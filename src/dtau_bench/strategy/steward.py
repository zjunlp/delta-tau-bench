"""All-tools skill context strategy with an on-demand explorer subagent.

Extends the skill-context base strategy with a virtual ``explore_capabilities``
tool. When the main agent calls it, an explorer subagent investigates the skill
library and the full tool catalog (read-only) and returns a compact report plus
a set of newly available tools.

The main agent decides when to delegate; there is no automatic trigger.

Key invariants:
- The explorer never executes a real tool (no DB state change, evaluation
  untouched).
- Only tools committed in the explorer's report enter the agent's visible set;
  merely "explored" tools do not (otherwise tool-retrieval precision would be
  destroyed by exploration breadth).
- The simulated user's private ``user_system_prompt`` is never passed to the
  explorer: only ``conversation`` is inspected.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from dtau_bench.explorer.agent import ExplorerAgent
from dtau_bench.explorer.backend import (
    ExplorerBackend,
    REF_ATOMIC,
    REF_TOOL_SKILL,
    make_ref,
)
from dtau_bench.explorer.store import ExploreTraceStore
from dtau_bench.explorer.types import ExploreRequest, ExplorerConfig, ExplorerResult
from dtau_bench.strategy.steward_base import StewardBaseStrategy
from dtau_bench.strategy.base import (
    StrategyRuntimeContext,
    StrategySessionState,
    StrategyToolInterceptionResult,
    StrategyUpdate,
    StrategyView,
)
from dtau_bench.strategy.registry import register_strategy
from dtau_bench.strategy.single_domain import (
    extract_initial_user_query,
    filter_tools_info_by_names,
    merge_tool_names_preserve_order,
)
from dtau_bench.utils.llm import LLMClient, LLMConfig

logger = logging.getLogger(__name__)

STRATEGY_NAME = "steward"

_COMPACT_PROMPT_HEADER = (
    "Reusable reference skill summaries. These are condensed: only names, one-line "
    "purposes and referenced tool names are shown. Tools mentioned here are "
    "references and may not exist in this environment; rely on the actual tool "
    "list of this session. If a summary looks relevant but you need the details, "
    "delegate an investigation with {tool_name}."
)


def build_explore_tool_definition(tool_name: str, max_actions: int) -> Dict[str, Any]:
    """OpenAI-format definition for the virtual explore tool."""
    return {
        "type": "function",
        "function": {
            "name": tool_name,
            "description": (
                "Delegate a focused investigation to a capability explorer subagent. "
                "Use this when the currently available tools are clearly insufficient, "
                "when you cannot tell which tool expresses a specific constraint the "
                "user mentioned, or when a tool call failed because a required or "
                "unsupported parameter. The explorer searches the reusable skill "
                "library and the full cross-domain tool catalog, inspects tool "
                f"schemas (up to {max_actions} lookups), and returns a compact report "
                "plus a set of tools that become available to you. When a stored "
                "procedure contains parameter-level or ordering detail, it can also "
                "add that procedure verbatim to your context under '# Added skills'. "
                "It never executes any tool, so it has no side effects and does not "
                "change any data. "
                "Prefer retrieve_tools for a cheap single keyword lookup; use this "
                "only for harder cases."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "goal": {
                        "type": "string",
                        "description": (
                            "What you are trying to accomplish, in one or two sentences. "
                            "Include the concrete user constraints you already know."
                        ),
                    },
                    "missing_capability": {
                        "type": "string",
                        "description": (
                            "The specific capability or parameter you could not find "
                            "with the tools you currently have."
                        ),
                    },
                    "known_failures": {
                        "type": "string",
                        "description": (
                            "Optional. Tool names you already tried and why they did "
                            "not work."
                        ),
                    },
                },
                "required": ["goal", "missing_capability"],
                "additionalProperties": False,
            },
        },
    }


@register_strategy(STRATEGY_NAME)
class StewardStrategy(StewardBaseStrategy):
    """skill context + iterative tool RAG + on-demand explorer subagent."""

    STRATEGY_NAME = STRATEGY_NAME

    def __init__(self, config: Dict[str, Any] = None, **kwargs):
        config = config or {}
        super().__init__(config=config, **kwargs)

        self.explorer_config = ExplorerConfig.from_dict(config.get("explorer"))
        self.explore_tool_name = self.explorer_config.tool_name

        run_context = config.get("_run_context") or {}
        self._run_output_dir = run_context.get("output_dir")
        self._run_name = run_context.get("run_name", "")
        # Authoritative domain/version. StrategyRuntimeContext derives them from
        # the pipeline's evaluation_context, which is empty on the test path
        # (rollout/pipeline.py), so relying on it would yield "unknown_domain".
        self._run_domain = str(run_context.get("domain") or "")
        self._run_version = str(run_context.get("version") or "")

        self._explorer_llm_client: Optional[LLMClient] = None
        self._explorer_llm_init_lock = threading.Lock()
        self._trace_stores: Dict[str, ExploreTraceStore] = {}
        self._trace_stores_lock = threading.Lock()

    def prewarm(self) -> None:
        """Load all shared Steward components before task workers start."""
        super().prewarm()
        if self.explorer_config.enabled:
            self._get_explorer_llm_client()

    # ─────────────────────────────────────────────────────────────────────
    # Lifecycle: start_session
    # ─────────────────────────────────────────────────────────────────────

    def start_session(
        self,
        sample: Dict[str, Any],
        all_tools_info: List[Dict[str, Any]],
        runtime_context: StrategyRuntimeContext,
    ) -> StrategySessionState:
        injection = self.explorer_config.initial_injection

        if injection == "none":
            state = self._start_session_without_skill_context(
                sample, all_tools_info, runtime_context
            )
        else:
            state = super().start_session(sample, all_tools_info, runtime_context)
            if injection == "compact":
                self._compact_initial_injection(state)

        state.custom_state["explore_tool_name"] = self.explore_tool_name
        state.custom_state["turn_index"] = 0
        state.custom_state["explore_calls"] = []
        state.custom_state["explore_rejected_calls"] = 0
        state.custom_state["explored_tool_names"] = []
        state.custom_state["explore_committed_tool_names"] = []
        state.custom_state["explore_initial_injection"] = injection
        state.custom_state["domain"] = self._run_domain or runtime_context.domain
        state.custom_state["version"] = self._run_version or runtime_context.version
        state.custom_state["tool_to_domain"] = runtime_context.tool_to_domain or {}
        return state

    def _start_session_without_skill_context(
        self,
        sample: Dict[str, Any],
        all_tools_info: List[Dict[str, Any]],
        runtime_context: StrategyRuntimeContext,
    ) -> StrategySessionState:
        """``initial_injection: none`` — skip the skill context pipeline entirely."""
        query_text = extract_initial_user_query(sample.get("messages", []))
        state = StrategySessionState(
            sample_id=sample.get("id", ""),
            query_text=query_text,
            visible_tool_names=[],
            visible_tools_info=[],
            prepended_messages=[],
            custom_state={
                "all_tools_info": list(all_tools_info),
                "retrieval_tool_name": self.retrieval_tool_name,
                "retrieval_call_count": 0,
                "skill_context_prompt": None,
                "skill_context_stats": {},
                "skill_context_trace": {},
            },
        )

        top_k_init = int(self._retrieval_config.get("top_k_init", 15))
        tool_rag = self._get_tool_rag_runtime()
        if query_text:
            retrieval = tool_rag.retrieve_tools(
                query_text=query_text,
                all_tools_info=all_tools_info,
                top_k=top_k_init,
            )
            state.visible_tool_names = list(retrieval["tool_names"])
            state.visible_tools_info = list(retrieval["tools_info"])

        state.retrieval_rounds.append({
            "type": "explore_initial_rag_only",
            "query": query_text,
            "top_k": top_k_init,
            "newly_added_tool_names": list(state.visible_tool_names),
            "newly_added_tool_count": len(state.visible_tool_names),
            "final_visible_tool_names": list(state.visible_tool_names),
            "final_visible_tool_count": len(state.visible_tool_names),
            "is_internal_retrieval_tool_round": False,
        })
        state.debug_history.append({
            "round": 0,
            "strategy": self.strategy_name,
            "visible_tool_count": len(state.visible_tool_names),
            "visible_tool_names": list(state.visible_tool_names),
            "initial_injection": "none",
            "iterative_retrieval_round_count": 0,
        })
        return state

    def _compact_initial_injection(self, state: StrategySessionState) -> None:
        """``initial_injection: compact`` — replace skill full text with L1 cards."""
        stats = state.custom_state.get("skill_context_stats") or {}
        names = [str(name) for name in (stats.get("selected_skill_names") or [])]
        if not names:
            return

        backend = self._build_backend(state)

        cards: List[str] = []
        for name in names:
            # Address by ref; fall back to the tool-skill layer for tool names.
            info = backend.read_skill(make_ref(REF_ATOMIC, name))
            if info.get("error"):
                info = backend.read_skill(make_ref(REF_TOOL_SKILL, name))
            if info.get("error"):
                continue
            summary = str(info.get("document") or "")
            summary = " ".join(summary.split())[: self.explorer_config.card_summary_chars]
            tools = ", ".join([str(t) for t in (info.get("tools") or [])][:6])
            cards.append(f"- {name}: {summary}" + (f" | tools: {tools}" if tools else ""))

        if not cards:
            return

        prompt = (
            _COMPACT_PROMPT_HEADER.format(tool_name=self.explore_tool_name)
            + "\n\n"
            + "\n".join(cards)
        )
        message = {"role": "system", "content": prompt}
        previous = state.custom_state.get("skill_context_prompt")
        state.prepended_messages = [
            item for item in state.prepended_messages if item is not previous
        ]
        state.prepended_messages.insert(0, message)
        state.custom_state["skill_context_prompt"] = message
        state.custom_state["skill_context_prompt_compacted"] = True

    # ─────────────────────────────────────────────────────────────────────
    # Lifecycle: get_agent_view
    # ─────────────────────────────────────────────────────────────────────

    def get_agent_view(
        self,
        state: StrategySessionState,
        conversation: List[Dict[str, Any]],
    ) -> StrategyView:
        view = super().get_agent_view(state, conversation)
        if not self.explorer_config.enabled:
            return view
        tools_info = list(view.visible_tools_info or [])
        tools_info.append(
            build_explore_tool_definition(
                self.explore_tool_name, self.explorer_config.max_actions
            )
        )
        view.visible_tools_info = tools_info
        return view

    # ─────────────────────────────────────────────────────────────────────
    # Lifecycle: on_turn_end
    # ─────────────────────────────────────────────────────────────────────

    def on_turn_end(
        self,
        state: StrategySessionState,
        conversation: List[Dict[str, Any]],
        turn_artifacts: Dict[str, Any],
    ) -> StrategyUpdate:
        turn_index = turn_artifacts.get("turn_index")
        if isinstance(turn_index, int):
            state.custom_state["turn_index"] = turn_index + 1
        else:
            state.custom_state["turn_index"] = (
                int(state.custom_state.get("turn_index", 0)) + 1
            )
        return super().on_turn_end(state, conversation, turn_artifacts)

    # ─────────────────────────────────────────────────────────────────────
    # Lifecycle: intercept_tool_calls
    # ─────────────────────────────────────────────────────────────────────

    def intercept_tool_calls(
        self,
        state: StrategySessionState,
        tool_calls: List[Dict[str, Any]],
        conversation: List[Dict[str, Any]],
    ) -> StrategyToolInterceptionResult:
        # Let the parent handle retrieve_tools; explore calls fall through as
        # "remaining" and are processed here.
        base = super().intercept_tool_calls(state, tool_calls, conversation)
        if not self.explorer_config.enabled:
            return base

        handled = list(base.handled_tool_responses)
        remaining: List[Dict[str, Any]] = []
        metadata = dict(base.metadata)

        for tool_call in base.remaining_tool_calls:
            function = tool_call.get("function") or {}
            if str(function.get("name") or "").strip() != self.explore_tool_name:
                remaining.append(tool_call)
                continue

            payload = self._handle_explore_call(state, tool_call, conversation)
            handled.append({
                "role": "tool",
                "content": json.dumps(payload, ensure_ascii=False),
                "tool_call_id": tool_call.get("id"),
            })
            metadata["explore_call_count"] = len(
                state.custom_state.get("explore_calls") or []
            )

        return StrategyToolInterceptionResult(
            handled_tool_responses=handled,
            remaining_tool_calls=remaining,
            metadata=metadata,
        )

    # ─────────────────────────────────────────────────────────────────────
    # Explore handling
    # ─────────────────────────────────────────────────────────────────────

    def _handle_explore_call(
        self,
        state: StrategySessionState,
        tool_call: Dict[str, Any],
        conversation: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        calls: List[Dict[str, Any]] = state.custom_state.setdefault("explore_calls", [])
        config = self.explorer_config

        if len(calls) >= config.max_explore_calls_per_session:
            state.custom_state["explore_rejected_calls"] = (
                int(state.custom_state.get("explore_rejected_calls", 0)) + 1
            )
            return {
                "error": "explore budget exhausted for this task",
                "hint": "Use retrieve_tools for further lookups.",
                "explore_calls_used": len(calls),
            }

        arguments = self._parse_arguments(tool_call)
        goal = str(arguments.get("goal") or "").strip()
        missing = str(arguments.get("missing_capability") or "").strip()
        if not goal and not missing:
            return {
                "error": (
                    f"{self.explore_tool_name} requires 'goal' and 'missing_capability'."
                ),
            }

        request = ExploreRequest(
            goal=goal,
            missing_capability=missing,
            known_failures=str(arguments.get("known_failures") or "").strip(),
            turn_index=int(state.custom_state.get("turn_index", 0)),
            call_index=len(calls),
            sample_id=state.sample_id,
            domain=str(state.custom_state.get("domain") or ""),
            version=str(state.custom_state.get("version") or ""),
            visible_tool_names=list(state.visible_tool_names),
            injected_skill_names=[
                str(name)
                for name in (
                    (state.custom_state.get("skill_context_stats") or {}).get(
                        "selected_skill_names"
                    )
                    or []
                )
            ],
            recent_user_messages=self._recent_user_messages(conversation),
            recent_tool_errors=self._recent_tool_errors(conversation),
            pseudo_plan=str(
                (state.custom_state.get("skill_context_stats") or {}).get("pseudo_plan") or ""
            ),
        )

        started = time.time()
        try:
            backend = self._build_backend(state)
            explorer = ExplorerAgent(
                backend=backend,
                llm_client=self._get_explorer_llm_client(),
                config=config,
            )
            result = explorer.run(request)
        except Exception as exc:
            logger.error("Explorer failed for sample %s: %r", state.sample_id, exc)
            state.custom_state["explore_error"] = str(exc)
            calls.append({
                "call_index": request.call_index,
                "turn_index": request.turn_index,
                "goal": goal,
                "missing_capability": missing,
                "stop_reason": "exception",
                "error": str(exc),
                "wall_time_s": round(time.time() - started, 3),
            })
            return {
                "error": f"exploration failed: {exc}",
                "hint": "Continue with retrieve_tools.",
            }

        newly_added = self._commit_tools(state, result)
        new_skill_names = self._commit_skills(state, result, backend)
        self._record_call(state, request, result, newly_added, calls)

        payload = {
            "report": result.report_final,
            "new_tool_names": newly_added,
            "new_tool_count": len(newly_added),
            "all_visible_tool_names": list(state.visible_tool_names),
            "all_visible_tool_count": len(state.visible_tool_names),
            "confidence": result.confidence,
            "explore_calls_left": max(
                config.max_explore_calls_per_session - len(calls), 0
            ),
        }
        if new_skill_names:
            payload["new_skill_names"] = new_skill_names
            payload["skills_note"] = (
                "Their full procedures were added to your context under "
                "'# Added skills'. A skill does NOT grant its tools."
            )
        return payload

    def _commit_tools(
        self, state: StrategySessionState, result: ExplorerResult
    ) -> List[str]:
        """Only committed tools become visible; explored ones never do."""
        all_tools_info = state.custom_state.get("all_tools_info") or []
        before = set(state.visible_tool_names)

        merged = merge_tool_names_preserve_order(
            state.visible_tool_names, result.committed_tool_names
        )
        newly_added = [name for name in merged if name not in before]

        if newly_added:
            state.visible_tool_names = merged
            state.visible_tools_info = filter_tools_info_by_names(all_tools_info, merged)

        committed_all: List[str] = state.custom_state.setdefault(
            "explore_committed_tool_names", []
        )
        for name in newly_added:
            if name not in committed_all:
                committed_all.append(name)

        explored_all: List[str] = state.custom_state.setdefault("explored_tool_names", [])
        for name in result.explored_tool_names:
            if name not in explored_all:
                explored_all.append(name)

        # A retrieval round so downstream metrics pick up the new visible set
        # without any change to run_evaluate*.py.
        state.retrieval_rounds.append({
            "type": "explore_commit",
            "query": result.report_final[:200],
            "top_k": self.explorer_config.max_commit_tools,
            "newly_added_tool_names": newly_added,
            "newly_added_tool_count": len(newly_added),
            "final_visible_tool_names": list(state.visible_tool_names),
            "final_visible_tool_count": len(state.visible_tool_names),
            "is_internal_retrieval_tool_round": True,
        })
        state.debug_history.append({
            "round": len(state.debug_history),
            "strategy": self.strategy_name,
            "explore_stop_reason": result.stop_reason,
            "explore_actions_used": result.actions_used,
            "explore_committed_tool_names": list(result.committed_tool_names),
            "explore_newly_added_tool_names": newly_added,
            "iterative_visible_tool_names": list(state.visible_tool_names),
            "iterative_visible_tool_count": len(state.visible_tool_names),
        })

        if self.explorer_config.persist_report_as_system and result.report_final:
            marker = "Capability exploration findings."
            state.prepended_messages = [
                message
                for message in state.prepended_messages
                if not str(message.get("content", "")).startswith(marker)
            ]
            state.prepended_messages.append({
                "role": "system",
                "content": f"{marker}\n{result.report_final}",
            })

        return newly_added

    def _commit_skills(
        self,
        state: StrategySessionState,
        result: ExplorerResult,
        backend: ExplorerBackend,
    ) -> List[str]:
        """Hand validated skills over to the main agent (P2).

        The injected block is a SINGLE prepended system message that is REBUILT
        on every commit rather than appended to. A session allows up to
        ``max_explore_calls_per_session`` explorations and we have already lost
        samples to the gateway's 1 MB request-body limit, so accumulating
        messages here is not acceptable.

        INVARIANT: committing a skill never exposes its tools. The predicted
        tool set feeds the retrieval metric and one skill may reference 3-6
        tools; auto-exposing them would wreck precision.
        """
        config = self.explorer_config
        entries = list(result.committed_skills or [])
        if not entries:
            return []

        committed: List[Dict[str, Any]] = state.custom_state.setdefault(
            "explore_committed_skills", []
        )
        known_refs = {str(item.get("ref") or "") for item in committed}
        stats = state.custom_state.get("skill_context_stats") or {}
        injected_names = {
            str(name) for name in (stats.get("selected_skill_names") or [])
        }

        newly_added: List[str] = []
        for entry in entries:
            ref = str(entry.get("ref") or "")
            name = str(entry.get("name") or "")
            if not ref or ref in known_refs or name in injected_names:
                continue
            known_refs.add(ref)
            committed.append(entry)
            newly_added.append(name)

        if not newly_added:
            return []

        if config.persist_skills_as_system:
            rendered = backend.render_committed_skills(
                committed, int(config.max_committed_skill_chars_per_session)
            )
            marker = "# Added skills (from capability investigation)"
            state.prepended_messages = [
                message
                for message in state.prepended_messages
                if marker not in str(message.get("content", ""))
            ]
            if rendered:
                state.prepended_messages.append({"role": "system", "content": rendered})
                state.custom_state["explore_committed_skill_chars"] = len(rendered)

        state.custom_state["explore_committed_skill_names"] = [
            str(item.get("name") or "") for item in committed
        ]
        state.custom_state["explore_committed_skill_refs"] = [
            str(item.get("ref") or "") for item in committed
        ]
        state.debug_history.append({
            "round": len(state.debug_history),
            "strategy": self.strategy_name,
            "explore_committed_skill_refs": list(
                state.custom_state["explore_committed_skill_refs"]
            ),
            "explore_newly_added_skill_names": newly_added,
        })
        return newly_added

    def _record_call(
        self,
        state: StrategySessionState,
        request: ExploreRequest,
        result: ExplorerResult,
        newly_added: List[str],
        calls: List[Dict[str, Any]],
    ) -> None:
        store = self._get_trace_store(state)
        line_id = f"{state.sample_id}#{request.call_index}"

        summary = {
            "call_index": request.call_index,
            "turn_index": request.turn_index,
            "goal": request.goal,
            "missing_capability": request.missing_capability,
            "newly_added_tool_names": newly_added,
            **result.summary(),
        }
        if store.available:
            summary["trace_ref"] = {
                "file": store.relative_path(self.explorer_config.trace_dir_name),
                "line_id": line_id,
            }
        else:
            state.custom_state["explore_trace_unavailable"] = True
        calls.append(summary)

        if not store.available:
            return

        store.append({
            "line_id": line_id,
            "sample_id": state.sample_id,
            "domain": request.domain,
            "version": request.version,
            "run_name": self._run_name,
            "strategy": self.strategy_name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "request": request.to_dict(),
            "actions": [action.to_dict() for action in result.actions],
            "messages": result.messages,
            "llm_usage": [item.to_dict() for item in result.llm_usage],
            "report_raw": result.report_raw,
            "report_final": result.report_final,
            "commit": {
                "requested": result.requested_tool_names,
                "accepted": result.committed_tool_names,
                "rejected_hallucinated": result.hallucinated_tool_names,
                "newly_added": newly_added,
            },
            "commit_skills": {
                "requested": list(result.requested_skill_refs),
                "accepted": [
                    {
                        "ref": str(entry.get("ref") or ""),
                        "kind": str(entry.get("kind") or ""),
                        "name": str(entry.get("name") or ""),
                        "chars": len(str(entry.get("body") or "")),
                        "truncated": bool(entry.get("truncated")),
                        "tools": list(entry.get("tools") or []),
                    }
                    for entry in (result.committed_skills or [])
                ],
                "rejected_unresolvable": list(result.rejected_skill_refs),
                "session_committed_names": list(
                    state.custom_state.get("explore_committed_skill_names") or []
                ),
            },
            "explored": {
                "tool_names": result.explored_tool_names,
                "skill_names": result.explored_skill_names,
            },
            "budget": {
                "max_actions": self.explorer_config.max_actions,
                "actions_used": result.actions_used,
                "max_llm_turns": self.explorer_config.max_llm_turns,
                "llm_turns_used": result.llm_calls,
                "stop_reason": result.stop_reason,
            },
            "report_chars_before_truncate": result.report_chars_before_truncate,
            "wall_time_s": round(result.wall_time_s, 3),
            "error": result.error,
        })

    # ─────────────────────────────────────────────────────────────────────
    # Context extraction (only what the main agent can also observe)
    # ─────────────────────────────────────────────────────────────────────

    def _recent_user_messages(self, conversation: List[Dict[str, Any]]) -> List[str]:
        limit = max(int(self.explorer_config.recent_user_messages), 0)
        if limit == 0:
            return []
        per_message = max(int(self.explorer_config.max_context_chars) // limit, 80)
        collected: List[str] = []
        for message in reversed(conversation or []):
            if str(message.get("role")) != "user":
                continue
            content = str(message.get("content") or "").strip()
            if not content:
                continue
            collected.append(content[:per_message])
            if len(collected) >= limit:
                break
        return list(reversed(collected))

    def _recent_tool_errors(
        self, conversation: List[Dict[str, Any]]
    ) -> List[Dict[str, str]]:
        limit = max(int(self.explorer_config.recent_tool_errors), 0)
        if limit == 0:
            return []

        name_by_call_id: Dict[str, str] = {}
        for message in conversation or []:
            for tool_call in message.get("tool_calls") or []:
                if not isinstance(tool_call, dict):
                    continue
                function = tool_call.get("function") or {}
                call_id = str(tool_call.get("id") or "")
                name = str(function.get("name") or "")
                if call_id and name:
                    name_by_call_id[call_id] = name

        errors: List[Dict[str, str]] = []
        for message in reversed(conversation or []):
            if str(message.get("role")) != "tool":
                continue
            content = str(message.get("content") or "")
            error_text = self._extract_error_text(content)
            if not error_text:
                continue
            tool_name = name_by_call_id.get(str(message.get("tool_call_id") or ""), "")
            if tool_name in {self.explore_tool_name, self.retrieval_tool_name}:
                continue
            errors.append({"tool": tool_name or "unknown", "error": error_text[:240]})
            if len(errors) >= limit:
                break
        return list(reversed(errors))

    @staticmethod
    def _extract_error_text(content: str) -> str:
        text = (content or "").strip()
        if not text:
            return ""
        try:
            payload = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            lowered = text.lower()
            if "error" in lowered or "failed" in lowered or "invalid" in lowered:
                return " ".join(text.split())
            return ""
        if isinstance(payload, dict):
            for key in ("error", "message", "detail"):
                value = payload.get(key)
                if isinstance(value, str) and value.strip():
                    if key != "error" and not payload.get("error"):
                        if str(payload.get("status", "")).lower() not in {"error", "failed"}:
                            continue
                    return " ".join(value.split())
        return ""

    # ─────────────────────────────────────────────────────────────────────
    # Component construction
    # ─────────────────────────────────────────────────────────────────────

    def _build_backend(self, state: StrategySessionState) -> ExplorerBackend:
        all_tools_info = state.custom_state.get("all_tools_info") or []
        tool_to_domain = state.custom_state.get("tool_to_domain") or {}
        return ExplorerBackend(
            config=self.explorer_config,
            all_tools_info=all_tools_info,
            skill_context_runtime=self._get_skill_context_runtime(),
            tool_rag_runtime=self._get_tool_rag_runtime(),
            tool_to_domain=tool_to_domain,
        )

    def _get_explorer_llm_client(self) -> LLMClient:
        if self._explorer_llm_client is not None:
            return self._explorer_llm_client
        with self._explorer_llm_init_lock:
            if self._explorer_llm_client is not None:
                return self._explorer_llm_client
            llm_cfg = self.config.get("_explorer_llm_config") or self.config.get(
                "_llm_config", {}
            )
            client = LLMClient(
                LLMConfig(
                    model_name=llm_cfg.get("model_name", "DeepSeek-V3.2"),
                    base_url=llm_cfg.get("base_url", "http://localhost:8000/v1"),
                    api_key=llm_cfg.get("api_key", "EMPTY"),
                    max_tokens=llm_cfg.get("max_tokens", 4096),
                    temperature=self.explorer_config.temperature,
                    timeout=llm_cfg.get("timeout", 180),
                    max_retries=2,
                    extra_params=dict(llm_cfg.get("request_options") or {}),
                )
            )
            self._explorer_llm_client = client
            logger.info("Explorer LLM client initialized: %s", client.model_name)
            return client

    def _get_trace_store(self, state: StrategySessionState) -> ExploreTraceStore:
        domain = str(
            state.custom_state.get("domain") or self._run_domain or "unknown_domain"
        )
        store = self._trace_stores.get(domain)
        if store is None:
            with self._trace_stores_lock:
                store = self._trace_stores.get(domain)
                if store is None:
                    store = ExploreTraceStore(
                        output_dir=self._run_output_dir,
                        domain=domain,
                        dir_name=self.explorer_config.trace_dir_name,
                        enabled=True,
                    )
                    self._trace_stores[domain] = store
        return store

    @staticmethod
    def _parse_arguments(tool_call: Dict[str, Any]) -> Dict[str, Any]:
        raw = (tool_call.get("function") or {}).get("arguments", {})
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, str):
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                return {}
            return parsed if isinstance(parsed, dict) else {}
        return {}

    # ─────────────────────────────────────────────────────────────────────
    # Lifecycle: get_artifacts
    # ─────────────────────────────────────────────────────────────────────

    def get_artifacts(self, state: StrategySessionState) -> Dict[str, Any]:
        artifacts = super().get_artifacts(state)
        calls = list(state.custom_state.get("explore_calls") or [])

        prompt_tokens = [c.get("prompt_tokens") for c in calls if c.get("prompt_tokens")]
        completion_tokens = [
            c.get("completion_tokens") for c in calls if c.get("completion_tokens")
        ]

        artifacts["explore"] = {
            "enabled": bool(self.explorer_config.enabled),
            "initial_injection": state.custom_state.get("explore_initial_injection"),
            "total_calls": len(calls),
            "rejected_calls": int(state.custom_state.get("explore_rejected_calls", 0)),
            "committed_tool_names": list(
                state.custom_state.get("explore_committed_tool_names") or []
            ),
            "explored_tool_names": list(
                state.custom_state.get("explored_tool_names") or []
            ),
            # P2 — kept OUT of retrieval.rounds on purpose: that trace defines the
            # predicted tool set, and skills are not tools.
            "committed_skill_names": list(
                state.custom_state.get("explore_committed_skill_names") or []
            ),
            "committed_skill_refs": list(
                state.custom_state.get("explore_committed_skill_refs") or []
            ),
            "committed_skill_chars": int(
                state.custom_state.get("explore_committed_skill_chars", 0)
            ),
            "total_llm_calls": sum(int(c.get("llm_calls") or 0) for c in calls),
            "prompt_tokens": sum(prompt_tokens) if prompt_tokens else None,
            "completion_tokens": sum(completion_tokens) if completion_tokens else None,
            "total_wall_time_s": round(
                sum(float(c.get("wall_time_s") or 0.0) for c in calls), 3
            ),
            "trace_unavailable": bool(
                state.custom_state.get("explore_trace_unavailable", False)
            ),
            "error": state.custom_state.get("explore_error"),
            "calls": calls,
        }
        return artifacts
