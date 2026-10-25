"""Explorer subagent main loop.

A deep-research style loop over a side-effect free action space, with a hard
budget (default: 20 actions) and a guaranteed report: if the budget is exhausted
or the model misbehaves, the loop still produces a usable (possibly fallback)
report instead of failing the rollout.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional

from dtau_bench.explorer.backend import ExplorerBackend
from dtau_bench.explorer.prompts import (
    FORCE_FINISH_MESSAGE,
    build_budget_warning,
    build_request_prompt,
    build_system_prompt,
)
from dtau_bench.explorer.report import build_fallback_report, validate_report
from dtau_bench.explorer.types import (
    ActionRecord,
    ExploreRequest,
    ExplorerConfig,
    ExplorerResult,
    LLMUsageRecord,
)

logger = logging.getLogger(__name__)

FINISH_ACTION = "finish"


def build_action_definitions(config: ExplorerConfig) -> List[Dict[str, Any]]:
    """OpenAI-format definitions for the explorer's virtual action space."""
    return [
        {
            "type": "function",
            "function": {
                "name": "search_skills",
                "description": (
                    "Search the reusable skill library (workflows and procedures). "
                    "Returns compact summary cards only. Each card has a `ref` "
                    "(e.g. 'atomic:xxx', 'planning:1a2b3c4d') which is the ONLY "
                    "valid handle for read_skill / expand_graph."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "What capability or workflow you are looking for."},
                        "kind": {
                            "type": "string",
                            "enum": ["any", "atomic", "planning"],
                            "description": (
                                "Skill layer: 'atomic' = one reusable procedure, "
                                "'planning' = a whole task plan. Default: any."
                            ),
                        },
                        "top_k": {"type": "integer", "description": f"1..{config.search_skills_top_k}"},
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "read_skill",
                "description": (
                    "Read one skill body. Pass the card's `ref`, NOT its display "
                    "name. Long bodies are truncated."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "ref": {
                            "type": "string",
                            "description": "A `ref` returned by search_skills / search_tool_skills / expand_graph.",
                        }
                    },
                    "required": ["ref"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "expand_graph",
                "description": (
                    "Walk one hop of the skill graph from a card `ref`. "
                    "to='tools': atomic skill -> its tool skills. "
                    "to='atomic_skills': planning skill -> its atomic skills. "
                    "to='plans': atomic skill -> plans that use it."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "ref": {
                            "type": "string",
                            "description": "A `ref` from a card (not a tool name, not a task sentence).",
                        },
                        "to": {
                            "type": "string",
                            "enum": ["tools", "atomic_skills", "plans"],
                            "description": "Which edge to follow. Defaults to the natural edge for the ref's layer.",
                        },
                        "limit": {"type": "integer", "description": f"1..{config.expand_graph_limit}"},
                    },
                    "required": ["ref"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_tool_skills",
                "description": (
                    "Search learned usage notes for tools: required parameters, "
                    "valid enum values, preconditions, call ordering. Use this "
                    "after search_tools when you need to know HOW to call a tool, "
                    "not just which tool exists."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Tool name or the operation you need to perform."},
                        "top_k": {"type": "integer", "description": f"1..{config.search_skills_top_k}"},
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_tools",
                "description": (
                    "Semantic search over the full cross-domain tool catalog of the "
                    "current version. Returns candidate tools; they are not yet "
                    "available to the main agent."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "top_k": {"type": "integer", "description": f"1..{config.search_tools_top_k}"},
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "inspect_tool",
                "description": (
                    "Inspect one tool's full schema: required/optional parameters, "
                    "enums, cost, latency, confirmation requirement."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "note",
                "description": "Record a short intermediate conclusion so you do not lose it.",
                "parameters": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": FINISH_ACTION,
                "description": (
                    "Finish the investigation. Provide the compact structured report "
                    "and the exact tool names the main agent should receive."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "report": {"type": "string", "description": "Structured report following the required layout."},
                        "commit_tools": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": f"Verified tool names to hand over (max {config.max_commit_tools}).",
                        },
                        "commit_skills": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "Card `ref`s of skills whose procedure the main agent "
                                "should receive verbatim (max "
                                f"{config.max_commit_skills}). Use this when the skill "
                                "body contains parameter-level or ordering detail that "
                                "would be lost if you paraphrased it."
                            ),
                        },
                        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                    },
                    "required": ["report", "commit_tools"],
                    "additionalProperties": False,
                },
            },
        },
    ]


class ExplorerAgent:
    """Runs one investigation and returns a validated, compressed report."""

    def __init__(
        self,
        *,
        backend: ExplorerBackend,
        llm_client: Any,
        config: ExplorerConfig,
    ):
        self.backend = backend
        self.llm_client = llm_client
        self.config = config
        self._action_definitions = build_action_definitions(config)

    # ─────────────────────────────────────────────────────────────────────

    def run(self, request: ExploreRequest) -> ExplorerResult:
        started = time.time()
        config = self.config

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": build_system_prompt(config)},
            {"role": "user", "content": build_request_prompt(request, config)},
        ]
        actions: List[ActionRecord] = []
        usage: List[LLMUsageRecord] = []
        empty_replies = 0
        stop_reason = "max_llm_turns"
        finish_payload: Optional[Dict[str, Any]] = None

        exclude_skill_names = (
            set(request.injected_skill_names) if config.exclude_injected_skills else set()
        )

        for _ in range(max(int(config.max_llm_turns), 1)):
            if time.time() - started > config.hard_timeout_s:
                stop_reason = "timeout"
                break
            if len(actions) >= config.max_actions:
                stop_reason = "max_actions"
                break

            response = self._chat(messages, usage, tools=self._action_definitions)
            if response is None:
                stop_reason = "llm_error"
                break

            content = self._safe_extract_content(response)
            tool_calls = self._safe_extract_tool_calls(response)
            messages.append(self._assistant_message(content, response))

            if not tool_calls:
                empty_replies += 1
                if empty_replies >= 2:
                    stop_reason = "no_action"
                    break
                messages.append({
                    "role": "user",
                    "content": (
                        "You must call one of the provided actions, or call finish "
                        "with your report."
                    ),
                })
                continue

            finish_call = None
            for call in tool_calls:
                name = str(call.get("name") or "")
                arguments = call.get("arguments") if isinstance(call.get("arguments"), dict) else {}

                if name == FINISH_ACTION:
                    finish_call = arguments
                    messages.append({
                        "role": "tool",
                        "tool_call_id": call.get("id"),
                        "content": json.dumps({"status": "received"}, ensure_ascii=False),
                    })
                    continue

                if len(actions) >= config.max_actions:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": call.get("id"),
                        "content": json.dumps(
                            {"error": "action budget exhausted", "hint": FORCE_FINISH_MESSAGE},
                            ensure_ascii=False,
                        ),
                    })
                    continue

                record = self._execute_action(
                    index=len(actions),
                    name=name,
                    arguments=arguments,
                    exclude_skill_names=exclude_skill_names,
                )
                actions.append(record)
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.get("id"),
                    "content": json.dumps(record.result_payload, ensure_ascii=False),
                })

            if finish_call is not None:
                finish_payload = finish_call
                stop_reason = "finish"
                break

            actions_left = config.max_actions - len(actions)
            if 0 < actions_left <= 3:
                messages.append({"role": "user", "content": build_budget_warning(actions_left)})

        #── Forced finish: give the model one last chance to summarize ──
        if finish_payload is None and stop_reason in {
            "max_actions",
            "max_llm_turns",
            "timeout",
            "no_action",
        }:
            forced = self._force_finish(messages, usage)
            if forced is not None:
                finish_payload = forced

        return self._finalize(
            request=request,
            finish_payload=finish_payload,
            actions=actions,
            usage=usage,
            messages=messages,
            stop_reason=stop_reason,
            started=started,
        )

    # ─────────────────────────────────────────────────────────────────────
    # Action dispatch
    # ─────────────────────────────────────────────────────────────────────

    def _execute_action(
        self,
        *,
        index: int,
        name: str,
        arguments: Dict[str, Any],
        exclude_skill_names: set,
    ) -> ActionRecord:
        started = time.time()
        error: Optional[str] = None
        try:
            payload = self._dispatch(name, arguments, exclude_skill_names)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Explorer action %s failed: %r", name, exc)
            payload = {"error": f"action failed: {exc}"}
            error = str(exc)

        serialized = json.dumps(payload, ensure_ascii=False)
        return ActionRecord(
            index=index,
            name=name,
            arguments=arguments,
            result_payload=payload,
            result_chars=len(serialized),
            latency_s=time.time() - started,
            error=error,
        )

    def _dispatch(
        self, name: str, arguments: Dict[str, Any], exclude_skill_names: set
    ) -> Any:
        config = self.config
        if name == "search_skills":
            return self.backend.search_skills(
                query=str(arguments.get("query") or ""),
                kind=str(arguments.get("kind") or "any"),
                top_k=int(arguments.get("top_k") or config.search_skills_top_k),
                exclude_names=exclude_skill_names,
            )
        if name == "read_skill":
            # Accept the legacy "name" key so an out-of-sync prompt cannot break it.
            return self.backend.read_skill(
                str(arguments.get("ref") or arguments.get("name") or "")
            )
        if name == "expand_graph":
            return self.backend.expand_graph(
                ref=str(arguments.get("ref") or arguments.get("name") or ""),
                to=arguments.get("to"),
                direction=arguments.get("direction"),
                limit=int(arguments.get("limit") or config.expand_graph_limit),
            )
        if name == "search_tool_skills":
            return self.backend.search_tool_skills(
                query=str(arguments.get("query") or ""),
                top_k=int(arguments.get("top_k") or config.search_skills_top_k),
            )
        if name == "search_tools":
            return self.backend.search_tools(
                query=str(arguments.get("query") or ""),
                top_k=int(arguments.get("top_k") or config.search_tools_top_k),
            )
        if name == "inspect_tool":
            return self.backend.inspect_tool(str(arguments.get("name") or ""))
        if name == "note":
            return {"status": "noted", "text": str(arguments.get("text") or "")[:200]}
        return {
            "error": f"unknown action '{name}'",
            "available_actions": [
                "search_skills",
                "read_skill",
                "expand_graph",
                "search_tool_skills",
                "search_tools",
                "inspect_tool",
                "note",
                FINISH_ACTION,
            ],
        }

    # ─────────────────────────────────────────────────────────────────────
    # LLM plumbing
    # ─────────────────────────────────────────────────────────────────────

    def _chat(
        self,
        messages: List[Dict[str, Any]],
        usage: List[LLMUsageRecord],
        *,
        tools: List[Dict[str, Any]],
        tool_choice: Optional[Any] = None,
    ) -> Optional[Any]:
        started = time.time()
        model_name = getattr(self.llm_client, "model_name", "unknown")
        try:
            kwargs: Dict[str, Any] = {
                "messages": messages,
                "tools": tools,
                "temperature": self.config.temperature,
            }
            if tool_choice is not None:
                kwargs["tool_choice"] = tool_choice
            response = self.llm_client.chat(**kwargs)
        except Exception as exc:
            logger.warning("Explorer LLM call failed: %r", exc)
            usage.append(
                LLMUsageRecord(
                    model=str(model_name),
                    prompt_tokens=None,
                    completion_tokens=None,
                    latency_s=time.time() - started,
                    error=str(exc),
                )
            )
            return None

        prompt_tokens, completion_tokens = self._extract_usage(response)
        usage.append(
            LLMUsageRecord(
                model=str(model_name),
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_s=time.time() - started,
            )
        )
        return response

    def _force_finish(
        self, messages: List[Dict[str, Any]], usage: List[LLMUsageRecord]
    ) -> Optional[Dict[str, Any]]:
        messages.append({"role": "user", "content": FORCE_FINISH_MESSAGE})
        finish_only = [
            definition
            for definition in self._action_definitions
            if definition["function"]["name"] == FINISH_ACTION
        ]
        response = self._chat(
            messages,
            usage,
            tools=finish_only,
        )
        if response is None:
            return None
        content = self._safe_extract_content(response)
        messages.append(self._assistant_message(content, response))
        for call in self._safe_extract_tool_calls(response):
            if str(call.get("name")) == FINISH_ACTION:
                arguments = call.get("arguments")
                return arguments if isinstance(arguments, dict) else {}
        # Some models answer in plain text when forced; accept it as the report.
        if content.strip():
            return {"report": content, "commit_tools": []}
        return None

    def _assistant_message(self, content: str, response: Any) -> Dict[str, Any]:
        message: Dict[str, Any] = {"role": "assistant", "content": content}
        has_reasoning, reasoning_content = self._extract_reasoning_content(response)
        if has_reasoning:
            # An explicit empty string is protocol-significant for reasoning
            # gateways after an assistant tool call; do not use a truthiness
            # check here.
            message["reasoning_content"] = reasoning_content
        try:
            raw_calls = self.llm_client.extract_tool_calls(response)
        except Exception:  # pragma: no cover - defensive
            raw_calls = []
        if raw_calls:
            message["tool_calls"] = raw_calls
        return message

    @staticmethod
    def _extract_reasoning_content(response: Any) -> tuple[bool, str]:
        """Extract provider reasoning from LiteLLM objects or raw dicts.

        Missing and ``None`` are omitted, while every string -- including
        ``""`` -- is replayed in the next Explorer request.
        """
        sentinel = object()
        if isinstance(response, dict):
            choices = response.get("choices")
        else:
            choices = getattr(response, "choices", None)
        if not isinstance(choices, (list, tuple)) or not choices:
            return False, ""

        choice = choices[0]
        if isinstance(choice, dict):
            provider_message = choice.get("message")
        else:
            provider_message = getattr(choice, "message", None)

        if isinstance(provider_message, dict):
            value = provider_message.get("reasoning_content", sentinel)
        else:
            value = getattr(provider_message, "reasoning_content", sentinel)
            if value is sentinel:
                # Some LiteLLM mapping-like response objects expose provider
                # extras via ``get`` without materialising an attribute.
                getter = getattr(provider_message, "get", None)
                if callable(getter):
                    value = getter("reasoning_content", sentinel)

        if isinstance(value, str):
            return True, value
        return False, ""

    def _safe_extract_content(self, response: Any) -> str:
        try:
            return self.llm_client.extract_content(response) or ""
        except Exception:  # pragma: no cover - defensive
            return ""

    def _safe_extract_tool_calls(self, response: Any) -> List[Dict[str, Any]]:
        try:
            return self.llm_client.extract_tool_calls_parsed(response) or []
        except Exception:  # pragma: no cover - defensive
            return []

    @staticmethod
    def _extract_usage(response: Any) -> tuple[Optional[int], Optional[int]]:
        usage = response.get("usage") if isinstance(response, dict) else getattr(response, "usage", None)
        if usage is None:
            return None, None
        prompt = getattr(usage, "prompt_tokens", None)
        completion = getattr(usage, "completion_tokens", None)
        if prompt is None and isinstance(usage, dict):
            prompt = usage.get("prompt_tokens")
            completion = usage.get("completion_tokens")
        try:
            prompt = int(prompt) if prompt is not None else None
        except (TypeError, ValueError):
            prompt = None
        try:
            completion = int(completion) if completion is not None else None
        except (TypeError, ValueError):
            completion = None
        return prompt, completion

    # ─────────────────────────────────────────────────────────────────────
    # Finalization
    # ─────────────────────────────────────────────────────────────────────

    def _finalize(
        self,
        *,
        request: ExploreRequest,
        finish_payload: Optional[Dict[str, Any]],
        actions: List[ActionRecord],
        usage: List[LLMUsageRecord],
        messages: List[Dict[str, Any]],
        stop_reason: str,
        started: float,
    ) -> ExplorerResult:
        config = self.config
        traced_messages = list(messages) if config.trace_full_messages else []

        if finish_payload is None:
            report_text, names = build_fallback_report(
                goal=request.goal,
                tool_cards=self._fallback_tool_cards(),
                max_report_chars=config.max_report_chars,
            )
            validation = validate_report(
                report_raw=report_text,
                requested_tools=names,
                known_tool_names=self.backend.known_tool_names,
                max_report_chars=config.max_report_chars,
                max_commit_tools=config.max_commit_tools,
                already_visible=request.visible_tool_names,
            )
            return ExplorerResult(
                report_final=validation.report_final,
                report_raw=report_text,
                committed_tool_names=validation.accepted_tool_names,
                requested_tool_names=validation.requested_tool_names,
                hallucinated_tool_names=validation.hallucinated_tool_names,
                explored_tool_names=list(self.backend.explored_tool_names),
                explored_skill_names=list(self.backend.explored_skill_names),
                confidence="low",
                stop_reason="fallback",
                actions=actions,
                llm_usage=usage,
                messages=traced_messages,
                report_chars_before_truncate=validation.chars_before_truncate,
                wall_time_s=time.time() - started,
                error=f"explorer did not finish (reason={stop_reason})",
            )

        report_raw = str(finish_payload.get("report") or "")
        commit_tools = finish_payload.get("commit_tools") or []
        if not isinstance(commit_tools, list):
            commit_tools = [commit_tools]
        confidence = str(finish_payload.get("confidence") or "unknown").strip().lower()

        committed_skills, requested_skill_refs, rejected_skill_refs = (
            self._resolve_commit_skills(finish_payload, request)
        )

        validation = validate_report(
            report_raw=report_raw,
            requested_tools=[str(name) for name in commit_tools],
            known_tool_names=self.backend.known_tool_names,
            max_report_chars=config.max_report_chars,
            max_commit_tools=config.max_commit_tools,
            already_visible=request.visible_tool_names,
        )

        return ExplorerResult(
            report_final=validation.report_final,
            report_raw=report_raw,
            committed_tool_names=validation.accepted_tool_names,
            requested_tool_names=validation.requested_tool_names,
            hallucinated_tool_names=validation.hallucinated_tool_names,
            explored_tool_names=list(self.backend.explored_tool_names),
            explored_skill_names=list(self.backend.explored_skill_names),
            committed_skills=committed_skills,
            requested_skill_refs=requested_skill_refs,
            rejected_skill_refs=rejected_skill_refs,
            confidence=confidence,
            stop_reason=stop_reason if stop_reason == "finish" else f"forced_finish:{stop_reason}",
            actions=actions,
            llm_usage=usage,
            messages=traced_messages,
            report_chars_before_truncate=validation.chars_before_truncate,
            wall_time_s=time.time() - started,
        )

    def _resolve_commit_skills(
        self,
        finish_payload: Dict[str, Any],
        request: ExploreRequest,
    ) -> tuple:
        """Validate ``commit_skills`` refs and render the ones worth handing over.

        Rejection reasons, in order:
          1. unresolvable ref (hallucination) -> rejected
          2. already injected into the main agent's context -> skipped
          3. duplicate within this call -> skipped
          4. over ``max_commit_skills`` -> skipped

        Committing a skill never exposes its tools (see ExplorerConfig).
        """
        raw = finish_payload.get("commit_skills") or []
        if not isinstance(raw, list):
            raw = [raw]

        requested: List[str] = []
        for item in raw:
            ref = str(item or "").strip()
            if ref and ref not in requested:
                requested.append(ref)

        already = {str(name).strip() for name in (request.injected_skill_names or [])}
        limit = max(int(self.config.max_commit_skills), 0)

        accepted: List[Dict[str, Any]] = []
        rejected: List[str] = []
        seen_refs: set = set()
        for ref in requested:
            if len(accepted) >= limit:
                break
            entry = self.backend.resolve_committed_skill(ref)
            if entry is None:
                rejected.append(ref)
                continue
            canonical = str(entry.get("ref") or "")
            if canonical in seen_refs:
                continue
            if str(entry.get("name") or "") in already:
                continue
            seen_refs.add(canonical)
            accepted.append(entry)

        if rejected:
            logger.debug("Explorer rejected unresolvable skill refs: %s", rejected)
        return accepted, requested, rejected

    def _fallback_tool_cards(self) -> List[Dict[str, Any]]:
        # Unverified candidates: keep the list short so a rare fallback cannot
        # inflate the visible tool set and wreck retrieval precision.
        limit = min(int(self.config.max_commit_tools), 3)
        return self.backend.fallback_candidates(limit=limit)
