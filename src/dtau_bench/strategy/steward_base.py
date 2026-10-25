"""
Skill context + iterative tool retrieval.

At session start the skill library is queried for relevant skills, which are
injected as prepended context, and their associated tools seed the agent's
visible tool set. During the conversation the agent can call the virtual
``retrieve_tools`` tool to pull more tools from the full cross-domain catalog.

Configured under ``skill_context``, ``retrieval`` and ``virtual_tools`` in
configs/strategies/steward.yaml. Not registered as a selectable strategy:
``steward`` is the concrete subclass.
"""

from __future__ import annotations

import json
import logging
import threading
from typing import Any, Dict, List, Optional

from dtau_bench.rag.runtime import ToolRAGRuntime
from dtau_bench.skill.runtime import SkillContextRuntime
from dtau_bench.strategy.base import (
    AgentStrategy,
    StrategyRuntimeContext,
    StrategySessionState,
    StrategyToolInterceptionResult,
    StrategyView,
)
from dtau_bench.strategy.single_domain import (
    extract_initial_user_query,
    filter_tools_info_by_names,
    merge_tool_names_preserve_order,
)
from dtau_bench.strategy.all_tools_rag import build_retrieve_tools_definition
from dtau_bench.utils.llm import LLMClient, LLMConfig

logger = logging.getLogger(__name__)


class StewardBaseStrategy(AgentStrategy):
    """Skill context injection plus iterative tool retrieval."""

    #: Set by the concrete subclass that registers itself.
    STRATEGY_NAME = "steward"

    def __init__(self, config: Dict[str, Any] = None, **kwargs):
        config = config or {}
        super().__init__(type(self).STRATEGY_NAME, config)

        # Extract sub-configs
        self._skill_context_config = config.get("skill_context", {})
        self._retrieval_config = config.get("retrieval", {})

        # Virtual tool name
        virtual_tools = config.get("virtual_tools", ["retrieve_tools"])
        self.retrieval_tool_name = (
            virtual_tools[0] if virtual_tools else "retrieve_tools"
        )

        # Top-k for iterative retrieval
        self.top_k_turn = int(self._retrieval_config.get("top_k_turn", 5))

        # Tool top-k per pseudo-plan step (initial retrieval)
        self.tool_top_k_per_step = int(self._skill_context_config.get("tool_top_k_per_step", 3))

        # Lazy-initialized runtimes
        self._skill_context_runtime: Optional[SkillContextRuntime] = None
        self._tool_rag_runtime: Optional[ToolRAGRuntime] = None
        self._skill_runtime_init_lock = threading.Lock()
        self._tool_rag_init_lock = threading.Lock()

    # ─────────────────────────────────────────────────────────────────────────
    # Runtime Initialization (lazy)
    # ─────────────────────────────────────────────────────────────────────────

    def _get_skill_context_runtime(self) -> SkillContextRuntime:
        """Lazy initialization of SkillContextRuntime."""
        if self._skill_context_runtime is not None:
            return self._skill_context_runtime
        with self._skill_runtime_init_lock:
            if self._skill_context_runtime is not None:
                return self._skill_context_runtime

            skill_context_cfg = dict(self._skill_context_config)
            if not skill_context_cfg.get("skill_library_dir"):
                raise ValueError(
                    "This strategy requires 'skill_library_dir' in skill_context config. "
                    "Pass --skill-library <path> on the CLI."
                )
            if not skill_context_cfg.get("index_dir"):
                raise ValueError(
                    "This strategy requires 'index_dir' in skill_context config. "
                    "Build the index first with python -m dtau_bench build-skill-index."
                )

            runtime = SkillContextRuntime.from_config(
                skill_context_config=skill_context_cfg,
                llm_client=LLMClient(self._build_llm_config()),
                embedding_config=self.config.get("_embedding_config"),
            )
            self._skill_context_runtime = runtime
            logger.info("StewardBaseStrategy: SkillContextRuntime initialized")
            return runtime

    def _get_tool_rag_runtime(self) -> ToolRAGRuntime:
        """Initialize the required ToolRAGRuntime exactly once."""
        if self._tool_rag_runtime is not None:
            return self._tool_rag_runtime
        with self._tool_rag_init_lock:
            if self._tool_rag_runtime is not None:
                return self._tool_rag_runtime

            retrieval_cfg = dict(self._retrieval_config)
            if not retrieval_cfg.get("index_dir"):
                raise ValueError(
                    "Steward requires retrieval.index_dir. Build the tool index "
                    "before starting rollout."
                )

            embedding_cfg = self.config.get("_embedding_config", {})
            if embedding_cfg:
                if "embedding_api_base" not in retrieval_cfg and embedding_cfg.get("base_url"):
                    retrieval_cfg["embedding_api_base"] = embedding_cfg["base_url"]
                if "embedding_backend" not in retrieval_cfg and embedding_cfg.get("backend"):
                    backend = embedding_cfg["backend"]
                    if backend == "openai_compatible":
                        backend = "openai_http"
                    retrieval_cfg["embedding_backend"] = backend
                if "embedding_model" not in retrieval_cfg and embedding_cfg.get("model"):
                    retrieval_cfg["embedding_model"] = embedding_cfg["model"]
                if "embedding_api_key" not in retrieval_cfg and embedding_cfg.get("api_key"):
                    retrieval_cfg["embedding_api_key"] = embedding_cfg["api_key"]

            runtime = ToolRAGRuntime.from_config(retrieval_cfg)
            self._tool_rag_runtime = runtime
            logger.info("StewardBaseStrategy: ToolRAGRuntime initialized")
            return runtime

    def prewarm(self) -> None:
        """Load shared retrieval infrastructure before task workers start."""
        self._get_skill_context_runtime()
        self._get_tool_rag_runtime()

    def _build_llm_config(self) -> LLMConfig:
        """Build LLM config for skill retrieval LLM calls."""
        llm_cfg = self.config.get("_llm_config", {})
        return LLMConfig(
            model_name=llm_cfg.get("model_name", "DeepSeek-V3.2"),
            base_url=llm_cfg.get("base_url", "http://localhost:8000/v1"),
            api_key=llm_cfg.get("api_key", "EMPTY"),
            max_tokens=llm_cfg.get("max_tokens", 4096),
            temperature=0.7,
            timeout=llm_cfg.get("timeout", 180),
            max_retries=2,
            extra_params=dict(llm_cfg.get("request_options") or {}),
        )

    def set_skill_context_runtime(self, runtime: SkillContextRuntime) -> None:
        """Inject a pre-built skill context runtime (for testing)."""
        self._skill_context_runtime = runtime

    def set_tool_rag_runtime(self, runtime: ToolRAGRuntime) -> None:
        """Inject a pre-built ToolRAG runtime (for testing)."""
        self._tool_rag_runtime = runtime

    # ─────────────────────────────────────────────────────────────────────────
    # Lifecycle: start_session
    # ─────────────────────────────────────────────────────────────────────────

    def start_session(
        self,
        sample: Dict[str, Any],
        all_tools_info: List[Dict[str, Any]],
        runtime_context: StrategyRuntimeContext,
    ) -> StrategySessionState:
        """Initialize session with skill context + initial tool retrieval.

        Steps:
        1. Extract user query
        2. Run skill context retrieval → skill context prompt + initial tool recommendations
        3. Use ToolRAGRuntime (if available) for per-step tool retrieval
        4. Merge initial tools
        5. Store skill context as prepended message
        """
        # Extract query
        query_text = extract_initial_user_query(sample.get("messages", []))

        # Initialize state (tools will be populated below)
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

        # Retrieval is required infrastructure.  Let any initialization or
        # query error fail the sample instead of silently running with no real
        # tools and misreporting an infrastructure failure as model quality.
        skill_context_runtime = self._get_skill_context_runtime()
        tool_rag_runtime = self._get_tool_rag_runtime()

        prompt_msg, initial_tool_names, stats = (
            skill_context_runtime.build_all_tools_context_for_query(
                query_text=query_text,
                all_tools_info=all_tools_info,
                tool_rag_runtime=tool_rag_runtime,
                tool_top_k_per_step=self.tool_top_k_per_step,
            )
        )

        if prompt_msg:
            state.prepended_messages = [prompt_msg]
            state.custom_state["skill_context_prompt"] = prompt_msg

        state.visible_tool_names = initial_tool_names
        state.visible_tools_info = filter_tools_info_by_names(
            all_tools_info, initial_tool_names
        )

        state.custom_state["skill_context_stats"] = stats
        state.custom_state["skill_context_trace"] = stats.get("retrieval_trace", {})

        logger.info(
            f"StewardBase: {stats.get('selected_skills_count', 0)} skills, "
            f"{len(initial_tool_names)} initial tools for sample {state.sample_id}"
        )

        # Record initial retrieval round
        initial_round = {
            "type": "skill_context_initial",
            "query": query_text,
            "top_k": self.tool_top_k_per_step,
            "final_visible_tool_names": list(state.visible_tool_names),
            "final_visible_tool_count": len(state.visible_tool_names),
            "newly_added_tool_names": list(state.visible_tool_names),
            "newly_added_tool_count": len(state.visible_tool_names),
            "is_internal_retrieval_tool_round": False,
        }
        state.retrieval_rounds.append(initial_round)

        # Debug history
        state.debug_history.append({
            "round": 0,
            "strategy": self.strategy_name,
            "visible_tool_count": len(state.visible_tool_names),
            "visible_tool_names": list(state.visible_tool_names),
            "skill_context_selected_skills_count": state.custom_state.get("skill_context_stats", {}).get(
                "selected_skills_count", 0
            ),
            "skill_context_selected_skill_names": state.custom_state.get("skill_context_stats", {}).get(
                "selected_skill_names", []
            ),
            "iterative_retrieval_round_count": 0,
        })

        return state

    # ─────────────────────────────────────────────────────────────────────────
    # Lifecycle: get_agent_view
    # ─────────────────────────────────────────────────────────────────────────

    def get_agent_view(
        self,
        state: StrategySessionState,
        conversation: List[Dict[str, Any]],
    ) -> StrategyView:
        """Return visible tools + retrieve_tools virtual tool + skill context."""
        latest_debug = state.debug_history[-1] if state.debug_history else {}

        # Build visible tools list with the virtual retrieve_tools appended
        retrieval_tool_name = str(
            state.custom_state.get("retrieval_tool_name", self.retrieval_tool_name)
        )
        retrieval_tool_info = build_retrieve_tools_definition(retrieval_tool_name)

        visible_tools_info = list(state.visible_tools_info or [])
        visible_tools_info.append(retrieval_tool_info)

        return StrategyView(
            visible_tools_info=visible_tools_info,
            prepended_messages=list(state.prepended_messages),
            metadata=latest_debug,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Lifecycle: intercept_tool_calls
    # ─────────────────────────────────────────────────────────────────────────

    def intercept_tool_calls(
        self,
        state: StrategySessionState,
        tool_calls: List[Dict[str, Any]],
        conversation: List[Dict[str, Any]],
    ) -> StrategyToolInterceptionResult:
        """Intercept retrieve_tools calls and perform iterative tool RAG.

        For each `retrieve_tools` call:
        1. Extract the query argument
        2. Perform RAG retrieval (top_k_turn)
        3. Merge new tools into visible set
        4. Return a synthetic tool response to the Agent
        """
        handled_tool_responses: List[Dict[str, Any]] = []
        remaining_tool_calls: List[Dict[str, Any]] = []
        turn_rounds: List[Dict[str, Any]] = []

        retrieval_tool_name = str(
            state.custom_state.get("retrieval_tool_name", self.retrieval_tool_name)
        )
        all_tools_info = state.custom_state.get("all_tools_info") or []
        if not isinstance(all_tools_info, list):
            all_tools_info = []

        for tc in tool_calls:
            function = tc.get("function") or {}
            function_name = str(function.get("name") or "").strip()

            # Not a retrieval tool call — pass through
            if function_name != retrieval_tool_name:
                remaining_tool_calls.append(tc)
                continue

            # Parse arguments
            raw_arguments = function.get("arguments", {})
            if isinstance(raw_arguments, str):
                try:
                    parsed_arguments = json.loads(raw_arguments)
                except json.JSONDecodeError:
                    parsed_arguments = {}
            elif isinstance(raw_arguments, dict):
                parsed_arguments = raw_arguments
            else:
                parsed_arguments = {}

            query = str(parsed_arguments.get("query") or "").strip()

            if query:
                # Perform RAG retrieval
                tool_rag = self._get_tool_rag_runtime()
                retrieval = tool_rag.retrieve_tools(
                    query_text=query,
                    all_tools_info=all_tools_info,
                    top_k=self.top_k_turn,
                )

                # Merge new tools into visible set
                new_visible_names = merge_tool_names_preserve_order(
                    state.visible_tool_names,
                    retrieval["tool_names"],
                )
                newly_added = [
                    name
                    for name in new_visible_names
                    if name not in set(state.visible_tool_names)
                ]

                # Update state
                state.visible_tool_names = new_visible_names
                state.visible_tools_info = filter_tools_info_by_names(
                    all_tools_info, new_visible_names
                )
                state.custom_state["retrieval_call_count"] = int(
                    state.custom_state.get("retrieval_call_count", 0)
                ) + 1

                # Record retrieval round
                round_trace = dict(retrieval["trace"])
                round_trace["type"] = "tool_rag_turn"
                round_trace["top_k"] = self.top_k_turn
                round_trace["trigger_tool_call_id"] = tc.get("id")
                round_trace["newly_added_tool_names"] = newly_added
                round_trace["newly_added_tool_count"] = len(newly_added)
                round_trace["final_visible_tool_names"] = list(state.visible_tool_names)
                round_trace["final_visible_tool_count"] = len(state.visible_tool_names)
                round_trace["is_internal_retrieval_tool_round"] = True
                state.retrieval_rounds.append(round_trace)
                turn_rounds.append(round_trace)

                # Debug history
                state.debug_history.append({
                    "round": len(state.debug_history),
                    "strategy": self.strategy_name,
                    "iterative_retrieval_round_count": max(
                        len(state.retrieval_rounds) - 1, 0
                    ),
                    "iterative_visible_tool_names": list(state.visible_tool_names),
                    "iterative_visible_tool_count": len(state.visible_tool_names),
                    "last_iterative_retrieval_query": query,
                    "last_iterative_new_tool_names": newly_added,
                    "last_iterative_new_tool_count": len(newly_added),
                })

                # Build response payload
                response_payload = {
                    "retrieval_query": query,
                    "new_tool_names": newly_added,
                    "new_tool_count": len(newly_added),
                    "all_visible_tool_names": list(state.visible_tool_names),
                    "all_visible_tool_count": len(state.visible_tool_names),
                }
            else:
                # Empty query
                response_payload = {
                    "retrieval_query": query,
                    "new_tool_names": [],
                    "new_tool_count": 0,
                    "all_visible_tool_names": list(state.visible_tool_names),
                    "all_visible_tool_count": len(state.visible_tool_names),
                    "error": "retrieve_tools requires a non-empty query",
                }

            # Build tool response message
            handled_tool_responses.append({
                "role": "tool",
                "content": json.dumps(response_payload, ensure_ascii=False),
                "tool_call_id": tc.get("id"),
            })

        # Metadata from this interception
        metadata = {}
        if turn_rounds:
            latest_round = turn_rounds[-1]
            metadata.update({
                "iterative_retrieval_round_count": max(
                    len(state.retrieval_rounds) - 1, 0
                ),
                "iterative_visible_tool_names": latest_round.get(
                    "final_visible_tool_names", []
                ),
                "iterative_visible_tool_count": latest_round.get(
                    "final_visible_tool_count", 0
                ),
            })

        return StrategyToolInterceptionResult(
            handled_tool_responses=handled_tool_responses,
            remaining_tool_calls=remaining_tool_calls,
            metadata=metadata,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Lifecycle: get_artifacts
    # ─────────────────────────────────────────────────────────────────────────

    def get_artifacts(self, state: StrategySessionState) -> Dict[str, Any]:
        """Return both skill context and retrieval artifacts for output.

        Matches the plan's output schema:
        artifacts.skill_context = { pseudo_plan, selected_skill_names, retrieval_trace }
        artifacts.retrieval = { rounds, total_retrieval_calls, ... }
        """
        stats = state.custom_state.get("skill_context_stats", {})
        trace = state.custom_state.get("skill_context_trace", {})

        # skill context artifacts
        skill_context_artifacts = {
            "pseudo_plan": stats.get("pseudo_plan", ""),
            "selected_skill_names": stats.get("selected_skill_names", []),
            "selected_skills_count": stats.get("selected_skills_count", 0),
            "graph_enabled": stats.get("graph_enabled", False),
            "skill_tool_names": stats.get("skill_tool_names", []),
            "per_step_rag_tool_names": stats.get("per_step_rag_tool_names", []),
            "initial_tool_count": stats.get("initial_tool_count", 0),
            "retrieval_trace": trace,
        }

        # Retrieval artifacts (iterative rounds)
        rounds = []
        for i, round_trace in enumerate(state.retrieval_rounds):
            rounds.append({
                "round_index": i,
                "type": round_trace.get("type", "tool_rag"),
                "query": round_trace.get("query", ""),
                "top_k": round_trace.get("top_k", 0),
                "newly_added_tool_names": round_trace.get("newly_added_tool_names", []),
                "newly_added_tool_count": round_trace.get("newly_added_tool_count", 0),
                "final_visible_tool_names": round_trace.get("final_visible_tool_names", []),
                "final_visible_tool_count": round_trace.get("final_visible_tool_count", 0),
                "is_internal_retrieval_tool_round": round_trace.get(
                    "is_internal_retrieval_tool_round", False
                ),
            })

        retrieval_artifacts = {
            "rounds": rounds,
            "total_retrieval_calls": int(
                state.custom_state.get("retrieval_call_count", 0)
            ),
            "final_visible_tool_count": len(state.visible_tool_names),
            "final_visible_tool_names": list(state.visible_tool_names),
        }

        return {
            "skill_context": skill_context_artifacts,
            "retrieval": retrieval_artifacts,
        }
