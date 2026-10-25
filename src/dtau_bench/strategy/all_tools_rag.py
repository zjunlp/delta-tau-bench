"""
All-tools iterative RAG strategy.

This strategy loads all tools from all domains but only exposes a subset
to the Agent based on semantic retrieval. The Agent can dynamically
retrieve more tools by calling the virtual `retrieve_tools` tool.

Flow:
1. start_session: RAG retrieves top_k_init tools based on initial user query
2. get_agent_view: Returns visible tools + the `retrieve_tools` virtual tool
3. intercept_tool_calls: Intercepts `retrieve_tools` calls, performs RAG,
   merges new tools into visible set, returns synthetic response
4. get_artifacts: Returns retrieval trace for evaluation

Used for:
- Ablation experiments (RAG-only, no skills)
- Evaluating tool retrieval performance at scale (all tools from all domains)
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

from dtau_bench.rag.runtime import ToolRAGRuntime
from dtau_bench.strategy.base import (
    AgentStrategy,
    StrategyRuntimeContext,
    StrategySessionState,
    StrategyToolInterceptionResult,
    StrategyView,
)
from dtau_bench.strategy.registry import register_strategy
from dtau_bench.strategy.single_domain import (
    extract_initial_user_query,
    filter_tools_info_by_names,
    merge_tool_names_preserve_order,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Virtual Tool Definition
# ============================================================================


def build_retrieve_tools_definition(tool_name: str = "retrieve_tools") -> Dict[str, Any]:
    """Build the OpenAI-format tool definition for the virtual retrieve_tools tool.

    This virtual tool is injected into the agent's visible tools. When the agent
    calls it, the strategy intercepts the call and performs RAG retrieval instead
    of executing it on an MCP server.
    """
    return {
        "type": "function",
        "function": {
            "name": tool_name,
            "description": (
                "Retrieve additional tools when the current tool list is insufficient. "
                "Pass one short, specific natural-language query describing the tool "
                "capability you still need. "
                "Use this only when the currently available tools do not cover the task."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "A short, specific description of the tool capability "
                            "you want to retrieve."
                        ),
                    }
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    }


# ============================================================================
# Strategy Implementation
# ============================================================================


@register_strategy("all_tools_iterative_rag")
class AllToolsIterativeRAGStrategy(AgentStrategy):
    """All-tools iterative RAG strategy.

    On session start, performs an initial RAG retrieval using the user's query
    to select top_k_init tools. The Agent also receives a virtual `retrieve_tools`
    tool that it can call to retrieve additional tools during the conversation.

    Config (from all_tools_iterative_rag.yaml):
        retrieval:
            embedding_model: Qwen3-Embedding-0.6B
            top_k_init: 15
            top_k_turn: 5
            similarity_threshold: 0.0
            embedding_backend: openai_http
            embedding_api_base: http://...
            index_dir: path/to/index
        virtual_tools:
            - retrieve_tools
    """

    def __init__(self, config: Dict[str, Any] = None, **kwargs):
        config = config or {}
        super().__init__("all_tools_iterative_rag", config)

        # Extract retrieval sub-config
        retrieval_cfg = config.get("retrieval", {})

        # RAG configuration parameters
        self.top_k_init = int(retrieval_cfg.get("top_k_init", retrieval_cfg.get("top_k", 15)))
        self.top_k_turn = int(retrieval_cfg.get("top_k_turn", retrieval_cfg.get("top_k", 5)))

        # Virtual tool name
        virtual_tools = config.get("virtual_tools", ["retrieve_tools"])
        self.retrieval_tool_name = (
            virtual_tools[0] if virtual_tools else "retrieve_tools"
        )

        # Initialize RAG runtime
        # The runtime is created lazily or from config to allow for index_dir override
        self._runtime: ToolRAGRuntime | None = None
        self._retrieval_config = retrieval_cfg

    def _get_runtime(self) -> ToolRAGRuntime:
        """Lazy initialization of RAG runtime."""
        if self._runtime is None:
            self._runtime = ToolRAGRuntime.from_config(self._retrieval_config)
        return self._runtime

    def set_runtime(self, runtime: ToolRAGRuntime) -> None:
        """Inject a pre-built runtime (useful for testing or sharing across domains)."""
        self._runtime = runtime

    # ─────────────────────────────────────────────────────────────────────────
    # Lifecycle: start_session
    # ─────────────────────────────────────────────────────────────────────────

    def start_session(
        self,
        sample: Dict[str, Any],
        all_tools_info: List[Dict[str, Any]],
        runtime_context: StrategyRuntimeContext,
    ) -> StrategySessionState:
        """Initialize session with initial RAG retrieval.

        Retrieves top_k_init tools based on the initial user query.
        """
        runtime = self._get_runtime()

        # Extract query from initial messages
        query_text = extract_initial_user_query(sample.get("messages", []))

        # Initial retrieval
        retrieval = runtime.retrieve_tools(
            query_text=query_text,
            all_tools_info=all_tools_info,
            top_k=self.top_k_init,
        )

        # Build session state
        state = StrategySessionState(
            sample_id=sample.get("id", ""),
            query_text=query_text,
            visible_tool_names=list(retrieval["tool_names"]),
            visible_tools_info=list(retrieval["tools_info"]),
            prepended_messages=[],
            custom_state={
                "all_tools_info": list(all_tools_info),
                "retrieval_tool_name": self.retrieval_tool_name,
                "retrieval_call_count": 0,
            },
        )

        # Record initial retrieval round
        initial_round = dict(retrieval["trace"])
        initial_round["type"] = "tool_rag_initial"
        initial_round["top_k"] = self.top_k_init
        initial_round["newly_added_tool_names"] = list(retrieval["tool_names"])
        initial_round["newly_added_tool_count"] = len(retrieval["tool_names"])
        initial_round["is_internal_retrieval_tool_round"] = False
        state.retrieval_rounds.append(initial_round)

        # Debug history
        state.debug_history.append({
            "round": 0,
            "strategy": self.strategy_name,
            "iterative_retrieval_round_count": 0,
            "iterative_visible_tool_names": list(state.visible_tool_names),
            "iterative_visible_tool_count": len(state.visible_tool_names),
            **retrieval["debug"],
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
        """Return visible tools + the virtual retrieve_tools tool."""
        latest = state.debug_history[-1] if state.debug_history else {}

        # Build visible tools list with the virtual retrieve_tools appended
        retrieval_tool_name = str(
            state.custom_state.get("retrieval_tool_name", self.retrieval_tool_name)
        )
        retrieval_tool_info = build_retrieve_tools_definition(retrieval_tool_name)

        visible_tools_info = list(state.visible_tools_info or [])
        visible_tools_info.append(retrieval_tool_info)

        return StrategyView(
            visible_tools_info=visible_tools_info,
            prepended_messages=[],
            metadata=latest,
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
        """Intercept retrieve_tools calls and perform RAG retrieval.

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
                runtime = self._get_runtime()
                retrieval = runtime.retrieve_tools(
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
                # Empty query - return error response
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
        """Return retrieval artifacts for output.

        Matches the plan's output schema:
        artifacts.retrieval.rounds = [
            {round_index, query, retrieved_tool_names, final_visible_tool_names, ...}
        ]
        """
        # Build simplified retrieval rounds for output
        rounds = []
        for i, round_trace in enumerate(state.retrieval_rounds):
            rounds.append({
                "round_index": i,
                "type": round_trace.get("type", "tool_rag"),
                "query": round_trace.get("query", ""),
                "top_k": round_trace.get("top_k", 0),
                "retrieved_tool_names": round_trace.get("final_visible_tool_names", []),
                "newly_added_tool_names": round_trace.get("newly_added_tool_names", []),
                "newly_added_tool_count": round_trace.get("newly_added_tool_count", 0),
                "final_visible_tool_names": round_trace.get("final_visible_tool_names", []),
                "final_visible_tool_count": round_trace.get("final_visible_tool_count", 0),
                "is_internal_retrieval_tool_round": round_trace.get(
                    "is_internal_retrieval_tool_round", False
                ),
            })

        return {
            "retrieval": {
                "rounds": rounds,
                "total_retrieval_calls": int(
                    state.custom_state.get("retrieval_call_count", 0)
                ),
                "final_visible_tool_count": len(state.visible_tool_names),
                "final_visible_tool_names": list(state.visible_tool_names),
            }
        }
