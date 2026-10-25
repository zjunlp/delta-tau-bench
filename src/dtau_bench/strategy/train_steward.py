"""
Single-domain skill context strategy, used for training rollouts.

All tools of the current domain are visible; the skill library is queried once
at session start and the resulting skills are injected as prepended context.
There is no interception and no per-turn update.

Configured under ``skill_context`` in configs/strategies/train_steward.yaml.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from dtau_bench.skill.runtime import SkillContextRuntime
from dtau_bench.strategy.base import (
    AgentStrategy,
    StrategyRuntimeContext,
    StrategySessionState,
    StrategyView,
)
from dtau_bench.strategy.registry import register_strategy
from dtau_bench.strategy.single_domain import (
    extract_initial_user_query,
    tool_name_from_info,
)
from dtau_bench.utils.llm import LLMClient, LLMConfig

logger = logging.getLogger(__name__)


@register_strategy("train_steward")
class TrainStewardStrategy(AgentStrategy):
    """Single-domain skill context strategy.

    All tools from the current domain are visible. Additionally, relevant
    skills are retrieved and injected as context to guide the Agent.

    The skill retrieval happens ONCE at session start (not iterative).
    The skill context is a system message prepended to the conversation.

    Config (via strategy_config["skill_context"]):
        - skill_library_dir: Path to skill library directory
        - index_dir: Path to pre-built skill context index
        - plan_top_k: Top-k planning skills to retrieve (default: 3)
        - skill_top_k_per_step: Max atomic skills per pseudo-plan step (default: 5)
        - final_max_skills: Max skills to select (default: 8)
        - dedup_similarity_threshold: Dedup threshold (default: 0.95)
        - mmr_lambda: MMR lambda (default: 0.75)
        - candidate_min_similarity: Min similarity for candidates (default: 0.45)
        - enable_skill_graph: Use skill graph (default: True)
        - embedding_model: Model name
        - embedding_backend: "openai_http" or "local"
        - embedding_api_base: API base URL
    """

    def __init__(self, config: Dict[str, Any] = None, **kwargs):
        config = config or {}
        super().__init__("train_steward", config)

        # skill context runtime is created lazily (needs skill_library_dir and index_dir)
        self._runtime: Optional[SkillContextRuntime] = None
        self._runtime_init_lock = threading.Lock()
        self._skill_context_config = config.get("skill_context", {})

    def _get_runtime(self) -> SkillContextRuntime:
        """Lazy initialization of SkillContextRuntime.

        The runtime is created once and shared across all sessions.
        It loads the skill indices from disk and initializes the LLM client.
        """
        if self._runtime is not None:
            return self._runtime
        with self._runtime_init_lock:
            if self._runtime is not None:
                return self._runtime

            skill_context_cfg = dict(self._skill_context_config)
            if not skill_context_cfg.get("skill_library_dir"):
                raise ValueError(
                    "train_steward requires 'skill_library_dir' in skill_context config. "
                    "Pass --skill-library <path> on the CLI."
                )
            if not skill_context_cfg.get("index_dir"):
                raise ValueError(
                    "train_steward requires 'index_dir' in skill_context config. "
                    "Build the index first with python -m dtau_bench build-skill-index."
                )

            runtime = SkillContextRuntime.from_config(
                skill_context_config=skill_context_cfg,
                llm_client=LLMClient(self._build_llm_config()),
                embedding_config=self.config.get("_embedding_config"),
            )
            self._runtime = runtime
            logger.info("TrainStewardStrategy: SkillContextRuntime initialized")
            return runtime

    def prewarm(self) -> None:
        """Load the shared skill runtime before task workers start."""
        self._get_runtime()

    def _build_llm_config(self) -> LLMConfig:
        """Build LLM config for skill retrieval LLM calls.

        Uses the agent model config if available, otherwise defaults.
        """
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

    def set_runtime(self, runtime: SkillContextRuntime) -> None:
        """Inject a pre-built runtime (for testing)."""
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
        """Initialize session with skill retrieval.

        Runs the full skill context pipeline:
        1. Extract user query
        2. Retrieve relevant skills
        3. Build skill context prompt
        4. Expose all domain tools + skill context
        """
        # Extract query
        query_text = extract_initial_user_query(sample.get("messages", []))

        # All tools visible (single-domain mode)
        visible_names = [
            name for name in (tool_name_from_info(item) for item in all_tools_info) if name
        ]

        # Initialize state
        state = StrategySessionState(
            sample_id=sample.get("id", ""),
            query_text=query_text,
            visible_tool_names=visible_names,
            visible_tools_info=list(all_tools_info),
            prepended_messages=[],
            custom_state={
                "skill_context_prompt": None,
                "skill_context_stats": {},
                "skill_context_trace": {},
            },
        )

        # Skill retrieval is required infrastructure for train_steward.  A
        # failure must fail the sample rather than silently changing strategy.
        runtime = self._get_runtime()
        prompt_msg, stats = runtime.build_context_for_query(
            query_text=query_text,
            tools_info=all_tools_info,
        )

        if prompt_msg:
            state.prepended_messages = [prompt_msg]
            state.custom_state["skill_context_prompt"] = prompt_msg

        state.custom_state["skill_context_stats"] = stats
        state.custom_state["skill_context_trace"] = stats.get("retrieval_trace", {})

        logger.info(
            f"skill context retrieved {stats.get('selected_skills_count', 0)} skills "
            f"for sample {state.sample_id}"
        )

        # Debug history
        state.debug_history.append({
            "round": 0,
            "strategy": self.strategy_name,
            "visible_tool_count": len(visible_names),
            "skill_context_selected_skills_count": state.custom_state.get("skill_context_stats", {}).get(
                "selected_skills_count", 0
            ),
            "skill_context_selected_skill_names": state.custom_state.get("skill_context_stats", {}).get(
                "selected_skill_names", []
            ),
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
        """Return all tools + skill context as prepended message."""
        latest_debug = state.debug_history[-1] if state.debug_history else {}
        return StrategyView(
            visible_tools_info=state.visible_tools_info,
            prepended_messages=list(state.prepended_messages),
            metadata=latest_debug,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Lifecycle: get_artifacts
    # ─────────────────────────────────────────────────────────────────────────

    def get_artifacts(self, state: StrategySessionState) -> Dict[str, Any]:
        """Return skill context artifacts for output.

        Matches the plan's output schema artifacts.skill_context:
        {
            pseudo_plan: str,
            selected_skill_names: [...],
            selected_skill_contents: [...],
            retrieval_trace: { planning, step_skill_retrieval, dedup, mmr, ... }
        }
        """
        stats = state.custom_state.get("skill_context_stats", {})
        trace = state.custom_state.get("skill_context_trace", {})

        return {
            "skill_context": {
                "pseudo_plan": stats.get("pseudo_plan", ""),
                "selected_skill_names": stats.get("selected_skill_names", []),
                "selected_skills_count": stats.get("selected_skills_count", 0),
                "graph_enabled": stats.get("graph_enabled", False),
                "retrieval_trace": trace,
            }
        }
