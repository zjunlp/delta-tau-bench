"""Data structures for the explorer subagent."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class ExplorerConfig:
    """Explorer configuration (``explorer`` block of the strategy YAML)."""

    enabled: bool = True
    tool_name: str = "explore_capabilities"

    # Session-level budget (trigger scheme B: main agent decides when to call)
    max_explore_calls_per_session: int = 3

    # Per-call budget
    # NOTE: models in practice emit ONE action per LLM turn, so max_llm_turns must
    # be >= max_actions or the action budget can never be reached. Keep a small
    # margin for turns where the model replies without calling an action.
    max_actions: int = 20
    max_llm_turns: int = 24
    hard_timeout_s: float = 360.0

    # Action top-k caps
    search_skills_top_k: int = 5
    search_tools_top_k: int = 8
    expand_graph_limit: int = 8

    # Compression (three levels, see design section 6)
    max_skill_chars: int = 1500
    # NOTE: 1200 caused a 34%/36% (v1/v2) report truncation rate, and truncation
    # drops whole sections by keep-priority (WORKFLOW / RULED_OUT go first), i.e.
    # exactly the reasoning content the main agent needs. Per-section entry caps
    # (report._MAX_ENTRIES) remain the real content bound.
    max_report_chars: int = 6400
    max_commit_tools: int = 8
    card_summary_chars: int = 120

    # Injection
    initial_injection: str = "full"  # full | compact | none
    persist_report_as_system: bool = False

    # ── Skill hand-over to the main agent (P2) ──
    # The explorer may return skills it judges useful; they are rendered into an
    # actionable block and injected as a prepended system message that is
    # REBUILT (never appended to) on every commit.
    max_commit_skills: int = 2
    max_committed_skill_chars_per_session: int = 4000
    persist_skills_as_system: bool = True
    # Ablation-only. Committing a skill must not auto-expose its tools: the
    # predicted tool set feeds the retrieval metric and one skill may reference
    # 3-6 tools, which would wreck precision.
    commit_skill_auto_tools: bool = False

    # Input context (design section 4.1)
    recent_user_messages: int = 3
    recent_tool_errors: int = 3
    include_pseudo_plan: bool = True
    exclude_injected_skills: bool = True
    max_context_chars: int = 1200

    # Safety: schema-only. Never execute a real tool.
    allow_readonly_probe: bool = False

    # Tracing
    trace_dir_name: str = "explore_traces"
    trace_full_messages: bool = True

    # LLM
    temperature: float = 0.3

    @classmethod
    def from_dict(cls, payload: Optional[Dict[str, Any]]) -> "ExplorerConfig":
        payload = payload or {}
        config = cls()
        for key, value in payload.items():
            if not hasattr(config, key):
                continue
            current = getattr(config, key)
            try:
                if isinstance(current, bool):
                    coerced: Any = bool(value)
                elif isinstance(current, int) and not isinstance(current, bool):
                    coerced = int(value)
                elif isinstance(current, float):
                    coerced = float(value)
                else:
                    coerced = value
            except (TypeError, ValueError):
                continue
            setattr(config, key, coerced)

        normalized = str(config.initial_injection or "full").strip().lower()
        if normalized not in {"full", "compact", "none"}:
            normalized = "full"
        config.initial_injection = normalized

        # Guard against the budget being unreachable: models typically emit one
        # action per LLM turn, so max_llm_turns < max_actions silently turns
        # max_llm_turns into the real (and misleading) cap.
        if config.max_llm_turns < config.max_actions:
            logger.warning(
                "explorer.max_llm_turns (%d) < explorer.max_actions (%d): the action "
                "budget would be unreachable. Raising max_llm_turns to %d.",
                config.max_llm_turns,
                config.max_actions,
                config.max_actions + 4,
            )
            config.max_llm_turns = config.max_actions + 4

        if config.allow_readonly_probe:
            # Hard safety guard: executing real tools would corrupt the shared
            # per-domain database and break gold-replay / DB-diff evaluation.
            raise ValueError(
                "explorer.allow_readonly_probe=true is not implemented. "
                "The explorer must stay schema-only."
            )
        return config


# ============================================================================
# Request / Result
# ============================================================================


@dataclass
class ExploreRequest:
    """Everything the explorer is allowed to see.

    NOTE: the simulated user's ``user_system_prompt`` must never appear here.
    """

    goal: str
    missing_capability: str
    known_failures: str = ""
    turn_index: int = 0
    call_index: int = 0
    sample_id: str = ""
    domain: str = ""
    version: str = ""

    # Session state injected by the strategy (not by the main agent)
    visible_tool_names: List[str] = field(default_factory=list)
    injected_skill_names: List[str] = field(default_factory=list)
    recent_user_messages: List[str] = field(default_factory=list)
    recent_tool_errors: List[Dict[str, str]] = field(default_factory=list)
    pseudo_plan: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "goal": self.goal,
            "missing_capability": self.missing_capability,
            "known_failures": self.known_failures,
            "turn_index": self.turn_index,
            "call_index": self.call_index,
            "sample_id": self.sample_id,
            "domain": self.domain,
            "version": self.version,
            "visible_tools_before": list(self.visible_tool_names),
            "injected_skill_names": list(self.injected_skill_names),
            "recent_user_messages": list(self.recent_user_messages),
            "recent_tool_errors": list(self.recent_tool_errors),
            "pseudo_plan": self.pseudo_plan,
        }


@dataclass
class ActionRecord:
    """One explorer action (virtual, side-effect free)."""

    index: int
    name: str
    arguments: Dict[str, Any]
    result_payload: Any
    result_chars: int
    latency_s: float
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "name": self.name,
            "arguments": self.arguments,
            "result_payload": self.result_payload,
            "result_chars": self.result_chars,
            "latency_s": round(self.latency_s, 3),
            "error": self.error,
        }


@dataclass
class LLMUsageRecord:
    """Token usage for a single explorer LLM call."""

    model: str
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    latency_s: float
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "latency_s": round(self.latency_s, 3),
            "error": self.error,
        }


@dataclass
class ExplorerResult:
    """Outcome of one explore call."""

    report_final: str
    report_raw: str = ""
    committed_tool_names: List[str] = field(default_factory=list)
    requested_tool_names: List[str] = field(default_factory=list)
    hallucinated_tool_names: List[str] = field(default_factory=list)
    explored_tool_names: List[str] = field(default_factory=list)
    explored_skill_names: List[str] = field(default_factory=list)
    # P2: skills handed over to the main agent (validated + rendered entries).
    committed_skills: List[Dict[str, Any]] = field(default_factory=list)
    requested_skill_refs: List[str] = field(default_factory=list)
    rejected_skill_refs: List[str] = field(default_factory=list)
    confidence: str = "unknown"
    stop_reason: str = "finish"
    actions: List[ActionRecord] = field(default_factory=list)
    llm_usage: List[LLMUsageRecord] = field(default_factory=list)
    messages: List[Dict[str, Any]] = field(default_factory=list)
    report_chars_before_truncate: int = 0
    wall_time_s: float = 0.0
    error: Optional[str] = None

    # ── Derived helpers ──

    @property
    def actions_used(self) -> int:
        return len(self.actions)

    @property
    def llm_calls(self) -> int:
        return len(self.llm_usage)

    @property
    def action_histogram(self) -> Dict[str, int]:
        histogram: Dict[str, int] = {}
        for action in self.actions:
            histogram[action.name] = histogram.get(action.name, 0) + 1
        return histogram

    @property
    def prompt_tokens(self) -> Optional[int]:
        values = [u.prompt_tokens for u in self.llm_usage if u.prompt_tokens is not None]
        return sum(values) if values else None

    @property
    def completion_tokens(self) -> Optional[int]:
        values = [
            u.completion_tokens for u in self.llm_usage if u.completion_tokens is not None
        ]
        return sum(values) if values else None

    def summary(self) -> Dict[str, Any]:
        """Compact summary that goes into ``artifacts.explore.calls[*]``."""
        return {
            "actions_used": self.actions_used,
            "action_histogram": self.action_histogram,
            "committed_tool_names": list(self.committed_tool_names),
            "requested_tool_names": list(self.requested_tool_names),
            "hallucinated_tool_names": list(self.hallucinated_tool_names),
            "explored_tool_count": len(self.explored_tool_names),
            "explored_skill_count": len(self.explored_skill_names),
            "committed_skill_refs": [
                str(entry.get("ref") or "") for entry in (self.committed_skills or [])
            ],
            "committed_skill_count": len(self.committed_skills or []),
            "rejected_skill_refs": list(self.rejected_skill_refs),
            "report_chars_before_truncate": self.report_chars_before_truncate,
            "report_chars": len(self.report_final),
            "confidence": self.confidence,
            "stop_reason": self.stop_reason,
            "llm_calls": self.llm_calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "wall_time_s": round(self.wall_time_s, 3),
            "error": self.error,
        }
