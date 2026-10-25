"""
Data models for the dtau_bench evaluator.

Defines the core data structures used across the evaluation pipeline:
- EvalSample: A single evaluation task loaded from rollout output
- SampleEvalResult: Evaluation result for a single sample
- VersionReport: Aggregated evaluation report for a version
- ActionRecord: A single agent tool call with metadata
- TrajectoryWeightSummary: Weight summary for an agent trajectory
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class EvalSample:
    """
    A single evaluation sample parsed from rollout JSONL output.

    Contains all information needed to evaluate one task:
    - Gold trajectory and execution context for replay
    - Agent trajectory and conversation for action extraction
    - Metadata (domains, version, run info)
    """

    id: str
    """Unique sample identifier (query_id)."""

    gold_trajectory: List[str]
    """Base-version gold tool name sequence (NOT version-translated).

    Rollout JSONL only carries the base sequence; use ``version_trajectories``
    for version-aware references.
    """

    entity_context: Dict[str, Any]
    """Entity instances and value domain samples for gold replay."""

    agent_traj: List[str]
    """Agent's tool call sequence (tool names only, from rollout)."""

    messages: List[Dict[str, Any]]
    """Full conversation messages (for action extraction)."""

    domains: List[str]
    """Domain(s) involved in this task."""

    version: str
    """Tool version used in this rollout."""

    instruction: str
    """User instruction / task description."""

    # Optional rich fields
    version_trajectories: Optional[Dict[str, List[str]]] = None
    """Gold trajectories for all versions (base, v1, ..., v6)."""

    gold_execution_context: Optional[Dict[str, Any]] = None
    """Pre-computed gold execution context (step_param_codes, etc.)."""

    artifacts: Dict[str, Any] = field(default_factory=dict)
    """Rollout artifacts (retrieval info, etc.)."""

    run_meta: Dict[str, Any] = field(default_factory=dict)
    """Run metadata (strategy, model, etc.)."""

    trajectory_hash: str = ""
    """Hash of the gold trajectory for caching."""

    domain_key: str = ""
    """Domain key (for cross-domain tasks, joined with '_')."""

    success: bool = False
    """Whether the rollout itself succeeded (agent completed task)."""

    error: Optional[str] = None
    """Error message if rollout failed."""


@dataclass
class ActionRecord:
    """
    Represents a single agent tool call extracted from conversation messages.

    Attributes:
        index: Position in the extracted action sequence (0-based).
        tool_name: Normalized tool name (without namespace prefix).
        tool_call_id: OpenAI-style tool_call_id for pairing with response.
        arguments: Tool call arguments dict.
        tool_result_raw: Raw tool response content (string or None).
        tool_result_parsed: Parsed JSON tool response (dict/list or None).
        has_error: True if tool response contains top-level "error" key.
    """

    index: int
    tool_name: str
    tool_call_id: Optional[str] = None
    arguments: Dict[str, Any] = field(default_factory=dict)
    tool_result_raw: Any = None
    tool_result_parsed: Any = None
    has_error: bool = False

    def to_summary_dict(self) -> Dict[str, Any]:
        """Compact representation for logging/debugging."""
        return {
            "index": self.index,
            "tool_name": self.tool_name,
            "has_error": self.has_error,
        }


@dataclass
class TrajectoryWeightSummary:
    """
    Agent trajectory weight Ŵ under a specific weight mode.

    Attributes:
        agent_traj: Tool name sequence.
        agent_weight: Total weight of the sequence.
        weight_mode: "unit" (function-call count) or "cost" (execution cost).
    """

    agent_traj: List[str]
    agent_weight: float
    weight_mode: str

    @property
    def length(self) -> int:
        return len(self.agent_traj)


@dataclass
class SampleEvalResult:
    """
    Evaluation result for a single sample.

    Populated after gold replay + agent replay + DB diff.
    """

    sample_id: str
    version: str
    passed: bool = False
    error: Optional[str] = None

    # Trajectory info
    gold_trajectory: List[str] = field(default_factory=list)
    agent_traj: List[str] = field(default_factory=list)

    # Domain info
    domains: List[str] = field(default_factory=list)
    domain_key: str = ""

    # Detailed results (populated by replay engine)
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "version": self.version,
            "passed": self.passed,
            "error": self.error,
            "gold_trajectory": self.gold_trajectory,
            "agent_traj": self.agent_traj,
            "domains": self.domains,
            "domain_key": self.domain_key,
            "details": self.details,
        }


@dataclass
class DomainStats:
    """Per-domain success statistics."""

    domain: str
    total: int = 0
    passed: int = 0
    failed: int = 0

    @property
    def sr(self) -> float:
        return self.passed / self.total if self.total > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "domain": self.domain,
            "total": self.total,
            "passed": self.passed,
            "failed": self.failed,
            "sr": self.sr,
        }


@dataclass
class VersionReport:
    """
    Aggregated evaluation report for a single version.

    Attributes:
        version: Tool version evaluated.
        total: Total number of samples.
        passed: Number of passing samples.
        failed: Number of failing samples.
        sr: Overall success rate.
        by_domain: Per-domain breakdown.
        retrieval_summary: Tool retrieval P/R/F1 if available.
    """

    version: str
    total: int = 0
    passed: int = 0
    failed: int = 0

    by_domain: Dict[str, DomainStats] = field(default_factory=dict)
    retrieval_summary: Dict[str, float] = field(default_factory=dict)

    @property
    def sr(self) -> float:
        return self.passed / self.total if self.total > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "total": self.total,
            "passed": self.passed,
            "failed": self.failed,
            "sr": self.sr,
            "by_domain": {k: v.to_dict() for k, v in self.by_domain.items()},
            "retrieval_summary": self.retrieval_summary,
        }
