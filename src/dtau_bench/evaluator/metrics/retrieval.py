"""
Tool retrieval quality: Precision, Recall and F1.

For task i at version v, let Â_i be the final tool set exposed to the agent
after retrieval and A_i the tools contained in the reference path:

    P_i = |Â_i ∩ A_i| / |Â_i|
    R_i = |Â_i ∩ A_i| / |A_i|
    F1_i = 2 P_i R_i / (P_i + R_i)

The reported values are macro averages over applicable tasks.

Gold tools MUST be the version-translated sequence for the evaluated catalog
version, i.e. ``sample.version_trajectories[sample.version]``.

The ``gold_trajectory`` field of a rollout record is the **base** version
sequence; base tool names are renamed / split / merged in later versions, so
using it as gold silently deflates recall and precision, and the deflation
differs per version. Samples must be enriched via
``data_loader.attach_version_trajectories`` before calling into this module. If
the version sequence is unavailable, the sample is reported as non-computable
instead of falling back to base.

Predicted tools come from ``artifacts.retrieval.rounds[-1].final_visible_tool_names``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Set

from dtau_bench.evaluator.models import EvalSample

logger = logging.getLogger(__name__)

# Retrieval strategies that produce tool_names output
SUPPORTED_RETRIEVAL_STRATEGIES = frozenset({
    "all_tools_iterative_rag",
    # For the explorer-subagent strategy the predicted set is still
    # artifacts.retrieval.rounds[-1].final_visible_tool_names, which only grows
    # via committed (not merely explored) tools.
    "steward",
})

# Where the gold tool set for a sample came from.
GOLD_SOURCE_VERSION = "version"        # version_trajectories[sample.version] (correct)
GOLD_SOURCE_BASE = "base"              # base gold_trajectory (only for base/unversioned runs)
GOLD_SOURCE_UNAVAILABLE = "unavailable"  # version sequence missing -> not computable
GOLD_SOURCE_NOT_RESOLVED = "not_resolved"  # gold was never looked up (e.g. no retrieval trace)

# Versions for which the base gold_trajectory IS the correct reference.
BASE_VERSION_ALIASES = frozenset({"", "base"})

# Internal/framework tools to exclude
INTERNAL_TOOL_NAMES = frozenset({
    "retrieve_tools",
    "retrieve_skills",
    # Virtual tool of steward: never a real MCP tool, must not
    # count against retrieval precision if it ever leaks into a visible set.
    "explore_capabilities",
})


@dataclass
class RetrievalSampleMetrics:
    """
    Per-sample tool retrieval metrics.

    Attributes:
        sample_id: Unique identifier.
        applicable: Whether retrieval metrics can be computed for this sample.
        strategy: Retrieval strategy used (e.g., "all_tools_rag").
        predicted_tool_names: Tools returned by retrieval.
        gold_tool_names: Expected tools for the evaluated version.
        gold_source: Where gold came from ("version" / "base" / "unavailable" /
            "not_resolved" when gold was never looked up).
        matched_tool_names: Correctly retrieved tools.
        missing_gold_tool_names: Gold tools not retrieved.
        extra_predicted_tool_names: Retrieved tools not in gold.
        precision: P = |matched| / |predicted|.
        recall: R = |matched| / |gold|.
        f1: F1 = 2*P*R / (P+R).
        error: Error message if extraction failed.
    """

    sample_id: str = ""
    applicable: bool = False
    strategy: Optional[str] = None
    predicted_tool_names: List[str] = field(default_factory=list)
    gold_tool_names: List[str] = field(default_factory=list)
    gold_source: str = GOLD_SOURCE_NOT_RESOLVED
    matched_tool_names: List[str] = field(default_factory=list)
    missing_gold_tool_names: List[str] = field(default_factory=list)
    extra_predicted_tool_names: List[str] = field(default_factory=list)
    predicted_tool_count: int = 0
    gold_tool_count: int = 0
    matched_tool_count: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RetrievalSummary:
    """
    Macro-averaged retrieval metrics across samples.

    Attributes:
        samples_with_tool_retrieval: Number of applicable samples.
        precision: Mean of per-sample precision.
        recall: Mean of per-sample recall.
        f1: Mean of per-sample F1.
        avg_predicted_tool_count: Average predicted tools per sample.
        avg_gold_tool_count: Average gold tools per sample.
        avg_matched_tool_count: Average matched tools per sample.
        gold_source_counts: Gold-source histogram over applicable samples.
        samples_gold_unavailable: Applicable samples whose version-translated
            gold was missing (excluded from the averages above). Distinct from
            samples whose predicted set could not be extracted at all, which are
            counted as "not_resolved" in gold_source_counts.
    """

    samples_with_tool_retrieval: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    avg_predicted_tool_count: float = 0.0
    avg_gold_tool_count: float = 0.0
    avg_matched_tool_count: float = 0.0
    gold_source_counts: Dict[str, int] = field(default_factory=dict)
    samples_gold_unavailable: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _normalize_tool_name(name: str) -> str:
    """Strip namespace prefix and whitespace from tool name."""
    return name.strip().split(".")[-1]


def _is_internal_tool(name: str) -> bool:
    """Check if a tool name is an internal framework tool."""
    normalized = _normalize_tool_name(name)
    return normalized in INTERNAL_TOOL_NAMES


def _safe_list_of_strings(values: Any) -> List[str]:
    """Safely extract a list of non-empty strings."""
    if not isinstance(values, list):
        return []
    result: List[str] = []
    for value in values:
        if not isinstance(value, str):
            continue
        normalized = value.strip()
        if normalized:
            result.append(normalized)
    return result


def _deduplicate_normalize(names: List[str]) -> List[str]:
    """Normalize and deduplicate tool names, preserving order."""
    seen: Set[str] = set()
    result: List[str] = []
    for name in names:
        normalized = _normalize_tool_name(name)
        if normalized and normalized not in seen and not _is_internal_tool(normalized):
            seen.add(normalized)
            result.append(normalized)
    return result


def _compute_prf(predicted: Set[str], gold: Set[str]) -> tuple:
    """Compute precision, recall, F1 given two sets."""
    matched = predicted & gold
    matched_count = len(matched)
    precision = (matched_count / len(predicted)) if predicted else 0.0
    recall = (matched_count / len(gold)) if gold else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return matched_count, precision, recall, f1


def _extract_predicted_tools_from_sample(sample: EvalSample) -> tuple:
    """
    Extract predicted tool names from sample artifacts.

    Returns:
        (strategy, predicted_tool_names_or_None, error_or_None)
    """
    # Determine strategy from run metadata
    run_meta = sample.run_meta if isinstance(sample.run_meta, dict) else {}
    strategy = run_meta.get("agent_strategy") or run_meta.get("tool_selection_mode")
    if not isinstance(strategy, str):
        strategy = None

    if strategy not in SUPPORTED_RETRIEVAL_STRATEGIES:
        return strategy, None, None

    # Extract from artifacts.retrieval
    artifacts = sample.artifacts if isinstance(sample.artifacts, dict) else {}
    retrieval = artifacts.get("retrieval")
    if not isinstance(retrieval, dict):
        return strategy, None, "structured retrieval trace missing"

    rounds = retrieval.get("rounds")
    if not isinstance(rounds, list) or not rounds:
        return strategy, None, "retrieval rounds missing"

    latest = rounds[-1]
    if not isinstance(latest, dict):
        return strategy, None, "latest retrieval round malformed"

    # The predicted tool set is the visible set at the end of the session.
    predicted = _deduplicate_normalize(
        _safe_list_of_strings(latest.get("final_visible_tool_names"))
    )
    return strategy, predicted, None


def _extract_gold_tools_from_sample(sample: EvalSample) -> tuple:
    """
    Extract the gold tool names for the evaluated catalog version.

    Resolution order:
      1. run has no version, or targets "base" -> base ``sample.gold_trajectory``
         (GOLD_SOURCE_BASE); for those runs the base sequence IS the reference.
      2. ``sample.version_trajectories[sample.version]`` -> GOLD_SOURCE_VERSION.
      3. otherwise -> GOLD_SOURCE_UNAVAILABLE with an empty list.

    Case 3 deliberately does NOT fall back to the base trajectory: base tool
    names do not exist in later catalog versions, so such a fallback would look
    like a low score instead of a missing input.

    Returns:
        (gold_tool_names, gold_source)
    """
    version = str(sample.version or "").strip()

    if version in BASE_VERSION_ALIASES:
        return _deduplicate_normalize(sample.gold_trajectory), GOLD_SOURCE_BASE

    version_trajectories = sample.version_trajectories
    if isinstance(version_trajectories, dict):
        tools = version_trajectories.get(version)
        if isinstance(tools, list) and tools:
            return _deduplicate_normalize([str(t) for t in tools if t]), GOLD_SOURCE_VERSION

    return [], GOLD_SOURCE_UNAVAILABLE


def compute_retrieval_metrics(sample: EvalSample) -> RetrievalSampleMetrics:
    """
    Compute tool retrieval P/R/F1 for a single sample.

    Args:
        sample: EvalSample with artifacts and gold_trajectory.

    Returns:
        RetrievalSampleMetrics for this sample.
    """
    strategy, predicted_names, extraction_error = _extract_predicted_tools_from_sample(sample)

    # Not applicable if strategy is not a retrieval strategy
    if strategy not in SUPPORTED_RETRIEVAL_STRATEGIES:
        return RetrievalSampleMetrics(
            sample_id=sample.id,
            applicable=False,
            strategy=strategy,
        )

    # Applicable but extraction failed
    if predicted_names is None:
        return RetrievalSampleMetrics(
            sample_id=sample.id,
            applicable=True,
            strategy=strategy,
            error=extraction_error,
        )

    # Resolve gold for the evaluated version
    gold_names, gold_source = _extract_gold_tools_from_sample(sample)

    if gold_source == GOLD_SOURCE_UNAVAILABLE:
        # Not a retrieval failure: the reference itself is missing. Surface it
        # instead of scoring against base tool names.
        return RetrievalSampleMetrics(
            sample_id=sample.id,
            applicable=True,
            strategy=strategy,
            predicted_tool_names=predicted_names,
            predicted_tool_count=len(predicted_names),
            gold_source=gold_source,
            error=(
                f"gold trajectory for version '{sample.version}' unavailable; "
                "enrich samples via data_loader.attach_version_trajectories"
            ),
        )

    predicted_set = set(predicted_names)
    gold_set = set(gold_names)

    matched_count, precision, recall, f1 = _compute_prf(predicted_set, gold_set)
    matched_names = sorted(predicted_set & gold_set)
    missing_names = sorted(gold_set - predicted_set)
    extra_names = sorted(predicted_set - gold_set)

    return RetrievalSampleMetrics(
        sample_id=sample.id,
        applicable=True,
        strategy=strategy,
        predicted_tool_names=predicted_names,
        gold_tool_names=gold_names,
        gold_source=gold_source,
        matched_tool_names=matched_names,
        missing_gold_tool_names=missing_names,
        extra_predicted_tool_names=extra_names,
        predicted_tool_count=len(predicted_names),
        gold_tool_count=len(gold_names),
        matched_tool_count=matched_count,
        precision=precision,
        recall=recall,
        f1=f1,
    )


def summarize_retrieval_metrics(
    sample_metrics: List[RetrievalSampleMetrics],
) -> RetrievalSummary:
    """
    Macro-average per-sample retrieval metrics.

    Only includes applicable samples without errors.

    Args:
        sample_metrics: List of per-sample retrieval metrics.

    Returns:
        RetrievalSummary with macro-averaged P/R/F1.
    """
    applicable = [m for m in sample_metrics if m.applicable and m.error is None]

    gold_source_counts: Dict[str, int] = {}
    for metric in sample_metrics:
        if not metric.applicable:
            continue
        gold_source_counts[metric.gold_source] = (
            gold_source_counts.get(metric.gold_source, 0) + 1
        )
    samples_gold_unavailable = gold_source_counts.get(GOLD_SOURCE_UNAVAILABLE, 0)

    if samples_gold_unavailable:
        logger.warning(
            "Tool retrieval: %d applicable samples had no version-translated gold "
            "and were excluded. Did you forget attach_version_trajectories()?",
            samples_gold_unavailable,
        )

    if not applicable:
        return RetrievalSummary(
            gold_source_counts=gold_source_counts,
            samples_gold_unavailable=samples_gold_unavailable,
        )

    n = len(applicable)

    return RetrievalSummary(
        samples_with_tool_retrieval=n,
        precision=sum(m.precision for m in applicable) / n,
        recall=sum(m.recall for m in applicable) / n,
        f1=sum(m.f1 for m in applicable) / n,
        avg_predicted_tool_count=sum(m.predicted_tool_count for m in applicable) / n,
        avg_gold_tool_count=sum(m.gold_tool_count for m in applicable) / n,
        avg_matched_tool_count=sum(m.matched_tool_count for m in applicable) / n,
        gold_source_counts=gold_source_counts,
        samples_gold_unavailable=samples_gold_unavailable,
    )
