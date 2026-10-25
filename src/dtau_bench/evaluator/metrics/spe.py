"""
Success-weighted Path Efficiency (SPE), a single-version metric.

    SPE^(v) = (1 / N_v) * Σ_i  S_i * min( W_i^(v) / Ŵ_i^(v), 1 )

where S_i is pass/fail, W_i the reference-path weight and Ŵ_i the agent
trajectory weight. N_v is the full evaluated task count for that version, the
same denominator as SR, so SPE <= SR always holds.

Weight modes: ``unit`` (function-call count) gives Step-SPE, ``cost``
(execution cost) gives Cost-SPE.

In cost mode the agent-side weight uses real per-tool costs only when
``data/meta/tool_costs.json`` exists. Without it the agent side falls back to
unit cost while the reference side keeps real costs, which inflates Cost-SPE.
``mean_gold_weight`` / ``mean_agent_weight`` make such a scale mismatch visible.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

from dtau_bench.evaluator.gold_weight import GoldWeightCalculator
from dtau_bench.evaluator.metrics.cross_version import TaskWeightInfo, _sort_versions

logger = logging.getLogger(__name__)

# Floating point tolerance, kept identical to cross_version.py
_EPS = 1e-9


@dataclass
class SPEMetrics:
    """
    SPE and SR for one version under one weight mode.

    Attributes:
        version: Catalog version (e.g. "v2").
        weight_mode: "unit" (Step-SPE) or "cost" (Cost-SPE).
        total_tasks: N_v — full evaluated task count, the SR denominator too.
        computable_tasks: Tasks where both W and Ŵ could be determined.
        passed_tasks: Tasks with S_i = 1.
        spe: Σ S_i·min(W/Ŵ, 1) / N_v.
        sr: passed_tasks / N_v.
        mean_gold_weight: Mean W over computable tasks (scale sanity check).
        mean_agent_weight: Mean Ŵ over computable tasks (scale sanity check).
        skipped_no_base_traj: Tasks without a base reference trajectory.
        skipped_no_calculator: Tasks whose domain has no weight calculator.
        skipped_passed: Skipped tasks that had passed — these lower the
            achievable SPE ceiling.
    """

    version: str
    weight_mode: str

    total_tasks: int = 0
    computable_tasks: int = 0
    passed_tasks: int = 0

    spe: Optional[float] = None
    sr: Optional[float] = None

    mean_gold_weight: Optional[float] = None
    mean_agent_weight: Optional[float] = None

    skipped_no_base_traj: int = 0
    skipped_no_calculator: int = 0
    skipped_passed: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_spe_metrics(
    task_weights: Dict[str, TaskWeightInfo],
    gold_weight_calculators: Dict[str, GoldWeightCalculator],
    version: str,
    weight_mode: str,
    total_tasks: int,
) -> SPEMetrics:
    """
    Compute SPE and SR for a single version.

    Args:
        task_weights: sample_id → TaskWeightInfo for this version, already built
            under the given weight mode.
        gold_weight_calculators: domain_key → GoldWeightCalculator.
        version: Version to compute the reference weight for.
        weight_mode: "unit" or "cost".
        total_tasks: N_v — the full evaluated task count for this version. Tasks
            missing from ``task_weights`` still count towards the denominator
            with S_i = 0, which keeps SPE comparable with SR.

    Returns:
        SPEMetrics for this (version, weight_mode).
    """
    ratio_sum = 0.0
    passed_tasks = 0
    computable = 0
    skipped_no_base = 0
    skipped_no_calc = 0
    skipped_passed = 0
    gold_weights: List[float] = []
    agent_weights: List[float] = []

    for info in task_weights.values():
        if info.passed:
            passed_tasks += 1

        if not info.base_gold_traj:
            skipped_no_base += 1
            if info.passed:
                skipped_passed += 1
            continue

        calculator = gold_weight_calculators.get(info.domain_key)
        if calculator is None:
            skipped_no_calc += 1
            if info.passed:
                skipped_passed += 1
            continue

        if weight_mode == "unit":
            gold_weight = float(
                calculator.get_optimal_gold_length(info.base_gold_traj, version)
            )
        else:
            gold_weight = calculator.get_optimal_gold_weight(
                info.base_gold_traj, version
            )

        agent_weight = float(info.agent_weight)

        computable += 1
        gold_weights.append(gold_weight)
        agent_weights.append(agent_weight)

        # Gated on success: a failed task contributes 0.
        if not info.passed:
            continue

        if agent_weight <= _EPS:
            # No domain tool call at all; scores a perfect ratio.
            ratio_sum += 1.0
            continue

        ratio_sum += min(gold_weight / agent_weight, 1.0)

    denominator = int(total_tasks)
    if denominator <= 0:
        logger.warning(
            "SPE for %s [%s]: total_tasks=%d, cannot compute ratios.",
            version, weight_mode, denominator,
        )

    return SPEMetrics(
        version=version,
        weight_mode=weight_mode,
        total_tasks=denominator,
        computable_tasks=computable,
        passed_tasks=passed_tasks,
        spe=(ratio_sum / denominator) if denominator > 0 else None,
        sr=(passed_tasks / denominator) if denominator > 0 else None,
        mean_gold_weight=(
            sum(gold_weights) / len(gold_weights) if gold_weights else None
        ),
        mean_agent_weight=(
            sum(agent_weights) / len(agent_weights) if agent_weights else None
        ),
        skipped_no_base_traj=skipped_no_base,
        skipped_no_calculator=skipped_no_calc,
        skipped_passed=skipped_passed,
    )


def compute_all_spe_metrics(
    task_weights_by_version: Dict[str, Dict[str, TaskWeightInfo]],
    gold_weight_calculators: Dict[str, GoldWeightCalculator],
    weight_mode: str,
    total_tasks_by_version: Dict[str, int],
    version_order: Optional[List[str]] = None,
) -> Dict[str, SPEMetrics]:
    """
    Compute SPE for every version under one weight mode.

    Args:
        task_weights_by_version: version → {sample_id → TaskWeightInfo}.
        gold_weight_calculators: domain_key → GoldWeightCalculator.
        weight_mode: "unit" or "cost".
        total_tasks_by_version: version → N_v (full evaluated task count).
        version_order: Ordered version list. If None, sorted naturally.

    Returns:
        version → SPEMetrics.
    """
    if version_order is None:
        version_order = _sort_versions(list(task_weights_by_version.keys()))

    out: Dict[str, SPEMetrics] = {}
    for version in version_order:
        task_weights = task_weights_by_version.get(version)
        if task_weights is None:
            continue
        total = total_tasks_by_version.get(version)
        if not total:
            # Falling back changes the denominator, so make it loud.
            total = len(task_weights)
            logger.warning(
                "SPE for %s [%s]: no roster size supplied, falling back to %d "
                "observed tasks. Cross-run comparisons may be inconsistent.",
                version, weight_mode, total,
            )
        out[version] = compute_spe_metrics(
            task_weights=task_weights,
            gold_weight_calculators=gold_weight_calculators,
            version=version,
            weight_mode=weight_mode,
            total_tasks=total,
        )
    return out
