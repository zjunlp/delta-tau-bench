"""
Cross-version metrics: Improvement Discovery Rate (IDR) and Degradation
Adaptation Rate (DAR).

For each task the reference-path weight is recomputed at two adjacent catalog
versions. Tasks are split by the direction in which that weight moved:

    D_down = { i | W_i^(v) <  W_i^(v-1) }    reference path became cheaper
    D_up   = { i | W_i^(v) >  W_i^(v-1) }    reference path became costlier

    IDR^(v) = (1 / |D_down|) * Σ_{i in D_down} S_i * min( W_i^(v) / Ŵ_i^(v), 1 )
    DAR^(v) = (1 / |D_up|)   * Σ_{i in D_up}   S_i * min( W_i^(v) / Ŵ_i^(v), 1 )

where S_i is pass/fail and Ŵ_i the agent trajectory weight. Failed tasks
contribute 0, so both rates penalise failure and over-usage jointly.

Weight modes: ``unit`` (function-call count) gives Step-IDR/DAR, ``cost``
(execution cost) gives Cost-IDR/DAR.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

from dtau_bench.evaluator.models import SampleEvalResult
from dtau_bench.evaluator.gold_weight import GoldWeightCalculator

logger = logging.getLogger(__name__)

# Floating point tolerance, kept identical to spe.py
_EPS = 1e-9


@dataclass
class TaskWeightInfo:
    """Per-task weight inputs for one version."""

    sample_id: str
    base_gold_traj: List[str]
    agent_weight: float
    passed: bool = False
    domain_key: str = ""


@dataclass
class CrossVersionMetrics:
    """
    IDR and DAR for one version pair under one weight mode.

    Attributes:
        version_pair: "v2_vs_v1" style label.
        weight_mode: "unit" or "cost".
        total_tasks: Tasks present in both versions with a usable reference path.
        improvement_tasks: |D_down| — reference path became cheaper.
        degradation_tasks: |D_up| — reference path became costlier.
        unchanged_tasks: Reference weight did not move.
        idr: Improvement Discovery Rate, None when |D_down| is 0.
        dar: Degradation Adaptation Rate, None when |D_up| is 0.
        passed_improvement_tasks: Passed tasks within D_down.
        passed_degradation_tasks: Passed tasks within D_up.
    """

    version_pair: str
    weight_mode: str
    total_tasks: int = 0
    improvement_tasks: int = 0
    degradation_tasks: int = 0
    unchanged_tasks: int = 0
    idr: Optional[float] = None
    dar: Optional[float] = None
    passed_improvement_tasks: int = 0
    passed_degradation_tasks: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_cross_version_metrics(
    task_weights_current: Dict[str, TaskWeightInfo],
    task_weights_prev: Dict[str, TaskWeightInfo],
    gold_weight_calculators: Dict[str, GoldWeightCalculator],
    current_version: str,
    prev_version: str,
    weight_mode: str,
) -> CrossVersionMetrics:
    """
    Compute IDR and DAR for one version pair.

    Args:
        task_weights_current: sample_id → TaskWeightInfo for the current version.
        task_weights_prev: sample_id → TaskWeightInfo for the previous version.
        gold_weight_calculators: domain_key → GoldWeightCalculator.
        current_version: Current version (e.g. "v2").
        prev_version: Previous version (e.g. "v1").
        weight_mode: "unit" or "cost".

    Returns:
        CrossVersionMetrics for this pair.
    """
    common_ids = set(task_weights_current.keys()) & set(task_weights_prev.keys())

    improvement_tasks = 0
    degradation_tasks = 0
    unchanged_tasks = 0
    passed_improvement = 0
    passed_degradation = 0
    ratio_sum_improvement = 0.0
    ratio_sum_degradation = 0.0
    total = 0

    for sample_id in common_ids:
        info = task_weights_current[sample_id]
        base_traj = info.base_gold_traj
        if not base_traj:
            continue

        calculator = gold_weight_calculators.get(info.domain_key)
        if calculator is None:
            continue

        total += 1

        if weight_mode == "unit":
            gold_current = float(
                calculator.get_optimal_gold_length(base_traj, current_version)
            )
            gold_prev = float(
                calculator.get_optimal_gold_length(base_traj, prev_version)
            )
        else:
            gold_current = calculator.get_optimal_gold_weight(base_traj, current_version)
            gold_prev = calculator.get_optimal_gold_weight(base_traj, prev_version)

        gold_delta = gold_current - gold_prev
        agent_weight = info.agent_weight

        if gold_delta < -_EPS:
            improvement_tasks += 1
            if info.passed:
                passed_improvement += 1
                ratio_sum_improvement += _weight_ratio(gold_current, agent_weight)
        elif gold_delta > _EPS:
            degradation_tasks += 1
            if info.passed:
                passed_degradation += 1
                ratio_sum_degradation += _weight_ratio(gold_current, agent_weight)
        else:
            unchanged_tasks += 1

    return CrossVersionMetrics(
        version_pair=f"{current_version}_vs_{prev_version}",
        weight_mode=weight_mode,
        total_tasks=total,
        improvement_tasks=improvement_tasks,
        degradation_tasks=degradation_tasks,
        unchanged_tasks=unchanged_tasks,
        idr=(
            ratio_sum_improvement / improvement_tasks if improvement_tasks > 0 else None
        ),
        dar=(
            ratio_sum_degradation / degradation_tasks if degradation_tasks > 0 else None
        ),
        passed_improvement_tasks=passed_improvement,
        passed_degradation_tasks=passed_degradation,
    )


def _weight_ratio(gold_weight: float, agent_weight: float) -> float:
    """min(W / Ŵ, 1), with an empty agent trajectory scoring 1."""
    if agent_weight <= _EPS:
        return 1.0
    return min(gold_weight / agent_weight, 1.0)


def compute_all_cross_version_metrics(
    task_weights_by_version: Dict[str, Dict[str, TaskWeightInfo]],
    gold_weight_calculators: Dict[str, GoldWeightCalculator],
    weight_mode: str,
    version_order: Optional[List[str]] = None,
) -> Dict[str, CrossVersionMetrics]:
    """
    Compute IDR/DAR for all adjacent version pairs.

    Args:
        task_weights_by_version: version → {sample_id → TaskWeightInfo}.
        gold_weight_calculators: domain_key → GoldWeightCalculator.
        weight_mode: "unit" or "cost".
        version_order: Ordered version list. If None, sorted naturally.

    Returns:
        version_pair → CrossVersionMetrics.
    """
    if version_order is None:
        version_order = _sort_versions(list(task_weights_by_version.keys()))

    out: Dict[str, CrossVersionMetrics] = {}
    for i in range(1, len(version_order)):
        prev_v = version_order[i - 1]
        curr_v = version_order[i]
        if (
            prev_v not in task_weights_by_version
            or curr_v not in task_weights_by_version
        ):
            continue
        metrics = compute_cross_version_metrics(
            task_weights_current=task_weights_by_version[curr_v],
            task_weights_prev=task_weights_by_version[prev_v],
            gold_weight_calculators=gold_weight_calculators,
            current_version=curr_v,
            prev_version=prev_v,
            weight_mode=weight_mode,
        )
        out[metrics.version_pair] = metrics

    return out


def build_task_weight_info_from_results(
    results: Dict[str, SampleEvalResult],
    base_gold_trajs: Dict[str, List[str]],
    agent_weights: Dict[str, float],
) -> Dict[str, TaskWeightInfo]:
    """
    Combine evaluation results with pre-computed agent weights.

    Args:
        results: sample_id → SampleEvalResult.
        base_gold_trajs: sample_id → base-version reference trajectory.
        agent_weights: sample_id → agent trajectory weight for this weight mode.

    Returns:
        sample_id → TaskWeightInfo.
    """
    return {
        sample_id: TaskWeightInfo(
            sample_id=sample_id,
            base_gold_traj=base_gold_trajs.get(sample_id, []),
            agent_weight=agent_weights.get(sample_id, 0.0),
            passed=result.passed,
            domain_key=result.domain_key,
        )
        for sample_id, result in results.items()
    }


def _sort_versions(versions: List[str]) -> List[str]:
    """Sort versions in natural order: base < v1 < v2 < ... < vN."""

    def _sort_key(v: str) -> Tuple[int, int, str]:
        if v == "base":
            return (0, 0, "")
        if v.startswith("v") and v[1:].isdigit():
            return (1, int(v[1:]), "")
        return (2, 9999, v)

    return sorted(versions, key=_sort_key)
