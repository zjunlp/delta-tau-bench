#!/usr/bin/env python3
"""Aggregate the per-version evaluation results of one experiment.

Reads each version's evaluation_results.jsonl, computes SPE per version and
IDR/DAR per adjacent version pair, and writes the combined report.

Usage:
    python -m dtau_bench evaluate-multi \
      --experiment-name my_run \
      --versions v1 v2 v3 v4 v5 v6 \
      --rollout-pattern "my_run_test_{version}" \
      --output-dir experiments/my_run_multi_eval/ \
      --weight-modes unit cost
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dtau_bench.evaluator.data_loader import (
    attach_version_trajectories,
    load_rollout_dir,
    TaskMetadataProvider,
)
from dtau_bench.evaluator.models import EvalSample, SampleEvalResult
from dtau_bench.evaluator.actions import (
    extract_action_records,
    compute_trajectory_weights,
)
from dtau_bench.evaluator.gold_weight import GoldWeightCalculator
from dtau_bench.evaluator.metrics.retrieval import (
    RetrievalSummary,
    compute_retrieval_metrics,
    summarize_retrieval_metrics,
)
from dtau_bench.evaluator.metrics.cross_version import (
    TaskWeightInfo,
    compute_all_cross_version_metrics,
    build_task_weight_info_from_results,
)
from dtau_bench.evaluator.metrics.spe import compute_all_spe_metrics
from dtau_bench.evaluator.report import (
    MultiVersionReport,
    build_multi_version_report,
)
from dtau_bench.utils.config import get_data_root

logger = logging.getLogger("dtau_bench.cli.evaluate_multi")


class IncompleteEvaluationError(RuntimeError):
    """Requested versions do not all have complete evaluation results."""


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def resolve_run_name(pattern: str, version: str) -> str:
    """
    Resolve run_name from a pattern.

    Replaces {version} in the pattern with the actual version string.

    Args:
        pattern: Run name pattern (e.g., "test_evolve_{version}").
        version: Version to substitute.

    Returns:
        Resolved run name.
    """
    return pattern.replace("{version}", version)


@lru_cache(maxsize=None)
def _load_test_task_roster(
    data_dir_str: str,
    strict: bool = False,
) -> Tuple[Tuple[str, str], ...]:
    """
    Load the canonical (gold) test task roster from data/tasks/test.

    Returns a tuple of (query_id, domain) for every gold test task. The result
    is cached (and hashable) so repeated calls across versions are cheap.
    The task set is version-agnostic (the same tasks are reused across all
    tool-schema versions).
    """
    base = Path(data_dir_str) / "tasks" / "test"
    if not base.is_dir():
        return tuple()
    task_files = sorted(base.glob("*/*.json"))
    if not strict:
        # Preserve the legacy, filename-based behaviour for callers which use
        # load_evaluation_results(..., strict=False).
        return tuple((p.stem, p.parent.name) for p in task_files)

    roster: List[Tuple[str, str]] = []
    seen_ids: set[str] = set()
    for task_path in task_files:
        try:
            with task_path.open("r", encoding="utf-8") as handle:
                task = json.load(handle)
            if not isinstance(task, dict):
                raise ValueError("task is not a JSON object")
            query_id = str(task.get("query_id") or task_path.stem).strip()
            domain = str(task.get("domain") or task_path.parent.name).strip()
            if not query_id or not domain:
                raise ValueError("task has no query_id/domain")
            if query_id in seen_ids:
                raise ValueError(f"duplicate query_id {query_id!r}")
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            raise IncompleteEvaluationError(
                f"Invalid canonical test task {task_path}: {exc}"
            ) from exc
        seen_ids.add(query_id)
        roster.append((query_id, domain))
    return tuple(roster)


def load_evaluation_results(
    eval_dir: Path,
    data_dir: Optional[Path] = None,
    version: str = "",
    *,
    strict: bool = False,
) -> Optional[List[SampleEvalResult]]:
    """
    Load pre-computed evaluation results from a directory.

    Looks for evaluation_results.jsonl (produced by the evaluate subcommand).

    If ``data_dir`` is provided, legacy callers backfill missing gold tasks as
    failed placeholders. With ``strict=True``, missing, duplicate, malformed,
    unexpected, or wrong-version rows raise ``IncompleteEvaluationError``.

    Args:
        eval_dir: Evaluation output directory.
        data_dir: Root data directory (for the gold task roster). If None,
            no backfill is performed (legacy behavior).
        version: Version label used when constructing placeholder rows.
        strict: Require an exact, valid result row for every gold test task.

    Returns:
        List of SampleEvalResult, or None if not found.
    """
    results_file = eval_dir / "evaluation_results.jsonl"
    if not results_file.exists():
        return None

    results: List[SampleEvalResult] = []
    seen_ids: set[str] = set()
    with open(results_file, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                if not isinstance(data, dict):
                    raise ValueError("row is not a JSON object")
                sample_id = str(data.get("sample_id") or "").strip()
                if strict and not sample_id:
                    raise ValueError("sample_id is missing")
                if strict and sample_id in seen_ids:
                    raise ValueError(f"duplicate sample_id {sample_id!r}")
                row_version = str(data.get("version") or "").strip()
                if strict and version and row_version != version:
                    raise ValueError(
                        f"sample {sample_id!r} has version {row_version!r}; "
                        f"expected {version!r}"
                    )
                passed = data.get("passed", False)
                if strict and not isinstance(passed, bool):
                    raise ValueError(
                        f"sample {sample_id!r} has non-boolean passed={passed!r}"
                    )
                results.append(SampleEvalResult(
                    sample_id=sample_id,
                    version=row_version,
                    passed=passed,
                    error=data.get("error"),
                    gold_trajectory=data.get("gold_trajectory", []),
                    agent_traj=data.get("agent_traj", []),
                    domains=data.get("domains", []),
                    domain_key=data.get("domain_key", ""),
                    details=data.get("details", {}),
                ))
                seen_ids.add(sample_id)
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
                if strict:
                    raise IncompleteEvaluationError(
                        f"Invalid evaluation result at {results_file}:{line_number}: {e}"
                    ) from e
                logger.warning("Skipping malformed result line: %s", e)
                continue

    if not results:
        return None

    # Strict multi-version aggregation must never turn a partial evaluation into
    # an apparently complete one. Legacy callers can retain the old backfill.
    if data_dir is not None:
        roster = _load_test_task_roster(str(data_dir), strict=strict)
        if roster:
            seen = {r.sample_id for r in results}
            expected = {query_id for query_id, _ in roster}
            expected_domains = dict(roster)
            missing_ids = sorted(expected - seen)
            unexpected_ids = sorted(seen - expected)
            wrong_domains = sorted(
                f"{result.sample_id}={result.domain_key!r}"
                for result in results
                if result.sample_id in expected_domains
                and result.domain_key != expected_domains[result.sample_id]
            )
            if strict and (missing_ids or unexpected_ids or wrong_domains):
                raise IncompleteEvaluationError(
                    f"Incomplete evaluation for {version or eval_dir.name}: "
                    f"missing={missing_ids[:20]}{'...' if len(missing_ids) > 20 else ''}, "
                    f"unexpected={unexpected_ids[:20]}, "
                    f"wrong_domain={wrong_domains[:20]}"
                )
            ver = version or results[0].version
            missing = [(qid, domain) for qid, domain in roster if qid in missing_ids]
            for qid, domain in missing:
                results.append(SampleEvalResult(
                    sample_id=qid,
                    version=ver,
                    passed=False,
                    error="missing",
                    gold_trajectory=[],
                    agent_traj=[],
                    domains=[domain],
                    domain_key=domain,
                    details={},
                ))
            if missing:
                logger.warning(
                    "  %s: backfilled %d missing gold task(s) as failed "
                    "(roster=%d, had=%d). Missing ids: %s",
                    ver, len(missing), len(roster), len(seen),
                    ", ".join(qid for qid, _ in missing[:20])
                    + (" ..." if len(missing) > 20 else ""),
                )

    return results


def _validate_rollout_samples(
    samples: List[EvalSample],
    *,
    roster: Tuple[Tuple[str, str], ...],
    version: str,
    source: Path,
    raw_record_count: Optional[int] = None,
) -> None:
    """Require one correctly-versioned rollout for every canonical test task."""
    problems: List[str] = []
    if raw_record_count is not None and raw_record_count != len(samples):
        problems.append(
            f"{raw_record_count} non-empty record(s) but only "
            f"{len(samples)} valid sample(s)"
        )

    seen: set[str] = set()
    duplicates: set[str] = set()
    wrong_versions: List[str] = []
    for sample in samples:
        sample_id = str(sample.id or "").strip()
        if not sample_id:
            problems.append("rollout sample has no id")
            continue
        if sample_id in seen:
            duplicates.add(sample_id)
        seen.add(sample_id)
        if sample.version != version:
            wrong_versions.append(
                f"{sample_id}={sample.version!r}"
            )

    expected_ids = {query_id for query_id, _ in roster}
    expected_domains = dict(roster)
    missing = sorted(expected_ids - seen)
    unexpected = sorted(seen - expected_ids)
    if duplicates:
        problems.append(f"duplicate={sorted(duplicates)[:20]}")
    if wrong_versions:
        problems.append(
            f"wrong_version={wrong_versions[:20]}"
            f"{'...' if len(wrong_versions) > 20 else ''}"
        )
    if missing:
        problems.append(
            f"missing={missing[:20]}{'...' if len(missing) > 20 else ''}"
        )
    if unexpected:
        problems.append(
            f"unexpected={unexpected[:20]}"
            f"{'...' if len(unexpected) > 20 else ''}"
        )
    wrong_domains = [
        f"{sample.id}={sample.domain_key!r}"
        for sample in samples
        if sample.id in expected_domains
        and sample.domain_key != expected_domains[sample.id]
    ]
    if wrong_domains:
        problems.append(
            f"wrong_domain={wrong_domains[:20]}"
            f"{'...' if len(wrong_domains) > 20 else ''}"
        )

    if problems:
        raise IncompleteEvaluationError(
            f"Incomplete rollout for {version} at {source}: " + "; ".join(problems)
        )


def _count_nonempty_jsonl_records(directory: Path) -> int:
    """Count source rows so loader-skipped malformed records cannot go unnoticed."""
    count = 0
    for jsonl_path in sorted(directory.glob("*.jsonl")):
        with jsonl_path.open("r", encoding="utf-8") as handle:
            count += sum(1 for line in handle if line.strip())
    return count


def _validate_weight_metric_coverage(
    cross_version_metrics: Dict[str, Dict[str, Any]],
    spe_metrics: Dict[str, Dict[str, Any]],
    *,
    versions: List[str],
    weight_modes: List[str],
    roster_size: int,
) -> None:
    """Reject reports with missing or only partially-computed paper metrics."""
    expected_pairs = [
        f"{versions[index]}_vs_{versions[index - 1]}"
        for index in range(1, len(versions))
    ]
    problems: List[str] = []

    for weight_mode in weight_modes:
        per_version = spe_metrics.get(weight_mode)
        if not isinstance(per_version, dict):
            problems.append(f"missing SPE mode {weight_mode!r}")
        else:
            for version in versions:
                metrics = per_version.get(version)
                if not isinstance(metrics, dict) or "error" in metrics:
                    problems.append(f"missing SPE for {weight_mode}/{version}")
                    continue
                if "spe" not in metrics:
                    problems.append(f"missing SPE value for {weight_mode}/{version}")
                if metrics.get("total_tasks") != roster_size:
                    problems.append(
                        f"SPE {weight_mode}/{version} total_tasks="
                        f"{metrics.get('total_tasks')!r}, expected {roster_size}"
                    )
                if metrics.get("computable_tasks") != roster_size:
                    problems.append(
                        f"SPE {weight_mode}/{version} computable_tasks="
                        f"{metrics.get('computable_tasks')!r}, expected {roster_size}"
                    )

        per_pair = cross_version_metrics.get(weight_mode)
        if not isinstance(per_pair, dict):
            problems.append(f"missing IDR/DAR mode {weight_mode!r}")
            continue
        for pair in expected_pairs:
            metrics = per_pair.get(pair)
            if not isinstance(metrics, dict) or "error" in metrics:
                problems.append(f"missing IDR/DAR for {weight_mode}/{pair}")
                continue
            if "idr" not in metrics or "dar" not in metrics:
                problems.append(f"missing IDR/DAR values for {weight_mode}/{pair}")
            if metrics.get("total_tasks") != roster_size:
                problems.append(
                    f"IDR/DAR {weight_mode}/{pair} total_tasks="
                    f"{metrics.get('total_tasks')!r}, expected {roster_size}"
                )

    if problems:
        raise IncompleteEvaluationError(
            "Cannot generate a partial multi-version metric report: "
            + "; ".join(problems)
        )


def _build_task_weights_for_version(
    results: List[SampleEvalResult],
    samples: List[EvalSample],
    weight_mode: str,
    tool_weights: Optional[Dict[str, float]] = None,
) -> Dict[str, TaskWeightInfo]:
    """
    Build TaskWeightInfo for one version under one weight mode.

    Args:
        results: Evaluation results for this version.
        samples: EvalSample list for this version.
        weight_mode: "unit" or "cost".
        tool_weights: Per-tool cost weights (for cost mode).

    Returns:
        sample_id → TaskWeightInfo.
    """
    results_by_id = {r.sample_id: r for r in results}

    agent_weights: Dict[str, float] = {}
    base_gold_trajs: Dict[str, List[str]] = {}

    for sample in samples:
        sample_id = sample.id
        if sample_id not in results_by_id:
            continue

        records = extract_action_records(sample.messages)
        tw = tool_weights if weight_mode == "cost" else None
        agent_weights[sample_id] = compute_trajectory_weights(
            records, tool_weights=tw, weight_mode=weight_mode,
        ).agent_weight

        # Reference trajectories are stored at base version and translated per
        # target version by GoldWeightCalculator.
        if sample.version_trajectories and "base" in sample.version_trajectories:
            base_gold_trajs[sample_id] = sample.version_trajectories["base"]
        else:
            base_gold_trajs[sample_id] = sample.gold_trajectory

    return build_task_weight_info_from_results(
        results=results_by_id,
        base_gold_trajs=base_gold_trajs,
        agent_weights=agent_weights,
    )


def compute_weight_based_metrics(
    eval_results_by_version: Dict[str, List[SampleEvalResult]],
    samples_by_version: Dict[str, List[EvalSample]],
    versions: List[str],
    weight_modes: List[str],
    gold_weight_calculators: Optional[Dict[str, GoldWeightCalculator]] = None,
    tool_weights: Optional[Dict[str, float]] = None,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """
    Compute IDR/DAR and SPE for every weight mode.

    Both families share the same per-task weight computation. IDR/DAR need
    adjacent version pairs; SPE is single-version and is therefore also defined
    for the first version.

    Args:
        eval_results_by_version: version → results, already backfilled to the
            full task roster so its length is N_v.
        samples_by_version: version → list of EvalSample.
        versions: Ordered version list.
        weight_modes: ["unit", "cost"].
        gold_weight_calculators: domain_key → GoldWeightCalculator.
        tool_weights: Per-tool cost weights (for cost mode).

    Returns:
        (cross_version_by_weight, spe_by_weight), each keyed by weight mode.
    """
    if not gold_weight_calculators:
        logger.warning(
            "No GoldWeightCalculators configured. Skipping IDR/DAR and SPE."
        )
        return {}, {}

    # N_v: the full evaluated task count, identical to the SR denominator. Tasks
    # with no rollout record were backfilled as failed and must still count.
    total_tasks_by_version: Dict[str, int] = {
        v: len(results) for v, results in eval_results_by_version.items()
    }

    cross_version_by_weight: Dict[str, Dict[str, Any]] = {}
    spe_by_weight: Dict[str, Dict[str, Any]] = {}

    for weight_mode in weight_modes:
        task_weights_by_version: Dict[str, Dict[str, TaskWeightInfo]] = {}

        for v in versions:
            if v not in eval_results_by_version:
                continue
            samples = samples_by_version.get(v, [])
            if not samples:
                logger.debug("Skipping version %s: no samples loaded", v)
                continue

            tw = tool_weights if weight_mode == "cost" else None
            try:
                task_weights_by_version[v] = _build_task_weights_for_version(
                    results=eval_results_by_version[v],
                    samples=samples,
                    weight_mode=weight_mode,
                    tool_weights=tw,
                )
            except Exception as e:
                logger.warning(
                    "Failed to build task weights for %s [%s]: %s", v, weight_mode, e
                )

        # ── SPE: per version, no pairing needed ──
        try:
            spe = compute_all_spe_metrics(
                task_weights_by_version=task_weights_by_version,
                gold_weight_calculators=gold_weight_calculators,
                weight_mode=weight_mode,
                total_tasks_by_version=total_tasks_by_version,
                version_order=versions,
            )
            spe_by_weight[weight_mode] = {
                version: metrics.to_dict() for version, metrics in spe.items()
            }
        except Exception as e:
            logger.warning("Failed to compute SPE for [%s]: %s", weight_mode, e)
            spe_by_weight[weight_mode] = {"error": str(e)}

        # ── IDR/DAR: needs at least two versions ──
        if len(task_weights_by_version) < 2:
            cross_version_by_weight[weight_mode] = {}
            continue

        try:
            comparisons = compute_all_cross_version_metrics(
                task_weights_by_version=task_weights_by_version,
                gold_weight_calculators=gold_weight_calculators,
                weight_mode=weight_mode,
                version_order=versions,
            )
            cross_version_by_weight[weight_mode] = {
                pair_name: metrics.to_dict()
                for pair_name, metrics in comparisons.items()
            }
        except Exception as e:
            logger.warning("Failed to compute IDR/DAR for [%s]: %s", weight_mode, e)
            cross_version_by_weight[weight_mode] = {"error": str(e)}

    return cross_version_by_weight, spe_by_weight


# ─────────────────────────────────────────────────────────────────────────────
# Core Logic
# ─────────────────────────────────────────────────────────────────────────────


def run_multi_version_evaluation(
    experiment_name: str,
    versions: List[str],
    rollout_pattern: str,
    output_dir: Path,
    workers: int = 8,
    weight_modes: Optional[List[str]] = None,
    report_format: str = "both",
    verbose: bool = False,
    data_dir: Optional[Path] = None,
) -> MultiVersionReport:
    """
    Run multi-version aggregated evaluation.

    Steps:
    1. Load per-version evaluation results (from previous the evaluate subcommand runs)
    2. Load samples for weight-based metric computation
    3. Compute SPE per version and IDR/DAR per adjacent pair
    4. Build and save the multi-version report

    Args:
        experiment_name: Experiment identifier.
        versions: Ordered list of versions to compare.
        rollout_pattern: Run name pattern (e.g., "my_run_test_{version}").
        output_dir: Output directory for the aggregated report.
        workers: Workers for per-version evaluation.
        weight_modes: Weight modes (default: ["unit", "cost"]).
        report_format: Output format ("json", "markdown", "both").
        verbose: Enable verbose output.
        data_dir: Data root override; otherwise use DTAU_BENCH_DATA_ROOT or project data/.

    Returns:
        MultiVersionReport with all aggregated results.
    """
    if weight_modes is None:
        weight_modes = ["unit", "cost"]
    if not weight_modes:
        raise IncompleteEvaluationError("At least one weight mode is required")
    if not versions:
        raise IncompleteEvaluationError("At least one version is required")
    duplicate_versions = sorted({v for v in versions if versions.count(v) > 1})
    if duplicate_versions:
        raise IncompleteEvaluationError(
            f"Duplicate requested versions are not allowed: {duplicate_versions}"
        )

    data_dir = get_data_root(data_dir)
    experiments_dir = Path.cwd() / "experiments"
    roster = _load_test_task_roster(str(data_dir), strict=True)
    if not roster:
        raise IncompleteEvaluationError(
            f"Canonical test roster is missing or empty under "
            f"{data_dir / 'tasks' / 'test'}"
        )

    t_start = time.perf_counter()
    output_dir = Path(output_dir)

    # Step 1: Load evaluation results for each version
    logger.info("Loading evaluation results for %d versions...", len(versions))

    # Authoritative source for version-translated gold trajectories.
    tasks_dir = Path(data_dir) / "tasks"
    task_provider: Optional[TaskMetadataProvider] = None
    if tasks_dir.is_dir():
        task_provider = TaskMetadataProvider(tasks_dir)
        logger.info("TaskMetadataProvider loaded: %d tasks indexed", len(task_provider))
    else:
        logger.warning(
            "Tasks directory not found at %s; tool-retrieval metrics will be "
            "reported as non-computable (version gold unavailable).",
            tasks_dir,
        )

    eval_results_by_version: Dict[str, List[SampleEvalResult]] = {}
    samples_by_version: Dict[str, List[EvalSample]] = {}
    retrieval_by_version: Dict[str, RetrievalSummary] = {}
    missing_versions: List[str] = []
    incomplete_reasons: List[str] = []

    for version in versions:
        run_name = resolve_run_name(rollout_pattern, version)
        eval_dir = experiments_dir / run_name / "evaluation"

        # Try to load pre-computed results (backfilling missing gold tasks)
        try:
            results = load_evaluation_results(
                eval_dir,
                data_dir=data_dir,
                version=version,
                strict=True,
            )
        except IncompleteEvaluationError as exc:
            missing_versions.append(version)
            incomplete_reasons.append(f"{version} evaluation: {exc}")
            logger.error("  %s: %s", version, exc)
            continue
        if not results:
            missing_versions.append(version)
            incomplete_reasons.append(
                f"{version} evaluation: no results at {eval_dir}"
            )
            logger.error(
                "  %s: no evaluation results at %s. Run "
                "'python -m dtau_bench evaluate --run-name %s --version %s' first.",
                version,
                eval_dir,
                run_name,
                version,
            )
            continue
        # Rollouts are mandatory inputs to SPE/IDR/DAR. Load without a version
        # filter so a mislabeled row is rejected instead of silently skipped.
        rollouts_dir = experiments_dir / run_name / "rollouts"
        rollout_source: Optional[Path] = None
        if rollouts_dir.is_dir():
            rollout_source = rollouts_dir
        elif (experiments_dir / run_name).is_dir():
            run_dir = experiments_dir / run_name
            jsonl_files = list(run_dir.glob("*.jsonl"))
            if jsonl_files:
                rollout_source = run_dir

        if rollout_source is None:
            missing_versions.append(version)
            incomplete_reasons.append(
                f"{version} rollout: no JSONL files under "
                f"{experiments_dir / run_name}"
            )
            continue

        try:
            samples = load_rollout_dir(rollout_source)
            _validate_rollout_samples(
                samples,
                roster=roster,
                version=version,
                source=rollout_source,
                raw_record_count=_count_nonempty_jsonl_records(rollout_source),
            )
        except Exception as exc:
            missing_versions.append(version)
            incomplete_reasons.append(f"{version} rollout: {exc}")
            logger.error("  %s: invalid rollout samples: %s", version, exc)
            continue

        eval_results_by_version[version] = results
        samples_by_version[version] = samples
        logger.info(
            "  %s: loaded %d evaluation result(s) and rollout sample(s)",
            version,
            len(results),
        )

        # Compute retrieval metrics from samples
        # Gold for retrieval must be version-translated; rollout JSONL only
        # carries the base sequence.
        if task_provider is not None:
            attach_version_trajectories(samples, task_provider)
        sample_metrics = [compute_retrieval_metrics(s) for s in samples]
        ret_summary = summarize_retrieval_metrics(sample_metrics)
        if ret_summary.samples_with_tool_retrieval > 0:
            retrieval_by_version[version] = ret_summary

    if missing_versions:
        raise IncompleteEvaluationError(
            "Cannot aggregate incomplete evaluations. Missing or invalid versions: "
            + ", ".join(dict.fromkeys(missing_versions))
            + ". "
            + " | ".join(incomplete_reasons)
        )

    if not eval_results_by_version:
        raise IncompleteEvaluationError(
            "No evaluation results available. Cannot generate multi-version report."
        )

    # Step 2: Compute comparison metrics
    logger.info("Computing cross-version metrics...")

    # Initialize gold weight calculators from data directory (one per domain)
    gold_weight_calculators: Dict[str, GoldWeightCalculator] = {}
    tool_weights: Optional[Dict[str, float]] = None

    version_mapping_dir = data_dir / "meta" / "version_mapping"
    if version_mapping_dir.is_dir():
        try:
            for vmap_file in version_mapping_dir.glob("*.json"):
                domain_key = vmap_file.stem
                with open(vmap_file, "r", encoding="utf-8") as f:
                    version_mapping = json.load(f)

                # Load corresponding evolution forest
                evolution_forest = None
                forest_path = (
                    data_dir / "meta" / "evolution_forests" / f"{domain_key}_forest.json"
                )
                if forest_path.exists():
                    with open(forest_path, "r", encoding="utf-8") as f:
                        evolution_forest = json.load(f)

                # Load corresponding tool weights by version
                # tool_attributes format: {tool_name: {cost: int, latency: int, ...}}
                # GoldWeightCalculator expects: {version: {tool_name: float}}
                tool_weights_by_version: Optional[Dict[str, Dict[str, float]]] = None
                domain_attrs_dir = data_dir / "meta" / "tool_attributes" / domain_key
                if domain_attrs_dir.is_dir():
                    tool_weights_by_version = {}
                    for tw_file in domain_attrs_dir.glob("*.json"):
                        try:
                            with open(tw_file, "r", encoding="utf-8") as f:
                                tw_data = json.load(f)
                            # Extract cost as weight; fall back to 1.0 if missing
                            tool_weights_by_version[tw_file.stem] = {
                                tool_name: float(attrs.get("cost", 1.0))
                                if isinstance(attrs, dict)
                                else float(attrs)
                                for tool_name, attrs in tw_data.items()
                            }
                        except Exception:
                            pass

                gold_weight_calculators[domain_key] = GoldWeightCalculator(
                    version_mapping=version_mapping,
                    evolution_forest=evolution_forest,
                    tool_weights_by_version=tool_weights_by_version,
                )

            logger.info(
                "Initialized %d GoldWeightCalculators from %s",
                len(gold_weight_calculators),
                version_mapping_dir,
            )
        except Exception as e:
            logger.warning("Failed to initialize GoldWeightCalculators: %s", e)
            gold_weight_calculators = {}
    else:
        logger.info(
            "No version_mapping directory found at %s. "
            "Version comparison metrics will be skipped.",
            version_mapping_dir,
        )

    # Load flat tool_costs for agent weight computation (cost mode)
    tool_costs_path = data_dir / "meta" / "tool_costs.json"
    if tool_costs_path.exists():
        with open(tool_costs_path, "r", encoding="utf-8") as f:
            tool_weights = json.load(f)

    cross_version_metrics, spe_metrics = compute_weight_based_metrics(
        eval_results_by_version=eval_results_by_version,
        samples_by_version=samples_by_version,
        # Preserve the user-requested order. If weight inputs for a middle
        # version are unavailable, cross-version metrics skip both adjacent
        # pairs instead of incorrectly joining its neighbours.
        versions=versions,
        weight_modes=weight_modes,
        gold_weight_calculators=gold_weight_calculators,
        tool_weights=tool_weights,
    )
    _validate_weight_metric_coverage(
        cross_version_metrics,
        spe_metrics,
        versions=versions,
        weight_modes=weight_modes,
        roster_size=len(roster),
    )

    if not tool_costs_path.exists() and "cost" in weight_modes:
        logger.warning(
            "%s not found: agent-side weights fall back to unit cost while "
            "reference weights keep real costs. Cost-SPE / Cost-IDR / Cost-DAR "
            "will be inflated.",
            tool_costs_path,
        )

    # Step 3: Build report
    logger.info("Building multi-version report...")

    report = build_multi_version_report(
        eval_results_by_version=eval_results_by_version,
        versions=versions,
        weight_modes=weight_modes,
        retrieval_by_version=retrieval_by_version if retrieval_by_version else None,
        cross_version_metrics_by_weight=cross_version_metrics,
        spe_metrics_by_weight=spe_metrics,
    )

    # Add extra metadata
    report.metadata["experiment_name"] = experiment_name
    report.metadata["rollout_pattern"] = rollout_pattern
    report.metadata["elapsed_seconds"] = round(time.perf_counter() - t_start, 2)
    report.metadata["versions_with_results"] = [
        v for v in versions if v in eval_results_by_version
    ]
    report.metadata["versions_missing"] = missing_versions
    report.metadata["versions_with_results"] = [
        v for v in versions if v in eval_results_by_version
    ]

    # Step 4: Save report
    output_dir.mkdir(parents=True, exist_ok=True)
    if report_format in ("json", "both"):
        report.save_json(output_dir / "multi_version_report.json")

    if report_format in ("markdown", "both"):
        report.save_markdown(output_dir / "multi_version_report.md")

    t_elapsed = time.perf_counter() - t_start
    logger.info(
        "Multi-version evaluation complete in %.1fs. Report: %s",
        t_elapsed,
        output_dir,
    )

    return report


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run multi-version aggregated evaluation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--experiment-name",
        required=True,
        help="Experiment name identifier",
    )
    parser.add_argument(
        "--versions",
        nargs="+",
        required=True,
        help='Ordered list of versions to evaluate (e.g., v1 v2 v3 v4 v5 v6)',
    )
    parser.add_argument(
        "--rollout-pattern",
        required=True,
        help='Run name pattern, use {version} placeholder (e.g., "test_evolve_{version}")',
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for the aggregated report",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel workers per version (default: 8)",
    )
    parser.add_argument(
        "--weight-modes",
        nargs="+",
        choices=["unit", "cost"],
        default=["unit", "cost"],
        help=(
            "Trajectory weight modes (default: unit cost). unit counts function "
            "calls, cost sums real execution cost."
        ),
    )
    parser.add_argument(
        "--report-format",
        choices=["json", "markdown", "both"],
        default="both",
        help="Report output format (default: both)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=False,
        help="Enable verbose output",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        default=False,
        help="Suppress info-level output",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=None,
        help="Data root (default: DTAU_BENCH_DATA_ROOT or {project_root}/data).",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Configure logging
    if args.quiet:
        log_level = logging.WARNING
    elif args.verbose:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


    # Run multi-version evaluation
    try:
        report = run_multi_version_evaluation(
            experiment_name=args.experiment_name,
            versions=args.versions,
            rollout_pattern=args.rollout_pattern,
            output_dir=Path(args.output_dir),
            workers=args.workers,
            weight_modes=args.weight_modes,
            report_format=args.report_format,
            verbose=args.verbose,
            data_dir=args.data_dir,
        )
    except IncompleteEvaluationError as exc:
        logger.error("Multi-version evaluation failed: %s", exc)
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)

    # Print summary
    print(f"\n{'='*60}")
    print(f"Multi-Version Evaluation Complete: {args.experiment_name}")
    print(f"{'='*60}")
    print(f"  Versions: {', '.join(args.versions)}")
    comparison_order = args.versions
    print(f"  Comparison Order: {' → '.join(comparison_order)}")
    versions_with = report.metadata.get("versions_with_results", [])
    versions_missing = report.metadata.get("versions_missing", [])
    print(f"  With Results: {', '.join(versions_with)}")
    if versions_missing:
        print(f"  Missing: {', '.join(versions_missing)}")
    print(f"  Output: {args.output_dir}")

    # Print SR
    if report.version_summaries:
        print("\n  SR:")
        for v in args.versions:
            vs = report.version_summaries.get(v, {})
            print(f"    {v}: {vs.get('sr', 0):.2%} ({vs.get('total', 0)} samples)")

    # Print SPE per weight mode
    for weight_mode, version_dict in sorted(report.spe_metrics_by_weight.items()):
        if not isinstance(version_dict, dict):
            continue
        print(f"\n  SPE [{weight_mode}]:")
        for v in args.versions:
            metrics = version_dict.get(v)
            if isinstance(metrics, dict) and metrics.get("spe") is not None:
                print(f"    {v}: {metrics['spe']:.4f}")

    # Print IDR/DAR per weight mode
    for weight_mode, pair_dict in sorted(
        report.cross_version_metrics_by_weight.items()
    ):
        if not isinstance(pair_dict, dict):
            continue
        print(f"\n  IDR / DAR [{weight_mode}]:")
        for pair_name, metrics in sorted(pair_dict.items()):
            if not isinstance(metrics, dict):
                continue
            idr = metrics.get("idr")
            dar = metrics.get("dar")
            print(
                f"    {pair_name}: IDR={'N/A' if idr is None else f'{idr:.4f}'} "
                f"DAR={'N/A' if dar is None else f'{dar:.4f}'}"
            )

    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
