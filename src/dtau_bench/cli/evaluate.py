#!/usr/bin/env python3
"""Run Evaluate — Single-version environment evaluation CLI.

Evaluates agent rollout samples against reference trajectories for one version:
1. Load rollout samples from experiments/{run_name}/rollouts/
2. Run environment evaluation (gold replay + agent replay + DB diff)
3. Compute SR and tool retrieval P/R/F1
4. Save evaluation report (JSON + Markdown)

Usage:
    python -m dtau_bench evaluate --run-name my_run_v2 --version v2 --workers 64
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Optional

from dtau_bench.evaluator.data_loader import (
    attach_version_trajectories,
    load_rollout_dir,
    load_rollout_samples,
    TaskMetadataProvider,
)
from dtau_bench.evaluator.engine import EnvironmentEvaluationEngine
from dtau_bench.evaluator.models import EvalSample, SampleEvalResult
from dtau_bench.evaluator.metrics.retrieval import (
    compute_retrieval_metrics,
    summarize_retrieval_metrics,
)
from dtau_bench.evaluator.report import (
    SingleVersionReport,
    build_single_version_report,
)
from dtau_bench.utils.config import get_data_root

logger = logging.getLogger("dtau_bench.cli.evaluate")


# ─────────────────────────────────────────────────────────────────────────────
# Core Logic
# ─────────────────────────────────────────────────────────────────────────────


def resolve_rollout_path(run_name: str) -> Path:
    """
    Resolve the rollout directory path for a run.

    Looks for:
    1. experiments/{run_name}/rollouts/
    2. experiments/{run_name}/ (if it contains .jsonl files directly)

    Args:
        run_name: Name of the rollout run.

    Returns:
        Path to the directory containing rollout JSONL files.

    Raises:
        FileNotFoundError: If no valid rollout path is found.
    """
    experiments_dir = Path.cwd() / "experiments"

    # Check experiments/{run_name}/rollouts/ first
    rollouts_dir = experiments_dir / run_name / "rollouts"
    if rollouts_dir.is_dir():
        return rollouts_dir

    # Fallback: experiments/{run_name}/ with JSONL files
    run_dir = experiments_dir / run_name
    if run_dir.is_dir():
        jsonl_files = list(run_dir.glob("*.jsonl"))
        if jsonl_files:
            return run_dir

    raise FileNotFoundError(
        f"No rollout data found for run '{run_name}' in {experiments_dir}. "
        f"Expected at: {rollouts_dir} or {run_dir}/*.jsonl"
    )


def run_single_version_evaluation(
    run_name: str,
    version: str,
    output_dir: Path,
    workers: int = 64,
    verbose: bool = False,
    data_dir: Optional[Path] = None,
) -> SingleVersionReport:
    """
    Run a complete single-version evaluation pipeline.

    Steps:
    1. Load rollout samples
    2. Run environment evaluation (engine)
    3. Compute tool retrieval metrics
    4. Build and save report

    Args:
        run_name: Rollout run name (experiments/{run_name}/).
        version: Tool version to evaluate.
        output_dir: Directory to save evaluation output.
        workers: Number of parallel evaluation workers.
        verbose: Enable verbose logging.
        data_dir: Data root override; otherwise use DTAU_BENCH_DATA_ROOT or project data/.

    Returns:
        SingleVersionReport with all results.
    """
    t_start = time.perf_counter()

    # Step 1: Resolve and load rollout samples
    logger.info("Loading rollout samples for run: %s, version: %s", run_name, version)

    data_dir = get_data_root(data_dir)
    rollout_path = resolve_rollout_path(run_name)
    logger.info("Rollout path resolved: %s", rollout_path)

    if rollout_path.is_file():
        samples = load_rollout_samples(rollout_path, version_filter=version)
    else:
        samples = load_rollout_dir(rollout_path, version_filter=version)

    if not samples:
        logger.warning("No samples loaded for version %s in %s", version, rollout_path)
        # Return empty report
        report = build_single_version_report(
            [], version=version, run_name=run_name, workers=workers,
        )
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        report.save_json(output_dir / "evaluation_report.json")
        return report

    logger.info("Loaded %d samples", len(samples))

    # Step 2: Run environment evaluation
    logger.info("Starting environment evaluation with %d workers...", workers)

    # Initialize TaskMetadataProvider — authoritative source for gold metadata
    tasks_dir = data_dir / "tasks"
    if not tasks_dir.is_dir():
        raise FileNotFoundError(
            f"Tasks directory not found at {tasks_dir}. "
            f"Gold metadata must come from data/tasks/ task JSON files."
        )
    task_provider = TaskMetadataProvider(tasks_dir)
    logger.info("TaskMetadataProvider loaded: %d tasks indexed", len(task_provider))

    # Tool-retrieval gold must be the version-translated sequence, which rollout
    # JSONL does not carry. Enrich from the canonical task data.
    attach_stats = attach_version_trajectories(samples, task_provider)
    logger.info(
        "Version gold attached: %d (already present: %d, missing task: %d)",
        attach_stats["attached"],
        attach_stats["already_present"],
        attach_stats["missing_task"],
    )

    engine = EnvironmentEvaluationEngine(
        task_provider=task_provider,
        data_dir=str(data_dir),
        verbose=verbose,
    )

    results = engine.evaluate(samples=samples, version=version, workers=workers)

    logger.info(
        "Evaluation complete: %d/%d passed (%.1f%%)",
        sum(1 for r in results if r.passed),
        len(results),
        100.0 * sum(1 for r in results if r.passed) / len(results) if results else 0,
    )

    # Step 3: Compute retrieval metrics
    logger.info("Computing retrieval metrics...")
    retrieval_metrics = [compute_retrieval_metrics(s) for s in samples]
    retrieval_summary = summarize_retrieval_metrics(retrieval_metrics)

    if retrieval_summary.samples_with_tool_retrieval > 0:
        logger.info(
            "Retrieval: %d applicable samples, F1=%.4f",
            retrieval_summary.samples_with_tool_retrieval,
            retrieval_summary.f1,
        )

    # Step 4: Build and save report
    t_elapsed = time.perf_counter() - t_start
    report = build_single_version_report(
        results,
        version=version,
        run_name=run_name,
        workers=workers,
        retrieval_summary=retrieval_summary,
    )
    report.metadata["elapsed_seconds"] = round(t_elapsed, 2)

    # Save outputs
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report.save_json(output_dir / "evaluation_report.json")
    report.save_markdown(output_dir / "evaluation_report.md")

    # Also save raw results JSONL for downstream consumption
    results_jsonl_path = output_dir / "evaluation_results.jsonl"
    with open(results_jsonl_path, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")
    logger.info("Saved %d raw results to %s", len(results), results_jsonl_path)

    logger.info(
        "Single-version evaluation complete in %.1fs. Report: %s",
        t_elapsed,
        output_dir / "evaluation_report.json",
    )

    return report


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run single-version environment evaluation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--run-name",
        required=True,
        help="Rollout run name (points to experiments/{run_name}/)",
    )
    parser.add_argument(
        "--version",
        required=True,
        help="Tool version to evaluate (e.g., v2)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Evaluation output directory (default: experiments/{run_name}/evaluation/)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=64,
        help="Number of parallel worker processes (default: 64)",
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

    # Resolve output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path.cwd() / "experiments" / args.run_name / "evaluation"

    report = run_single_version_evaluation(
        run_name=args.run_name,
        version=args.version,
        output_dir=output_dir,
        workers=args.workers,
        verbose=args.verbose,
        data_dir=args.data_dir,
    )

    # Print summary
    summary = report.summary
    print(f"\n{'='*60}")
    print(f"Evaluation Complete: {args.version}")
    print(f"{'='*60}")
    print(f"  Total:  {summary.get('total', 0)}")
    print(f"  Passed: {summary.get('passed', 0)}")
    print(f"  Failed: {summary.get('failed', 0)}")
    print(f"  SR:     {summary.get('sr', 0):.2%}")
    print(f"  Output: {output_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
