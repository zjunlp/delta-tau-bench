#!/usr/bin/env python3
"""Skill evolution — evolve a skill library from one catalog version to the next.

Six steps: feedback summary -> proposal extraction -> tool skill finalize ->
atomic skill finalize -> planning propagation -> rebuild graph and partition.

LLM and embedding endpoints are read from configs/models.yaml; the remaining
hyperparameters live in the pipeline defaults.

Usage:
    python -m dtau_bench evolve \
      --run-name evolve_to_v2 \
      --prev-library experiments/bootstrap_v1/skills/ \
      --rollout-dir experiments/train_v2/rollouts/ \
      --eval-results-file experiments/train_v2/evaluation/evaluation_results.jsonl \
      --tool-schema-version v2 \
      --model DeepSeek-V3.2
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path

from dtau_bench.utils.config import get_data_root, resolve_embedding_config, resolve_model_endpoint


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run offline graph-backed skill evolution",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--run-name", type=str, required=True,
        help="Experiment run name (used for the output directory)",
    )
    parser.add_argument(
        "--prev-library", type=Path, default=None,
        help="Previous skill library directory (omit for a cold start)",
    )
    parser.add_argument(
        "--tool-schema-version", type=str, required=True,
        help="Target tool schema version for the active/inactive partition",
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help="Model key from configs/models.yaml",
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--rollout-dir", type=Path,
        help="Rollout output directory (a dataset is built from it)",
    )
    input_group.add_argument(
        "--eval-file", type=Path,
        help="Pre-built skill-context JSONL (alternative to --rollout-dir)",
    )

    parser.add_argument(
        "--eval-results-file", type=Path, default=None,
        help=(
            "evaluation_results.jsonl from the evaluate subcommand, carrying the "
            "authoritative pass/fail signal. Required with --rollout-dir."
        ),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory (default: ./experiments/{run_name}/skills)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from the last completed step",
    )
    parser.add_argument(
        "--batch-size", type=_positive_int, default=2,
        help="LLM processing batch size (default: 2)",
    )
    parser.add_argument(
        "--max-concurrent", type=_positive_int, default=2,
        help="Maximum concurrent LLM batches (default: 2)",
    )
    parser.add_argument(
        "--merge-embedding-batch-size", type=_positive_int, default=8,
        help="Embedding batch size for skill merging/clustering (default: 8)",
    )
    parser.add_argument("--quiet", action="store_true")

    parser.add_argument(
        "--data-dir", type=Path, default=None,
        help="Data root (default: DTAU_BENCH_DATA_ROOT or {project_root}/data).",
    )
    args = parser.parse_args()
    if args.rollout_dir is not None and args.eval_results_file is None:
        parser.error("--eval-results-file is required when using --rollout-dir")
    return args


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    endpoint = resolve_model_endpoint(args.model)
    embedding = resolve_embedding_config()

    from dtau_bench.skill.evolve import EvolveConfig, EvolvePipeline

    data_dir = get_data_root(args.data_dir)
    config = EvolveConfig(
        output_dir=args.output_dir or Path("experiments") / args.run_name / "skills",
        prev_library=args.prev_library,
        rollout_dir=args.rollout_dir,
        eval_file=args.eval_file,
        eval_results_file=args.eval_results_file,
        query_dir=data_dir / "tasks" / "train",
        tool_schemas_dir=data_dir / "tools",
        tool_attributes_dir=data_dir / "meta" / "tool_attributes",
        tool_schema_version=args.tool_schema_version,
        model=endpoint["model_name"],
        base_url=endpoint["base_url"],
        api_key=endpoint["api_key"],
        max_tokens=endpoint["max_tokens"],
        llm_request_options=endpoint["request_options"],
        embedding_api_base=embedding["base_url"],
        embedding_api_key=embedding["api_key"],
        embedding_model=embedding["model"],
        resume=args.resume,
        quiet=args.quiet,
        batch_size=args.batch_size,
        max_concurrent=args.max_concurrent,
        merge_embedding_batch_size=args.merge_embedding_batch_size,
    )

    logging.getLogger("dtau_bench.cli.evolve").info(
        "Resources: batch_size=%d, max_concurrent=%d, "
        "merge_embedding_batch_size=%d",
        args.batch_size,
        args.max_concurrent,
        args.merge_embedding_batch_size,
    )
    pipeline = EvolvePipeline(config)
    asyncio.run(pipeline.run())


if __name__ == "__main__":
    main()
