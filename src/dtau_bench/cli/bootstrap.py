#!/usr/bin/env python3
"""Bootstrap the first skill library from rollout trajectories.

Four stages: dataset -> planning/atomic extraction -> tool skills -> graph.

LLM and embedding endpoints are read from configs/models.yaml; the remaining
hyperparameters live in the pipeline defaults.

Usage:
    python -m dtau_bench bootstrap \
      --run-name bootstrap_v1 \
      --rollout-dir experiments/train_v1/rollouts/ \
      --eval-file experiments/train_v1/evaluation/evaluation_results.jsonl \
      --tool-schema-version v1 \
      --model DeepSeek-V3.2
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from dtau_bench.utils.config import get_data_root, resolve_embedding_config, resolve_model_endpoint


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bootstrap the first skill library from rollout + eval artifacts.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--run-name", type=str, required=True,
        help="Experiment name (determines the output directory).",
    )
    parser.add_argument(
        "--rollout-dir", type=Path, required=True,
        help="Rollout output directory (one *.jsonl per domain).",
    )
    parser.add_argument(
        "--tool-schema-version", type=str, required=True,
        help="Tool schema version (e.g., v1).",
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help="Model key from configs/models.yaml.",
    )
    parser.add_argument(
        "--eval-file", type=Path, required=True,
        help=(
            "Required evaluation_results.jsonl from the evaluate subcommand; "
            "this supplies the authoritative pass/fail reward signal."
        ),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory. Default: experiments/{run_name}/skills.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from the last completed stage.",
    )
    parser.add_argument(
        "--batch-size", type=_positive_int, default=2,
        help="LLM processing batch size (default: 2).",
    )
    parser.add_argument(
        "--max-concurrent", type=_positive_int, default=2,
        help="Maximum concurrent LLM batches (default: 2).",
    )
    parser.add_argument(
        "--merge-embedding-batch-size", type=_positive_int, default=8,
        help="Embedding batch size for skill merging/clustering (default: 8).",
    )

    parser.add_argument(
        "--data-dir", type=Path, default=None,
        help="Data root (default: DTAU_BENCH_DATA_ROOT or {project_root}/data).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("dtau_bench.cli.bootstrap")

    endpoint = resolve_model_endpoint(args.model)
    embedding = resolve_embedding_config()

    cwd = Path.cwd()
    data_dir = get_data_root(args.data_dir)
    output_dir = args.output_dir or (cwd / "experiments" / args.run_name / "skills")
    query_dir = data_dir / "tasks" / "train"
    tool_schemas_dir = data_dir / "tools"
    tool_attributes_dir = data_dir / "meta" / "tool_attributes"

    if not args.rollout_dir.exists():
        logger.error(f"Rollout directory not found: {args.rollout_dir}")
        sys.exit(1)

    from dtau_bench.skill.bootstrap.pipeline import BootstrapConfig, BootstrapPipeline

    config = BootstrapConfig(
        output_dir=output_dir,
        rollout_dir=args.rollout_dir,
        query_dir=query_dir,
        tool_schemas_dir=tool_schemas_dir,
        tool_attributes_dir=tool_attributes_dir,
        tool_schema_version=args.tool_schema_version,
        model=endpoint["model_name"],
        base_url=endpoint["base_url"],
        api_key=endpoint["api_key"],
        max_tokens=endpoint["max_tokens"],
        llm_request_options=endpoint["request_options"],
        embedding_api_base=embedding["base_url"],
        embedding_api_key=embedding["api_key"],
        embedding_model=embedding["model"],
        eval_file=args.eval_file,
        resume=args.resume,
        batch_size=args.batch_size,
        max_concurrent=args.max_concurrent,
        merge_embedding_batch_size=args.merge_embedding_batch_size,
    )

    logger.info("=" * 72)
    logger.info(f"Bootstrap Pipeline: {args.run_name}")
    logger.info("=" * 72)
    logger.info(f"  Output:       {output_dir}")
    logger.info(f"  Rollout dir:  {args.rollout_dir}")
    logger.info(f"  Eval file:    {args.eval_file}")
    logger.info(f"  Tool version: {args.tool_schema_version}")
    logger.info(f"  Model:        {endpoint['model_name']} @ {endpoint['base_url']}")
    logger.info(f"  Embedding:    {embedding['model']} @ {embedding['base_url']}")
    logger.info(
        "  Resources:    batch_size=%d, max_concurrent=%d, "
        "merge_embedding_batch_size=%d",
        args.batch_size,
        args.max_concurrent,
        args.merge_embedding_batch_size,
    )
    logger.info(f"  Resume:       {args.resume}")
    logger.info("=" * 72)

    pipeline = BootstrapPipeline(config)
    try:
        manifest = pipeline.run()
        logger.info("Bootstrap pipeline completed successfully!")
        logger.info(f"Library output: {manifest['outputs']['library_dir']}")
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
    except Exception as e:
        logger.error(f"Bootstrap pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
