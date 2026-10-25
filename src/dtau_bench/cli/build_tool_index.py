#!/usr/bin/env python3
"""Build the tool retrieval index for one catalog version.

Embeds every tool in data/tools/{domain}/{version}_tools.json so that
retrieval strategies can load the index instantly at rollout time. The
embedding endpoint comes from configs/models.yaml (``embedding.default``).

Usage:
    python -m dtau_bench build-tool-index --version v1
    python -m dtau_bench build-tool-index --version v1 --output-dir ./rag_index_cache/v1

Output:
    {output_dir}/
    ├── embeddings.npy      # (N, dim) float32
    ├── documents.json      # tool metadata
    └── metadata.json       # index metadata
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

from dtau_bench.rag.config import RAGConfig
from dtau_bench.rag.indexer import ToolIndexer
from dtau_bench.utils.config import get_data_root, resolve_embedding_config


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

QUERY_INSTRUCTION = (
    "Given a user request, retrieve relevant tools that can help complete the task"
)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _validate_saved_index(indexer: ToolIndexer, output_dir: Path, version: str) -> dict:
    """Load and validate every persisted tool-index artifact."""
    required = [
        output_dir / "embeddings.npy",
        output_dir / "documents.json",
        output_dir / "metadata.json",
    ]
    missing = [str(path) for path in required if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise ValueError(f"missing or empty index artifacts: {missing}")

    index = indexer.load_index(str(output_dir))
    embeddings = index.get("embeddings")
    documents = index.get("documents")
    metadata = index.get("metadata")
    if not isinstance(embeddings, np.ndarray) or embeddings.ndim != 2:
        raise ValueError(f"embeddings must be a rank-2 array, got {getattr(embeddings, 'shape', None)}")
    if embeddings.shape[0] <= 0 or embeddings.shape[1] <= 0:
        raise ValueError(f"embeddings have invalid shape {embeddings.shape}")
    if not isinstance(documents, list) or len(documents) != embeddings.shape[0]:
        raise ValueError(
            f"document/embedding count mismatch: documents="
            f"{len(documents) if isinstance(documents, list) else 'invalid'}, "
            f"embeddings={embeddings.shape[0]}"
        )
    if not isinstance(metadata, dict):
        raise ValueError("metadata.json must contain a JSON object")
    if metadata.get("source_version") != version:
        raise ValueError(
            f"index source_version={metadata.get('source_version')!r}, expected {version!r}"
        )
    if metadata.get("total_tools") != len(documents):
        raise ValueError(
            f"metadata total_tools={metadata.get('total_tools')!r}, expected {len(documents)}"
        )
    if metadata.get("embedding_dim") != embeddings.shape[1]:
        raise ValueError(
            f"metadata embedding_dim={metadata.get('embedding_dim')!r}, "
            f"expected {embeddings.shape[1]}"
        )
    document_domains = sorted({doc.domain for doc in documents})
    if metadata.get("domains") != document_domains:
        raise ValueError(
            f"metadata domains do not match indexed documents: "
            f"{metadata.get('domains')!r} != {document_domains!r}"
        )
    if not indexer.validate_index(index):
        raise ValueError("index metadata does not match the current embedding config")
    return index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the tool retrieval index for one catalog version.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--version", type=str, required=True,
        help="Catalog version to index (e.g., v1).",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: ./rag_index_cache/{version}).",
    )
    parser.add_argument(
        "--validate-only", action="store_true",
        help="Only validate an existing index.",
    )
    parser.add_argument(
        "--batch-size", type=_positive_int, default=None,
        help=(
            "Embedding request batch size "
            "(default: embedding.default.batch_size in configs/models.yaml)."
        ),
    )
    parser.add_argument(
        "--data-dir", type=Path, default=None,
        help="Data root (default: DTAU_BENCH_DATA_ROOT or {project_root}/data).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    embedding = resolve_embedding_config()
    config = RAGConfig(
        embedding_model=embedding["model"],
        embedding_backend=embedding["backend"],
        embedding_api_base=embedding["base_url"],
        embedding_api_key=embedding["api_key"],
        query_instruction=QUERY_INSTRUCTION,
        embedding_http_timeout=embedding.get("request_timeout", 600.0),
        embedding_http_max_retries=embedding.get("max_retries", 2),
        embedding_http_retry_delay=embedding.get("retry_delay", 1.0),
    )
    batch_size = (
        args.batch_size if args.batch_size is not None else embedding["batch_size"]
    )

    tools_dir = get_data_root(args.data_dir) / "tools"
    if not tools_dir.exists():
        print(f"[ERROR] Tools directory not found: {tools_dir}")
        sys.exit(1)

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path.cwd() / "rag_index_cache" / args.version
    )

    if args.validate_only:
        indexer = ToolIndexer(
            config=config,
            embedding_batch_size=batch_size,
        )
        try:
            index = _validate_saved_index(indexer, output_dir, args.version)
        except Exception as exc:
            print(f"[ERROR] Invalid tool index at {output_dir}: {exc}")
            sys.exit(1)
        print(f"Index at {output_dir}:")
        print(f"  Tools:   {len(index['documents'])}")
        print(f"  Domains: {len(index['metadata'].get('domains', []))}")
        print("  Valid:   yes")
        return

    print(f"\n{'═' * 60}")
    print("  Tool Index Builder")
    print(f"{'═' * 60}")
    print(f"  Version:    {args.version}")
    print(f"  Tools dir:  {tools_dir}")
    print(f"  Output dir: {output_dir}")
    print(f"  Model:      {embedding['model']} ({embedding['backend']})")
    print(f"  Batch size: {batch_size}")
    if embedding["backend"] == "openai_http":
        print(f"  API base:   {embedding['base_url']}")
    print(f"{'═' * 60}\n")

    indexer = ToolIndexer(
        config=config,
        tools_dir=tools_dir,
        version=args.version,
        embedding_batch_size=batch_size,
    )

    start_time = time.time()
    index = indexer.build_index()
    build_time = time.time() - start_time

    indexer.save_index(index, str(output_dir))
    try:
        index = _validate_saved_index(indexer, output_dir, args.version)
    except Exception as exc:
        print(f"[ERROR] Tool index failed post-write validation: {exc}")
        sys.exit(1)

    domains = index["metadata"].get("domains", [])
    print(f"\n{'═' * 60}")
    print("  Index built")
    print(f"{'═' * 60}")
    print(f"  Tools:      {len(index['documents'])}")
    print(f"  Domains:    {len(domains)}")
    print(f"  Embeddings: {index['embeddings'].shape}")
    print(f"  Time:       {build_time:.1f}s")
    print(f"  Output:     {output_dir}")
    print(f"{'═' * 60}")


if __name__ == "__main__":
    main()
