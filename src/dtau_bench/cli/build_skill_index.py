#!/usr/bin/env python3
"""Build the skill retrieval indices for one skill library.

Embeds the planning, atomic-skill and tool-skill layers of a skill library so
that rollout can retrieve from them without recomputing embeddings. The
embedding endpoint comes from configs/models.yaml (``embedding.default``).

Usage:
    python -m dtau_bench build-skill-index \\
      --skill-library-dir experiments/bootstrap_v1/skills/

Output:
    {output_dir}/
    ├── planning/      embeddings.npy  documents.json  metadata.json
    ├── skills/        ...
    └── tool_skills/   ...
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

from dtau_bench.utils.config import resolve_embedding_config


QUERY_INSTRUCTION = (
    "Given a novel and complex agent task, retrieve relevant reusable skills "
    "that can guide solving this task"
)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _required_index_kinds(skill_library_dir: Path) -> list[str]:
    """Return persisted sub-indices required by the current library."""
    from dtau_bench.skill.loader import SkillLibraryLoader

    artifacts = SkillLibraryLoader.load(skill_library_dir, mode="normal")
    if not artifacts.plans:
        raise ValueError(f"skill library contains no plans: {skill_library_dir}")
    if not artifacts.skills:
        raise ValueError(f"skill library contains no active skills: {skill_library_dir}")
    kinds = ["planning", "skills"]
    if artifacts.tool_skills:
        kinds.append("tool_skills")
    return kinds


def _validate_saved_indices(indexer, skill_library_dir: Path, output_dir: Path) -> dict:
    """Load every required sub-index and verify files, shapes, and metadata."""
    loaded = {}
    for kind in _required_index_kinds(skill_library_dir):
        index_dir = output_dir / kind
        required = [
            index_dir / "embeddings.npy",
            index_dir / "documents.json",
            index_dir / "metadata.json",
        ]
        missing = [
            str(path) for path in required
            if not path.is_file() or path.stat().st_size == 0
        ]
        if missing:
            raise ValueError(f"{kind} index is missing artifacts: {missing}")

        index = indexer.load_index(index_dir, kind=kind)
        embeddings = index.get("embeddings")
        documents = index.get("documents")
        metadata = index.get("metadata")
        if not isinstance(embeddings, np.ndarray) or embeddings.ndim != 2:
            raise ValueError(
                f"{kind} embeddings must be rank 2, got {getattr(embeddings, 'shape', None)}"
            )
        if embeddings.shape[0] <= 0 or embeddings.shape[1] <= 0:
            raise ValueError(f"{kind} embeddings have invalid shape {embeddings.shape}")
        if not isinstance(documents, list) or len(documents) != embeddings.shape[0]:
            raise ValueError(
                f"{kind} document/embedding count mismatch: documents="
                f"{len(documents) if isinstance(documents, list) else 'invalid'}, "
                f"embeddings={embeddings.shape[0]}"
            )
        if not isinstance(metadata, dict):
            raise ValueError(f"{kind} metadata must be a JSON object")
        if metadata.get("kind") != kind:
            raise ValueError(
                f"{kind} metadata kind={metadata.get('kind')!r}"
            )
        if metadata.get("total_documents") != len(documents):
            raise ValueError(
                f"{kind} metadata total_documents="
                f"{metadata.get('total_documents')!r}, expected {len(documents)}"
            )
        if metadata.get("embedding_dim") != embeddings.shape[1]:
            raise ValueError(
                f"{kind} metadata embedding_dim={metadata.get('embedding_dim')!r}, "
                f"expected {embeddings.shape[1]}"
            )
        source_dir = metadata.get("source_dir")
        if not source_dir or Path(source_dir).resolve() != skill_library_dir.resolve():
            raise ValueError(
                f"{kind} index source_dir={source_dir!r}, "
                f"expected {str(skill_library_dir.resolve())!r}"
            )
        if not indexer.validate_index(index):
            raise ValueError(
                f"{kind} metadata does not match the current embedding config"
            )
        loaded[kind] = index
    return loaded


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build skill retrieval indices for a skill library.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--skill-library-dir", type=str, required=True,
        help="Skill library directory (plan_library.json, *_skills.prepared.jsonl).",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: {skill_library_dir}/index_cache).",
    )
    parser.add_argument(
        "--validate", action="store_true",
        help="Only validate an existing index.",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Verbose output.",
    )
    parser.add_argument(
        "--batch-size", type=_positive_int, default=None,
        help=(
            "Embedding request batch size "
            "(default: embedding.default.batch_size in configs/models.yaml)."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    from dtau_bench.rag.config import RAGConfig
    from dtau_bench.skill.indexer import SkillIndexer

    skill_library_dir = Path(args.skill_library_dir).resolve()
    if not skill_library_dir.exists():
        print(f"[ERROR] Skill library not found: {skill_library_dir}")
        sys.exit(1)

    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else skill_library_dir / "index_cache"
    )

    embedding = resolve_embedding_config()
    rag_config = RAGConfig(
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
    indexer = SkillIndexer(
        config=rag_config,
        embedding_batch_size=batch_size,
    )

    if args.validate:
        print(f"Validating skill index at {output_dir}...")
        try:
            loaded = _validate_saved_indices(indexer, skill_library_dir, output_dir)
        except Exception as exc:
            print(f"[ERROR] Invalid skill index at {output_dir}: {exc}")
            sys.exit(1)
        for kind, index in loaded.items():
            print(f"  {kind}: {len(index['documents'])} documents, valid=True")
        print("\nIndex is valid.")
        return

    print(f"\n{'═' * 60}")
    print("  Skill Index Builder")
    print(f"{'═' * 60}")
    print(f"  Source:  {skill_library_dir}")
    print(f"  Output:  {output_dir}")
    print(f"  Model:   {embedding['model']} ({embedding['backend']})")
    print(f"  Batch:   {batch_size}")
    if embedding["backend"] == "openai_http":
        print(f"  API base: {embedding['base_url']}")
    print(f"{'═' * 60}\n")

    indices = indexer.build_indices(skill_library_dir)
    indexer.save_indices(indices, output_dir)
    try:
        _validate_saved_indices(indexer, skill_library_dir, output_dir)
    except Exception as exc:
        print(f"[ERROR] Skill index failed post-write validation: {exc}")
        sys.exit(1)

    def _count(kind: str) -> int:
        return int(indices.get(kind, {}).get("metadata", {}).get("total_documents", 0))

    print(f"\n{'═' * 60}")
    print("  Index built")
    print(f"{'═' * 60}")
    print(f"  Planning documents:   {_count('planning')}")
    print(f"  Skill documents:      {_count('skills')}")
    print(f"  Tool-skill documents: {_count('tool_skills')}")
    print(f"  Output:               {output_dir}")
    print(f"{'═' * 60}")


if __name__ == "__main__":
    main()
