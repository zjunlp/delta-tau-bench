"""Stage 2: Planning & Atomic skill extraction.

Pipeline:
  1. Load skill context-ready dataset
  2. Filter by reward threshold (keep high-quality trajectories)
  3. Optional long tool-feedback summarization
  4. Plan extraction + plan combination per user task
  5. Atomic skill extraction (τ²-Bench style)
  6. Embedding clustering (complete-linkage) + LLM merge
  7. Two-stage filtering (general + tool-schema)
  8. Final prepared skill export

Delegates to the skill context library for core extraction logic.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)



def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Tool schema loading (from data/tools/{domain}/{version}_tools.json)
# ─────────────────────────────────────────────────────────────────────────────


def _load_plan_library(path: Path) -> Any:
    """Load a previously extracted plan library.

    Accepts both the on-disk wrapper written by this stage
    (``{"plan": <library>}``) and a bare library, so a library copied out of
    ``skills/library/`` works as-is.
    """
    if not path.is_file():
        raise FileNotFoundError(f"reuse_plan_library not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    library = payload.get("plan", payload) if isinstance(payload, dict) else payload
    if not library:
        raise ValueError(
            f"reuse_plan_library is empty: {path}. Reusing it would silently "
            "produce a library with no planning layer."
        )
    return library


def _load_domain_tool_schemas(
    tool_schemas_dir: Path,
    version: str,
    domain: str,
    cache: Dict[Tuple[str, str], Dict[str, Dict[str, Any]]],
) -> Dict[str, Dict[str, Any]]:
    """Load tool schemas for a specific domain and version."""
    cache_key = (version, domain)
    if cache_key in cache:
        return cache[cache_key]

    # dtau_bench uses: data/tools/{domain}/{version}_tools.json
    path = tool_schemas_dir / domain / f"{version}_tools.json"
    if not path.exists():
        # Fallback: try base_tools.json
        path = tool_schemas_dir / domain / "base_tools.json"
    if not path.exists():
        logger.warning(f"Tool schema not found for {domain}/{version}, returning empty")
        cache[cache_key] = {}
        return {}

    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, list):
        raise ValueError(f"Tool schema file must contain a list: {path}")

    schema_map: Dict[str, Dict[str, Any]] = {}
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        schema = entry.get("tool") if isinstance(entry.get("tool"), dict) else entry
        function = schema.get("function") if isinstance(schema.get("function"), dict) else None
        name = str(function.get("name", "") if function else "").strip()
        if not name:
            continue
        schema_map[name] = schema

    cache[cache_key] = schema_map
    return schema_map


# ─────────────────────────────────────────────────────────────────────────────
# Skill post-processing utilities
# ─────────────────────────────────────────────────────────────────────────────


def _collect_skills_from_results(extraction_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Collect individual skill items from batched extraction results."""
    all_skills: List[Dict[str, Any]] = []

    for result in extraction_results:
        if not result or "plan_step_metadata" not in result:
            continue

        source_metadata = result.get("metadata", {})
        source_task_id = result.get("task_id", "")
        source_user_task = result.get("user_task", "")

        for skill_items in result["plan_step_metadata"].values():
            for item in skill_items:
                if not isinstance(item, dict):
                    continue
                option = item.get("option")
                skill = item.get("skill")

                if option == "keep":
                    continue
                if option not in {"add", "modify"} or not isinstance(skill, dict):
                    continue

                enriched = dict(item)
                enriched["source_metadata"] = dict(source_metadata) if isinstance(source_metadata, dict) else {}
                enriched["source_task_id"] = source_task_id
                enriched["source_user_task"] = source_user_task
                all_skills.append(enriched)

    return all_skills


def _group_by_domain(items: List[Dict[str, Any]], metadata_key: str = "metadata") -> DefaultDict[str, List[Dict[str, Any]]]:
    """Group items by their domain field in metadata."""
    grouped: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in items:
        metadata = item.get(metadata_key, {})
        if not isinstance(metadata, dict):
            metadata = {}
        domain = str(metadata.get("domain", "")).strip()
        if not domain:
            domain = "unknown"
        grouped[domain].append(item)
    return grouped


def _normalize_merged_results(merge_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Flatten merge results into individual skill records."""
    normalized: List[Dict[str, Any]] = []
    for result in merge_results:
        if not isinstance(result, dict):
            continue

        if "merge_skills" not in result:
            normalized.append(result)
            continue

        payload = result["merge_skills"]
        merged_skills = payload if isinstance(payload, list) else [payload]
        source_items = result.get("source", [])

        # Extract source metadata
        domains = sorted({
            str(s.get("source_metadata", {}).get("domain", "")).strip()
            for s in source_items
            if isinstance(s.get("source_metadata"), dict)
        } - {""})
        source_metadata = source_items[0].get("source_metadata", {}) if source_items else {}

        for skill in merged_skills:
            if not isinstance(skill, dict):
                continue
            normalized.append({
                "option": "add",
                "skill": skill,
                "source_metadata": source_metadata,
                "source_count": len(source_items),
            })

    return normalized


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────


async def _run_extraction_pipeline(
    items: List[Dict[str, Any]],
    output_dir: Path,
    config: Any,  # BootstrapConfig
) -> Dict[str, Any]:
    """Async implementation of the extraction pipeline."""
    from ..core.data.loaders import TrajectoryLoader
    from ..core.llm.client import LLM
    from ..core.extraction.plan_extractor import PlanExtractor, PlanCombiner
    from ..core.extraction.tool_summary import ToolSummary
    from ..core.extraction.skill_extractor import (
        AtomicSkillExtractor,
        prepare_skills_for_clustering,
    )
    from ..core.clustering.hierarchical import CompleteLinkageClustering
    from ..core.clustering.embedding import EmbeddingService
    from ..core.clustering.merger import SkillMerger
    from ..core.filtering.pipeline import TwoStageFilterPipeline
    from ..cost import COST_AWARE_GUIDANCE

    prompt_suffix = COST_AWARE_GUIDANCE if config.cost_aware_skills else ""

    # Filter by reward threshold (keep high-quality trajectories)
    filtered = [
        item for item in items
        if float(item.get("reward", item.get("after_score", 0.0))) > config.filter_threshold
    ]
    if not filtered:
        raise ValueError(
            f"No items above filter threshold {config.filter_threshold}. "
            f"Total items: {len(items)}. Try a lower threshold."
        )

    logger.info(f"Filtered {len(filtered)} of {len(items)} items (threshold={config.filter_threshold})")

    # Initialize LLM
    llm = LLM(
        model=config.model,
        api_key=config.api_key,
        base_url=config.base_url,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        max_retries=config.max_retries,
        timeout=config.timeout,
        request_options=config.llm_request_options,
    )

    schema_cache: Dict[Tuple[str, str], Dict[str, Dict[str, Any]]] = {}

    # Optional: Summarize long tool feedback
    if config.max_feedback_length > 0:
        summarizer = ToolSummary(
            llm=llm,
            max_len=config.max_feedback_length,
            verbose=True,
        )
        filtered = await summarizer.summarize_multiple(
            filtered,
            batch_size=config.batch_size,
            max_concurrent=config.max_concurrent,
            show_progress=True,
        )

    # --- Steps 1-2: Plan extraction + combination (or reuse) ---
    reuse_plan_path = getattr(config, "reuse_plan_library", None)
    plan_library_reused = False
    if reuse_plan_path:
        plan_library = _load_plan_library(Path(reuse_plan_path))
        plan_library_reused = True
        logger.info(
            "Reusing %d plan(s) from %s; skipping plan extraction",
            len(plan_library),
            reuse_plan_path,
        )
    else:
        plan_extractor = PlanExtractor(
            llm=llm,
            benchmark="atomic",
            max_retries=config.max_retries,
            verbose=True,
            prompt_suffix=prompt_suffix,
        )
        grouped_plans = await plan_extractor.extract_and_group(
            filtered,
            filter_threshold=config.filter_threshold,
            batch_size=config.batch_size,
            max_concurrent=config.max_concurrent,
        )

        plan_combiner = PlanCombiner(
            llm=llm,
            benchmark="atomic",
            max_retries=config.max_retries,
            verbose=True,
            prompt_suffix=prompt_suffix,
        )
        plan_library = await plan_combiner.combine_grouped_plans(
            grouped_plans,
            concurrent=config.concurrent_plan_combine,
        )

    # --- Step 3: Atomic skill extraction ---
    atomic_pipeline_results = {
        "raw": [],
        "merge_input": [],
        "merged": [],
        "filtered": [],
        "prepared": [],
        "cluster_count": 0,
    }

    if config.run_atomic:
        atomic_items = [
            {
                "task_id": item.get("task_id", ""),
                "user_task": item.get("user_task", ""),
                "successful_trajectory": item.get("successful_trajectory") or item.get("trajectory", []),
                "exp_metadata": {"skills": []},
                "reward": item.get("reward", 0.0),
                "metadata": item.get("metadata", {}),
                "cost_summary": item.get("cost_summary") if config.cost_aware_skills else None,
                "observed_action_costs": (
                    [
                        {
                            "tool_name": call.get("tool_name"),
                            "arguments": call.get("arguments"),
                            "has_error": call.get("has_error"),
                            "cost": call.get("cost"),
                        }
                        for call in item.get("tool_calls", [])
                        if isinstance(call, dict) and "cost" in call
                    ]
                    if config.cost_aware_skills
                    else []
                ),
            }
            for item in filtered
        ]

        atomic_extractor = AtomicSkillExtractor(
            llm=llm,
            benchmark="atomic",
            existing_skills={},
            max_retries=config.max_retries,
            verbose=True,
            prompt_suffix=prompt_suffix,
        )
        atomic_results = await atomic_extractor.extract_batch(
            atomic_items,
            batch_size=config.batch_size,
            max_concurrent=config.max_concurrent,
            show_progress=True,
        )

        # Collect raw skills
        atomic_raw = _collect_skills_from_results(atomic_results)
        atomic_pipeline_results["raw"] = atomic_raw

        if atomic_raw:
            # Prepare for clustering
            merge_input = prepare_skills_for_clustering(atomic_raw)
            atomic_pipeline_results["merge_input"] = merge_input

            if merge_input:
                # Embedding clustering + merge (per domain)
                embedding_service = EmbeddingService(
                    model=config.embedding_model,
                    base_url=config.embedding_api_base,
                    batch_size=config.merge_embedding_batch_size,
                    api_key=config.embedding_api_key,
                )
                clustering = CompleteLinkageClustering(
                    distance_threshold=1 - config.merge_similarity_threshold,
                )

                grouped = _group_by_domain(merge_input, metadata_key="source_metadata")
                merged_results: List[Dict[str, Any]] = []
                cluster_count = 0

                for domain in sorted(grouped.keys()):
                    domain_skills = grouped[domain]
                    embeddings = await embedding_service.embed_batch(
                        [skill["embedding_text"] for skill in domain_skills],
                        show_progress=True,
                    )
                    clusters = clustering.cluster_skills(domain_skills, embeddings)
                    truncated_clusters = [cluster[:15] for cluster in clusters]
                    cluster_count += len(truncated_clusters)

                    merger = SkillMerger(
                        llm=llm,
                        benchmark="atomic",
                        skill_type="atomic",
                        tool_schemas=_load_domain_tool_schemas(
                            config.tool_schemas_dir,
                            config.tool_schema_version,
                            domain,
                            schema_cache,
                        ),
                        max_group_size=15,
                        verbose=True,
                        max_concurrent=config.max_concurrent,
                    )
                    domain_merged = await merger.merge_clusters(truncated_clusters)
                    merged_results.extend(_normalize_merged_results(domain_merged))

                atomic_pipeline_results["merged"] = merged_results
                atomic_pipeline_results["cluster_count"] = cluster_count

                # Two-stage filtering
                filtered_results: List[Dict[str, Any]] = []
                merged_grouped = _group_by_domain(merged_results, metadata_key="source_metadata")

                for domain in sorted(merged_grouped.keys()):
                    pipeline = TwoStageFilterPipeline(
                        llm=llm,
                        benchmark="atomic",
                        tool_schemas=_load_domain_tool_schemas(
                            config.tool_schemas_dir,
                            config.tool_schema_version,
                            domain,
                            schema_cache,
                        ),
                        skip_stage1=config.skip_general_filter,
                        verbose=True,
                    )
                    domain_filtered = await pipeline.filter(
                        merged_grouped[domain],
                        batch_size=config.batch_size,
                        max_concurrent=config.max_concurrent,
                        show_progress=True,
                    )
                    filtered_results.extend(domain_filtered)

                atomic_pipeline_results["filtered"] = filtered_results
                atomic_pipeline_results["prepared"] = prepare_skills_for_clustering(filtered_results)

    # Every extractor call degrades to a warning, so a stage whose LLM was
    # unreachable (auth, quota, endpoint) still reaches this point and writes
    # empty files, then reports success. Downstream that becomes a library with
    # zero skills, which silently turns the method into its own RAG baseline.
    # Fail loudly instead; the caller can resume once the cause is fixed.
    if config.run_atomic and not atomic_pipeline_results["raw"]:
        raise RuntimeError(
            f"Atomic skill extraction produced 0 skills from {len(filtered)} "
            "trajectories. Every extractor call failed (check LLM auth/quota/"
            "endpoint above). Refusing to write an empty skill library."
        )

    # --- Save all artifacts ---
    _write_json(output_dir / "plan_library.json", {"plan": plan_library})

    if config.run_atomic:
        _write_jsonl(output_dir / "atomic_skills.raw.jsonl", atomic_pipeline_results["raw"])
        _write_jsonl(output_dir / "atomic_skills.merge_input.jsonl", atomic_pipeline_results["merge_input"])
        _write_jsonl(output_dir / "atomic_skills.merged.jsonl", atomic_pipeline_results["merged"])
        _write_jsonl(output_dir / "atomic_skills.filtered.jsonl", atomic_pipeline_results["filtered"])
        _write_jsonl(output_dir / "atomic_skills.prepared.jsonl", atomic_pipeline_results["prepared"])

    # Summary
    stats = {
        "input_items": len(items),
        "filtered_items": len(filtered),
        "plans": len(plan_library),
        "plan_library_reused": plan_library_reused,
        "plan_library_source": str(reuse_plan_path) if plan_library_reused else None,
        "atomic_enabled": config.run_atomic,
        "atomic_skills_raw": len(atomic_pipeline_results["raw"]),
        "atomic_skills_clusters": atomic_pipeline_results["cluster_count"],
        "atomic_skills_merged": len(atomic_pipeline_results["merged"]),
        "atomic_skills_filtered": len(atomic_pipeline_results["filtered"]),
        "atomic_skills_prepared": len(atomic_pipeline_results["prepared"]),
        "tool_schema_version": config.tool_schema_version,
        "embedding_model": config.embedding_model,
        "output_dir": str(output_dir),
    }

    # Save token usage
    token_path = output_dir / "token_usage.json"
    llm.token_tracker.save(
        str(token_path),
        extra_info={"stage": "bootstrap_extraction", "model": config.model},
    )
    stats["token_usage"] = llm.token_tracker.get_summary()
    logger.info(f"Extraction token usage: {llm.token_tracker}")

    _write_json(output_dir / "summary.json", stats)

    return stats


def run_extraction_stage(
    input_path: Path,
    output_dir: Path,
    config: Any,  # BootstrapConfig
) -> Dict[str, Any]:
    """Run the extraction stage (Stage 2).

    Synchronous wrapper around the async pipeline.

    Args:
        input_path: Path to skill_context_ready.jsonl (output from Stage 1).
        output_dir: Directory to write extraction artifacts.
        config: BootstrapConfig instance.

    Returns:
        Statistics dict.
    """
    # Load dataset
    items = _read_jsonl(input_path)
    if not items:
        raise FileNotFoundError(f"No records loaded from {input_path}")

    logger.info(f"Stage 2: Loaded {len(items)} records from {input_path}")

    # Run async pipeline
    return asyncio.run(_run_extraction_pipeline(items, output_dir, config))
