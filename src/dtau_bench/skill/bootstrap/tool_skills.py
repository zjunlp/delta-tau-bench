"""Stage 3: Tool Skill Extraction.

Pipeline:
  1. Load skill context-ready dataset
  2. Project rows into tool evidence records
  3. Load global tool schemas
  4. Aggregate evidence groups by tool (filter by min_distinct_tasks, min_error_events)
  5. LLM facet extraction (one facet per evidence group)
  6. Per-tool merge of facets into final tool skills

Delegates to skill_pipeline.tool_skills for core extraction logic.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

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
# Tool Schema Loading (adapted for dtau_bench directory structure)
# ─────────────────────────────────────────────────────────────────────────────


def _load_schema_from_file(path: Path) -> Dict[str, Dict[str, Any]]:
    """Load tool schemas from a single JSON file."""
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

    return schema_map


def _load_global_tool_schemas(
    tool_schemas_dir: Path,
    version: str,
    tool_attributes_dir: Path | None = None,
) -> Dict[str, Dict[str, Any]]:
    """Load all tool schemas across domains for a given version.

    dtau_bench uses: data/tools/{domain}/{version}_tools.json
    """
    global_map: Dict[str, Dict[str, Any]] = {}

    if not tool_schemas_dir.exists():
        raise FileNotFoundError(f"Tool schemas directory not found: {tool_schemas_dir}")

    attribute_loader = None
    if tool_attributes_dir is not None:
        from ..cost import ToolAttributeLoader

        attribute_loader = ToolAttributeLoader(tool_attributes_dir)

    for domain_dir in sorted(tool_schemas_dir.iterdir()):
        if not domain_dir.is_dir():
            continue

        # Try version-specific file first
        path = domain_dir / f"{version}_tools.json"
        if not path.exists():
            path = domain_dir / "base_tools.json"
        if not path.exists():
            continue

        schema_map = _load_schema_from_file(path)
        for tool_name, schema in schema_map.items():
            if tool_name not in global_map:
                schema_with_attributes = dict(schema)
                if attribute_loader is not None:
                    cost = attribute_loader.cost_for(domain_dir.name, version, tool_name)
                    if cost is not None:
                        schema_with_attributes["x_dtau_bench_cost"] = cost
                global_map[tool_name] = schema_with_attributes

    if not global_map:
        raise ValueError(f"No tool schemas found for version '{version}' in {tool_schemas_dir}")

    return global_map


# ─────────────────────────────────────────────────────────────────────────────
# Evidence Projection
# ─────────────────────────────────────────────────────────────────────────────


def _project_to_tool_evidence(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Project skill context-ready rows into tool-skill evidence records."""
    records: List[Dict[str, Any]] = []

    for row in rows:
        if not isinstance(row, dict):
            continue
        metadata = row.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}

        domain = row.get("domain") or metadata.get("domain")
        if not isinstance(domain, str) or not domain.strip():
            continue

        query_id = row.get("query_id") or row.get("task_id")
        if query_id is None:
            continue

        passed = row.get("passed", metadata.get("passed"))
        if not isinstance(passed, bool):
            continue

        tool_calls = row.get("tool_calls", [])
        tool_errors = row.get("tool_errors", [])
        if not isinstance(tool_calls, list) or not isinstance(tool_errors, list):
            continue

        records.append({
            "record_id": row.get("trajectory_id"),
            "query_id": str(query_id),
            "task_id": str(row.get("task_id", "")),
            "version": row.get("version"),
            "domain": domain.strip(),
            "user_task": str(row.get("user_task", "") or ""),
            "passed": passed,
            "reward": row.get("reward"),
            "total_score": row.get("total_score", metadata.get("total_score")),
            "tool_calls": tool_calls,
            "tool_errors": tool_errors,
            "selected_skills": row.get("selected_skills", []),
            "final_visible_tools": row.get("final_visible_tools", []),
            "metadata": {"skill_context_row_metadata": metadata},
        })

    return records


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────


async def _run_tool_skill_pipeline(
    rows: List[Dict[str, Any]],
    output_dir: Path,
    config: Any,  # BootstrapConfig
) -> Dict[str, Any]:
    """Async implementation of the tool skill extraction pipeline."""
    from ..core.tool_skills import (
        ToolSkillExtractor,
        aggregate_tool_skill_evidence,
        merge_tool_facets_per_tool,
        prepare_tool_skills_for_output,
    )
    from ..core.llm.client import LLM

    # Step 1: Project to evidence records
    records = _project_to_tool_evidence(rows)
    if not records:
        logger.warning("No tool-skill evidence records projected from input")
        return {"tool_skills_prepared": 0, "skipped": True}

    logger.info(f"[2/6] Projected {len(records)} tool evidence records")

    # Step 2: Load global tool schemas
    global_tool_schemas = _load_global_tool_schemas(
        config.tool_schemas_dir,
        config.tool_schema_version,
        config.tool_attributes_dir if config.cost_aware_skills else None,
    )
    logger.info(f"[3/6] Loaded {len(global_tool_schemas)} tool schemas")

    # Step 3: Aggregate evidence groups
    evidence_groups = aggregate_tool_skill_evidence(
        records,
        global_tool_schemas=global_tool_schemas,
        min_distinct_tasks=config.min_distinct_tasks,
        min_error_events=config.min_error_events,
        max_examples_per_group=config.max_examples_per_group,
    )
    if not evidence_groups:
        logger.warning("No tool-skill evidence groups met thresholds")
        return {"tool_skills_prepared": 0, "no_evidence_groups": True}

    logger.info(f"[4/6] Aggregated {len(evidence_groups)} evidence groups")

    # Step 4: LLM facet extraction
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

    extractor = ToolSkillExtractor(
        llm=llm,
        max_retries=config.max_retries,
        verbose=True,
        cost_aware=config.cost_aware_skills,
    )
    raw_skills = await extractor.extract_groups(
        evidence_groups,
        batch_size=config.batch_size,
        max_concurrent=config.max_concurrent,
        show_progress=True,
    )
    if not raw_skills:
        logger.warning("Tool skill extraction returned no raw skills")
        return {"tool_skills_prepared": 0, "no_raw_skills": True}

    logger.info(f"[5/6] Extracted {len(raw_skills)} raw tool-skill facets")

    # Step 5: Per-tool merge
    merge_candidates, merged_skills = await merge_tool_facets_per_tool(
        raw_facets=raw_skills,
        llm=llm,
        global_tool_schemas=global_tool_schemas,
        batch_size=config.batch_size,
        max_concurrent=config.max_concurrent,
        max_facets_per_tool=config.max_facets_per_tool,
        random_seed=0,
        quiet=False,
        show_progress=True,
        cost_aware=config.cost_aware_skills,
    )
    if not merged_skills:
        logger.warning("Tool facet merge returned no merged tool skills")
        return {"tool_skills_prepared": 0, "no_merged_skills": True}

    # Step 6: Prepare final output
    prepared_skills = prepare_tool_skills_for_output(merged_skills)
    logger.info(f"[6/6] Prepared {len(prepared_skills)} final tool skills")

    # Write artifacts
    _write_jsonl(output_dir / "tool_skill_groups.jsonl",
                 [group.to_record() for group in evidence_groups])
    _write_jsonl(output_dir / "tool_skill_facets.raw.jsonl", raw_skills)
    _write_jsonl(output_dir / "tool_skills.merged.jsonl", merged_skills)
    _write_jsonl(output_dir / "tool_skills.prepared.jsonl", prepared_skills)

    stats = {
        "input_rows": len(rows),
        "projected_evidence_records": len(records),
        "evidence_group_count": len(evidence_groups),
        "tool_skill_facets_raw": len(raw_skills),
        "tool_skills_merged": len(merged_skills),
        "tool_skills_prepared": len(prepared_skills),
        "tool_schema_version": config.tool_schema_version,
    }

    # Save token usage
    token_path = output_dir / "token_usage.json"
    llm.token_tracker.save(
        str(token_path),
        extra_info={"stage": "bootstrap_tool_skills", "model": config.model},
    )
    stats["token_usage"] = llm.token_tracker.get_summary()
    logger.info(f"Tool skill token usage: {llm.token_tracker}")

    _write_json(output_dir / "summary.json", stats)

    return stats


def run_tool_skill_stage(
    input_path: Path,
    output_dir: Path,
    config: Any,  # BootstrapConfig
) -> Dict[str, Any]:
    """Run the tool skill extraction stage (Stage 3).

    Synchronous wrapper around the async pipeline.

    Args:
        input_path: Path to skill_context_ready.jsonl (output from Stage 1).
        output_dir: Directory to write tool skill artifacts.
        config: BootstrapConfig instance.

    Returns:
        Statistics dict.
    """
    rows = _read_jsonl(input_path)
    if not rows:
        raise FileNotFoundError(f"No records loaded from {input_path}")

    logger.info(f"Stage 3: Loaded {len(rows)} records from {input_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    return asyncio.run(_run_tool_skill_pipeline(rows, output_dir, config))
