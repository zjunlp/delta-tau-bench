"""Step 6: Rebuild Skill Graph + Active/Inactive Partition.

Merges all finalized skills with existing library, partitions by target
tool schema version, writes final library files, and rebuilds the graph.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

from .proposals import EvolutionArtifacts, _read_jsonl, _write_jsonl

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Skill Row Helpers (carry over from existing library)
# ─────────────────────────────────────────────────────────────────────────────


def _copy_existing_atomic_rows(artifacts: EvolutionArtifacts) -> List[Dict[str, Any]]:
    """Create carry-over rows for existing active atomic skills."""
    rows: List[Dict[str, Any]] = []
    for item in artifacts.atomic_skills.values():
        rows.append({
            "option": "add",
            "skill": item.get("raw_skill", item),
            "embedding_text": item.get("embedding_text", ""),
            "source_metadata": {"domain": "global", "carryover": True},
            "source_task_ids": [],
        })
    return rows


def _copy_existing_tool_rows(artifacts: EvolutionArtifacts) -> List[Dict[str, Any]]:
    """Create carry-over rows for existing active tool skills."""
    rows: List[Dict[str, Any]] = []
    for item in artifacts.tool_skills.values():
        rows.append({
            "option": "add",
            "skill": item.get("raw_skill", item),
            "embedding_text": item.get("embedding_text", ""),
            "source_metadata": {"tool_name": item.get("name", ""), "carryover": True},
            "source_task_ids": [],
        })
    return rows


def _copy_inactive_atomic_rows(artifacts: EvolutionArtifacts) -> List[Dict[str, Any]]:
    """Create carry-over rows for existing inactive atomic skills."""
    rows: List[Dict[str, Any]] = []
    for item in artifacts.inactive_atomic_skills:
        source_row = item.get("source_row")
        if isinstance(source_row, dict):
            rows.append(dict(source_row))
            continue
        rows.append({
            "option": "add",
            "skill": item.get("raw_skill", item),
            "embedding_text": item.get("embedding_text", ""),
            "source_metadata": {"domain": "global", "carryover_inactive": True},
            "source_task_ids": [],
        })
    return rows


def _copy_inactive_tool_rows(artifacts: EvolutionArtifacts) -> List[Dict[str, Any]]:
    """Create carry-over rows for existing inactive tool skills."""
    rows: List[Dict[str, Any]] = []
    for item in artifacts.inactive_tool_skills:
        source_row = item.get("source_row")
        if isinstance(source_row, dict):
            rows.append(dict(source_row))
            continue
        rows.append({
            "option": "add",
            "skill": item.get("raw_skill", item),
            "embedding_text": item.get("embedding_text", ""),
            "source_metadata": {"tool_name": item.get("name", ""), "carryover_inactive": True},
            "source_task_ids": [],
        })
    return rows


def _merge_by_skill_name(
    existing_rows: List[Dict[str, Any]],
    updated_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Merge skill rows by name — updated rows override existing ones."""
    by_name: Dict[str, Dict[str, Any]] = {}
    for row in existing_rows + updated_rows:
        skill = row.get("skill", {})
        if not isinstance(skill, dict):
            continue
        name = str(skill.get("name", "")).strip()
        if not name:
            continue
        by_name[name] = row
    return [by_name[name] for name in sorted(by_name.keys())]


# ─────────────────────────────────────────────────────────────────────────────
# Active/Inactive Partition
# ─────────────────────────────────────────────────────────────────────────────


def _load_active_tool_names(tools_dir: Path, tool_schema_version: str) -> Set[str]:
    """Load all tool names from a specific version across all domains.

    Supports multiple tool JSON formats:
    - [{"node_id": ..., "tool": {"function": {"name": ...}}}]  (dtau_bench format)
    - [{"function": {"name": ...}}]  (OpenAI function calling format)
    - {"tool_name": {...}}  (dict keyed by name)
    """
    active_tools: Set[str] = set()

    if not tools_dir.exists():
        logger.warning(f"  Tools directory not found: {tools_dir}")
        return active_tools

    for domain_dir in tools_dir.iterdir():
        if not domain_dir.is_dir():
            continue
        version_file = domain_dir / f"{tool_schema_version}_tools.json"
        if not version_file.exists():
            continue
        with version_file.open("r", encoding="utf-8") as f:
            tools_data = json.load(f)
        if isinstance(tools_data, list):
            for tool in tools_data:
                if not isinstance(tool, dict):
                    continue
                # Format: {"node_id": ..., "tool": {"function": {"name": ...}}}
                inner_tool = tool.get("tool", {})
                if isinstance(inner_tool, dict):
                    func = inner_tool.get("function", {})
                    if isinstance(func, dict):
                        name = str(func.get("name", "")).strip()
                        if name:
                            active_tools.add(name)
                            continue
                # Format: {"function": {"name": ...}}
                func = tool.get("function", {})
                if isinstance(func, dict):
                    name = str(func.get("name", "")).strip()
                    if name:
                        active_tools.add(name)
                        continue
                # Format: {"name": ...}
                name = str(tool.get("name", "")).strip()
                if name:
                    active_tools.add(name)
        elif isinstance(tools_data, dict):
            for name in tools_data.keys():
                active_tools.add(str(name).strip())

    return active_tools


def partition_skill_rows(
    *,
    tool_rows: List[Dict[str, Any]],
    atomic_rows: List[Dict[str, Any]],
    tools_dir: Path,
    tool_schema_version: str,
) -> Tuple[
    List[Dict[str, Any]],  # active_tool_rows
    List[Dict[str, Any]],  # inactive_tool_rows
    List[Dict[str, Any]],  # active_atomic_rows
    List[Dict[str, Any]],  # inactive_atomic_rows
    Dict[str, Any],        # partition_metadata
]:
    """Partition skills into active/inactive based on tool availability in target version."""
    active_tool_names = _load_active_tool_names(tools_dir, tool_schema_version)

    active_tool_rows: List[Dict[str, Any]] = []
    inactive_tool_rows: List[Dict[str, Any]] = []
    active_atomic_rows: List[Dict[str, Any]] = []
    inactive_atomic_rows: List[Dict[str, Any]] = []

    inactive_tool_summary: List[Dict[str, Any]] = []
    inactive_atomic_summary: List[Dict[str, Any]] = []

    for row in tool_rows:
        skill = row.get("skill", {}) if isinstance(row.get("skill"), dict) else {}
        tool_name = str(skill.get("name", "")).strip()
        normalized_row = dict(row)
        source_metadata = (
            dict(normalized_row.get("source_metadata", {}))
            if isinstance(normalized_row.get("source_metadata"), dict)
            else {}
        )
        if tool_name and tool_name in active_tool_names:
            source_metadata.pop("inactive_reason", None)
            source_metadata["active_in_version"] = tool_schema_version
            normalized_row["source_metadata"] = source_metadata
            active_tool_rows.append(normalized_row)
        else:
            source_metadata["inactive_reason"] = "missing_tool_in_version"
            source_metadata["inactive_in_version"] = tool_schema_version
            normalized_row["source_metadata"] = source_metadata
            inactive_tool_rows.append(normalized_row)
            inactive_tool_summary.append({
                "name": tool_name,
                "reason": "missing_tool_in_version",
            })

    for row in atomic_rows:
        skill = row.get("skill", {}) if isinstance(row.get("skill"), dict) else {}
        atomic_name = str(skill.get("name", "")).strip()
        tools = [str(tool).strip() for tool in skill.get("tools", []) if str(tool).strip()]
        missing_tools = sorted(tool for tool in tools if tool not in active_tool_names)
        normalized_row = dict(row)
        source_metadata = (
            dict(normalized_row.get("source_metadata", {}))
            if isinstance(normalized_row.get("source_metadata"), dict)
            else {}
        )
        if not missing_tools:
            source_metadata.pop("inactive_reason", None)
            source_metadata["active_in_version"] = tool_schema_version
            normalized_row["source_metadata"] = source_metadata
            active_atomic_rows.append(normalized_row)
        else:
            source_metadata["inactive_reason"] = "atomic_depends_on_missing_tool"
            source_metadata["inactive_in_version"] = tool_schema_version
            source_metadata["missing_tools"] = missing_tools
            normalized_row["source_metadata"] = source_metadata
            inactive_atomic_rows.append(normalized_row)
            inactive_atomic_summary.append({
                "name": atomic_name,
                "reason": "atomic_depends_on_missing_tool",
                "missing_tools": missing_tools,
            })

    metadata = {
        "target_tool_schema_version": tool_schema_version,
        "inactive_summary": {
            "tool_count": len(inactive_tool_rows),
            "atomic_count": len(inactive_atomic_rows),
            "inactive_tool_skills": inactive_tool_summary,
            "inactive_atomic_skills": inactive_atomic_summary,
        },
    }
    return active_tool_rows, inactive_tool_rows, active_atomic_rows, inactive_atomic_rows, metadata


# ─────────────────────────────────────────────────────────────────────────────
# Prepare skills for output (add embedding_text)
# ─────────────────────────────────────────────────────────────────────────────


def _prepare_skills_for_output(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Ensure each row has an embedding_text field."""
    prepared: List[Dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        skill = item.get("skill", {})
        if isinstance(skill, dict) and "embedding_text" not in item:
            item["embedding_text"] = "\n".join(
                part
                for part in [
                    skill.get("name", ""),
                    skill.get("document", ""),
                    skill.get("content", ""),
                ]
                if part
            )
        prepared.append(item)
    return prepared


# ─────────────────────────────────────────────────────────────────────────────
# Step 6: Main Entry
# ─────────────────────────────────────────────────────────────────────────────


async def run_rebuild_and_partition(
    *,
    output_dir: Path,
    artifacts: EvolutionArtifacts,
    config: Any,
) -> None:
    """Step 6: Rebuild skill graph + active/inactive partition.

    1. Merge existing skills with finalized updates
    2. Partition into active/inactive based on target tool version
    3. Write final library files
    4. Rebuild skill graph
    """
    # Load finalized skill rows from previous steps
    finalized_tool_path = output_dir / "finalized_tool_rows.jsonl"
    finalized_atomic_modified_path = output_dir / "finalized_atomic_modified.jsonl"
    finalized_atomic_added_path = output_dir / "finalized_atomic_added.jsonl"

    finalized_tool_rows = _read_jsonl(finalized_tool_path) if finalized_tool_path.exists() else []
    finalized_atomic_modified = _read_jsonl(finalized_atomic_modified_path) if finalized_atomic_modified_path.exists() else []
    finalized_atomic_added = _read_jsonl(finalized_atomic_added_path) if finalized_atomic_added_path.exists() else []
    finalized_atomic_rows = finalized_atomic_modified + finalized_atomic_added

    # Merge with existing library
    tool_rows = _merge_by_skill_name(
        _copy_existing_tool_rows(artifacts) + _copy_inactive_tool_rows(artifacts),
        finalized_tool_rows,
    )
    atomic_rows = _merge_by_skill_name(
        _copy_existing_atomic_rows(artifacts) + _copy_inactive_atomic_rows(artifacts),
        finalized_atomic_rows,
    )

    # Partition
    (
        active_tool_rows,
        inactive_tool_rows,
        active_atomic_rows,
        inactive_atomic_rows,
        partition_metadata,
    ) = partition_skill_rows(
        tool_rows=tool_rows,
        atomic_rows=atomic_rows,
        tools_dir=config.tool_schemas_dir,
        tool_schema_version=config.tool_schema_version,
    )

    # Prepare for output (ensure embedding_text)
    prepared_tool_rows = _prepare_skills_for_output(active_tool_rows)
    prepared_atomic_rows = _prepare_skills_for_output(active_atomic_rows)

    # Write final library files
    _write_jsonl(output_dir / "tool_skills.prepared.jsonl", prepared_tool_rows)
    _write_jsonl(output_dir / "tool_skills.finalized.jsonl", tool_rows)
    _write_jsonl(output_dir / "tool_skills.inactive.jsonl", inactive_tool_rows)
    _write_jsonl(output_dir / "atomic_skills.prepared.jsonl", prepared_atomic_rows)
    _write_jsonl(output_dir / "atomic_skills.finalized.jsonl", atomic_rows)
    _write_jsonl(output_dir / "atomic_skills.inactive.jsonl", inactive_atomic_rows)

    # Write library_metadata.json
    metadata = {
        "library_version": config.tool_schema_version,
        "source_library_dir": str(config.prev_library),
        "target_tool_schema_version": config.tool_schema_version,
        "inherits_from": str(config.prev_library),
        **partition_metadata,
    }
    metadata_path = output_dir / "library_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    logger.info(
        f"  Partition: {len(active_tool_rows)} active tool + "
        f"{len(inactive_tool_rows)} inactive tool, "
        f"{len(active_atomic_rows)} active atomic + "
        f"{len(inactive_atomic_rows)} inactive atomic"
    )

    # Rebuild skill graph
    logger.info("  Rebuilding skill graph...")
    from ..bootstrap.graph import build_skill_graph

    build_skill_graph(
        skill_library_dir=output_dir,
        output_dir=output_dir,
        embedding_api_base=config.embedding_api_base,
        embedding_api_key=config.embedding_api_key,
        embedding_model=config.embedding_model,
        plan_atomic_coverage_threshold=config.plan_atomic_coverage_threshold,
        embedding_batch_size=config.merge_embedding_batch_size,
    )

    logger.info(f"  Skill graph rebuilt → {output_dir / 'skill_graph.json'}")
