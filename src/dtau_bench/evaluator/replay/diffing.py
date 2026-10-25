"""
Database state diffing for evaluation.

Provides:
- compute_state_diff: Compute what changed between initial and final states
- evaluate_gold_subset_relation: Check if gold changes are subset of agent changes

Key concepts:
- Entity IDs (e.g., athlete_id) are stable — they exist pre-loaded in database
- Relationship IDs (e.g., creation_id) are generated — differ across sessions
- Dynamic fields (timestamps, free text, generated IDs) are excluded from comparison
- Placeholder values are detected and excluded
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from dtau_bench.evaluator.replay.executor import DatabaseState

logger_name = __name__

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

UUID_PATTERN = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)

TIMESTAMP_FIELDS = {
    "created_at",
    "updated_at",
    "deleted_at",
    "timestamp",
    "createdon",
    "updatedon",
}


# ─────────────────────────────────────────────────────────────────────────────
# Diff Metadata
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class DiffMetadata:
    """
    Metadata for smart field filtering during database diff.

    Attributes:
        domain: Domain name.
        entity_id_names: Set of field names that are entity IDs (stable).
        free_text_fields: Set of field names that are free-form text (ignore in diff).
        ignored_fields: Set of field names explicitly marked ignore_in_eval.
    """

    domain: str
    entity_id_names: Set[str] = field(default_factory=set)
    free_text_fields: Set[str] = field(default_factory=set)
    ignored_fields: Set[str] = field(default_factory=set)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict form used by the diffing functions."""
        return {
            "domain": self.domain,
            "entity_id_names": self.entity_id_names,
            "free_text_fields": self.free_text_fields,
            "ignored_fields": self.ignored_fields,
        }


def _is_placeholder(value: Any) -> bool:
    """Check if a value looks like a placeholder."""
    if not isinstance(value, str):
        return False
    val_lower = value.lower().strip()
    placeholder_keywords = [
        "placeholder", "tbd", "todo", "n/a", "n-a", "unknown",
        "pending", "temp", "dummy", "sample", "example", "test-",
    ]
    if any(keyword in val_lower for keyword in placeholder_keywords):
        return True
    return (
        (value.startswith("<") and value.endswith(">"))
        or (value.startswith("[") and value.endswith("]"))
        or (value.startswith("{") and value.endswith("}"))
    )


def _is_uuid(value: Any) -> bool:
    """Check if a value is a UUID string."""
    return isinstance(value, str) and bool(UUID_PATTERN.match(value))


def _normalize_value(value: Any) -> Any:
    """Normalize a value for comparison (case-insensitive strings, rounded floats)."""
    if isinstance(value, float):
        return round(value, 2)
    if isinstance(value, str):
        return value.lower()
    if isinstance(value, dict):
        return {k: _normalize_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_normalize_value(v) for v in value]
    return value


def _is_dynamic_field(key: str, value: Any, diff_meta: Dict[str, Any]) -> bool:
    """
    Check if a field should be excluded from comparison as dynamic.

    Dynamic fields include:
    - Explicitly ignored fields
    - Free text fields (unless they're UUIDs)
    - Non-entity ID fields (generated relationship IDs)
    - Timestamp fields
    - Sequence number fields
    """
    key_lower = key.lower()
    if key in diff_meta["ignored_fields"]:
        return True
    if key in diff_meta["free_text_fields"] and isinstance(value, str) and not _is_uuid(value):
        return True
    if key_lower.endswith("_id") and key_lower not in diff_meta["entity_id_names"]:
        return True
    if key_lower in TIMESTAMP_FIELDS:
        return True
    if "number" in key_lower and isinstance(value, str) and value.isdigit():
        return True
    if any(part in key_lower for part in ["opening_date", "registration_date", "creation_date"]):
        return True
    return False


def normalize_record(
    record: Dict[str, Any],
    diff_meta: Dict[str, Any],
    extra_ignore: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    """
    Normalize a database record for comparison.

    Strips dynamic fields and normalizes remaining values.

    Args:
        record: Raw database record dict.
        diff_meta: Diff metadata dict (from DiffMetadata.to_dict()).
        extra_ignore: Additional field names to ignore.

    Returns:
        Normalized record with only comparable fields.
    """
    if not isinstance(record, dict):
        raise ValueError("record must be a dict")

    normalized: Dict[str, Any] = {}
    ignored = extra_ignore or set()
    for key, value in record.items():
        if key in ignored:
            continue
        if _is_dynamic_field(key, value, diff_meta):
            continue
        normalized[key] = _normalize_value(value)
    return normalized


def _collect_placeholder_fields(*record_lists: List[Dict[str, Any]]) -> Set[str]:
    """Find fields that contain placeholder values in any record."""
    placeholder_fields: Set[str] = set()
    for records in record_lists:
        for record in records:
            if not isinstance(record, dict):
                continue
            for key, value in record.items():
                if _is_placeholder(value):
                    placeholder_fields.add(key)
    return placeholder_fields


def _record_hash(
    record: Dict[str, Any],
    diff_meta: Dict[str, Any],
    extra_ignore: Optional[Set[str]] = None,
) -> str:
    """Compute MD5 hash of a normalized record for quick comparison."""
    normalized = normalize_record(record, diff_meta, extra_ignore)
    return hashlib.md5(
        json.dumps(normalized, sort_keys=True, default=str).encode()
    ).hexdigest()


def _build_hash_to_record_map(
    records: List[Dict[str, Any]],
    diff_meta: Dict[str, Any],
    extra_ignore: Optional[Set[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Build map from record hash -> normalized record."""
    result: Dict[str, Dict[str, Any]] = {}
    for record in records:
        record_h = _record_hash(record, diff_meta, extra_ignore)
        result[record_h] = normalize_record(record, diff_meta, extra_ignore)
    return result


def _find_closest_record(
    target_record: Dict[str, Any],
    candidate_records: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Find the most similar candidate record to the target (for diagnostic output)."""
    if not candidate_records:
        return None

    best_match = None
    min_diff_count = float("inf")
    best_diff_fields: Dict[str, Any] = {}

    for candidate in candidate_records:
        diff_fields: Dict[str, Any] = {}
        all_keys = set(target_record.keys()) | set(candidate.keys())
        for key in all_keys:
            target_value = target_record.get(key)
            candidate_value = candidate.get(key)
            if target_value != candidate_value:
                diff_fields[key] = {
                    "gold": target_value,
                    "agent": candidate_value,
                }
        if len(diff_fields) < min_diff_count:
            best_match = candidate
            min_diff_count = len(diff_fields)
            best_diff_fields = diff_fields

    return {
        "closest_record": best_match,
        "differing_fields": best_diff_fields,
        "diff_count": min_diff_count,
    }


def _get_table_records(
    state: Optional[DatabaseState], table: str
) -> List[Dict[str, Any]]:
    """Get records from a table in a database state (entities or relationships)."""
    if state is None:
        return []
    return state.entities.get(table, []) or state.relationships.get(table, [])


# ─────────────────────────────────────────────────────────────────────────────
# Core Diff Functions
# ─────────────────────────────────────────────────────────────────────────────


def compute_state_diff(
    initial_state: DatabaseState,
    final_state: DatabaseState,
    diff_meta: DiffMetadata,
) -> Dict[str, Any]:
    """
    Compute what records were added between initial and final states.

    Used to understand what a replay (gold or agent) actually changed.

    Args:
        initial_state: State before replay.
        final_state: State after replay.
        diff_meta: Metadata for field filtering.

    Returns:
        Dict with per-table added records and totals.
    """
    if initial_state is None or final_state is None:
        raise ValueError("initial_state and final_state are required")

    meta_dict = diff_meta.to_dict() if isinstance(diff_meta, DiffMetadata) else diff_meta

    all_tables = (
        set(initial_state.entities.keys()) | set(initial_state.relationships.keys())
        | set(final_state.entities.keys()) | set(final_state.relationships.keys())
    )

    tables: Dict[str, Any] = {}
    total_added = 0

    for table in all_tables:
        initial_records = _get_table_records(initial_state, table)
        final_records = _get_table_records(final_state, table)
        placeholder_fields = _collect_placeholder_fields(initial_records, final_records)

        initial_map = _build_hash_to_record_map(initial_records, meta_dict, placeholder_fields)
        final_map = _build_hash_to_record_map(final_records, meta_dict, placeholder_fields)

        initial_hashes = set(initial_map.keys())
        final_hashes = set(final_map.keys())
        added_hashes = final_hashes - initial_hashes
        total_added += len(added_hashes)

        tables[table] = {
            "added_count": len(added_hashes),
            "added_records": [final_map[h] for h in added_hashes],
            "placeholder_fields_ignored": sorted(placeholder_fields),
        }

    return {
        "domain": meta_dict["domain"],
        "total_added": total_added,
        "tables": tables,
    }


def evaluate_gold_subset_relation(
    initial_state: DatabaseState,
    gold_state: DatabaseState,
    agent_state: DatabaseState,
    diff_meta: DiffMetadata,
) -> Dict[str, Any]:
    """
    Check whether gold DB changes are a subset of agent DB changes.

    This is the core pass/fail logic:
    - Compute gold diff (what gold replay added vs initial)
    - Compute agent diff (what agent replay added vs initial)
    - Check if every gold-added record hash also exists in agent-added

    If all gold changes are present in agent changes → PASS.

    Args:
        initial_state: DB state before any replay (baseline).
        gold_state: DB state after gold replay.
        agent_state: DB state after agent replay.
        diff_meta: Metadata for field filtering.

    Returns:
        Dict with:
        - passed (bool): Whether all gold changes are present in agent
        - reason (str): Human-readable failure reason (empty if passed)
        - stats: Summary counts
        - missing_changes: Detailed info about missing records (for debugging)
    """
    meta_dict = diff_meta.to_dict() if isinstance(diff_meta, DiffMetadata) else diff_meta

    all_tables = (
        set(gold_state.entities.keys()) | set(gold_state.relationships.keys())
        | set(agent_state.entities.keys()) | set(agent_state.relationships.keys())
    )

    missing_changes: List[Dict[str, Any]] = []
    stats = {
        "gold_diff_total": 0,
        "agent_diff_total": 0,
        "tables_checked": 0,
    }

    for table in all_tables:
        initial_records = _get_table_records(initial_state, table)
        gold_records = _get_table_records(gold_state, table)
        agent_records = _get_table_records(agent_state, table)
        placeholder_fields = _collect_placeholder_fields(
            initial_records, gold_records, agent_records,
        )

        initial_map = _build_hash_to_record_map(initial_records, meta_dict, placeholder_fields)
        gold_map = _build_hash_to_record_map(gold_records, meta_dict, placeholder_fields)
        agent_map = _build_hash_to_record_map(agent_records, meta_dict, placeholder_fields)

        initial_hashes = set(initial_map.keys())
        gold_hashes = set(gold_map.keys())
        agent_hashes = set(agent_map.keys())

        gold_diff = gold_hashes - initial_hashes
        agent_diff = agent_hashes - initial_hashes

        stats["gold_diff_total"] += len(gold_diff)
        stats["agent_diff_total"] += len(agent_diff)
        stats["tables_checked"] += 1

        missing = gold_diff - agent_diff
        extra = agent_diff - gold_diff
        if not missing:
            continue

        missing_records = [gold_map[h] for h in missing]
        extra_records = [agent_map[h] for h in extra]

        field_differences = []
        for missing_record in missing_records:
            closest_match = _find_closest_record(missing_record, extra_records)
            if closest_match is None:
                field_differences.append({
                    "type": "missing_in_agent",
                    "gold_record": missing_record,
                    "closest_agent_record": None,
                    "differing_fields": {},
                    "diff_count": None,
                    "note": "No candidate records in agent diff to compare",
                })
            else:
                field_differences.append({
                    "type": "missing_in_agent",
                    "gold_record": missing_record,
                    "closest_agent_record": closest_match["closest_record"],
                    "differing_fields": closest_match["differing_fields"],
                    "diff_count": closest_match["diff_count"],
                })

        missing_changes.append({
            "table": table,
            "missing_count": len(missing),
            "extra_count": len(extra),
            "gold_diff_count": len(gold_diff),
            "agent_diff_count": len(agent_diff),
            "missing_records": missing_records,
            "extra_records": extra_records,
            "field_differences": field_differences,
            "placeholder_fields_ignored": sorted(placeholder_fields),
        })

    passed = len(missing_changes) == 0
    return {
        "passed": passed,
        "reason": "" if passed else f"{len(missing_changes)} table(s) missing expected changes",
        "stats": stats,
        "missing_changes": missing_changes,
    }
