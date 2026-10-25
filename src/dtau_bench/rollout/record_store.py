"""Durable per-task rollout records and recoverable JSONL summaries.

Each completed task is first written to an independent, verified checkpoint.
The per-domain JSONL remains the public aggregate format, but can be rebuilt
from those checkpoints after an interrupted or corrupted append.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import threading
from typing import Any, Dict, Iterable, Optional, Set, Tuple


def _safe_component(value: str, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value in {".", ".."}
        or "\x00" in value
        or Path(value).name != value
    ):
        raise ValueError(f"Invalid {label}: {value!r}")
    return value


def _validate_record(
    record: Dict[str, Any],
    domain: str,
    query_id: str,
    expected_version: Optional[str] = None,
) -> None:
    """Reject records that do not belong to the requested task/run."""
    if not isinstance(record, dict) or record.get("id") != query_id:
        raise ValueError(f"Rollout record query ID mismatch: {query_id}")

    task = record.get("task")
    if not isinstance(task, dict):
        raise ValueError(f"Rollout record has no task object: {query_id}")
    task_query_id = task.get("query_id")
    if task_query_id != query_id:
        raise ValueError(
            f"Task query ID mismatch for {query_id}: {task_query_id!r}"
        )

    scalar_domains = []
    for field_name in ("domain", "domain_key"):
        value = task.get(field_name)
        if value not in (None, ""):
            if not isinstance(value, str):
                raise ValueError(
                    f"Invalid task {field_name} for {query_id}: {value!r}"
                )
            scalar_domains.append(value)
    if any(value != domain for value in scalar_domains):
        raise ValueError(
            f"Task domain mismatch for {query_id}: {scalar_domains!r}"
        )

    task_domains = task.get("domains")
    if task_domains is not None:
        if (
            not isinstance(task_domains, list)
            or not task_domains
            or not all(isinstance(value, str) and value for value in task_domains)
        ):
            raise ValueError(f"Invalid task domains for {query_id}: {task_domains!r}")
        if domain not in task_domains:
            raise ValueError(f"Task domains mismatch for {query_id}: {task_domains!r}")
    if not scalar_domains and task_domains is None:
        raise ValueError(f"Task domain missing for {query_id}")

    if expected_version is not None:
        run = record.get("run")
        actual_version = run.get("version") if isinstance(run, dict) else None
        if actual_version != expected_version:
            raise ValueError(
                f"Run version mismatch for {query_id}: "
                f"expected {expected_version!r}, got {actual_version!r}"
            )


def _serialize(record: Dict[str, Any]) -> bytes:
    payload = (json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8")
    parsed = json.loads(payload)
    if parsed != record:
        raise ValueError("Rollout record changed during JSON serialization")
    return payload


def _checkpoint_dir(output_dir: Path, domain: str) -> Path:
    domain = _safe_component(domain, "domain")
    return Path(output_dir) / "record_checkpoints" / domain


def _checkpoint_path(output_dir: Path, domain: str, query_id: str) -> Path:
    query_id = _safe_component(query_id, "query id")
    return _checkpoint_dir(output_dir, domain) / f"{query_id}.json"


def _retry_marker_path(output_dir: Path, domain: str, query_id: str) -> Path:
    query_id = _safe_component(query_id, "query id")
    return _checkpoint_dir(output_dir, domain) / f"{query_id}.retry"


def _rollout_path(output_dir: Path, domain: str) -> Path:
    domain = _safe_component(domain, "domain")
    return Path(output_dir) / "rollouts" / f"{domain}.jsonl"


def _fsync_directory(path: Path) -> None:
    """Persist directory-entry changes made by rename, create, or unlink."""
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(str(path), flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _mkdir_parents_durable(path: Path) -> None:
    missing = []
    cursor = Path(path)
    while not cursor.exists():
        missing.append(cursor)
        if cursor.parent == cursor:
            break
        cursor = cursor.parent
    path.mkdir(parents=True, exist_ok=True)
    for created in reversed(missing):
        _fsync_directory(created.parent)


def _unlink_durable(path: Path) -> bool:
    if not path.exists():
        return False
    path.unlink()
    _fsync_directory(path.parent)
    return True


def atomic_write_verified(path: Path, payload: bytes) -> None:
    """Atomically publish bytes after fsync and before/after readback checks."""
    path = Path(path)
    _mkdir_parents_durable(path.parent)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    try:
        with temporary.open("wb") as stream:
            written = stream.write(payload)
            if written != len(payload):
                raise OSError(
                    f"Short write for {path.name}: {written}/{len(payload)} bytes"
                )
            stream.flush()
            os.fsync(stream.fileno())
        if temporary.read_bytes() != payload:
            raise OSError(f"Temporary-file readback mismatch: {path.name}")
        temporary.replace(path)
        _fsync_directory(path.parent)
        if path.read_bytes() != payload:
            raise OSError(f"Published-file readback mismatch: {path.name}")
    finally:
        if temporary.exists():
            temporary.unlink()


def save_record(
    output_dir: Path,
    domain: str,
    query_id: str,
    record: Dict[str, Any],
    expected_version: Optional[str] = None,
) -> str:
    """Save one verified checkpoint and return its JSONL serialization."""
    domain = _safe_component(domain, "domain")
    query_id = _safe_component(query_id, "query id")
    _validate_record(record, domain, query_id, expected_version)
    payload = _serialize(record)
    atomic_write_verified(
        _checkpoint_path(output_dir, domain, query_id),
        payload,
    )
    # A successful replacement completes any pending retry for this task.
    _unlink_durable(_retry_marker_path(output_dir, domain, query_id))
    return payload.decode("utf-8")


def _load_retry_markers(
    output_dir: Path,
    domain: str,
    expected_ids: Optional[Set[str]] = None,
    expected_version: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """Load old checkpoints atomically renamed aside for a pending retry."""
    directory = _checkpoint_dir(output_dir, domain)
    if not directory.exists():
        return {}
    records: Dict[str, Dict[str, Any]] = {}
    for path in sorted(directory.glob("*.retry")):
        try:
            record = json.loads(path.read_bytes())
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError(f"Invalid retry marker: {path}") from exc
        query_id = record.get("id") if isinstance(record, dict) else None
        if (
            not isinstance(query_id, str)
            or path.name != f"{query_id}.retry"
            or query_id in records
        ):
            raise ValueError(f"Retry marker ID mismatch or duplicate: {path}")
        if expected_ids is not None and query_id not in expected_ids:
            raise ValueError(f"Unexpected retry marker ID for {domain}: {query_id}")
        _validate_record(record, domain, query_id, expected_version)
        records[query_id] = record
    return records


def load_domain_records(
    output_dir: Path,
    domain: str,
    expected_ids: Optional[Iterable[str]] = None,
    expected_version: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """Load and validate all independent checkpoints for one domain."""
    allowed = set(expected_ids) if expected_ids is not None else None
    records: Dict[str, Dict[str, Any]] = {}
    directory = _checkpoint_dir(output_dir, domain)
    if not directory.exists():
        return records
    for path in sorted(directory.glob("*.json")):
        record = json.loads(path.read_bytes())
        if not isinstance(record, dict):
            raise ValueError(f"Checkpoint is not an object: {path}")
        query_id = record.get("id")
        if (
            not isinstance(query_id, str)
            or path.stem != query_id
            or query_id in records
        ):
            raise ValueError(f"Checkpoint ID mismatch or duplicate: {path}")
        if allowed is not None and query_id not in allowed:
            raise ValueError(f"Unexpected checkpoint ID for {domain}: {query_id}")
        _validate_record(record, domain, query_id, expected_version)
        records[query_id] = record
    return records


def _read_jsonl_records(
    path: Path,
    domain: str,
    expected_ids: Optional[Set[str]],
    expected_version: Optional[str],
    authoritative: Optional[Dict[str, Dict[str, Any]]] = None,
    ignored_ids: Optional[Set[str]] = None,
) -> Tuple[Dict[str, Dict[str, Any]], list[int], list[str]]:
    records: Dict[str, Dict[str, Any]] = {}
    malformed: list[int] = []
    conflicts: Set[str] = set()
    authoritative = authoritative or {}
    ignored_ids = ignored_ids or set()
    if not path.exists():
        return records, malformed, []
    for line_number, line in enumerate(path.read_bytes().split(b"\n"), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            malformed.append(line_number)
            continue
        if not isinstance(record, dict) or not isinstance(record.get("id"), str):
            malformed.append(line_number)
            continue
        query_id = record["id"]
        if expected_ids is not None and query_id not in expected_ids:
            raise ValueError(f"Unexpected rollout ID at {path}:{line_number}: {query_id}")
        if query_id in ignored_ids:
            continue
        if query_id in authoritative:
            if record != authoritative[query_id]:
                conflicts.add(query_id)
            records.setdefault(query_id, record)
            continue
        if query_id in records:
            raise ValueError(f"Duplicate rollout ID at {path}:{line_number}: {query_id}")
        _validate_record(record, domain, query_id, expected_version)
        records[query_id] = record
    return records, malformed, sorted(conflicts)


def rebuild_domain_jsonl(
    output_dir: Path,
    domain: str,
    expected_ids: Optional[Iterable[str]] = None,
    expected_version: Optional[str] = None,
) -> Dict[str, Any]:
    """Rebuild one aggregate JSONL from its verified task checkpoints."""
    allowed = set(expected_ids) if expected_ids is not None else None
    records = load_domain_records(
        output_dir,
        domain,
        allowed,
        expected_version=expected_version,
    )
    output_path = _rollout_path(output_dir, domain)
    payload = b"".join(_serialize(records[qid]) for qid in sorted(records))
    if records or output_path.exists():
        atomic_write_verified(output_path, payload)
    return {
        "record_ids": set(records),
        "records": len(records),
        "output_path": output_path,
    }


def recover_domain_records(
    output_dir: Path,
    domain: str,
    expected_ids: Iterable[str],
    expected_version: Optional[str] = None,
) -> Dict[str, Any]:
    """Migrate valid legacy rows, prefer checkpoints, and repair the JSONL."""
    allowed = set(expected_ids)
    output_path = _rollout_path(output_dir, domain)
    retry_markers = _load_retry_markers(
        output_dir,
        domain,
        allowed,
        expected_version=expected_version,
    )
    checkpoints = load_domain_records(
        output_dir,
        domain,
        allowed,
        expected_version=expected_version,
    )
    # q.json appearing beside q.retry means a replacement was committed and
    # the process stopped before it could remove the old retry marker.
    completed_retries = set(retry_markers) & set(checkpoints)
    for query_id in completed_retries:
        _unlink_durable(_retry_marker_path(output_dir, domain, query_id))
    pending_retries = set(retry_markers) - completed_retries
    aggregate, malformed, conflicts = _read_jsonl_records(
        output_path,
        domain,
        allowed,
        expected_version,
        authoritative=checkpoints,
        ignored_ids=pending_retries,
    )
    for query_id, record in aggregate.items():
        if query_id not in checkpoints:
            save_record(
                output_dir,
                domain,
                query_id,
                record,
                expected_version=expected_version,
            )
    rebuilt = rebuild_domain_jsonl(
        output_dir,
        domain,
        allowed,
        expected_version=expected_version,
    )
    rebuilt.update(
        malformed_lines=malformed,
        restored_from_checkpoints=sorted(set(checkpoints) - set(aggregate)),
        checkpoint_conflicts=conflicts,
        pending_retry_ids=sorted(pending_retries),
        missing_ids=sorted(allowed - rebuilt["record_ids"]),
    )
    return rebuilt


def remove_record_checkpoints(
    output_dir: Path,
    domain: str,
    query_ids: Iterable[str],
    expected_ids: Optional[Iterable[str]] = None,
    expected_version: Optional[str] = None,
) -> int:
    """Durably mark records for retry, remove them, and rebuild the aggregate."""
    allowed = set(expected_ids) if expected_ids is not None else None
    retry_ids = set(query_ids)
    if allowed is not None and not retry_ids <= allowed:
        unexpected = sorted(retry_ids - allowed)
        raise ValueError(f"Unexpected retry IDs for {domain}: {unexpected}")

    existing_markers = _load_retry_markers(
        output_dir,
        domain,
        allowed,
        expected_version=expected_version,
    )
    checkpoints = load_domain_records(
        output_dir,
        domain,
        allowed,
        expected_version=expected_version,
    )

    # Atomically rename each old checkpoint into the retry marker. The two
    # names encode the state: only .retry is pending; a new .json is committed.
    removed = 0
    for query_id in retry_ids:
        query_id = _safe_component(query_id, "query id")
        checkpoint_path = _checkpoint_path(output_dir, domain, query_id)
        marker_path = _retry_marker_path(output_dir, domain, query_id)
        if query_id in checkpoints:
            checkpoint_path.replace(marker_path)
            _fsync_directory(checkpoint_path.parent)
            removed += 1
        elif query_id not in existing_markers:
            raise ValueError(f"No checkpoint to retry for {domain}: {query_id}")
    rebuild_domain_jsonl(
        output_dir,
        domain,
        expected_ids,
        expected_version=expected_version,
    )
    return removed


def clear_domain_records(output_dir: Path, domain: str) -> None:
    """Clear one domain for an explicitly requested non-resume run."""
    checkpoint_dir = _checkpoint_dir(output_dir, domain)
    if checkpoint_dir.exists():
        entries = list(checkpoint_dir.iterdir())
        for path in entries:
            if path.is_dir():
                raise ValueError(f"Unexpected checkpoint entry: {path}")
        for path in entries:
            path.unlink()
        checkpoint_dir.rmdir()
        _fsync_directory(checkpoint_dir.parent)
    output_path = _rollout_path(output_dir, domain)
    _unlink_durable(output_path)
