#!/usr/bin/env python3
"""Version-chain orchestration for the Steward method.

Per version, in order:

1. **Train phase**
   - v1 (bootstrap): cold-start rollout with ``--bootstrap-strategy``, no skill
     library, then bootstrap the first library
   - v2+: rollout with the library of v(i-1) using ``--train-strategy``, then
     evolve the library
   - after each hop: build the library's retrieval index
2. **Test phase**: test rollout with the matching library, then evaluate

Sub-stages run as ``python -m dtau_bench <subcommand>`` subprocesses. Stage
progress is persisted in ``experiments/{name}/run_state.json``, so re-running
with ``--resume`` skips completed stages.

Usage:
    python -m dtau_bench chain \\
      --experiment-name my_run \\
      --train-versions v1 v2 v3 v4 v5 v6 \\
      --test-versions v1 v2 v3 v4 v5 v6 \\
      --bootstrap-strategy single_domain_full \\
      --train-strategy train_steward \\
      --test-strategy steward \\
      --agent-model my-model
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

from dtau_bench.strategy.registry import uses_skill_library
from dtau_bench.utils.config import (
    get_project_root,
    extract_model_request_options,
    get_data_root,
    load_defaults_config,
    load_models_config,
    load_strategy_config,
)

logger = logging.getLogger("dtau_bench.cli.chain")


class ArtifactValidationError(RuntimeError):
    """A stage returned successfully but did not produce usable artifacts."""


def _positive_int(value: str) -> int:
    """Argparse type for worker and batch settings."""
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _configuration_fingerprint(metadata: Dict[str, Any]) -> str:
    """Return a stable fingerprint for the configuration stored in run state."""
    encoded = json.dumps(
        metadata,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _experiment_runtime_signature(
    *,
    agent_model: str,
    user_model: str,
    evolve_model: str,
    bootstrap_strategy: str,
    train_strategy: str,
    test_strategy: str,
    rollout_domain_workers: int = 1,
    rollout_query_workers: int = 4,
    evaluate_workers: int = 2,
    skill_batch_size: int = 2,
    skill_max_concurrent: int = 2,
    merge_embedding_batch_size: int = 8,
    embedding_batch_size: int = 8,
) -> Dict[str, Any]:
    """Describe result-affecting config without persisting credentials.

    Model keys alone are insufficient for safe resume: the endpoint model name,
    reasoning effort, token limits, or strategy YAML may change under the same
    key. Request options are represented by a digest because they may contain
    custom headers; API keys are deliberately excluded altogether.
    """
    models_config = load_models_config()
    models = models_config.get("models") or {}

    def model_signature(model_key: str) -> Dict[str, Any]:
        entry = models.get(model_key)
        if not isinstance(entry, dict):
            raise KeyError(
                f"Model key {model_key!r} is not in configs/models.yaml. "
                f"Available: {sorted(models)}"
            )
        endpoints = entry.get("base_urls")
        if not isinstance(endpoints, list):
            endpoints = [entry.get("base_url", "")]
        request_options = extract_model_request_options(entry)
        return {
            "key": model_key,
            "model_name": entry.get("model_name", model_key),
            "base_urls": endpoints,
            "max_tokens": entry.get("max_tokens"),
            "reasoning_effort": request_options.get("reasoning_effort"),
            "request_options_fingerprint": _configuration_fingerprint(request_options),
        }

    strategy_signatures: Dict[str, str] = {}
    for strategy_name in {
        bootstrap_strategy,
        train_strategy,
        test_strategy,
    }:
        strategy_signatures[strategy_name] = _configuration_fingerprint(
            load_strategy_config(strategy_name)
        )

    embedding = (models_config.get("embedding") or {}).get("default") or {}
    return {
        "models": {
            "agent": model_signature(agent_model),
            "user": model_signature(user_model),
            "evolve": model_signature(evolve_model),
        },
        "embedding": {
            key: embedding.get(key)
            for key in ("backend", "model", "base_url")
        },
        "resources": {
            "rollout_domain_workers": rollout_domain_workers,
            "rollout_query_workers": rollout_query_workers,
            "evaluate_workers": evaluate_workers,
            "skill_batch_size": skill_batch_size,
            "skill_max_concurrent": skill_max_concurrent,
            "merge_embedding_batch_size": merge_embedding_batch_size,
            "embedding_batch_size": embedding_batch_size,
        },
        "defaults_fingerprint": _configuration_fingerprint(load_defaults_config()),
        "strategy_fingerprints": strategy_signatures,
    }


def _metadata_mismatches(
    existing: Dict[str, Any], requested: Dict[str, Any]
) -> List[str]:
    """Describe configuration fields that differ across a resume boundary."""
    mismatches: List[str] = []
    for key in sorted(set(existing) | set(requested)):
        if existing.get(key) != requested.get(key):
            mismatches.append(
                f"{key}: state={existing.get(key)!r}, requested={requested.get(key)!r}"
            )
    return mismatches


# ─────────────────────────────────────────────────────────────────────────────
# Run State Management
# ─────────────────────────────────────────────────────────────────────────────


def _now_iso() -> str:
    """UTC ISO timestamp."""
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    """Atomically write JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


# A dry run must not leave a trace on disk. Persisting stage state while skipping
# the actual work marks every stage "completed" with no artifacts behind it, and
# the next real --resume then skips the whole train chain and dies on a missing
# skill library. So all state/manifest persistence funnels through the two
# helpers below, which become no-ops under --dry-run.
_DRY_RUN = False


def _set_dry_run(enabled: bool) -> None:
    global _DRY_RUN
    _DRY_RUN = bool(enabled)


def _write_state(path: Path, data: Dict[str, Any]) -> None:
    """Persist run state unless this is a dry run."""
    if _DRY_RUN:
        return
    _write_json(path, data)


def _write_manifest(path: Path, data: Dict[str, Any]) -> None:
    """Persist the experiment manifest unless this is a dry run."""
    if _DRY_RUN:
        print(f"  [DRY-RUN] manifest not written: {path}")
        return
    _write_json(path, data)


def _load_json(path: Path) -> Dict[str, Any]:
    """Load JSON from file."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _init_run_state(
    state_path: Path,
    *,
    resume: bool,
    metadata: Dict[str, Any],
    allow_resource_change_on_resume: bool = False,
) -> Dict[str, Any]:
    """Initialize or load run state."""
    if allow_resource_change_on_resume and not resume:
        raise ValueError(
            "--allow-resource-change-on-resume requires --resume"
        )

    requested_fingerprint = _configuration_fingerprint(metadata)
    if state_path.exists():
        if not resume:
            raise FileExistsError(
                f"Run state already exists at {state_path}. "
                f"Pass --resume to continue, or remove the directory."
            )
        state = _load_json(state_path)
        if not isinstance(state, dict):
            raise ValueError(f"Run state must be a JSON object: {state_path}")

        existing_metadata = state.get("metadata")
        if not isinstance(existing_metadata, dict):
            raise ValueError(
                f"Cannot resume {state_path}: state metadata is missing or invalid."
            )

        existing_fingerprint = _configuration_fingerprint(existing_metadata)
        recorded_fingerprint = state.get("config_fingerprint")
        if recorded_fingerprint is not None and recorded_fingerprint != existing_fingerprint:
            raise ValueError(
                f"Cannot resume {state_path}: config fingerprint does not match "
                "the state's stored metadata. The state may have been edited or corrupted."
            )

        mismatches = _metadata_mismatches(existing_metadata, metadata)
        if existing_fingerprint != requested_fingerprint or mismatches:
            existing_without_resources = copy.deepcopy(existing_metadata)
            requested_without_resources = copy.deepcopy(metadata)
            existing_runtime = existing_without_resources.get("runtime_signature")
            requested_runtime = requested_without_resources.get("runtime_signature")
            existing_resources = None
            requested_resources = None
            if isinstance(existing_runtime, dict) and isinstance(requested_runtime, dict):
                existing_resources = existing_runtime.pop("resources", None)
                requested_resources = requested_runtime.pop("resources", None)

            resource_only_change = (
                recorded_fingerprint is not None
                and isinstance(existing_resources, dict)
                and isinstance(requested_resources, dict)
                and existing_resources != requested_resources
                and existing_without_resources == requested_without_resources
            )
            if not (allow_resource_change_on_resume and resource_only_change):
                details = "; ".join(mismatches) or "configuration fingerprint changed"
                raise ValueError(
                    f"Cannot resume {state_path} with a different configuration: {details}"
                )

            changed_at = _now_iso()
            state.setdefault("resource_change_history", []).append({
                "changed_at": changed_at,
                "previous": existing_resources,
                "requested": requested_resources,
            })
            state["metadata"] = metadata
            state["config_fingerprint"] = requested_fingerprint
            state["updated_at"] = changed_at
            _write_state(state_path, state)
            logger.warning(
                "Resuming %s with an explicit resource-only configuration change: %s -> %s",
                state_path,
                existing_resources,
                requested_resources,
            )

        # Legacy states predate the explicit fingerprint. Exact metadata equality
        # above makes it safe to upgrade them in place.
        if recorded_fingerprint is None:
            state["config_fingerprint"] = requested_fingerprint
            state["schema_version"] = max(int(state.get("schema_version", 1)), 2)
            state["updated_at"] = _now_iso()
            _write_state(state_path, state)
        logger.info(f"Resuming from existing run state: {state_path}")
    else:
        state = {
            "schema_version": 2,
            "run_type": "experiment",
            "status": "running",
            "started_at": _now_iso(),
            "updated_at": _now_iso(),
            "current_stage": None,
            "stages": {},
            "metadata": metadata,
            "config_fingerprint": requested_fingerprint,
        }
        _write_state(state_path, state)
    return state


def _should_run_stage(state: Dict[str, Any], stage_name: str) -> bool:
    """Check if a stage needs to be executed."""
    stages = state.get("stages", {})
    stage = stages.get(stage_name, {})
    return stage.get("status") != "completed"


def _stage_needs_run(
    state_path: Path,
    state: Dict[str, Any],
    stage_name: str,
    validator: Callable[[], None],
) -> bool:
    """Return whether a stage must run, revalidating completed stage output.

    A completed marker is only a resume hint. If its artifacts were removed,
    truncated, or replaced, mark the stage invalid and run it again.
    """
    if _should_run_stage(state, stage_name):
        return True
    try:
        validator()
    except (ArtifactValidationError, FileNotFoundError) as exc:
        logger.warning(
            "Completed stage %s has invalid artifacts and will be rerun: %s",
            stage_name,
            exc,
        )
        stage = state.setdefault("stages", {}).setdefault(stage_name, {})
        stage.update({
            "status": "invalid",
            "validation_error": str(exc),
            "updated_at": _now_iso(),
        })
        state["status"] = "running"
        state["current_stage"] = stage_name
        state["updated_at"] = _now_iso()
        _write_state(state_path, state)
        return True
    return False


def _mark_stage_running(
    state_path: Path,
    state: Dict[str, Any],
    stage_name: str,
    *,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Mark a stage as currently running."""
    stages = state.setdefault("stages", {})
    stage = stages.setdefault(stage_name, {})
    attempts = int(stage.get("attempts", 0)) + 1
    stage.update({
        "status": "running",
        "attempts": attempts,
        "started_at": _now_iso(),
        "updated_at": _now_iso(),
    })
    if metadata:
        stage["metadata"] = metadata
    state["status"] = "running"
    state["current_stage"] = stage_name
    state["updated_at"] = _now_iso()
    _write_state(state_path, state)


def _mark_stage_completed(
    state_path: Path,
    state: Dict[str, Any],
    stage_name: str,
    *,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Mark a stage as completed."""
    stage = state.setdefault("stages", {}).setdefault(stage_name, {})
    stage.update({
        "status": "completed",
        "completed_at": _now_iso(),
        "updated_at": _now_iso(),
    })
    if metadata:
        stage["metadata"] = metadata
    state["current_stage"] = None
    state["updated_at"] = _now_iso()
    _write_state(state_path, state)


def _mark_stage_failed(
    state_path: Path,
    state: Dict[str, Any],
    stage_name: str,
    *,
    error: str,
) -> None:
    """Mark a stage as failed."""
    stage = state.setdefault("stages", {}).setdefault(stage_name, {})
    stage.update({
        "status": "failed",
        "last_error": error,
        "failed_at": _now_iso(),
        "updated_at": _now_iso(),
    })
    state["status"] = "failed"
    state["current_stage"] = stage_name
    state["updated_at"] = _now_iso()
    _write_state(state_path, state)


def _mark_run_completed(state_path: Path, state: Dict[str, Any]) -> None:
    """Mark the entire run as completed."""
    state["status"] = "completed"
    state["current_stage"] = None
    state["completed_at"] = _now_iso()
    state["updated_at"] = _now_iso()
    _write_state(state_path, state)


# ─────────────────────────────────────────────────────────────────────────────
# Subprocess Execution
# ─────────────────────────────────────────────────────────────────────────────


def _run_cli(
    subcommand: str,
    args: List[str],
    *,
    dry_run: bool = False,
) -> None:
    """Run another dtau_bench subcommand as a subprocess.

    Args:
        subcommand: Subcommand name (e.g., "rollout").
        args: CLI arguments to pass.
        dry_run: If True, print the command without executing.
    """
    cmd = [sys.executable, "-m", "dtau_bench", subcommand] + args

    logger.info(f"Running: {' '.join(cmd[:6])}...")
    if dry_run:
        print(f"[DRY-RUN] Would execute: {' '.join(cmd)}")
        return

    # cwd is inherited: every path is resolved against the project root.
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, cmd)


# ─────────────────────────────────────────────────────────────────────────────
# Artifact Acceptance
# ─────────────────────────────────────────────────────────────────────────────


def _artifact_error(message: str) -> ArtifactValidationError:
    return ArtifactValidationError(message)


@lru_cache(maxsize=None)
def _load_task_roster(data_dir: Path, split: str) -> Dict[str, str]:
    """Load the authoritative query-id to domain mapping for one split."""
    task_root = data_dir / "tasks" / split
    task_files = sorted(task_root.glob("*/*.json")) if task_root.is_dir() else []
    if not task_files:
        raise _artifact_error(f"No {split} task roster found under {task_root}")

    roster: Dict[str, str] = {}
    for task_path in task_files:
        try:
            data = _load_json(task_path)
        except (OSError, json.JSONDecodeError) as exc:
            raise _artifact_error(f"Invalid task definition {task_path}: {exc}") from exc
        query_id = str(data.get("query_id") or task_path.stem).strip()
        domain = str(data.get("domain") or task_path.parent.name).strip()
        if not query_id or not domain:
            raise _artifact_error(f"Task definition lacks query_id/domain: {task_path}")
        if query_id in roster:
            raise _artifact_error(f"Duplicate task query_id {query_id!r} in {task_root}")
        roster[query_id] = domain
    return roster


def _load_jsonl_objects(path: Path, *, label: str) -> List[Dict[str, Any]]:
    if not path.is_file() or path.stat().st_size == 0:
        raise _artifact_error(f"Missing or empty {label}: {path}")
    rows: List[Dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("row is not a JSON object")
                rows.append(value)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise _artifact_error(
            f"Invalid {label} at {path} (line {line_number if 'line_number' in locals() else '?'}): {exc}"
        ) from exc
    if not rows:
        raise _artifact_error(f"No records in {label}: {path}")
    return rows


def _validate_rollout_artifacts(
    run_dir: Path,
    *,
    data_dir: Path,
    split: str,
    version: str,
    run_name: str,
) -> None:
    """Accept a rollout only when every task has one valid versioned record."""
    roster = _load_task_roster(data_dir, split)
    rollout_dir = run_dir / "rollouts"
    files = sorted(rollout_dir.glob("*.jsonl")) if rollout_dir.is_dir() else []
    if not files:
        raise _artifact_error(f"No rollout JSONL files found in {rollout_dir}")

    seen: Dict[str, str] = {}
    for path in files:
        for row in _load_jsonl_objects(path, label="rollout output"):
            task = row.get("task") if isinstance(row.get("task"), dict) else {}
            query_id = str(row.get("id") or task.get("query_id") or "").strip()
            if not query_id:
                raise _artifact_error(f"Rollout row has no query id: {path}")
            if query_id in seen:
                raise _artifact_error(
                    f"Duplicate rollout result for {query_id!r}: {seen[query_id]} and {path}"
                )
            run = row.get("run") if isinstance(row.get("run"), dict) else {}
            if run.get("version") != version:
                raise _artifact_error(
                    f"Rollout {query_id!r} has version {run.get('version')!r}; "
                    f"expected {version!r}"
                )
            row_domain = str(
                task.get("domain_key")
                or task.get("domain")
                or (task.get("domains") or [path.stem])[0]
            )
            expected_domain = roster.get(query_id)
            if expected_domain is not None and row_domain != expected_domain:
                raise _artifact_error(
                    f"Rollout {query_id!r} belongs to {row_domain!r}; "
                    f"expected {expected_domain!r}"
                )
            seen[query_id] = str(path)

    expected_ids = set(roster)
    actual_ids = set(seen)
    missing = sorted(expected_ids - actual_ids)
    unexpected = sorted(actual_ids - expected_ids)
    if missing or unexpected:
        raise _artifact_error(
            f"Incomplete {split} rollout for {run_name}/{version}: "
            f"missing={missing[:10]}{'...' if len(missing) > 10 else ''}, "
            f"unexpected={unexpected[:10]}{'...' if len(unexpected) > 10 else ''}"
        )

    summary_path = run_dir / "run_summary.json"
    try:
        summary = _load_json(summary_path)
    except (OSError, json.JSONDecodeError) as exc:
        raise _artifact_error(f"Missing or invalid rollout summary {summary_path}: {exc}") from exc
    if not isinstance(summary, dict):
        raise _artifact_error(f"Rollout summary must be a JSON object: {summary_path}")
    expected_summary = {
        "run_name": run_name,
        "split": split,
        "version": version,
        "total_tasks": len(roster),
    }
    mismatches = [
        f"{key}={summary.get(key)!r} (expected {value!r})"
        for key, value in expected_summary.items()
        if summary.get(key) != value
    ]
    if summary.get("failed") != 0:
        mismatches.append(f"failed={summary.get('failed')!r} (expected 0)")
    if mismatches:
        raise _artifact_error(
            f"Rollout summary does not describe a complete run: {', '.join(mismatches)}"
        )


def _validate_evaluation_artifacts(
    eval_dir: Path,
    *,
    data_dir: Path,
    split: str,
    version: str,
    run_name: str,
) -> None:
    """Accept an evaluation only when it covers the complete task roster."""
    roster = _load_task_roster(data_dir, split)
    results_path = eval_dir / "evaluation_results.jsonl"
    results = _load_jsonl_objects(results_path, label="evaluation results")
    seen: Set[str] = set()
    for row in results:
        sample_id = str(row.get("sample_id") or "").strip()
        if not sample_id:
            raise _artifact_error(f"Evaluation row has no sample_id: {results_path}")
        if sample_id in seen:
            raise _artifact_error(f"Duplicate evaluation result for {sample_id!r}")
        if row.get("version") != version:
            raise _artifact_error(
                f"Evaluation {sample_id!r} has version {row.get('version')!r}; "
                f"expected {version!r}"
            )
        if not isinstance(row.get("passed"), bool):
            raise _artifact_error(
                f"Evaluation {sample_id!r} has non-boolean passed="
                f"{row.get('passed')!r}"
            )
        row_domains = row.get("domains")
        row_domain = str(
            row.get("domain_key")
            or (
                "_".join(str(item) for item in row_domains if item)
                if isinstance(row_domains, list)
                else ""
            )
        ).strip()
        if row_domain != roster.get(sample_id):
            raise _artifact_error(
                f"Evaluation {sample_id!r} belongs to {row_domain!r}; "
                f"expected {roster.get(sample_id)!r}"
            )
        seen.add(sample_id)

    expected_ids = set(roster)
    missing = sorted(expected_ids - seen)
    unexpected = sorted(seen - expected_ids)
    if missing or unexpected:
        raise _artifact_error(
            f"Incomplete {split} evaluation for {run_name}/{version}: "
            f"missing={missing[:10]}{'...' if len(missing) > 10 else ''}, "
            f"unexpected={unexpected[:10]}{'...' if len(unexpected) > 10 else ''}"
        )

    report_path = eval_dir / "evaluation_report.json"
    try:
        report = _load_json(report_path)
    except (OSError, json.JSONDecodeError) as exc:
        raise _artifact_error(f"Missing or invalid evaluation report {report_path}: {exc}") from exc
    metadata = report.get("metadata") if isinstance(report, dict) else None
    summary = report.get("summary") if isinstance(report, dict) else None
    if not isinstance(metadata, dict) or not isinstance(summary, dict):
        raise _artifact_error(f"Invalid evaluation report structure: {report_path}")
    expected_fields = {
        "version": version,
        "run_name": run_name,
        "total_samples": len(roster),
    }
    mismatches = [
        f"{key}={metadata.get(key)!r} (expected {value!r})"
        for key, value in expected_fields.items()
        if metadata.get(key) != value
    ]
    if summary.get("total") != len(roster):
        mismatches.append(
            f"summary.total={summary.get('total')!r} (expected {len(roster)!r})"
        )
    expected_errored = sum(1 for row in results if row.get("error"))
    expected_passed = sum(
        1 for row in results if not row.get("error") and row["passed"]
    )
    expected_failed = sum(
        1 for row in results if not row.get("error") and not row["passed"]
    )
    for key, expected in (
        ("passed", expected_passed),
        ("failed", expected_failed),
        ("errored", expected_errored),
    ):
        actual = summary.get(key)
        if type(actual) is not int or actual != expected:
            mismatches.append(
                f"summary.{key}={actual!r} (expected {expected!r})"
            )
    if mismatches:
        raise _artifact_error(
            f"Evaluation report does not describe the requested run: {', '.join(mismatches)}"
        )


def _validate_skill_library(library_dir: Path, *, version: str) -> None:
    """Load and sanity-check the final skill library used by Steward."""
    required = (
        library_dir / "plan_library.json",
        library_dir / "library_metadata.json",
        library_dir / "skill_graph.json",
    )
    missing = [str(path) for path in required if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise _artifact_error(f"Skill library is missing required artifacts: {missing}")
    try:
        from dtau_bench.skill.loader import SkillLibraryLoader

        artifacts = SkillLibraryLoader.load(library_dir, mode="normal")
    except Exception as exc:
        raise _artifact_error(f"Cannot load skill library {library_dir}: {exc}") from exc
    if not artifacts.plans:
        raise _artifact_error(f"Skill library contains no plans: {library_dir}")
    if not artifacts.skills:
        raise _artifact_error(f"Skill library contains no active skills: {library_dir}")
    if not artifacts.skill_graph:
        raise _artifact_error(f"Skill library has an empty skill graph: {library_dir}")
    metadata = artifacts.library_metadata or {}
    recorded_version = (
        metadata.get("tool_schema_version")
        or metadata.get("target_tool_schema_version")
        or metadata.get("library_version")
    )
    if recorded_version != version:
        raise _artifact_error(
            f"Skill library {library_dir} targets {recorded_version!r}; expected {version!r}"
        )


def _validate_embedding_index_dir(
    index_dir: Path,
    *,
    kind: str,
    expected_metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Validate the three persisted files and their row/count metadata."""
    required = [
        index_dir / "embeddings.npy",
        index_dir / "documents.json",
        index_dir / "metadata.json",
    ]
    missing = [str(path) for path in required if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise _artifact_error(f"{kind} index is missing required artifacts: {missing}")
    try:
        import numpy as np

        embeddings = np.load(required[0], allow_pickle=False, mmap_mode="r")
        documents = _load_json(required[1])
        metadata = _load_json(required[2])
    except Exception as exc:
        raise _artifact_error(f"Cannot load {kind} index at {index_dir}: {exc}") from exc
    if embeddings.ndim != 2 or embeddings.shape[0] <= 0 or embeddings.shape[1] <= 0:
        raise _artifact_error(f"{kind} embeddings have invalid shape {embeddings.shape}")
    if not isinstance(documents, list) or len(documents) != embeddings.shape[0]:
        raise _artifact_error(
            f"{kind} index document/embedding count mismatch: "
            f"documents={len(documents) if isinstance(documents, list) else 'invalid'}, "
            f"embeddings={embeddings.shape[0]}"
        )
    if not isinstance(metadata, dict):
        raise _artifact_error(f"{kind} index metadata must be a JSON object")
    if metadata.get("kind") != kind:
        raise _artifact_error(
            f"{kind} index metadata kind is {metadata.get('kind')!r}"
        )
    if metadata.get("total_documents") != len(documents):
        raise _artifact_error(
            f"{kind} index metadata count is {metadata.get('total_documents')!r}; "
            f"expected {len(documents)}"
        )
    if metadata.get("embedding_dim") != embeddings.shape[1]:
        raise _artifact_error(
            f"{kind} index metadata dimension is {metadata.get('embedding_dim')!r}; "
            f"expected {embeddings.shape[1]}"
        )
    for key, expected in (expected_metadata or {}).items():
        actual = metadata.get(key)
        if key == "embedding_batch_size" and key not in metadata:
            actual = 64  # Historical hard-coded index-builder batch size.
        if actual != expected:
            raise _artifact_error(
                f"{kind} index metadata {key}={actual!r}; "
                f"expected {expected!r}"
            )


def _validate_skill_index_artifacts(
    library_dir: Path,
    *,
    embedding_batch_size: Optional[int] = None,
) -> None:
    """Validate every required persisted sub-index for a skill library."""
    try:
        from dtau_bench.cli.build_skill_index import QUERY_INSTRUCTION
        from dtau_bench.rag.config import RAGConfig
        from dtau_bench.skill.loader import SkillLibraryLoader
        from dtau_bench.utils.config import resolve_embedding_config

        artifacts = SkillLibraryLoader.load(library_dir, mode="normal")
    except Exception as exc:
        raise _artifact_error(f"Cannot inspect skill library {library_dir}: {exc}") from exc
    kinds = ["planning", "skills"]
    if artifacts.tool_skills:
        kinds.append("tool_skills")
    embedding = resolve_embedding_config()
    rag_config = RAGConfig(
        embedding_model=embedding["model"],
        embedding_backend=embedding["backend"],
        embedding_api_base=embedding["base_url"],
        embedding_api_key=embedding["api_key"],
        query_instruction=QUERY_INSTRUCTION,
    )
    expected_metadata = {
        "embedding_model": rag_config.embedding_model,
        "embedding_backend": rag_config.embedding_backend,
        "query_instruction": rag_config.query_instruction,
        "query_format_version": rag_config.query_format_version,
        "embedding_batch_size": (
            embedding["batch_size"]
            if embedding_batch_size is None
            else embedding_batch_size
        ),
        "source_dir": str(library_dir.resolve()),
    }
    index_root = library_dir / "index_cache"
    for kind in kinds:
        _validate_embedding_index_dir(
            index_root / kind,
            kind=kind,
            expected_metadata=expected_metadata,
        )


def _accept_artifacts(validator: Callable[[], None], *, dry_run: bool) -> None:
    if not dry_run:
        validator()


# ─────────────────────────────────────────────────────────────────────────────
# Library Resolution
# ─────────────────────────────────────────────────────────────────────────────


def _resolve_evolved_library(
    evolve_run_name: str,
    experiments_dir: Path,
) -> Path:
    """Resolve path to a skill library output (from bootstrap or evolve).

    Handles two output layouts:
      - EvolvePipeline:    experiments/{run_name}/skills/  (library files at top level)
      - BootstrapPipeline: experiments/{run_name}/skills/library/  (library in subdirectory)

    Returns the most specific existing path (prefers skills/library/ if it exists,
    otherwise falls back to skills/).
    """
    base_dir = experiments_dir / evolve_run_name / "skills"

    # BootstrapPipeline output: skills/library/ contains the final assembled library
    library_subdir = base_dir / "library"
    if library_subdir.exists():
        return library_subdir

    # EvolvePipeline output: skills/ itself is the library directory
    if base_dir.exists():
        return base_dir

    raise FileNotFoundError(
        f"Skill library not found. Checked:\n"
        f"  - {library_subdir}  (BootstrapPipeline output)\n"
        f"  - {base_dir}  (EvolvePipeline output)"
    )


def _init_manifest(
    manifest_path: Path,
    *,
    experiment_name: str,
    bootstrap_strategy: str,
    train_versions: List[str],
    test_versions: List[str],
    resume: bool,
) -> Dict[str, Any]:
    """Create a manifest or resume it without discarding train-phase history."""
    existing: Dict[str, Any] = {}
    if resume and manifest_path.is_file():
        loaded = _load_json(manifest_path)
        if not isinstance(loaded, dict):
            raise ValueError(f"Experiment manifest must be a JSON object: {manifest_path}")
        if loaded.get("experiment_name") not in (None, experiment_name):
            raise ValueError(
                f"Manifest {manifest_path} belongs to {loaded.get('experiment_name')!r}, "
                f"not {experiment_name!r}"
            )
        existing = loaded

    manifest: Dict[str, Any] = dict(existing)
    manifest.update({
        "experiment_name": experiment_name,
        "bootstrap_strategy": bootstrap_strategy,
        "train_chain": list(train_versions),
        "test_versions": list(test_versions),
        "status": "running",
    })
    manifest.setdefault("started_at", _now_iso())
    manifest.setdefault("hops", [])
    manifest.setdefault("tests", [])
    manifest.setdefault("library_map", {})
    return manifest


def _upsert_manifest_entry(
    entries: List[Dict[str, Any]], entry: Dict[str, Any], *, key: str
) -> None:
    """Replace a version entry in place, preserving deterministic ordering."""
    value = entry.get(key)
    for index, existing in enumerate(entries):
        if isinstance(existing, dict) and existing.get(key) == value:
            entries[index] = entry
            return
    entries.append(entry)


# ─────────────────────────────────────────────────────────────────────────────
# Build Stage Commands
# ─────────────────────────────────────────────────────────────────────────────


def _build_train_rollout_args(
    run_name: str,
    version: str,
    strategy: str,
    agent_model: str,
    user_model: str,
    resume: bool,
    library_dir: Optional[Path] = None,
    rollout_domain_workers: int = 1,
    rollout_query_workers: int = 4,
) -> List[str]:
    """Build CLI args for a train rollout stage.

    Args:
        library_dir: Path to skill library. None for bootstrap (cold-start) rollouts.
    """
    args = [
        "--run-name", run_name,
        "--strategy", strategy,
        "--split", "train",
        "--version", version,
        "--agent-model", agent_model,
        "--user-model", user_model,
        "--max-workers", str(rollout_domain_workers),
        "--query-workers", str(rollout_query_workers),
    ]
    if library_dir is not None:
        args.extend(["--skill-library", str(library_dir)])
    if resume:
        # the rollout subcommand doesn't have --resume, it auto-resumes by default
        pass
    return args


def _build_eval_args(
    run_name: str,
    version: str,
    eval_workers: int = 2,
) -> List[str]:
    """Build CLI args for evaluating one rollout run.

    Produces evaluation_results.jsonl, which carries the authoritative
    pass/fail signal for both skill extraction and reporting.

    Args:
        run_name: Rollout run name (experiments/{run_name}/rollouts/).
        version: Tool schema version being evaluated.
        eval_workers: Number of parallel evaluation workers.
    """
    args = [
        "--run-name", run_name,
        "--version", version,
        "--workers", str(eval_workers),
    ]
    return args


def _build_bootstrap_args(
    run_name: str,
    rollout_dir: Path,
    eval_file: Path,
    tool_schema_version: str,
    model: str,
    resume: bool,
    batch_size: int = 2,
    max_concurrent: int = 2,
    merge_embedding_batch_size: int = 8,
) -> List[str]:
    """Build CLI args for a bootstrap stage (first-hop cold-start).

    Uses the bootstrap subcommand which runs the 4-stage BootstrapPipeline:
    dataset → extraction → tool_skills → graph construction.

    Args:
        run_name: Experiment run name (determines output directory).
        rollout_dir: Path to rollout output directory.
        eval_file: Path to evaluation results JSONL (from the evaluate subcommand).
        tool_schema_version: Tool schema version (e.g., v1).
        model: LLM model key from models.yaml.
        resume: Whether to resume from last completed stage.
    """
    args = [
        "--run-name", run_name,
        "--rollout-dir", str(rollout_dir),
        "--eval-file", str(eval_file),
        "--tool-schema-version", tool_schema_version,
        "--model", model,
        "--batch-size", str(batch_size),
        "--max-concurrent", str(max_concurrent),
        "--merge-embedding-batch-size", str(merge_embedding_batch_size),
    ]
    if resume:
        args.append("--resume")
    return args


def _build_evolve_args(
    run_name: str,
    prev_library: Optional[Path],
    rollout_dir: Path,
    eval_results_file: Path,
    tool_schema_version: str,
    model: str,
    resume: bool,
    batch_size: int = 2,
    max_concurrent: int = 2,
    merge_embedding_batch_size: int = 8,
) -> List[str]:
    """Build CLI args for an evolve stage (v2+ incremental evolution).

    Uses the evolve subcommand which runs the 6-step EvolvePipeline:
    feedback_summary → proposal_extraction → tool_skill_finalize →
    atomic_skill_finalize → planning_propagation → rebuild_and_partition.

    We pass --rollout-dir (full trajectory data) + --eval-results-file
    (authoritative passed/failed signals from the evaluate subcommand). The pipeline
    will build a skill context-ready JSONL internally using both sources.

    Args:
        prev_library: Path to previous skill library (must not be None for evolve).
        rollout_dir: Path to rollout output directory.
        eval_results_file: Path to evaluation results JSONL (from the evaluate subcommand).
    """
    args = [
        "--run-name", run_name,
        "--rollout-dir", str(rollout_dir),
        "--eval-results-file", str(eval_results_file),
        "--tool-schema-version", tool_schema_version,
        "--model", model,
        "--batch-size", str(batch_size),
        "--max-concurrent", str(max_concurrent),
        "--merge-embedding-batch-size", str(merge_embedding_batch_size),
    ]
    if prev_library is not None:
        args.extend(["--prev-library", str(prev_library)])
    if resume:
        args.append("--resume")
    return args


def _build_test_rollout_args(
    run_name: str,
    version: str,
    strategy: str,
    library_dir: Path,
    agent_model: str,
    user_model: str,
    rag_index_dir: Optional[str] = None,
    rollout_domain_workers: int = 1,
    rollout_query_workers: int = 4,
) -> List[str]:
    """Build CLI args for a test rollout stage."""
    args = [
        "--run-name", run_name,
        "--strategy", strategy,
        "--split", "test",
        "--version", version,
        "--agent-model", agent_model,
        "--user-model", user_model,
        "--skill-library", str(library_dir),
        "--max-workers", str(rollout_domain_workers),
        "--query-workers", str(rollout_query_workers),
    ]
    if rag_index_dir:
        args.extend(["--rag-index-dir", rag_index_dir])
    return args


# ─────────────────────────────────────────────────────────────────────────────
# Main Orchestration Logic
# ─────────────────────────────────────────────────────────────────────────────


def run_experiment(
    experiment_name: str,
    train_versions: List[str],
    test_versions: List[str],
    bootstrap_strategy: str,
    train_strategy: str,
    test_strategy: str,
    agent_model: str,
    user_model: str,
    evolve_model: Optional[str],
    experiments_dir: Path,
    resume: bool,
    dry_run: bool,
    skip_train: bool = False,
    skip_test: bool = False,
    rollout_domain_workers: int = 1,
    rollout_query_workers: int = 4,
    evaluate_workers: int = 2,
    skill_batch_size: int = 2,
    skill_max_concurrent: int = 2,
    merge_embedding_batch_size: int = 8,
    embedding_batch_size: int = 8,
    allow_resource_change_on_resume: bool = False,
    data_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Execute the version chain: train phase, then test phase.

    Args:
        experiment_name: Experiment name (determines the output directories).
        train_versions: Ordered versions for iterative training.
        test_versions: Versions to test.
        bootstrap_strategy: Strategy for the first hop, which has no library yet.
        train_strategy: Strategy for later training hops.
        test_strategy: Strategy for test rollouts.
        agent_model: Model key for the agent.
        user_model: Model key for the simulated user.
        evolve_model: Model for skill evolution (defaults to agent_model).
        experiments_dir: Base experiments directory.
        resume: Skip stages already marked completed in the run state.
        dry_run: Print commands without executing.
        skip_train: Only run the test phase, reusing an earlier run's libraries.
        skip_test: Only run the train phase.
        rollout_domain_workers: Number of domains processed in parallel per rollout.
        rollout_query_workers: Number of queries processed in parallel per domain.
        evaluate_workers: Number of parallel evaluation workers.
        skill_batch_size: Batch size used by bootstrap and evolve LLM stages.
        skill_max_concurrent: Maximum concurrent bootstrap/evolve LLM batches.
        merge_embedding_batch_size: Embedding batch size used for skill merging.
        embedding_batch_size: Batch size used to build skill retrieval indexes.
        allow_resource_change_on_resume: Permit an explicitly requested change
            to runtime resource controls while preserving all model, strategy,
            data, and embedding configuration on resume.
        data_dir: Data root override; otherwise use DTAU_BENCH_DATA_ROOT or project data/.

    Returns:
        Experiment manifest dict.
    """
    if evolve_model is None:
        evolve_model = agent_model

    resource_settings = {
        "rollout_domain_workers": rollout_domain_workers,
        "rollout_query_workers": rollout_query_workers,
        "evaluate_workers": evaluate_workers,
        "skill_batch_size": skill_batch_size,
        "skill_max_concurrent": skill_max_concurrent,
        "merge_embedding_batch_size": merge_embedding_batch_size,
        "embedding_batch_size": embedding_batch_size,
    }
    invalid_resources = {
        name: value for name, value in resource_settings.items() if value < 1
    }
    if invalid_resources:
        raise ValueError(
            "Resource settings must be positive integers: "
            + ", ".join(
                f"{name}={value!r}" for name, value in invalid_resources.items()
            )
        )

    runtime_signature = _experiment_runtime_signature(
        agent_model=agent_model,
        user_model=user_model,
        evolve_model=evolve_model,
        bootstrap_strategy=bootstrap_strategy,
        train_strategy=train_strategy,
        test_strategy=test_strategy,
        rollout_domain_workers=rollout_domain_workers,
        rollout_query_workers=rollout_query_workers,
        evaluate_workers=evaluate_workers,
        skill_batch_size=skill_batch_size,
        skill_max_concurrent=skill_max_concurrent,
        merge_embedding_batch_size=merge_embedding_batch_size,
        embedding_batch_size=embedding_batch_size,
    )

    # ── Setup experiment directory ──
    # Must precede any state/manifest write.
    _set_dry_run(dry_run)

    experiment_dir = experiments_dir / experiment_name
    data_dir = get_data_root(data_dir)
    # A dry run also must not create the experiment directory.
    if not dry_run:
        experiment_dir.mkdir(parents=True, exist_ok=True)

    # ── Initialize run state ──
    state_path = experiment_dir / "run_state.json"
    state = _init_run_state(
        state_path,
        resume=resume,
        allow_resource_change_on_resume=allow_resource_change_on_resume,
        metadata={
            "experiment_name": experiment_name,
            "bootstrap_strategy": bootstrap_strategy,
            "train_versions": train_versions,
            "test_versions": test_versions,
            "train_strategy": train_strategy,
            "test_strategy": test_strategy,
            "agent_model": agent_model,
            "user_model": user_model,
            "evolve_model": evolve_model,
            "runtime_signature": runtime_signature,
            "data_dir": str(data_dir),
        },
    )

    # ── Manifest tracking ──
    # Test-only is a second invocation of the same experiment. Resume the
    # train manifest so its hops and library map are not replaced by empty lists.
    manifest_path = experiment_dir / "experiment_manifest.json"
    manifest = _init_manifest(
        manifest_path,
        experiment_name=experiment_name,
        bootstrap_strategy=bootstrap_strategy,
        train_versions=train_versions,
        test_versions=test_versions,
        resume=resume,
    )

    # ── Train Chain (multi-hop) ──
    current_library: Optional[Path] = None  # No library at the start
    # Map: version → library path (for test phase to look up)
    library_map: Dict[str, Path] = {}

    print(f"\n{'═' * 70}")
    print(f"  Experiment: {experiment_name}")
    print(f"  Bootstrap strategy: {bootstrap_strategy}")
    print(f"  Train strategy: {train_strategy}")
    print(f"  Train chain: {' → '.join(train_versions)}")
    print(f"  Test versions: {', '.join(test_versions)}")
    print(
        "  Resources: "
        f"rollout={rollout_domain_workers} domain x {rollout_query_workers} query, "
        f"evaluate={evaluate_workers}, "
        f"skill_batch={skill_batch_size}, "
        f"skill_concurrent={skill_max_concurrent}, "
        f"merge_embedding_batch={merge_embedding_batch_size}, "
        f"index_embedding_batch={embedding_batch_size}"
    )
    print(f"{'═' * 70}\n")

    if skip_train:
        # Rebuild the library map from the evolve outputs of a previous run. Run
        # names are deterministic, so no state beyond the directories is needed.
        print(f"{'═' * 70}")
        print("  Train Phase: SKIPPED (--skip-train)")
        print(f"{'═' * 70}")
        for version in train_versions:
            library = _resolve_evolved_library(
                f"{experiment_name}_evolve_to_{version}", experiments_dir
            )
            _accept_artifacts(
                lambda library=library, version=version: _validate_skill_library(
                    library, version=version
                ),
                dry_run=dry_run,
            )
            if uses_skill_library(test_strategy):
                _accept_artifacts(
                    lambda library=library: _validate_skill_index_artifacts(
                        library,
                        embedding_batch_size=embedding_batch_size,
                    ),
                    dry_run=dry_run,
                )
            library_map[version] = library
            print(f"    {version}: {library}")
        if not library_map:
            raise FileNotFoundError(
                f"No skill library found under {experiments_dir} for experiment "
                f"{experiment_name!r}. Run the train phase first."
            )
        print("")

    train_hops = [] if skip_train else list(enumerate(train_versions))
    for i, version in train_hops:
        is_bootstrap = (i == 0)

        # Determine strategy and library for this hop
        if is_bootstrap:
            hop_strategy = bootstrap_strategy
            hop_library = None  # No skill library for cold-start
        else:
            hop_strategy = train_strategy
            hop_library = current_library  # Use previous hop's evolved library

        # ── Stage A: Train rollout ──
        train_stage = f"train_rollout:{version}"
        train_run_name = f"{experiment_name}_train_{version}"
        train_run_dir = experiments_dir / train_run_name
        train_rollout_validator = lambda: _validate_rollout_artifacts(
            train_run_dir,
            data_dir=data_dir,
            split="train",
            version=version,
            run_name=train_run_name,
        )

        if _stage_needs_run(
            state_path, state, train_stage, train_rollout_validator
        ):
            print(f"\n{'─' * 50}")
            print(f"  [HOP {i+1}/{len(train_versions)}] Train rollout: version={version}")
            print(f"    Strategy: {hop_strategy}")
            if hop_library:
                print(f"    Library: {hop_library}")
            else:
                print(f"    Library: (none — cold-start bootstrap)")
            print(f"{'─' * 50}")

            _mark_stage_running(
                state_path, state, train_stage,
                metadata={
                    "version": version,
                    "library": str(hop_library) if hop_library else None,
                    "strategy": hop_strategy,
                    "run_name": train_run_name,
                },
            )

            try:
                rollout_args = _build_train_rollout_args(
                    run_name=train_run_name,
                    version=version,
                    strategy=hop_strategy,
                    agent_model=agent_model,
                    user_model=user_model,
                    resume=resume,
                    library_dir=hop_library,
                    rollout_domain_workers=rollout_domain_workers,
                    rollout_query_workers=rollout_query_workers,
                )
                rollout_args.extend(["--data-dir", str(data_dir)])
                _run_cli("rollout", rollout_args, dry_run=dry_run)
                _accept_artifacts(train_rollout_validator, dry_run=dry_run)

                _mark_stage_completed(
                    state_path, state, train_stage,
                    metadata={"version": version, "run_name": train_run_name},
                )
                print(f"  ✓ Train rollout for {version} completed")
            except Exception as exc:
                _mark_stage_failed(state_path, state, train_stage, error=str(exc))
                logger.error(f"Train rollout for {version} failed: {exc}")
                raise
        else:
            print(f"  [SKIP] Train rollout for {version} already completed")

        # ── Stage A.5: Evaluate train rollout ──
        # Produces evaluation_results.jsonl with authoritative passed/failed signals
        eval_stage = f"train_eval:{version}"
        train_eval_dir = experiments_dir / train_run_name / "evaluation"
        train_eval_file = train_eval_dir / "evaluation_results.jsonl"
        train_eval_validator = lambda: _validate_evaluation_artifacts(
            train_eval_dir,
            data_dir=data_dir,
            split="train",
            version=version,
            run_name=train_run_name,
        )

        if _stage_needs_run(state_path, state, eval_stage, train_eval_validator):
            print(f"\n{'─' * 50}")
            print(f"  [HOP {i+1}/{len(train_versions)}] Evaluate train rollout: {train_run_name}")
            print(f"    Output: {train_eval_file}")
            print(f"{'─' * 50}")

            _mark_stage_running(
                state_path, state, eval_stage,
                metadata={"version": version, "run_name": train_run_name},
            )

            try:
                eval_args = _build_eval_args(
                    run_name=train_run_name,
                    version=version,
                    eval_workers=evaluate_workers,
                )
                eval_args.extend(["--data-dir", str(data_dir)])
                _run_cli("evaluate", eval_args, dry_run=dry_run)
                _accept_artifacts(train_eval_validator, dry_run=dry_run)

                _mark_stage_completed(
                    state_path, state, eval_stage,
                    metadata={
                        "version": version,
                        "run_name": train_run_name,
                        "eval_file": str(train_eval_file),
                    },
                )
                print(f"  ✓ Train evaluation for {version} completed")
            except Exception as exc:
                _mark_stage_failed(state_path, state, eval_stage, error=str(exc))
                logger.error(f"Train evaluation for {version} failed: {exc}")
                raise
        else:
            print(f"  [SKIP] Train evaluation for {version} already completed")

        # ── Stage B: Evolve (or Bootstrap for first hop) ──
        evolve_stage = f"evolve:{version}"
        evolve_run_name = f"{experiment_name}_evolve_to_{version}"
        train_rollout_dir = experiments_dir / train_run_name / "rollouts"
        evolve_validator = lambda: _validate_skill_library(
            _resolve_evolved_library(evolve_run_name, experiments_dir),
            version=version,
        )

        # For the first hop, prev_library is None → use BootstrapPipeline
        prev_library_for_evolve = current_library

        if _stage_needs_run(state_path, state, evolve_stage, evolve_validator):
            print(f"\n{'─' * 50}")
            if is_bootstrap:
                print(f"  [HOP {i+1}/{len(train_versions)}] Bootstrap: cold-start → Skill@{version}")
                print(f"    Pipeline: BootstrapPipeline (the bootstrap subcommand)")
            else:
                print(f"  [HOP {i+1}/{len(train_versions)}] Evolve: {train_versions[i-1]} → {version}")
                print(f"    Pipeline: EvolvePipeline (the evolve subcommand)")
            print(f"    Eval file: {train_eval_file}")
            print(f"    Prev library: {prev_library_for_evolve or '(none — cold-start)'}")
            print(f"{'─' * 50}")

            _mark_stage_running(
                state_path, state, evolve_stage,
                metadata={
                    "version": version,
                    "mode": "bootstrap" if is_bootstrap else "evolve",
                    "prev_library": str(prev_library_for_evolve) if prev_library_for_evolve else None,
                    "eval_file": str(train_eval_file),
                    "run_name": evolve_run_name,
                },
            )

            try:
                if is_bootstrap:
                    # First hop: use BootstrapPipeline (correct for cold-start data
                    # which lacks retrieved_*_skills fields)
                    bootstrap_args = _build_bootstrap_args(
                        run_name=evolve_run_name,
                        rollout_dir=train_rollout_dir,
                        eval_file=train_eval_file,
                        tool_schema_version=version,
                        model=evolve_model,
                        resume=resume,
                        batch_size=skill_batch_size,
                        max_concurrent=skill_max_concurrent,
                        merge_embedding_batch_size=merge_embedding_batch_size,
                    )
                    bootstrap_args.extend(["--data-dir", str(data_dir)])
                    _run_cli("bootstrap", bootstrap_args, dry_run=dry_run)
                else:
                    # Subsequent hops: use EvolvePipeline (incremental evolution
                    # with feedback from retrieved skills)
                    evolve_args = _build_evolve_args(
                        run_name=evolve_run_name,
                        prev_library=prev_library_for_evolve,
                        rollout_dir=train_rollout_dir,
                        eval_results_file=train_eval_file,
                        tool_schema_version=version,
                        model=evolve_model,
                        resume=resume,
                        batch_size=skill_batch_size,
                        max_concurrent=skill_max_concurrent,
                        merge_embedding_batch_size=merge_embedding_batch_size,
                    )
                    evolve_args.extend(["--data-dir", str(data_dir)])
                    _run_cli("evolve", evolve_args, dry_run=dry_run)

                _accept_artifacts(evolve_validator, dry_run=dry_run)

                _mark_stage_completed(
                    state_path, state, evolve_stage,
                    metadata={
                        "version": version,
                        "mode": "bootstrap" if is_bootstrap else "evolve",
                        "run_name": evolve_run_name,
                    },
                )
                print(f"  ✓ {'Bootstrap' if is_bootstrap else 'Evolve'} to {version} completed")
            except Exception as exc:
                _mark_stage_failed(state_path, state, evolve_stage, error=str(exc))
                logger.error(f"{'Bootstrap' if is_bootstrap else 'Evolve'} to {version} failed: {exc}")
                raise
        else:
            print(f"  [SKIP] {'Bootstrap' if is_bootstrap else 'Evolve'} to {version} already completed")

        # Update current library for next hop
        if not dry_run:
            evolved_lib = _resolve_evolved_library(evolve_run_name, experiments_dir)
            current_library = evolved_lib
        else:
            # In dry-run mode, predict where the library would be created
            if is_bootstrap:
                # BootstrapPipeline outputs to: skills/library/
                current_library = experiments_dir / evolve_run_name / "skills" / "library"
            else:
                # EvolvePipeline outputs to: skills/
                current_library = experiments_dir / evolve_run_name / "skills"
        library_map[version] = current_library

        # ── Stage C: Build skill retrieval index ──
        # Needed for the next hop, and for the test strategy on the last hop.
        needs_index_for_next_hop = i < len(train_versions) - 1
        needs_index_for_test = (
            uses_skill_library(test_strategy) and i == len(train_versions) - 1
        )
        if needs_index_for_next_hop or needs_index_for_test:
            index_stage = f"build_index:{version}"
            index_validator = lambda: _validate_skill_index_artifacts(
                current_library,
                embedding_batch_size=embedding_batch_size,
            )
            if _stage_needs_run(state_path, state, index_stage, index_validator):
                print(f"  [INDEX] Building skill context index for Skill@{version}...")
                _mark_stage_running(
                    state_path, state, index_stage,
                    metadata={"version": version, "library": str(current_library)},
                )
                try:
                    _accept_artifacts(evolve_validator, dry_run=dry_run)
                    index_args = [
                        "--skill-library-dir", str(current_library),
                        "--output-dir", str(current_library / "index_cache"),
                        "--batch-size", str(embedding_batch_size),
                    ]
                    _run_cli("build-skill-index", index_args, dry_run=dry_run)
                    _accept_artifacts(index_validator, dry_run=dry_run)
                    _mark_stage_completed(
                        state_path, state, index_stage,
                        metadata={"version": version, "index_dir": str(current_library / "index_cache")},
                    )
                    print(f"  ✓ skill context index for {version} built")
                except Exception as exc:
                    _mark_stage_failed(state_path, state, index_stage, error=str(exc))
                    logger.error(f"Build skill context index for {version} failed: {exc}")
                    raise
            else:
                print(f"  [SKIP] skill context index for {version} already built")

        _upsert_manifest_entry(manifest["hops"], {
            "version": version,
            "train_run_name": train_run_name,
            "evolve_run_name": evolve_run_name,
            "library": str(current_library),
            "status": "completed",
        }, key="version")

    persisted_library_map = manifest.get("library_map", {})
    if not isinstance(persisted_library_map, dict):
        persisted_library_map = {}
    persisted_library_map.update({k: str(v) for k, v in library_map.items()})
    manifest["library_map"] = persisted_library_map

    # Library used when a test version has no matching train hop.
    first_library = library_map.get(train_versions[0]) if train_versions else None

    # ── Test Phase ──
    if skip_test:
        print(f"\n{'═' * 70}")
        print("  Test Phase: SKIPPED (--skip-test)")
        print(f"{'═' * 70}")
        print("  Train chain finished. Library map:")
        for version, library in library_map.items():
            print(f"    {version}: {library}")

        _mark_run_completed(state_path, state)
        manifest["status"] = "train_completed"
        manifest["test_skipped"] = True
        manifest["train_completed_at"] = _now_iso()
        _write_manifest(manifest_path, manifest)
        print(f"\n  Manifest: {manifest_path}\n")
        return manifest

    print(f"\n{'═' * 70}")
    print(f"  Test Phase: {len(test_versions)} versions")
    print(f"{'═' * 70}")

    for version in test_versions:
        test_stage = f"test_rollout:{version}"
        test_run_name = f"{experiment_name}_test_{version}"

        # Determine which library to use for this test version
        test_library = _resolve_test_library(
            version=version,
            library_map=library_map,
            first_library=first_library,
            train_versions=train_versions,
        )
        library_version = version if version in library_map else train_versions[0]
        _accept_artifacts(
            lambda: _validate_skill_library(test_library, version=library_version),
            dry_run=dry_run,
        )
        if uses_skill_library(test_strategy):
            _accept_artifacts(
                lambda: _validate_skill_index_artifacts(
                    test_library,
                    embedding_batch_size=embedding_batch_size,
                ),
                dry_run=dry_run,
            )
        test_run_dir = experiments_dir / test_run_name
        test_rollout_validator = lambda: _validate_rollout_artifacts(
            test_run_dir,
            data_dir=data_dir,
            split="test",
            version=version,
            run_name=test_run_name,
        )

        if _stage_needs_run(
            state_path, state, test_stage, test_rollout_validator
        ):
            print(f"\n  [TEST] version={version}, library={test_library}")

            _mark_stage_running(
                state_path, state, test_stage,
                metadata={
                    "version": version,
                    "library": str(test_library),
                    "run_name": test_run_name,
                },
            )

            try:
                # Resolve RAG index dir for this version (if exists)
                version_rag_index = experiments_dir.parent / "rag_index_cache" / version
                rag_index_for_test = str(version_rag_index) if version_rag_index.exists() else None

                test_args = _build_test_rollout_args(
                    run_name=test_run_name,
                    version=version,
                    strategy=test_strategy,
                    library_dir=test_library,
                    agent_model=agent_model,
                    user_model=user_model,
                    rag_index_dir=rag_index_for_test,
                    rollout_domain_workers=rollout_domain_workers,
                    rollout_query_workers=rollout_query_workers,
                )
                test_args.extend(["--data-dir", str(data_dir)])
                _run_cli("rollout", test_args, dry_run=dry_run)
                _accept_artifacts(test_rollout_validator, dry_run=dry_run)

                _mark_stage_completed(
                    state_path, state, test_stage,
                    metadata={"version": version, "run_name": test_run_name},
                )
                print(f"  ✓ Test rollout for {version} completed")
            except Exception as exc:
                _mark_stage_failed(state_path, state, test_stage, error=str(exc))
                logger.error(f"Test rollout for {version} failed: {exc}")
                raise
        else:
            print(f"  [SKIP] Test rollout for {version} already completed")

        # ── Evaluate the test rollout ──
        # Produces evaluation_results.jsonl, which the cross-version
        # aggregation reads. Without it there are no test metrics at all.
        test_eval_stage = f"test_eval:{version}"
        test_eval_file = (
            experiments_dir / test_run_name / "evaluation" / "evaluation_results.jsonl"
        )
        test_eval_validator = lambda: _validate_evaluation_artifacts(
            test_eval_file.parent,
            data_dir=data_dir,
            split="test",
            version=version,
            run_name=test_run_name,
        )

        if _stage_needs_run(
            state_path, state, test_eval_stage, test_eval_validator
        ):
            print(f"\n  [EVAL] {test_run_name} → {test_eval_file}")
            _mark_stage_running(
                state_path, state, test_eval_stage,
                metadata={"version": version, "run_name": test_run_name},
            )
            try:
                _run_cli(
                    "evaluate",
                    _build_eval_args(
                        run_name=test_run_name,
                        version=version,
                        eval_workers=evaluate_workers,
                    ) + ["--data-dir", str(data_dir)],
                    dry_run=dry_run,
                )
                _accept_artifacts(test_eval_validator, dry_run=dry_run)
                _mark_stage_completed(
                    state_path, state, test_eval_stage,
                    metadata={
                        "version": version,
                        "run_name": test_run_name,
                        "eval_file": str(test_eval_file),
                    },
                )
                print(f"  ✓ Test evaluation for {version} completed")
            except Exception as exc:
                _mark_stage_failed(state_path, state, test_eval_stage, error=str(exc))
                logger.error(f"Test evaluation for {version} failed: {exc}")
                raise
        else:
            print(f"  [SKIP] Test evaluation for {version} already completed")

        _upsert_manifest_entry(manifest["tests"], {
            "version": version,
            "test_run_name": test_run_name,
            "library": str(test_library),
            "status": "completed",
        }, key="version")

    # ── Finalize ──
    _mark_run_completed(state_path, state)
    manifest["status"] = "completed"
    manifest.pop("test_skipped", None)
    manifest["completed_at"] = _now_iso()

    # Write manifest
    _write_manifest(manifest_path, manifest)

    print(f"\n{'═' * 70}")
    if dry_run:
        print(f"  ✓ Dry run of '{experiment_name}' finished — nothing executed,")
        print(f"    nothing persisted. Stage state was NOT written, so a later")
        print(f"    --resume still starts from the beginning.")
    else:
        print(f"  ✓ Experiment '{experiment_name}' completed!")
        print(f"  Manifest: {manifest_path}")
    print(f"{'═' * 70}\n")

    return manifest


# ─────────────────────────────────────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────────────────────────────────────


def _resolve_test_library(
    version: str,
    library_map: Dict[str, Path],
    first_library: Optional[Path],
    train_versions: List[str],
) -> Path:
    """Determine which library to use for testing a given version.

    Strategy:
    - If version is in library_map (has been evolved to), use that library.
    - Otherwise, use the first evolved library (Skill@v1) as fallback.
    """
    # Direct match: use the library evolved for this version
    if version in library_map:
        return library_map[version]

    # Fallback: use the first library (typically Skill@v1)
    if first_library:
        return first_library

    raise FileNotFoundError(
        f"No library available for test version {version}. "
        f"library_map keys: {list(library_map.keys())}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments per plan §4.4."""
    parser = argparse.ArgumentParser(
        description="dtau_bench Multi-Hop Experiment Orchestration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── Core experiment parameters ──
    parser.add_argument(
        "--experiment-name", type=str, required=True,
        help="Experiment name (determines output directory: experiments/{name}).",
    )
    parser.add_argument(
        "--train-versions", nargs="+", type=str, required=True,
        help="Ordered list of training versions (e.g., v1 v2 v3 v4 v5 v6).",
    )
    parser.add_argument(
        "--test-versions", nargs="+", type=str, required=True,
        help="List of versions to run test rollouts on (e.g., v1 v2 v3 v4 v5 v6).",
    )

    # ── Strategy ──
    parser.add_argument(
        "--bootstrap-strategy", type=str, default="single_domain_full",
        help=(
            "Strategy for the first-hop cold-start rollout (no skill library needed). "
            "Default: single_domain_full."
        ),
    )
    parser.add_argument(
        "--train-strategy", type=str, default="train_steward",
        help="Strategy for v2+ train rollouts (requires skill library). Default: train_steward.",
    )
    parser.add_argument(
        "--test-strategy", type=str, default="steward",
        help="Strategy for test rollouts (default: steward).",
    )

    # ── Models ──
    parser.add_argument(
        "--agent-model", type=str, default="deepseek-v4-pro",
        help="Agent model key from models.yaml (default: deepseek-v4-pro).",
    )
    parser.add_argument(
        "--user-model", type=str, default="deepseek-flash",
        help="Simulated user model key (default: deepseek-flash).",
    )
    parser.add_argument(
        "--evolve-model", type=str, default=None,
        help="Model for skill evolution (default: same as agent-model).",
    )

    # ── Resource controls ──
    parser.add_argument(
        "--rollout-domain-workers", type=_positive_int, default=1,
        help="Parallel domains per rollout (default: 1).",
    )
    parser.add_argument(
        "--rollout-query-workers", type=_positive_int, default=4,
        help="Parallel queries per rollout domain (default: 4).",
    )
    parser.add_argument(
        "--evaluate-workers", type=_positive_int, default=2,
        help="Parallel workers for each evaluate stage (default: 2).",
    )
    parser.add_argument(
        "--skill-batch-size", type=_positive_int, default=2,
        help="Bootstrap/evolve LLM batch size (default: 2).",
    )
    parser.add_argument(
        "--skill-max-concurrent", type=_positive_int, default=2,
        help="Maximum concurrent bootstrap/evolve LLM batches (default: 2).",
    )
    parser.add_argument(
        "--merge-embedding-batch-size", type=_positive_int, default=8,
        help="Bootstrap/evolve merge embedding batch size (default: 8).",
    )
    parser.add_argument(
        "--embedding-batch-size", type=_positive_int, default=8,
        help="Skill retrieval index embedding batch size (default: 8).",
    )

    # ── Execution control ──
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from a previous run (skip completed stages).",
    )
    parser.add_argument(
        "--allow-resource-change-on-resume",
        action="store_true",
        help=(
            "With --resume, allow only runtime resource controls to change. "
            "All model, strategy, data, and embedding configuration must match."
        ),
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print commands without executing them.",
    )

    # ── Phase selection ──
    parser.add_argument(
        "--skip-train", action="store_true",
        help=(
            "Only run the test phase, reusing the skill libraries produced by an "
            "earlier train run of the same experiment name."
        ),
    )
    parser.add_argument(
        "--skip-test", action="store_true",
        help="Only run the train phase (rollout → eval → bootstrap/evolve → index).",
    )

    # ── Logging ──
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output.")
    parser.add_argument("--quiet", "-q", action="store_true", help="Quiet mode.")

    parser.add_argument(
        "--data-dir", type=Path, default=None,
        help="Data root (default: DTAU_BENCH_DATA_ROOT or {project_root}/data).",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    # ── Logging ──
    log_level = logging.DEBUG if args.verbose else (logging.WARNING if args.quiet else logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ── Resolve experiments directory ──
    experiments_dir = (get_project_root() / "experiments").resolve()
    if not args.dry_run:
        experiments_dir.mkdir(parents=True, exist_ok=True)

    # ── Run the experiment ──
    try:
        manifest = run_experiment(
            experiment_name=args.experiment_name,
            train_versions=args.train_versions,
            test_versions=args.test_versions,
            bootstrap_strategy=args.bootstrap_strategy,
            train_strategy=args.train_strategy,
            test_strategy=args.test_strategy,
            agent_model=args.agent_model,
            user_model=args.user_model,
            evolve_model=args.evolve_model,
            experiments_dir=experiments_dir,
            resume=args.resume,
            dry_run=args.dry_run,
            skip_train=args.skip_train,
            skip_test=args.skip_test,
            rollout_domain_workers=args.rollout_domain_workers,
            rollout_query_workers=args.rollout_query_workers,
            evaluate_workers=args.evaluate_workers,
            skill_batch_size=args.skill_batch_size,
            skill_max_concurrent=args.skill_max_concurrent,
            merge_embedding_batch_size=args.merge_embedding_batch_size,
            embedding_batch_size=args.embedding_batch_size,
            allow_resource_change_on_resume=args.allow_resource_change_on_resume,
            data_dir=args.data_dir,
        )
    except subprocess.CalledProcessError as exc:
        logger.error(f"Subprocess failed with return code {exc.returncode}")
        print(f"\n✗ Experiment failed. Check logs above for details.")
        print(f"  Use --resume to continue from where it stopped.")
        sys.exit(1)
    except FileNotFoundError as exc:
        logger.error(str(exc))
        print(f"\n✗ {exc}")
        sys.exit(1)
    except FileExistsError as exc:
        logger.error(str(exc))
        print(f"\n✗ {exc}")
        sys.exit(1)
    except (ArtifactValidationError, ValueError) as exc:
        logger.error(str(exc))
        print(f"\n✗ {exc}")
        sys.exit(1)
    except KeyboardInterrupt:
        print(f"\n\n✗ Interrupted by user. Use --resume to continue.")
        sys.exit(130)


if __name__ == "__main__":
    main()
