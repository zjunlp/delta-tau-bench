"""
Data loader for dtau_bench evaluator.

Loads evaluation samples from rollout JSONL output files.
Supports schema_version 2 with both single-domain and all-tools formats.

The ``gold_trajectory`` carried by a rollout record is the **base** version
tool sequence. Version-translated sequences are not persisted in rollout JSONL,
so anything needing them must go through :class:`TaskMetadataProvider` /
:func:`attach_version_trajectories`.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from dtau_bench.evaluator.models import EvalSample

logger = logging.getLogger(__name__)


def _extract_gold_trajectory(task: Dict[str, Any]) -> List[str]:
    """
    Extract gold trajectory tool list from task data.

    Handles two formats:
    - Format A (single_domain_full): task["gold_trajectory"] is List[str]
    - Format B (all_tools_*): task["gold_trajectory"] is Dict with "tools" field
    """
    gold_traj = task.get("gold_trajectory", [])

    if isinstance(gold_traj, list):
        # Format A: direct list of tool names
        return [str(t) for t in gold_traj if t]

    if isinstance(gold_traj, dict):
        # Format B: dict with "tools" key
        tools = gold_traj.get("tools", [])
        return [str(t) for t in tools if t]

    logger.warning("Unexpected gold_trajectory type: %s", type(gold_traj))
    return []


def _extract_version_trajectories(task: Dict[str, Any]) -> Optional[Dict[str, List[str]]]:
    """
    Extract version_trajectories from task data (if present).

    Two layouts are supported:
    - Canonical task JSON (data/tasks/**): top-level "version_trajectories".
    - Legacy/nested: task["gold_trajectory"]["version_trajectories"].

    Rollout JSONL records contain NEITHER, so this returns None for them; use
    :func:`attach_version_trajectories` to enrich samples from the canonical
    task data.

    Returns dict mapping version -> list of tool names, or None.
    """
    candidates: List[Any] = [task.get("version_trajectories")]
    gold_traj = task.get("gold_trajectory")
    if isinstance(gold_traj, dict):
        candidates.append(gold_traj.get("version_trajectories"))

    for vt in candidates:
        if isinstance(vt, dict) and vt:
            return {
                str(version): [str(t) for t in tools if t]
                for version, tools in vt.items()
                if isinstance(tools, list)
            }
    return None


def _extract_entity_context(task: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract entity context for gold replay.

    Checks both:
    - task["gold_trajectory"]["entity_instances"] + ["value_domain_samples"] (Format B)
    - record-level "gold_execution_context" (fallback)
    """
    gold_traj = task.get("gold_trajectory")
    if isinstance(gold_traj, dict):
        context = {}
        entity_instances = gold_traj.get("entity_instances")
        if isinstance(entity_instances, dict):
            context["entity_instances"] = entity_instances
        value_domain_samples = gold_traj.get("value_domain_samples")
        if isinstance(value_domain_samples, dict):
            context["value_domain_samples"] = value_domain_samples
        if context:
            return context
    return {}


def _extract_trajectory_hash(task: Dict[str, Any]) -> str:
    """Extract trajectory hash from task data."""
    gold_traj = task.get("gold_trajectory")
    if isinstance(gold_traj, dict):
        return gold_traj.get("trajectory_hash", "") or ""
    return task.get("trajectory_hash", "") or ""


def _extract_instruction(task: Dict[str, Any]) -> str:
    """Extract instruction from task data."""
    return task.get("instruction", "") or ""


def _extract_domains(task: Dict[str, Any]) -> List[str]:
    """Extract domains from task data."""
    # Try "domains" field first (list)
    domains = task.get("domains")
    if isinstance(domains, list) and domains:
        return [str(d) for d in domains if d]

    # Try "domain" field (single string)
    domain = task.get("domain")
    if isinstance(domain, str) and domain:
        return [domain]

    # Try domain_key
    domain_key = task.get("domain_key")
    if isinstance(domain_key, str) and domain_key:
        return domain_key.split("_")

    return []


def _extract_messages(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract conversation messages from rollout record.

    Handles:
    - record["messages"] (legacy)
    - record["conversation"]["messages"] (schema v2)
    """
    messages = record.get("messages")
    if isinstance(messages, list):
        return messages

    conversation = record.get("conversation")
    if isinstance(conversation, dict):
        conv_messages = conversation.get("messages")
        if isinstance(conv_messages, list):
            return conv_messages

    return []


def _extract_agent_traj(record: Dict[str, Any]) -> List[str]:
    """Extract agent trajectory (tool names) from rollout record."""
    agent_traj = record.get("agent_traj", [])
    if isinstance(agent_traj, list):
        return [str(t) for t in agent_traj if t]
    return []


def parse_rollout_record(record: Dict[str, Any]) -> EvalSample:
    """
    Parse a single rollout record into an EvalSample.

    Args:
        record: A single JSON record from rollout JSONL file.

    Returns:
        EvalSample with all fields populated from the record.

    Raises:
        ValueError: If required fields are missing.
    """
    sample_id = record.get("id", "")
    if not sample_id:
        raise ValueError("Rollout record missing 'id' field")

    task = record.get("task", {})
    if not isinstance(task, dict):
        raise ValueError(f"Rollout record {sample_id} has invalid 'task' field")

    # Extract core fields
    gold_trajectory = _extract_gold_trajectory(task)
    entity_context = _extract_entity_context(task)
    messages = _extract_messages(record)
    agent_traj = _extract_agent_traj(record)
    domains = _extract_domains(task)
    instruction = _extract_instruction(task)
    version_trajectories = _extract_version_trajectories(task)
    trajectory_hash = _extract_trajectory_hash(task)

    # Version comes from run metadata or task
    run_meta = record.get("run", {}) or {}
    version = run_meta.get("version", "") or task.get("version", "") or ""

    # Gold execution context (if pre-computed)
    gold_execution_context = record.get("gold_execution_context")
    if gold_execution_context is None:
        # Also check inside task.gold_trajectory for step_param_codes
        gold_traj_data = task.get("gold_trajectory")
        if isinstance(gold_traj_data, dict):
            step_param_codes = gold_traj_data.get("step_param_codes")
            if step_param_codes:
                gold_execution_context = {"step_param_codes": step_param_codes}

    # Domain key
    domain_key = "_".join(domains) if domains else ""

    return EvalSample(
        id=sample_id,
        gold_trajectory=gold_trajectory,
        entity_context=entity_context,
        agent_traj=agent_traj,
        messages=messages,
        domains=domains,
        version=version,
        instruction=instruction,
        version_trajectories=version_trajectories,
        gold_execution_context=gold_execution_context,
        artifacts=record.get("artifacts", {}) or {},
        run_meta=run_meta,
        trajectory_hash=trajectory_hash,
        domain_key=domain_key,
        success=record.get("success", False),
        error=record.get("error"),
    )


def iter_rollout_records(filepath: Path) -> Iterator[Dict[str, Any]]:
    """
    Iterate over records in a rollout JSONL file.

    Args:
        filepath: Path to the JSONL file.

    Yields:
        Parsed JSON records.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                if not isinstance(record, dict):
                    logger.warning(
                        "Skipping line %d in %s: not a JSON object", line_num, filepath
                    )
                    continue
                yield record
            except json.JSONDecodeError as e:
                logger.warning(
                    "Skipping line %d in %s: JSON parse error: %s",
                    line_num,
                    filepath,
                    e,
                )
                continue


def load_rollout_samples(
    filepath: Path,
    *,
    max_samples: Optional[int] = None,
    require_success: bool = False,
    version_filter: Optional[str] = None,
) -> List[EvalSample]:
    """
    Load evaluation samples from a rollout JSONL file.

    Args:
        filepath: Path to the JSONL file.
        max_samples: Maximum number of samples to load (for debugging).
        require_success: If True, skip samples where rollout failed.
        version_filter: If set, only load samples matching this version.

    Returns:
        List of EvalSample objects.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Rollout file not found: {filepath}")

    samples: List[EvalSample] = []
    skipped = 0

    for record in iter_rollout_records(filepath):
        try:
            sample = parse_rollout_record(record)
        except (ValueError, KeyError) as e:
            logger.warning("Skipping record: %s", e)
            skipped += 1
            continue

        # Apply filters
        if require_success and not sample.success:
            skipped += 1
            continue
        if version_filter and sample.version != version_filter:
            skipped += 1
            continue

        samples.append(sample)

        if max_samples and len(samples) >= max_samples:
            break

    logger.info(
        "Loaded %d samples from %s (skipped %d)",
        len(samples),
        filepath.name,
        skipped,
    )
    return samples


class TaskMetadataProvider:
    """
    Authoritative source for gold trajectory metadata.

    Loads task JSON files from data/tasks/{split}/{domain}/*.json and indexes
    them by query_id. Evaluator uses this to fetch gold tools, step_param_codes,
    and entity context — ensuring these always come from the canonical task data,
    not from rollout JSONL (which may contain version-specific tools).

    Usage:
        provider = TaskMetadataProvider(data_dir / "tasks")
        task_meta = provider.get("1aa57fc8")
        # task_meta["gold_trajectory"]["tools"] → base version tools
        # task_meta["gold_trajectory"]["step_param_codes"] → matching codes
    """

    def __init__(self, tasks_dir: Path):
        """
        Initialize provider by scanning task JSON files.

        Args:
            tasks_dir: Path to data/tasks/ directory (contains split/domain/*.json).
        """
        self._cache: Dict[str, Dict[str, Any]] = {}  # query_id → task json dict
        self._load_all(Path(tasks_dir))

    def _load_all(self, tasks_dir: Path) -> None:
        """Scan data/tasks/{split}/{domain}/*.json and index by query_id."""
        if not tasks_dir.is_dir():
            logger.warning("TaskMetadataProvider: tasks directory not found: %s", tasks_dir)
            return

        count = 0
        for json_file in tasks_dir.rglob("*.json"):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if not isinstance(data, dict):
                    continue
                query_id = data.get("query_id", "")
                if query_id:
                    self._cache[query_id] = data
                    count += 1
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("TaskMetadataProvider: failed to load %s: %s", json_file, e)

        logger.info("TaskMetadataProvider: loaded %d tasks from %s", count, tasks_dir)

    def get(self, query_id: str) -> Optional[Dict[str, Any]]:
        """
        Look up a task by query_id.

        Args:
            query_id: The unique task identifier.

        Returns:
            Full task dict if found, None otherwise.
        """
        return self._cache.get(query_id)

    def get_gold_trajectory(self, query_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the gold_trajectory dict for a task.

        Returns the canonical gold_trajectory with tools, step_param_codes,
        entity_instances, and value_domain_samples from the base version.

        Args:
            query_id: The unique task identifier.

        Returns:
            gold_trajectory dict if found, None otherwise.
        """
        task = self._cache.get(query_id)
        if task is None:
            return None
        gold_traj = task.get("gold_trajectory")
        if isinstance(gold_traj, dict):
            return gold_traj
        return None

    def get_version_trajectories(self, query_id: str) -> Optional[Dict[str, List[str]]]:
        """
        Get the version -> gold-tool-sequence map for a task.

        Sourced from the canonical task JSON's top-level "version_trajectories"
        (keys: "base", "v1", ..., "v6").

        Args:
            query_id: The unique task identifier.

        Returns:
            Mapping version -> tool names, or None if unknown/absent.
        """
        task = self._cache.get(query_id)
        if task is None:
            return None
        return _extract_version_trajectories(task)

    def get_version_tools(self, query_id: str, version: str) -> Optional[List[str]]:
        """
        Get the version-translated gold tool sequence for a task.

        Args:
            query_id: The unique task identifier.
            version: Catalog version (e.g. "v2").

        Returns:
            Tool names for that version, or None when the task is unknown or the
            version is absent.

        Callers must not fall back to the base trajectory: base tool names are
        renamed, split or merged in later versions, so using them as gold
        deflates retrieval metrics with no visible error.
        """
        version = str(version or "").strip()
        if not version:
            return None
        version_trajectories = self.get_version_trajectories(query_id)
        if not version_trajectories:
            return None
        tools = version_trajectories.get(version)
        if tools is None:
            return None
        return list(tools)

    def __len__(self) -> int:
        return len(self._cache)

    def __contains__(self, query_id: str) -> bool:
        return query_id in self._cache


def attach_version_trajectories(
    samples: List[EvalSample],
    provider: TaskMetadataProvider,
) -> Dict[str, int]:
    """
    Fill ``EvalSample.version_trajectories`` from the canonical task JSON.

    Rollout records only carry the base gold trajectory, so version-aware
    metrics (currently tool retrieval) cannot work without this enrichment.
    Samples are mutated in place.

    Args:
        samples: Samples loaded from rollout JSONL.
        provider: Provider indexed over data/tasks/**.

    Returns:
        Stats dict with keys "attached", "already_present", "missing_task".
    """
    stats = {"attached": 0, "already_present": 0, "missing_task": 0}

    for sample in samples:
        if sample.version_trajectories:
            stats["already_present"] += 1
            continue
        version_trajectories = provider.get_version_trajectories(sample.id)
        if version_trajectories:
            sample.version_trajectories = version_trajectories
            stats["attached"] += 1
        else:
            stats["missing_task"] += 1

    if stats["missing_task"]:
        logger.warning(
            "attach_version_trajectories: %d/%d samples have no canonical task "
            "entry; version-aware gold will be unavailable for them.",
            stats["missing_task"],
            len(samples),
        )
    return stats


def load_rollout_dir(
    rollout_dir: Path,
    *,
    max_samples: Optional[int] = None,
    require_success: bool = False,
    version_filter: Optional[str] = None,
    domain_filter: Optional[str] = None,
) -> List[EvalSample]:
    """
    Load evaluation samples from all JSONL files in a rollout directory.

    Args:
        rollout_dir: Path to the rollouts/ directory.
        max_samples: Maximum total samples to load.
        require_success: If True, skip failed rollouts.
        version_filter: Only load samples matching this version.
        domain_filter: Only load files matching this domain name.

    Returns:
        List of EvalSample objects from all files.
    """
    rollout_dir = Path(rollout_dir)
    if not rollout_dir.is_dir():
        raise FileNotFoundError(f"Rollout directory not found: {rollout_dir}")

    all_samples: List[EvalSample] = []

    for jsonl_file in sorted(rollout_dir.glob("*.jsonl")):
        # Apply domain filter if specified
        if domain_filter:
            domain_name = jsonl_file.stem
            if domain_name != domain_filter:
                continue

        remaining = None
        if max_samples is not None:
            remaining = max_samples - len(all_samples)
            if remaining <= 0:
                break

        samples = load_rollout_samples(
            jsonl_file,
            max_samples=remaining,
            require_success=require_success,
            version_filter=version_filter,
        )
        all_samples.extend(samples)

    logger.info(
        "Loaded %d total samples from %s",
        len(all_samples),
        rollout_dir,
    )
    return all_samples
