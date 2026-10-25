"""
Data loader - loads task data and tool definitions from the dtau_bench data directory.

Provides:
- TaskData: Data class representing a single rollout task
- load_tasks(): Load tasks for a given split/domain/version
- load_domain_tasks(): Load all tasks for a specific domain
- discover_domains(): List all available domains
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from dtau_bench.utils.logging import get_logger

logger = get_logger("tools.data_loader")


# ─────────────────────────────────────────────────────────────────────────────
# Data Models
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GoldTrajectory:
    """Gold (reference) trajectory for a task.
    
    Attributes:
        tools: Ordered list of gold tool function names.
        trajectory_hash: Hash for tracking/dedup.
        step_param_codes: Python code snippets for parameter generation at each step.
        entity_instances: Entity data used in this trajectory.
        value_domain_samples: Concrete parameter values for each tool call.
    """
    tools: List[str]
    trajectory_hash: str = ""
    step_param_codes: List[str] = field(default_factory=list)
    entity_instances: Dict[str, Any] = field(default_factory=dict)
    value_domain_samples: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskData:
    """A single rollout task loaded from data/tasks/.
    
    Contains all information needed to run a rollout for one sample:
    - The conversation setup (messages, user prompt)
    - The gold trajectory (for evaluation)
    - Version-specific trajectories
    - Metadata for tracking
    
    Attributes:
        query_id: Unique task identifier.
        domain: Domain name (e.g., 'AthleteContentDashboard').
        split: Data split ('train' or 'test').
        instruction: Full instruction text (with entity values embedded).
        messages: Initial conversation messages.
        user_system_prompt: System prompt for the simulated user.
        gold_trajectory: The base gold trajectory.
        version_trajectories: Dict mapping version → tool sequence.
        metadata: Additional metadata (combo_id, trajectory_hash, reason_for_call, etc.).
    """
    query_id: str
    domain: str
    split: str
    instruction: str
    messages: List[Dict[str, str]]
    user_system_prompt: str
    gold_trajectory: GoldTrajectory
    version_trajectories: Dict[str, List[str]]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_version_tools(self, version: str) -> List[str]:
        """Get the gold tool sequence for a specific version.
        
        Args:
            version: Version string (e.g., 'v1', 'base').
        
        Returns:
            List of tool function names in order.
        
        Raises:
            KeyError: If version not found in version_trajectories.
        """
        if version not in self.version_trajectories:
            available = list(self.version_trajectories.keys())
            raise KeyError(
                f"Version '{version}' not found for task {self.query_id}. "
                f"Available: {available}"
            )
        return self.version_trajectories[version]

    def get_initial_messages(self) -> List[Dict[str, str]]:
        """Get the initial conversation messages (typically one user message)."""
        return list(self.messages)

    @property
    def combo_id(self) -> str:
        """Shortcut to metadata combo_id."""
        return self.metadata.get("combo_id", "")

    @property
    def reason_for_call(self) -> str:
        """Shortcut to metadata reason_for_call."""
        return self.metadata.get("reason_for_call", "")

    def to_dict(self) -> Dict[str, Any]:
        """Convert back to dict (for serialization in rollout output)."""
        return {
            "query_id": self.query_id,
            "domain": self.domain,
            "split": self.split,
            "instruction": self.instruction,
            "gold_trajectory": {
                "tools": self.gold_trajectory.tools,
                "trajectory_hash": self.gold_trajectory.trajectory_hash,
                "step_param_codes": self.gold_trajectory.step_param_codes,
                "entity_instances": self.gold_trajectory.entity_instances,
                "value_domain_samples": self.gold_trajectory.value_domain_samples,
            },
            "version_trajectories": self.version_trajectories,
            "metadata": self.metadata,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Parsing
# ─────────────────────────────────────────────────────────────────────────────

def _parse_task_json(data: Dict[str, Any]) -> TaskData:
    """Parse a task JSON dict into a TaskData object.
    
    Args:
        data: Raw JSON dict from task file.
    
    Returns:
        Parsed TaskData object.
    """
    gold_traj_data = data.get("gold_trajectory", {})
    gold_trajectory = GoldTrajectory(
        tools=gold_traj_data.get("tools", []),
        trajectory_hash=gold_traj_data.get("trajectory_hash", ""),
        step_param_codes=gold_traj_data.get("step_param_codes", []),
        entity_instances=gold_traj_data.get("entity_instances", {}),
        value_domain_samples=gold_traj_data.get("value_domain_samples", {}),
    )
    
    return TaskData(
        query_id=data["query_id"],
        domain=data["domain"],
        split=data["split"],
        instruction=data.get("instruction", ""),
        messages=data.get("messages", []),
        user_system_prompt=data.get("user_system_prompt", ""),
        gold_trajectory=gold_trajectory,
        version_trajectories=data.get("version_trajectories", {}),
        metadata=data.get("metadata", {}),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Loading Functions
# ─────────────────────────────────────────────────────────────────────────────

def load_task(task_path: Path) -> TaskData:
    """Load a single task from a JSON file.
    
    Args:
        task_path: Path to the task JSON file.
    
    Returns:
        Parsed TaskData object.
    """
    with task_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return _parse_task_json(data)


def load_domain_tasks(
    domain: str,
    split: str,
    data_dir: Path,
) -> List[TaskData]:
    """Load all tasks for a specific domain and split.
    
    Args:
        domain: Domain name (e.g., 'AthleteContentDashboard').
        split: Data split ('train' or 'test').
        data_dir: Root data directory.
    
    Returns:
        List of TaskData objects, sorted by query_id.
    
    Raises:
        FileNotFoundError: If the domain/split directory doesn't exist.
    """
    tasks_dir = data_dir / "tasks" / split / domain
    if not tasks_dir.exists():
        raise FileNotFoundError(
            f"Tasks directory not found: {tasks_dir}\n"
            f"Available splits: {[d.name for d in (data_dir / 'tasks').iterdir() if d.is_dir()]}"
        )
    
    tasks = []
    for task_file in sorted(tasks_dir.glob("*.json")):
        try:
            task = load_task(task_file)
            tasks.append(task)
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to load task {task_file}: {e}")
            continue
    
    logger.info(f"Loaded {len(tasks)} tasks for {domain}/{split}")
    return tasks


def load_tasks(
    split: str,
    data_dir: Path,
    domains: Optional[List[str]] = None,
    version: Optional[str] = None,
) -> Dict[str, List[TaskData]]:
    """Load tasks for one or more domains.
    
    Args:
        split: Data split ('train' or 'test').
        data_dir: Root data directory.
        domains: List of domain names. If None, loads all available domains.
        version: If specified, filters out tasks that don't have this version
            in their version_trajectories.
    
    Returns:
        Dict mapping domain name to list of TaskData objects.
    """
    if domains is None:
        domains = discover_domains(data_dir, split)
    
    all_tasks: Dict[str, List[TaskData]] = {}
    
    for domain in domains:
        tasks = load_domain_tasks(domain, split, data_dir)
        
        # Filter by version if specified
        if version:
            tasks = [t for t in tasks if version in t.version_trajectories]
            if len(tasks) == 0:
                logger.warning(f"No tasks with version '{version}' found for {domain}/{split}")
        
        all_tasks[domain] = tasks
    
    total = sum(len(ts) for ts in all_tasks.values())
    logger.info(
        f"Loaded {total} tasks across {len(all_tasks)} domains "
        f"(split={split}, version={version or 'all'})"
    )
    return all_tasks


def discover_domains(data_dir: Path, split: Optional[str] = None) -> List[str]:
    """Discover all available domain names.
    
    Args:
        data_dir: Root data directory.
        split: Optional split to check within. If None, checks tools directory.
    
    Returns:
        Sorted list of domain names.
    """
    if split:
        base = data_dir / "tasks" / split
    else:
        base = data_dir / "tools"
    
    if not base.exists():
        return []
    
    return sorted([d.name for d in base.iterdir() if d.is_dir()])


def get_available_versions(domain: str, data_dir: Path) -> List[str]:
    """Get all available tool versions for a domain.
    
    Args:
        domain: Domain name.
        data_dir: Root data directory.
    
    Returns:
        Sorted list of version strings (e.g., ['base', 'v1', 'v2', ...]).
    """
    tools_dir = data_dir / "tools" / domain
    if not tools_dir.exists():
        return []
    
    versions = []
    for f in tools_dir.glob("*_tools.json"):
        # Extract version from filename: "v1_tools.json" → "v1"
        version = f.stem.replace("_tools", "")
        versions.append(version)
    
    # Sort: base first, then v1, v2, ...
    def sort_key(v):
        if v == "base":
            return (0, 0)
        try:
            return (1, int(v[1:]))
        except (ValueError, IndexError):
            return (2, 0)
    
    return sorted(versions, key=sort_key)


def get_task_count(split: str, data_dir: Path, domain: Optional[str] = None) -> int:
    """Get the number of tasks available.
    
    Args:
        split: Data split.
        data_dir: Root data directory.
        domain: Optional domain filter.
    
    Returns:
        Total number of task JSON files.
    """
    if domain:
        tasks_dir = data_dir / "tasks" / split / domain
        return len(list(tasks_dir.glob("*.json"))) if tasks_dir.exists() else 0
    
    total = 0
    for d in discover_domains(data_dir, split):
        tasks_dir = data_dir / "tasks" / split / d
        total += len(list(tasks_dir.glob("*.json")))
    return total
