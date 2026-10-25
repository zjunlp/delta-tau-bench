"""Trajectory data models for skill context."""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Literal
import json


@dataclass
class ToolCall:
    """Represents a tool/API call in a trajectory."""
    id: str
    name: str
    arguments: Dict[str, Any]
    requestor: str = "assistant"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "arguments": self.arguments,
            "requestor": self.requestor,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolCall":
        return cls(
            id=data.get("id", ""),
            name=data["name"],
            arguments=data.get("arguments", {}),
            requestor=data.get("requestor", "assistant"),
        )


@dataclass
class TrajectoryStep:
    """
    A single step in an agent trajectory.

    Represents one turn of interaction: assistant action + environment feedback.
    """
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    tool_calls: Optional[List[ToolCall]] = None
    tool_call_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "role": self.role,
            "content": self.content,
        }
        if self.tool_calls:
            result["tool_calls"] = [tc.to_dict() for tc in self.tool_calls]
        if self.tool_call_id:
            result["id"] = self.tool_call_id
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrajectoryStep":
        tool_calls = None
        if data.get("tool_calls"):
            tool_calls = [ToolCall.from_dict(tc) for tc in data["tool_calls"]]
        return cls(
            role=data["role"],
            content=data.get("content", ""),
            tool_calls=tool_calls,
            tool_call_id=data.get("id"),
        )

    def get_tool_names(self) -> List[str]:
        """Extract tool names from this step."""
        if not self.tool_calls:
            return []
        return [tc.name for tc in self.tool_calls]


@dataclass
class Trajectory:
    """
    Complete agent trajectory for a task.

    Contains the full interaction history including system prompts,
    user messages, assistant actions, and tool responses.
    """
    trajectory_id: str
    benchmark: str
    task_id: str
    user_task: str
    steps: List[TrajectoryStep]
    reward: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trajectory_id": self.trajectory_id,
            "benchmark": self.benchmark,
            "task_id": self.task_id,
            "user_task": self.user_task,
            "task_history": [step.to_dict() for step in self.steps],
            "reward": self.reward,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Trajectory":
        # Handle different field names for task history
        task_history = data.get("task_history", data.get("trajectory", []))
        steps = [TrajectoryStep.from_dict(s) for s in task_history]

        return cls(
            trajectory_id=data.get("trajectory_id", ""),
            benchmark=data.get("benchmark", "unknown"),
            task_id=data.get("task_id", ""),
            user_task=data.get("user_task", ""),
            steps=steps,
            reward=data.get("reward", data.get("after_score", 0.0)),
            metadata=data.get("metadata", {}),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "Trajectory":
        return cls.from_dict(json.loads(json_str))

    def is_successful(self, threshold: float = 0.999) -> bool:
        """Check if trajectory is successful based on reward threshold."""
        return self.reward > threshold

    def get_all_tool_calls(self) -> List[ToolCall]:
        """Get all tool calls from the trajectory."""
        tool_calls = []
        for step in self.steps:
            if step.tool_calls:
                tool_calls.extend(step.tool_calls)
        return tool_calls

    def get_all_tools_used(self) -> set:
        """
        Get set of all tool names used in the trajectory.

        This is used for omission-based atomic skill extraction.
        """
        tools = set()
        for step in self.steps:
            if step.role == "assistant" and step.tool_calls:
                for tc in step.tool_calls:
                    tools.add(tc.name)
        return tools

    def get_assistant_steps(self) -> List[TrajectoryStep]:
        """Get only assistant steps from the trajectory."""
        return [s for s in self.steps if s.role == "assistant"]

    def get_tool_response_steps(self) -> List[TrajectoryStep]:
        """Get only tool response steps from the trajectory."""
        return [s for s in self.steps if s.role == "tool"]

    def get_step_pairs(self) -> List[tuple]:
        """
        Get pairs of (assistant_action, tool_response).

        Useful for skill extraction where we need to analyze
        action-feedback pairs.
        """
        pairs = []
        i = 0
        while i < len(self.steps) - 1:
            if self.steps[i].role == "assistant" and self.steps[i].tool_calls:
                # Find corresponding tool response
                if self.steps[i + 1].role == "tool":
                    pairs.append((self.steps[i], self.steps[i + 1]))
            i += 1
        return pairs

    def __len__(self) -> int:
        return len(self.steps)


@dataclass
class TrajectoryBatch:
    """Container for multiple trajectories grouped by task."""
    task_id: str
    user_task: str
    successful: List[Trajectory] = field(default_factory=list)
    failed: List[Trajectory] = field(default_factory=list)
    plan: Optional[str] = None
    exp_metadata: Dict[str, Any] = field(default_factory=dict)

    def add_trajectory(self, trajectory: Trajectory, threshold: float = 0.999) -> None:
        """Add a trajectory to the appropriate list based on success."""
        if trajectory.is_successful(threshold):
            self.successful.append(trajectory)
        else:
            self.failed.append(trajectory)

    def get_shortest_successful(self) -> Optional[Trajectory]:
        """Get the shortest successful trajectory."""
        if not self.successful:
            return None
        return min(self.successful, key=len)

    def has_successful(self) -> bool:
        return len(self.successful) > 0

    def has_failed(self) -> bool:
        return len(self.failed) > 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "user_task": self.user_task,
            "successful_trajectories": [t.to_dict() for t in self.successful],
            "failed_trajectories": [t.to_dict() for t in self.failed],
            "plan": self.plan,
            "exp_metadata": self.exp_metadata,
        }


def group_trajectories_by_task(
    trajectories: List[Trajectory],
    threshold: float = 0.999
) -> Dict[str, TrajectoryBatch]:
    """
    Group trajectories by task ID.

    Args:
        trajectories: List of trajectories to group
        threshold: Success threshold for reward

    Returns:
        Dictionary mapping task_id to TrajectoryBatch
    """
    batches: Dict[str, TrajectoryBatch] = {}

    for traj in trajectories:
        task_id = traj.task_id or traj.user_task

        if task_id not in batches:
            batches[task_id] = TrajectoryBatch(
                task_id=task_id,
                user_task=traj.user_task,
            )

        batches[task_id].add_trajectory(traj, threshold)

    return batches
