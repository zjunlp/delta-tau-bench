"""
Skill data models for the skill context system.

Provides:
- PlanningDocument: A planning skill (task → plan) for retrieval.
- SkillDocument: A functional or atomic skill for retrieval.
- SkillContextArtifacts: Container for loaded skill library artifacts.

These are the core retrieval-oriented models used by the skill context runtime.
The 5-key schema for a skill: name, document, content, tools, metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set


# ============================================================================
# Planning Document (for plan retrieval)
# ============================================================================


@dataclass
class PlanningDocument:
    """Planning skill document for retrieval.

    Represents a (task, plan) pair from plan_library.json.
    The embedding_text is used for vector search.

    Attributes:
        task: The task description (key in plan_library.json).
        plan: The step-by-step plan text.
        embedding_text: Combined text for embedding (task + plan).
        metadata: Additional metadata from the plan library.
        embedding: The embedding vector (set after indexing/loading).
    """

    task: str
    plan: str
    embedding_text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None

    def to_dict(self, include_embedding: bool = False) -> Dict[str, Any]:
        """Serialize to dict (for saving to documents.json)."""
        result = {
            "task": self.task,
            "plan": self.plan,
            "embedding_text": self.embedding_text,
            "metadata": self.metadata,
        }
        if include_embedding and self.embedding is not None:
            result["embedding"] = self.embedding
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PlanningDocument":
        """Deserialize from dict."""
        return cls(
            task=data["task"],
            plan=data["plan"],
            embedding_text=data["embedding_text"],
            metadata=data.get("metadata", {}),
            embedding=data.get("embedding"),
        )


# ============================================================================
# Skill Document (for skill retrieval: functional + atomic)
# ============================================================================


@dataclass
class SkillDocument:
    """Functional or atomic skill document for retrieval.

    Follows the 5-key schema: name, document, content, tools, metadata.
    The skill_type distinguishes functional vs atomic skills.

    Attributes:
        skill_type: "functional" or "atomic".
        name: Skill name (unique identifier within type).
        document: Short description of what the skill does.
        content: Detailed content (step-by-step usage).
        tools: List of tool names this skill references.
        embedding_text: Text used for vector embedding.
        raw_skill: Original skill dict (preserved for full fidelity).
        metadata: Additional metadata.
        embedding: The embedding vector (set after indexing/loading).
    """

    skill_type: str
    name: str
    document: str
    content: str
    tools: List[str]
    embedding_text: str
    raw_skill: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None

    def to_dict(self, include_embedding: bool = False) -> Dict[str, Any]:
        """Serialize to dict (for saving to documents.json)."""
        result = {
            "skill_type": self.skill_type,
            "name": self.name,
            "document": self.document,
            "content": self.content,
            "tools": self.tools,
            "embedding_text": self.embedding_text,
            "raw_skill": self.raw_skill,
            "metadata": self.metadata,
        }
        if include_embedding and self.embedding is not None:
            result["embedding"] = self.embedding
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SkillDocument":
        """Deserialize from dict."""
        return cls(
            skill_type=data["skill_type"],
            name=data["name"],
            document=data.get("document", ""),
            content=data.get("content", ""),
            tools=data.get("tools", []),
            embedding_text=data["embedding_text"],
            raw_skill=data.get("raw_skill", {}),
            metadata=data.get("metadata", {}),
            embedding=data.get("embedding"),
        )


# ============================================================================
# Skill Graph State (parsed from skill_graph.json)
# ============================================================================


@dataclass
class SkillGraphState:
    """Parsed state from the skill graph, ready for runtime use.

    The skill graph has three layers:
    - Planning → Atomic (which atomics are linked to which plans)
    - Atomic → Tool (which tool skills attach to which atomics)

    Attributes:
        planning_to_atomic: Mapping: planning skill name → set of linked atomic skill names.
        atomic_to_tool: Mapping: atomic skill name → set of linked tool names.
        tool_skill_by_tool_name: Mapping: tool name → tool skill dict.
        graph_payload: Raw graph JSON for serialization if needed.
    """

    planning_to_atomic: Dict[str, Set[str]] = field(default_factory=dict)
    atomic_to_tool: Dict[str, Set[str]] = field(default_factory=dict)
    tool_skill_by_tool_name: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    graph_payload: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# skill context Artifacts (loaded from disk)
# ============================================================================


@dataclass
class SkillContextArtifacts:
    """Container for all artifacts loaded from a skill library directory.

    A skill library directory contains:
    - plan_library.json: Planning skills (task → plan mappings)
    - {functional,atomic}_skills.prepared.jsonl: Active skills
    - tool_skills.prepared.jsonl: Active tool skills
    - {atomic,tool}_skills.inactive.jsonl: Inactive skills
    - skill_graph.json: Three-layer graph (optional)
    - library_metadata.json: Library metadata (optional)
    """

    source_dir: str
    plans: List[Dict[str, Any]]
    skills: List[Dict[str, Any]]
    tool_skills: List[Dict[str, Any]]
    inactive_atomic_skills: List[Dict[str, Any]] = field(default_factory=list)
    inactive_tool_skills: List[Dict[str, Any]] = field(default_factory=list)
    skill_graph: Optional[Dict[str, Any]] = None
    library_metadata: Optional[Dict[str, Any]] = None
