"""Pydantic schemas for data validation."""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json


@dataclass
class SkillSchema:
    """Schema for a single skill."""
    name: str
    document: str
    content: str
    tools: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> tuple:
        """
        Validate skill data.

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        if not self.name:
            errors.append("name is required")

        if not self.document:
            errors.append("document is required")

        if not self.content:
            errors.append("content is required")

        if not self.tools or len(self.tools) == 0:
            errors.append("at least one tool is required")

        return (len(errors) == 0, errors)


@dataclass
class PlanSchema:
    """Schema for a plan."""
    task: str
    plan: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> tuple:
        """Validate plan data."""
        errors = []

        if not self.task:
            errors.append("task is required")

        if not self.plan:
            errors.append("plan is required")

        # Check plan format (should have steps)
        if self.plan and "# step" not in self.plan:
            errors.append("plan should contain step markers")

        return (len(errors) == 0, errors)


@dataclass
class SkillLibrarySchema:
    """Schema for the entire skill library."""
    version: str = "1.0"
    benchmark: str = "atomic"
    created_at: Optional[str] = None
    planning: Dict[str, Dict] = field(default_factory=dict)
    functional: List[Dict] = field(default_factory=list)
    atomic: List[Dict] = field(default_factory=list)
    embeddings: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()

    def validate(self) -> tuple:
        """Validate entire skill library."""
        errors = []

        # Validate functional skills
        for i, skill in enumerate(self.functional):
            schema = SkillSchema(
                name=skill.get("name", ""),
                document=skill.get("document", ""),
                content=skill.get("content", ""),
                tools=skill.get("tools", []),
                metadata=skill.get("metadata", {})
            )
            is_valid, skill_errors = schema.validate()
            if not is_valid:
                errors.extend([f"functional[{i}]: {e}" for e in skill_errors])

        # Validate atomic skills
        for i, skill in enumerate(self.atomic):
            schema = SkillSchema(
                name=skill.get("name", ""),
                document=skill.get("document", ""),
                content=skill.get("content", ""),
                tools=skill.get("tools", []),
                metadata=skill.get("metadata", {})
            )
            is_valid, skill_errors = schema.validate()
            if not is_valid:
                errors.extend([f"atomic[{i}]: {e}" for e in skill_errors])

        return (len(errors) == 0, errors)

    def to_dict(self) -> Dict:
        """Convert to dictionary format."""
        return {
            "version": self.version,
            "benchmark": self.benchmark,
            "created_at": self.created_at,
            "skills": {
                "planning": self.planning,
                "functional": self.functional,
                "atomic": self.atomic,
            },
            "embeddings": self.embeddings,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "SkillLibrarySchema":
        """Create from dictionary."""
        skills = data.get("skills", {})
        return cls(
            version=data.get("version", "1.0"),
            benchmark=data.get("benchmark", "atomic"),
            created_at=data.get("created_at"),
            planning=skills.get("planning", {}),
            functional=skills.get("functional", []),
            atomic=skills.get("atomic", []),
            embeddings=data.get("embeddings", {}),
        )

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> "SkillLibrarySchema":
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))
