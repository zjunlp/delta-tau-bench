"""
skill context Artifact Loader — Load skill library from disk.

Loads the skill library directory structure:
- plan_library.json
- {functional,atomic}_skills.prepared.jsonl
- tool_skills.prepared.jsonl
- {atomic,tool}_skills.inactive.jsonl
- skill_graph.json
- library_metadata.json

Supports three loading modes:
- "normal": Only active skills (for rollout)
- "all_active": All skills treated as active (ignore version partition)
- "evolution": Load all, keeping active/inactive separation

Usage:
    artifacts = SkillLibraryLoader.load("/path/to/skills", mode="normal")
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from .models import SkillContextArtifacts

logger = logging.getLogger(__name__)


class SkillLibraryLoader:
    """Loads skill context artifacts from a skill library directory.

    The directory is expected to contain:
    - plan_library.json: Dict mapping task → {plan, metadata}
    - functional_skills.prepared.jsonl: Active functional skills
    - atomic_skills.prepared.jsonl: Active atomic skills
    - tool_skills.prepared.jsonl: Active tool skills
    - atomic_skills.inactive.jsonl: Inactive atomic skills (optional)
    - tool_skills.inactive.jsonl: Inactive tool skills (optional)
    - skill_graph.json: Skill graph (optional)
    - library_metadata.json: Library metadata (optional)
    """

    @classmethod
    def load(
        cls,
        skill_library_dir: str | Path,
        mode: str = "normal",
    ) -> SkillContextArtifacts:
        """Load skill library artifacts from a directory.

        Args:
            skill_library_dir: Path to the skill library directory.
            mode: Loading mode:
                - "normal": Only active skills (*.prepared.jsonl)
                - "all_active": Active + inactive treated as all active
                - "evolution": Load all, keeping separation

        Returns:
            SkillContextArtifacts container with loaded data.

        Raises:
            FileNotFoundError: If directory or required files don't exist.
        """
        path = Path(skill_library_dir)
        if not path.exists():
            raise FileNotFoundError(f"skill context directory not found: {path}")

        plan_path = path / "plan_library.json"
        if not plan_path.exists():
            raise FileNotFoundError(f"Missing skill context plan library: {plan_path}")

        # Load plans
        plans = cls._load_plans(plan_path)

        # Load active skills
        skills = cls._load_skills(path)

        # Load active tool skills
        tool_skills = cls._load_tool_skills(path)

        # Handle inactive skills based on mode
        inactive_atomic_skills: List[Dict[str, Any]] = []
        inactive_tool_skills: List[Dict[str, Any]] = []

        if mode in ("all_active", "evolution"):
            inactive_atomic_skills = cls._load_inactive_skills(path, skill_type="atomic")
            inactive_tool_skills = cls._load_inactive_tool_skills(path)

        if mode == "all_active":
            # Merge inactive into active
            skills = skills + inactive_atomic_skills
            tool_skills = tool_skills + inactive_tool_skills
            inactive_atomic_skills = []
            inactive_tool_skills = []

        # Load optional files
        skill_graph = cls._load_skill_graph(path)
        library_metadata = cls._load_library_metadata(path)

        logger.info(
            f"Loaded skill context library from {path}: "
            f"{len(plans)} plans, {len(skills)} skills, "
            f"{len(tool_skills)} tool skills, "
            f"graph={'yes' if skill_graph else 'no'}, "
            f"mode={mode}"
        )

        return SkillContextArtifacts(
            source_dir=str(path),
            plans=plans,
            skills=skills,
            tool_skills=tool_skills,
            inactive_atomic_skills=inactive_atomic_skills,
            inactive_tool_skills=inactive_tool_skills,
            skill_graph=skill_graph,
            library_metadata=library_metadata,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Loading Helpers
    # ─────────────────────────────────────────────────────────────────────────

    @classmethod
    def _load_plans(cls, plan_path: Path) -> List[Dict[str, Any]]:
        """Load plan_library.json into normalized plan entries."""
        with plan_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        # plan_library.json has format: {"plan": {task: {plan: ..., metadata: ...}}}
        # or just {task: {plan: ..., metadata: ...}}
        plan_map = data.get("plan", data)
        plans: List[Dict[str, Any]] = []

        for task, value in plan_map.items():
            if isinstance(value, dict):
                plan = value.get("plan", "")
                metadata = value.get("metadata", {})
            else:
                plan = str(value)
                metadata = {}

            plans.append({
                "task": task,
                "plan": plan,
                "metadata": metadata,
                "embedding_text": f"{task}\n{plan}".strip(),
            })

        return plans

    @classmethod
    def _load_skills(cls, path: Path) -> List[Dict[str, Any]]:
        """Load active functional + atomic skills from prepared JSONL files."""
        skill_rows: List[Dict[str, Any]] = []

        for skill_type in ("functional", "atomic"):
            prepared_path = path / f"{skill_type}_skills.prepared.jsonl"
            raw_path = path / f"{skill_type}_skills.raw.jsonl"

            if prepared_path.exists():
                rows = cls._load_jsonl(prepared_path)
            elif raw_path.exists():
                rows = cls._load_jsonl(raw_path)
            else:
                continue

            for row in rows:
                skill = row.get("skill", row)
                if not isinstance(skill, dict):
                    continue
                metadata = dict(skill.get("metadata", {}))
                metadata.setdefault("skill_type", skill_type)
                embedding_text = row.get("embedding_text") or cls._build_embedding_text(skill)
                skill_rows.append({
                    "skill_type": metadata.get("skill_type", skill_type),
                    "name": skill.get("name", ""),
                    "document": skill.get("document", ""),
                    "content": skill.get("content", ""),
                    "tools": skill.get("tools", []),
                    "metadata": metadata,
                    "embedding_text": embedding_text,
                    "raw_skill": skill,
                })

        return skill_rows

    @classmethod
    def _load_tool_skills(cls, path: Path) -> List[Dict[str, Any]]:
        """Load active tool skills from prepared JSONL."""
        prepared_path = path / "tool_skills.prepared.jsonl"
        raw_path = path / "tool_skills.merged.jsonl"

        if prepared_path.exists():
            rows = cls._load_jsonl(prepared_path)
        elif raw_path.exists():
            rows = cls._load_jsonl(raw_path)
        else:
            return []

        skill_rows: List[Dict[str, Any]] = []
        for row in rows:
            skill = row.get("skill", row)
            if not isinstance(skill, dict):
                continue
            metadata = dict(skill.get("metadata", {}))
            metadata.setdefault("skill_type", "tool")
            embedding_text = row.get("embedding_text") or cls._build_embedding_text(skill)
            skill_rows.append({
                "skill_type": "tool",
                "name": skill.get("name", ""),
                "document": skill.get("document", ""),
                "content": skill.get("content", ""),
                "tools": skill.get("tools", []),
                "metadata": metadata,
                "embedding_text": embedding_text,
                "raw_skill": skill,
            })
        return skill_rows

    @classmethod
    def _load_inactive_skills(cls, path: Path, *, skill_type: str) -> List[Dict[str, Any]]:
        """Load inactive skills of given type."""
        inactive_path = path / f"{skill_type}_skills.inactive.jsonl"
        if not inactive_path.exists():
            return []
        rows = cls._load_jsonl(inactive_path)
        skill_rows: List[Dict[str, Any]] = []
        for row in rows:
            skill = row.get("skill", row)
            if not isinstance(skill, dict):
                continue
            metadata = dict(skill.get("metadata", {}))
            metadata.setdefault("skill_type", skill_type)
            embedding_text = row.get("embedding_text") or cls._build_embedding_text(skill)
            skill_rows.append({
                "skill_type": metadata.get("skill_type", skill_type),
                "name": skill.get("name", ""),
                "document": skill.get("document", ""),
                "content": skill.get("content", ""),
                "tools": skill.get("tools", []),
                "metadata": metadata,
                "embedding_text": embedding_text,
                "raw_skill": skill,
            })
        return skill_rows

    @classmethod
    def _load_inactive_tool_skills(cls, path: Path) -> List[Dict[str, Any]]:
        """Load inactive tool skills."""
        inactive_path = path / "tool_skills.inactive.jsonl"
        if not inactive_path.exists():
            return []
        rows = cls._load_jsonl(inactive_path)
        skill_rows: List[Dict[str, Any]] = []
        for row in rows:
            skill = row.get("skill", row)
            if not isinstance(skill, dict):
                continue
            metadata = dict(skill.get("metadata", {}))
            metadata.setdefault("skill_type", "tool")
            embedding_text = row.get("embedding_text") or cls._build_embedding_text(skill)
            skill_rows.append({
                "skill_type": "tool",
                "name": skill.get("name", ""),
                "document": skill.get("document", ""),
                "content": skill.get("content", ""),
                "tools": skill.get("tools", []),
                "metadata": metadata,
                "embedding_text": embedding_text,
                "raw_skill": skill,
            })
        return skill_rows

    @classmethod
    def _load_skill_graph(cls, path: Path) -> Optional[Dict[str, Any]]:
        """Load skill_graph.json if present."""
        graph_path = path / "skill_graph.json"
        if not graph_path.exists():
            return None
        with graph_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(f"Skill graph must be a JSON object: {graph_path}")
        return data

    @classmethod
    def _load_library_metadata(cls, path: Path) -> Optional[Dict[str, Any]]:
        """Load library_metadata.json if present."""
        metadata_path = path / "library_metadata.json"
        if not metadata_path.exists():
            return None
        with metadata_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(f"Library metadata must be a JSON object: {metadata_path}")
        return data

    # ─────────────────────────────────────────────────────────────────────────
    # Utilities
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
        """Load a JSONL file into a list of dicts."""
        rows: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    @staticmethod
    def _build_embedding_text(skill: Dict[str, Any]) -> str:
        """Build embedding text from skill fields (name + document + content)."""
        return "\n".join(
            part
            for part in [
                skill.get("name", ""),
                skill.get("document", ""),
                skill.get("content", ""),
            ]
            if part
        )
