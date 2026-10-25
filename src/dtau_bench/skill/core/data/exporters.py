"""Data exporters for skill libraries."""

import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime

from ..models.skill import SkillLibrary, Skill
from .schemas import SkillLibrarySchema

logger = logging.getLogger(__name__)


class SkillLibraryExporter:
    """Export skill libraries to various formats."""

    @staticmethod
    def export_json(
        library: SkillLibrary,
        path: str,
        indent: int = 2
    ) -> None:
        """
        Export skill library to JSON file.

        Args:
            library: SkillLibrary instance
            path: Output path
            indent: JSON indentation
        """
        with open(path, "w", encoding="utf-8") as f:
            json.dump(library.to_dict(), f, ensure_ascii=False, indent=indent)

        logger.info(f"Exported skill library to {path}")

    @staticmethod
    def export_skills_jsonl(
        skills: List[Dict],
        path: str
    ) -> None:
        """
        Export skills to JSONL file.

        Args:
            skills: List of skill dictionaries
            path: Output path
        """
        with open(path, "w", encoding="utf-8") as f:
            for skill in skills:
                f.write(json.dumps(skill, ensure_ascii=False) + "\n")

        logger.info(f"Exported {len(skills)} skills to {path}")

    @staticmethod
    def export_plan_library(
        plans: Dict[str, str],
        path: str,
        epoch: int = 1
    ) -> None:
        """
        Export plan library to JSON file.

        Args:
            plans: Dictionary mapping task to plan
            path: Output path
            epoch: Training epoch number
        """
        output = {
            "train_epoch": epoch,
            "plan": {}
        }

        for task, plan in plans.items():
            output["plan"][task] = {
                "plan": plan,
                "metadata": {
                    "created_at": datetime.now().isoformat()
                }
            }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        logger.info(f"Exported {len(plans)} plans to {path}")

    @staticmethod
    def validate_and_export(
        library: SkillLibrary,
        path: str
    ) -> tuple:
        """
        Validate and export skill library.

        Args:
            library: SkillLibrary instance
            path: Output path

        Returns:
            Tuple of (success, errors)
        """
        # Convert to schema for validation
        schema = SkillLibrarySchema(
            version=library.version,
            benchmark=library.benchmark,
            created_at=library.created_at,
            planning={k: v.to_dict() for k, v in library.planning.items()},
            functional=[s.to_dict() for s in library.functional],
            atomic=[s.to_dict() for s in library.atomic],
            embeddings_config=library.embeddings_config,
        )

        is_valid, errors = schema.validate()

        if is_valid:
            SkillLibraryExporter.export_json(library, path)
            return (True, [])
        else:
            logger.error(f"Validation failed: {errors}")
            return (False, errors)

    @staticmethod
    def create_library_from_skills(
        functional_skills: List[Dict],
        atomic_skills: List[Dict],
        plans: Dict[str, str],
        benchmark: str = "atomic",
        version: str = "1.0"
    ) -> SkillLibrary:
        """
        Create a SkillLibrary from skill and plan data.

        Args:
            functional_skills: List of functional skill dictionaries
            atomic_skills: List of atomic skill dictionaries
            plans: Dictionary of plans
            benchmark: Benchmark name
            version: Library version

        Returns:
            SkillLibrary instance
        """
        library = SkillLibrary(
            version=version,
            benchmark=benchmark
        )

        # Add functional skills
        for skill_data in functional_skills:
            skill = Skill.from_dict(skill_data.get("skill", skill_data))
            library.add_functional_skill(skill)

        # Add atomic skills
        for skill_data in atomic_skills:
            skill = Skill.from_dict(skill_data.get("skill", skill_data))
            library.add_atomic_skill(skill)

        # Add plans
        from ..models.skill import PlanSkill
        for task, plan in plans.items():
            library.add_plan(task, PlanSkill(task=task, plan=plan))

        return library
