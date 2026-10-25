"""Data loaders for trajectories and skill libraries."""

import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

from ..models.trajectory import Trajectory
from ..models.skill import SkillLibrary

logger = logging.getLogger(__name__)


class TrajectoryLoader:
    """Load trajectories from various formats."""

    @staticmethod
    def load_jsonl(path: str) -> List[Dict]:
        """
        Load trajectories from JSONL file.

        Args:
            path: Path to JSONL file

        Returns:
            List of trajectory dictionaries
        """
        trajectories = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    trajectories.append(json.loads(line))

        logger.info(f"Loaded {len(trajectories)} trajectories from {path}")
        return trajectories

    @staticmethod
    def load_json(path: str) -> List[Dict]:
        """
        Load trajectories from JSON file.

        Args:
            path: Path to JSON file

        Returns:
            List of trajectory dictionaries
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Handle different formats
        if isinstance(data, list):
            trajectories = data
        elif isinstance(data, dict):
            # Could be a dict with plan key
            if "plan" in data:
                trajectories = list(data["plan"].values())
            else:
                trajectories = [data]
        else:
            trajectories = []

        logger.info(f"Loaded {len(trajectories)} trajectories from {path}")
        return trajectories

    @staticmethod
    def load(path: str) -> List[Dict]:
        """
        Load trajectories from file (auto-detect format).

        Args:
            path: Path to file

        Returns:
            List of trajectory dictionaries
        """
        path_obj = Path(path)

        if path_obj.suffix == ".jsonl":
            return TrajectoryLoader.load_jsonl(path)
        elif path_obj.suffix == ".json":
            return TrajectoryLoader.load_json(path)
        else:
            raise ValueError(f"Unsupported file format: {path_obj.suffix}")

    @staticmethod
    def filter_by_reward(
        trajectories: List[Dict],
        threshold: float = 0.999
    ) -> List[Dict]:
        """
        Filter trajectories by reward threshold.

        Args:
            trajectories: List of trajectory dictionaries
            threshold: Minimum reward threshold

        Returns:
            Filtered list of trajectories
        """
        filtered = [
            t for t in trajectories
            if t.get("reward", t.get("after_score", 0)) > threshold
        ]

        logger.info(
            f"Filtered {len(filtered)}/{len(trajectories)} trajectories "
            f"(threshold: {threshold})"
        )

        return filtered

    @staticmethod
    def to_trajectory_objects(data: List[Dict]) -> List[Trajectory]:
        """Convert raw dictionaries to Trajectory objects."""
        return [Trajectory.from_dict(d) for d in data]


class SkillLibraryLoader:
    """Load skill libraries from files."""

    @staticmethod
    def load(path: str) -> SkillLibrary:
        """
        Load skill library from JSON file.

        Args:
            path: Path to skill library JSON file

        Returns:
            SkillLibrary instance
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        library = SkillLibrary.from_dict(data)
        logger.info(
            f"Loaded skill library from {path}: "
            f"{len(library.functional)} functional, "
            f"{len(library.atomic)} atomic skills"
        )

        return library

    @staticmethod
    def load_skills_list(path: str) -> List[Dict]:
        """
        Load a list of skills from JSONL or JSON file.

        Args:
            path: Path to skills file

        Returns:
            List of skill dictionaries
        """
        path_obj = Path(path)

        if path_obj.suffix == ".jsonl":
            skills = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        skills.append(json.loads(line))
            return skills
        else:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
                return [data]

    @staticmethod
    def load_plan_library(path: str) -> Dict[str, Dict]:
        """
        Load plan library from JSON file.

        Args:
            path: Path to plan library file

        Returns:
            Dictionary mapping task to plan data
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Handle different formats
        if "plan" in data:
            return data["plan"]
        return data
