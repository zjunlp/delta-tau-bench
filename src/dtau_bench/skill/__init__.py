"""
Skill system — skill library management, loading, indexing, and runtime retrieval.

Provides:
- SkillLibraryLoader: Load skill library from disk (plan + skills + graph)
- SkillIndexer: Build/load embedding indices for skill retrieval
- SkillContextRuntime: End-to-end skill retrieval and prompt injection

Data Models:
- PlanningDocument: Planning skill (task → plan)
- SkillDocument: Functional/atomic skill
- SkillGraphState: Parsed skill graph state
- SkillContextArtifacts: Container for loaded artifacts
"""

from dtau_bench.skill.models import (
    PlanningDocument,
    SkillDocument,
    SkillGraphState,
    SkillContextArtifacts,
)
from dtau_bench.skill.loader import SkillLibraryLoader
from dtau_bench.skill.indexer import SkillIndexer
from dtau_bench.skill.runtime import SkillContextRuntime

__all__ = [
    # Models
    "PlanningDocument",
    "SkillDocument",
    "SkillGraphState",
    "SkillContextArtifacts",
    # Loader
    "SkillLibraryLoader",
    # Indexer
    "SkillIndexer",
    # Runtime
    "SkillContextRuntime",
]
