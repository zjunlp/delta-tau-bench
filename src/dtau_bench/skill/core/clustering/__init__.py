"""Clustering module for skill deduplication and merging."""

from .embedding import EmbeddingService
from .dbscan import DBSCANClustering
from .hierarchical import CompleteLinkageClustering
from .merger import SkillMerger

__all__ = [
    "EmbeddingService",
    "DBSCANClustering",
    "CompleteLinkageClustering",
    "SkillMerger",
]
