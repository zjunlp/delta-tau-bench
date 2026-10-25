"""Skill bootstrap pipeline - extract skills from successful rollout trajectories.

Four-stage pipeline:
  Stage 1 (Dataset):     Build skill context-ready dataset from rollout + eval artifacts
  Stage 2 (Extraction):  Plan extraction + Atomic skill extraction + clustering/merge/filter
  Stage 3 (Tool Skills): Tool skill extraction from evidence
  Stage 4 (Graph):       Skill graph construction (planning → atomic → tool)
"""

from .pipeline import BootstrapPipeline, BootstrapConfig

__all__ = ["BootstrapPipeline", "BootstrapConfig"]
