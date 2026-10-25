"""Skill evolution pipeline - evolve skill libraries across tool versions.

This package implements the 6-step evolution pipeline:
  Step 1: Tool Feedback Summary (compress long trajectories)
  Step 2: Direct Proposal Extraction (per-session LLM proposals)
  Step 3: Tool Skill Finalize (merge proposals per tool)
  Step 4: Atomic Skill Finalize (merge proposals + clustering)
  Step 5: Planning Propagation (graph-driven plan updates)
  Step 6: Rebuild Skill Graph + Active/Inactive Partition
"""

from .pipeline import EvolveConfig, EvolvePipeline

__all__ = ["EvolveConfig", "EvolvePipeline"]
