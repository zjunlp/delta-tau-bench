"""
Rollout module - Core agent-user interaction loop.

Provides:
- Agent: LLM-based assistant for tool-calling conversations
- SimulatedUser: Multi-turn customer simulator with control tokens
- Pipeline: Fault-tolerant trajectory generation (single/multi-turn)
- CheckpointManager: Persistent save/resume for interrupted rollouts
"""

from dtau_bench.rollout.agent import Agent, AgentConfig
from dtau_bench.rollout.user import (
    SimulatedUser,
    ConversationState,
    STOP_TOKEN,
    TRANSFER_TOKEN,
    OUT_OF_SCOPE_TOKEN,
)
from dtau_bench.rollout.pipeline import (
    Pipeline,
    PipelineConfig,
    PipelineMode,
    TrajectoryResult,
    clear_reasoning_content,
    copy_messages_without_reasoning,
)
from dtau_bench.rollout.checkpoint import (
    CheckpointManager,
    CheckpointData,
    TrajectoryStatus,
    ErrorType,
)
from dtau_bench.rollout.reasoning import build_reasoning_control_extra_body

__all__ = [
    # Agent
    "Agent",
    "AgentConfig",
    # User
    "SimulatedUser",
    "ConversationState",
    "STOP_TOKEN",
    "TRANSFER_TOKEN",
    "OUT_OF_SCOPE_TOKEN",
    # Pipeline
    "Pipeline",
    "PipelineConfig",
    "PipelineMode",
    "TrajectoryResult",
    "clear_reasoning_content",
    "copy_messages_without_reasoning",
    # Checkpoint
    "CheckpointManager",
    "CheckpointData",
    "TrajectoryStatus",
    "ErrorType",
    # Reasoning
    "build_reasoning_control_extra_body",
]
