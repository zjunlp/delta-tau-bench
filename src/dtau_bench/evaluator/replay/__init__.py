"""
Replay engine for dtau_bench evaluator.

Provides:
- GoldReplayExecutor: Executes gold trajectories using pre-generated step_param_codes
- AgentReplayExecutor: Replays agent tool calls with ID mapping
- DatabaseState: Captured DB state for diff comparison
- compute_state_diff / evaluate_gold_subset_relation: DB diffing logic
"""

from dtau_bench.evaluator.replay.executor import (
    DatabaseState,
    GoldReplayExecutor,
    GoldReplayResult,
)
from dtau_bench.evaluator.replay.agent_replay import (
    AgentReplayExecutor,
    AgentReplayResult,
)
from dtau_bench.evaluator.replay.diffing import (
    DiffMetadata,
    compute_state_diff,
    evaluate_gold_subset_relation,
)

__all__ = [
    "DatabaseState",
    "GoldReplayExecutor",
    "GoldReplayResult",
    "AgentReplayExecutor",
    "AgentReplayResult",
    "DiffMetadata",
    "compute_state_diff",
    "evaluate_gold_subset_relation",
]
