"""Explorer subagent package.

Implements a deep-research style capability explorer that the main agent can
invoke on demand.

Key invariants:
1. The explorer NEVER executes a real tool. It only reads skill documents and
   tool schemas, so it produces no database state change and does not affect
   the gold-replay / DB-diff evaluation.
2. The explorer only influences the main agent through the tools it commits in
   its final report("explored" != "committed").
3. The simulated user's private``user_system_prompt`` must NEVER be passed
   into the explorer input.
"""

from dtau_bench.explorer.types import (
    ExplorerConfig,
    ExploreRequest,
    ExplorerResult,
    ActionRecord,
    LLMUsageRecord,
)
from dtau_bench.explorer.backend import ExplorerBackend
from dtau_bench.explorer.agent import ExplorerAgent
from dtau_bench.explorer.store import ExploreTraceStore
from dtau_bench.explorer.report import ReportValidation, validate_report

__all__ = [
    "ExplorerConfig",
    "ExploreRequest",
    "ExplorerResult",
    "ActionRecord",
    "LLMUsageRecord",
    "ExplorerBackend",
    "ExplorerAgent",
    "ExploreTraceStore",
    "ReportValidation",
    "validate_report",
]
