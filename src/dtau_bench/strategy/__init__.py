"""
Strategy layer - pluggable agent strategies.

Registered strategies:
- single_domain_full: all tools of one domain visible, no retrieval
- all_tools_iterative_rag: cross-domain iterative tool retrieval
- train_steward: skill context over one domain's tools
- steward: skill context + iterative tool retrieval + explorer subagent

``StewardBaseStrategy`` is the unregistered parent of ``steward``.
"""

# Base classes and data structures
from dtau_bench.strategy.base import (
    AgentStrategy,
    StrategyRuntimeContext,
    StrategyView,
    StrategySessionState,
    StrategyUpdate,
    StrategyToolInterceptionResult,
)

# Registry
from dtau_bench.strategy.registry import (
    register_strategy,
    build_strategy,
    list_strategies,
    get_strategy_class,
    is_registered,
    uses_all_tools,
    uses_skill_library,
    uses_explorer,
)

# Concrete strategies (importing triggers registration via decorators)
from dtau_bench.strategy.single_domain import (
    SingleDomainFullStrategy,
    # Helper functions (exported for use by other strategy modules)
    extract_initial_user_query,
    tool_name_from_info,
    filter_tools_info_by_names,
    merge_tool_names_preserve_order,
)
from dtau_bench.strategy.all_tools_rag import (
    AllToolsIterativeRAGStrategy,
    build_retrieve_tools_definition,
)
from dtau_bench.strategy.train_steward import (
    TrainStewardStrategy,
)
from dtau_bench.strategy.steward_base import (
    StewardBaseStrategy,
)
from dtau_bench.strategy.steward import (
    StewardStrategy,
    build_explore_tool_definition,
)

__all__ = [
    # Base
    "AgentStrategy",
    "StrategyRuntimeContext",
    "StrategyView",
    "StrategySessionState",
    "StrategyUpdate",
    "StrategyToolInterceptionResult",
    # Registry
    "register_strategy",
    "build_strategy",
    "list_strategies",
    "get_strategy_class",
    "is_registered",
    "uses_all_tools",
    "uses_skill_library",
    "uses_explorer",
    # Strategies
    "SingleDomainFullStrategy",
    "AllToolsIterativeRAGStrategy",
    "TrainStewardStrategy",
    "StewardBaseStrategy",
    "StewardStrategy",
    # Helpers
    "extract_initial_user_query",
    "tool_name_from_info",
    "filter_tools_info_by_names",
    "merge_tool_names_preserve_order",
    "build_retrieve_tools_definition",
    "build_explore_tool_definition",
]
