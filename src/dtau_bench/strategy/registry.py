"""
Strategy registry - decorator-based registration and factory function.

Usage:
    from dtau_bench.strategy.registry import register_strategy, build_strategy

    @register_strategy("my_strategy")
    class MyStrategy(AgentStrategy):
        ...

    strategy = build_strategy("my_strategy", config={...})
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Type

from dtau_bench.strategy.base import AgentStrategy

logger = logging.getLogger(__name__)


# Global registry mapping strategy names to their classes
_STRATEGY_REGISTRY: Dict[str, Type[AgentStrategy]] = {}


def register_strategy(name: str):
    """Decorator to register a strategy class.
    
    Args:
        name: The strategy name (used in CLI and config files).
        
    Example:
        @register_strategy("single_domain_full")
        class SingleDomainFullStrategy(AgentStrategy):
            ...
    """
    def decorator(cls: Type[AgentStrategy]) -> Type[AgentStrategy]:
        if name in _STRATEGY_REGISTRY:
            logger.warning(
                f"Strategy '{name}' is being re-registered: "
                f"{_STRATEGY_REGISTRY[name].__name__} → {cls.__name__}"
            )
        _STRATEGY_REGISTRY[name] = cls
        return cls
    return decorator


def build_strategy(
    name: str,
    config: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> AgentStrategy:
    """Factory function: instantiate a strategy by name.
    
    Args:
        name: Strategy name (must be registered).
        config: Strategy configuration dict (from YAML).
        **kwargs: Additional keyword arguments passed to the strategy constructor.
        
    Returns:
        An instantiated AgentStrategy.
        
    Raises:
        ValueError: If the strategy name is not registered.
    """
    normalized = name.strip().lower()
    if normalized not in _STRATEGY_REGISTRY:
        available = list(_STRATEGY_REGISTRY.keys())
        raise ValueError(
            f"Unknown strategy: '{name}'. Available strategies: {available}"
        )
    cls = _STRATEGY_REGISTRY[normalized]
    return cls(config=config or {}, **kwargs)


def list_strategies() -> List[str]:
    """List all registered strategy names."""
    return sorted(_STRATEGY_REGISTRY.keys())


def get_strategy_class(name: str) -> Optional[Type[AgentStrategy]]:
    """Get the strategy class for a given name (or None if not found)."""
    return _STRATEGY_REGISTRY.get(name.strip().lower())


def is_registered(name: str) -> bool:
    """Check if a strategy name is registered."""
    return name.strip().lower() in _STRATEGY_REGISTRY


# Strategy capability sets. Callers must test membership here instead of
# matching substrings of the strategy name.
ALL_TOOLS_STRATEGIES = frozenset({
    "all_tools_iterative_rag",
    "steward",
})

SKILL_LIBRARY_STRATEGIES = frozenset({
    "train_steward",
    "steward",
})

EXPLORER_STRATEGIES = frozenset({"steward"})


def uses_all_tools(name: str) -> bool:
    """Whether the strategy exposes tools from every domain."""
    return name.strip().lower() in ALL_TOOLS_STRATEGIES


def uses_skill_library(name: str) -> bool:
    """Whether the strategy needs a skill library and its retrieval index."""
    return name.strip().lower() in SKILL_LIBRARY_STRATEGIES


def uses_explorer(name: str) -> bool:
    """Whether the strategy runs the explorer subagent."""
    return name.strip().lower() in EXPLORER_STRATEGIES
