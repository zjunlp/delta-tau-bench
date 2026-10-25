"""
Single-domain full strategy.

This is the simplest baseline strategy: all tools from the current domain
are made visible to the Agent without any retrieval or filtering.

Used for:
- Bootstrap rollouts (v1 train with all tools exposed)
- Baseline experiments (no RAG, no skills)
"""

from __future__ import annotations

from typing import Any, Dict, List

from dtau_bench.strategy.base import (
    AgentStrategy,
    StrategyRuntimeContext,
    StrategySessionState,
    StrategyView,
)
from dtau_bench.strategy.registry import register_strategy


# ============================================================================
# Helper Functions (shared with other strategies)
# ============================================================================


def extract_initial_user_query(messages: List[Dict[str, Any]]) -> str:
    """Extract the first non-empty user message content from a conversation."""
    for message in messages:
        if message.get("role") != "user":
            continue
        content = (message.get("content") or "").strip()
        if content:
            return content
    return ""


def tool_name_from_info(item: Dict[str, Any]) -> str:
    """Extract tool name from an OpenAI-format tool definition.
    
    Args:
        item: Tool definition dict (e.g., {"type": "function", "function": {"name": "..."}})
        
    Returns:
        Tool name string, or empty string if not found.
    """
    if not isinstance(item, dict):
        return ""
    function = item.get("function")
    if not isinstance(function, dict):
        return ""
    return str(function.get("name") or "").strip()


def filter_tools_info_by_names(
    all_tools_info: List[Dict[str, Any]],
    selected_names: List[str],
) -> List[Dict[str, Any]]:
    """Filter tool definitions by name, preserving the order of selected_names.
    
    Args:
        all_tools_info: All available tool definitions.
        selected_names: Names of tools to include (in desired order).
        
    Returns:
        List of tool definitions matching selected_names, ordered accordingly.
    """
    if not selected_names:
        return []
    name_to_item: Dict[str, Dict[str, Any]] = {}
    for item in all_tools_info:
        name = tool_name_from_info(item)
        if name and name not in name_to_item:
            name_to_item[name] = item
    ordered: List[Dict[str, Any]] = []
    seen: set = set()
    for name in selected_names:
        if not name or name in seen:
            continue
        item = name_to_item.get(name)
        if item is None:
            continue
        ordered.append(item)
        seen.add(name)
    return ordered


def merge_tool_names_preserve_order(
    existing_names: List[str],
    new_names: List[str],
) -> List[str]:
    """Merge two lists of tool names, preserving order and deduplicating."""
    merged = list(existing_names)
    seen = set(existing_names)
    for name in new_names:
        if not name or name in seen:
            continue
        merged.append(name)
        seen.add(name)
    return merged


# ============================================================================
# Base Full Strategy (shared by SingleDomain and AllTools full variants)
# ============================================================================


class _BaseFullStrategy(AgentStrategy):
    """Base for strategies that expose all given tools without filtering.
    
    All tools provided via all_tools_info are made visible to the Agent.
    No retrieval, no interception, no prepended messages.
    """

    def start_session(
        self,
        sample: Dict[str, Any],
        all_tools_info: List[Dict[str, Any]],
        runtime_context: StrategyRuntimeContext,
    ) -> StrategySessionState:
        """Initialize session: expose all tools."""
        query_text = extract_initial_user_query(sample.get("messages", []))
        visible_names = [
            name for name in (tool_name_from_info(item) for item in all_tools_info) if name
        ]
        state = StrategySessionState(
            sample_id=sample.get("id", ""),
            query_text=query_text,
            visible_tool_names=visible_names,
            visible_tools_info=list(all_tools_info),
            prepended_messages=[],
        )
        state.debug_history.append({
            "round": 0,
            "strategy": self.strategy_name,
            "visible_tool_count": len(visible_names),
            "visible_tool_names": visible_names,
        })
        return state

    def get_agent_view(
        self,
        state: StrategySessionState,
        conversation: List[Dict[str, Any]],
    ) -> StrategyView:
        """Return view with all visible tools (no filtering per turn)."""
        latest_debug = state.debug_history[-1] if state.debug_history else {}
        return StrategyView(
            visible_tools_info=state.visible_tools_info,
            prepended_messages=list(state.prepended_messages),
            metadata=latest_debug,
        )


# ============================================================================
# Registered Strategy Implementations
# ============================================================================


@register_strategy("single_domain_full")
class SingleDomainFullStrategy(_BaseFullStrategy):
    """Single-domain full tool exposure strategy.

    All tools from the current domain are exposed to the Agent.
    No retrieval, no filtering, no interception.
    """

    def __init__(self, config: Dict[str, Any] = None, **kwargs):
        super().__init__("single_domain_full", config or {})
