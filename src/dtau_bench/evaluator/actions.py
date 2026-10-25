"""
Agent action extraction and trajectory weight computation.

Every agent tool call is charged against the trajectory weight used by IDR, DAR
and SPE. Internal framework tools are excluded.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from dtau_bench.evaluator.models import ActionRecord, TrajectoryWeightSummary

logger = logging.getLogger(__name__)

# Framework tools (tool retrieval, skill retrieval, delegation to the explorer
# subagent) rather than domain tools, so they are not charged against the agent
# trajectory weight.
#
# Keep this in sync with the `virtual_tools` lists in configs/strategies/*.yaml
# and with `explorer.tool_name`.
INTERNAL_TOOL_NAMES = frozenset(
    {
        "retrieve_tools",
        "retrieve_skills",
        "explore_capabilities",
    }
)


def is_internal_tool_name(name: object) -> bool:
    """Check if a tool name is an internal framework tool (not a domain tool)."""
    if not isinstance(name, str):
        return False
    normalized = name.strip().split(".")[-1]
    return normalized in INTERNAL_TOOL_NAMES


def _parse_json_maybe(content: Any) -> Any:
    """Attempt to parse content as JSON. Returns parsed value or None."""
    if content is None:
        return None
    if isinstance(content, (dict, list)):
        return content
    if isinstance(content, str):
        text = content.strip()
        if not text:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None
    return None


def _tool_message_has_top_level_key(content: Any, key: str) -> bool:
    """Check if parsed tool message content has a specific top-level key."""
    parsed = _parse_json_maybe(content)
    return isinstance(parsed, dict) and key in parsed


def get_rollout_messages(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Return rollout conversation messages from a rollout record.

    Supports:
    - record["messages"] (legacy format)
    - record["conversation"]["messages"] (schema v2)
    """
    if not isinstance(record, dict):
        raise ValueError("rollout record must be a dict")

    messages = record.get("messages")
    if isinstance(messages, list):
        return messages

    conversation = record.get("conversation")
    if isinstance(conversation, dict):
        conversation_messages = conversation.get("messages")
        if isinstance(conversation_messages, list):
            return conversation_messages

    raise ValueError("Rollout record missing conversation messages")


def extract_action_records(messages: List[Dict[str, Any]]) -> List[ActionRecord]:
    """
    Extract ordered assistant tool calls paired with their tool responses.

    Args:
        messages: List of conversation messages (OpenAI chat format).

    Returns:
        Ordered list of ActionRecord objects, excluding internal framework tools.
    """
    # First pass: index tool responses by tool_call_id
    tool_response_by_id: Dict[str, Any] = {}
    tool_response_parsed_by_id: Dict[str, Any] = {}

    for msg in messages:
        if msg.get("role") != "tool":
            continue
        tool_call_id = msg.get("tool_call_id")
        if not tool_call_id:
            continue
        content = msg.get("content")
        tool_response_by_id[tool_call_id] = content
        tool_response_parsed_by_id[tool_call_id] = _parse_json_maybe(content)

    # Second pass: extract tool calls from assistant messages
    records: List[ActionRecord] = []
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        for tc in msg.get("tool_calls") or []:
            func = tc.get("function") or {}
            name = func.get("name", "")
            if not name:
                continue
            if is_internal_tool_name(name):
                continue

            args = func.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}

            call_id = tc.get("id")
            raw_result = tool_response_by_id.get(call_id)
            parsed_result = tool_response_parsed_by_id.get(call_id)

            # Strip namespace prefix (e.g., "domain.tool_name" -> "tool_name")
            normalized_name = name.split(".")[-1]

            records.append(
                ActionRecord(
                    index=len(records),
                    tool_name=normalized_name,
                    tool_call_id=call_id,
                    arguments=args if isinstance(args, dict) else {},
                    tool_result_raw=raw_result,
                    tool_result_parsed=parsed_result,
                    has_error=_tool_message_has_top_level_key(raw_result, "error"),
                )
            )

    return records


def compute_trajectory_weights(
    records: List[ActionRecord],
    tool_weights: Optional[Dict[str, float]] = None,
    weight_mode: Optional[str] = None,
) -> TrajectoryWeightSummary:
    """
    Compute the agent trajectory weight Ŵ for an action sequence.

    Args:
        records: ActionRecord list.
        tool_weights: Per-tool weight mapping (for cost mode). If None, unit mode
            is used (all weights = 1.0).
        weight_mode: Explicit mode label. If None, inferred from tool_weights.

    Returns:
        TrajectoryWeightSummary with the tool sequence and its total weight.
    """
    if weight_mode is None:
        weight_mode = "cost" if tool_weights else "unit"

    tool_weights = tool_weights or {}

    def _weight_for(tool_name: str) -> float:
        if not tool_weights:
            return 1.0
        weight = float(tool_weights.get(tool_name, 1.0))
        return weight if weight > 0 else 1.0

    return TrajectoryWeightSummary(
        agent_traj=[record.tool_name for record in records],
        agent_weight=sum(_weight_for(record.tool_name) for record in records),
        weight_mode=weight_mode,
    )
