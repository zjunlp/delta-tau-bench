"""Reasoning-mode control passed to the LLM endpoint."""

from __future__ import annotations

from typing import Any, Dict, Optional


def build_reasoning_control_extra_body(
    enable_thinking: bool,
) -> Optional[Dict[str, Any]]:
    """Build the ``extra_body`` payload that turns reasoning mode on.

    Both keys are sent because servers disagree on where the switch lives:
    vLLM reads ``chat_template_kwargs``, other gateways read the top-level
    ``enable_thinking``. An unknown key is ignored.

    Args:
        enable_thinking: Whether reasoning mode is requested.

    Returns:
        The extra_body dict, or None when reasoning is off.
    """
    if not enable_thinking:
        return None
    return {
        "enable_thinking": True,
        "chat_template_kwargs": {"enable_thinking": True},
    }
