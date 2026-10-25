"""Cost metadata helpers used by cost-aware skill extraction experiments."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


INTERNAL_TOOL_NAMES = frozenset({"retrieve_tools", "retrieve_skills"})


def is_internal_tool(name: object) -> bool:
    """Return whether *name* is a framework action rather than a business tool."""
    if not isinstance(name, str):
        return True
    normalized = name.strip().split(".")[-1]
    return not normalized or normalized in INTERNAL_TOOL_NAMES or normalized.startswith("__")


class ToolAttributeLoader:
    """Load versioned tool attributes with a per-domain ``base.json`` fallback."""

    def __init__(self, attributes_dir: Path):
        self.attributes_dir = Path(attributes_dir)
        self._cache: Dict[tuple[str, str], Dict[str, Dict[str, Any]]] = {}

    def attributes_for(self, domain: str, version: object) -> Dict[str, Dict[str, Any]]:
        domain = str(domain or "").strip()
        version = str(version or "").strip()
        key = (domain, version)
        if key in self._cache:
            return self._cache[key]

        domain_dir = self.attributes_dir / domain
        version_path = domain_dir / f"{version}.json"
        base_path = domain_dir / "base.json"
        # Version fields override base fields while allowing absent version files.
        merged: Dict[str, Dict[str, Any]] = {}
        for path in (base_path, version_path):
            if not path.exists():
                continue
            with path.open("r", encoding="utf-8") as handle:
                raw = json.load(handle)
            if not isinstance(raw, dict):
                continue
            for tool_name, value in raw.items():
                if isinstance(value, dict):
                    merged[str(tool_name)] = dict(value)
                else:
                    merged[str(tool_name)] = {"cost": value}
        self._cache[key] = merged
        return merged

    def cost_for(self, domain: str, version: object, tool_name: object) -> Optional[float]:
        if is_internal_tool(tool_name):
            return None
        attrs = self.attributes_for(domain, version).get(str(tool_name).strip().split(".")[-1])
        if not isinstance(attrs, dict) or "cost" not in attrs:
            return None
        try:
            cost = float(attrs["cost"])
        except (TypeError, ValueError):
            return None
        return cost if cost > 0 else None


def build_cost_summary(events: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarize real business-action cost without inferring unobserved actions."""
    event_list = [event for event in events if isinstance(event, dict)]
    seen_calls: Counter[tuple[str, str]] = Counter()
    total_cost = 0.0
    successful_cost = 0.0
    error_cost = 0.0
    repeated_cost = 0.0
    missing_attribute_count = 0

    for event in event_list:
        cost = event.get("cost")
        if not isinstance(cost, (int, float)) or isinstance(cost, bool) or cost <= 0:
            missing_attribute_count += 1
            continue
        numeric_cost = float(cost)
        total_cost += numeric_cost
        if event.get("has_error"):
            error_cost += numeric_cost
        else:
            successful_cost += numeric_cost

        key = (str(event.get("tool_name", "")), json.dumps(event.get("arguments", {}), sort_keys=True, default=str))
        seen_calls[key] += 1
        if seen_calls[key] > 1:
            repeated_cost += numeric_cost

    return {
        "action_count": len(event_list),
        "total_cost": total_cost,
        "successful_action_cost": successful_cost,
        "error_action_cost": error_cost,
        "repeated_call_cost": repeated_cost,
        "missing_attribute_count": missing_attribute_count,
    }


COST_AWARE_GUIDANCE = """\

# Cost-aware skill guidance
Correctness is the primary objective. Preserve every necessary state change, required
precondition, confirmation, and task-required verification. Only when rollout evidence
shows that candidates are semantically equivalent with the same preconditions and state
effects may you prefer the lower-cost tool. Never describe a cheaper, merely similar tool
as a substitute. Capture cost guidance only when evidence demonstrates an avoidable
redundant call, an error/known-invalid retry, or an equivalent alternative. Prefer the
minimal correct workflow: authenticate once, obtain required IDs once, perform required
writes, and verify or report only when the task requires it.
"""
