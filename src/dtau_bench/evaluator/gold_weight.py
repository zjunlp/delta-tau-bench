"""
Gold Weight Calculator for cross-version trajectory comparison.

Computes the minimum achievable gold trajectory weight when translating
a base trajectory to a target version, using:
- version_mapping: Tool availability and alternatives per version
- evolution_forest: Tree structure of tool evolution (parent/child relationships)
- tool_weights_by_version: Per-tool cost weights per version

Supports two weight modes:
- unit: All tools have weight 1.0 (weight = trajectory length)
- cost: Each tool has an independent weight from tool_attributes

Key concepts:
- A base tool might map to 1 tool (rename), N tools (split), or be removed
- The "optimal" translation minimizes total weight
- Uses recursive tree traversal to find min-weight path through evolution forest
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from dtau_bench.utils.config import get_data_root

logger = logging.getLogger(__name__)


@dataclass
class TrajTranslation:
    """Result of translating a trajectory to a target version."""

    translated_traj: List[str]
    """Translated tool names in the target version."""

    original_length: int
    """Length of the original trajectory."""

    translated_length: int
    """Length of the translated trajectory."""

    delta: int
    """Change in length (translated_length - original_length)."""

    translated_weight: float = 0.0
    """Total weight of the translated trajectory."""


class GoldWeightCalculator:
    """
    Computes optimal gold trajectory weight for a target version.

    Given a base trajectory (list of tool names), determines the minimum-weight
    equivalent trajectory in a target version by navigating the tool evolution tree.

    Usage:
        calculator = GoldWeightCalculator(version_mapping, evolution_forest, tool_weights_by_version)
        weight = calculator.get_optimal_gold_weight(base_traj=["tool_a", "tool_b"], target_version="v3")
    """

    def __init__(
        self,
        version_mapping: Dict[str, Any],
        evolution_forest: Optional[Dict[str, Any]] = None,
        tool_weights_by_version: Optional[Dict[str, Dict[str, float]]] = None,
        default_tool_weight: float = 1.0,
    ):
        """
        Initialize calculator with version mapping metadata.

        Args:
            version_mapping: Loaded from version_mapping.json. Contains:
                - versions: {base: {tools: [...]}, v1: {tools: [...]}, ...}
                - versions[vX].efficiency.alternatives: {tool: {optimal: [...]}}
                - tool_mappings: {tool: {base: [...], v1: [...], ...}}
            evolution_forest: Loaded from evolution_forests/<domain>_forest.json.
                Contains nodes with parent/child relationships.
            tool_weights_by_version: {version: {tool_name: cost_weight}}.
                Loaded from tool_attributes/<domain>/<version>.json.
            default_tool_weight: Default weight for tools not in tool_weights.
        """
        self.version_mapping = version_mapping
        self.tool_mappings = version_mapping.get("tool_mappings", {})
        self.versions = version_mapping.get("versions", {})
        self.evolution_forest = evolution_forest
        self.tool_weights_by_version = tool_weights_by_version or {}
        self.default_tool_weight = default_tool_weight

        # Precompute node index from evolution forest
        self._nodes: Dict[str, Dict[str, Any]] = {}
        self._version_node_ids: Dict[str, Set[str]] = {}
        self._original_to_root_ids: Dict[str, List[str]] = {}

        if self.evolution_forest:
            self._nodes = self.evolution_forest.get("nodes", {})

        # Build version -> available node_ids mapping
        for version, version_data in self.versions.items():
            node_ids: Set[str] = set()
            for tool in version_data.get("tools", []):
                if isinstance(tool, dict) and tool.get("node_id"):
                    node_ids.add(tool["node_id"])
            self._version_node_ids[version] = node_ids

        # Build original_tool_name -> root node_ids (generation 0 nodes)
        if self._nodes:
            for node_id, node in self._nodes.items():
                if node.get("generation") != 0:
                    continue
                tool_schema = node.get("tool_schema") or {}
                original_name = tool_schema.get("name")
                if not original_name:
                    continue
                self._original_to_root_ids.setdefault(original_name, []).append(node_id)

    def _tool_weight(self, tool_name: str, target_version: str) -> float:
        """Get weight for a tool in a specific version."""
        weight = float(
            self.tool_weights_by_version.get(target_version, {}).get(
                tool_name,
                self.default_tool_weight,
            )
        )
        return weight if weight > 0 else self.default_tool_weight

    def _fallback_translation_for_tool(
        self,
        tool_name: str,
        target_version: str,
    ) -> List[str]:
        """
        Fallback translation when evolution forest is not available.

        Checks version_mapping.alternatives and tool_mappings.
        """
        # Try version-level efficiency.alternatives
        version_data = self.versions.get(target_version, {})
        efficiency = version_data.get("efficiency", {})
        alternatives = efficiency.get("alternatives", {}).get(tool_name, {})

        # Prefer explicit optimal selection
        selected = alternatives.get("optimal") or alternatives.get("selected")
        if selected:
            # selected can be node_ids or tool names
            if all(s in self._nodes for s in selected):
                translated = []
                for node_id in selected:
                    node = self._nodes.get(node_id, {})
                    schema = node.get("tool_schema") or {}
                    if schema.get("name"):
                        translated.append(schema["name"])
                if translated:
                    return translated
            return [str(s) for s in selected]

        # Try top-level alternatives (some version_mapping formats)
        top_alternatives = self.version_mapping.get("alternatives", {}).get(tool_name, {})
        version_alternatives = top_alternatives.get(target_version, {})
        top_selected = version_alternatives.get("optimal") or version_alternatives.get("selected")
        if top_selected:
            if all(s in self._nodes for s in top_selected):
                translated = []
                for node_id in top_selected:
                    node = self._nodes.get(node_id, {})
                    schema = node.get("tool_schema") or {}
                    if schema.get("name"):
                        translated.append(schema["name"])
                if translated:
                    return translated
            return [str(s) for s in top_selected]

        # Try tool_mappings
        mapped = self.tool_mappings.get(tool_name, {}).get(target_version)
        if mapped:
            return mapped

        # Default: tool is unchanged
        return [tool_name]

    def _best_option_for_node(
        self,
        node_id: str,
        target_version: str,
        available_node_ids: Set[str],
        memo: Dict[str, Optional[Tuple[float, List[str]]]],
        stack: Optional[Set[str]] = None,
    ) -> Optional[Tuple[float, List[str]]]:
        """
        Compute best (min-weight) translation option for one node in a target version.

        Uses recursive tree traversal with memoization:
        - Option A: Keep current node if available in target version (weight = tool_weight)
        - Option B: Use all children (split semantics: weight = sum of child weights)

        Returns (weight, [tool_names]) or None if no feasible option.
        """
        if node_id in memo:
            return memo[node_id]

        if stack is None:
            stack = set()
        if node_id in stack:
            # Cycle detection
            memo[node_id] = None
            return None
        stack.add(node_id)

        node = self._nodes.get(node_id)
        if not node:
            memo[node_id] = None
            stack.discard(node_id)
            return None

        tool_schema = node.get("tool_schema") or {}
        tool_name = tool_schema.get("name", "")
        options: List[Tuple[float, List[str]]] = []

        # Option A: keep this node as one tool call
        if node_id in available_node_ids and tool_name:
            options.append((self._tool_weight(tool_name, target_version), [tool_name]))

        # Option B: use all children (conjunctive replacement for split)
        # Child order is part of a split's execution plan. Node identifiers
        # carry no dependency information; sorting them can put a consumer
        # before the tool that creates its input or required state.
        child_ids = list(node.get("child_ids") or [])
        if child_ids:
            child_total_weight = 0.0
            child_traj: List[str] = []
            feasible = True
            for child_id in child_ids:
                child_best = self._best_option_for_node(
                    child_id,
                    target_version,
                    available_node_ids,
                    memo,
                    stack,
                )
                if child_best is None:
                    feasible = False
                    break
                child_total_weight += child_best[0]
                child_traj.extend(child_best[1])

            if feasible:
                options.append((child_total_weight, child_traj))

        best = min(options, key=lambda x: x[0]) if options else None
        memo[node_id] = best
        stack.discard(node_id)
        return best

    def _translate_single_tool(
        self,
        tool_name: str,
        target_version: str,
    ) -> Tuple[List[str], float]:
        """
        Translate one base tool into a min-weight target-version trajectory fragment.

        Returns:
            (translated_tool_names, total_weight)
        """
        if not self._nodes or not self._version_node_ids.get(target_version):
            fallback = self._fallback_translation_for_tool(tool_name, target_version)
            return fallback, sum(self._tool_weight(t, target_version) for t in fallback)

        root_ids = self._original_to_root_ids.get(tool_name, [])
        available_node_ids = self._version_node_ids.get(target_version, set())

        best: Optional[Tuple[float, List[str]]] = None
        memo: Dict[str, Optional[Tuple[float, List[str]]]] = {}
        for root_id in root_ids:
            candidate = self._best_option_for_node(
                root_id,
                target_version,
                available_node_ids,
                memo,
            )
            if candidate is None:
                continue
            if best is None or candidate[0] < best[0]:
                best = candidate

        if best is not None:
            return best[1], best[0]

        # Fallback when no root node found in forest
        fallback = self._fallback_translation_for_tool(tool_name, target_version)
        return fallback, sum(self._tool_weight(t, target_version) for t in fallback)

    def translate_trajectory(
        self,
        base_traj: List[str],
        target_version: str,
    ) -> TrajTranslation:
        """
        Translate a base trajectory to the target version.

        For each tool in base_traj, finds the minimum-weight equivalent
        in the target version.

        Args:
            base_traj: List of tool names from base version.
            target_version: Target version identifier (e.g., "v1", "v2").

        Returns:
            TrajTranslation with translated trajectory and weight.
        """
        translated: List[str] = []
        total_weight = 0.0

        for tool in base_traj:
            translated_part, part_weight = self._translate_single_tool(tool, target_version)
            translated.extend(translated_part)
            total_weight += part_weight

        return TrajTranslation(
            translated_traj=translated,
            original_length=len(base_traj),
            translated_length=len(translated),
            delta=len(translated) - len(base_traj),
            translated_weight=total_weight,
        )

    def get_optimal_gold_weight(
        self,
        base_traj: List[str],
        target_version: str,
    ) -> float:
        """
        Return minimum achievable gold trajectory weight in target version.

        This is the primary API for cross-version comparison metrics.

        Args:
            base_traj: Base trajectory (list of tool names).
            target_version: Target version to compute weight for.

        Returns:
            Minimum total weight achievable in the target version.
        """
        return self.translate_trajectory(base_traj, target_version).translated_weight

    def get_optimal_gold_length(
        self,
        base_traj: List[str],
        target_version: str,
    ) -> int:
        """
        Return optimal trajectory length (number of tools) in target version.

        Equivalent to get_optimal_gold_weight with unit weights.
        """
        result = self.translate_trajectory(base_traj, target_version)
        return result.translated_length


def load_version_mapping(
    domain: str,
    data_dir: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """
    Load version mapping for a domain.

    Args:
        domain: Domain name (e.g., "AthleteContentDashboard").
        data_dir: Base data directory.

    Returns:
        Version mapping dictionary.

    Raises:
        FileNotFoundError: If version_mapping.json doesn't exist.
    """
    mapping_path = get_data_root(data_dir) / "meta" / "version_mapping" / f"{domain}.json"
    if not mapping_path.exists():
        raise FileNotFoundError(f"Version mapping not found: {mapping_path}")
    with open(mapping_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_evolution_forest(
    domain: str,
    data_dir: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """
    Load evolution forest for a domain.

    Args:
        domain: Domain name (e.g., "AthleteContentDashboard").
        data_dir: Base data directory.

    Returns:
        Evolution forest dictionary.

    Raises:
        FileNotFoundError: If forest file doesn't exist.
    """
    forest_path = get_data_root(data_dir) / "meta" / "evolution_forests" / f"{domain}_forest.json"
    if not forest_path.exists():
        raise FileNotFoundError(f"Evolution forest not found: {forest_path}")
    with open(forest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_tool_weights_for_domain(
    domain: str,
    data_dir: Optional[str | Path] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Load tool weights (cost field) from tool_attributes for a domain.

    Returns:
        Dict mapping version -> {tool_name: cost_weight}.
        E.g., {"v1": {"authorize_athlete": 1.0, "schedule_content_release": 2.0}, ...}
    """
    attrs_dir = get_data_root(data_dir) / "meta" / "tool_attributes" / domain
    if not attrs_dir.is_dir():
        logger.warning("Tool attributes directory not found: %s", attrs_dir)
        return {}

    weights_by_version: Dict[str, Dict[str, float]] = {}
    for attrs_file in sorted(attrs_dir.glob("*.json")):
        version = attrs_file.stem  # e.g., "v1", "v2", "base"
        try:
            with open(attrs_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to load tool attributes %s: %s", attrs_file, e)
            continue

        version_weights: Dict[str, float] = {}
        for tool_name, attrs in (data or {}).items():
            if not isinstance(attrs, dict):
                continue
            raw_cost = attrs.get("cost", 1)
            cost = float(raw_cost)
            if cost <= 0:
                cost = 1.0
            version_weights[str(tool_name)] = cost
        weights_by_version[version] = version_weights

    return weights_by_version


def build_gold_weight_calculator(
    domain: str,
    data_dir: Optional[str | Path] = None,
    *,
    include_cost_weights: bool = True,
) -> GoldWeightCalculator:
    """
    Convenience function to build a fully-configured GoldWeightCalculator for a domain.

    Args:
        domain: Domain name.
        data_dir: Base data directory.
        include_cost_weights: If True, load tool_weights_by_version for cost mode.

    Returns:
        Configured GoldWeightCalculator instance.
    """
    version_mapping = load_version_mapping(domain, data_dir)

    try:
        evolution_forest = load_evolution_forest(domain, data_dir)
    except FileNotFoundError:
        logger.warning("No evolution forest for domain %s, using fallback translation", domain)
        evolution_forest = None

    tool_weights_by_version = None
    if include_cost_weights:
        tool_weights_by_version = load_tool_weights_for_domain(domain, data_dir)

    return GoldWeightCalculator(
        version_mapping=version_mapping,
        evolution_forest=evolution_forest,
        tool_weights_by_version=tool_weights_by_version,
    )
