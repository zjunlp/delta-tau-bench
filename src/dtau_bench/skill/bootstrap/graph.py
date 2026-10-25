"""Stage 4: Skill Graph Construction.

Builds a three-layer skill graph: Planning → Atomic → Tool.

  - Planning → Atomic: an edge exists when the atomic skill's tools are
    sufficiently covered by the tools referenced in the plan's steps,

        coverage = |plan_tools ∩ atomic_tools| / |atomic_tools|

    thresholded by ``plan_atomic_coverage_threshold``. Embedding text
    similarity is recorded as the edge weight but does not decide the edge.

  - Atomic → Tool: direct link from the atomic skill's ``tools`` field.
"""

from __future__ import annotations

import json
import logging
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Read JSONL file."""
    items: List[Dict[str, Any]] = []
    if not path.exists():
        return items
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def _normalize_skill_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """Unwrap a skill JSONL row into a flat dict with name/tools/text.

    Rows are written as ``{"embedding_text": ..., "skill": {...}, ...}``, so
    reading ``row["name"]`` / ``row["tools"]`` directly yields ``None`` / ``[]``
    and silently produces nameless nodes and zero edges. Mirrors the runtime
    loader, which uses ``row.get("skill", row)``.
    """
    if not isinstance(row, dict):
        return {"name": "", "tools": [], "content": "", "document": "", "embedding_text": "", "raw": {}}

    skill = row.get("skill", row)
    if not isinstance(skill, dict):
        skill = {}

    tools = [
        str(name).strip()
        for name in (skill.get("tools") or [])
        if str(name or "").strip()
    ]
    name = str(skill.get("name", "") or "").strip()
    content = str(skill.get("content", "") or "")
    document = str(skill.get("document", "") or "")
    embedding_text = str(row.get("embedding_text", "") or "")
    if not embedding_text:
        embedding_text = f"{name}: {content}" if content else name

    return {
        "name": name,
        "tools": tools,
        "content": content,
        "document": document,
        "embedding_text": embedding_text,
        "raw": skill,
    }


def _cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    """Compute cosine similarity between two vectors."""
    numerator = sum(float(a) * float(b) for a, b in zip(vec_a, vec_b))
    denom_a = math.sqrt(sum(float(a) * float(a) for a in vec_a))
    denom_b = math.sqrt(sum(float(b) * float(b) for b in vec_b))
    if denom_a == 0 or denom_b == 0:
        return 0.0
    return numerator / (denom_a * denom_b)


def _normalize_text(text: str) -> str:
    """Normalize whitespace in text."""
    return " ".join(part.strip() for part in re.split(r"\s+", text) if part.strip())


def _read_plan_steps(plan: str) -> List[Tuple[str, List[str]]]:
    """Parse plan text into (step_text, tool_names) pairs."""
    steps: List[Tuple[str, List[str]]] = []
    for raw_line in plan.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        lowered = line.lower()
        if "# step" not in lowered and not re.match(r"^\d+[.)]", line):
            continue
        # Strip step prefix
        text = re.sub(r"^\s*(?:#\s*)?step\s*\d+\s*:\s*", "", line, flags=re.I)
        text = re.sub(r"^\s*\d+[.)]\s*", "", text).strip()
        # Extract key APIs
        api_match = re.search(r"key APIs used:\s*(.+)$", line, flags=re.I)
        tools: List[str] = []
        if api_match:
            tools = [part.strip() for part in api_match.group(1).split(",") if part.strip()]
        steps.append((text, tools))
    return steps


def _build_node(
    node_id: str,
    node_type: str,
    name: str,
    tools: List[str],
    source_path: str,
    raw_ref: Dict[str, Any],
) -> Dict[str, Any]:
    """Build a graph node."""
    return {
        "node_id": node_id,
        "node_type": node_type,
        "name": name,
        "tools": list(tools),
        "source_path": source_path,
        "raw_ref": raw_ref,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Embedding
# ─────────────────────────────────────────────────────────────────────────────


def _embed_texts(
    texts: List[str],
    embedding_api_base: str,
    embedding_api_key: str,
    embedding_model: str,
    batch_size: int,
) -> List[List[float]]:
    """Generate embeddings without sending the whole graph in one request."""
    from dtau_bench.rag.config import RAGConfig
    from dtau_bench.rag.indexer import ToolIndexer

    config = RAGConfig(
        embedding_model=embedding_model,
        embedding_backend="openai_http",
        embedding_api_base=embedding_api_base,
        embedding_api_key=embedding_api_key,
    )
    indexer = ToolIndexer(
        config=config,
        embedding_batch_size=batch_size,
    )
    embeddings: List[List[float]] = []
    for start in range(0, len(texts), indexer.embedding_batch_size):
        batch = texts[start:start + indexer.embedding_batch_size]
        embeddings.extend(indexer._get_embeddings_batch(batch))
    return embeddings


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def build_skill_graph(
    skill_library_dir: Path,
    output_dir: Path,
    embedding_api_base: str,
    embedding_api_key: str,
    embedding_model: str,
    plan_atomic_coverage_threshold: float = 0.5,
    embedding_batch_size: int = 64,
) -> Dict[str, Any]:
    """Build the skill graph from library artifacts.

    Reads plan_library.json, atomic_skills.prepared.jsonl, and
    tool_skills.prepared.jsonl from skill_library_dir, then constructs
    a three-layer graph.

    Args:
        skill_library_dir: Directory containing the assembled skill library.
        output_dir: Directory to write skill_graph.json.
        embedding_api_base: Embedding API base URL.
        embedding_api_key: Embedding API key.
        embedding_model: Embedding model name.
        plan_atomic_coverage_threshold: Minimum fraction of an atomic skill's
            tools that must appear in the plan's steps for an edge to exist.
        embedding_batch_size: Maximum graph texts sent in one embedding request.

    Returns:
        Statistics dict.

    Raises:
        ValueError: If the resulting graph is degenerate (no edges at all, or
            no atomic→tool edges while both layers are non-empty). Such a
            graph is silently useless at inference time, so it must fail loudly
            rather than be written out.
    """
    # --- Load artifacts ---
    plan_library_path = skill_library_dir / "plan_library.json"
    if not plan_library_path.exists():
        raise FileNotFoundError(f"plan_library.json not found in {skill_library_dir}")

    with plan_library_path.open("r", encoding="utf-8") as f:
        plan_library_raw = json.load(f)

    # plan_library.json stores {"plan": {task: plan_text, ...}}
    plan_dict = plan_library_raw.get("plan", {})
    if not isinstance(plan_dict, dict):
        plan_dict = {}

    # Build plan entries
    plans: List[Dict[str, Any]] = []
    for task, plan_text in plan_dict.items():
        plans.append({"task": task, "plan": plan_text})

    # Load atomic skills (rows are nested under "skill", see _normalize_skill_row)
    atomic_rows = [_normalize_skill_row(row) for row in _read_jsonl(skill_library_dir / "atomic_skills.prepared.jsonl")]
    # Load tool skills
    tool_skills = [_normalize_skill_row(row) for row in _read_jsonl(skill_library_dir / "tool_skills.prepared.jsonl")]

    if not atomic_rows:
        logger.warning("No atomic skills found; graph will have limited structure")
    if not tool_skills:
        logger.warning("No tool skills found; graph will have no tool layer")

    # --- Build plan step payloads for embedding ---
    plan_step_payloads: List[Dict[str, Any]] = []
    for plan_item in plans:
        steps = _read_plan_steps(str(plan_item.get("plan", "")))
        embedding_text = "\n".join(
            _normalize_text(step_text) for step_text, _ in steps if _normalize_text(step_text)
        ) or str(plan_item.get("task", ""))
        plan_step_payloads.append({
            "task": plan_item.get("task", ""),
            "steps": steps,
            "embedding_text": embedding_text,
        })

    # --- Build atomic embedding texts ---
    atomic_embedding_texts = [item["embedding_text"] for item in atomic_rows]

    # --- Compute embeddings ---
    all_texts = (
        [p["embedding_text"] for p in plan_step_payloads]
        + atomic_embedding_texts
    )

    embeddings: List[List[float]] = []
    if all_texts:
        logger.info(f"Computing embeddings for {len(all_texts)} texts...")
        embeddings = _embed_texts(
            all_texts,
            embedding_api_base=embedding_api_base,
            embedding_api_key=embedding_api_key,
            embedding_model=embedding_model,
            batch_size=embedding_batch_size,
        )

    num_plans = len(plan_step_payloads)
    plan_embeddings = embeddings[:num_plans]
    atomic_embeddings = embeddings[num_plans:]

    # --- Build nodes ---
    nodes: List[Dict[str, Any]] = []

    for idx, plan_item in enumerate(plans):
        nodes.append(_build_node(
            node_id=f"planning::{idx}",
            node_type="planning",
            name=str(plan_item.get("task", "")),
            tools=[],
            source_path="plan_library.json",
            raw_ref={"task": plan_item.get("task", "")},
        ))

    for idx, atomic_item in enumerate(atomic_rows):
        nodes.append(_build_node(
            node_id=f"atomic::{idx}",
            node_type="atomic",
            name=atomic_item["name"],
            tools=list(atomic_item["tools"]),
            source_path="atomic_skills.prepared.jsonl",
            raw_ref={"name": atomic_item["name"]},
        ))

    for idx, tool_item in enumerate(tool_skills):
        # A tool skill documents exactly one tool; fall back to its own name.
        tool_node_tools = list(tool_item["tools"]) or (
            [tool_item["name"]] if tool_item["name"] else []
        )
        nodes.append(_build_node(
            node_id=f"tool::{idx}",
            node_type="tool",
            name=tool_item["name"],
            tools=tool_node_tools,
            source_path="tool_skills.prepared.jsonl",
            raw_ref={"name": tool_item["name"]},
        ))

    # --- Build edges ---
    edges: List[Dict[str, Any]] = []

    # Name-to-node-id maps (skip unnamed rows: they cannot be linked)
    atomic_name_to_node_id = {
        item["name"]: f"atomic::{idx}"
        for idx, item in enumerate(atomic_rows)
        if item["name"]
    }
    tool_name_to_node_id = {
        item["name"]: f"tool::{idx}"
        for idx, item in enumerate(tool_skills)
        if item["name"]
    }

    # Planning → Atomic edges: does this plan provide the tools this atomic needs?
    # Text similarity is recorded as the edge weight, not as an edge criterion.
    for plan_idx, plan_payload in enumerate(plan_step_payloads):
        plan_name = str(plans[plan_idx].get("task", ""))
        # Collect tool names from plan steps
        step_tool_names = {
            tool_name
            for _, tool_names in plan_payload["steps"]
            for tool_name in tool_names
        }

        for atomic_idx, atomic_item in enumerate(atomic_rows):
            atomic_name = atomic_item["name"]
            atomic_tools = set(atomic_item["tools"])
            if not atomic_name or not atomic_tools or not step_tool_names:
                continue

            # Tool coverage: fraction of the atomic's tools provided by this plan.
            coverage = len(step_tool_names & atomic_tools) / len(atomic_tools)
            if coverage < plan_atomic_coverage_threshold:
                continue

            # Text similarity is kept as the edge weight for ranking/pruning.
            text_score = 0.0
            if plan_idx < len(plan_embeddings) and atomic_idx < len(atomic_embeddings):
                text_score = _cosine_similarity(
                    plan_embeddings[plan_idx], atomic_embeddings[atomic_idx]
                )

            edges.append({
                "source": f"planning::{plan_idx}",
                "target": atomic_name_to_node_id[atomic_name],
                "edge_type": "plan_to_atomic",
                "score": round(text_score, 4),
                "tool_coverage_score": round(coverage, 4),
                "text_similarity_score": round(text_score, 4),
                "planning_name": plan_name,
                "atomic_name": atomic_name,
            })

    # Atomic → Tool edges (direct tool name linkage)
    for atomic_idx, atomic_item in enumerate(atomic_rows):
        atomic_name = atomic_item["name"]
        if not atomic_name:
            continue
        for tool_name in atomic_item["tools"]:
            if tool_name not in tool_name_to_node_id:
                continue
            edges.append({
                "source": f"atomic::{atomic_idx}",
                "target": tool_name_to_node_id[tool_name],
                "edge_type": "atomic_to_tool",
                "atomic_name": atomic_name,
                "tool_name": tool_name,
            })

    # --- Write output ---
    output_dir.mkdir(parents=True, exist_ok=True)
    graph_payload = {
        "metadata": {
            "skill_library_dir": str(skill_library_dir),
            "embedding_model": embedding_model,
            "plan_atomic_edge_rule": "tool_coverage",
            "plan_atomic_coverage_threshold": plan_atomic_coverage_threshold,
            "edge_score_field": "text_similarity (weight only, not a criterion)",
        },
        "nodes": nodes,
        "edges": edges,
    }

    # Summary stats
    planning_nodes = sum(1 for n in nodes if n["node_type"] == "planning")
    atomic_nodes = sum(1 for n in nodes if n["node_type"] == "atomic")
    tool_nodes = sum(1 for n in nodes if n["node_type"] == "tool")
    plan_to_atomic_edges = sum(1 for e in edges if e["edge_type"] == "plan_to_atomic")
    atomic_to_tool_edges = sum(1 for e in edges if e["edge_type"] == "atomic_to_tool")

    unnamed_atomic = sum(1 for item in atomic_rows if not item["name"])
    unnamed_tool = sum(1 for item in tool_skills if not item["name"])

    # ── Fail loudly on a degenerate graph ──
    # A graph without edges is silently useless at inference time: graph
    # prioritization becomes a no-op and the explorer's expand_graph action can
    # never return anything. Writing it out and returning success is what let an
    # earlier field-nesting bug go unnoticed across every experiment.
    problems: List[str] = []
    if unnamed_atomic or unnamed_tool:
        problems.append(
            f"{unnamed_atomic}/{len(atomic_rows)} atomic and "
            f"{unnamed_tool}/{len(tool_skills)} tool rows have no name "
            "(skill row nesting changed?)"
        )
    if atomic_rows and tool_skills and atomic_to_tool_edges == 0:
        problems.append(
            "0 atomic→tool edges while both layers are non-empty; this link is a "
            "plain tool-name match, so 0 means the names never line up"
        )
    if plans and atomic_rows and not edges:
        problems.append("0 edges in total")
    if problems:
        raise ValueError(
            "Degenerate skill graph, refusing to write: "
            + "; ".join(problems)
            + f" [planning={planning_nodes}, atomic={atomic_nodes}, tool={tool_nodes}]"
        )

    with (output_dir / "skill_graph.json").open("w", encoding="utf-8") as f:
        json.dump(graph_payload, f, ensure_ascii=False, indent=2)

    # Degree stats make an unhealthy graph (too sparse / hub-dominated) visible
    # without having to reload the whole payload.
    atomic_out_degree: Dict[str, int] = {}
    plan_out_degree: Dict[str, int] = {}
    for edge in edges:
        if edge["edge_type"] == "plan_to_atomic":
            plan_out_degree[edge["source"]] = plan_out_degree.get(edge["source"], 0) + 1
        elif edge["edge_type"] == "atomic_to_tool":
            atomic_out_degree[edge["source"]] = atomic_out_degree.get(edge["source"], 0) + 1
    linked_atomics = len({e["target"] for e in edges if e["edge_type"] == "plan_to_atomic"})

    stats = {
        "planning_nodes": planning_nodes,
        "atomic_nodes": atomic_nodes,
        "tool_nodes": tool_nodes,
        "total_nodes": len(nodes),
        "plan_to_atomic_edges": plan_to_atomic_edges,
        "atomic_to_tool_edges": atomic_to_tool_edges,
        "total_edges": len(edges),
        "plan_atomic_coverage_threshold": plan_atomic_coverage_threshold,
        "plans_with_atomic_edges": len(plan_out_degree),
        "atomics_linked_from_plans": linked_atomics,
        "avg_atomics_per_plan": (
            round(plan_to_atomic_edges / planning_nodes, 3) if planning_nodes else 0.0
        ),
        "max_atomics_per_plan": max(plan_out_degree.values()) if plan_out_degree else 0,
        "avg_tools_per_atomic": (
            round(atomic_to_tool_edges / atomic_nodes, 3) if atomic_nodes else 0.0
        ),
    }

    # Write summary separately
    with (output_dir / "skill_graph_summary.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    logger.info(
        f"Skill graph: {planning_nodes} planning, {atomic_nodes} atomic, "
        f"{tool_nodes} tool nodes; {plan_to_atomic_edges} plan→atomic "
        f"(coverage>={plan_atomic_coverage_threshold}, "
        f"{stats['avg_atomics_per_plan']} per plan, "
        f"{len(plan_out_degree)}/{planning_nodes} plans linked), "
        f"{atomic_to_tool_edges} atomic→tool edges"
    )

    return stats
