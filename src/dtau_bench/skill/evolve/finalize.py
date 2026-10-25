"""Step 3 (Tool Skill Finalize) and Step 4 (Atomic Skill Finalize).

Step 3: For each tool with modify/add proposals, finalize into a single skill.
Step 4: For each atomic with modify proposals, finalize; cluster new adds.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from .proposals import (
    EvolutionArtifacts,
    _extract_tagged_json,
    _read_jsonl,
    _run_batched_tasks,
    _sort_proposals,
    _write_jsonl,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Prompts
# ─────────────────────────────────────────────────────────────────────────────


TOOL_FINALIZE_PROMPT = """You are finalizing one tool skill from the current canonical version plus a set of modify/add proposals.

Output exactly one JSON object inside <skill>...</skill> with keys:
- name
- document
- content

Constraints:
- This final skill must describe exactly one tool.
- Do not output a tools field.
- Merge overlapping guidance and keep only reusable, tool-centric advice.
- The final skill should primarily capture agent-facing pitfalls, prerequisites, verification steps, confirmation semantics, ordering constraints, argument/format traps, and recovery behavior.
- Prefer compact, high-value guidance that helps a future agent avoid mistakes.
- Remove or compress generic API-description material unless it is needed to explain a non-obvious constraint.
- Do not spend much space restating obvious inputs/outputs or simple happy-path usage.
- If the evidence does not justify a meaningful update, preserve the old skill content as closely as possible.
"""


ATOMIC_FINALIZE_PROMPT = """You are finalizing one atomic skill from the current canonical version plus a set of modify proposals.

Output exactly one JSON object inside <skill>...</skill> with keys:
- name
- document
- content
- tools

Constraints:
- Keep this as one reusable atomic capability.
- The final skill should emphasize reusable multi-step subtask guidance: sequencing, branching, validation, retry, recovery, and cross-tool coordination.
- Prefer skills that help with harder or failure-prone subtasks, not trivial one-step actions.
- Remove or compress task-specific narration and simple happy-path detail unless it carries a non-obvious reusable lesson.
- Preserve important non-obvious workflow constraints even if they appeared only in a small number of sessions.
- Do not delete core useful guidance from the old skill unless the proposals clearly correct it.
- The tools field must remain accurate and explicit.
"""


def _cost_aware_prompt(prompt: str, config: Any) -> str:
    if not config.cost_aware_skills:
        return prompt
    from ..cost import COST_AWARE_GUIDANCE

    return prompt + COST_AWARE_GUIDANCE


# ─────────────────────────────────────────────────────────────────────────────
# Parsers
# ─────────────────────────────────────────────────────────────────────────────


def _extract_skill_payload(text: str) -> Optional[Dict[str, Any]]:
    """Extract a finalized skill from <skill>...</skill>."""
    payload = _extract_tagged_json(text, "skill")
    if payload is None:
        return None
    if not isinstance(payload.get("name"), str) or not payload["name"].strip():
        return None
    if not isinstance(payload.get("document"), str):
        return None
    if not isinstance(payload.get("content"), str):
        return None
    return payload


def _extract_atomic_skill_payload(text: str) -> Optional[Dict[str, Any]]:
    """Extract a finalized atomic skill (with tools field)."""
    payload = _extract_skill_payload(text)
    if payload is None:
        return None
    if not isinstance(payload.get("tools"), list):
        return None
    return payload


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Tool Skill Finalize
# ─────────────────────────────────────────────────────────────────────────────


async def run_tool_skill_finalize(
    *,
    tool_proposals_path: Path,
    trace_path: Path,
    output_dir: Path,
    artifacts: EvolutionArtifacts,
    llm: Any,
    config: Any,
) -> List[Dict[str, Any]]:
    """Step 3: Finalize tool skills from proposals.

    Groups proposals by tool_name, then for each:
    - Existing tools with modify proposals: LLM finalize
    - New tools with add proposals: merge facets → LLM finalize

    Returns finalized tool skill rows.
    """
    tool_updates = _read_jsonl(tool_proposals_path)

    # Group by tool name and option
    grouped_existing: Dict[str, List[Dict[str, Any]]] = {}
    grouped_new: Dict[str, List[Dict[str, Any]]] = {}
    for update in tool_updates:
        tool_name = str(update.get("tool_name", "")).strip()
        if not tool_name:
            continue
        if update.get("option") == "modify":
            grouped_existing.setdefault(tool_name, []).append(update)
        elif update.get("option") == "add":
            grouped_new.setdefault(tool_name, []).append(update)

    finalized: List[Dict[str, Any]] = []
    proposal_trace: List[Dict[str, Any]] = []
    semaphore = asyncio.Semaphore(config.max_concurrent)

    from ..core.tool_skills import ToolSkillMerger
    merger = ToolSkillMerger(
        llm=llm,
        verbose=False,
        cost_aware=config.cost_aware_skills,
    )

    async def finalize_existing(
        tool_name: str, proposals: List[Dict[str, Any]]
    ) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
        existing = artifacts.tool_skills.get(tool_name)
        if existing is None:
            return None
        selected = _sort_proposals(proposals)[:config.max_modify_proposals_per_skill]
        trace = {
            "tool_name": tool_name,
            "mode": "modify",
            "proposal_count": len(proposals),
            "selected_proposal_count": len(selected),
            "selected_query_ids": [str(item.get("query_id", "")) for item in selected],
        }
        async with semaphore:
            response = await llm.ainvoke(
                messages=[
                    ("system", _cost_aware_prompt(TOOL_FINALIZE_PROMPT, config)),
                    ("human", json.dumps(
                        {
                            "tool_name": tool_name,
                            "existing_skill": existing,
                            "proposals": selected,
                        },
                        ensure_ascii=False,
                        indent=2,
                    )),
                ],
                regex_extractor=_extract_skill_payload,
            )
        skill = _extract_skill_payload(response)
        if skill is None:
            return None
        skill["name"] = tool_name
        skill["tools"] = [tool_name]
        row = {
            "option": "add",
            "skill": skill,
            "source_metadata": {
                "tool_name": tool_name,
                "facet_count": len(selected),
                "evolution_mode": "modify_finalize",
            },
            "source_task_ids": sorted({
                str(item.get("query_id", "")).strip()
                for item in selected
                if str(item.get("query_id", "")).strip()
            }),
            "merge_source_count": len(selected),
        }
        return row, trace

    async def finalize_new(
        tool_name: str, proposals: List[Dict[str, Any]]
    ) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
        if tool_name in artifacts.tool_skills:
            return None
        selected = _sort_proposals(proposals)[:config.max_modify_proposals_per_skill]
        trace = {
            "tool_name": tool_name,
            "mode": "add",
            "proposal_count": len(proposals),
            "selected_proposal_count": len(selected),
            "selected_query_ids": [str(item.get("query_id", "")) for item in selected],
        }
        selected_facets = []
        for item in selected:
            skill = item.get("skill")
            if not isinstance(skill, dict):
                continue
            selected_facets.append({
                "skill": {
                    "name": tool_name,
                    "document": skill.get("document", ""),
                    "content": skill.get("content", ""),
                    "tools": [tool_name],
                },
                "source_metadata": {"tool_name": tool_name},
                "source_task_ids": [item.get("query_id")],
            })
        if not selected_facets:
            return None
        async with semaphore:
            merged_skill = await merger.merge_tool(
                tool_name=tool_name,
                selected_facets=selected_facets,
            )
        if merged_skill is None:
            return None
        row = {
            "option": "add",
            "skill": merged_skill,
            "source_metadata": {
                "tool_name": tool_name,
                "facet_count": len(selected_facets),
                "evolution_mode": "add_finalize",
            },
            "source_task_ids": sorted({
                str(item.get("query_id", "")).strip()
                for item in selected
                if str(item.get("query_id", "")).strip()
            }),
            "merge_source_count": len(selected_facets),
        }
        return row, trace

    # Process existing tool skill modifications
    existing_tasks = [
        finalize_existing(tool_name, proposals)
        for tool_name, proposals in sorted(grouped_existing.items())
    ]
    for result in await _run_batched_tasks(
        existing_tasks,
        batch_size=config.batch_size,
        max_concurrent=config.max_concurrent,
        show_progress=not config.quiet,
        progress_desc="Tool Finalize Modify",
        progress_unit="tool",
    ):
        if isinstance(result, Exception) or result is None:
            continue
        row, trace = result
        finalized.append(row)
        proposal_trace.append(trace)

    # Process new tool skill adds
    new_tasks = [
        finalize_new(tool_name, proposals)
        for tool_name, proposals in sorted(grouped_new.items())
    ]
    for result in await _run_batched_tasks(
        new_tasks,
        batch_size=config.batch_size,
        max_concurrent=config.max_concurrent,
        show_progress=not config.quiet,
        progress_desc="Tool Finalize Add",
        progress_unit="tool",
    ):
        if isinstance(result, Exception) or result is None:
            continue
        row, trace = result
        finalized.append(row)
        proposal_trace.append(trace)

    # Write outputs
    _write_jsonl(trace_path, proposal_trace)
    _write_jsonl(output_dir / "finalized_tool_rows.jsonl", finalized)

    logger.info(f"  Finalized {len(finalized)} tool skills ({len(grouped_existing)} modify + {len(grouped_new)} add groups)")
    return finalized


# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Atomic Skill Finalize
# ─────────────────────────────────────────────────────────────────────────────


async def _filter_atomic_candidates(
    *,
    llm: Any,
    benchmark: str,
    candidates: Sequence[Dict[str, Any]],
    batch_size: int,
    max_concurrent: int,
    skip_general_filter: bool,
    show_progress: bool = False,
) -> List[Dict[str, Any]]:
    """Run the TwoStageFilterPipeline on atomic skill candidates."""
    if not candidates:
        return []
    from ..core.filtering.pipeline import TwoStageFilterPipeline

    pipeline = TwoStageFilterPipeline(
        llm=llm,
        benchmark=benchmark,
        tool_schemas={},
        skip_stage1=skip_general_filter,
        skip_stage2=True,
        verbose=False,
    )
    passed = await pipeline.filter(
        list(candidates),
        batch_size=batch_size,
        max_concurrent=max_concurrent,
        show_progress=show_progress,
    )
    return list(passed)


async def _merge_new_atomic_adds(
    *,
    llm: Any,
    add_updates: Sequence[Dict[str, Any]],
    config: Any,
) -> List[Dict[str, Any]]:
    """Cluster and merge new atomic add proposals using embedding similarity."""
    if not add_updates:
        return []

    from ..core.clustering.embedding import EmbeddingService
    from ..core.clustering.hierarchical import CompleteLinkageClustering
    from ..core.clustering.merger import SkillMerger
    from ..core.extraction.skill_extractor import prepare_skills_for_clustering

    # Build raw items from add proposals
    raw_items: List[Dict[str, Any]] = []
    for update in add_updates:
        skill = update.get("skill")
        if not isinstance(skill, dict):
            continue
        raw_items.append({
            "option": "add",
            "skill": skill,
            "source_metadata": {
                "domain": "global",
                "evolution_mode": "atomic_add",
            },
            "source_task_ids": [update.get("query_id")],
            "source_task_id": update.get("query_id"),
        })

    merge_input = prepare_skills_for_clustering(raw_items)
    if not merge_input:
        return []

    # Embed and cluster
    embedding_service = EmbeddingService(
        model=config.embedding_model,
        base_url=config.embedding_api_base,
        batch_size=config.merge_embedding_batch_size,
        api_key=config.embedding_api_key,
    )
    embeddings = await embedding_service.embed_batch(
        [item["embedding_text"] for item in merge_input],
        show_progress=not config.quiet,
    )
    clustering = CompleteLinkageClustering(
        distance_threshold=1 - config.merge_similarity_threshold,
    )
    clusters = clustering.cluster_skills(merge_input, embeddings)
    clusters = [cluster[:15] for cluster in clusters]

    # Merge clusters
    merger = SkillMerger(
        llm=llm,
        benchmark="atomic",
        skill_type="atomic",
        tool_schemas={},
        max_group_size=15,
        verbose=False,
        max_concurrent=config.max_concurrent,
    )
    merged = await merger.merge_clusters(clusters)

    # Normalize merged results
    from ..core.merge_filter import normalize_merged_results
    normalized = normalize_merged_results(merged)

    # Filter
    filtered = await _filter_atomic_candidates(
        llm=llm,
        benchmark="atomic",
        candidates=normalized,
        batch_size=config.batch_size,
        max_concurrent=config.max_concurrent,
        skip_general_filter=config.skip_general_filter,
        show_progress=not config.quiet,
    )
    return filtered


async def run_atomic_skill_finalize(
    *,
    atomic_proposals_path: Path,
    trace_path: Path,
    output_dir: Path,
    artifacts: EvolutionArtifacts,
    llm: Any,
    config: Any,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Step 4: Finalize atomic skills from proposals.

    - Modify proposals: LLM finalize each existing atomic
    - Add proposals: Cluster → merge → filter

    Returns (finalized_modified, finalized_added).
    """
    atomic_updates = _read_jsonl(atomic_proposals_path)

    # Group by option type
    grouped_modify: Dict[str, List[Dict[str, Any]]] = {}
    add_updates: List[Dict[str, Any]] = []
    for update in atomic_updates:
        option = update.get("option")
        if option == "modify":
            grouped_modify.setdefault(str(update.get("skill_name", "")).strip(), []).append(update)
        elif option == "add":
            add_updates.append(update)

    finalized_modified: List[Dict[str, Any]] = []
    proposal_trace: List[Dict[str, Any]] = []
    semaphore = asyncio.Semaphore(config.max_concurrent)

    async def finalize_modified(
        skill_name: str, proposals: List[Dict[str, Any]]
    ) -> Optional[Tuple[List[Dict[str, Any]], Dict[str, Any]]]:
        existing = artifacts.atomic_skills.get(skill_name)
        if existing is None:
            return None
        selected = _sort_proposals(proposals)[:config.max_modify_proposals_per_skill]
        trace = {
            "skill_name": skill_name,
            "mode": "modify",
            "proposal_count": len(proposals),
            "selected_proposal_count": len(selected),
            "selected_query_ids": [str(item.get("query_id", "")) for item in selected],
        }
        async with semaphore:
            response = await llm.ainvoke(
                messages=[
                    ("system", _cost_aware_prompt(ATOMIC_FINALIZE_PROMPT, config)),
                    ("human", json.dumps(
                        {
                            "skill_name": skill_name,
                            "existing_skill": existing,
                            "proposals": selected,
                        },
                        ensure_ascii=False,
                        indent=2,
                    )),
                ],
                regex_extractor=_extract_atomic_skill_payload,
            )
        skill = _extract_atomic_skill_payload(response)
        if skill is None:
            return None
        raw_item = {
            "option": "add",
            "skill": skill,
            "source_metadata": {
                "domain": "global",
                "evolution_mode": "modify_finalize",
                "original_skill_name": skill_name,
            },
            "source_task_ids": sorted({
                str(item.get("query_id", "")).strip()
                for item in selected
                if str(item.get("query_id", "")).strip()
            }),
            "merge_source_count": len(selected),
        }
        # Filter through general filter
        filtered = await _filter_atomic_candidates(
            llm=llm,
            benchmark="atomic",
            candidates=[raw_item],
            batch_size=config.batch_size,
            max_concurrent=config.max_concurrent,
            skip_general_filter=config.skip_general_filter,
        )
        return filtered, trace

    # Process modify proposals
    modify_tasks = [
        finalize_modified(skill_name, proposals)
        for skill_name, proposals in sorted(grouped_modify.items())
    ]
    for result in await _run_batched_tasks(
        modify_tasks,
        batch_size=config.batch_size,
        max_concurrent=config.max_concurrent,
        show_progress=not config.quiet,
        progress_desc="Atomic Finalize Modify",
        progress_unit="skill",
    ):
        if isinstance(result, Exception) or result is None:
            continue
        filtered, trace = result
        proposal_trace.append(trace)
        if filtered:
            finalized_modified.extend(filtered)

    # Process add proposals: cluster + merge + filter
    finalized_added = await _merge_new_atomic_adds(
        llm=llm,
        add_updates=add_updates,
        config=config,
    )

    # Write outputs
    _write_jsonl(trace_path, proposal_trace)
    _write_jsonl(output_dir / "finalized_atomic_modified.jsonl", finalized_modified)
    _write_jsonl(output_dir / "finalized_atomic_added.jsonl", finalized_added)

    logger.info(
        f"  Finalized atomic: {len(finalized_modified)} modified + "
        f"{len(finalized_added)} added (from {len(add_updates)} add proposals)"
    )
    return finalized_modified, finalized_added
