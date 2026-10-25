"""Step 5: Planning Propagation.

Uses the skill graph to propagate atomic skill changes up to planning skills.
For each modified atomic skill linked to a planning skill via the graph,
the LLM decides whether the planning skill needs updating.
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


PLANNING_PROPAGATION_PROMPT = """You are deciding whether an existing planning skill should change because one linked atomic skill changed.

You will be given:
- the current planning skill
- the old atomic skill
- the new atomic skill
- graph linkage metadata
- compact session snippets and proposal rationales supporting the atomic change

Output exactly one JSON object inside <proposal>...</proposal> with keys:
- option: "modify" | "keep"
- rationale: string
- plan: string

Rules:
- Use "keep" if the planning skill does not need to change.
- Use "modify" only if the planning text should materially change because of the atomic update.
- Do not rename the planning skill key.
- Do not create new planning skills.
"""


PLANNING_FINALIZE_PROMPT = """You are finalizing one planning skill from the current canonical version plus a set of propagation modify proposals.

Output exactly one JSON object inside <plan>...</plan> with keys:
- plan

Constraints:
- Preserve the existing planning identity; only revise the plan text.
- Keep the result reusable and high-level.
- Integrate propagation-driven updates without overfitting to one specific rollout.
"""


def _cost_aware_prompt(prompt: str, config: Any) -> str:
    if not config.cost_aware_skills:
        return prompt
    from ..cost import COST_AWARE_GUIDANCE

    return prompt + COST_AWARE_GUIDANCE


# ─────────────────────────────────────────────────────────────────────────────
# Parsers
# ─────────────────────────────────────────────────────────────────────────────


def _extract_plan_proposal(text: str) -> Optional[Dict[str, Any]]:
    """Parse a planning propagation proposal from LLM output."""
    payload = _extract_tagged_json(text, "proposal")
    if payload is None:
        return None
    option = str(payload.get("option", "")).strip()
    if option not in {"modify", "keep"}:
        return None
    if not isinstance(payload.get("rationale"), str):
        return None
    if not isinstance(payload.get("plan"), str):
        return None
    return payload


def _extract_plan_finalize(text: str) -> Optional[Dict[str, Any]]:
    """Parse a finalized plan from LLM output."""
    payload = _extract_tagged_json(text, "plan")
    if payload is None:
        return None
    if not isinstance(payload.get("plan"), str):
        return None
    return payload


# ─────────────────────────────────────────────────────────────────────────────
# Planning Graph Matcher (fallback for missing graph edges)
# ─────────────────────────────────────────────────────────────────────────────


class PlanningGraphMatcher:
    """Match atomic skills to planning skills by tool coverage.

    Used when the skill graph doesn't have edges for a given atomic skill
    (e.g., newly added atomics). Uses the same criterion as the offline graph
    builder: an atomic skill is linked to a plan when the plan's steps provide
    at least ``plan_atomic_coverage_threshold`` of the atomic skill's tools.
    """

    def __init__(
        self,
        *,
        embedding_api_base: str,
        embedding_api_key: str,
        embedding_model: str,
        plan_atomic_coverage_threshold: float = 0.5,
    ):
        self.embedding_api_base = embedding_api_base
        self.embedding_api_key = embedding_api_key
        self.embedding_model = embedding_model
        self.plan_atomic_coverage_threshold = plan_atomic_coverage_threshold

    def match_atomic_to_plans(
        self,
        *,
        atomic_skill_name: str,
        atomic_skill: Dict[str, Any],
        existing_plans: Dict[str, Dict[str, Any]],
    ) -> List[str]:
        """Find planning skills linked to this atomic skill by tool coverage."""
        from ..bootstrap.graph import _read_plan_steps

        plan_items = list(existing_plans.values())
        if not plan_items:
            return []

        atomic_tools = {
            str(name).strip()
            for name in (atomic_skill.get("tools") or [])
            if str(name or "").strip()
        }
        if not atomic_tools:
            return []

        matched: List[str] = []
        for plan in plan_items:
            steps = _read_plan_steps(str(plan.get("plan", "")))
            step_tools = {
                tool_name
                for _, tool_names in steps
                for tool_name in tool_names
            }
            if not step_tools:
                continue
            coverage = len(step_tools & atomic_tools) / len(atomic_tools)
            if coverage >= self.plan_atomic_coverage_threshold:
                matched.append(str(plan.get("task", "")).strip())

        return matched


# ─────────────────────────────────────────────────────────────────────────────
# Step 5: Planning Propagation
# ─────────────────────────────────────────────────────────────────────────────


async def run_planning_propagation(
    *,
    atomic_proposals_path: Path,
    output_dir: Path,
    artifacts: EvolutionArtifacts,
    llm: Any,
    config: Any,
) -> Tuple[Dict[str, str], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Step 5: Propagate atomic skill changes to planning skills.

    For each modified atomic skill, find linked planning skills via the graph,
    then ask LLM whether each planning skill should be updated.

    Returns (finalized_plans, propagation_proposals, finalize_trace).
    """
    # Load finalized atomic updates
    finalized_atomic_modified_path = output_dir / "finalized_atomic_modified.jsonl"
    finalized_atomic_added_path = output_dir / "finalized_atomic_added.jsonl"

    finalized_atomic_updates: List[Dict[str, Any]] = []
    if finalized_atomic_modified_path.exists():
        finalized_atomic_updates.extend(_read_jsonl(finalized_atomic_modified_path))
    if finalized_atomic_added_path.exists():
        finalized_atomic_updates.extend(_read_jsonl(finalized_atomic_added_path))

    # Load raw atomic proposals for supporting evidence
    atomic_updates = _read_jsonl(atomic_proposals_path) if atomic_proposals_path.exists() else []

    # Build map: changed atomic name → finalized item
    changed_atomic_by_name: Dict[str, Dict[str, Any]] = {
        str(item.get("skill", {}).get("name", "")).strip(): item
        for item in finalized_atomic_updates
        if isinstance(item.get("skill"), dict)
        and str(item.get("skill", {}).get("name", "")).strip()
    }

    # Build supporting evidence by atomic name
    support_updates_by_atomic: Dict[str, List[Dict[str, Any]]] = {}
    for update in atomic_updates:
        if update.get("option") != "modify":
            continue
        support_updates_by_atomic.setdefault(
            str(update.get("skill_name", "")).strip(), []
        ).append(update)

    # Initialize graph matcher for fallback
    graph_matcher = PlanningGraphMatcher(
        embedding_api_base=config.embedding_api_base,
        embedding_api_key=config.embedding_api_key,
        embedding_model=config.embedding_model,
        plan_atomic_coverage_threshold=config.plan_atomic_coverage_threshold,
    )

    semaphore = asyncio.Semaphore(config.max_concurrent)
    propagation_proposals: List[Dict[str, Any]] = []

    async def analyze_edge(
        atomic_name: str,
        planning_name: str,
        old_atomic: Optional[Dict[str, Any]],
        new_atomic: Dict[str, Any],
        existing_plan: Dict[str, Any],
        support_query_ids: List[str],
        support_records: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        async with semaphore:
            response = await llm.ainvoke(
                messages=[
                    ("system", _cost_aware_prompt(PLANNING_PROPAGATION_PROMPT, config)),
                    ("human", json.dumps(
                        {
                            "planning_name": planning_name,
                            "existing_plan": existing_plan,
                            "old_atomic_skill": old_atomic,
                            "new_atomic_skill": new_atomic,
                            "graph_link": {
                                "planning_name": planning_name,
                                "atomic_name": atomic_name,
                            },
                            "supporting_snippets": support_records,
                        },
                        ensure_ascii=False,
                        indent=2,
                    )),
                ],
                regex_extractor=_extract_plan_proposal,
            )
        proposal = _extract_plan_proposal(response)
        if proposal is None:
            return None
        return {
            "planning_name": planning_name,
            "atomic_name": atomic_name,
            "option": proposal.get("option"),
            "rationale": proposal.get("rationale"),
            "plan": proposal.get("plan"),
            "source_task_ids": support_query_ids,
        }

    # Build edge analysis tasks
    edge_tasks: List[Any] = []
    for atomic_name, finalized in sorted(changed_atomic_by_name.items()):
        old_atomic = artifacts.atomic_skills.get(atomic_name)
        new_atomic = finalized.get("skill")
        if not isinstance(new_atomic, dict):
            continue

        # Find linked planning skills
        parent_plans = set(artifacts.atomic_to_planning.get(atomic_name, set()))
        if not parent_plans:
            # Fallback: use embedding similarity to find linked plans
            parent_plans = set(graph_matcher.match_atomic_to_plans(
                atomic_skill_name=atomic_name,
                atomic_skill=new_atomic,
                existing_plans=artifacts.plans,
            ))

        # Get supporting evidence
        support_updates = _sort_proposals(
            support_updates_by_atomic.get(atomic_name, [])
        )[:config.max_modify_proposals_per_skill]
        support_query_ids = [
            str(item.get("query_id", "")).strip()
            for item in support_updates
            if str(item.get("query_id", "")).strip()
        ]
        support_records = [
            {
                "query_id": item.get("query_id"),
                "passed": item.get("passed"),
                "reasoning_summary": item.get("reasoning_summary"),
                "rationale": item.get("rationale"),
                "session_snippet": item.get("session_snippet"),
            }
            for item in support_updates
        ]

        for planning_name in sorted(parent_plans):
            existing_plan = artifacts.plans.get(planning_name)
            if existing_plan is None:
                continue
            edge_tasks.append(
                analyze_edge(
                    atomic_name,
                    planning_name,
                    old_atomic,
                    new_atomic,
                    existing_plan,
                    support_query_ids,
                    support_records,
                )
            )

    # Run propagation analysis
    for result in await _run_batched_tasks(
        edge_tasks,
        batch_size=config.batch_size,
        max_concurrent=config.max_concurrent,
        show_progress=not config.quiet,
        progress_desc="Planning Propagation",
        progress_unit="edge",
    ):
        if isinstance(result, Exception) or result is None:
            continue
        propagation_proposals.append(result)

    # Group modify proposals by planning name
    grouped_modify: Dict[str, List[Dict[str, Any]]] = {}
    for proposal in propagation_proposals:
        if proposal.get("option") == "modify":
            grouped_modify.setdefault(
                str(proposal.get("planning_name", "")).strip(), []
            ).append(proposal)

    # Initialize finalized plans with all existing plans
    finalized_plans: Dict[str, str] = {
        name: str(item.get("plan", "")).strip()
        for name, item in artifacts.plans.items()
    }
    finalize_trace: List[Dict[str, Any]] = []

    # Finalize plans with modifications
    async def finalize_plan(
        planning_name: str, proposals: List[Dict[str, Any]]
    ) -> Optional[Tuple[str, str, Dict[str, Any]]]:
        existing_plan = artifacts.plans.get(planning_name)
        if existing_plan is None:
            return None
        selected = list(_sort_proposals(proposals)[:config.max_modify_proposals_per_skill])
        trace = {
            "planning_name": planning_name,
            "proposal_count": len(proposals),
            "selected_proposal_count": len(selected),
            "source_task_ids": sorted({
                str(task_id).strip()
                for item in selected
                for task_id in item.get("source_task_ids", [])
                if str(task_id).strip()
            }),
        }
        async with semaphore:
            response = await llm.ainvoke(
                messages=[
                    ("system", _cost_aware_prompt(PLANNING_FINALIZE_PROMPT, config)),
                    ("human", json.dumps(
                        {
                            "planning_name": planning_name,
                            "existing_plan": existing_plan,
                            "proposals": selected,
                        },
                        ensure_ascii=False,
                        indent=2,
                    )),
                ],
                regex_extractor=_extract_plan_finalize,
            )
        finalized = _extract_plan_finalize(response)
        if finalized is None:
            return None
        return planning_name, str(finalized.get("plan", "")).strip(), trace

    plan_tasks = [
        finalize_plan(planning_name, proposals)
        for planning_name, proposals in sorted(grouped_modify.items())
    ]
    for result in await _run_batched_tasks(
        plan_tasks,
        batch_size=config.batch_size,
        max_concurrent=config.max_concurrent,
        show_progress=not config.quiet,
        progress_desc="Planning Finalize",
        progress_unit="plan",
    ):
        if isinstance(result, Exception) or result is None:
            continue
        planning_name, plan_text, trace = result
        finalized_plans[planning_name] = plan_text
        finalize_trace.append(trace)

    # Write outputs
    _write_jsonl(output_dir / "planning_propagation_proposals.jsonl", propagation_proposals)
    _write_jsonl(output_dir / "planning_finalize_trace.jsonl", finalize_trace)
    _write_jsonl(
        output_dir / "planning_skills.finalized.jsonl",
        [
            {"planning_name": name, "plan": plan_text}
            for name, plan_text in sorted(finalized_plans.items())
        ],
    )

    # Write plan_library.json (intermediate, before graph rebuild)
    plan_library_path = output_dir / "plan_library.json"
    with plan_library_path.open("w", encoding="utf-8") as f:
        json.dump({"plan": finalized_plans}, f, indent=2, ensure_ascii=False)

    logger.info(
        f"  Planning propagation: {len(propagation_proposals)} edges analyzed, "
        f"{len(finalize_trace)} plans modified"
    )
    return finalized_plans, propagation_proposals, finalize_trace
