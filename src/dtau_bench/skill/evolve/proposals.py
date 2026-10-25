"""Step 1 (Tool Feedback Summary) and Step 2 (Direct Proposal Extraction).

Step 1: Summarize overly-long tool feedback in trajectories.
Step 2: For each session, extract add/modify/keep/skip proposals via LLM.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Artifact Loading (reuse dtau_bench's own loader)
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class EvolutionArtifacts:
    """Container for the previous skill library artifacts needed for evolution."""

    skill_library_dir: Path
    plans: Dict[str, Dict[str, Any]]
    atomic_skills: Dict[str, Dict[str, Any]]
    tool_skills: Dict[str, Dict[str, Any]]
    inactive_atomic_skills: List[Dict[str, Any]]
    inactive_tool_skills: List[Dict[str, Any]]
    planning_to_atomic: Dict[str, Set[str]]
    atomic_to_planning: Dict[str, Set[str]]
    atomic_to_tool: Dict[str, Set[str]]


def load_evolution_artifacts(skill_library_dir: Optional[Path]) -> EvolutionArtifacts:
    """Load all artifacts needed for evolution from a skill library directory.

    If skill_library_dir is None or does not exist, returns an empty EvolutionArtifacts
    (cold-start bootstrap with no previous skills).
    """
    if skill_library_dir is None or not skill_library_dir.exists():
        logger.info(
            "No previous skill library provided (cold-start bootstrap). "
            "Starting evolution from scratch."
        )
        return EvolutionArtifacts(
            skill_library_dir=skill_library_dir or Path("."),
            plans={},
            atomic_skills={},
            tool_skills={},
            inactive_atomic_skills=[],
            inactive_tool_skills=[],
            planning_to_atomic={},
            atomic_to_planning={},
            atomic_to_tool={},
        )

    from ..loader import SkillLibraryLoader

    raw = SkillLibraryLoader.load(skill_library_dir, mode="evolution")

    # Index plans by task name
    plans: Dict[str, Dict[str, Any]] = {}
    for item in raw.plans:
        task = str(item.get("task", "")).strip()
        if task:
            plans[task] = item

    # Index atomic skills by name
    atomic_skills: Dict[str, Dict[str, Any]] = {}
    for item in raw.skills:
        name = str(item.get("name", "")).strip()
        skill_type = str(item.get("skill_type", "")).strip()
        if name and skill_type in ("atomic", "functional"):
            atomic_skills[name] = item

    # Index tool skills by name
    tool_skills: Dict[str, Dict[str, Any]] = {}
    for item in raw.tool_skills:
        name = str(item.get("name", "")).strip()
        if name:
            tool_skills[name] = item

    # Parse graph edges
    planning_to_atomic: Dict[str, Set[str]] = {}
    atomic_to_planning: Dict[str, Set[str]] = {}
    atomic_to_tool: Dict[str, Set[str]] = {}

    graph_payload = raw.skill_graph or {}
    nodes = graph_payload.get("nodes", []) if isinstance(graph_payload, dict) else []
    edges = graph_payload.get("edges", []) if isinstance(graph_payload, dict) else []
    node_name_by_id = {
        str(node.get("node_id", "")).strip(): str(node.get("name", "")).strip()
        for node in nodes
        if isinstance(node, dict)
    }
    for edge in edges:
        if not isinstance(edge, dict):
            continue
        source_name = node_name_by_id.get(str(edge.get("source", "")).strip(), "")
        target_name = node_name_by_id.get(str(edge.get("target", "")).strip(), "")
        if not source_name or not target_name:
            continue
        edge_type = str(edge.get("edge_type", "")).strip()
        if edge_type == "plan_to_atomic":
            planning_to_atomic.setdefault(source_name, set()).add(target_name)
            atomic_to_planning.setdefault(target_name, set()).add(source_name)
        elif edge_type == "atomic_to_tool":
            atomic_to_tool.setdefault(source_name, set()).add(target_name)

    return EvolutionArtifacts(
        skill_library_dir=skill_library_dir,
        plans=plans,
        atomic_skills=atomic_skills,
        tool_skills=tool_skills,
        inactive_atomic_skills=raw.inactive_atomic_skills,
        inactive_tool_skills=raw.inactive_tool_skills,
        planning_to_atomic=planning_to_atomic,
        atomic_to_planning=atomic_to_planning,
        atomic_to_tool=atomic_to_tool,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Read a JSONL file into a list of dicts."""
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    """Write a list of dicts to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _safe_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _compact_skill(skill: Dict[str, Any]) -> Dict[str, Any]:
    """Compact a skill to essential fields for prompt context."""
    return {
        "name": str(skill.get("name", "")).strip(),
        "document": str(skill.get("document", "")),
        "content": str(skill.get("content", "")),
        "tools": [
            str(tool).strip()
            for tool in _safe_list(skill.get("tools"))
            if str(tool).strip()
        ],
        "skill_type": str(
            skill.get("skill_type", skill.get("metadata", {}).get("skill_type", ""))
        ).strip(),
    }


def _compact_planning_skill(skill: Dict[str, Any]) -> Dict[str, Any]:
    """Compact a planning skill for prompt context."""
    return {
        "name": str(skill.get("name", skill.get("task", ""))).strip(),
        "plan": str(skill.get("plan", "")),
        "skill_type": "planning",
    }


def _trajectory_field(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Get the best available trajectory from a record."""
    successful = row.get("successful_trajectory")
    if isinstance(successful, list):
        return successful
    trajectory = row.get("trajectory")
    return trajectory if isinstance(trajectory, list) else []


def _compact_tool_errors(row: Dict[str, Any], limit: int = 6) -> List[Dict[str, Any]]:
    """Extract compact tool error records."""
    errors: List[Dict[str, Any]] = []
    for item in _safe_list(row.get("tool_errors"))[:limit]:
        if not isinstance(item, dict):
            continue
        errors.append({
            "step_index": item.get("step_index"),
            "tool_name": item.get("tool_name"),
            "arguments": item.get("arguments"),
            "tool_result": item.get("tool_result"),
        })
    return errors


def _compact_cost_evidence(row: Dict[str, Any], limit: int = 20) -> List[Dict[str, Any]]:
    """Return observed action costs only; never derive them from gold trajectories."""
    evidence: List[Dict[str, Any]] = []
    for item in _safe_list(row.get("tool_calls"))[:limit]:
        if not isinstance(item, dict) or "cost" not in item:
            continue
        evidence.append({
            "tool_name": item.get("tool_name"),
            "arguments": item.get("arguments"),
            "has_error": item.get("has_error"),
            "cost": item.get("cost"),
        })
    return evidence


def _compact_session_snippet(row: Dict[str, Any], limit: int = 12) -> List[Dict[str, Any]]:
    """Extract a compact trajectory snippet for trace."""
    snippet: List[Dict[str, Any]] = []
    for msg in _trajectory_field(row)[:limit]:
        if not isinstance(msg, dict):
            continue
        snippet.append({
            "role": msg.get("role"),
            "content": msg.get("content"),
            "tool_calls": msg.get("tool_calls"),
            "id": msg.get("id"),
        })
    return snippet


def _names_from_skill_refs(value: Any) -> Set[str]:
    """Extract selected skill names from rollout dataset references."""
    names: Set[str] = set()
    for item in _safe_list(value):
        if isinstance(item, str) and item.strip():
            names.add(item.strip())
        elif isinstance(item, dict):
            name = str(item.get("name", "")).strip()
            if name:
                names.add(name)
    return names


def _restore_retrieved_context(
    row: Dict[str, Any],
    artifacts: EvolutionArtifacts,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Restore skill context context from names when legacy datasets lack full objects.

    Rollouts record the exact selected atomic names and planning hits, while older
    dataset exports only kept a compact name list. Resolve those names against the
    previous library rather than widening the context to all available skills.
    """
    trace = row.get("skill_context_retrieval_trace", {})
    if not isinstance(trace, dict):
        trace = {}

    plan_names = _names_from_skill_refs(trace.get("planning_hit_names"))
    selected_names = _names_from_skill_refs(row.get("selected_skills"))
    selected_names.update(_names_from_skill_refs(trace.get("selected_atomic_skill_names")))

    plans = [
        _compact_planning_skill(artifacts.plans[name])
        for name in sorted(plan_names)
        if name in artifacts.plans
    ]
    atomic = [
        _compact_skill(artifacts.atomic_skills[name])
        for name in sorted(selected_names)
        if name in artifacts.atomic_skills
    ]
    return plans, atomic


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Session Packet Building + Feedback Summary
# ─────────────────────────────────────────────────────────────────────────────


def prepare_session_packets(
    rows: Sequence[Dict[str, Any]],
    *,
    artifacts: EvolutionArtifacts,
) -> Tuple[List[Dict[str, Any]], int]:
    """Convert input rows into evolution-ready session packets.

    Each packet contains the record's key info + retrieved skills context.
    Returns (packets, backfilled_tool_skill_count).
    """
    packets: List[Dict[str, Any]] = []
    backfilled_count = 0

    for row in rows:
        if not isinstance(row, dict):
            continue

        retrieved_plans_raw = _safe_list(row.get("retrieved_plan_skills"))
        retrieved_atomic_raw = _safe_list(row.get("retrieved_atomic_skills"))
        retrieved_tool_raw = _safe_list(row.get("retrieved_tool_skills"))

        retrieved_plans = [
            _compact_planning_skill(item)
            for item in retrieved_plans_raw
            if isinstance(item, dict) and str(item.get("name", item.get("task", ""))).strip()
        ]
        retrieved_atomic = [
            _compact_skill(item)
            for item in retrieved_atomic_raw
            if isinstance(item, dict) and str(item.get("name", "")).strip()
        ]
        retrieved_tools = [
            _compact_skill(item)
            for item in retrieved_tool_raw
            if isinstance(item, dict) and str(item.get("name", "")).strip()
        ]

        # Legacy datasets stored only skill context names. Reconstruct the exact context
        # from the library used for this rollout instead of dropping the session.
        if not retrieved_plans and not retrieved_atomic:
            restored_plans, restored_atomic = _restore_retrieved_context(row, artifacts)
            retrieved_plans = restored_plans
            retrieved_atomic = restored_atomic

        # Backfill tool skills from graph if missing
        if not retrieved_tools and retrieved_atomic:
            seen: Set[str] = set()
            backfilled: List[Dict[str, Any]] = []
            for atomic in retrieved_atomic:
                for tool_name in artifacts.atomic_to_tool.get(atomic["name"], set()):
                    tool_skill = artifacts.tool_skills.get(tool_name)
                    if tool_skill is None or tool_name in seen:
                        continue
                    seen.add(tool_name)
                    backfilled.append(_compact_skill(tool_skill))
            if backfilled:
                retrieved_tools = backfilled
                backfilled_count += 1

        packet = {
            "record_id": str(
                row.get("record_id")
                or row.get("trajectory_id")
                or row.get("query_id")
                or row.get("task_id")
                or ""
            ).strip(),
            "query_id": str(row.get("query_id", "")).strip(),
            "task_id": str(row.get("task_id", "")).strip(),
            "version": row.get("version"),
            "domain": row.get("domain"),
            "user_task": str(row.get("user_task", "")),
            "passed": row.get("passed"),
            "reward": row.get("reward"),
            "total_score": row.get("total_score"),
            "trajectory": _trajectory_field(row),
            "tool_errors": _compact_tool_errors(row),
            "cost_summary": row.get("cost_summary") if isinstance(row.get("cost_summary"), dict) else {},
            "observed_action_costs": _compact_cost_evidence(row),
            "retrieved_plan_skills": retrieved_plans,
            "retrieved_atomic_skills": retrieved_atomic,
            "retrieved_tool_skills": retrieved_tools,
            "metadata": row.get("metadata", {}) if isinstance(row.get("metadata"), dict) else {},
        }

        # Skip records without trajectory or context
        if not packet["trajectory"]:
            continue
        if not (
            packet["retrieved_plan_skills"]
            or packet["retrieved_atomic_skills"]
            or packet["retrieved_tool_skills"]
        ):
            continue
        packets.append(packet)

    return packets, backfilled_count


async def run_feedback_summary(
    *,
    input_path: Path,
    output_path: Path,
    artifacts: EvolutionArtifacts,
    llm: Any,
    config: Any,
) -> None:
    """Step 1: Build session packets and summarize long tool feedback."""
    input_rows = _read_jsonl(input_path)
    if not input_rows:
        raise AssertionError(f"No input rows found in {input_path}")

    session_packets, backfilled_count = prepare_session_packets(input_rows, artifacts=artifacts)
    if not session_packets:
        raise AssertionError(f"No evolution-eligible session packets found in {input_path}")

    logger.info(
        f"  Prepared {len(session_packets)} session packets "
        f"(backfilled tool skills for {backfilled_count} sessions)"
    )

    # Summarize long feedback if enabled
    if config.summarize_long_feedback:
        from ..core.extraction.tool_summary import ToolSummary

        summarizer = ToolSummary(
            llm=llm,
            max_len=config.max_feedback_length,
            verbose=not config.quiet,
        )
        session_packets = await summarizer.summarize_multiple(
            session_packets,
            batch_size=config.batch_size,
            max_concurrent=config.max_concurrent,
            show_progress=not config.quiet,
        )

    _write_jsonl(output_path, session_packets)
    logger.info(f"  Wrote {len(session_packets)} summarized sessions to {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Direct Proposal Extraction
# ─────────────────────────────────────────────────────────────────────────────


DIRECT_PROPOSAL_PROMPT = """You are updating a graph-backed skill library from one rollout session.

You will be given:
- task metadata and reward outcome
- a summarized trajectory
- planning skills retrieved into context
- atomic skills retrieved into context
- tool skills retrieved/attached into context

Your job is to decide which tool skills and atomic skills should be kept, modified, or newly added.
The resulting skills are for a future agent to reuse while solving user tasks.
They should describe agent-side workflow guidance, tool usage constraints, recovery behavior, and reusable execution patterns.
Do not turn user behavior or conversation style into a skill unless it directly constrains how the agent should act.

Prioritization:
- Prefer updates that capture non-trivial, reusable agent guidance.
- Prefer sessions that reveal complex workflow, multi-step coordination, hidden dependency, recovery behavior, or easy-to-miss constraints.
- Prefer guidance that would help a future agent avoid mistakes, avoid dead ends, or handle a subtask that is somewhat hard rather than obvious.
- A successful session can still justify an update if it demonstrates a non-obvious workflow or a subtle but important control rule.
- If a subtask was solved in a simple, direct, obvious way and the session adds no meaningful new guidance, prefer "keep" or "skip".

High-value signals:
- lookup -> act, validate -> execute, preview -> confirm, or retrieve ID -> perform update patterns
- tool errors, retries, repair steps, argument correction, format correction, or fallback handling
- multi-tool coordination, order dependence, or context-building before action
- cases where the correct action is not just "call the tool directly"
- reusable guidance for difficult or failure-prone subtasks

Low-value signals:
- merely restating what a tool does
- retelling the task or conversation
- documenting a one-step happy-path call that is already obvious from the tool itself
- adding generic advice with no concrete workflow value
- overfitting to task-specific facts instead of extracting reusable guidance

Rules:
- Tool skills are one-tool-one-skill.
- If a tool already has a tool skill in context, you may only use option "modify" or "keep" for that tool.
- If a tool does not yet have a tool skill in context, you may only use option "add" or "skip".
- For tool skills, do NOT output a `tools` field; the system will set it automatically.
- Tool skill updates should focus on agent-facing pitfalls, prerequisites, verification steps, confirmation semantics, argument traps, and common recovery patterns for that tool.
- Tool skills should NOT mainly be API summaries, generic capability descriptions, or obvious parameter recitations.
- For existing atomic skills in context:
  - use "modify" only if the skill needs a real update
  - use "keep" if no update is needed
- You may use atomic "add" only for a genuinely new reusable capability that is not just a clarification of an existing atomic skill.
- Atomic skill updates should focus on reusable multi-step subtask workflow: sequencing, branching, validation, retry, recovery, and tool coordination.
- Atomic skills should NOT mainly capture trivial single-call behavior or task retelling.
- Prioritize reusable agent workflow guidance over task-specific narration.
- Do NOT force updates. If the session does not contribute meaningful new reusable guidance, choose keep/skip.
- Do NOT propose any planning skill updates here.

Output exactly one JSON object inside <proposal>...</proposal> with these keys:
- confidence: number between 0 and 1
- reasoning_summary: string
- tool_skill_updates: list[object]
- atomic_skill_updates: list[object]

Each tool update object must have:
- option: "add" | "modify" | "keep" | "skip"
- tool_name: string
- rationale: string
- skill: object or null

Each atomic update object must have:
- option: "add" | "modify" | "keep" | "skip"
- skill_name: string
- modified_from: string or ""
- rationale: string
- skill: object or null

For add/modify skill objects:
- atomic skill must contain name/document/content/tools
- tool skill must contain name/document/content
"""


def _proposal_prompt(cost_aware: bool) -> str:
    if not cost_aware:
        return DIRECT_PROPOSAL_PROMPT
    from ..cost import COST_AWARE_GUIDANCE

    return DIRECT_PROPOSAL_PROMPT + COST_AWARE_GUIDANCE


import re


def _extract_tagged_json(text: str, tag: str) -> Optional[Dict[str, Any]]:
    """Extract JSON object from <tag>...</tag> in LLM response."""
    match = re.search(rf"<{tag}>(.*?)</{tag}>", text, flags=re.S)
    if not match:
        return None
    try:
        payload = json.loads(match.group(1).strip())
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _extract_update_proposal(text: str) -> Optional[Dict[str, Any]]:
    """Parse a proposal from LLM output."""
    payload = _extract_tagged_json(text, "proposal")
    if payload is None:
        return None
    if not isinstance(payload.get("tool_skill_updates", []), list):
        return None
    if not isinstance(payload.get("atomic_skill_updates", []), list):
        return None
    confidence = payload.get("confidence")
    if not isinstance(confidence, (int, float)):
        payload["confidence"] = 0.0
    if not isinstance(payload.get("reasoning_summary"), str):
        payload["reasoning_summary"] = ""
    return payload


def _normalize_tool_option(
    update: Dict[str, Any],
    *,
    tool_name: str,
    has_existing_skill: bool,
) -> Optional[Dict[str, Any]]:
    """Normalize and validate a tool skill proposal."""
    option = str(update.get("option", "")).strip()
    if option == "skip":
        return None
    if has_existing_skill and option not in {"modify", "keep"}:
        return None
    if not has_existing_skill and option not in {"add", "skip"}:
        return None
    payload: Dict[str, Any] = {
        "option": option,
        "tool_name": tool_name,
        "rationale": str(update.get("rationale", "")).strip(),
    }
    skill = update.get("skill")
    if option in {"add", "modify"}:
        if not isinstance(skill, dict):
            return None
        if not isinstance(skill.get("name"), str):
            return None
        if not isinstance(skill.get("document"), str):
            return None
        if not isinstance(skill.get("content"), str):
            return None
        payload["skill"] = {
            "name": tool_name,
            "document": skill.get("document", ""),
            "content": skill.get("content", ""),
            "tools": [tool_name],
        }
    return payload


def _normalize_atomic_option(
    update: Dict[str, Any],
    *,
    existing_atomic_names: Set[str],
    retrieved_atomic_names: Set[str],
) -> Optional[Dict[str, Any]]:
    """Normalize and validate an atomic skill proposal."""
    option = str(update.get("option", "")).strip()
    if option == "skip":
        return None
    skill_name = str(update.get("skill_name", "")).strip()
    modified_from = str(update.get("modified_from", "")).strip()

    if option == "keep":
        if skill_name not in retrieved_atomic_names:
            return None
        return {
            "option": "keep",
            "skill_name": skill_name,
            "modified_from": modified_from or skill_name,
            "rationale": str(update.get("rationale", "")).strip(),
        }
    if option == "modify":
        target_name = modified_from or skill_name
        if target_name not in retrieved_atomic_names or target_name not in existing_atomic_names:
            return None
        skill = update.get("skill")
        if not isinstance(skill, dict):
            return None
        if not isinstance(skill.get("name"), str):
            return None
        if not isinstance(skill.get("document"), str):
            return None
        if not isinstance(skill.get("content"), str):
            return None
        if not isinstance(skill.get("tools"), list):
            return None
        return {
            "option": "modify",
            "skill_name": target_name,
            "modified_from": target_name,
            "rationale": str(update.get("rationale", "")).strip(),
            "skill": skill,
        }
    if option == "add":
        skill = update.get("skill")
        if not isinstance(skill, dict):
            return None
        if not isinstance(skill.get("name"), str) or not skill.get("name", "").strip():
            return None
        if not isinstance(skill.get("document"), str):
            return None
        if not isinstance(skill.get("content"), str):
            return None
        if not isinstance(skill.get("tools"), list):
            return None
        return {
            "option": "add",
            "skill_name": str(skill.get("name", "")).strip(),
            "modified_from": "",
            "rationale": str(update.get("rationale", "")).strip(),
            "skill": skill,
        }
    return None


def _sort_proposals(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Sort proposals: failures first, then by confidence descending."""
    def _key(item: Dict[str, Any]) -> Tuple[int, float, str]:
        passed = item.get("passed")
        failure_rank = 0 if passed is False else 1
        confidence = float(item.get("analysis_confidence", 0.0) or 0.0)
        query_id = str(item.get("query_id", "")).strip()
        return (failure_rank, -confidence, query_id)
    return sorted(records, key=_key)


async def _run_batched_tasks(
    tasks: Sequence[Any],
    *,
    batch_size: int,
    max_concurrent: Optional[int] = None,
    show_progress: bool,
    progress_desc: str,
    progress_unit: str,
) -> List[Any]:
    """Run async tasks with bounded concurrency while preserving input order.

    ``batch_size`` remains the concurrency fallback for legacy callers.  New
    callers should pass ``max_concurrent`` explicitly so a small processing
    batch does not cap independent LLM requests or add a barrier between them.
    """
    if not tasks:
        return []

    concurrency = batch_size if max_concurrent is None else max_concurrent
    if isinstance(concurrency, bool) or not isinstance(concurrency, int) or concurrency < 1:
        raise ValueError(
            f"max_concurrent must be a positive integer, got {concurrency!r}"
        )

    try:
        from tqdm.auto import tqdm as _tqdm
    except ImportError:
        _tqdm = None

    missing = object()
    results: List[Any] = [missing] * len(tasks)
    progress = None
    if show_progress and _tqdm is not None:
        progress = _tqdm(total=len(tasks), desc=progress_desc, unit=progress_unit)
    try:
        next_index = 0

        async def worker() -> None:
            nonlocal next_index
            while next_index < len(tasks):
                index = next_index
                next_index += 1
                try:
                    results[index] = await tasks[index]
                except Exception as exc:  # match gather(return_exceptions=True)
                    results[index] = exc
                if progress is not None:
                    progress.update(1)

        workers = [
            asyncio.create_task(worker())
            for _ in range(min(concurrency, len(tasks)))
        ]
        await asyncio.gather(*workers)
    finally:
        if progress is not None:
            progress.close()
    if any(result is missing for result in results):
        raise RuntimeError("Async task scheduler exited before all tasks completed")
    return results


async def _ainvoke_with_retry(
    llm: Any,
    *,
    messages: List[Tuple[str, str]],
    regex_extractor: Any,
    max_attempts: int = 5,
    base_delay: float = 2.0,
) -> Any:
    """Invoke the LLM, retrying transient failures with exponential backoff.

    Without this, a temporary outage makes every single session fail; the caller
    used to swallow those failures and write an empty proposal file, producing a
    library that inherits the previous version unchanged while every stage is
    marked completed.
    """
    attempts = max(1, int(max_attempts))
    last_error: Optional[BaseException] = None
    for attempt in range(1, attempts + 1):
        try:
            return await llm.ainvoke(messages=messages, regex_extractor=regex_extractor)
        except Exception as e:  # noqa: BLE001 - transient API/network failures
            last_error = e
            if attempt >= attempts:
                break
            delay = base_delay * (2 ** (attempt - 1))
            logger.debug(
                "  Proposal LLM call failed (attempt %d/%d): %s; retrying in %.1fs",
                attempt, attempts, e, delay,
            )
            await asyncio.sleep(delay)
    assert last_error is not None
    raise last_error


async def run_proposal_extraction(
    *,
    summarized_path: Path,
    tool_proposals_path: Path,
    atomic_proposals_path: Path,
    trace_path: Path,
    artifacts: EvolutionArtifacts,
    llm: Any,
    config: Any,
) -> None:
    """Step 2: Extract update proposals from each summarized session."""
    session_packets = _read_jsonl(summarized_path)
    if not session_packets:
        raise AssertionError(f"No summarized sessions in {summarized_path}")

    semaphore = asyncio.Semaphore(config.max_concurrent)
    tool_results: List[Dict[str, Any]] = []
    atomic_results: List[Dict[str, Any]] = []
    proposal_traces: List[Dict[str, Any]] = []

    async def run_one(packet: Dict[str, Any]) -> Tuple[
        List[Dict[str, Any]], List[Dict[str, Any]], Optional[Dict[str, Any]]
    ]:
        retrieved_atomic_names = {
            str(item.get("name", "")).strip()
            for item in packet.get("retrieved_atomic_skills", [])
            if isinstance(item, dict) and str(item.get("name", "")).strip()
        }
        payload = {
            "query_id": packet.get("query_id"),
            "task_id": packet.get("task_id"),
            "domain": packet.get("domain"),
            "user_task": packet.get("user_task"),
            "passed": packet.get("passed"),
            "reward": packet.get("reward"),
            "total_score": packet.get("total_score"),
            "trajectory": packet.get("trajectory"),
            "tool_errors": packet.get("tool_errors"),
            "cost_summary": packet.get("cost_summary"),
            "observed_action_costs": packet.get("observed_action_costs"),
            "retrieved_plan_skills": packet.get("retrieved_plan_skills"),
            "retrieved_atomic_skills": packet.get("retrieved_atomic_skills"),
            "retrieved_tool_skills": packet.get("retrieved_tool_skills"),
        }
        async with semaphore:
            response = await _ainvoke_with_retry(
                llm,
                messages=[
                    ("system", _proposal_prompt(config.cost_aware_skills)),
                    ("human", json.dumps(payload, ensure_ascii=False, indent=2)),
                ],
                regex_extractor=_extract_update_proposal,
                max_attempts=int(getattr(config, "proposal_max_attempts", 5) or 5),
            )
        parsed = _extract_update_proposal(response)
        if parsed is None:
            return [], [], None

        confidence = float(parsed.get("confidence", 0.0) or 0.0)
        reasoning_summary = str(parsed.get("reasoning_summary", "")).strip()

        normalized_tool: List[Dict[str, Any]] = []
        for update in parsed.get("tool_skill_updates", []):
            if not isinstance(update, dict):
                continue
            tool_name = str(update.get("tool_name", "")).strip()
            if not tool_name:
                continue
            normalized = _normalize_tool_option(
                update,
                tool_name=tool_name,
                has_existing_skill=(tool_name in artifacts.tool_skills),
            )
            if normalized is None:
                continue
            normalized.update({
                "record_id": packet.get("record_id"),
                "query_id": packet.get("query_id"),
                "passed": packet.get("passed"),
                "analysis_confidence": confidence,
                "reasoning_summary": reasoning_summary,
                "session_snippet": _compact_session_snippet(packet),
            })
            normalized_tool.append(normalized)

        normalized_atomic: List[Dict[str, Any]] = []
        for update in parsed.get("atomic_skill_updates", []):
            if not isinstance(update, dict):
                continue
            normalized = _normalize_atomic_option(
                update,
                existing_atomic_names=set(artifacts.atomic_skills.keys()),
                retrieved_atomic_names=retrieved_atomic_names,
            )
            if normalized is None:
                continue
            normalized.update({
                "record_id": packet.get("record_id"),
                "query_id": packet.get("query_id"),
                "passed": packet.get("passed"),
                "analysis_confidence": confidence,
                "reasoning_summary": reasoning_summary,
                "session_snippet": _compact_session_snippet(packet),
            })
            normalized_atomic.append(normalized)

        trace = {
            "record_id": packet.get("record_id"),
            "query_id": packet.get("query_id"),
            "passed": packet.get("passed"),
            "confidence": confidence,
            "reasoning_summary": reasoning_summary,
            "tool_update_count": len(normalized_tool),
            "atomic_update_count": len(normalized_atomic),
        }
        return normalized_tool, normalized_atomic, trace

    tasks = [run_one(packet) for packet in session_packets]
    batch_results = await _run_batched_tasks(
        tasks,
        batch_size=config.batch_size,
        max_concurrent=config.max_concurrent,
        show_progress=not config.quiet,
        progress_desc="Proposal Extraction",
        progress_unit="record",
    )

    failed = 0
    for result in batch_results:
        if isinstance(result, Exception):
            failed += 1
            logger.warning(f"  Proposal extraction error: {result}")
            continue
        tool_updates, atomic_updates, trace = result
        tool_results.extend(tool_updates)
        atomic_results.extend(atomic_updates)
        if trace is not None:
            proposal_traces.append(trace)

    # Fail loudly instead of writing empty proposal files: an all-failed run
    # would otherwise be indistinguishable from "the model proposed no changes",
    # and the resulting library silently inherits the previous version.
    total = len(session_packets)
    max_failure_ratio = float(getattr(config, "proposal_max_failure_ratio", 0.2) or 0.2)
    if failed and failed / max(total, 1) > max_failure_ratio:
        raise RuntimeError(
            f"Proposal extraction failed for {failed}/{total} sessions "
            f"({failed / max(total, 1):.1%} > {max_failure_ratio:.0%} allowed). "
            "Refusing to write empty proposals. Check LLM connectivity/quota, "
            "then rerun this stage."
        )
    if failed:
        logger.warning(
            "  Proposal extraction: %d/%d sessions failed (within the %.0f%% "
            "tolerance), continuing.", failed, total, max_failure_ratio * 100,
        )

    _write_jsonl(tool_proposals_path, tool_results)
    _write_jsonl(atomic_proposals_path, atomic_results)
    _write_jsonl(trace_path, proposal_traces)

    logger.info(
        f"  Extracted {len(tool_results)} tool proposals + "
        f"{len(atomic_results)} atomic proposals from {len(session_packets)} sessions"
    )
