"""Stage 1: Build skill context-ready dataset from rollout + eval artifacts.

Merges three required sources:
  1) Rollout output JSONL files (per-domain)
  2) Query files (task definitions)
  3) Evaluator results

Output: A single JSONL file with unified skill context-compatible records.

"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Read a JSONL file into a list of dicts."""
    items: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def _safe_json_loads(raw: Any) -> Any:
    """Safely parse JSON string, returning raw if already parsed."""
    if isinstance(raw, (dict, list)):
        return raw
    if raw is None:
        return {}
    if not isinstance(raw, str):
        return {"__raw": str(raw)}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"__raw": raw}


def _jsonable(value: Any) -> Any:
    """Ensure a value is JSON-serializable."""
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def _normalize_tool_calls(tool_calls: Any) -> Optional[List[Dict[str, Any]]]:
    """Normalize tool_calls into a consistent format."""
    if not tool_calls:
        return None

    normalized: List[Dict[str, Any]] = []
    for tc in tool_calls:
        tc_id = tc.get("id", "") if isinstance(tc, dict) else ""
        name = ""
        arguments: Any = {}

        if isinstance(tc, dict):
            if "name" in tc:
                name = str(tc.get("name", ""))
                arguments = _safe_json_loads(tc.get("arguments", {}))
            elif "function" in tc and isinstance(tc["function"], dict):
                name = str(tc["function"].get("name", ""))
                arguments = _safe_json_loads(tc["function"].get("arguments", {}))

        if not name:
            continue

        normalized.append({
            "id": tc_id,
            "name": name,
            "arguments": arguments if isinstance(arguments, dict) else {"value": arguments},
            "requestor": "assistant",
        })

    return normalized or None


def _normalize_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalize message list for skill context compatibility."""
    normalized: List[Dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if content is None:
            content = ""
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)

        out: Dict[str, Any] = {"role": role, "content": content}

        tool_calls = _normalize_tool_calls(msg.get("tool_calls"))
        if tool_calls:
            out["tool_calls"] = tool_calls

        # skill context trajectory model expects tool response id under key "id"
        if role == "tool" and msg.get("tool_call_id"):
            out["id"] = msg["tool_call_id"]

        normalized.append(out)

    return normalized


def _is_internal_tool(name: str) -> bool:
    """Check if a tool name is an internal/virtual tool (not user-facing)."""
    internal_prefixes = ("retrieve_tools", "__")
    return any(name.startswith(p) for p in internal_prefixes)


def _extract_tool_events(messages: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Extract tool call events and errors from message history."""
    tool_calls: List[Dict[str, Any]] = []
    tool_errors: List[Dict[str, Any]] = []
    step_index = 0

    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        msg_tool_calls = msg.get("tool_calls")
        if not msg_tool_calls:
            continue

        for tc in msg_tool_calls:
            if not isinstance(tc, dict):
                continue
            # Extract tool name
            name = ""
            arguments: Any = {}
            tc_id = tc.get("id", "")
            if "name" in tc:
                name = str(tc["name"])
                arguments = _safe_json_loads(tc.get("arguments", {}))
            elif "function" in tc and isinstance(tc["function"], dict):
                name = str(tc["function"].get("name", ""))
                arguments = _safe_json_loads(tc["function"].get("arguments", {}))

            if not name or _is_internal_tool(name):
                continue

            # Find the corresponding tool response
            tool_result = None
            has_error = False
            for resp_msg in messages:
                if resp_msg.get("role") == "tool" and resp_msg.get("tool_call_id") == tc_id:
                    tool_result = resp_msg.get("content", "")
                    if isinstance(tool_result, str):
                        has_error = "error" in tool_result.lower()[:200]
                    break

            event = {
                "step_index": step_index,
                "tool_name": name,
                "tool_call_id": tc_id,
                "arguments": _jsonable(arguments),
                "tool_result": _jsonable(tool_result),
                "has_error": has_error,
            }
            tool_calls.append(event)
            if has_error:
                tool_errors.append(event)
            step_index += 1

    return tool_calls, tool_errors


def _extract_retrieval_info(artifacts: Dict[str, Any]) -> Dict[str, Any]:
    """Extract skill retrieval information from rollout artifacts."""
    retrieval = artifacts.get("retrieval", {})
    skill_context = artifacts.get("skill_context", {})

    if not isinstance(retrieval, dict):
        retrieval = {}
    if not isinstance(skill_context, dict):
        skill_context = {}

    # From RAG retrieval rounds
    rounds = retrieval.get("rounds", [])
    if not isinstance(rounds, list):
        rounds = []

    final_visible_tool_names: List[str] = []
    for round_item in rounds:
        if not isinstance(round_item, dict):
            continue
        visible = round_item.get("final_visible_tool_names")
        if isinstance(visible, list):
            final_visible_tool_names = [str(n) for n in visible if isinstance(n, str)]

    # From skill context artifacts
    selected_skill_names = skill_context.get("selected_skill_names", [])
    if not isinstance(selected_skill_names, list):
        selected_skill_names = []
    skill_context_trace = skill_context.get("retrieval_trace", {})
    if not isinstance(skill_context_trace, dict):
        skill_context_trace = {}

    return {
        "strategy": retrieval.get("strategy"),
        "selected_skills": [{"name": n} for n in selected_skill_names],
        "final_visible_tool_names": final_visible_tool_names,
        "retrieval_trace": _jsonable(retrieval),
        # Keep the skill context trace separate from tool-RAG retrieval for evolution.
        "skill_context_retrieval_trace": _jsonable(skill_context_trace),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Index builders
# ─────────────────────────────────────────────────────────────────────────────


def _build_query_index(query_dir: Path) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """Index query records by (domain, sample_id)."""
    index: Dict[Tuple[str, str], Dict[str, Any]] = {}

    # Support both flat JSONL files and subdirectory structure
    jsonl_files = sorted(query_dir.glob("*.jsonl"))
    if not jsonl_files:
        # Try data/tasks/{split}/{domain}/*.json structure
        for domain_dir in sorted(query_dir.iterdir()):
            if not domain_dir.is_dir():
                continue
            domain = domain_dir.name
            for json_file in sorted(domain_dir.glob("*.json")):
                with json_file.open("r", encoding="utf-8") as f:
                    item = json.load(f)
                sample_id = str(item.get("id", json_file.stem))
                index[(domain, sample_id)] = item
    else:
        for fp in jsonl_files:
            domain = fp.stem
            for item in _read_jsonl(fp):
                sample_id = str(item.get("id", ""))
                if sample_id:
                    index[(domain, sample_id)] = item

    return index


def _build_eval_index(eval_file: Optional[Path]) -> Dict[str, Dict[str, Any]]:
    """Index evaluator results by sample_id.

    Supports two key formats from different evaluation outputs:
      - "id": legacy format
      - "sample_id": format from python -m dtau_bench evaluate (SampleEvalResult.to_dict())
    """
    if eval_file is None:
        raise ValueError(
            "eval_file is required for reliable skill reward signals; run the "
            "environment evaluator before bootstrap/evolve"
        )
    if not eval_file.is_file():
        raise FileNotFoundError(f"Evaluator results file not found: {eval_file}")

    index: Dict[str, Dict[str, Any]] = {}

    for line_number, item in enumerate(_read_jsonl(eval_file), start=1):
        # Support both "id" and "sample_id" key names
        sample_id = str(item.get("id") or item.get("sample_id") or "")
        if not sample_id:
            raise ValueError(
                f"Evaluator result at {eval_file}:{line_number} has no id/sample_id"
            )
        if sample_id in index:
            raise ValueError(f"Duplicate evaluator result for sample {sample_id!r}")
        if not isinstance(item.get("passed"), bool):
            raise ValueError(
                f"Evaluator result for sample {sample_id!r} has no boolean 'passed'"
            )
        index[sample_id] = item

    if not index:
        raise ValueError(f"Evaluator results file is empty: {eval_file}")

    return index


def _get_reward(
    sample_id: str,
    eval_index: Dict[str, Dict[str, Any]],
    rollout_item: Dict[str, Any],
    reward_mode: str,
) -> Tuple[float, Optional[bool], Optional[float]]:
    """Compute reward from the authoritative environment evaluation result."""
    ev = eval_index.get(sample_id)

    if ev is None:
        raise KeyError(
            f"Missing authoritative evaluator result for sample {sample_id!r}"
        )
    passed = ev["passed"]
    total_score = float(ev.get("total_score", 1.0 if passed else 0.0))

    if reward_mode == "total_score":
        reward = total_score
    elif reward_mode == "passed":
        reward = 1.0 if passed else 0.0
    else:
        raise ValueError(f"Unsupported reward mode: {reward_mode}")

    return reward, passed, total_score


# ─────────────────────────────────────────────────────────────────────────────
# Record builder
# ─────────────────────────────────────────────────────────────────────────────


def _build_skill_context_record(
    domain: str,
    rollout_item: Dict[str, Any],
    query_item: Optional[Dict[str, Any]],
    eval_index: Dict[str, Dict[str, Any]],
    reward_mode: str,
    include_successful_trajectory: bool,
    tool_attributes: Optional[Any] = None,
) -> Dict[str, Any]:
    """Build a single skill context-compatible record from a rollout item."""
    sample_id = str(rollout_item.get("id", ""))
    if not sample_id:
        raise ValueError(f"Rollout item missing id in domain {domain}")

    # Extract fields from rollout v2 schema
    task = rollout_item.get("task", {})
    if not isinstance(task, dict):
        task = {}
    conversation = rollout_item.get("conversation", {})
    if not isinstance(conversation, dict):
        conversation = {}
    run_info = rollout_item.get("run", {})
    if not isinstance(run_info, dict):
        run_info = {}
    artifacts = rollout_item.get("artifacts", {})
    if not isinstance(artifacts, dict):
        artifacts = {}

    messages = conversation.get("messages", [])
    if not isinstance(messages, list):
        messages = []

    # Normalize messages for skill context
    trajectory = _normalize_messages(messages)
    tool_calls, tool_errors = _extract_tool_events(messages)
    if tool_attributes is not None:
        version = run_info.get("version")
        for event in tool_calls:
            cost = tool_attributes.cost_for(domain, version, event.get("tool_name"))
            if cost is not None:
                event["cost"] = cost
        # Preserve error event identity while carrying cost into its evidence record.
        costs_by_step = {
            event.get("step_index"): event.get("cost")
            for event in tool_calls
            if "cost" in event
        }
        for event in tool_errors:
            cost = costs_by_step.get(event.get("step_index"))
            if cost is not None:
                event["cost"] = cost
    retrieval_info = _extract_retrieval_info(artifacts)

    # Get instruction
    instruction = str(task.get("instruction", "") or "")
    if not instruction:
        # Fallback: extract from first user message
        for msg in messages:
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str) and content:
                    instruction = content
                    break

    # Compute reward
    reward, passed, total_score = _get_reward(sample_id, eval_index, rollout_item, reward_mode)

    # Build record
    record: Dict[str, Any] = {
        "trajectory_id": f"{domain}:{sample_id}",
        "benchmark": "atomic",
        "task_id": sample_id,
        "query_id": str(task.get("query_id", sample_id) or sample_id),
        "version": run_info.get("version"),
        "domain": domain,
        "domains": task.get("domains") or [domain],
        "domain_key": task.get("domain_key"),
        "user_task": instruction,
        "trajectory": trajectory,
        "passed": passed,
        "reward": reward,
        "total_score": total_score,
        "expected_trajectory": list(task.get("trajectory", []) or []),
        "tool_calls": tool_calls,
        "tool_errors": tool_errors,
        "rollout_agent_traj": _jsonable(rollout_item.get("agent_traj", [])),
        "selected_skills": retrieval_info["selected_skills"],
        "final_visible_tools": retrieval_info["final_visible_tool_names"],
        "retrieval_trace": retrieval_info["retrieval_trace"],
        "skill_context_retrieval_trace": retrieval_info["skill_context_retrieval_trace"],
        "tool_schema_version": run_info.get("version"),
        "metadata": {
            "domain": domain,
            "task_domains": task.get("domains") or [],
            "task_domain_key": task.get("domain_key"),
            "query_id": task.get("query_id"),
            "trajectory_hash": task.get("trajectory_hash"),
            "combo_id": task.get("combo_id"),
            "source_id": sample_id,
            "passed": passed,
            "total_score": total_score,
            "rollout_success": bool(rollout_item.get("success", False)),
            "rollout_total_turns": conversation.get("total_turns"),
            "retrieval_strategy": retrieval_info["strategy"],
        },
    }

    # Add successful trajectory for extraction (only if reward > 0)
    if include_successful_trajectory and reward > 0.0:
        record["successful_trajectory"] = trajectory

    if tool_attributes is not None:
        from ..cost import build_cost_summary

        record["cost_summary"] = build_cost_summary(tool_calls)

    return record


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────


def build_skill_dataset(
    rollout_dir: Path,
    query_dir: Path,
    eval_file: Optional[Path],
    output_path: Path,
    expected_version: str,
    reward_mode: str = "passed",
    include_successful_trajectory: bool = True,
    tool_attributes_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Build the skill context-ready dataset from rollout artifacts.

    Args:
        rollout_dir: Directory containing rollout JSONL files (one per domain).
        query_dir: Directory containing query/task files.
        eval_file: Evaluator results JSONL file. It must contain exactly one
            authoritative result for every rollout record.
        output_path: Output JSONL file path.
        expected_version: Tool schema version this dataset is being built for.
        reward_mode: Reward computation mode ('passed' or 'total_score').
        include_successful_trajectory: Whether to include successful_trajectory field.

    Returns:
        Statistics dict with counts.
    """
    expected_version = str(expected_version or "").strip()
    if not expected_version:
        raise ValueError("expected_version is required")

    tool_attributes = None
    if tool_attributes_dir is not None:
        from ..cost import ToolAttributeLoader

        tool_attributes = ToolAttributeLoader(tool_attributes_dir)

    # Discover rollout files
    rollout_files = sorted(rollout_dir.glob("*.jsonl"))
    if not rollout_files:
        raise FileNotFoundError(f"No rollout JSONL files found in {rollout_dir}")

    # Read and validate all inputs before producing any output.  Skill
    # distillation must never fall back to a rollout transport-success flag.
    query_index = _build_query_index(query_dir)
    eval_index = _build_eval_index(eval_file)
    rollout_rows: List[Tuple[str, Dict[str, Any]]] = []
    rollout_ids: Set[str] = set()
    for rollout_file in rollout_files:
        domain = rollout_file.stem
        for item in _read_jsonl(rollout_file):
            sample_id = str(item.get("id") or "")
            if not sample_id:
                raise ValueError(f"Rollout record in {rollout_file} has no id")
            if sample_id in rollout_ids:
                raise ValueError(f"Duplicate rollout record for sample {sample_id!r}")
            rollout_ids.add(sample_id)
            rollout_rows.append((domain, item))

    if not rollout_rows:
        raise ValueError(f"Rollout directory contains no records: {rollout_dir}")

    eval_ids = set(eval_index)
    missing_eval = sorted(rollout_ids - eval_ids)
    extra_eval = sorted(eval_ids - rollout_ids)
    if missing_eval or extra_eval:
        raise ValueError(
            "Rollout/evaluator sample roster mismatch: "
            f"rollouts={len(rollout_ids)}, eval={len(eval_ids)}, "
            f"missing_eval={missing_eval[:5]}, extra_eval={extra_eval[:5]}"
        )

    # ``SampleEvalResult.to_dict()`` stores the evaluated catalog version in
    # the top-level ``version`` field, while rollout artifacts store it in
    # ``run.version``.  Matching only by sample id can otherwise attach a vN
    # reward to the same task rolled out under a different tool catalog.
    for _, rollout_item in rollout_rows:
        sample_id = str(rollout_item["id"])
        run_info = rollout_item.get("run")
        rollout_version = (
            str(run_info.get("version") or "").strip()
            if isinstance(run_info, dict)
            else ""
        )
        eval_version = str(eval_index[sample_id].get("version") or "").strip()
        if not rollout_version:
            raise ValueError(
                f"Rollout record for sample {sample_id!r} has no run.version"
            )
        if not eval_version:
            raise ValueError(
                f"Evaluator result for sample {sample_id!r} has no version"
            )
        if eval_version != rollout_version:
            raise ValueError(
                f"Evaluator/rollout version mismatch for sample {sample_id!r}: "
                f"eval={eval_version!r}, rollout={rollout_version!r}"
            )
        if rollout_version != expected_version:
            raise ValueError(
                f"Wrong catalog version for sample {sample_id!r}: "
                f"eval={eval_version!r}, rollout={rollout_version!r}, "
                f"expected={expected_version!r}"
            )

    # Process all rollout files
    records: List[Dict[str, Any]] = []
    missing_query_count = 0
    domains_seen: Set[str] = set()
    passed_count = 0

    for domain, rollout_item in rollout_rows:
        domains_seen.add(domain)
        sample_id = str(rollout_item["id"])

        query_item = query_index.get((domain, sample_id))
        if query_item is None:
            missing_query_count += 1
            continue

        # A transport-level failure without a conversation is not useful skill
        # evidence, but its evaluator row was still required above.
        if not rollout_item.get("success", False) and not rollout_item.get("conversation"):
            continue

        try:
            record = _build_skill_context_record(
                domain=domain,
                rollout_item=rollout_item,
                query_item=query_item,
                eval_index=eval_index,
                reward_mode=reward_mode,
                include_successful_trajectory=include_successful_trajectory,
                tool_attributes=tool_attributes,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to build skill record for {domain}:{sample_id}: {exc}"
            ) from exc
        records.append(record)
        if record.get("passed"):
            passed_count += 1

    if missing_query_count:
        raise ValueError(
            f"{missing_query_count} rollout records have no matching task definition"
        )

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_output = output_path.with_suffix(output_path.suffix + ".tmp")
    with tmp_output.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    tmp_output.replace(output_path)

    stats = {
        "total_records": len(records),
        "passed_records": passed_count,
        "domains": len(domains_seen),
        "domain_list": sorted(domains_seen),
        "missing_query_count": missing_query_count,
        "eval_file_used": True,
        "eval_records": len(eval_index),
        "cost_aware": tool_attributes is not None,
    }

    logger.info(f"Built {len(records)} records ({passed_count} passed) from {len(domains_seen)} domains")
    return stats
