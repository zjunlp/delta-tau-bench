"""
Environment Evaluation Engine.

Provides parallel evaluation of agent rollouts against gold trajectories:
1. Gold Replay: Execute gold trajectory (step_param_codes) → gold DB state
2. Agent Replay: Re-execute agent tool calls → agent DB state
3. DB Diff: Compare gold vs agent states → pass/fail

"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from dtau_bench.evaluator.models import EvalSample, SampleEvalResult
from dtau_bench.evaluator.data_loader import TaskMetadataProvider
from dtau_bench.utils.config import get_data_root
from dtau_bench.evaluator.replay.executor import (
    DatabaseState,
    GoldReplayExecutor,
    capture_database_state,
)
from dtau_bench.evaluator.replay.agent_replay import (
    AgentReplayExecutor,
    AgentReplayResult,
)
from dtau_bench.evaluator.replay.diffing import (
    DiffMetadata,
)
from dtau_bench.evaluator.replay.evaluation_strategies import (
    EvaluationStrategyError,
    evaluate_by_strategy,
    load_identity_schema,
    parse_evaluation_config,
    validate_strategy_schema,
)

from dtau_bench.evaluator.replay.replay_identity import (
    action_source_errors, get_replay_contracts, load_replay_identity_schema,
)

from dtau_bench.evaluator.replay.state_compatibility import (
    get_state_contracts, load_state_compatibility_schema, parse_state_comparison, bind_state_targets,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data Models
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class EvalTaskSpec:
    """
    Pickle-safe evaluation task specification for worker processes.

    Contains all information needed to evaluate one sample without
    referencing any non-picklable objects (servers, etc.).
    """

    sample_id: str
    domain: str
    version: str
    instruction: str

    # Gold replay inputs
    gold_trajectory: Tuple[str, ...]
    step_param_codes: Tuple[str, ...]
    entity_context: Dict[str, Any]
    trajectory_hash: str

    # Agent replay inputs
    agent_steps: Tuple[Dict[str, Any], ...]

    # Metadata
    domains: Tuple[str, ...]
    evaluation: Dict[str, Any] = field(default_factory=lambda: {
        "schema_version": 1, "strategy": "default", "targets": [],
    })

    state_comparison: Optional[Dict[str, Any]] = None


@dataclass
class EvalTimings:
    """Timing breakdown for a single sample evaluation."""

    gold_replay_ms: float = 0.0
    agent_replay_ms: float = 0.0
    diff_ms: float = 0.0
    total_ms: float = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Task Building (Precheck)
# ─────────────────────────────────────────────────────────────────────────────


def build_eval_task_spec(
    sample: EvalSample,
    task_provider: TaskMetadataProvider,
) -> EvalTaskSpec:
    """
    Build a pickle-safe EvalTaskSpec from an EvalSample.

    Validates required fields and extracts agent steps from messages.
    Gold trajectory metadata (tools, step_param_codes, entity_context) is
    always sourced from the canonical task JSON via TaskMetadataProvider,
    ensuring correctness regardless of rollout JSONL content.

    Args:
        sample: EvalSample loaded from rollout output.
        task_provider: Provider for authoritative gold metadata (required).

    Returns:
        EvalTaskSpec ready for worker execution.

    Raises:
        ValueError: If required fields are missing or invalid.
    """
    # --- Resolve gold and evaluation metadata from the canonical task ---
    canonical_task = task_provider.get(sample.id)
    if not isinstance(canonical_task, dict):
        raise ValueError(f"Sample {sample.id}: not found in TaskMetadataProvider")
    metadata = canonical_task.get("metadata", {})
    if not isinstance(metadata, dict):
        raise EvaluationStrategyError("Canonical task metadata must be an object")
    if "evaluation" in metadata and not isinstance(metadata["evaluation"], dict):
        raise EvaluationStrategyError("Canonical metadata.evaluation must be an object")
    evaluation = parse_evaluation_config(metadata.get("evaluation"))
    gold_traj_meta = task_provider.get_gold_trajectory(sample.id)
    if gold_traj_meta is None:
        raise ValueError(
            f"Sample {sample.id}: not found in TaskMetadataProvider. "
            f"Ensure task JSON exists in data/tasks/ with matching query_id."
        )

    gold_trajectory = [str(t) for t in gold_traj_meta.get("tools", []) if t]
    step_param_codes = gold_traj_meta.get("step_param_codes", [])
    trajectory_hash = gold_traj_meta.get("trajectory_hash", "") or ""

    entity_context: Dict[str, Any] = {}
    entity_instances = gold_traj_meta.get("entity_instances")
    value_domain_samples = gold_traj_meta.get("value_domain_samples")
    if isinstance(entity_instances, dict):
        entity_context["entity_instances"] = entity_instances
    if isinstance(value_domain_samples, dict):
        entity_context["value_domain_samples"] = value_domain_samples

    # Validate gold trajectory
    if not gold_trajectory:
        raise ValueError(f"Sample {sample.id}: empty gold_trajectory in task JSON")

    # Validate step_param_codes
    if not step_param_codes:
        raise ValueError(f"Sample {sample.id}: missing step_param_codes in task JSON")
    if len(step_param_codes) != len(gold_trajectory):
        raise ValueError(
            f"Sample {sample.id}: step_param_codes length ({len(step_param_codes)}) "
            f"!= gold_trajectory length ({len(gold_trajectory)})"
        )

    # Validate domain
    if not sample.domains:
        raise ValueError(f"Sample {sample.id}: no domain specified")
    domain = canonical_task.get("domain")
    if not isinstance(domain, str) or not domain or sample.domains != [domain]:
        raise ValueError(f"Sample {sample.id}: rollout domain does not match canonical task")
    instruction = canonical_task.get("instruction")
    if not isinstance(instruction, str):
        raise ValueError(f"Sample {sample.id}: canonical instruction must be a string")
    if evaluation["strategy"] != "default" and sample.version not in canonical_task.get("version_trajectories", {}):
        raise EvaluationStrategyError(f"Task has no declared version: {sample.version}")

    # Extract agent steps from messages
    from dtau_bench.evaluator.actions import extract_action_records

    action_records = extract_action_records(sample.messages)
    source_errors = action_source_errors(sample.messages, action_records)
    agent_steps = tuple(
        {
            "tool_name": rec.tool_name,
            "arguments": rec.arguments,
            "tool_result": rec.tool_result_raw,
            "tool_call_id": rec.tool_call_id,
            "source_index": rec.index,
            "source_verified": rec.index not in source_errors,
            "source_error": source_errors.get(rec.index),
        }
        for rec in action_records
    )

    # Compute trajectory hash if not already present
    traj_hash = trajectory_hash
    if not traj_hash:
        traj_hash = hashlib.md5(
            json.dumps(gold_trajectory, sort_keys=True).encode()
        ).hexdigest()[:16]

    return EvalTaskSpec(
        sample_id=sample.id,
        domain=domain,
        version=sample.version,
        instruction=instruction,
        gold_trajectory=tuple(gold_trajectory),
        step_param_codes=tuple(step_param_codes),
        entity_context=entity_context,
        trajectory_hash=traj_hash,
        agent_steps=agent_steps,
        domains=(domain,),
        evaluation=evaluation,
        state_comparison=parse_state_comparison(canonical_task),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Worker Context (Module-level for ProcessPoolExecutor)
# ─────────────────────────────────────────────────────────────────────────────

# These module-level variables are set by _init_worker() in child processes
_worker_servers: Dict[str, Any] = {}
_worker_diff_meta_cache: Dict[str, DiffMetadata] = {}
_worker_field_meta: Dict[str, Any] = {}
_worker_data_dir: str = ""
_worker_confirmation_tools: Dict[str, Any] = {}
_worker_identity_schema: Optional[Dict[str, Any]] = None
_worker_replay_identity_schema: Optional[Dict[str, Any]] = None


def _init_worker(
    data_dir: str,
    field_meta: Dict[str, Any],
) -> None:
    """
    Initialize worker process globals.

    Called as ProcessPoolExecutor initializer. Sets up shared state
    that persists across tasks within the same worker process.

    Args:
        data_dir: Path to the data directory.
        field_meta: domain -> eval_field_meta entry.
    """
    global _worker_servers, _worker_diff_meta_cache, _worker_field_meta
    global _worker_data_dir, _worker_confirmation_tools, _worker_identity_schema
    global _worker_replay_identity_schema

    _worker_data_dir = data_dir
    _worker_field_meta = field_meta
    _worker_servers = {}
    _worker_diff_meta_cache = {}
    _worker_confirmation_tools = {}
    _worker_identity_schema = None
    _worker_replay_identity_schema = None


def _get_server(domain: str, version: str) -> Any:
    """
    Get or create an MCP server instance for a domain/version pair.

    Lazy-loads servers on first access within a worker process.
    """
    global _worker_servers

    cache_key = f"{domain}:{version}"
    if cache_key in _worker_servers:
        return _worker_servers[cache_key]

    from dtau_bench.tools.mcp_wrapper import setup_domain_tools

    server = setup_domain_tools(
        domain=domain,
        version=version,
        data_dir=Path(_worker_data_dir),
        inject_attributes=False,
    )
    _worker_servers[cache_key] = server
    return server


def _get_diff_metadata(domain: str) -> DiffMetadata:
    """
    Get or build diff metadata for a domain (cached per worker).

    Reads the pre-classified field sets of data/meta/eval_field_meta/{domain}.json.

    Raises:
        ValueError: If no metadata is available for the domain.
    """
    global _worker_diff_meta_cache, _worker_field_meta

    if domain in _worker_diff_meta_cache:
        return _worker_diff_meta_cache[domain]

    meta = _worker_field_meta.get(domain)
    if meta is None:
        raise ValueError(
            f"Domain '{domain}' not found in data/meta/eval_field_meta/. "
            f"Loaded domains: {sorted(_worker_field_meta)[:5]}..."
        )

    diff_meta = DiffMetadata(
        domain=domain,
        entity_id_names=set(meta.get("entity_id_names", [])),
        free_text_fields=set(meta.get("free_text_fields", [])),
        ignored_fields=set(meta.get("ignored_fields", [])),
    )
    _worker_diff_meta_cache[domain] = diff_meta
    return diff_meta


def _get_confirmation_tools(domain: str) -> Any:
    """Get or lazily load confirmation tools for a domain (cached per worker)."""
    global _worker_confirmation_tools

    if domain in _worker_confirmation_tools:
        return _worker_confirmation_tools[domain]

    from dtau_bench.tools.mcp_wrapper import get_confirmation_tools

    ct = get_confirmation_tools(domain, Path(_worker_data_dir))
    _worker_confirmation_tools[domain] = ct
    return ct


def _agent_replay_details(result: AgentReplayResult) -> Dict[str, Any]:
    return {
        "steps_attempted": result.steps_attempted,
        "steps_succeeded": result.steps_succeeded,
        "steps_total": result.steps_total,
        "steps_failed": result.steps_failed,
        "steps_previewed": result.steps_previewed,
        "steps_unknown": result.steps_unknown,
        "id_mapping_size": result.id_mapping_size,
        "errors": result.replay_errors[:5],
        "replay_complete": result.replay_complete,
        "tool_failures": result.tool_failures,
        "replay_integrity_errors": result.replay_integrity_errors,
        "identity_bindings": result.identity_bindings,
        **({"state_compatibility_evidence": result.state_compatibility_evidence}
           if result.state_compatibility_evidence is not None else {}),
    }


def _execute_eval_task(task: EvalTaskSpec) -> SampleEvalResult:
    """
    Execute a single evaluation task in a worker process.

    This is the core worker function passed to ProcessPoolExecutor.map().

    Steps:
    1. Gold Replay → initial_state + gold_state
    2. Agent Replay → agent_state
    3. DB Diff → pass/fail

    Args:
        task: EvalTaskSpec with all required inputs.

    Returns:
        SampleEvalResult with pass/fail and details.
    """
    t_start = time.perf_counter()
    timings = EvalTimings()

    try:
        # Validate the declared route before running any tools.
        evaluation = parse_evaluation_config(task.evaluation)
        identity_schema = None
        if evaluation["strategy"] != "default":
            global _worker_identity_schema
            if task.version not in {"base", "v1", "v2", "v3", "v4", "v5", "v6"}:
                raise EvaluationStrategyError(f"Unsupported evaluation version: {task.version}")
            if _worker_identity_schema is None:
                _worker_identity_schema = load_identity_schema(Path(_worker_data_dir))
            identity_schema = _worker_identity_schema
            validate_strategy_schema(evaluation, task.domain, identity_schema)

        # Transport contracts apply independently of the final comparison route.
        global _worker_replay_identity_schema
        if _worker_replay_identity_schema is None:
            _worker_replay_identity_schema = load_replay_identity_schema(Path(_worker_data_dir))
        replay_contracts = get_replay_contracts(_worker_replay_identity_schema, task.domain, task.version)

        state_contracts = get_state_contracts(
            load_state_compatibility_schema(Path(_worker_data_dir)), task.domain, task.version, task.state_comparison,
        )

        if state_contracts.get("audited") and identity_schema is None:
            if _worker_identity_schema is None:
                _worker_identity_schema = load_identity_schema(Path(_worker_data_dir))
            identity_schema = _worker_identity_schema

        # --- Step 1: Gold Replay ---
        t_gold_start = time.perf_counter()

        # Replay this sample's base reference with its own parameters and context.
        gold_server = _get_server(task.domain, "base")
        confirm_tools = _get_confirmation_tools(task.domain)
        executor = GoldReplayExecutor(gold_server, confirmation_tools=confirm_tools)
        gold_result = executor.execute(
            trajectory=list(task.gold_trajectory),
            step_param_codes=list(task.step_param_codes),
            entity_context=dict(task.entity_context),
        )
        initial_state = gold_result.initial_state or DatabaseState()
        gold_state = gold_result.final_state or DatabaseState()

        timings.gold_replay_ms = (time.perf_counter() - t_gold_start) * 1000

        # Check gold replay success
        if not gold_result.success:
            return SampleEvalResult(
                sample_id=task.sample_id,
                version=task.version,
                passed=False,
                error=f"Gold replay failed: {'; '.join(gold_result.errors[:3])}",
                gold_trajectory=list(task.gold_trajectory),
                agent_traj=[s["tool_name"] for s in task.agent_steps],
                domains=list(task.domains),
                domain_key="_".join(task.domains),
                details={
                    "gold_replay_errors": gold_result.errors,
                    "gold_steps_succeeded": gold_result.success_count,
                    "gold_steps_total": gold_result.step_count,
                    "timings": {
                        "gold_replay_ms": timings.gold_replay_ms,
                        "total_ms": (time.perf_counter() - t_start) * 1000,
                    },
                },
            )

        state_contracts = bind_state_targets(
            state_contracts, gold_result.steps, initial_state, gold_state,
        )

        # --- Step 2: Agent Replay ---
        t_agent_start = time.perf_counter()

        if not task.agent_steps:
            # No agent actions → automatic fail (agent did nothing)
            return SampleEvalResult(
                sample_id=task.sample_id,
                version=task.version,
                passed=False,
                error="No agent actions to replay",
                gold_trajectory=list(task.gold_trajectory),
                agent_traj=[],
                domains=list(task.domains),
                domain_key="_".join(task.domains),
                details={"reason": "empty_agent_traj"},
            )

        # Agent replay uses the evaluation version's server
        agent_server = _get_server(task.domain, task.version)

        # Get entity ID names for ID mapping
        diff_meta = _get_diff_metadata(task.domain)
        entity_id_names = diff_meta.entity_id_names

        agent_executor = AgentReplayExecutor(agent_server, domain=task.domain, contracts=replay_contracts,
                                              state_contracts=state_contracts)
        agent_result = agent_executor.execute(
            agent_steps=list(task.agent_steps),
            entity_id_names=entity_id_names,
        )
        agent_state = agent_result.final_state or DatabaseState()

        timings.agent_replay_ms = (time.perf_counter() - t_agent_start) * 1000

        if not agent_result.replay_complete:
            timings.total_ms = (time.perf_counter() - t_start) * 1000
            return SampleEvalResult(
                sample_id=task.sample_id, version=task.version, passed=False,
                error="Agent replay integrity error: " + "; ".join(
                    entry["error"] for entry in agent_result.replay_integrity_errors[:3]),
                gold_trajectory=list(task.gold_trajectory),
                agent_traj=[step["tool_name"] for step in task.agent_steps],
                domains=list(task.domains), domain_key="_".join(task.domains),
                details={
                    "reason": "agent_replay_integrity_error",
                    "agent_replay": _agent_replay_details(agent_result),
                    "gold_replay": {"success": gold_result.success,
                                    "steps_succeeded": gold_result.success_count,
                                    "steps_total": gold_result.step_count},
                    "timings": {"gold_replay_ms": round(timings.gold_replay_ms, 2),
                                "agent_replay_ms": round(timings.agent_replay_ms, 2),
                                "total_ms": round(timings.total_ms, 2)},
                },
            )

        # --- Step 3: DB Diff ---
        t_diff_start = time.perf_counter()

        diff_result = evaluate_by_strategy(
            initial_state=initial_state,
            gold_state=gold_state,
            agent_state=agent_state,
            diff_meta=diff_meta,
            evaluation=evaluation,
            identity_schema=identity_schema,
            agent_initial_state=agent_result.initial_state,
            state_contracts=state_contracts,
            state_evidence=agent_result.state_compatibility_evidence,
        )

        timings.diff_ms = (time.perf_counter() - t_diff_start) * 1000
        timings.total_ms = (time.perf_counter() - t_start) * 1000

        # Determine pass/fail
        gold_diff_total = diff_result.get("stats", {}).get("gold_diff_total", 0)
        if evaluation["strategy"] == "default":
            passed = gold_diff_total == 0 or diff_result.get("passed", False)
        else:
            # Identity validation must never be bypassed by empty/missing stats.
            passed = diff_result.get("passed") is True
        if state_contracts:
            # The legacy empty-gold shortcut cannot bypass compatibility guards.
            passed = passed and diff_result.get("state_compatibility", {}).get("passed") is True

        return SampleEvalResult(
            sample_id=task.sample_id,
            version=task.version,
            passed=passed,
            error=None if passed else diff_result.get("reason", "Unknown diff failure"),
            gold_trajectory=list(task.gold_trajectory),
            agent_traj=[s["tool_name"] for s in task.agent_steps],
            domains=list(task.domains),
            domain_key="_".join(task.domains),
            details={
                "diff_result": {
                    "passed": diff_result.get("passed"),
                    "reason": diff_result.get("reason", ""),
                    "stats": diff_result.get("stats", {}),
                    "missing_changes_count": len(diff_result.get("missing_changes", [])),
                    "missing_changes": diff_result.get("missing_changes", []),
                    **({"identity_comparison": diff_result["identity_comparison"]}
                       if "identity_comparison" in diff_result else {}),
                    **({"state_compatibility": diff_result["state_compatibility"]}
                       if "state_compatibility" in diff_result else {}),
                },
                "gold_replay": {
                    "success": gold_result.success,
                    "steps_succeeded": gold_result.success_count,
                    "steps_total": gold_result.step_count,
                },
                "agent_replay": _agent_replay_details(agent_result),
                "timings": {
                    "gold_replay_ms": round(timings.gold_replay_ms, 2),
                    "agent_replay_ms": round(timings.agent_replay_ms, 2),
                    "diff_ms": round(timings.diff_ms, 2),
                    "total_ms": round(timings.total_ms, 2),
                },
            },
        )

    except Exception as e:
        timings.total_ms = (time.perf_counter() - t_start) * 1000
        error_msg = f"Evaluation error: {type(e).__name__}: {e}"
        logger.error(f"Task {task.sample_id}: {error_msg}\n{traceback.format_exc()}")
        return SampleEvalResult(
            sample_id=task.sample_id,
            version=task.version,
            passed=False,
            error=error_msg,
            gold_trajectory=list(task.gold_trajectory),
            agent_traj=[s.get("tool_name", "") for s in task.agent_steps],
            domains=list(task.domains),
            domain_key="_".join(task.domains),
            details={
                "exception": str(e),
                "traceback": traceback.format_exc()[-500:],
                "timings": {"total_ms": round(timings.total_ms, 2)},
            },
        )


# ─────────────────────────────────────────────────────────────────────────────
# Main Engine
# ─────────────────────────────────────────────────────────────────────────────


class EnvironmentEvaluationEngine:
    """
    Parallel environment-based evaluation engine.

    Evaluates agent rollout samples by:
    1. Replaying gold trajectories to get expected DB states
    2. Replaying agent tool calls to get actual DB states
    3. Comparing states to determine pass/fail

    Supports:
    - Multi-process parallelism (ProcessPoolExecutor)
    - Independent gold replay for each sample
    - Graceful error isolation (one sample failure doesn't affect others)

    Usage:
        engine = EnvironmentEvaluationEngine(task_provider=task_provider)
        results = engine.evaluate(samples, version="v2", workers=8)
    """

    def __init__(
        self,
        task_provider: TaskMetadataProvider,
        data_dir: Optional[str | Path] = None,
        verbose: bool = False,
    ):
        """
        Initialize evaluation engine.

        Args:
            task_provider: TaskMetadataProvider for authoritative gold metadata.
                          Gold trajectory info (tools, step_param_codes, entity_context)
                          is sourced from canonical task JSON files.
            data_dir: Root data directory containing servers/, tools/, database/, meta/.
            verbose: Enable verbose logging.
        """
        self.task_provider = task_provider
        self.data_dir = str(get_data_root(data_dir))
        self.verbose = verbose
        self.field_meta: Dict[str, Any] = {}
        self._load_eval_field_meta()

    def _load_eval_field_meta(self) -> None:
        """
        Load the per-domain field classification used by the DB diff.

        Each data/meta/eval_field_meta/{domain}.json provides entity_id_names,
        free_text_fields and ignored_fields.

        Raises:
            FileNotFoundError: If the directory is missing or has no valid JSON.
        """
        meta_dir = Path(self.data_dir) / "meta" / "eval_field_meta"
        if not meta_dir.is_dir():
            raise FileNotFoundError(
                f"Evaluation field metadata not found at {meta_dir}."
            )

        for meta_file in sorted(meta_dir.glob("*.json")):
            try:
                with open(meta_file, "r", encoding="utf-8") as f:
                    self.field_meta[meta_file.stem] = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Failed to load eval_field_meta %s: %s", meta_file, e)

        if not self.field_meta:
            raise FileNotFoundError(
                f"{meta_dir} contains no valid JSON."
            )
        logger.info("Loaded eval_field_meta for %d domains", len(self.field_meta))

    def evaluate(
        self,
        samples: List[EvalSample],
        version: str,
        workers: int = 8,
    ) -> List[SampleEvalResult]:
        """
        Evaluate samples in parallel using process pool.

        Args:
            samples: List of EvalSample objects to evaluate.
            version: Tool version being evaluated (e.g., "v2").
            workers: Number of parallel worker processes.
        Returns:
            List of SampleEvalResult (same order as input samples).
        """
        if not samples:
            return []

        logger.info(
            "Starting evaluation: %d samples, version=%s, workers=%d",
            len(samples), version, workers,
        )

        # Phase 1: Precheck — build task specs and catch invalid samples early
        tasks: List[Optional[EvalTaskSpec]] = []
        precheck_errors: List[Optional[SampleEvalResult]] = []

        for sample in samples:
            try:
                task_spec = build_eval_task_spec(sample, task_provider=self.task_provider)
                tasks.append(task_spec)
                precheck_errors.append(None)
            except (ValueError, KeyError, TypeError) as e:
                # A rejected rollout must not move its failure to a self-reported domain.
                canonical = self.task_provider.get(sample.id)
                canonical_domain = canonical.get("domain") if isinstance(canonical, dict) else None
                error_domains = ([canonical_domain] if isinstance(canonical_domain, str) and canonical_domain
                                 else sample.domains)
                tasks.append(None)
                precheck_errors.append(
                    SampleEvalResult(
                        sample_id=sample.id,
                        version=version,
                        passed=False,
                        error=f"Precheck failed: {e}",
                        gold_trajectory=sample.gold_trajectory,
                        agent_traj=sample.agent_traj,
                        domains=error_domains,
                        domain_key="_".join(error_domains) if error_domains else "",
                        details={"precheck_error": str(e)},
                    )
                )

        valid_tasks = [t for t in tasks if t is not None]
        precheck_pass_count = len(valid_tasks)
        precheck_fail_count = len(samples) - precheck_pass_count

        if precheck_fail_count > 0:
            logger.warning(
                "Precheck: %d/%d samples failed validation",
                precheck_fail_count, len(samples),
            )

        if not valid_tasks:
            logger.error("No valid tasks after precheck")
            return [pe for pe in precheck_errors if pe is not None]

        # Phase 2: Execute evaluation
        if workers <= 1:
            results_map = self._evaluate_sequential(valid_tasks)
        else:
            results_map = self._evaluate_parallel(valid_tasks, workers)

        # Phase 3: Merge results in original sample order
        final_results: List[SampleEvalResult] = []
        for i, (task, precheck_err) in enumerate(zip(tasks, precheck_errors)):
            if precheck_err is not None:
                final_results.append(precheck_err)
            elif task is not None:
                result = results_map.get(task.sample_id)
                if result is None:
                    # Should not happen, but defensive
                    final_results.append(
                        SampleEvalResult(
                            sample_id=task.sample_id,
                            version=version,
                            passed=False,
                            error="Result missing after evaluation",
                            domains=list(task.domains),
                            domain_key="_".join(task.domains),
                        )
                    )
                else:
                    final_results.append(result)

        # Log summary
        passed = sum(1 for r in final_results if r.passed)
        failed = len(final_results) - passed
        logger.info(
            "Evaluation complete: %d passed, %d failed (%.1f%% pass rate)",
            passed, failed, 100.0 * passed / len(final_results) if final_results else 0,
        )

        return final_results

    def _evaluate_sequential(
        self,
        tasks: List[EvalTaskSpec],
    ) -> Dict[str, SampleEvalResult]:
        """Execute tasks sequentially in the current process."""
        # Initialize worker state in current process
        _init_worker(self.data_dir, self.field_meta)

        results: Dict[str, SampleEvalResult] = {}
        for i, task in enumerate(tasks):
            if self.verbose:
                logger.info(
                    "Evaluating [%d/%d] %s ...", i + 1, len(tasks), task.sample_id
                )
            result = _execute_eval_task(task)
            results[task.sample_id] = result
        return results

    def _evaluate_parallel(
        self,
        tasks: List[EvalTaskSpec],
        workers: int,
    ) -> Dict[str, SampleEvalResult]:
        """Execute tasks in parallel using ProcessPoolExecutor."""
        results: Dict[str, SampleEvalResult] = {}

        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_worker,
            initargs=(self.data_dir, self.field_meta),
        ) as pool:
            future_to_id = {
                pool.submit(_execute_eval_task, task): task.sample_id
                for task in tasks
            }

            completed = 0
            for future in as_completed(future_to_id):
                sample_id = future_to_id[future]
                completed += 1
                try:
                    result = future.result()
                    results[sample_id] = result
                except Exception as e:
                    logger.error(f"Worker exception for {sample_id}: {e}")
                    results[sample_id] = SampleEvalResult(
                        sample_id=sample_id,
                        version=tasks[0].version if tasks else "",
                        passed=False,
                        error=f"Worker exception: {e}",
                    )

                if self.verbose and completed % 10 == 0:
                    logger.info("Progress: %d/%d completed", completed, len(tasks))

        return results

    def evaluate_single(
        self,
        sample: EvalSample,
        version: str,
    ) -> SampleEvalResult:
        """
        Evaluate a single sample (convenience method, sequential).

        Args:
            sample: Single EvalSample to evaluate.
            version: Tool version.

        Returns:
            SampleEvalResult.
        """
        results = self.evaluate([sample], version=version, workers=1)
        return results[0]
