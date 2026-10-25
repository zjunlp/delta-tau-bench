"""Evolve pipeline orchestrator — 6-step skill evolution with resume support.

Usage:
    config = EvolveConfig(...)
    pipeline = EvolvePipeline(config)
    asyncio.run(pipeline.run())
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from dtau_bench.utils.config import get_data_root

logger = logging.getLogger(__name__)


def _json_artifact_error(
    path: Path,
    required_keys: tuple[str, ...] = (),
    expected_values: Optional[Dict[str, Any]] = None,
    nonempty_keys: tuple[str, ...] = (),
) -> Optional[str]:
    """Return a diagnostic when a JSON stage artifact is missing or malformed."""
    if not path.is_file():
        return f"missing artifact: {path}"
    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return f"invalid JSON artifact {path}: {exc}"
    if not isinstance(payload, dict):
        return f"invalid JSON artifact {path}: expected an object"
    missing = [key for key in required_keys if key not in payload]
    if missing:
        return f"invalid JSON artifact {path}: missing keys {missing}"
    empty = [key for key in nonempty_keys if not payload.get(key)]
    if empty:
        return f"invalid JSON artifact {path}: empty keys {empty}"
    for key, expected in (expected_values or {}).items():
        if payload.get(key) != expected:
            return (
                f"invalid JSON artifact {path}: {key}={payload.get(key)!r}, "
                f"expected {expected!r}"
            )
    return None


def _jsonl_artifact_error(
    path: Path,
    *,
    require_nonempty: bool = False,
    expected_version: Optional[str] = None,
) -> Optional[str]:
    """Return a diagnostic when a JSONL artifact cannot be consumed safely."""
    if not path.is_file():
        return f"missing artifact: {path}"
    row_count = 0
    try:
        with path.open("r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    return f"invalid JSONL artifact {path}:{line_number}: {exc}"
                if not isinstance(row, dict):
                    return (
                        f"invalid JSONL artifact {path}:{line_number}: "
                        "expected an object"
                    )
                if expected_version is not None:
                    versions = [
                        str(row[key]).strip()
                        for key in ("version", "tool_schema_version")
                        if row.get(key) is not None
                    ]
                    if not versions or any(
                        version != expected_version for version in versions
                    ):
                        return (
                            f"wrong version in {path}:{line_number}: "
                            f"found={versions}, expected={expected_version!r}"
                        )
                row_count += 1
    except (OSError, UnicodeError) as exc:
        return f"invalid JSONL artifact {path}: {exc}"
    if require_nonempty and row_count == 0:
        return f"empty artifact: {path}"
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class EvolveConfig:
    """Full configuration for the skill evolution pipeline.

    Attributes:
        output_dir: Root output directory for evolved skill library.
        prev_library: Path to previous skill library directory.
        rollout_dir: Path to rollout output JSONL files (with eval results).
        eval_file: Optional evaluated skill context-ready JSONL (if not building from rollout).
        eval_results_file: Optional evaluation results JSONL from python -m dtau_bench evaluate.
            Used when building dataset from rollout_dir to provide authoritative
            passed/failed signals (instead of relying on rollout's success field).
        tool_schemas_dir: Path to tools directory (data/tools/).
        query_dir: Canonical training task directory used to build the dataset.
        tool_schema_version: Target tool version for active/inactive partition.
        model: LLM model key.
        base_url: LLM API base URL.
        api_key: LLM API key.
        embedding_api_base: Embedding API base URL.
        embedding_api_key: Embedding API key.
        embedding_model: Embedding model name.
        resume: Whether to resume from last completed step.
    """

    # Required paths
    output_dir: Path = field(default_factory=lambda: Path("experiments/evolve/skills"))
    prev_library: Optional[Path] = field(default=None)
    rollout_dir: Optional[Path] = None
    eval_file: Optional[Path] = None
    eval_results_file: Optional[Path] = None
    tool_schemas_dir: Path = field(default_factory=lambda: get_data_root() / "tools")
    tool_attributes_dir: Optional[Path] = None
    tool_schema_version: str = "v1"

    # LLM config
    model: str = "deepseek-v3"
    base_url: str = ""
    api_key: str = ""

    # Embedding config
    embedding_api_base: str = ""
    embedding_api_key: str = ""
    embedding_model: str = ""

    # Resume
    resume: bool = False

    # LLM parameters
    max_tokens: int = 40960
    temperature: float = 0.2
    max_retries: int = 3
    timeout: int = 600
    llm_request_options: Dict[str, Any] = field(default_factory=dict)

    # Batch processing
    batch_size: int = 8
    max_concurrent: int = 32

    # Embedding & clustering
    merge_embedding_batch_size: int = 64
    merge_similarity_threshold: float = 0.93

    # Graph construction: fraction of an atomic skill's tools that a plan must
    # provide for a plan→atomic edge to exist.
    plan_atomic_coverage_threshold: float = 0.5

    # Step 1: Feedback summary
    summarize_long_feedback: bool = True
    max_feedback_length: int = 1500

    # Step 2: Proposals
    max_modify_proposals_per_skill: int = 15

    # Step 4: Atomic finalize
    skip_general_filter: bool = False

    # Query dir (for building dataset from rollout)
    query_dir: Path = field(default_factory=lambda: get_data_root() / "tasks" / "train")

    # Quiet mode
    quiet: bool = False
    cost_aware_skills: bool = False


# ─────────────────────────────────────────────────────────────────────────────
# Run State Management
# ─────────────────────────────────────────────────────────────────────────────


STEPS = [
    "feedback_summary",
    "proposal_extraction",
    "tool_skill_finalize",
    "atomic_skill_finalize",
    "planning_propagation",
    "rebuild_graph_partition",
]


def _load_run_state(path: Path) -> Dict[str, Any]:
    """Load or initialize the pipeline run state."""
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "steps": {step: {"status": "pending"} for step in STEPS},
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def _save_run_state(path: Path, state: Dict[str, Any]) -> None:
    """Persist run state to JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)
    temporary.replace(path)


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────


class EvolvePipeline:
    """Orchestrates the 6-step skill evolution pipeline.

    Steps:
      1. Tool feedback summary (compress long tool results)
      2. Direct proposal extraction (add/modify/keep/skip per session)
      3. Tool skill finalize (merge proposals per tool → final tool skills)
      4. Atomic skill finalize (merge proposals + cluster new adds)
      5. Planning propagation (graph-driven plan updates)
      6. Rebuild skill graph + active/inactive partition
    """

    def __init__(self, config: EvolveConfig):
        self.config = config
        self.state_path = config.output_dir / "run_state.json"
        self.state = _load_run_state(self.state_path) if config.resume else {
            "steps": {step: {"status": "pending"} for step in STEPS},
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

    async def run(self) -> Dict[str, Any]:
        """Run all 6 steps, skipping completed ones if resume=True."""
        cfg = self.config
        cfg.output_dir.mkdir(parents=True, exist_ok=True)

        # Ensure we have input data
        input_path = self._resolve_input()

        # Initialize LLM
        from ..core.llm.client import LLM

        llm = LLM(
            model=cfg.model,
            api_key=cfg.api_key,
            base_url=cfg.base_url,
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
            max_retries=cfg.max_retries,
            timeout=cfg.timeout,
            request_options=cfg.llm_request_options,
        )

        # Load previous skill library artifacts
        from .proposals import load_evolution_artifacts

        artifacts = load_evolution_artifacts(cfg.prev_library)

        # ─────────────────────────────────────────────────────────────────
        # Step 1: Tool Feedback Summary
        # ─────────────────────────────────────────────────────────────────
        summarized_path = cfg.output_dir / "summarized_sessions.jsonl"

        if self._should_run("feedback_summary"):
            logger.info("[1/6] Tool feedback summary")
            from .proposals import run_feedback_summary

            await run_feedback_summary(
                input_path=input_path,
                output_path=summarized_path,
                artifacts=artifacts,
                llm=llm,
                config=cfg,
            )
            self._require_step_artifacts("feedback_summary")
            self._mark_completed("feedback_summary")
        else:
            logger.info("[1/6] Tool feedback summary — already completed, skipping")

        # ─────────────────────────────────────────────────────────────────
        # Step 2: Proposal Extraction
        # ─────────────────────────────────────────────────────────────────
        tool_proposals_path = cfg.output_dir / "tool_skill_modify_proposals.jsonl"
        atomic_proposals_path = cfg.output_dir / "atomic_skill_modify_proposals.jsonl"
        proposal_trace_path = cfg.output_dir / "proposal_trace.jsonl"

        if self._should_run("proposal_extraction"):
            logger.info("[2/6] Direct proposal extraction")
            from .proposals import run_proposal_extraction

            await run_proposal_extraction(
                summarized_path=summarized_path,
                tool_proposals_path=tool_proposals_path,
                atomic_proposals_path=atomic_proposals_path,
                trace_path=proposal_trace_path,
                artifacts=artifacts,
                llm=llm,
                config=cfg,
            )
            self._require_step_artifacts("proposal_extraction")
            self._mark_completed("proposal_extraction")
        else:
            logger.info("[2/6] Direct proposal extraction — already completed, skipping")

        # ─────────────────────────────────────────────────────────────────
        # Step 3: Tool Skill Finalize
        # ─────────────────────────────────────────────────────────────────
        tool_finalize_path = cfg.output_dir / "tool_skill_finalize_trace.jsonl"

        if self._should_run("tool_skill_finalize"):
            logger.info("[3/6] Tool skill finalize")
            from .finalize import run_tool_skill_finalize

            await run_tool_skill_finalize(
                tool_proposals_path=tool_proposals_path,
                trace_path=tool_finalize_path,
                output_dir=cfg.output_dir,
                artifacts=artifacts,
                llm=llm,
                config=cfg,
            )
            self._require_step_artifacts("tool_skill_finalize")
            self._mark_completed("tool_skill_finalize")
        else:
            logger.info("[3/6] Tool skill finalize — already completed, skipping")

        # ─────────────────────────────────────────────────────────────────
        # Step 4: Atomic Skill Finalize
        # ─────────────────────────────────────────────────────────────────
        atomic_finalize_path = cfg.output_dir / "atomic_skill_finalize_trace.jsonl"

        if self._should_run("atomic_skill_finalize"):
            logger.info("[4/6] Atomic skill finalize")
            from .finalize import run_atomic_skill_finalize

            await run_atomic_skill_finalize(
                atomic_proposals_path=atomic_proposals_path,
                trace_path=atomic_finalize_path,
                output_dir=cfg.output_dir,
                artifacts=artifacts,
                llm=llm,
                config=cfg,
            )
            self._require_step_artifacts("atomic_skill_finalize")
            self._mark_completed("atomic_skill_finalize")
        else:
            logger.info("[4/6] Atomic skill finalize — already completed, skipping")

        # ─────────────────────────────────────────────────────────────────
        # Step 5: Planning Propagation
        # ─────────────────────────────────────────────────────────────────
        if self._should_run("planning_propagation"):
            logger.info("[5/6] Planning propagation and finalize")
            from .propagation import run_planning_propagation

            await run_planning_propagation(
                atomic_proposals_path=atomic_proposals_path,
                output_dir=cfg.output_dir,
                artifacts=artifacts,
                llm=llm,
                config=cfg,
            )
            self._require_step_artifacts("planning_propagation")
            self._mark_completed("planning_propagation")
        else:
            logger.info("[5/6] Planning propagation — already completed, skipping")

        # ─────────────────────────────────────────────────────────────────
        # Step 6: Rebuild Skill Graph + Active/Inactive Partition
        # ─────────────────────────────────────────────────────────────────
        if self._should_run("rebuild_graph_partition"):
            logger.info("[6/6] Rebuild skill graph + partition")
            from .partition import run_rebuild_and_partition

            await run_rebuild_and_partition(
                output_dir=cfg.output_dir,
                artifacts=artifacts,
                config=cfg,
            )
            self._require_step_artifacts("rebuild_graph_partition")
            self._mark_completed("rebuild_graph_partition")
        else:
            logger.info("[6/6] Rebuild graph + partition — already completed, skipping")

        # ─────────────────────────────────────────────────────────────────
        # Write final summary
        # ─────────────────────────────────────────────────────────────────
        summary = self._build_summary(llm)
        summary_path = cfg.output_dir / "summary.json"
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        # Token usage
        token_path = cfg.output_dir / "token_usage.json"
        llm.token_tracker.save(
            str(token_path),
            extra_info={"stage": "skill_evolution", "model": cfg.model},
        )
        if not cfg.quiet:
            logger.info(f"Token Usage: {llm.token_tracker}")
            logger.info(f"Evolution complete. Output: {cfg.output_dir}")

        return summary

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _resolve_input(self) -> Path:
        """Resolve input data path: either eval_file or build from rollout.

        Resolution order:
          1. cfg.eval_file → pre-built skill context-ready JSONL (skip dataset construction)
          2. cfg.rollout_dir → build skill context-ready JSONL from rollout + eval_results_file
        """
        cfg = self.config
        if cfg.eval_file and cfg.eval_file.exists():
            error = _jsonl_artifact_error(
                cfg.eval_file,
                require_nonempty=True,
                expected_version=cfg.tool_schema_version,
            )
            if error is not None:
                raise ValueError(f"Invalid pre-built evolution dataset: {error}")
            return cfg.eval_file

        if cfg.rollout_dir is not None and cfg.eval_results_file is None:
            raise ValueError(
                "eval_results_file is required when building evolution input "
                "from rollout_dir"
            )

        # Build from rollout using bootstrap dataset builder
        dataset_path = cfg.output_dir / "skill_context_ready.jsonl"
        if dataset_path.exists() and cfg.resume:
            error = _jsonl_artifact_error(
                dataset_path,
                require_nonempty=True,
                expected_version=cfg.tool_schema_version,
            )
            if error is None:
                return dataset_path
            logger.warning(
                "Cached evolution dataset is invalid (%s); rebuilding it.", error
            )

        if cfg.rollout_dir is None or not cfg.rollout_dir.exists():
            raise FileNotFoundError(
                f"Either --eval-file must point to a valid JSONL, "
                f"or --rollout-dir must be a valid rollout output directory. "
                f"Got: eval_file={cfg.eval_file}, rollout_dir={cfg.rollout_dir}"
            )

        # Use bootstrap dataset builder
        from ..bootstrap.dataset import build_skill_dataset

        logger.info(f"Building skill context-ready dataset from rollout: {cfg.rollout_dir}")
        build_skill_dataset(
            rollout_dir=cfg.rollout_dir,
            query_dir=cfg.query_dir,
            eval_file=cfg.eval_results_file,
            output_path=dataset_path,
            expected_version=cfg.tool_schema_version,
            reward_mode="passed",
            include_successful_trajectory=True,
            tool_attributes_dir=(cfg.tool_attributes_dir if cfg.cost_aware_skills else None),
        )
        return dataset_path

    def _should_run(self, step_name: str) -> bool:
        """Revalidate completed checkpoints and self-heal broken ones on resume."""
        step_state = self.state.get("steps", {}).get(step_name, {})
        if step_state.get("status") == "completed" and self.config.resume:
            error = self._step_artifact_error(step_name)
            if error is None:
                return False
            logger.warning(
                "Completed evolution step %s has invalid artifacts (%s); rerunning it.",
                step_name,
                error,
            )
            self._invalidate_from(step_name, error, include_step=True)
            return True
        if self.config.resume:
            self._invalidate_from(
                step_name,
                f"upstream step {step_name!r} is being rerun",
                include_step=False,
            )
        return True

    def _step_artifact_error(self, step_name: str) -> Optional[str]:
        """Validate the artifacts required to safely skip one completed step."""
        output_dir = self.config.output_dir
        checks: List[Optional[str]] = []
        if step_name == "feedback_summary":
            checks.append(
                _jsonl_artifact_error(
                    output_dir / "summarized_sessions.jsonl",
                    require_nonempty=True,
                )
            )
        elif step_name == "proposal_extraction":
            for filename in (
                "tool_skill_modify_proposals.jsonl",
                "atomic_skill_modify_proposals.jsonl",
                "proposal_trace.jsonl",
            ):
                checks.append(_jsonl_artifact_error(output_dir / filename))
        elif step_name == "tool_skill_finalize":
            for filename in (
                "tool_skill_finalize_trace.jsonl",
                "finalized_tool_rows.jsonl",
            ):
                checks.append(_jsonl_artifact_error(output_dir / filename))
        elif step_name == "atomic_skill_finalize":
            for filename in (
                "atomic_skill_finalize_trace.jsonl",
                "finalized_atomic_modified.jsonl",
                "finalized_atomic_added.jsonl",
            ):
                checks.append(_jsonl_artifact_error(output_dir / filename))
        elif step_name == "planning_propagation":
            for filename in (
                "planning_propagation_proposals.jsonl",
                "planning_finalize_trace.jsonl",
                "planning_skills.finalized.jsonl",
            ):
                checks.append(_jsonl_artifact_error(output_dir / filename))
            checks.append(
                _json_artifact_error(
                    output_dir / "plan_library.json",
                    ("plan",),
                    nonempty_keys=("plan",),
                )
            )
        elif step_name == "rebuild_graph_partition":
            for filename in (
                "tool_skills.prepared.jsonl",
                "tool_skills.finalized.jsonl",
                "tool_skills.inactive.jsonl",
                "atomic_skills.prepared.jsonl",
                "atomic_skills.finalized.jsonl",
                "atomic_skills.inactive.jsonl",
            ):
                checks.append(
                    _jsonl_artifact_error(
                        output_dir / filename,
                        require_nonempty=(
                            filename == "atomic_skills.prepared.jsonl"
                        ),
                    )
                )
            checks.extend([
                _json_artifact_error(
                    output_dir / "library_metadata.json",
                    ("target_tool_schema_version",),
                    {
                        "target_tool_schema_version": (
                            self.config.tool_schema_version
                        )
                    },
                ),
                _json_artifact_error(
                    output_dir / "skill_graph.json",
                    ("nodes", "edges"),
                    nonempty_keys=("nodes",),
                ),
                _json_artifact_error(output_dir / "skill_graph_summary.json"),
            ])
        else:
            return f"unknown evolution step: {step_name}"
        return next((error for error in checks if error is not None), None)

    def _require_step_artifacts(self, step_name: str) -> None:
        error = self._step_artifact_error(step_name)
        if error is not None:
            raise RuntimeError(f"Step {step_name!r} produced invalid artifacts: {error}")

    def _invalidate_from(
        self,
        step_name: str,
        reason: str,
        *,
        include_step: bool,
    ) -> None:
        """Invalidate stale completed checkpoints without deleting their artifacts."""
        start = STEPS.index(step_name)
        first = start if include_step else start + 1
        changed = False
        now = datetime.now(timezone.utc).isoformat()
        steps = self.state.setdefault("steps", {})
        for name in STEPS[first:]:
            info = steps.setdefault(name, {"status": "pending"})
            if name != step_name and info.get("status") != "completed":
                continue
            info["status"] = "invalid"
            info["invalidated_at"] = now
            info["invalid_reason"] = (
                reason
                if name == step_name
                else f"upstream step {step_name!r} must be rerun"
            )
            changed = True
        if changed:
            _save_run_state(self.state_path, self.state)

    def _mark_completed(self, step_name: str) -> None:
        """Mark a step as completed and persist state."""
        self.state["steps"][step_name] = {
            "status": "completed",
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        _save_run_state(self.state_path, self.state)

    def _build_summary(self, llm: Any) -> Dict[str, Any]:
        """Build a summary of the evolution run from output files."""
        cfg = self.config
        summary: Dict[str, Any] = {
            "prev_library": str(cfg.prev_library),
            "output_dir": str(cfg.output_dir),
            "tool_schema_version": cfg.tool_schema_version,
            "model": cfg.model,
        }

        # Count outputs from trace files
        for trace_name, key in [
            ("tool_skill_modify_proposals.jsonl", "tool_proposal_count"),
            ("atomic_skill_modify_proposals.jsonl", "atomic_proposal_count"),
            ("tool_skill_finalize_trace.jsonl", "tool_finalized_count"),
            ("atomic_skill_finalize_trace.jsonl", "atomic_finalized_count"),
            ("planning_propagation_proposals.jsonl", "planning_propagation_count"),
            ("planning_finalize_trace.jsonl", "planning_modified_count"),
        ]:
            path = cfg.output_dir / trace_name
            if path.exists():
                with path.open() as f:
                    summary[key] = sum(1 for line in f if line.strip())

        return summary
