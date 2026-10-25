"""Bootstrap pipeline orchestrator — runs 4 stages sequentially with resume support.

Usage:
    config = BootstrapConfig(...)
    pipeline = BootstrapPipeline(config)
    pipeline.run()
"""

from __future__ import annotations

import json
import logging
import shutil
import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

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
class BootstrapConfig:
    """Configuration for the bootstrap pipeline.

    Paths, the LLM/embedding endpoints and the resume flag come from the CLI;
    the remaining fields are pipeline hyperparameters with fixed defaults.
    """

    # Required paths
    output_dir: Path
    rollout_dir: Path
    query_dir: Path
    tool_schemas_dir: Path
    tool_schema_version: str
    model: str
    base_url: str
    api_key: str
    embedding_api_base: str
    embedding_model: str

    # Optional paths
    eval_file: Optional[Path] = None
    tool_attributes_dir: Optional[Path] = None
    embedding_api_key: str = ""
    llm_request_options: Dict[str, Any] = field(default_factory=dict)

    # Resume
    resume: bool = False

    # LLM parameters
    max_tokens: int = 40960
    temperature: float = 0.2
    max_retries: int = 3
    timeout: int = 600

    # Batch processing
    batch_size: int = 8
    max_concurrent: int = 32

    # Embedding & clustering
    merge_embedding_batch_size: int = 64
    merge_similarity_threshold: float = 0.93

    # Graph construction: fraction of an atomic skill's tools that a plan must
    # provide for a plan→atomic edge to exist.
    plan_atomic_coverage_threshold: float = 0.5

    # Stage 1 (dataset)
    reward_mode: str = "passed"
    include_successful_trajectory: bool = True

    # Stage 2 (extraction)
    run_atomic: bool = True
    run_functional: bool = False
    skip_general_filter: bool = False
    concurrent_plan_combine: bool = True
    max_feedback_length: int = 1500
    filter_threshold: float = 0.999
    # Path to an existing plan_library.json to load instead of re-extracting the
    # planning layer. Plan extraction and atomic extraction live in one
    # indivisible stage, so a stage that dies after the plans are written (e.g.
    # on an exhausted API balance) would otherwise force paying for the plans a
    # second time. Reusing them also lets the two layers be distilled by
    # different models on purpose: the plans stay whatever model produced them
    # while a cheaper model summarises the atomic layer. Both layers are still
    # derived from the same trajectories, so the knowledge source is unchanged.
    reuse_plan_library: Optional[str] = None

    # Stage 3 (tool skills)
    max_facets_per_tool: int = 15
    min_distinct_tasks: int = 2
    min_error_events: int = 3
    max_examples_per_group: int = 6
    cost_aware_skills: bool = False


# ─────────────────────────────────────────────────────────────────────────────
# Run State Management (simple JSON-based resume)
# ─────────────────────────────────────────────────────────────────────────────


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class _RunState:
    """Tracks pipeline stage progress for resume support."""

    STAGES = ["dataset", "planning_atomic_extraction", "tool_skill_extraction", "graph_construction"]

    def __init__(self, state_path: Path, resume: bool = False):
        self._path = state_path
        if state_path.exists() and resume:
            with state_path.open("r", encoding="utf-8") as f:
                self._state = json.load(f)
        elif state_path.exists() and not resume:
            raise FileExistsError(
                f"Run state already exists at {state_path}. "
                "Pass --resume to continue, or delete the output directory."
            )
        else:
            self._state = {
                "schema_version": 1,
                "run_type": "bootstrap",
                "status": "running",
                "started_at": _now_iso(),
                "stages": {},
            }
            self._save()

    def should_run(self, stage: str) -> bool:
        """Check if a stage should be run (not already completed)."""
        stage_info = self._state.get("stages", {}).get(stage, {})
        return stage_info.get("status") != "completed"

    def mark_running(self, stage: str) -> None:
        """Mark a stage as running."""
        stages = self._state.setdefault("stages", {})
        stages[stage] = {
            "status": "running",
            "started_at": _now_iso(),
        }
        self._state["current_stage"] = stage
        self._state["status"] = "running"
        self._save()

    def mark_invalid(
        self,
        stage: str,
        reason: str,
        dependents: List[str],
    ) -> None:
        """Invalidate a completed stage and any completed dependent stages."""
        stages = self._state.setdefault("stages", {})
        targets = [stage] + [
            name
            for name in dependents
            if stages.get(name, {}).get("status") == "completed"
        ]
        now = _now_iso()
        for name in targets:
            info = stages.setdefault(name, {})
            info["status"] = "invalid"
            info["invalidated_at"] = now
            info["invalid_reason"] = (
                reason if name == stage else f"upstream stage {stage!r} must be rerun"
            )
        self._state["status"] = "running"
        self._save()

    def invalidate_completed_dependents(
        self,
        stage: str,
        dependents: List[str],
    ) -> None:
        """Prevent stale downstream checkpoints when an upstream stage reruns."""
        stages = self._state.setdefault("stages", {})
        completed = [
            name
            for name in dependents
            if stages.get(name, {}).get("status") == "completed"
        ]
        if not completed:
            return
        now = _now_iso()
        for name in completed:
            info = stages[name]
            info["status"] = "invalid"
            info["invalidated_at"] = now
            info["invalid_reason"] = f"upstream stage {stage!r} is being rerun"
        self._state["status"] = "running"
        self._save()

    def mark_completed(self, stage: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Mark a stage as completed."""
        stages = self._state.setdefault("stages", {})
        stage_info = stages.setdefault(stage, {})
        stage_info["status"] = "completed"
        stage_info["completed_at"] = _now_iso()
        if metadata:
            stage_info["metadata"] = metadata
        self._state["current_stage"] = None
        self._save()

    def mark_failed(self, stage: str, error: str) -> None:
        """Mark a stage as failed."""
        stages = self._state.setdefault("stages", {})
        stage_info = stages.setdefault(stage, {})
        stage_info["status"] = "failed"
        stage_info["error"] = error
        stage_info["failed_at"] = _now_iso()
        self._state["status"] = "failed"
        self._save()

    def mark_all_completed(self) -> None:
        """Mark the entire run as completed."""
        self._state["status"] = "completed"
        self._state["completed_at"] = _now_iso()
        self._save()

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._state["updated_at"] = _now_iso()
        temporary = self._path.with_suffix(self._path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as f:
            json.dump(self._state, f, ensure_ascii=False, indent=2)
        temporary.replace(self._path)


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline Orchestrator
# ─────────────────────────────────────────────────────────────────────────────


class BootstrapPipeline:
    """Orchestrates the 4-stage bootstrap pipeline.

    Stages:
        1. Dataset:     Merge rollout + eval + queries → skill_context_ready.jsonl
        2. Extraction:  Plan + Atomic/Functional skill extraction (LLM + clustering + merge + filter)
        3. Tool Skills: Tool skill extraction from evidence (LLM facet extraction + merge)
        4. Graph:       Skill graph construction (embedding similarity + tool overlap → edges)

    Supports --resume: skips already-completed stages.
    """

    def __init__(self, config: BootstrapConfig):
        self.config = config
        self._output_dir = Path(config.output_dir)

        # Sub-directories
        self._dataset_dir = self._output_dir / "dataset"
        self._extraction_dir = self._output_dir / "extraction"
        self._tool_skill_dir = self._output_dir / "tool_skills"
        self._library_dir = self._output_dir / "library"

    _STAGE_DEPENDENTS = {
        "dataset": [
            "planning_atomic_extraction",
            "tool_skill_extraction",
            "graph_construction",
        ],
        "planning_atomic_extraction": ["graph_construction"],
        "tool_skill_extraction": ["graph_construction"],
        "graph_construction": [],
    }

    def _stage_artifact_error(self, stage: str) -> Optional[str]:
        """Validate artifacts that make a completed stage safe to skip."""
        checks: List[Optional[str]] = []
        if stage == "dataset":
            checks.append(
                _jsonl_artifact_error(
                    self._dataset_dir / "skill_context_ready.jsonl",
                    require_nonempty=True,
                    expected_version=self.config.tool_schema_version,
                )
            )
        elif stage == "planning_atomic_extraction":
            checks.extend([
                _json_artifact_error(
                    self._extraction_dir / "plan_library.json",
                    ("plan",),
                    nonempty_keys=("plan",),
                ),
                _json_artifact_error(self._extraction_dir / "summary.json"),
                _json_artifact_error(self._extraction_dir / "token_usage.json"),
            ])
            if self.config.run_atomic:
                for filename in (
                    "atomic_skills.raw.jsonl",
                    "atomic_skills.merge_input.jsonl",
                    "atomic_skills.merged.jsonl",
                    "atomic_skills.filtered.jsonl",
                    "atomic_skills.prepared.jsonl",
                ):
                    checks.append(
                        _jsonl_artifact_error(
                            self._extraction_dir / filename,
                            require_nonempty=(
                                filename == "atomic_skills.prepared.jsonl"
                            ),
                        )
                    )
        elif stage == "tool_skill_extraction":
            for filename in (
                "tool_skill_groups.jsonl",
                "tool_skill_facets.raw.jsonl",
                "tool_skills.merged.jsonl",
                "tool_skills.prepared.jsonl",
            ):
                checks.append(_jsonl_artifact_error(self._tool_skill_dir / filename))
            checks.extend([
                _json_artifact_error(self._tool_skill_dir / "summary.json"),
                _json_artifact_error(self._tool_skill_dir / "token_usage.json"),
            ])
        elif stage == "graph_construction":
            checks.extend([
                _json_artifact_error(
                    self._library_dir / "skill_graph.json",
                    ("nodes", "edges"),
                    nonempty_keys=("nodes",),
                ),
                _json_artifact_error(self._library_dir / "skill_graph_summary.json"),
                _json_artifact_error(
                    self._library_dir / "library_metadata.json",
                    ("tool_schema_version",),
                    {"tool_schema_version": self.config.tool_schema_version},
                ),
            ])
        else:
            return f"unknown bootstrap stage: {stage}"
        return next((error for error in checks if error is not None), None)

    def _stage_should_run(self, state: _RunState, stage: str) -> bool:
        """Revalidate completed checkpoints and self-heal broken ones on resume."""
        dependents = self._STAGE_DEPENDENTS[stage]
        if state.should_run(stage):
            if self.config.resume:
                state.invalidate_completed_dependents(stage, dependents)
            return True

        error = self._stage_artifact_error(stage)
        if error is None:
            return False

        logger.warning(
            "Completed bootstrap stage %s has invalid artifacts (%s); rerunning it.",
            stage,
            error,
        )
        state.mark_invalid(stage, error, dependents)
        return True

    def _require_stage_artifacts(self, stage: str) -> None:
        error = self._stage_artifact_error(stage)
        if error is not None:
            raise RuntimeError(f"Stage {stage!r} produced invalid artifacts: {error}")

    def run(self) -> Dict[str, Any]:
        """Run the full bootstrap pipeline.

        Returns:
            Manifest dict describing the pipeline run and outputs.
        """
        # Initialize directories
        for d in (self._dataset_dir, self._extraction_dir, self._tool_skill_dir, self._library_dir):
            d.mkdir(parents=True, exist_ok=True)

        # State tracking
        state = _RunState(self._output_dir / "run_state.json", resume=self.config.resume)

        manifest: Dict[str, Any] = {
            "mode": "bootstrap",
            "started_at": _now_iso(),
            "status": "running",
            "config": {
                "rollout_dir": str(self.config.rollout_dir),
                "query_dir": str(self.config.query_dir),
                "eval_file": str(self.config.eval_file) if self.config.eval_file else None,
                "tool_schema_version": self.config.tool_schema_version,
                "cost_aware_skills": self.config.cost_aware_skills,
                "model": self.config.model,
                "embedding_model": self.config.embedding_model,
            },
            "stages": {},
        }

        # --- Stage 1: Dataset ---
        dataset_output = self._dataset_dir / "skill_context_ready.jsonl"
        if self._stage_should_run(state, "dataset"):
            state.mark_running("dataset")
            try:
                from .dataset import build_skill_dataset

                stats = build_skill_dataset(
                    rollout_dir=self.config.rollout_dir,
                    query_dir=self.config.query_dir,
                    eval_file=self.config.eval_file,
                    output_path=dataset_output,
                    expected_version=self.config.tool_schema_version,
                    reward_mode=self.config.reward_mode,
                    include_successful_trajectory=self.config.include_successful_trajectory,
                    tool_attributes_dir=(
                        self.config.tool_attributes_dir
                        if self.config.cost_aware_skills
                        else None
                    ),
                )
                self._require_stage_artifacts("dataset")
                state.mark_completed("dataset", metadata={"records": stats["total_records"]})
                manifest["stages"]["dataset"] = {"output": str(dataset_output), **stats}
                logger.info(f"Stage 1 (Dataset) complete: {stats['total_records']} records")
            except Exception as e:
                state.mark_failed("dataset", str(e))
                raise
        else:
            logger.info("Stage 1 (Dataset) already completed, skipping.")
            manifest["stages"]["dataset"] = {"output": str(dataset_output), "skipped": True}

        # --- Stage 2: Planning & Atomic Extraction ---
        if self._stage_should_run(state, "planning_atomic_extraction"):
            state.mark_running("planning_atomic_extraction")
            try:
                from .extraction import run_extraction_stage

                extraction_stats = run_extraction_stage(
                    input_path=dataset_output,
                    output_dir=self._extraction_dir,
                    config=self.config,
                )
                self._require_stage_artifacts("planning_atomic_extraction")
                state.mark_completed("planning_atomic_extraction", metadata=extraction_stats)
                manifest["stages"]["planning_atomic_extraction"] = extraction_stats
                logger.info(
                    f"Stage 2 (Extraction) complete: "
                    f"{extraction_stats.get('plans', 0)} plans, "
                    f"{extraction_stats.get('atomic_skills_prepared', 0)} atomic skills"
                )
            except Exception as e:
                state.mark_failed("planning_atomic_extraction", str(e))
                raise
        else:
            logger.info("Stage 2 (Extraction) already completed, skipping.")
            manifest["stages"]["planning_atomic_extraction"] = {"skipped": True}

        # --- Stage 3: Tool Skill Extraction ---
        if self._stage_should_run(state, "tool_skill_extraction"):
            state.mark_running("tool_skill_extraction")
            try:
                from .tool_skills import run_tool_skill_stage

                tool_skill_stats = run_tool_skill_stage(
                    input_path=dataset_output,
                    output_dir=self._tool_skill_dir,
                    config=self.config,
                )
                self._require_stage_artifacts("tool_skill_extraction")
                state.mark_completed("tool_skill_extraction", metadata=tool_skill_stats)
                manifest["stages"]["tool_skill_extraction"] = tool_skill_stats
                logger.info(
                    f"Stage 3 (Tool Skills) complete: "
                    f"{tool_skill_stats.get('tool_skills_prepared', 0)} tool skills"
                )
            except Exception as e:
                state.mark_failed("tool_skill_extraction", str(e))
                raise
        else:
            logger.info("Stage 3 (Tool Skills) already completed, skipping.")
            manifest["stages"]["tool_skill_extraction"] = {"skipped": True}

        # --- Assemble final library ---
        self._assemble_library()

        # --- Stage 4: Graph Construction ---
        if self._stage_should_run(state, "graph_construction"):
            state.mark_running("graph_construction")
            try:
                from .graph import build_skill_graph

                graph_stats = build_skill_graph(
                    skill_library_dir=self._library_dir,
                    output_dir=self._library_dir,
                    embedding_api_base=self.config.embedding_api_base,
                    embedding_api_key=self.config.embedding_api_key,
                    embedding_model=self.config.embedding_model,
                    plan_atomic_coverage_threshold=self.config.plan_atomic_coverage_threshold,
                    embedding_batch_size=self.config.merge_embedding_batch_size,
                )
                self._require_stage_artifacts("graph_construction")
                state.mark_completed("graph_construction", metadata=graph_stats)
                manifest["stages"]["graph_construction"] = graph_stats
                logger.info(
                    f"Stage 4 (Graph) complete: "
                    f"{graph_stats.get('total_nodes', 0)} nodes, "
                    f"{graph_stats.get('total_edges', 0)} edges"
                )
            except Exception as e:
                state.mark_failed("graph_construction", str(e))
                raise
        else:
            logger.info("Stage 4 (Graph) already completed, skipping.")
            manifest["stages"]["graph_construction"] = {"skipped": True}

        # --- Finalize ---
        state.mark_all_completed()
        manifest["status"] = "completed"
        manifest["completed_at"] = _now_iso()
        manifest["outputs"] = {
            "library_dir": str(self._library_dir),
            "dataset_dir": str(self._dataset_dir),
            "extraction_dir": str(self._extraction_dir),
            "tool_skill_dir": str(self._tool_skill_dir),
        }

        # Aggregate token usage from all stages
        total_token_usage = {
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_tokens": 0,
            "call_count": 0,
            "stages": {},
        }
        for stage_name in ("planning_atomic_extraction", "tool_skill_extraction"):
            stage_info = manifest["stages"].get(stage_name, {})
            if "token_usage" in stage_info:
                usage = stage_info["token_usage"]
                total_token_usage["total_prompt_tokens"] += usage.get("total_prompt_tokens", 0)
                total_token_usage["total_completion_tokens"] += usage.get("total_completion_tokens", 0)
                total_token_usage["total_tokens"] += usage.get("total_tokens", 0)
                total_token_usage["call_count"] += usage.get("call_count", 0)
                total_token_usage["stages"][stage_name] = usage

        manifest["token_usage"] = total_token_usage

        # Save standalone token usage summary
        token_summary_path = self._output_dir / "token_usage.json"
        with token_summary_path.open("w", encoding="utf-8") as f:
            json.dump(total_token_usage, f, ensure_ascii=False, indent=2)

        # Write manifest
        manifest_path = self._output_dir / "bootstrap_manifest.json"
        with manifest_path.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

        logger.info(f"Bootstrap pipeline complete. Library at: {self._library_dir}")
        logger.info(f"Bootstrap total token usage: {total_token_usage['total_tokens']} tokens across {total_token_usage['call_count']} calls")
        return manifest

    def _assemble_library(self) -> None:
        """Copy final artifacts from extraction and tool_skill stages into the library directory."""
        # From extraction stage (Stage 2)
        for filename in (
            "plan_library.json",
            "atomic_skills.prepared.jsonl",
            "atomic_skills.merged.jsonl",
            "atomic_skills.raw.jsonl",
            "atomic_skills.filtered.jsonl",
            "functional_skills.prepared.jsonl",
        ):
            source = self._extraction_dir / filename
            if source.exists():
                target = self._library_dir / filename
                shutil.copy2(source, target)

        # From tool skill stage (Stage 3)
        for filename in (
            "tool_skills.prepared.jsonl",
            "tool_skills.merged.jsonl",
        ):
            source = self._tool_skill_dir / filename
            if source.exists():
                target = self._library_dir / filename
                shutil.copy2(source, target)

        # Write library metadata
        metadata = {
            "created_at": _now_iso(),
            "tool_schema_version": self.config.tool_schema_version,
            "model": self.config.model,
            "embedding_model": self.config.embedding_model,
            "source": "bootstrap",
        }
        metadata_path = self._library_dir / "library_metadata.json"
        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
