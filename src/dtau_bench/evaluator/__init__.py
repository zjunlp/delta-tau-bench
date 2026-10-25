"""
dtau_bench Evaluator Module

Provides environment-based evaluation of agent rollouts:
- Data loading from rollout JSONL files
- Agent action extraction and trajectory weight computation
- Reference-path weight calculation for cross-version comparison
- SR, SPE, IDR, DAR and tool retrieval P/R/F1
- Parallel evaluation engine (gold replay + agent replay + DB diff)
- Report generation (JSON + Markdown, single + multi version)
"""

from dtau_bench.evaluator.models import (
    EvalSample,
    SampleEvalResult,
    VersionReport,
    ActionRecord,
    TrajectoryWeightSummary,
)
from dtau_bench.evaluator.data_loader import load_rollout_samples
from dtau_bench.evaluator.actions import (
    extract_action_records,
    compute_trajectory_weights,
    get_rollout_messages,
)
from dtau_bench.evaluator.gold_weight import GoldWeightCalculator
from dtau_bench.evaluator.engine import (
    EnvironmentEvaluationEngine,
    EvalTaskSpec,
    build_eval_task_spec,
)
from dtau_bench.evaluator.report import (
    SingleVersionReport,
    MultiVersionReport,
    build_single_version_report,
    build_multi_version_report,
    generate_single_version_markdown,
    generate_multi_version_markdown,
)

__all__ = [
    "EvalSample",
    "SampleEvalResult",
    "VersionReport",
    "ActionRecord",
    "TrajectoryWeightSummary",
    "load_rollout_samples",
    "extract_action_records",
    "compute_trajectory_weights",
    "get_rollout_messages",
    "GoldWeightCalculator",
    "EnvironmentEvaluationEngine",
    "EvalTaskSpec",
    "build_eval_task_spec",
    "SingleVersionReport",
    "MultiVersionReport",
    "build_single_version_report",
    "build_multi_version_report",
    "generate_single_version_markdown",
    "generate_multi_version_markdown",
]
