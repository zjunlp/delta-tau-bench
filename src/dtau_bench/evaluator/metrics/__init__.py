"""
dtau_bench Evaluator Metrics Module

- retrieval: tool retrieval Precision / Recall / F1
- sr: single-version Success Rate, aggregated by domain
- cross_version: IDR / DAR between adjacent versions
- spe: Success-weighted Path Efficiency, single-version
"""

from dtau_bench.evaluator.metrics.retrieval import (
    RetrievalSampleMetrics,
    RetrievalSummary,
    compute_retrieval_metrics,
    summarize_retrieval_metrics,
)
from dtau_bench.evaluator.metrics.sr import (
    SRReport,
    compute_domain_sr,
    compute_sr_report,
)
from dtau_bench.evaluator.metrics.cross_version import (
    CrossVersionMetrics,
    TaskWeightInfo,
    build_task_weight_info_from_results,
    compute_all_cross_version_metrics,
    compute_cross_version_metrics,
)
from dtau_bench.evaluator.metrics.spe import (
    SPEMetrics,
    compute_all_spe_metrics,
    compute_spe_metrics,
)

__all__ = [
    # Retrieval
    "RetrievalSampleMetrics",
    "RetrievalSummary",
    "compute_retrieval_metrics",
    "summarize_retrieval_metrics",
    # SR
    "SRReport",
    "compute_sr_report",
    "compute_domain_sr",
    # Cross-version
    "CrossVersionMetrics",
    "TaskWeightInfo",
    "build_task_weight_info_from_results",
    "compute_cross_version_metrics",
    "compute_all_cross_version_metrics",
    # SPE
    "SPEMetrics",
    "compute_spe_metrics",
    "compute_all_spe_metrics",
]
