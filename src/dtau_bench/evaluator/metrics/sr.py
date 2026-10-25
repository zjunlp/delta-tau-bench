"""
Single-version Success Rate (SR).

    SR^(v) = passed_tasks / N_v

Aggregates per-sample pass/fail results into an overall rate plus a per-domain
breakdown. ``N_v`` is the full evaluated task count, the same denominator as SPE.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from dtau_bench.evaluator.models import DomainStats, SampleEvalResult, VersionReport
from dtau_bench.evaluator.metrics.retrieval import RetrievalSummary


@dataclass
class SRReport:
    """
    SR report for a single version.

    Attributes:
        version: Tool version evaluated.
        total: N_v — total number of evaluated samples.
        passed: Number of samples that passed.
        failed: Number of samples that failed.
        errored: Number of samples with evaluation errors (counted as failures).
        sr: passed / total.
        by_domain: Per-domain breakdown.
        retrieval_summary: Tool retrieval metrics (if applicable).
    """

    version: str
    total: int = 0
    passed: int = 0
    failed: int = 0
    errored: int = 0
    sr: float = 0.0
    by_domain: Dict[str, DomainStats] = field(default_factory=dict)
    retrieval_summary: Optional[RetrievalSummary] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result: Dict[str, Any] = {
            "version": self.version,
            "total": self.total,
            "passed": self.passed,
            "failed": self.failed,
            "errored": self.errored,
            "sr": self.sr,
            "by_domain": {k: v.to_dict() for k, v in self.by_domain.items()},
        }
        if self.retrieval_summary is not None:
            result["tool_retrieval"] = self.retrieval_summary.to_dict()
        return result

    def to_version_report(self) -> VersionReport:
        """Convert to the VersionReport model."""
        return VersionReport(
            version=self.version,
            total=self.total,
            passed=self.passed,
            failed=self.failed,
            by_domain=dict(self.by_domain),
            retrieval_summary=(
                self.retrieval_summary.to_dict() if self.retrieval_summary else {}
            ),
        )


def compute_sr_report(
    results: List[SampleEvalResult],
    version: str,
    *,
    retrieval_summary: Optional[RetrievalSummary] = None,
) -> SRReport:
    """
    Compute the SR report for a single version.

    Args:
        results: List of per-sample evaluation results.
        version: Version identifier being evaluated.
        retrieval_summary: Optional pre-computed retrieval summary.

    Returns:
        SRReport with overall and per-domain statistics.
    """
    total = len(results)
    passed = 0
    failed = 0
    errored = 0
    domain_stats: Dict[str, DomainStats] = {}

    for result in results:
        domain_key = result.domain_key or (
            "_".join(result.domains) if result.domains else "unknown"
        )

        if domain_key not in domain_stats:
            domain_stats[domain_key] = DomainStats(domain=domain_key)

        ds = domain_stats[domain_key]
        ds.total += 1

        if result.error:
            errored += 1
            ds.failed += 1  # errors count as failures
        elif result.passed:
            passed += 1
            ds.passed += 1
        else:
            failed += 1
            ds.failed += 1

    return SRReport(
        version=version,
        total=total,
        passed=passed,
        failed=failed,
        errored=errored,
        sr=passed / total if total > 0 else 0.0,
        by_domain=domain_stats,
        retrieval_summary=retrieval_summary,
    )


def compute_domain_sr(
    results: List[SampleEvalResult],
) -> Dict[str, DomainStats]:
    """
    Compute per-domain SR when only the domain breakdown is needed.

    Args:
        results: List of per-sample evaluation results.

    Returns:
        domain_key → DomainStats.
    """
    domain_stats: Dict[str, DomainStats] = {}

    for result in results:
        domain_key = result.domain_key or (
            "_".join(result.domains) if result.domains else "unknown"
        )
        if domain_key not in domain_stats:
            domain_stats[domain_key] = DomainStats(domain=domain_key)

        ds = domain_stats[domain_key]
        ds.total += 1
        if result.passed and not result.error:
            ds.passed += 1
        else:
            ds.failed += 1

    return domain_stats
