"""
Evaluation Report Generation.

Produces evaluation reports in JSON and Markdown formats:
- SingleVersionReport: SR and tool retrieval P/R/F1 for one version
- MultiVersionReport: SR, SPE, IDR/DAR and retrieval P/R/F1 across versions
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dtau_bench.evaluator.models import SampleEvalResult
from dtau_bench.evaluator.metrics.sr import compute_sr_report
from dtau_bench.evaluator.metrics.retrieval import RetrievalSummary

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Formatting Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _fmt(value: Optional[float]) -> str:
    """Format a float to 4 decimal places, or 'N/A' if None."""
    if value is None:
        return "N/A"
    return f"{value:.4f}"


def _version_sort_key(version: str) -> Tuple[int, int, str]:
    """Sort versions naturally: base < v1 < v2 < ... < vN < anything else."""
    if version == "base":
        return (0, 0, "")
    if version.startswith("v") and version[1:].isdigit():
        return (1, int(version[1:]), "")
    return (2, 9999, version)


def _format_gold_source(retrieval: Dict[str, Any]) -> str:
    """Render the gold-source histogram of a retrieval summary."""
    counts = retrieval.get("gold_source_counts") or {}
    if not counts:
        return "unknown"
    return ", ".join(f"{source}: {count}" for source, count in sorted(counts.items()))


# ─────────────────────────────────────────────────────────────────────────────
# Single Version Report
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class SingleVersionReport:
    """
    Evaluation report for a single version.

    Attributes:
        metadata: Run metadata (timestamp, version, run_name, totals).
        summary: SR aggregation plus the tool retrieval summary.
        samples: Per-sample detailed results.
    """

    metadata: Dict[str, Any] = field(default_factory=dict)
    summary: Dict[str, Any] = field(default_factory=dict)
    samples: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON output."""
        return {
            "metadata": self.metadata,
            "summary": self.summary,
            "samples": self.samples,
        }

    def save_json(self, output_path: Path) -> None:
        """Save report to JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        logger.info("Saved single-version report to %s", output_path)

    def save_markdown(self, output_path: Path) -> None:
        """Save Markdown report."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        md = generate_single_version_markdown(self)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(md)
        logger.info("Saved single-version markdown to %s", output_path)


def build_single_version_report(
    results: List[SampleEvalResult],
    *,
    version: str,
    run_name: str = "",
    workers: int = 1,
    retrieval_summary: Optional[RetrievalSummary] = None,
    include_sample_details: bool = True,
) -> SingleVersionReport:
    """
    Build a single-version evaluation report.

    Args:
        results: Per-sample evaluation results from the engine.
        version: Version being evaluated.
        run_name: Rollout run name.
        workers: Number of workers used.
        retrieval_summary: Pre-computed retrieval summary.
        include_sample_details: Whether to include per-sample details.

    Returns:
        SingleVersionReport ready for serialization.
    """
    sr_report = compute_sr_report(
        results, version, retrieval_summary=retrieval_summary
    )

    metadata = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": version,
        "run_name": run_name,
        "total_samples": sr_report.total,
        "workers": workers,
    }

    summary: Dict[str, Any] = {
        "total": sr_report.total,
        "passed": sr_report.passed,
        "failed": sr_report.failed,
        "errored": sr_report.errored,
        "sr": round(sr_report.sr, 4),
        "by_domain": {k: v.to_dict() for k, v in sr_report.by_domain.items()},
    }

    if retrieval_summary is not None:
        summary["tool_retrieval"] = retrieval_summary.to_dict()

    samples: List[Dict[str, Any]] = []
    if include_sample_details:
        samples = [result.to_dict() for result in results]

    return SingleVersionReport(
        metadata=metadata,
        summary=summary,
        samples=samples,
    )


def generate_single_version_markdown(report: SingleVersionReport) -> str:
    """
    Generate the Markdown report for a single version.

    Args:
        report: SingleVersionReport to format.

    Returns:
        Markdown string.
    """
    lines: List[str] = []
    meta = report.metadata
    summary = report.summary

    lines.append(f"# Evaluation Report — {meta.get('version', '?')}")
    lines.append("")
    lines.append(f"**Run**: {meta.get('run_name', 'N/A')}")
    lines.append(f"**Timestamp**: {meta.get('timestamp', 'N/A')}")
    lines.append(f"**Workers**: {meta.get('workers', 'N/A')}")
    lines.append("")

    lines.append("## Summary")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Total Samples | {summary.get('total', 0)} |")
    lines.append(f"| Passed | {summary.get('passed', 0)} |")
    lines.append(f"| Failed | {summary.get('failed', 0)} |")
    lines.append(f"| Errored | {summary.get('errored', 0)} |")
    lines.append(f"| SR | {summary.get('sr', 0):.2%} |")
    lines.append("")

    by_domain = summary.get("by_domain", {})
    if by_domain:
        lines.append("## Domain Breakdown")
        lines.append("")
        lines.append("| Domain | Total | Passed | Failed | SR |")
        lines.append("|--------|-------|--------|--------|----|")
        for domain_key, stats in sorted(by_domain.items()):
            if isinstance(stats, dict):
                total = stats.get("total", 0)
                passed = stats.get("passed", 0)
                failed = stats.get("failed", 0)
                rate = stats.get("sr", 0)
            else:
                total = passed = failed = 0
                rate = 0
            lines.append(
                f"| {domain_key} | {total} | {passed} | {failed} | {rate:.2%} |"
            )
        lines.append("")

    retrieval = summary.get("tool_retrieval")
    if retrieval:
        lines.append("## Tool Retrieval")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(
            f"| Samples with Retrieval "
            f"| {retrieval.get('samples_with_tool_retrieval', 0)} |"
        )
        lines.append(f"| Precision | {retrieval.get('precision', 0):.4f} |")
        lines.append(f"| Recall | {retrieval.get('recall', 0):.4f} |")
        lines.append(f"| F1 | {retrieval.get('f1', 0):.4f} |")
        lines.append(f"| Avg Gold Tools | {retrieval.get('avg_gold_tool_count', 0):.2f} |")
        lines.append(
            f"| Avg Predicted Tools | {retrieval.get('avg_predicted_tool_count', 0):.2f} |"
        )
        lines.append(f"| Gold Source | {_format_gold_source(retrieval)} |")
        unavailable = retrieval.get("samples_gold_unavailable", 0)
        if unavailable:
            lines.append(f"| Excluded (version gold unavailable) | {unavailable} |")
        lines.append("")
        lines.append(
            "> Gold = version-translated trajectory for the evaluated version "
            "(`version_trajectories[version]`), not the base trajectory."
        )
        lines.append("")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Multi-Version Report
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class MultiVersionReport:
    """
    Aggregated evaluation report across multiple versions.

    Attributes:
        metadata: Versions and weight modes.
        version_summaries: Per-version SR summaries.
        tool_retrieval_by_version: Per-version retrieval P/R/F1.
        cross_version_metrics_by_weight: weight_mode → version_pair → IDR/DAR.
        spe_metrics_by_weight: weight_mode → version → SPE/SR. SPE is a
            single-version metric, so it is also defined for the first version
            where IDR/DAR are not.
    """

    metadata: Dict[str, Any] = field(default_factory=dict)
    version_summaries: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    tool_retrieval_by_version: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    cross_version_metrics_by_weight: Dict[str, Dict[str, Any]] = field(
        default_factory=dict
    )
    spe_metrics_by_weight: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON output."""
        return {
            "metadata": self.metadata,
            "version_summaries": self.version_summaries,
            "tool_retrieval_by_version": self.tool_retrieval_by_version,
            "cross_version_metrics_by_weight": self.cross_version_metrics_by_weight,
            "spe_metrics_by_weight": self.spe_metrics_by_weight,
        }

    def save_json(self, output_path: Path) -> None:
        """Save report to JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        logger.info("Saved multi-version report to %s", output_path)

    def save_markdown(self, output_path: Path) -> None:
        """Save Markdown report."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        md = generate_multi_version_markdown(self)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(md)
        logger.info("Saved multi-version markdown to %s", output_path)


def build_multi_version_report(
    eval_results_by_version: Dict[str, List[SampleEvalResult]],
    *,
    versions: List[str],
    weight_modes: Optional[List[str]] = None,
    retrieval_by_version: Optional[Dict[str, RetrievalSummary]] = None,
    cross_version_metrics_by_weight: Optional[Dict[str, Dict[str, Any]]] = None,
    spe_metrics_by_weight: Optional[Dict[str, Dict[str, Any]]] = None,
) -> MultiVersionReport:
    """
    Build a multi-version evaluation report.

    Args:
        eval_results_by_version: version → list of SampleEvalResult.
        versions: Ordered list of evaluated versions. Adjacent pairs in this
            order define the cross-version comparisons.
        weight_modes: Weight modes used (default: ["unit", "cost"]).
        retrieval_by_version: Optional per-version retrieval summaries.
        cross_version_metrics_by_weight: weight_mode → version_pair → metrics.
        spe_metrics_by_weight: weight_mode → version → metrics.

    Returns:
        MultiVersionReport ready for serialization.
    """
    if weight_modes is None:
        weight_modes = ["unit", "cost"]

    metadata = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "versions": versions,
        "weight_modes": weight_modes,
    }

    version_summaries: Dict[str, Dict[str, Any]] = {}
    for v in versions:
        results = eval_results_by_version.get(v, [])
        if results:
            version_summaries[v] = compute_sr_report(results, v).to_dict()
        else:
            version_summaries[v] = {"version": v, "total": 0, "sr": 0.0}

    retrieval_dict: Dict[str, Dict[str, Any]] = {}
    if retrieval_by_version:
        for v, summary in retrieval_by_version.items():
            retrieval_dict[v] = summary.to_dict()

    return MultiVersionReport(
        metadata=metadata,
        version_summaries=version_summaries,
        tool_retrieval_by_version=retrieval_dict,
        cross_version_metrics_by_weight=cross_version_metrics_by_weight or {},
        spe_metrics_by_weight=spe_metrics_by_weight or {},
    )


def generate_multi_version_markdown(report: MultiVersionReport) -> str:
    """
    Generate the Markdown report for a multi-version evaluation.

    Args:
        report: MultiVersionReport to format.

    Returns:
        Markdown string.
    """
    lines: List[str] = []
    meta = report.metadata
    versions = meta.get("versions", [])

    lines.append("# Multi-Version Evaluation Report")
    lines.append("")
    lines.append(f"**Timestamp**: {meta.get('timestamp', 'N/A')}")
    lines.append(f"**Versions**: {', '.join(versions)}")
    lines.append("")
    comparison_order = versions
    if comparison_order:
        lines.append(f"**Comparison Order**: {' → '.join(comparison_order)}")
    lines.append(f"**Weight Modes**: {', '.join(meta.get('weight_modes', []))}")
    lines.append("")

    # ── SR ──
    lines.append("## SR")
    lines.append("")
    lines.append("| Version | Total | Passed | Failed | SR |")
    lines.append("|---------|-------|--------|--------|----|")
    for v in versions:
        vs = report.version_summaries.get(v, {})
        lines.append(
            f"| {v} | {vs.get('total', 0)} | {vs.get('passed', 0)} "
            f"| {vs.get('failed', 0)} | {vs.get('sr', 0):.2%} |"
        )
    lines.append("")

    # ── Tool retrieval ──
    if report.tool_retrieval_by_version:
        lines.append("## Tool Retrieval")
        lines.append("")
        lines.append(
            "| Version | Samples | Precision | Recall | F1 | Avg Gold | Gold Source |"
        )
        lines.append(
            "|---------|---------|-----------|--------|----|----------|-------------|"
        )
        for v in versions:
            r = report.tool_retrieval_by_version.get(v)
            if not r:
                continue
            lines.append(
                f"| {v} "
                f"| {r.get('samples_with_tool_retrieval', 0)} "
                f"| {r.get('precision', 0):.4f} "
                f"| {r.get('recall', 0):.4f} "
                f"| {r.get('f1', 0):.4f} "
                f"| {r.get('avg_gold_tool_count', 0):.2f} "
                f"| {_format_gold_source(r)} |"
            )
        lines.append("")
        lines.append(
            "> Gold = `version_trajectories[version]`. `Avg Gold` is expected to "
            "differ across versions (tools get split/merged); a constant value "
            "means the base trajectory is being used by mistake."
        )
        lines.append("")

    # ── SPE ──
    if report.spe_metrics_by_weight:
        lines.append("## SPE")
        lines.append("")
        lines.append(
            "> `SPE = (1/N) Σ Sᵢ·min(Wᵢ/Ŵᵢ, 1)`, where `Sᵢ` is pass/fail, `Wᵢ` "
            "the reference-path weight and `Ŵᵢ` the agent trajectory weight. `N` "
            "is the full evaluated task count, so `SPE ≤ SR` always holds. "
            "`weight=unit` gives Step-SPE, `weight=cost` gives Cost-SPE."
        )
        lines.append("")
        for weight_mode, version_dict in sorted(report.spe_metrics_by_weight.items()):
            lines.append(f"### Weight: {weight_mode}")
            lines.append("")
            lines.append(
                "| Version | N | Passed | SR | SPE | Computable | Mean W | Mean Ŵ |"
            )
            lines.append(
                "|---------|---|--------|----|-----|------------|--------|--------|"
            )
            for version, metrics in sorted(
                version_dict.items(), key=lambda kv: _version_sort_key(kv[0])
            ):
                if not isinstance(metrics, dict):
                    continue
                gold_w = metrics.get("mean_gold_weight")
                agent_w = metrics.get("mean_agent_weight")
                lines.append(
                    f"| {version} "
                    f"| {metrics.get('total_tasks', 0)} "
                    f"| {metrics.get('passed_tasks', 0)} "
                    f"| {_fmt(metrics.get('sr'))} "
                    f"| {_fmt(metrics.get('spe'))} "
                    f"| {metrics.get('computable_tasks', 0)} "
                    f"| {f'{gold_w:.2f}' if gold_w is not None else 'N/A'} "
                    f"| {f'{agent_w:.2f}' if agent_w is not None else 'N/A'} |"
                )
            lines.append("")

            metric_dicts = [m for m in version_dict.values() if isinstance(m, dict)]

            skipped_passed = sum(m.get("skipped_passed", 0) for m in metric_dicts)
            if skipped_passed:
                lines.append(
                    f"> Note: {skipped_passed} passed task(s) had no usable "
                    "reference weight and can never contribute to SPE."
                )
                lines.append("")

            # Real per-tool costs must be configured on both sides, otherwise Ŵ
            # is unit-scaled while W is cost-scaled.
            if weight_mode == "cost":
                gold_means = [
                    m["mean_gold_weight"]
                    for m in metric_dicts
                    if m.get("mean_gold_weight")
                ]
                agent_means = [
                    m["mean_agent_weight"]
                    for m in metric_dicts
                    if m.get("mean_agent_weight")
                ]
                if gold_means and agent_means:
                    gold_avg = sum(gold_means) / len(gold_means)
                    agent_avg = sum(agent_means) / len(agent_means)
                    if gold_avg > 2.0 * agent_avg:
                        lines.append(
                            f"> **Warning**: mean reference weight ({gold_avg:.2f}) "
                            f"is far above mean agent weight ({agent_avg:.2f}). The "
                            "agent side is likely falling back to unit cost — check "
                            "that `data/meta/tool_costs.json` exists."
                        )
                        lines.append("")

    # ── IDR / DAR ──
    if report.cross_version_metrics_by_weight:
        lines.append("## IDR / DAR")
        lines.append("")
        lines.append(
            "> `IDR` averages `Sᵢ·min(Wᵢ/Ŵᵢ, 1)` over the tasks whose reference "
            "path became cheaper at this version, `DAR` over those whose "
            "reference path became costlier. `weight=unit` gives Step-IDR/DAR, "
            "`weight=cost` gives Cost-IDR/DAR."
        )
        lines.append("")
        for weight_mode, pair_dict in sorted(
            report.cross_version_metrics_by_weight.items()
        ):
            lines.append(f"### Weight: {weight_mode}")
            lines.append("")
            lines.append(
                "| Pair | Total | Cheaper | Costlier | Unchanged | IDR | DAR |"
            )
            lines.append(
                "|------|-------|---------|----------|-----------|-----|-----|"
            )
            for pair_name, metrics in sorted(pair_dict.items()):
                if not isinstance(metrics, dict):
                    continue
                lines.append(
                    f"| {pair_name} "
                    f"| {metrics.get('total_tasks', 0)} "
                    f"| {metrics.get('improvement_tasks', 0)} "
                    f"| {metrics.get('degradation_tasks', 0)} "
                    f"| {metrics.get('unchanged_tasks', 0)} "
                    f"| {_fmt(metrics.get('idr'))} "
                    f"| {_fmt(metrics.get('dar'))} |"
                )
            lines.append("")

    return "\n".join(lines)
