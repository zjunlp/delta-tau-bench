"""Offline regression tests for evaluation report rendering."""

from __future__ import annotations

import unittest

from dtau_bench.evaluator.report import (
    MultiVersionReport,
    generate_multi_version_markdown,
)


class MultiVersionMarkdownTests(unittest.TestCase):
    def test_render_with_section_separating_blank_lines_includes_major_sections(self) -> None:
        report = MultiVersionReport(
            metadata={
                "timestamp": "2026-01-02T03:04:05+00:00",
                "versions": ["base", "v1"],
                "weight_modes": ["unit"],
            },
            version_summaries={
                "base": {"total": 2, "passed": 1, "failed": 1, "sr": 0.5},
                "v1": {"total": 2, "passed": 2, "failed": 0, "sr": 1.0},
            },
            tool_retrieval_by_version={
                "base": {
                    "samples_with_tool_retrieval": 2,
                    "precision": 0.5,
                    "recall": 1.0,
                    "f1": 2 / 3,
                    "avg_gold_tool_count": 1.0,
                    "gold_source_counts": {"version_trajectory": 2},
                }
            },
            spe_metrics_by_weight={
                "unit": {
                    "base": {
                        "total_tasks": 2,
                        "passed_tasks": 1,
                        "sr": 0.5,
                        "spe": 0.5,
                        "computable_tasks": 2,
                        "mean_gold_weight": 1.0,
                        "mean_agent_weight": 1.0,
                    }
                }
            },
            cross_version_metrics_by_weight={
                "unit": {
                    "base->v1": {
                        "total_tasks": 2,
                        "improvement_tasks": 1,
                        "degradation_tasks": 0,
                        "unchanged_tasks": 1,
                        "idr": 1.0,
                        "dar": None,
                    }
                }
            },
        )

        markdown = generate_multi_version_markdown(report)

        self.assertIn("\n\n", markdown)
        for heading in (
            "# Multi-Version Evaluation Report",
            "## SR",
            "## Tool Retrieval",
            "## SPE",
            "## IDR / DAR",
        ):
            with self.subTest(heading=heading):
                self.assertIn(heading, markdown)


if __name__ == "__main__":
    unittest.main()
