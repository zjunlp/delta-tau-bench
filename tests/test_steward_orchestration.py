"""Offline regression tests for Steward orchestration acceptance rules."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from dtau_bench.cli import chain, evaluate_multi
from dtau_bench.cli.build_tool_index import _validate_saved_index


ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _eval_row(sample_id: str, version: str) -> dict:
    return {
        "sample_id": sample_id,
        "version": version,
        "passed": True,
        "error": None,
        "gold_trajectory": [],
        "agent_traj": [],
        "domains": ["Domain"],
        "domain_key": "Domain",
        "details": {},
    }


def _rollout_row(sample_id: str, version: str) -> dict:
    return {
        "id": sample_id,
        "task": {
            "domain": "Domain",
            "gold_trajectory": ["example_tool"],
        },
        "run": {"version": version},
        "messages": [],
        "success": True,
    }


class ShellFailurePropagationTests(unittest.TestCase):
    def test_run_step_preserves_failure_and_does_not_print_done(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "step.log"
            result = subprocess.run(
                [
                    "bash",
                    "-c",
                    (
                        'source scripts/_common.sh; DRY_RUN=0; '
                        'run_step "$1" bash -c "exit 23"; echo FALSE_DONE'
                    ),
                    "steward-test",
                    str(log_path),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

        self.assertEqual(result.returncode, 23)
        self.assertNotIn("FALSE_DONE", result.stdout)

    def test_train_and_test_entrypoints_stop_before_done_banner(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp)
            scripts_dir = project / "scripts"
            scripts_dir.mkdir()
            for name in (
                "_common.sh",
                "_steward_env.sh",
                "run_steward_train.sh",
                "run_steward_test.sh",
            ):
                shutil.copy2(ROOT / "scripts" / name, scripts_dir / name)
            fake_python = project / "fake-python"
            fake_python.write_text("#!/bin/bash\nexit 29\n", encoding="utf-8")
            fake_python.chmod(0o755)
            env = {
                **os.environ,
                "PY": str(fake_python),
                "VERSIONS": "v1",
                "EXP_NAME": "failure_test",
                "RAG_INDEX_BASE": str(project / "tool-index"),
            }

            train = subprocess.run(
                ["bash", "scripts/run_steward_train.sh"],
                cwd=project,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
            test = subprocess.run(
                ["bash", "scripts/run_steward_test.sh"],
                cwd=project,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )

        self.assertEqual(train.returncode, 29, train.stdout + train.stderr)
        self.assertNotIn("Train phase done", train.stdout)
        self.assertEqual(test.returncode, 29, test.stdout + test.stderr)
        self.assertNotIn("Done --", test.stdout)


class ChainStateTests(unittest.TestCase):
    def setUp(self) -> None:
        chain._set_dry_run(False)

    def test_resume_rejects_configuration_mismatch(self) -> None:
        metadata = {
            "experiment_name": "exp",
            "train_versions": ["v1", "v2"],
            "test_versions": ["v1", "v2"],
            "agent_model": "model-a",
        }
        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "run_state.json"
            chain._init_run_state(state_path, resume=False, metadata=metadata)
            changed = dict(metadata, agent_model="model-b")

            with self.assertRaisesRegex(ValueError, "different configuration"):
                chain._init_run_state(state_path, resume=True, metadata=changed)

    def test_resume_rejects_tampered_fingerprint(self) -> None:
        metadata = {"experiment_name": "exp", "train_versions": ["v1"]}
        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "run_state.json"
            state = chain._init_run_state(
                state_path, resume=False, metadata=metadata
            )
            state["config_fingerprint"] = "tampered"
            _write_json(state_path, state)

            with self.assertRaisesRegex(ValueError, "fingerprint"):
                chain._init_run_state(state_path, resume=True, metadata=metadata)

    def test_completed_stage_with_missing_artifact_is_rerun(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "run_state.json"
            state = {"status": "completed", "stages": {"eval:v1": {"status": "completed"}}}

            def missing() -> None:
                raise chain.ArtifactValidationError("result file missing")

            should_run = chain._stage_needs_run(
                state_path, state, "eval:v1", missing
            )

            self.assertTrue(should_run)
            self.assertEqual(state["stages"]["eval:v1"]["status"], "invalid")
            persisted = json.loads(state_path.read_text(encoding="utf-8"))
            self.assertEqual(persisted["stages"]["eval:v1"]["status"], "invalid")

    def test_evaluation_artifact_requires_full_roster(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for query_id in ("q1", "q2"):
                _write_json(
                    root / "data" / "tasks" / "test" / "Domain" / f"{query_id}.json",
                    {"query_id": query_id, "domain": "Domain"},
                )
            eval_dir = root / "experiments" / "run" / "evaluation"
            _write_jsonl(eval_dir / "evaluation_results.jsonl", [_eval_row("q1", "v1")])
            _write_json(
                eval_dir / "evaluation_report.json",
                {
                    "metadata": {
                        "version": "v1",
                        "run_name": "run",
                        "total_samples": 1,
                    },
                    "summary": {"total": 1},
                },
            )

            with self.assertRaisesRegex(
                chain.ArtifactValidationError, "Incomplete test evaluation"
            ):
                chain._validate_evaluation_artifacts(
                    eval_dir,
                    data_dir=root / "data",
                    split="test",
                    version="v1",
                    run_name="run",
                )

    def test_evaluation_artifact_requires_boolean_passed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_json(
                root / "data" / "tasks" / "test" / "Domain" / "q1.json",
                {"query_id": "q1", "domain": "Domain"},
            )
            eval_dir = root / "experiments" / "run" / "evaluation"
            row = _eval_row("q1", "v1")
            row["passed"] = "false"
            _write_jsonl(eval_dir / "evaluation_results.jsonl", [row])

            with self.assertRaisesRegex(
                chain.ArtifactValidationError, "non-boolean passed"
            ):
                chain._validate_evaluation_artifacts(
                    eval_dir,
                    data_dir=root / "data",
                    split="test",
                    version="v1",
                    run_name="run",
                )

    def test_evaluation_artifact_rejects_wrong_domain(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_json(
                root / "data" / "tasks" / "test" / "Domain" / "q1.json",
                {"query_id": "q1", "domain": "Domain"},
            )
            eval_dir = root / "experiments" / "run" / "evaluation"
            row = _eval_row("q1", "v1")
            row["domains"] = ["OtherDomain"]
            row["domain_key"] = "OtherDomain"
            _write_jsonl(eval_dir / "evaluation_results.jsonl", [row])

            with self.assertRaisesRegex(
                chain.ArtifactValidationError, "expected 'Domain'"
            ):
                chain._validate_evaluation_artifacts(
                    eval_dir,
                    data_dir=root / "data",
                    split="test",
                    version="v1",
                    run_name="run",
                )

    def test_evaluation_artifact_rejects_stale_summary_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for query_id in ("q1", "q2"):
                _write_json(
                    root / "data" / "tasks" / "test" / "Domain" / f"{query_id}.json",
                    {"query_id": query_id, "domain": "Domain"},
                )
            eval_dir = root / "experiments" / "run" / "evaluation"
            passed = _eval_row("q1", "v1")
            errored = _eval_row("q2", "v1")
            errored["passed"] = False
            errored["error"] = "replay failed"
            _write_jsonl(
                eval_dir / "evaluation_results.jsonl", [passed, errored]
            )
            _write_json(
                eval_dir / "evaluation_report.json",
                {
                    "metadata": {
                        "version": "v1",
                        "run_name": "run",
                        "total_samples": 2,
                    },
                    "summary": {
                        "total": 2,
                        "passed": 2,
                        "failed": 0,
                        "errored": 0,
                    },
                },
            )

            with self.assertRaisesRegex(
                chain.ArtifactValidationError, "summary.passed"
            ):
                chain._validate_evaluation_artifacts(
                    eval_dir,
                    data_dir=root / "data",
                    split="test",
                    version="v1",
                    run_name="run",
                )

    def test_test_only_manifest_preserves_train_hops(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "experiment_manifest.json"
            original_hops = [{"version": "v1", "status": "completed"}]
            _write_json(
                path,
                {
                    "experiment_name": "exp",
                    "status": "train_completed",
                    "hops": original_hops,
                    "tests": [],
                    "library_map": {"v1": "/skills/v1"},
                },
            )

            manifest = chain._init_manifest(
                path,
                experiment_name="exp",
                bootstrap_strategy="bootstrap",
                train_versions=["v1"],
                test_versions=["v1"],
                resume=True,
            )

            self.assertEqual(manifest["hops"], original_hops)
            self.assertEqual(manifest["library_map"], {"v1": "/skills/v1"})


class ExperimentRuntimeSignatureTests(unittest.TestCase):
    @staticmethod
    def _signature(models_config: dict) -> dict:
        with patch.object(
            chain, "load_models_config", return_value=models_config
        ), patch.object(
            chain, "load_defaults_config", return_value={"max_steps": 20}
        ), patch.object(
            chain,
            "load_strategy_config",
            side_effect=lambda name: {"name": name},
        ):
            return chain._experiment_runtime_signature(
                agent_model="shared-model-key",
                user_model="shared-model-key",
                evolve_model="shared-model-key",
                bootstrap_strategy="bootstrap",
                train_strategy="train",
                test_strategy="test",
            )

    def test_reasoning_effort_change_under_same_model_key_changes_signature(self) -> None:
        def config(reasoning_effort: str) -> dict:
            return {
                "models": {
                    "shared-model-key": {
                        "model_name": "provider-model",
                        "base_url": "https://example.invalid/v1",
                        "max_tokens": 4096,
                        "reasoning_effort": reasoning_effort,
                    }
                },
                "embedding": {
                    "default": {
                        "backend": "local",
                        "model": "fixture-embedding",
                    }
                },
            }

        low = self._signature(config("low"))
        high = self._signature(config("high"))

        self.assertEqual(low["models"]["agent"]["key"], "shared-model-key")
        self.assertEqual(high["models"]["agent"]["key"], "shared-model-key")
        self.assertEqual(low["models"]["agent"]["reasoning_effort"], "low")
        self.assertEqual(high["models"]["agent"]["reasoning_effort"], "high")
        self.assertNotEqual(low, high)

    def test_signature_does_not_expose_api_keys_or_secret_values(self) -> None:
        model_secret = "model-secret-must-not-leak"
        request_secret = "request-secret-must-not-leak"
        embedding_secret = "embedding-secret-must-not-leak"
        signature = self._signature(
            {
                "models": {
                    "shared-model-key": {
                        "model_name": "provider-model",
                        "base_url": "https://example.invalid/v1",
                        "api_key": model_secret,
                        "request_options": {
                            "api_key": request_secret,
                            "extra_headers": {
                                "Authorization": f"Bearer {request_secret}"
                            },
                        },
                    }
                },
                "embedding": {
                    "default": {
                        "backend": "openai_http",
                        "model": "fixture-embedding",
                        "base_url": "https://embedding.invalid/v1",
                        "api_key": embedding_secret,
                    }
                },
            }
        )

        serialized = json.dumps(signature, ensure_ascii=False, sort_keys=True)
        self.assertNotIn('"api_key"', serialized)
        self.assertNotIn(model_secret, serialized)
        self.assertNotIn(request_secret, serialized)
        self.assertNotIn(embedding_secret, serialized)


class MultiVersionCompletenessTests(unittest.TestCase):
    @staticmethod
    def _write_roster(root: Path, query_ids=("q1",)) -> None:
        for query_id in query_ids:
            _write_json(
                root / "data" / "tasks" / "test" / "Domain" / f"{query_id}.json",
                {"query_id": query_id, "domain": "Domain"},
            )

    def test_missing_requested_version_fails_aggregation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_roster(root)
            _write_jsonl(
                root
                / "experiments"
                / "run_v1"
                / "evaluation"
                / "evaluation_results.jsonl",
                [_eval_row("q1", "v1")],
            )

            with patch.object(evaluate_multi.Path, "cwd", return_value=root), patch.object(
                evaluate_multi, "TaskMetadataProvider", return_value=[]
            ):
                with self.assertRaisesRegex(
                    evaluate_multi.IncompleteEvaluationError, "v2"
                ):
                    evaluate_multi.run_multi_version_evaluation(
                        experiment_name="exp",
                        data_dir=root / "data",
                        versions=["v1", "v2"],
                        rollout_pattern="run_{version}",
                        output_dir=root / "out",
                        report_format="json",
                    )

    def test_partial_result_file_is_not_backfilled_in_strict_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_roster(root, ("q1", "q2"))
            eval_dir = root / "evaluation"
            _write_jsonl(
                eval_dir / "evaluation_results.jsonl",
                [_eval_row("q1", "v1")],
            )

            with self.assertRaisesRegex(
                evaluate_multi.IncompleteEvaluationError, "missing"
            ):
                evaluate_multi.load_evaluation_results(
                    eval_dir,
                    data_dir=root / "data",
                    version="v1",
                    strict=True,
                )

    def test_strict_result_requires_boolean_passed_but_legacy_remains_compatible(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_roster(root)
            eval_dir = root / "evaluation"
            row = _eval_row("q1", "v1")
            row["passed"] = "false"
            _write_jsonl(eval_dir / "evaluation_results.jsonl", [row])

            legacy = evaluate_multi.load_evaluation_results(
                eval_dir,
                data_dir=root / "data",
                version="v1",
                strict=False,
            )
            self.assertIsNotNone(legacy)
            self.assertEqual(legacy[0].passed, "false")

            with self.assertRaisesRegex(
                evaluate_multi.IncompleteEvaluationError,
                "non-boolean passed",
            ):
                evaluate_multi.load_evaluation_results(
                    eval_dir,
                    data_dir=root / "data",
                    version="v1",
                    strict=True,
                )

    def test_wrong_version_rollout_is_not_silently_filtered(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_roster(root)
            _write_jsonl(
                root
                / "experiments"
                / "run_v1"
                / "evaluation"
                / "evaluation_results.jsonl",
                [_eval_row("q1", "v1")],
            )
            _write_jsonl(
                root / "experiments" / "run_v1" / "rollouts" / "Domain.jsonl",
                [_rollout_row("q1", "v2")],
            )

            with patch.object(evaluate_multi.Path, "cwd", return_value=root):
                with self.assertRaisesRegex(
                    evaluate_multi.IncompleteEvaluationError, "wrong_version"
                ):
                    evaluate_multi.run_multi_version_evaluation(
                        experiment_name="exp",
                        data_dir=root / "data",
                        versions=["v1"],
                        rollout_pattern="run_{version}",
                        output_dir=root / "out",
                        weight_modes=["unit"],
                        report_format="json",
                    )
            self.assertFalse((root / "out" / "multi_version_report.json").exists())

    def test_strict_result_rejects_wrong_domain(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_roster(root)
            eval_dir = root / "evaluation"
            row = _eval_row("q1", "v1")
            row["domains"] = ["OtherDomain"]
            row["domain_key"] = "OtherDomain"
            _write_jsonl(eval_dir / "evaluation_results.jsonl", [row])

            with self.assertRaisesRegex(
                evaluate_multi.IncompleteEvaluationError, "wrong_domain"
            ):
                evaluate_multi.load_evaluation_results(
                    eval_dir,
                    data_dir=root / "data",
                    version="v1",
                    strict=True,
                )

    def test_duplicate_and_missing_rollout_samples_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_roster(root, ("q1", "q2"))
            _write_jsonl(
                root
                / "experiments"
                / "run_v1"
                / "evaluation"
                / "evaluation_results.jsonl",
                [_eval_row("q1", "v1"), _eval_row("q2", "v1")],
            )
            _write_jsonl(
                root / "experiments" / "run_v1" / "rollouts" / "Domain.jsonl",
                [_rollout_row("q1", "v1"), _rollout_row("q1", "v1")],
            )

            with patch.object(evaluate_multi.Path, "cwd", return_value=root):
                with self.assertRaisesRegex(
                    evaluate_multi.IncompleteEvaluationError,
                    "duplicate=.*q1.*missing=.*q2",
                ):
                    evaluate_multi.run_multi_version_evaluation(
                        experiment_name="exp",
                        data_dir=root / "data",
                        versions=["v1"],
                        rollout_pattern="run_{version}",
                        output_dir=root / "out",
                        weight_modes=["unit"],
                        report_format="json",
                    )

    def test_complete_rollouts_generate_only_complete_paper_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_roster(root)
            for version in ("v1", "v2"):
                _write_jsonl(
                    root
                    / "experiments"
                    / f"run_{version}"
                    / "evaluation"
                    / "evaluation_results.jsonl",
                    [_eval_row("q1", version)],
                )
                _write_jsonl(
                    root
                    / "experiments"
                    / f"run_{version}"
                    / "rollouts"
                    / "Domain.jsonl",
                    [_rollout_row("q1", version)],
                )

            cross = {
                "unit": {
                    "v2_vs_v1": {
                        "total_tasks": 1,
                        "idr": None,
                        "dar": None,
                    }
                }
            }
            spe = {
                "unit": {
                    version: {
                        "total_tasks": 1,
                        "computable_tasks": 1,
                        "spe": 1.0,
                    }
                    for version in ("v1", "v2")
                }
            }
            with patch.object(evaluate_multi.Path, "cwd", return_value=root), patch.object(
                evaluate_multi,
                "compute_weight_based_metrics",
                return_value=(cross, spe),
            ):
                report = evaluate_multi.run_multi_version_evaluation(
                    experiment_name="exp",
                    data_dir=root / "data",
                    versions=["v1", "v2"],
                    rollout_pattern="run_{version}",
                    output_dir=root / "out",
                    weight_modes=["unit"],
                    report_format="json",
                )

            self.assertEqual(report.spe_metrics_by_weight, spe)
            self.assertEqual(report.cross_version_metrics_by_weight, cross)
            self.assertTrue((root / "out" / "multi_version_report.json").is_file())

    def test_missing_cross_version_metrics_prevents_report_write(self) -> None:
        with self.assertRaisesRegex(
            evaluate_multi.IncompleteEvaluationError, "missing IDR/DAR"
        ):
            evaluate_multi._validate_weight_metric_coverage(
                {"unit": {}},
                {
                    "unit": {
                        version: {
                            "total_tasks": 1,
                            "computable_tasks": 1,
                            "spe": 1.0,
                        }
                        for version in ("v1", "v2")
                    }
                },
                versions=["v1", "v2"],
                weight_modes=["unit"],
                roster_size=1,
            )


class IndexArtifactTests(unittest.TestCase):
    def test_tool_index_rejects_missing_companion_files(self) -> None:
        class UnusedIndexer:
            def load_index(self, path):  # pragma: no cover - must not be reached
                raise AssertionError("load_index should not run")

        with tempfile.TemporaryDirectory() as tmp:
            index_dir = Path(tmp)
            (index_dir / "embeddings.npy").write_bytes(b"partial")
            with self.assertRaisesRegex(ValueError, "missing or empty"):
                _validate_saved_index(UnusedIndexer(), index_dir, "v1")

    def test_skill_subindex_rejects_missing_companion_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            index_dir = Path(tmp) / "planning"
            index_dir.mkdir()
            (index_dir / "metadata.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(
                chain.ArtifactValidationError, "missing required artifacts"
            ):
                chain._validate_embedding_index_dir(index_dir, kind="planning")


if __name__ == "__main__":
    unittest.main()
