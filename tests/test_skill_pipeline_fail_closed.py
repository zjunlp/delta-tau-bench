import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from dtau_bench.cli.bootstrap import parse_args as parse_bootstrap_args
from dtau_bench.cli.evolve import parse_args as parse_evolve_args
from dtau_bench.skill.bootstrap.pipeline import (
    BootstrapConfig,
    BootstrapPipeline,
    _RunState,
)
from dtau_bench.skill.evolve.pipeline import EvolveConfig, EvolvePipeline, STEPS


class SkillPipelineCliTests(unittest.TestCase):
    def test_bootstrap_requires_evaluator_results(self):
        argv = [
            "dtau-bootstrap",
            "--run-name",
            "test",
            "--rollout-dir",
            "rollouts",
            "--tool-schema-version",
            "v1",
            "--model",
            "model",
        ]
        with patch.object(sys, "argv", argv), redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                parse_bootstrap_args()
        self.assertEqual(2, raised.exception.code)

    def test_bootstrap_help_does_not_claim_reward_fallback(self):
        output = io.StringIO()
        with patch.object(sys, "argv", ["dtau-bootstrap", "--help"]), redirect_stdout(output):
            with self.assertRaises(SystemExit) as raised:
                parse_bootstrap_args()
        self.assertEqual(0, raised.exception.code)
        self.assertNotIn("falls back", output.getvalue().lower())
        self.assertIn("authoritative pass/fail", output.getvalue().lower())

    def test_evolve_rejects_rollout_without_evaluator_results_during_parse(self):
        argv = [
            "dtau-evolve",
            "--run-name",
            "test",
            "--rollout-dir",
            "rollouts",
            "--tool-schema-version",
            "v2",
            "--model",
            "model",
        ]
        error = io.StringIO()
        with patch.object(sys, "argv", argv), redirect_stderr(error):
            with self.assertRaises(SystemExit) as raised:
                parse_evolve_args()
        self.assertEqual(2, raised.exception.code)
        self.assertIn("--eval-results-file is required", error.getvalue())


class SkillPipelineResumeValidationTests(unittest.TestCase):
    @staticmethod
    def _bootstrap_config(root: Path) -> BootstrapConfig:
        return BootstrapConfig(
            output_dir=root,
            rollout_dir=root / "rollouts",
            query_dir=root / "queries",
            tool_schemas_dir=root / "tools",
            tool_schema_version="v1",
            model="model",
            base_url="",
            api_key="",
            embedding_api_base="",
            embedding_model="embedding",
            eval_file=root / "evaluation_results.jsonl",
            resume=True,
        )

    def test_bootstrap_invalid_completed_artifact_reruns_and_invalidates_dependents(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state = _RunState(root / "run_state.json")
            for stage in _RunState.STAGES:
                state.mark_completed(stage)
            pipeline = BootstrapPipeline(self._bootstrap_config(root))

            self.assertTrue(pipeline._stage_should_run(state, "dataset"))

            persisted = json.loads((root / "run_state.json").read_text(encoding="utf-8"))
            self.assertEqual("invalid", persisted["stages"]["dataset"]["status"])
            for dependent in BootstrapPipeline._STAGE_DEPENDENTS["dataset"]:
                self.assertEqual("invalid", persisted["stages"][dependent]["status"])

    def test_bootstrap_valid_completed_artifact_is_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state = _RunState(root / "run_state.json")
            state.mark_completed("dataset")
            pipeline = BootstrapPipeline(self._bootstrap_config(root))
            pipeline._dataset_dir.mkdir(parents=True)
            (pipeline._dataset_dir / "skill_context_ready.jsonl").write_text(
                json.dumps(
                    {
                        "record_id": "task-1",
                        "version": "v1",
                        "tool_schema_version": "v1",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            self.assertFalse(pipeline._stage_should_run(state, "dataset"))

    def test_evolve_invalid_completed_artifact_reruns_without_deleting_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state = {"steps": {name: {"status": "completed"} for name in STEPS}}
            (root / "run_state.json").write_text(
                json.dumps(state), encoding="utf-8"
            )
            corrupt = root / "summarized_sessions.jsonl"
            corrupt.write_text("{broken\n", encoding="utf-8")
            pipeline = EvolvePipeline(EvolveConfig(output_dir=root, resume=True))

            self.assertTrue(pipeline._should_run("feedback_summary"))

            self.assertEqual("{broken\n", corrupt.read_text(encoding="utf-8"))
            persisted = json.loads((root / "run_state.json").read_text(encoding="utf-8"))
            for step in STEPS:
                self.assertEqual("invalid", persisted["steps"][step]["status"])

    def test_evolve_valid_completed_artifact_is_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state = {"steps": {name: {"status": "pending"} for name in STEPS}}
            state["steps"]["feedback_summary"] = {"status": "completed"}
            (root / "run_state.json").write_text(
                json.dumps(state), encoding="utf-8"
            )
            (root / "summarized_sessions.jsonl").write_text(
                json.dumps({"record_id": "task-1"}) + "\n", encoding="utf-8"
            )
            pipeline = EvolvePipeline(EvolveConfig(output_dir=root, resume=True))

            self.assertFalse(pipeline._should_run("feedback_summary"))

    def test_evolve_programmatic_rollout_input_also_requires_eval_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rollout_dir = root / "rollouts"
            rollout_dir.mkdir()
            cached = root / "skill_context_ready.jsonl"
            cached.write_text(json.dumps({"record_id": "task-1"}) + "\n", encoding="utf-8")
            pipeline = EvolvePipeline(
                EvolveConfig(
                    output_dir=root,
                    rollout_dir=rollout_dir,
                    eval_results_file=None,
                    resume=True,
                )
            )

            with self.assertRaisesRegex(ValueError, "eval_results_file is required"):
                pipeline._resolve_input()

    def test_evolve_prebuilt_dataset_must_match_target_version(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "prebuilt.jsonl"
            dataset.write_text(
                json.dumps(
                    {
                        "record_id": "task-1",
                        "version": "v1",
                        "tool_schema_version": "v1",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            pipeline = EvolvePipeline(
                EvolveConfig(
                    output_dir=root / "out",
                    eval_file=dataset,
                    tool_schema_version="v2",
                    resume=True,
                )
            )

            with self.assertRaisesRegex(ValueError, "expected='v2'"):
                pipeline._resolve_input()


if __name__ == "__main__":
    unittest.main()
