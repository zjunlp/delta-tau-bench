"""Offline tests for Steward CPU and memory resource controls."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from dtau_bench.cli import bootstrap, chain, evolve


ROOT = Path(__file__).resolve().parents[1]


def _option_value(args: list[str], option: str) -> str:
    return args[args.index(option) + 1]


class ChainResourcePropagationTests(unittest.TestCase):
    def setUp(self) -> None:
        chain._set_dry_run(False)

    def test_chain_forwards_every_resource_control_to_subcommands(self) -> None:
        calls: list[tuple[str, list[str]]] = []

        def record(subcommand: str, args: list[str], *, dry_run: bool) -> None:
            self.assertTrue(dry_run)
            calls.append((subcommand, args))

        with tempfile.TemporaryDirectory() as tmp, patch.object(
            chain, "_experiment_runtime_signature", return_value={"fixture": True}
        ), patch.object(chain, "_run_cli", side_effect=record):
            chain.run_experiment(
                experiment_name="resource_fixture",
                train_versions=["v1", "v2"],
                test_versions=["v1"],
                bootstrap_strategy="single_domain_full",
                train_strategy="train_steward",
                test_strategy="steward",
                agent_model="agent",
                user_model="user",
                evolve_model="evolve",
                experiments_dir=Path(tmp) / "experiments",
                data_dir=Path(tmp) / "external-dataset",
                resume=True,
                dry_run=True,
                rollout_domain_workers=3,
                rollout_query_workers=5,
                evaluate_workers=7,
                skill_batch_size=11,
                skill_max_concurrent=13,
                merge_embedding_batch_size=17,
                embedding_batch_size=19,
            )

        rollout_calls = [args for command, args in calls if command == "rollout"]
        evaluate_calls = [args for command, args in calls if command == "evaluate"]
        bootstrap_calls = [args for command, args in calls if command == "bootstrap"]
        evolve_calls = [args for command, args in calls if command == "evolve"]
        skill_index_calls = [
            args for command, args in calls if command == "build-skill-index"
        ]

        self.assertGreaterEqual(len(rollout_calls), 3)
        for args in rollout_calls:
            self.assertEqual(_option_value(args, "--max-workers"), "3")
            self.assertEqual(_option_value(args, "--query-workers"), "5")
        self.assertGreaterEqual(len(evaluate_calls), 3)
        for args in evaluate_calls:
            self.assertEqual(_option_value(args, "--workers"), "7")
        self.assertEqual(len(bootstrap_calls), 1)
        self.assertEqual(len(evolve_calls), 1)
        for args in bootstrap_calls + evolve_calls:
            self.assertEqual(_option_value(args, "--batch-size"), "11")
            self.assertEqual(_option_value(args, "--max-concurrent"), "13")
            self.assertEqual(
                _option_value(args, "--merge-embedding-batch-size"), "17"
            )
        self.assertGreaterEqual(len(skill_index_calls), 1)
        for args in skill_index_calls:
            self.assertEqual(_option_value(args, "--batch-size"), "19")

        for args in rollout_calls + evaluate_calls + bootstrap_calls + evolve_calls:
            self.assertEqual(
                _option_value(args, "--data-dir"),
                str(Path(tmp) / "external-dataset"),
            )

    def test_resume_rejects_resource_setting_change(self) -> None:
        metadata = {
            "experiment_name": "fixture",
            "runtime_signature": {
                "resources": {"rollout_domain_workers": 1}
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run_state.json"
            chain._init_run_state(path, resume=False, metadata=metadata)
            changed = json.loads(json.dumps(metadata))
            changed["runtime_signature"]["resources"]["rollout_domain_workers"] = 2

            with self.assertRaisesRegex(ValueError, "different configuration"):
                chain._init_run_state(path, resume=True, metadata=changed)

    def test_resume_allows_only_resource_change_with_opt_in(self) -> None:
        metadata = {
            "experiment_name": "fixture",
            "runtime_signature": {
                "models": {"agent": {"key": "model-a"}},
                "resources": {"skill_max_concurrent": 2},
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run_state.json"
            chain._init_run_state(path, resume=False, metadata=metadata)
            changed = json.loads(json.dumps(metadata))
            changed["runtime_signature"]["resources"]["skill_max_concurrent"] = 4

            resumed = chain._init_run_state(
                path,
                resume=True,
                metadata=changed,
                allow_resource_change_on_resume=True,
            )

            self.assertEqual(
                resumed["metadata"]["runtime_signature"]["resources"][
                    "skill_max_concurrent"
                ],
                4,
            )
            self.assertEqual(
                resumed["config_fingerprint"],
                chain._configuration_fingerprint(changed),
            )
            self.assertEqual(len(resumed["resource_change_history"]), 1)
            persisted = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(persisted, resumed)
            chain._init_run_state(path, resume=True, metadata=changed)

    def test_resource_change_opt_in_rejects_non_resource_change(self) -> None:
        metadata = {
            "experiment_name": "fixture",
            "runtime_signature": {
                "models": {"agent": {"key": "model-a"}},
                "resources": {"skill_max_concurrent": 2},
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run_state.json"
            chain._init_run_state(path, resume=False, metadata=metadata)
            changed = json.loads(json.dumps(metadata))
            changed["runtime_signature"]["resources"]["skill_max_concurrent"] = 4
            changed["runtime_signature"]["models"]["agent"]["key"] = "model-b"

            with self.assertRaisesRegex(ValueError, "different configuration"):
                chain._init_run_state(
                    path,
                    resume=True,
                    metadata=changed,
                    allow_resource_change_on_resume=True,
                )

    def test_resource_change_opt_in_does_not_bypass_tamper_check(self) -> None:
        metadata = {
            "experiment_name": "fixture",
            "runtime_signature": {
                "resources": {"skill_max_concurrent": 2},
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run_state.json"
            chain._init_run_state(path, resume=False, metadata=metadata)
            state = json.loads(path.read_text(encoding="utf-8"))
            state["metadata"]["runtime_signature"]["resources"][
                "skill_max_concurrent"
            ] = 3
            path.write_text(json.dumps(state), encoding="utf-8")
            changed = json.loads(json.dumps(metadata))
            changed["runtime_signature"]["resources"]["skill_max_concurrent"] = 4

            with self.assertRaisesRegex(ValueError, "fingerprint"):
                chain._init_run_state(
                    path,
                    resume=True,
                    metadata=changed,
                    allow_resource_change_on_resume=True,
                )

    def test_skill_index_validation_uses_chain_batch_override(self) -> None:
        embedding = {
            "model": "fixture-embedding",
            "backend": "local",
            "base_url": "",
            "api_key": "",
            "batch_size": 64,
        }
        with patch(
            "dtau_bench.skill.loader.SkillLibraryLoader.load",
            return_value=SimpleNamespace(tool_skills=[]),
        ), patch(
            "dtau_bench.utils.config.resolve_embedding_config",
            return_value=embedding,
        ), patch.object(chain, "_validate_embedding_index_dir") as validate_dir:
            chain._validate_skill_index_artifacts(
                Path("/tmp/fixture-library"),
                embedding_batch_size=8,
            )

        for call in validate_dir.call_args_list:
            self.assertEqual(
                call.kwargs["expected_metadata"]["embedding_batch_size"], 8
            )


class SkillCliResourceTests(unittest.TestCase):
    @staticmethod
    def _endpoint() -> dict:
        return {
            "model_name": "fixture-model",
            "base_url": "https://example.invalid/v1",
            "api_key": "",
            "max_tokens": 1024,
            "request_options": {},
        }

    @staticmethod
    def _embedding() -> dict:
        return {
            "model": "fixture-embedding",
            "base_url": "https://embedding.invalid/v1",
            "api_key": "",
        }

    def test_bootstrap_cli_writes_resource_values_to_config(self) -> None:
        captured = []

        class FakePipeline:
            def __init__(self, config) -> None:
                captured.append(config)

            def run(self) -> dict:
                return {"outputs": {"library_dir": "/tmp/library"}}

        with tempfile.TemporaryDirectory() as tmp:
            rollout_dir = Path(tmp) / "rollouts"
            rollout_dir.mkdir()
            eval_file = Path(tmp) / "evaluation.jsonl"
            eval_file.touch()
            argv = [
                "bootstrap",
                "--run-name", "fixture",
                "--rollout-dir", str(rollout_dir),
                "--eval-file", str(eval_file),
                "--tool-schema-version", "v1",
                "--model", "fixture",
                "--batch-size", "3",
                "--max-concurrent", "4",
                "--merge-embedding-batch-size", "5",
            ]
            with patch.object(sys, "argv", argv), patch.object(
                bootstrap, "resolve_model_endpoint", return_value=self._endpoint()
            ), patch.object(
                bootstrap, "resolve_embedding_config", return_value=self._embedding()
            ), patch(
                "dtau_bench.skill.bootstrap.pipeline.BootstrapPipeline", FakePipeline
            ):
                bootstrap.main()

        self.assertEqual(captured[0].batch_size, 3)
        self.assertEqual(captured[0].max_concurrent, 4)
        self.assertEqual(captured[0].merge_embedding_batch_size, 5)

    def test_evolve_cli_writes_resource_values_to_config(self) -> None:
        captured = []

        class FakePipeline:
            def __init__(self, config) -> None:
                captured.append(config)

            async def run(self) -> None:
                return None

        argv = [
            "evolve",
            "--run-name", "fixture",
            "--eval-file", "/tmp/eval.jsonl",
            "--tool-schema-version", "v2",
            "--model", "fixture",
            "--batch-size", "6",
            "--max-concurrent", "7",
            "--merge-embedding-batch-size", "9",
        ]
        with patch.object(sys, "argv", argv), patch.object(
            evolve, "resolve_model_endpoint", return_value=self._endpoint()
        ), patch.object(
            evolve, "resolve_embedding_config", return_value=self._embedding()
        ), patch("dtau_bench.skill.evolve.EvolvePipeline", FakePipeline):
            evolve.main()

        self.assertEqual(captured[0].batch_size, 6)
        self.assertEqual(captured[0].max_concurrent, 7)
        self.assertEqual(captured[0].merge_embedding_batch_size, 9)


class StewardShellDefaultsTests(unittest.TestCase):
    def test_run_step_appends_across_resume_invocations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "resume.log"
            for value in ("before", "after"):
                subprocess.run(
                    [
                        "bash",
                        "-c",
                        "source scripts/_common.sh; "
                        f"run_step {str(log)!r} printf '%s\\n' {value!r}",
                    ],
                    cwd=ROOT,
                    text=True,
                    capture_output=True,
                    check=True,
                )

            self.assertEqual(log.read_text().splitlines(), ["before", "after"])

    def test_cpu_safe_defaults_are_forwarded_by_shared_shell_config(self) -> None:
        env = {
            **os.environ,
            "DRY_RUN": "1",
            "EXP_NAME": "resource_fixture",
        }
        for name in (
            "ROLLOUT_DOMAIN_WORKERS",
            "ROLLOUT_QUERY_WORKERS",
            "EVALUATE_WORKERS",
            "SKILL_BATCH_SIZE",
            "SKILL_MAX_CONCURRENT",
            "MERGE_EMBEDDING_BATCH_SIZE",
            "EMBEDDING_BATCH_SIZE",
        ):
            env.pop(name, None)
        result = subprocess.run(
            [
                "bash",
                "-c",
                "source scripts/_common.sh; source scripts/_steward_env.sh; "
                "printf '%s\\n' \"${CHAIN_OPTS[@]}\"",
            ],
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=True,
        )
        args = result.stdout.splitlines()

        self.assertEqual(_option_value(args, "--rollout-domain-workers"), "1")
        self.assertEqual(_option_value(args, "--rollout-query-workers"), "4")
        self.assertEqual(_option_value(args, "--evaluate-workers"), "2")
        self.assertEqual(_option_value(args, "--skill-batch-size"), "2")
        self.assertEqual(_option_value(args, "--skill-max-concurrent"), "2")
        self.assertEqual(_option_value(args, "--merge-embedding-batch-size"), "8")
        self.assertEqual(_option_value(args, "--embedding-batch-size"), "8")
        self.assertNotIn("--allow-resource-change-on-resume", args)

    def test_shell_can_explicitly_allow_resource_change_on_resume(self) -> None:
        env = {
            **os.environ,
            "DRY_RUN": "1",
            "EXP_NAME": "resource_fixture",
            "ALLOW_RESOURCE_CHANGE_ON_RESUME": "1",
        }
        result = subprocess.run(
            [
                "bash",
                "-c",
                "source scripts/_common.sh; source scripts/_steward_env.sh; "
                "printf '%s\\n' \"${CHAIN_OPTS[@]}\"",
            ],
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=True,
        )

        self.assertIn(
            "--allow-resource-change-on-resume", result.stdout.splitlines()
        )

    def test_tool_index_build_and_validation_receive_embedding_batch_size(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env = {
                **os.environ,
                "DRY_RUN": "1",
                "EMBEDDING_BATCH_SIZE": "8",
                "RAG_INDEX_BASE": str(Path(tmp) / "indexes"),
            }
            result = subprocess.run(
                [
                    "bash",
                    "-c",
                    "source scripts/_common.sh; LOG_DIR=/tmp/steward-test-logs; "
                    "ensure_tool_index v1",
                ],
                cwd=ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=True,
            )

        commands = [line for line in result.stdout.splitlines() if "build-tool-index" in line]
        self.assertEqual(len(commands), 2)
        self.assertTrue(all("--batch-size 8" in command for command in commands))
        self.assertTrue(any("--validate-only" in command for command in commands))


if __name__ == "__main__":
    unittest.main()
