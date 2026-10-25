"""Offline regression coverage for a dataset stored outside the code tree."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from dtau_bench.evaluator.data_loader import TaskMetadataProvider
from dtau_bench.evaluator.engine import EnvironmentEvaluationEngine
from dtau_bench.evaluator.gold_weight import (
    load_evolution_forest,
    load_tool_weights_for_domain,
    load_version_mapping,
)
from dtau_bench.skill.evolve.pipeline import EvolveConfig, EvolvePipeline
from dtau_bench.tools.mcp_wrapper import setup_domain_tools


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


class ExternalDataCallsiteTests(unittest.TestCase):
    def test_evaluator_and_gold_loaders_use_the_selected_data_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "external-dataset"
            override = root / "override-dataset"
            _write_json(dataset / "meta/version_mapping/Domain.json", {"source": "env"})
            _write_json(override / "meta/version_mapping/Domain.json", {"source": "explicit"})
            _write_json(dataset / "meta/evolution_forests/Domain_forest.json", {"nodes": []})
            _write_json(dataset / "meta/tool_attributes/Domain/v2.json", {"tool": {"cost": 3}})
            _write_json(dataset / "meta/eval_field_meta/Domain.json", {"entity_id_names": []})
            with patch.dict(os.environ, {"DTAU_BENCH_DATA_ROOT": str(dataset)}):
                self.assertEqual(load_version_mapping("Domain"), {"source": "env"})
                self.assertEqual(
                    load_version_mapping("Domain", override), {"source": "explicit"}
                )
                self.assertEqual(load_evolution_forest("Domain"), {"nodes": []})
                self.assertEqual(load_tool_weights_for_domain("Domain"), {"v2": {"tool": 3.0}})
                engine = EnvironmentEvaluationEngine(TaskMetadataProvider(dataset / "tasks"))
                self.assertEqual(engine.data_dir, str(dataset))
                self.assertIn("Domain", engine.field_meta)

    def test_evolve_builds_input_from_external_training_tasks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "external-dataset"
            rollout_dir = root / "rollouts"
            eval_file = root / "evaluation.jsonl"
            _write_json(dataset / "tasks/train/Domain/q1.json", {"id": "q1"})
            _write_json(
                rollout_dir / "Domain.jsonl",
                {
                    "id": "q1",
                    "success": True,
                    "task": {"query_id": "q1", "instruction": "External dataset task"},
                    "conversation": {
                        "messages": [{"role": "user", "content": "External dataset task"}],
                        "total_turns": 1,
                    },
                    "run": {"version": "v2"},
                    "artifacts": {},
                },
            )
            _write_json(eval_file, {"sample_id": "q1", "version": "v2", "passed": True})
            with patch.dict(os.environ, {"DTAU_BENCH_DATA_ROOT": str(dataset)}):
                config = EvolveConfig(
                    output_dir=root / "evolved",
                    rollout_dir=rollout_dir,
                    eval_results_file=eval_file,
                    tool_schema_version="v2",
                )
                result_path = EvolvePipeline(config)._resolve_input()
            record = json.loads(result_path.read_text(encoding="utf-8"))
            self.assertEqual(record["task_id"], "q1")
            self.assertTrue(record["passed"])

    def test_server_database_is_bound_to_selected_root_for_each_version(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp) / "external-dataset"
            _write_json(dataset / "database/entities/table.json", [{"origin": "global"}])
            _write_json(dataset / "database/entities/Domain/table.json", [{"origin": "domain"}])
            _write_json(dataset / "database/relationships/Domain/links.json", [{"id": "r1"}])
            source = (
                "class DomainMCP:\n"
                "    def __init__(self, domain_name):\n"
                "        self.domain_name = domain_name\n"
                "        self.base_database = {'legacy': [{'origin': 'wrong-path'}]}\n"
                "        self.sessions = {'stale': object()}\n"
            )
            for version in ("base", "v2"):
                with self.subTest(version=version):
                    server_dir = dataset / "servers"
                    if version != "base":
                        server_dir /= version
                    server_dir.mkdir(parents=True, exist_ok=True)
                    (server_dir / "Domain.py").write_text(source, encoding="utf-8")
                    _write_json(dataset / "tools/Domain" / f"{version}_tools.json", [])
                    wrapper = setup_domain_tools("Domain", version, dataset)
                    self.assertEqual(
                        wrapper.server.base_database,
                        {"table": [{"origin": "domain"}], "links": [{"id": "r1"}]},
                    )
                    self.assertEqual(wrapper.server.sessions, {})

    def test_constructor_and_later_reload_never_use_legacy_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp) / "external-dataset"
            table = dataset / "database/entities/table.json"
            _write_json(table, [{"origin": "selected-dataset"}])
            source = (
                "class DomainMCP:\n"
                "    def __init__(self, domain_name):\n"
                "        self.domain_name = domain_name\n"
                "        self.base_database = {}\n"
                "        self.sessions = {}\n"
                "        self._load_database(domain_name)\n"
                "    def _load_database(self, domain_name):\n"
                "        raise AssertionError('legacy loader must not run')\n"
            )
            server_dir = dataset / "servers/v5"
            server_dir.mkdir(parents=True)
            (server_dir / "Domain.py").write_text(source)
            _write_json(dataset / "tools/Domain/v5_tools.json", [])
            server = setup_domain_tools("Domain", "v5", dataset).server
            self.assertEqual(server.base_database["table"], [{"origin": "selected-dataset"}])
            session = object()
            server.sessions["existing"] = session
            original_database = server.base_database
            server.base_database["unchanged"] = []
            _write_json(table, [{"origin": "updated-dataset"}])
            self.assertIsNone(server._load_database("Domain"))
            self.assertIs(server.base_database, original_database)
            self.assertEqual(server.base_database["table"], [{"origin": "updated-dataset"}])
            self.assertIn("unchanged", server.base_database)
            self.assertIs(server.sessions["existing"], session)


if __name__ == "__main__":
    unittest.main()
