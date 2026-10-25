import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from dtau_bench.skill.bootstrap.dataset import build_skill_dataset
from dtau_bench.skill.models import SkillDocument
from dtau_bench.skill.runtime import SkillContextRuntime, VectorRetriever


class _StaticIndexer:
    def get_query_embedding(self, text):
        return np.array([1.0, 0.0], dtype=np.float32)


def _empty_index():
    return {
        "documents": [],
        "embeddings": np.empty((0, 2), dtype=np.float32),
    }


class SkillRetrievalLimitTests(unittest.TestCase):
    def test_per_step_skill_limit_is_applied_after_threshold(self):
        skills = [
            SkillDocument(
                skill_type="atomic",
                name=f"skill-{i}",
                document="",
                content="",
                tools=[],
                embedding_text=f"skill {i}",
            )
            for i in range(8)
        ]
        embeddings = np.array(
            [[1.0, float(i) / 100.0] for i in range(8)], dtype=np.float32
        )
        runtime = SkillContextRuntime(
            planning_retriever=VectorRetriever(_empty_index()),
            functional_retriever=VectorRetriever(_empty_index()),
            atomic_retriever=VectorRetriever(
                {"documents": skills, "embeddings": embeddings}
            ),
            indexer=_StaticIndexer(),
            llm_client=None,
            config={
                "enable_functional_skills": False,
                "candidate_min_similarity": 0.0,
                "skill_top_k_per_step": 5,
                "final_max_skills": 8,
            },
        )
        runtime._select_skills = lambda query, plan, candidates: candidates

        _, _, _, stats, trace = runtime._retrieve_selected_skills(
            query_text="one step", tools_info=[]
        )

        per_step = trace["step_skill_retrieval"]
        self.assertEqual(5, per_step["skill_top_k_per_step"])
        self.assertEqual(5, len(per_step["per_step_hits"][0]["atomic_hits"]))
        self.assertEqual(5, stats["step_skill_hits_count_before_dedup"])


class SkillDatasetEvaluationTests(unittest.TestCase):
    def _fixture(self, root: Path):
        rollout_dir = root / "rollouts"
        query_dir = root / "queries"
        domain = "DemoDomain"
        (query_dir / domain).mkdir(parents=True)
        rollout_dir.mkdir()

        (query_dir / domain / "task-1.json").write_text(
            json.dumps({"id": "task-1"}), encoding="utf-8"
        )
        rollout = {
            "id": "task-1",
            "success": True,
            "task": {"query_id": "task-1", "instruction": "do it"},
            "conversation": {
                "messages": [{"role": "user", "content": "do it"}],
                "total_turns": 1,
            },
            "run": {"version": "v1"},
            "artifacts": {},
        }
        (rollout_dir / f"{domain}.jsonl").write_text(
            json.dumps(rollout) + "\n", encoding="utf-8"
        )
        return rollout_dir, query_dir

    def test_missing_eval_file_fails_without_writing_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rollout_dir, query_dir = self._fixture(root)
            output = root / "out.jsonl"

            with self.assertRaises(FileNotFoundError):
                build_skill_dataset(
                    rollout_dir=rollout_dir,
                    query_dir=query_dir,
                    eval_file=root / "missing.jsonl",
                    output_path=output,
                    expected_version="v1",
                )
            self.assertFalse(output.exists())

    def test_incomplete_eval_roster_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rollout_dir, query_dir = self._fixture(root)
            eval_file = root / "eval.jsonl"
            eval_file.write_text(
                json.dumps(
                    {"sample_id": "another-task", "version": "v1", "passed": True}
                )
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "roster mismatch"):
                build_skill_dataset(
                    rollout_dir=rollout_dir,
                    query_dir=query_dir,
                    eval_file=eval_file,
                    output_path=root / "out.jsonl",
                    expected_version="v1",
                )

    def test_environment_result_is_the_only_reward_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rollout_dir, query_dir = self._fixture(root)
            eval_file = root / "eval.jsonl"
            eval_file.write_text(
                json.dumps(
                    {
                        "sample_id": "task-1",
                        "version": "v1",
                        "passed": False,
                        "total_score": 0.0,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            output = root / "out.jsonl"

            stats = build_skill_dataset(
                rollout_dir=rollout_dir,
                query_dir=query_dir,
                eval_file=eval_file,
                output_path=output,
                expected_version="v1",
            )

            record = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(1, stats["eval_records"])
            self.assertFalse(record["passed"])
            self.assertEqual(0.0, record["reward"])
            self.assertNotIn("successful_trajectory", record)

    def test_evaluator_version_must_match_rollout_version(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rollout_dir, query_dir = self._fixture(root)
            eval_file = root / "eval.jsonl"
            eval_file.write_text(
                json.dumps(
                    {"sample_id": "task-1", "version": "v2", "passed": True}
                )
                + "\n",
                encoding="utf-8",
            )
            output = root / "out.jsonl"

            with self.assertRaisesRegex(ValueError, "version mismatch"):
                build_skill_dataset(
                    rollout_dir=rollout_dir,
                    query_dir=query_dir,
                    eval_file=eval_file,
                    output_path=output,
                    expected_version="v1",
                )
            self.assertFalse(output.exists())

    def test_evaluator_and_rollout_must_match_target_version(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rollout_dir, query_dir = self._fixture(root)
            eval_file = root / "eval.jsonl"
            eval_file.write_text(
                json.dumps(
                    {"sample_id": "task-1", "version": "v1", "passed": True}
                )
                + "\n",
                encoding="utf-8",
            )
            output = root / "out.jsonl"

            with self.assertRaisesRegex(ValueError, "expected='v2'"):
                build_skill_dataset(
                    rollout_dir=rollout_dir,
                    query_dir=query_dir,
                    eval_file=eval_file,
                    output_path=output,
                    expected_version="v2",
                )
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
