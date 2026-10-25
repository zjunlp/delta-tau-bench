import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from dtau_bench.cli.rollout import process_domain
from dtau_bench.rollout.record_store import (
    clear_domain_records,
    load_domain_records,
    rebuild_domain_jsonl,
    recover_domain_records,
    remove_record_checkpoints,
    save_record,
)
from dtau_bench.tools.data_loader import GoldTrajectory, TaskData


DOMAIN = "ExampleDomain"
VERSION = "v1"


def make_record(
    query_id: str,
    *,
    success: bool = True,
    error=None,
    domain: str = DOMAIN,
    version: str = VERSION,
):
    return {
        "schema_version": 2,
        "id": query_id,
        "success": success,
        "error": error,
        "task": {
            "query_id": query_id,
            "domain_key": domain,
            "domains": [domain],
        },
        "agent_traj": [],
        "conversation": {"messages": [], "total_turns": 0},
        "run": {"version": version},
        "artifacts": {},
    }


class RecordStoreTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.tempdir.name)

    def tearDown(self):
        self.tempdir.cleanup()

    @property
    def rollout_path(self):
        return self.output_dir / "rollouts" / f"{DOMAIN}.jsonl"

    def save(self, query_id, record=None):
        save_record(
            self.output_dir,
            DOMAIN,
            query_id,
            record or make_record(query_id),
            expected_version=VERSION,
        )

    def read_aggregate(self):
        return [
            json.loads(line)
            for line in self.rollout_path.read_text(encoding="utf-8").splitlines()
            if line
        ]

    def test_checkpoints_repair_malformed_and_missing_jsonl_rows(self):
        self.save("q1")
        self.save("q2")
        self.rollout_path.parent.mkdir(parents=True, exist_ok=True)
        self.rollout_path.write_bytes(b'{"id":"overwritten"')

        recovered = recover_domain_records(
            self.output_dir,
            DOMAIN,
            {"q1", "q2"},
            expected_version=VERSION,
        )

        self.assertEqual(recovered["malformed_lines"], [1])
        self.assertEqual(recovered["restored_from_checkpoints"], ["q1", "q2"])
        self.assertEqual([row["id"] for row in self.read_aggregate()], ["q1", "q2"])

    def test_legacy_jsonl_is_imported_before_rebuild(self):
        record = make_record("legacy")
        self.rollout_path.parent.mkdir(parents=True, exist_ok=True)
        self.rollout_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

        recovered = recover_domain_records(
            self.output_dir,
            DOMAIN,
            {"legacy"},
            expected_version=VERSION,
        )

        self.assertEqual(recovered["record_ids"], {"legacy"})
        self.assertEqual(
            load_domain_records(
                self.output_dir,
                DOMAIN,
                {"legacy"},
                expected_version=VERSION,
            )["legacy"],
            record,
        )

    def test_verified_checkpoint_wins_over_conflicting_jsonl(self):
        checkpoint = make_record("q1")
        self.save("q1", checkpoint)
        conflicting = make_record(
            "q1",
            success=False,
            error="different",
            domain="OtherDomain",
            version="v2",
        )
        self.rollout_path.parent.mkdir(parents=True, exist_ok=True)
        self.rollout_path.write_text(json.dumps(conflicting) + "\n", encoding="utf-8")

        recovered = recover_domain_records(
            self.output_dir,
            DOMAIN,
            {"q1"},
            expected_version=VERSION,
        )

        self.assertEqual(recovered["checkpoint_conflicts"], ["q1"])
        self.assertEqual(self.read_aggregate(), [checkpoint])

    def test_retry_removal_updates_both_representations(self):
        self.save("retry", make_record("retry", success=False, error="HTTP 503"))
        self.save("done")
        rebuild_domain_jsonl(
            self.output_dir,
            DOMAIN,
            {"retry", "done"},
            expected_version=VERSION,
        )

        removed = remove_record_checkpoints(
            self.output_dir,
            DOMAIN,
            {"retry"},
            expected_ids={"retry", "done"},
            expected_version=VERSION,
        )
        recovered = recover_domain_records(
            self.output_dir,
            DOMAIN,
            {"retry", "done"},
            expected_version=VERSION,
        )

        self.assertEqual(removed, 1)
        self.assertEqual(recovered["record_ids"], {"done"})
        self.assertEqual(recovered["missing_ids"], ["retry"])
        self.assertEqual([row["id"] for row in self.read_aggregate()], ["done"])

        replacement = make_record("retry", success=False, error="new attempt")
        self.save("retry", replacement)
        recovered = recover_domain_records(
            self.output_dir,
            DOMAIN,
            {"retry", "done"},
            expected_version=VERSION,
        )
        self.assertEqual(recovered["pending_retry_ids"], [])
        self.assertEqual(recovered["record_ids"], {"retry", "done"})

    def test_retry_marker_survives_interruption_before_jsonl_rebuild(self):
        self.save("retry", make_record("retry", success=False, error="HTTP 503"))
        rebuild_domain_jsonl(
            self.output_dir,
            DOMAIN,
            {"retry"},
            expected_version=VERSION,
        )

        with patch(
            "dtau_bench.rollout.record_store.rebuild_domain_jsonl",
            side_effect=RuntimeError("simulated interruption"),
        ):
            with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                remove_record_checkpoints(
                    self.output_dir,
                    DOMAIN,
                    {"retry"},
                    expected_ids={"retry"},
                    expected_version=VERSION,
                )

        recovered = recover_domain_records(
            self.output_dir,
            DOMAIN,
            {"retry"},
            expected_version=VERSION,
        )
        self.assertEqual(recovered["pending_retry_ids"], ["retry"])
        self.assertEqual(recovered["record_ids"], set())
        self.assertEqual(recovered["missing_ids"], ["retry"])

    def test_committed_replacement_wins_if_marker_cleanup_is_interrupted(self):
        original = make_record("retry", success=False, error="HTTP 503")
        replacement = make_record("retry", success=True)
        self.save("retry", original)
        rebuild_domain_jsonl(
            self.output_dir,
            DOMAIN,
            {"retry"},
            expected_version=VERSION,
        )
        remove_record_checkpoints(
            self.output_dir,
            DOMAIN,
            {"retry"},
            expected_ids={"retry"},
            expected_version=VERSION,
        )

        with patch(
            "dtau_bench.rollout.record_store._unlink_durable",
            side_effect=RuntimeError("simulated cleanup interruption"),
        ):
            with self.assertRaisesRegex(RuntimeError, "cleanup interruption"):
                self.save("retry", replacement)

        recovered = recover_domain_records(
            self.output_dir,
            DOMAIN,
            {"retry"},
            expected_version=VERSION,
        )
        self.assertEqual(recovered["pending_retry_ids"], [])
        self.assertEqual(recovered["record_ids"], {"retry"})
        self.assertEqual(self.read_aggregate(), [replacement])

    def test_retry_marker_is_bound_to_its_version(self):
        self.save("retry", make_record("retry", success=False, error="HTTP 503"))
        rebuild_domain_jsonl(
            self.output_dir,
            DOMAIN,
            {"retry"},
            expected_version=VERSION,
        )
        remove_record_checkpoints(
            self.output_dir,
            DOMAIN,
            {"retry"},
            expected_ids={"retry"},
            expected_version=VERSION,
        )

        with self.assertRaisesRegex(ValueError, "version mismatch"):
            recover_domain_records(
                self.output_dir,
                DOMAIN,
                {"retry"},
                expected_version="v2",
            )

    def test_clear_removes_jsonl_and_checkpoints(self):
        self.save("q1")
        rebuild_domain_jsonl(
            self.output_dir,
            DOMAIN,
            {"q1"},
            expected_version=VERSION,
        )

        clear_domain_records(self.output_dir, DOMAIN)

        self.assertFalse(self.rollout_path.exists())
        self.assertFalse((self.output_dir / "record_checkpoints" / DOMAIN).exists())

    def test_clear_removes_stale_atomic_write_temporary_file(self):
        checkpoint_dir = self.output_dir / "record_checkpoints" / DOMAIN
        checkpoint_dir.mkdir(parents=True)
        (checkpoint_dir / ".q1.json.123.456.tmp").write_text(
            "partial",
            encoding="utf-8",
        )
        self.rollout_path.parent.mkdir(parents=True)
        self.rollout_path.write_text("partial", encoding="utf-8")

        clear_domain_records(self.output_dir, DOMAIN)

        self.assertFalse(checkpoint_dir.exists())
        self.assertFalse(self.rollout_path.exists())

    def test_record_identity_domain_and_version_are_validated(self):
        with self.assertRaisesRegex(ValueError, "query ID mismatch"):
            self.save("q1", make_record("q2"))
        with self.assertRaisesRegex(ValueError, "domain mismatch"):
            self.save("q1", make_record("q1", domain="OtherDomain"))
        with self.assertRaisesRegex(ValueError, "version mismatch"):
            self.save("q1", make_record("q1", version="v2"))
        missing_query_id = make_record("q1")
        del missing_query_id["task"]["query_id"]
        with self.assertRaisesRegex(ValueError, "query ID mismatch"):
            self.save("q1", missing_query_id)
        missing_domain = make_record("q1")
        del missing_domain["task"]["domain_key"]
        del missing_domain["task"]["domains"]
        with self.assertRaisesRegex(ValueError, "domain missing"):
            self.save("q1", missing_domain)
        conflicting_domains = make_record("q1")
        conflicting_domains["task"]["domain"] = "OtherDomain"
        with self.assertRaisesRegex(ValueError, "domain mismatch"):
            self.save("q1", conflicting_domains)

        checkpoint_dir = self.output_dir / "record_checkpoints" / DOMAIN
        checkpoint_dir.mkdir(parents=True)
        (checkpoint_dir / "q1.json").write_text(
            json.dumps(make_record("q2")) + "\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "ID mismatch"):
            load_domain_records(self.output_dir, DOMAIN, {"q1", "q2"})


class ProcessDomainFailureTests(unittest.TestCase):
    def test_tool_setup_failure_is_not_reported_as_a_completed_domain(self):
        task = TaskData(
            query_id="q1",
            domain=DOMAIN,
            split="test",
            instruction="test",
            messages=[{"role": "user", "content": "test"}],
            user_system_prompt="test",
            gold_trajectory=GoldTrajectory(tools=[]),
            version_trajectories={VERSION: []},
        )
        with tempfile.TemporaryDirectory() as output_tmp, tempfile.TemporaryDirectory() as data_tmp:
            with patch(
                "dtau_bench.cli.rollout.setup_domain_tools",
                side_effect=RuntimeError("broken tools"),
            ):
                with self.assertRaisesRegex(RuntimeError, "Tool setup failed"):
                    process_domain(
                        domain=DOMAIN,
                        tasks=[task],
                        version=VERSION,
                        run_config={"strategy": "single_domain_full"},
                        output_dir=Path(output_tmp),
                        data_dir=Path(data_tmp),
                        resume=False,
                        verbose=False,
                    )


if __name__ == "__main__":
    unittest.main()
