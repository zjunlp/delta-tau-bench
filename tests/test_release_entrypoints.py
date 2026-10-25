"""Portable entry scripts produce complete metrics without dry-run artifacts."""
import os
from pathlib import Path
import shutil
import subprocess
from unittest.mock import patch

import pytest

from dtau_bench.cli import chain


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("entry", ["run_baseline.sh", "run_steward_test.sh"])
def test_one_version_dry_run_includes_metrics_and_writes_nothing(tmp_path, entry):
    project = tmp_path / "project with spaces"
    shutil.copytree(ROOT / "scripts", project / "scripts")
    result = subprocess.run(
        ["bash", str(project / "scripts" / entry)],
        env={**os.environ, "DRY_RUN": "1", "VERSIONS": "v2", "EXP_NAME": "dry_fixture"},
        text=True, capture_output=True, check=True,
    )
    assert "evaluate-multi" in result.stdout
    assert "--weight-modes unit cost" in result.stdout
    assert not (project / "logs").exists()
    assert not (project / "experiments").exists()
    assert not (project / "rag_index_cache").exists()


def test_python_executable_path_can_contain_spaces(tmp_path):
    fake = tmp_path / "python with spaces"
    fake.write_text('#!/bin/bash\n[ "$1" = "-m" ] && [ "$2" = "dtau_bench" ] || exit 88\nexit 29\n')
    fake.chmod(0o755)
    result = subprocess.run(
        ["bash", "-c", 'source scripts/_common.sh; "${DTAU[@]}" --help'],
        cwd=ROOT, env={**os.environ, "PY": str(fake)},
        text=True, capture_output=True,
    )
    assert result.returncode == 29


def test_chain_dry_run_uses_project_root_without_creating_outputs(tmp_path, monkeypatch):
    project = tmp_path / "project"
    (project / "configs").mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DTAU_BENCH_ROOT", str(project))
    monkeypatch.setattr("sys.argv", [
        "chain", "--experiment-name", "fixture", "--train-versions", "v1",
        "--test-versions", "v1", "--dry-run",
    ])
    with patch.object(chain, "run_experiment", return_value={}) as run:
        chain.main()
    assert run.call_args.kwargs["experiments_dir"] == project / "experiments"
    assert not (project / "experiments").exists()
    assert not (tmp_path / "experiments").exists()
