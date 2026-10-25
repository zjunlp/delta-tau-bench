"""Offline dataset discovery and release-integrity checks."""

import hashlib
import json
from pathlib import Path

import pytest

from dtau_bench.cli import validate_data
from dtau_bench.utils.config import get_data_root


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / "dataset"
    domain = "ExampleDomain"
    versions = validate_data.VERSIONS
    mapping = {"versions": {}}
    for version in versions:
        name = f"operation_{version}"
        _write_json(root / "tools" / domain / f"{version}_tools.json", [
            {"tool": {"type": "function", "function": {"name": name}}},
        ])
        _write_json(root / "meta/tool_attributes" / domain / f"{version}.json", {name: {"cost": 1}})
        mapping["versions"][version] = {"tools": [{"tool_name": name}]}
        server_dir = root / "servers" if version == "base" else root / "servers" / version
        server_dir.mkdir(parents=True, exist_ok=True)
        (server_dir / f"{domain}.py").write_text("# Static validation never executes servers.\n")
    for split in ("train", "test"):
        _write_json(root / "tasks" / split / domain / f"{split}-task.json", {
            "query_id": f"{split}-task", "domain": domain, "split": split,
            "version_trajectories": {v: [f"operation_{v}"] for v in versions},
        })
    _write_json(root / "meta/version_mapping" / f"{domain}.json", mapping)
    _write_json(root / "meta/evolution_forests" / f"{domain}_forest.json", {})
    _write_json(root / "meta/eval_field_meta" / f"{domain}.json", {})
    for name in ("tool_costs", "eval_identity_schema", "replay_identity_schema", "state_compatibility_schema"):
        _write_json(root / "meta" / f"{name}.json", {})
    _write_json(root / "database/entities/example.json", [])
    _write_json(root / "database/relationships/ExampleDomain/example.json", [])
    return root


def _validate(root):
    return validate_data.validate_dataset(root, expected_splits={"train": 1, "test": 1})


def test_data_root_precedence_and_relative_paths(tmp_path, monkeypatch):
    project = tmp_path / "code"
    (project / "configs").mkdir(parents=True)
    monkeypatch.setenv("DTAU_BENCH_ROOT", str(project))
    monkeypatch.delenv("DTAU_BENCH_DATA_ROOT", raising=False)
    assert get_data_root() == project / "data"
    monkeypatch.setenv("DTAU_BENCH_DATA_ROOT", "../dataset")
    assert get_data_root() == tmp_path / "dataset"
    assert get_data_root("override") == project / "override"
    assert get_data_root(tmp_path / "explicit") == tmp_path / "explicit"


def test_absolute_data_root_does_not_require_code_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DTAU_BENCH_ROOT", raising=False)
    monkeypatch.setenv("DTAU_BENCH_DATA_ROOT", str(tmp_path / "dataset"))
    assert get_data_root() == tmp_path / "dataset"


def test_complete_fixture_validates_offline(dataset, monkeypatch):
    import dtau_bench.utils.config as config

    def no_model_configuration(*args, **kwargs):
        pytest.fail("Dataset validation must not read model configuration")

    monkeypatch.setattr(config, "load_models_config", no_model_configuration)
    report = _validate(dataset)
    assert report["valid"], report["errors"]
    assert report["unique_tasks"] == 2
    assert report["catalogs"] == 7
    assert report["parsed_json_files"] == report["json_files"]


@pytest.mark.parametrize("failure", ["missing_catalog", "missing_server", "invalid_json", "wrong_gold", "mapping_mismatch", "duplicate_task"])
def test_incomplete_or_inconsistent_dataset_is_rejected(dataset, failure):
    if failure == "missing_catalog":
        (dataset / "tools/ExampleDomain/v2_tools.json").unlink()
    elif failure == "missing_server":
        (dataset / "servers/v2/ExampleDomain.py").unlink()
    elif failure == "invalid_json":
        (dataset / "database/entities/example.json").write_text("{broken")
    elif failure == "mapping_mismatch":
        path = dataset / "meta/version_mapping/ExampleDomain.json"
        payload = json.loads(path.read_text())
        payload["versions"]["v2"]["tools"] = []
        _write_json(path, payload)
    else:
        path = dataset / "tasks/test/ExampleDomain/test-task.json"
        payload = json.loads(path.read_text())
        if failure == "wrong_gold":
            payload["version_trajectories"]["v2"] = ["operation_base"]
        else:
            payload["query_id"] = "train-task"
            path.unlink()
            path = path.with_name("train-task.json")
        _write_json(path, payload)
    report = _validate(dataset)
    assert not report["valid"]
    assert report["errors"]


def test_release_task_counts_are_enforced(dataset):
    report = validate_data.validate_dataset(dataset)
    assert not report["valid"]
    assert any("expected 1191" in error for error in report["errors"])
    assert any("expected 397" in error for error in report["errors"])


def test_optional_checksums_cover_all_dataset_files(dataset):
    files = sorted(p for p in dataset.rglob("*") if p.is_file())
    (dataset / "README.md").write_text("Dataset documentation\n")
    (dataset / "checksums.sha256").write_text("".join(
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(dataset).as_posix()}\n"
        for path in files
    ))
    report = _validate(dataset)
    assert report["valid"], report["errors"]
    assert report["checksum_files"] == len(files)
    cache_dir = dataset / "servers/v2/__pycache__"
    cache_dir.mkdir()
    (cache_dir / "ExampleDomain.cpython-312.pyc").write_bytes(b"regenerable bytecode")
    (dataset / "servers/ExampleDomain.pyo").write_bytes(b"legacy bytecode")
    report = _validate(dataset)
    assert report["valid"], report["errors"]
    assert report["checksum_files"] == len(files)
    (dataset / "database/entities/example.json").write_text('[{"changed": true}]')
    report = _validate(dataset)
    assert not report["valid"]
    assert "Checksum mismatch: database/entities/example.json" in report["errors"]
    _write_json(dataset / "database/entities/unexpected.json", [])
    report = _validate(dataset)
    assert "Dataset file is absent from checksums.sha256: database/entities/unexpected.json" in report["errors"]


def test_checksum_manifest_cannot_silently_omit_data(dataset):
    (dataset / "checksums.sha256").write_text("")
    report = _validate(dataset)
    assert not report["valid"]
    assert any("absent from checksums.sha256" in error for error in report["errors"])


def test_empty_dataset_cli_reports_setup_instructions(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("DTAU_BENCH_DATA_ROOT", str(tmp_path))
    monkeypatch.setattr("sys.argv", ["validate-data"])
    with pytest.raises(SystemExit) as exc:
        validate_data.main()
    assert exc.value.code == 1
    assert "DTAU_BENCH_DATA_ROOT" in capsys.readouterr().err
