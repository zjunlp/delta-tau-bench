"""Check the released dataset without loading models or executing servers."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping

from dtau_bench.utils.config import get_data_root


VERSIONS = ("base", "v1", "v2", "v3", "v4", "v5", "v6")
EXPECTED_SPLITS = {"train": 1191, "test": 397}
REQUIRED_DIRECTORIES = ("tasks", "tools", "database", "servers", "meta")
DATA_HELP = (
    "Set DTAU_BENCH_DATA_ROOT or pass --data-dir to the extracted dataset "
    "directory containing tasks/, tools/, database/, servers/ and meta/. "
    "The code archive does not include the dataset."
)


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-JSON numeric constant {value}")


def _verify_checksums(root: Path, errors: list[str]) -> int | None:
    """Check an optional release checksum file, including its file coverage."""
    manifest = root / "checksums.sha256"
    if not manifest.is_file():
        return None
    listed: set[str] = set()
    for line_number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        match = re.fullmatch(r"([0-9a-fA-F]{64}) [ *](.+)", line)
        if match is None:
            errors.append(f"Invalid checksum entry at checksums.sha256:{line_number}")
            continue
        digest, name = match.groups()
        relative = Path(name)
        if not relative.parts or relative.is_absolute() or ".." in relative.parts or relative.parts[0] not in REQUIRED_DIRECTORIES:
            errors.append(f"Invalid dataset path in checksums.sha256:{line_number}: {name}")
            continue
        name = relative.as_posix()
        if name in listed:
            errors.append(f"Duplicate checksum entry: {name}")
            continue
        listed.add(name)
        path = root / relative
        if not path.is_file():
            errors.append(f"Checksum file is missing: {name}")
            continue
        actual = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                actual.update(chunk)
        if actual.hexdigest() != digest.lower():
            errors.append(f"Checksum mismatch: {name}")
    actual_files = {
        path.relative_to(root).as_posix()
        for directory in REQUIRED_DIRECTORIES
        for path in (root / directory).rglob("*")
        if path.is_file()
        and "__pycache__" not in path.relative_to(root).parts
        and path.suffix not in (".pyc", ".pyo")
    }
    for name in sorted(actual_files - listed):
        errors.append(f"Dataset file is absent from checksums.sha256: {name}")
    return len(listed)


def validate_dataset(
    data_root: Path,
    *,
    expected_splits: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Validate file structure, JSON, task rosters and catalog references.

    ``expected_splits`` allows small offline fixtures; the CLI always checks
    the released 1191 training tasks and 397 test tasks. This is a static
    integrity check, not a replay of the reference trajectories.
    """
    root = Path(data_root).resolve()
    expected = dict(EXPECTED_SPLITS if expected_splits is None else expected_splits)
    errors: list[str] = []
    report: dict[str, Any] = {
        "data_root": str(root),
        "valid": False,
        "json_files": 0,
        "parsed_json_files": 0,
        "task_counts": {split: 0 for split in expected},
        "unique_tasks": 0,
        "domains": 0,
        "catalogs": 0,
        "checksum_files": None,
        "versions": list(VERSIONS),
        "errors": errors,
    }
    if not root.is_dir():
        errors.append(f"Dataset directory is missing: {root}. {DATA_HELP}")
        return report

    def require_file(path: Path) -> None:
        if not path.is_file() or path.stat().st_size == 0:
            errors.append(f"Missing or empty file: {path.relative_to(root)}")

    for name in REQUIRED_DIRECTORIES:
        if not (root / name).is_dir():
            errors.append(f"Missing dataset directory: {name}/. {DATA_HELP}")
    for name in ("database/entities", "database/relationships"):
        if not (root / name).is_dir():
            errors.append(f"Missing dataset directory: {name}/")
        elif not any((root / name).rglob("*.json")):
            errors.append(f"No JSON database records found under {name}/")

    json_files = sorted(root.rglob("*.json"))
    report["json_files"] = len(json_files)
    if not json_files:
        errors.append(f"The dataset contains no JSON files. {DATA_HELP}")
        return report

    tasks: list[tuple[Path, str, dict[str, Any]]] = []
    seen_ids: set[str] = set()
    catalogs: dict[tuple[str, str], set[str]] = {}
    mappings: dict[str, dict[str, set[str]]] = {}
    attributes: dict[tuple[str, str], set[str]] = {}
    for path in json_files:
        relative = path.relative_to(root)
        parts = relative.parts
        if len(parts) == 4 and parts[0] == "tasks" and parts[1] in expected:
            report["task_counts"][parts[1]] += 1
        try:
            with path.open(encoding="utf-8") as handle:
                document = json.load(handle, parse_constant=_reject_constant)
        except (OSError, UnicodeError, ValueError) as exc:
            errors.append(f"Invalid JSON at {relative}: {exc}")
            continue
        report["parsed_json_files"] += 1

        if parts[0] == "tasks":
            if len(parts) != 4 or parts[1] not in expected:
                errors.append(f"Unexpected task location: {relative}")
                continue
            if not isinstance(document, dict):
                errors.append(f"Task must be a JSON object: {relative}")
                continue
            split, domain = parts[1:3]
            query_id = document.get("query_id")
            if not isinstance(query_id, str) or query_id != path.stem:
                errors.append(f"Task query_id does not match its filename: {relative}")
            elif query_id in seen_ids:
                errors.append(f"Duplicate task query_id {query_id!r}: {relative}")
            else:
                seen_ids.add(query_id)
            if document.get("domain") != domain or document.get("split") != split:
                errors.append(f"Task domain/split does not match its location: {relative}")
            trajectories = document.get("version_trajectories")
            if not isinstance(trajectories, dict):
                errors.append(f"Task has no version_trajectories object: {relative}")
                trajectories = {}
            tasks.append((relative, domain, trajectories))
        elif len(parts) == 3 and parts[0] == "tools" and path.stem.endswith("_tools"):
            domain, version = parts[1], path.stem[:-6]
            names: set[str] = set()
            if not isinstance(document, list) or not document:
                errors.append(f"Tool catalog must be a nonempty list: {relative}")
            else:
                for index, row in enumerate(document):
                    tool = row.get("tool") if isinstance(row, dict) else None
                    function = tool.get("function") if isinstance(tool, dict) else None
                    name = function.get("name") if isinstance(function, dict) else None
                    if not isinstance(name, str) or not name:
                        errors.append(f"Tool catalog row {index} has no function name: {relative}")
                    elif name in names:
                        errors.append(f"Duplicate tool {name!r} in {relative}")
                    else:
                        names.add(name)
            catalogs[(domain, version)] = names
        elif len(parts) == 3 and parts[:2] == ("meta", "version_mapping"):
            versions = document.get("versions") if isinstance(document, dict) else None
            if not isinstance(versions, dict):
                errors.append(f"Version mapping must contain a versions object: {relative}")
                continue
            mapped: dict[str, set[str]] = {}
            for version, entry in versions.items():
                rows = entry.get("tools") if isinstance(entry, dict) else None
                if not isinstance(rows, list) or any(
                    not isinstance(row, dict) or not isinstance(row.get("tool_name"), str)
                    for row in rows
                ):
                    errors.append(f"Invalid tool list for {version} in {relative}")
                    continue
                mapped[version] = {row["tool_name"] for row in rows}
            mappings[path.stem] = mapped
        elif len(parts) == 4 and parts[:2] == ("meta", "tool_attributes"):
            if not isinstance(document, dict):
                errors.append(f"Tool attributes must be a JSON object: {relative}")
            else:
                attributes[(parts[2], path.stem)] = set(document)

    report["unique_tasks"] = len(seen_ids)
    for split, count in expected.items():
        actual = report["task_counts"][split]
        if actual != count:
            errors.append(f"Task count for {split}: expected {count}, found {actual}")

    domains = sorted({domain for domain, _ in catalogs} | {domain for _, domain, _ in tasks})
    report["domains"] = len(domains)
    report["catalogs"] = len(catalogs)
    if not domains:
        errors.append("No domain tool catalogs were found under tools/.")
    for domain in domains:
        for path in (
            root / "meta/version_mapping" / f"{domain}.json",
            root / "meta/evolution_forests" / f"{domain}_forest.json",
            root / "meta/eval_field_meta" / f"{domain}.json",
        ):
            require_file(path)
        for version in VERSIONS:
            key = (domain, version)
            require_file(root / "tools" / domain / f"{version}_tools.json")
            server_dir = root / "servers" if version == "base" else root / "servers" / version
            require_file(server_dir / f"{domain}.py")
            require_file(root / "meta/tool_attributes" / domain / f"{version}.json")
            names = catalogs.get(key)
            if names is not None:
                if mappings.get(domain, {}).get(version) != names:
                    errors.append(f"Version mapping/catalog mismatch for {domain}/{version}")
                if attributes.get(key) != names:
                    errors.append(f"Tool attributes/catalog mismatch for {domain}/{version}")
    for relative, domain, trajectories in tasks:
        for version in VERSIONS:
            names = trajectories.get(version)
            if not isinstance(names, list) or not names or not all(isinstance(n, str) and n for n in names):
                errors.append(f"Missing or invalid {version} gold trajectory: {relative}")
                continue
            unknown = sorted(set(names) - catalogs.get((domain, version), set()))
            if unknown:
                errors.append(f"Gold tools absent from {domain}/{version} catalog at {relative}: {unknown}")
    for name in ("tool_costs.json", "eval_identity_schema.json", "replay_identity_schema.json", "state_compatibility_schema.json"):
        require_file(root / "meta" / name)
    report["checksum_files"] = _verify_checksums(root, errors)
    report["valid"] = not errors
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", help="Dataset root (overrides DTAU_BENCH_DATA_ROOT)")
    parser.add_argument("--json", action="store_true", help="Print a JSON validation report")
    args = parser.parse_args()
    try:
        report = validate_dataset(get_data_root(args.data_dir))
    except (OSError, ValueError) as exc:
        print(f"Cannot inspect dataset: {exc}\n{DATA_HELP}", file=sys.stderr)
        raise SystemExit(1) from exc
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(f"Dataset: {report['data_root']}")
        print(f"JSON: {report['parsed_json_files']}/{report['json_files']} parsed")
        print("Tasks: " + ", ".join(f"{split}={count}" for split, count in report["task_counts"].items()))
        print(f"Catalogs: {report['catalogs']} across {report['domains']} domains")
        if report["checksum_files"] is not None:
            print(f"Checksums: {report['checksum_files']} dataset files checked")
        if report["valid"]:
            print("Dataset validation passed (no model or API calls).")
        else:
            for error in report["errors"][:30]:
                print(f"[ERROR] {error}", file=sys.stderr)
            if len(report["errors"]) > 30:
                print(f"... {len(report['errors']) - 30} additional errors; use --json to see all.", file=sys.stderr)
    raise SystemExit(0 if report["valid"] else 1)


if __name__ == "__main__":
    main()
