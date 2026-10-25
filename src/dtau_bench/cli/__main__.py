"""Command-line entry point: ``python -m dtau_bench <subcommand>``."""

from __future__ import annotations

import sys
from typing import Callable, Dict

SUBCOMMANDS: Dict[str, str] = {
    "chain": "Run the full version chain (train and/or test phases)",
    "rollout": "Run one rollout (train or test) for a single version",
    "evaluate": "Evaluate one rollout run",
    "evaluate-multi": "Aggregate evaluation results across versions",
    "bootstrap": "Build the first skill library from trajectories",
    "evolve": "Evolve a skill library to the next version",
    "build-tool-index": "Build the tool retrieval index for one version",
    "build-skill-index": "Build a skill library's retrieval index",
    "validate-data": "Validate the local dataset without model or API calls",
}


def _load(subcommand: str) -> Callable[[], None]:
    module_name = subcommand.replace("-", "_")
    module = __import__(f"dtau_bench.cli.{module_name}", fromlist=["main"])
    return module.main


def _usage() -> str:
    width = max(len(name) for name in SUBCOMMANDS)
    lines = ["usage: python -m dtau_bench <subcommand> [options]", "", "subcommands:"]
    lines += [f"  {name:<{width}}  {desc}" for name, desc in SUBCOMMANDS.items()]
    lines += ["", "Run 'python -m dtau_bench <subcommand> --help' for details."]
    return "\n".join(lines)


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(_usage())
        raise SystemExit(0 if len(sys.argv) >= 2 else 1)

    subcommand = sys.argv[1]
    if subcommand not in SUBCOMMANDS:
        print(f"unknown subcommand: {subcommand!r}\n", file=sys.stderr)
        print(_usage(), file=sys.stderr)
        raise SystemExit(2)

    # Let the subcommand's own parser see argv without the subcommand name.
    sys.argv = [f"{sys.argv[0]} {subcommand}"] + sys.argv[2:]
    _load(subcommand)()


if __name__ == "__main__":
    main()
