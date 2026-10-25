"""Replay recorded agent actions, transporting audited generated identities.

The action sequence and object choices are preserved. Final business correctness
is decided separately by the task's evaluation strategy.
"""

from __future__ import annotations

import json
import uuid
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from dtau_bench.evaluator.replay.executor import DatabaseState, capture_database_state
from dtau_bench.evaluator.replay.replay_identity import (
    ReplayIdentityError, ReplayIdentityTracker, capture_replay_state, result_outcome,
)

from dtau_bench.evaluator.replay.state_compatibility import (
    StateCompatibilityError, StateCompatibilityObserver,
)


@dataclass
class AgentReplayResult:
    initial_state: Optional[DatabaseState] = None
    final_state: Optional[DatabaseState] = None
    replay_errors: List[str] = field(default_factory=list)
    id_mapping_size: int = 0
    steps_attempted: int = 0
    steps_succeeded: int = 0
    steps_total: int = 0
    steps_failed: int = 0
    steps_previewed: int = 0
    steps_unknown: int = 0
    replay_complete: bool = True
    tool_failures: List[Dict[str, Any]] = field(default_factory=list)
    replay_integrity_errors: List[Dict[str, Any]] = field(default_factory=list)
    identity_bindings: List[Dict[str, Any]] = field(default_factory=list)
    state_compatibility_evidence: Optional[Dict[str, Any]] = None

    @property
    def success_rate(self) -> float:
        return self.steps_succeeded / self.steps_attempted if self.steps_attempted else 0.0


def _validate_agent_step(step: Dict[str, Any], step_index: int) -> None:
    if not isinstance(step, dict):
        raise ValueError(f"agent_traj[{step_index}] must be a dict")
    tool_name = step.get("tool_name")
    if not isinstance(tool_name, str) or not tool_name.strip():
        raise ValueError(f"agent_traj[{step_index}] missing tool_name")
    if not isinstance(step.get("arguments", {}), dict):
        raise ValueError(f"agent_traj[{step_index}] arguments must be a dict")


def _parse_result_payload(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _apply_id_mapping(arguments: Dict[str, Any], id_mapping: Dict[str, str]) -> Dict[str, Any]:
    """Compatibility transport for unaudited legacy relationship/session IDs."""
    return {key: id_mapping.get(value, value) if isinstance(value, str) else deepcopy(value)
            for key, value in arguments.items()}


def _extract_relationship_ids(result: Dict[str, Any], entity_id_names: Set[str]) -> Dict[str, str]:
    """Legacy heuristic; audited identities are handled by ReplayIdentityTracker."""
    return {key: value for key, value in result.items()
            if isinstance(value, str) and key.endswith("_id") and key.lower() not in entity_id_names}


class AgentReplayExecutor:
    """Re-execute recorded actions in one fresh session.

    The evaluation engine supplies tool contracts and source-verified steps.
    Direct callers without contracts retain the legacy ID transport behavior.
    """

    def __init__(self, server: Any, *, domain: str = "", contracts: Optional[Dict[str, Any]] = None,
                 state_contracts: Optional[Dict[str, Any]] = None):
        self.server = server
        self.domain = domain
        self.contracts = contracts or {}
        self.state_contracts = state_contracts or {}

    def execute(self, agent_steps: List[Dict[str, Any]], entity_id_names: Set[str],
                session_id: Optional[str] = None) -> AgentReplayResult:
        session_id = session_id or f"agent_{uuid.uuid4().hex[:8]}"
        for index, step in enumerate(agent_steps):
            _validate_agent_step(step, index)
        result = AgentReplayResult(steps_total=len(agent_steps))
        names = [s["tool_name"].split(".")[-1] for s in agent_steps]
        protected = any(name in self.contracts for name in names)
        legacy_mapping: Dict[str, str] = {}
        observer = StateCompatibilityObserver(self.state_contracts) if self.state_contracts else None

        def integrity(index: int, name: str, error: str) -> None:
            result.replay_complete = False
            result.replay_integrity_errors.append({"step_index": index, "tool_name": name, "error": error})
            result.replay_errors.append(f"{name}: replay integrity: {error}")

        try:
            capture = capture_replay_state if protected else capture_database_state
            result.initial_state = capture(self.server, session_id)
            result.final_state = result.initial_state
            tracker = ReplayIdentityTracker(self.server, session_id, self.domain, self.contracts, names)
            if observer is not None:
                observer.start(self.server, session_id)
        except (ReplayIdentityError, StateCompatibilityError) as exc:
            integrity(-1, "initial_state", str(exc))
            return result

        for index, (step, name) in enumerate(zip(agent_steps, names)):
            original_arguments = step.get("arguments", {})
            original_result = _parse_result_payload(step.get("tool_result"))
            if not original_result:
                original_result = _parse_result_payload(step.get("tool_result_raw"))
            audited = name in self.contracts
            if audited and step.get("source_verified") is not True:
                integrity(index, name, step.get("source_error") or "Original call/result association is unverified")
                break
            try:
                mapped = tracker.map_arguments(name, original_arguments,
                                               _apply_id_mapping(original_arguments, legacy_mapping))
                prepared = tracker.before_call(name, original_arguments, mapped)
            except ReplayIdentityError as exc:
                integrity(index, name, str(exc))
                break
            result.steps_attempted += 1
            observed_arguments = deepcopy(mapped) if observer is not None else None
            try:
                replay_payload = self._invoke_tool(session_id, name, mapped)
                replay_result = _parse_result_payload(replay_payload)
            except Exception as exc:
                replay_result = {"error": f"{type(exc).__name__}: {exc}"}
            outcome = result_outcome(replay_result)
            if outcome == "success":
                result.steps_succeeded += 1
            elif outcome == "preview":
                result.steps_previewed += 1
            elif outcome == "failure":
                result.steps_failed += 1
                detail = {"step_index": index, "tool_name": name, "result": deepcopy(replay_result)}
                result.tool_failures.append(detail)
                result.replay_errors.append(f"{name}: tool failure: {replay_result}")
            else:
                result.steps_unknown += 1
            if observer is not None:
                try:
                    observer.after_call(index, name, observed_arguments, replay_result)
                except (ReplayIdentityError, StateCompatibilityError) as exc:
                    integrity(index, name, str(exc))
                    break
            if audited:
                try:
                    tracker.after_call(name, index, step.get("tool_call_id"), prepared,
                                       original_result, replay_result)
                except ReplayIdentityError as exc:
                    integrity(index, name, str(exc))
                    break
            # Do not learn identities from failures, previews, or missing results.
            if outcome == "success" and result_outcome(original_result) == "success":
                old_ids = tracker.legacy_result_ids(name, _extract_relationship_ids(original_result, entity_id_names))
                new_ids = tracker.legacy_result_ids(name, _extract_relationship_ids(replay_result, entity_id_names))
                for field_name, old_id in old_ids.items():
                    new_id = new_ids.get(field_name)
                    if new_id and new_id != old_id:
                        legacy_mapping[old_id] = new_id

        result.identity_bindings = tracker.provenance
        result.id_mapping_size = len(tracker.bindings) + len(legacy_mapping)
        try:
            result.final_state = capture(self.server, session_id)
            if observer is not None:
                result.state_compatibility_evidence = observer.finish()
        except (ReplayIdentityError, StateCompatibilityError) as exc:
            result.final_state = None
            integrity(result.steps_attempted, "final_state", str(exc))
        return result

    def _invoke_tool(self, session_id: str, tool_name: str, params: Dict[str, Any]) -> Any:
        if hasattr(self.server, "invoke"):
            return self.server.invoke(session_id=session_id, tool_name=tool_name, **params)
        if hasattr(self.server, "server"):
            return self.server.server.invoke(session_id, tool_name, **params)
        raise RuntimeError(f"Server does not support invoke: {type(self.server)}")
