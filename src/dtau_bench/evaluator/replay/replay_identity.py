"""Tool-scoped identity transport for replay, independent of final-state scoring.

Only a paired, successful server-generated creation may bind an original ID.
Bindings rewrite declared references; they never infer the intended object.
"""

from __future__ import annotations

import json
import re
import uuid
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dtau_bench.evaluator.replay.executor import DatabaseState


class ReplayIdentityError(ValueError):
    """The recorded operation cannot be faithfully transported to this replay."""


_MISSING = object()


def _valid_path(path: Any, *, wildcard: bool = False) -> bool:
    return (isinstance(path, list) and bool(path)
            and all(isinstance(p, str) and bool(p) and (wildcard or p != "*") for p in path))


def validate_replay_identity_schema(schema: Any) -> Dict[str, Any]:
    if not isinstance(schema, dict) or type(schema.get("schema_version")) is not int or schema["schema_version"] != 1:
        raise ReplayIdentityError("Unsupported replay identity schema")
    domains = schema.get("domains")
    if not isinstance(domains, dict):
        raise ReplayIdentityError("Replay identity schema requires domains")
    for domain, versions in domains.items():
        if not isinstance(domain, str) or not domain or not isinstance(versions, dict):
            raise ReplayIdentityError("Invalid replay domain")
        for version, tools in versions.items():
            if version not in {"base", "v1", "v2", "v3", "v4", "v5", "v6"} or not isinstance(tools, dict):
                raise ReplayIdentityError(f"Invalid replay version: {domain}/{version}")
            keys: Dict[Tuple[str, str], str] = {}
            for name, contract in tools.items():
                if not isinstance(name, str) or not name or not isinstance(contract, dict):
                    raise ReplayIdentityError("Invalid replay tool contract")
                if set(contract) - {"produces", "references", "success_field", "session_token"}:
                    raise ReplayIdentityError(f"Unknown contract fields: {domain}/{version}/{name}")
                if "success_field" in contract and (
                        not isinstance(contract["success_field"], str)
                        or contract["success_field"] not in {"success", "permitted"}
                        or not contract.get("produces")):
                    raise ReplayIdentityError("Invalid producer success field")
                owned = set()
                for section in ("produces", "references"):
                    rules = contract.get(section, [])
                    if not isinstance(rules, list):
                        raise ReplayIdentityError("Identity rules must be lists")
                    for rule in rules:
                        if (not isinstance(rule, dict) or rule.get("kind") not in {"entity", "relationship"}
                                or not isinstance(rule.get("table"), str) or not rule["table"]):
                            raise ReplayIdentityError("Invalid identity rule type/table")
                        if section == "references":
                            if set(rule) - {"kind", "table", "argument_path", "codec"} or not _valid_path(rule.get("argument_path"), wildcard=True):
                                raise ReplayIdentityError("Invalid reference path")
                            codec = rule.get("codec")
                            if codec is not None and (not isinstance(codec, dict)
                                    or set(codec) - {"prefix", "uuid_format"}
                                    or not isinstance(codec.get("prefix"), str) or not codec["prefix"]
                                    or codec.get("uuid_format") not in {None, "preserve"}):
                                raise ReplayIdentityError("Invalid identity reference codec")
                            path = tuple(rule["argument_path"])
                            if path in owned:
                                raise ReplayIdentityError("Conflicting reference path ownership")
                            owned.add(path)
                            continue
                        if set(rule) - {"kind", "table", "primary_key", "result_path", "generation", "table_optional_before"}:
                            raise ReplayIdentityError("Unknown producer fields")
                        if not isinstance(rule.get("primary_key"), str) or not rule["primary_key"] or not _valid_path(rule.get("result_path")):
                            raise ReplayIdentityError("Invalid producer primary key/result path")
                        if type(rule.get("table_optional_before", False)) is not bool:
                            raise ReplayIdentityError("Invalid optional-table flag")
                        table = (rule["kind"], rule["table"])
                        if table in keys and keys[table] != rule["primary_key"]:
                            raise ReplayIdentityError("Conflicting primary keys")
                        keys[table] = rule["primary_key"]
                        generation = rule.get("generation")
                        if not isinstance(generation, dict):
                            raise ReplayIdentityError("Missing generation provenance")
                        if generation.get("mode") == "always":
                            if set(generation) != {"mode"}:
                                raise ReplayIdentityError("Invalid unconditional generation")
                        elif generation.get("mode") == "when_absent":
                            if (set(generation) - {"mode", "argument_path", "absent_values"}
                                    or not _valid_path(generation.get("argument_path"))
                                    or not isinstance(generation.get("absent_values", [None]), list)):
                                raise ReplayIdentityError("Invalid conditional generation")
                        else:
                            raise ReplayIdentityError("Unsupported generation branch")
            _validate_session_token_contracts(tools)
    return schema


def load_replay_identity_schema(data_dir: Path) -> Dict[str, Any]:
    path = Path(data_dir) / "meta" / "replay_identity_schema.json"
    try:
        schema = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ReplayIdentityError(f"Cannot read replay identity schema: {path}") from exc
    return validate_replay_identity_schema(schema)


def get_replay_contracts(schema: Dict[str, Any], domain: str, version: str) -> Dict[str, Any]:
    versions = schema["domains"].get(domain)
    if versions is None:
        return {}
    if version not in versions:
        raise ReplayIdentityError(f"No replay identity contract for {domain}/{version}")
    return versions[version]


def action_source_errors(messages: List[Dict[str, Any]], records: Any) -> Dict[int, str]:
    """Validate before the legacy parser's response dictionary can hide duplicates.

    The engine supplies this evidence; a rollout cannot set its own verified flag.
    Only contracted steps depend on it, preserving unrelated legacy input formats.
    """
    calls: Dict[str, list] = defaultdict(list)
    responses: Dict[str, list] = defaultdict(list)
    for position, message in enumerate(messages):
        if not isinstance(message, dict):
            continue
        if message.get("role") == "assistant":
            for call in message.get("tool_calls") or []:
                identifier = call.get("id")
                if isinstance(identifier, str) and identifier:
                    calls[identifier].append(position)
        elif message.get("role") == "tool":
            identifier = message.get("tool_call_id")
            if isinstance(identifier, str) and identifier:
                responses[identifier].append((position, message))
    errors = {}
    for record in records:
        identifier = record.tool_call_id
        if not isinstance(identifier, str) or not identifier:
            errors[record.index] = "Missing tool_call_id"
        elif len(calls[identifier]) != 1 or len(responses[identifier]) != 1:
            errors[record.index] = "Missing, duplicated, or reused tool call/result association"
        else:
            position, response = responses[identifier][0]
            response_name = response.get("name")
            if position <= calls[identifier][0]:
                errors[record.index] = "Tool response precedes its call"
            elif response_name and str(response_name).split(".")[-1] != record.tool_name:
                errors[record.index] = "Tool response name disagrees with its call"
            elif response.get("content") is None:
                errors[record.index] = "Missing original tool result"
    return errors


def result_outcome(result: Any) -> str:
    """Recognize explicit protocol markers, never recursively inspect business data."""
    if not isinstance(result, dict) or not result:
        return "unknown"
    if "success" in result and type(result["success"]) is not bool:
        return "unknown"
    if result.get("error"):
        return "failure"
    if result.get("needs_confirmation") is True:
        return "preview"
    if result.get("success") is False or result.get("permitted") is False:
        return "failure"
    return "success"


def read_path(value: Any, path: List[str]) -> Any:
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return _MISSING
        value = value[key]
    return value


def _rewrite_path(source: Any, target: Any, path: List[str], replacement: Any) -> None:
    """Rewrite only present dictionary fields or explicitly declared list leaves."""
    key, *rest = path
    if key == "*":
        if not isinstance(source, list) or not isinstance(target, list):
            return
        for i, value in enumerate(source):
            if rest:
                _rewrite_path(value, target[i], rest, replacement)
            else:
                target[i] = replacement(value)
    elif isinstance(source, dict) and isinstance(target, dict) and key in source:
        if rest:
            _rewrite_path(source[key], target[key], rest, replacement)
        else:
            target[key] = replacement(source[key])


def _generated(rule: Dict[str, Any], arguments: Dict[str, Any]) -> bool:
    generation = rule["generation"]
    if generation["mode"] == "always":
        return True
    value = read_path(arguments, generation["argument_path"])
    return value is _MISSING or any(type(value) is type(v) and value == v
                                   for v in generation.get("absent_values", [None]))


def strict_database(server: Any, session_id: str) -> Dict[str, Any]:
    """Obtain the live session without converting a read failure into an empty DB."""
    try:
        if hasattr(server, "get_session"):
            session = server.get_session(session_id)
        elif hasattr(server, "_get_session"):
            session = server._get_session(session_id)
        elif hasattr(server, "sessions"):
            session = server.sessions.get(session_id)
        else:
            raise ReplayIdentityError("Server has no session access")
        if session is None and hasattr(server, "create_session"):
            session = server.create_session(session_id)
        if session is None:
            raise ReplayIdentityError("Replay session is unavailable")
        for attribute in ("database", "databases", "db_copy"):
            database = getattr(session, attribute, None)
            if database is not None:
                if not isinstance(database, dict):
                    raise ReplayIdentityError("Session database is not a dictionary")
                return database
        raise ReplayIdentityError("Session has no database")
    except ReplayIdentityError:
        raise
    except Exception as exc:
        raise ReplayIdentityError(f"Cannot read replay database: {exc}") from exc


def capture_replay_state(server: Any, session_id: str) -> DatabaseState:
    """Use the same table classification as gold capture, but propagate read errors.

    Capture the database returned by the strict read itself; a separate successful
    probe cannot establish that a subsequent best-effort capture succeeded.
    """
    database = strict_database(server, session_id)
    entities, relationships = {}, {}
    try:
        for table, rows in database.items():
            if not isinstance(rows, list) or not rows or not isinstance(rows[0], dict):
                continue
            is_entity = any(key.endswith("_id") and key.replace("_id", "").lower() == table.lower()
                            for key in rows[0])
            (entities if is_entity else relationships)[table] = deepcopy(rows)
    except Exception as exc:
        raise ReplayIdentityError(f"Cannot capture replay state: {exc}") from exc
    return DatabaseState(entities=entities, relationships=relationships)


def _table_ids(database: Dict[str, Any], rule: Dict[str, Any], *, before: bool = False) -> set:
    table, key = rule["table"], rule["primary_key"]
    if table not in database:
        if before and rule.get("table_optional_before", False):
            return set()
        raise ReplayIdentityError(f"Missing identity table: {table}")
    rows = database[table]
    if not isinstance(rows, list):
        raise ReplayIdentityError(f"Unreadable identity table: {table}")
    values = []
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get(key), str) or not row[key]:
            raise ReplayIdentityError(f"Invalid primary key: {table}.{key}")
        values.append(row[key])
    if len(set(values)) != len(values):
        raise ReplayIdentityError(f"Duplicate primary keys: {table}.{key}")
    return set(values)


def _validate_session_token_contracts(tools: Dict[str, Any]) -> None:
    """Validate token ownership separately from persistent entity identities."""
    producers, stores = {}, set()
    for name, contract in tools.items():
        if "session_token" not in contract:
            continue
        token = contract["session_token"]
        if (not isinstance(token, dict) or not isinstance(token.get("role"), str)
                or token["role"] not in {"producer", "consumer"}):
            raise ReplayIdentityError("Invalid session token role")
        if not isinstance(token.get("family"), str) or not token["family"]:
            raise ReplayIdentityError("Invalid session token family")
        if contract.get("produces"):
            raise ReplayIdentityError("Session token and persistent ID producers cannot share a call")
        if token["role"] == "producer":
            if set(token) != {"role", "family", "store", "result_path", "bound_arguments"}:
                raise ReplayIdentityError("Unknown or missing session token producer fields")
            if (not isinstance(token["store"], str) or not token["store"]
                    or not _valid_path(token["result_path"])):
                raise ReplayIdentityError("Invalid session token store/result path")
            arguments = token["bound_arguments"]
            if (not isinstance(arguments, dict) or not arguments
                    or any(not isinstance(key, str) or not key or not _valid_path(path)
                           for key, path in arguments.items())):
                raise ReplayIdentityError("Invalid session token bound arguments")
            if len({tuple(path) for path in arguments.values()}) != len(arguments):
                raise ReplayIdentityError("Duplicate session token parameter ownership")
            if token["family"] in producers or token["store"] in stores:
                raise ReplayIdentityError("Conflicting session token producer/store ownership")
            producers[token["family"]] = name
            stores.add(token["store"])
        else:
            if set(token) != {"role", "family", "producer", "argument_path"}:
                raise ReplayIdentityError("Unknown or missing session token consumer fields")
            if (not isinstance(token["producer"], str) or not token["producer"]
                    or not _valid_path(token["argument_path"])):
                raise ReplayIdentityError("Invalid session token producer/reference path")
            path = token["argument_path"]
            for reference in contract.get("references", []):
                other = reference["argument_path"]
                if all(a == b or a == "*" or b == "*" for a, b in zip(path, other)):
                    raise ReplayIdentityError("Conflicting token/identity reference paths")
    for contract in tools.values():
        token = contract.get("session_token", {})
        if token.get("role") == "consumer" and producers.get(token["family"]) != token["producer"]:
            raise ReplayIdentityError("Session token consumer has no matching producer")


class _PreparedTokenCall:
    """Opaque preparation passed through the existing agent replay interface."""

    def __init__(self, identities: list, token: Dict[str, Any]):
        self.identities, self.token = identities, token


class _SessionTokenTracker:
    """Observe dictionary-backed, single-use tokens without changing tool state."""

    def __init__(self, owner: Any, tool_names: List[str]):
        self.owner = owner
        self.rules = {name: contract["session_token"] for name, contract in owner.contracts.items()
                      if "session_token" in contract}
        self.producers = {rule["family"]: rule for rule in self.rules.values()
                          if rule["role"] == "producer"}
        self.records: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
        self.initial = {}
        active = {self.rules[name]["family"] for name in set(tool_names) if name in self.rules}
        for family in active:
            self.initial[family] = set(self._snapshot(family) or {})

    def _snapshot(self, family: str) -> Optional[Dict[str, Any]]:
        store = self.producers[family]["store"]
        database = strict_database(self.owner.server, self.owner.session_id)
        if store not in database:
            return None
        value = database[store]
        if (not isinstance(value, dict)
                or any(not isinstance(key, str) or not key or not isinstance(item, dict)
                       for key, item in value.items())):
            raise ReplayIdentityError(f"Unreadable session token store: {store}")
        return deepcopy(value)

    def map_arguments(self, name: str, original: Dict[str, Any], mapped: Dict[str, Any]) -> None:
        rule = self.rules.get(name)
        if rule is None or rule["role"] != "consumer":
            return

        def replace(value: Any) -> Any:
            if not isinstance(value, str):
                return deepcopy(value)
            return self.owner.bindings.get(("session_token", rule["family"], value), value)

        _rewrite_path(original, mapped, rule["argument_path"], replace)

    def before_call(self, name: str, original: Dict[str, Any], mapped: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        rule = self.rules.get(name)
        if rule is None:
            return None
        return {"rule": rule, "before": self._snapshot(rule["family"]),
                "original": deepcopy(original), "mapped": deepcopy(mapped)}

    def _expected_parameters(self, family: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        result = {}
        for field, path in self.producers[family]["bound_arguments"].items():
            value = read_path(arguments, path)
            if value is _MISSING:
                raise ReplayIdentityError(f"Missing witnessed token parameter: {field}")
            result[field] = value
        return result

    def after_call(self, name: str, index: int, call_id: Optional[str], prepared: Dict[str, Any],
                   original_result: Dict[str, Any], replay_result: Dict[str, Any], outcome: str) -> None:
        rule, before = prepared["rule"], prepared["before"]
        family = rule["family"]
        after = self._snapshot(family)
        if outcome != "success":
            if after != before:
                raise ReplayIdentityError("Failed or previewed call changed the session token store")
            self._record_use(prepared, index, call_id, outcome)
            return
        if original_result.get("success") is not True or replay_result.get("success") is not True:
            raise ReplayIdentityError("Session tokens require explicit original and replay success")
        if not isinstance(call_id, str) or not call_id:
            raise ReplayIdentityError("Session token operation has no source call ID")
        if rule["role"] == "producer":
            old = read_path(original_result, rule["result_path"])
            new = read_path(replay_result, rule["result_path"])
            if not isinstance(old, str) or not old or not isinstance(new, str) or not new:
                raise ReplayIdentityError("Missing single generated session token")
            expected = dict(before or {})
            if new in expected:
                raise ReplayIdentityError("Returned token was not newly generated")
            expected[new] = self._expected_parameters(family, prepared["mapped"])
            if after != expected:
                raise ReplayIdentityError("Token generation lacks an exact store/parameter witness")
            key, inverse_key = ("session_token", family, old), ("session_token", family, new)
            if old in self.initial[family] or new in self.initial[family]:
                raise ReplayIdentityError("Generated session token collides with initial state")
            if key in self.owner.bindings or inverse_key in self.owner.inverse:
                raise ReplayIdentityError("Session token identity reused or ambiguously bound")
            record = {"domain": self.owner.domain, "kind": "session_token", "family": family,
                      "table": rule["store"], "original_id": old, "replay_id": new,
                      "step_index": index, "tool_call_id": call_id, "tool_name": name,
                      "result_path": list(rule["result_path"]), "consumed": False, "uses": []}
            self.owner.bindings[key], self.owner.inverse[inverse_key] = new, old
            self.records[key] = record
            self.owner.provenance.append(record)
            return
        old = read_path(prepared["original"], rule["argument_path"])
        new = read_path(prepared["mapped"], rule["argument_path"])
        if not isinstance(old, str) or not isinstance(new, str):
            raise ReplayIdentityError("Successful consumption has no token reference")
        key = ("session_token", family, old)
        record = self.records.get(key)
        if record is None or self.owner.bindings.get(key) != new or record["consumed"]:
            raise ReplayIdentityError("Successful consumption has no live witnessed producer")
        if before is None or new not in before:
            raise ReplayIdentityError("Consumed session token was not present before the call")
        if before[new] != self._expected_parameters(family, prepared["mapped"]):
            raise ReplayIdentityError("Consumed token disagrees with the applied parameters")
        expected = dict(before)
        del expected[new]
        if after != expected:
            raise ReplayIdentityError("Successful application did not consume exactly its session token")
        record["consumed"] = True
        self._record_use(prepared, index, call_id, outcome)

    def _record_use(self, prepared: Dict[str, Any], index: int, call_id: Optional[str], outcome: str) -> None:
        rule = prepared["rule"]
        if rule["role"] != "consumer":
            return
        value = read_path(prepared["original"], rule["argument_path"])
        record = self.records.get(("session_token", rule["family"], value)) if isinstance(value, str) else None
        if record is not None:
            record["uses"].append({"step_index": index, "tool_call_id": call_id, "outcome": outcome})


class ReplayIdentityTracker:
    def __init__(self, server: Any, session_id: str, domain: str,
                 contracts: Dict[str, Any], tool_names: List[str]):
        self.server, self.session_id, self.domain = server, session_id, domain
        self.contracts = contracts
        self.bindings: Dict[Tuple[str, str, str], str] = {}
        self.inverse: Dict[Tuple[str, str, str], str] = {}
        self.provenance: List[Dict[str, Any]] = []
        self.initial: Dict[Tuple[str, str], set] = {}
        if any("session_token" in contracts.get(name, {}) for name in tool_names):
            _validate_session_token_contracts(contracts)
            self.tokens = _SessionTokenTracker(self, tool_names)
        else:
            self.tokens = None
        rules = [rule for name in set(tool_names) for rule in contracts.get(name, {}).get("produces", [])]
        if rules:
            database = strict_database(server, session_id)
            for rule in rules:
                self.initial[(rule["kind"], rule["table"])] = _table_ids(database, rule, before=True)

    def map_arguments(self, name: str, original: Dict[str, Any], legacy_mapped: Dict[str, Any]) -> Dict[str, Any]:
        # An audited tool may only rewrite explicitly declared reference leaves.
        # Old untyped bindings must not leak into its business text or other data.
        mapped = deepcopy(original if name in self.contracts else legacy_mapped)
        contract = self.contracts.get(name, {})
        for rule in contract.get("references", []):
            def replace(value: Any, rule: Dict[str, Any] = rule) -> Any:
                if not isinstance(value, str):
                    return deepcopy(value)
                codec = rule.get("codec")
                if codec:
                    prefix = codec["prefix"]
                    if not value.startswith(prefix):
                        return value
                    suffix = value[len(prefix):]
                    original_id = suffix
                    if codec.get("uuid_format") == "preserve":
                        if not re.fullmatch(r"(?:[0-9a-fA-F]{32}|[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12})", suffix):
                            return value
                        original_id = str(uuid.UUID(suffix))
                    new_id = self.bindings.get((rule["kind"], rule["table"], original_id))
                    if new_id is None:
                        return value
                    if codec.get("uuid_format") == "preserve":
                        try:
                            parsed = uuid.UUID(new_id)
                        except ValueError as exc:
                            raise ReplayIdentityError("Bound identity cannot preserve its declared UUID encoding") from exc
                        new_id = parsed.hex if len(suffix) == 32 else str(parsed)
                        if suffix.upper() == suffix:
                            new_id = new_id.upper()
                        elif suffix.lower() != suffix:
                            new_id = "".join(char.upper() if old.isupper() else char
                                             for char, old in zip(new_id, suffix))
                    return prefix + new_id
                return self.bindings.get((rule["kind"], rule["table"], value), value)
            # Restore original values at owned paths even when no binding exists.
            _rewrite_path(original, mapped, rule["argument_path"], replace)
        for rule in contract.get("produces", []):
            path = rule["generation"].get("argument_path")
            if path:
                _rewrite_path(original, mapped, path, deepcopy)
        if self.tokens is not None:
            self.tokens.map_arguments(name, original, mapped)
        return mapped

    def before_call(self, name: str, original: Dict[str, Any], mapped: Dict[str, Any]) -> Any:
        producers = self.contracts.get(name, {}).get("produces", [])
        prepared = []
        database = strict_database(self.server, self.session_id) if producers else None
        for rule in producers:
            enabled = _generated(rule, original)
            if enabled != _generated(rule, mapped):
                raise ReplayIdentityError("Original and replay generation branches differ")
            prepared.append((rule, enabled, _table_ids(database, rule, before=True)))
        if self.tokens is not None:
            token = self.tokens.before_call(name, original, mapped)
            if token is not None:
                return _PreparedTokenCall(prepared, token)
        return prepared

    def after_call(self, name: str, index: int, call_id: Optional[str], prepared: Any,
                   original_result: Dict[str, Any], replay_result: Dict[str, Any]) -> None:
        original_outcome, replay_outcome = result_outcome(original_result), result_outcome(replay_result)
        if original_outcome == "unknown" or replay_outcome == "unknown":
            raise ReplayIdentityError("Unrecognized original or replay tool result")
        if original_outcome != replay_outcome:
            raise ReplayIdentityError(f"Tool outcome diverged: {original_outcome} -> {replay_outcome}")
        if isinstance(prepared, _PreparedTokenCall):
            self.tokens.after_call(name, index, call_id, prepared.token,
                                   original_result, replay_result, original_outcome)
            prepared = prepared.identities
        if original_outcome != "success":
            return
        success_field = self.contracts.get(name, {}).get("success_field", "success")
        if any(enabled for _, enabled, _ in prepared) and (
                original_result.get(success_field) is not True or replay_result.get(success_field) is not True):
            raise ReplayIdentityError("Generated identity requires the audited explicit success protocol")
        pending = []
        database = strict_database(self.server, self.session_id) if prepared else None
        for rule, enabled, before in prepared:
            if not enabled:
                continue
            old_id = read_path(original_result, rule["result_path"])
            new_id = read_path(replay_result, rule["result_path"])
            if not isinstance(old_id, str) or not old_id or not isinstance(new_id, str) or not new_id:
                raise ReplayIdentityError(f"Missing single generated ID at {rule['result_path']}")
            table = (rule["kind"], rule["table"])
            after = _table_ids(database, rule)
            if new_id not in after - before:
                raise ReplayIdentityError(f"Returned ID was not newly persisted in {rule['table']}")
            if old_id in self.initial[table] or new_id in self.initial[table]:
                raise ReplayIdentityError("Generated identity collides with an initial object")
            key, inverse_key = (*table, old_id), (*table, new_id)
            if key in self.bindings:
                raise ReplayIdentityError("Original identity reused or bound to multiple replay objects")
            if inverse_key in self.inverse:
                raise ReplayIdentityError("Replay identity reused or bound to multiple original objects")
            pending.append((key, inverse_key, old_id, new_id, rule))
        # Validate the entire call atomically; don't leave a partial identity map.
        if len({p[0] for p in pending}) != len(pending) or len({p[1] for p in pending}) != len(pending):
            raise ReplayIdentityError("Ambiguous identities in one creation result")
        for key, inverse_key, old_id, new_id, rule in pending:
            self.bindings[key], self.inverse[inverse_key] = new_id, old_id
            self.provenance.append({"domain": self.domain, "kind": rule["kind"], "table": rule["table"],
                                    "original_id": old_id, "replay_id": new_id, "step_index": index,
                                    "tool_call_id": call_id, "tool_name": name,
                                    "result_path": list(rule["result_path"])})

    def legacy_result_ids(self, name: str, values: Dict[str, str]) -> Dict[str, str]:
        """Typed identities cannot leak back into the old untyped global map."""
        owned = {rule["result_path"][0] for rule in self.contracts.get(name, {}).get("produces", [])
                 if len(rule["result_path"]) == 1}
        typed_values = {key[2] for key in self.bindings} | set(self.bindings.values())
        return {field: value for field, value in values.items() if field not in owned and value not in typed_values}
