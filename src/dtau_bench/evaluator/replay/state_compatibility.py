"""Evidence-backed compatibility for audited state fields.

Contracts are trusted dataset metadata. They never change replay arguments or
database snapshots, and they are independent of generated-identity strategies.
Unreviewed values/operations remain literal comparison inputs.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from functools import lru_cache
import hashlib
import json
from pathlib import Path
from typing import Any

from dtau_bench.evaluator.replay.replay_identity import strict_database, result_outcome


class StateCompatibilityError(ValueError):
    """Invalid contract or incomplete internal replay evidence."""


_VERSIONS = {"base", "v1", "v2", "v3", "v4", "v5", "v6"}
_RULE_FIELDS = {"id", "tool", "table", "primary_key", "field", "parameter",
                "value_kind", "default_value", "selector", "core_fields",
                "status_field", "status_parameter"}


def _encoded(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":"))


def _equal(left, right):
    # JSON encoding keeps bool/int, absent/null and list order distinct.
    return _encoded(left) == _encoded(right)


def _digest(value):
    return hashlib.sha256(_encoded(value).encode()).hexdigest()


def _name(value):
    return type(value) is str and value.isidentifier()


def _validate_rules(rules):
    if type(rules) is not list:
        raise StateCompatibilityError("State rules must be a list")
    seen, table_specs = set(), {}
    for rule in rules:
        if type(rule) is not dict or set(rule) != _RULE_FIELDS:
            raise StateCompatibilityError("Unknown or missing state rule fields")
        if any(not _name(rule[k]) for k in _RULE_FIELDS - {
                "selector", "core_fields", "default_value"}):
            raise StateCompatibilityError("Invalid state rule name")
        field = rule["field"]
        if field.endswith("_id") or field in {"id", rule["primary_key"]}:
            raise StateCompatibilityError("Compatibility cannot remove an identity")
        selector, core = rule["selector"], rule["core_fields"]
        if (type(selector) is not dict or not selector
                or any(not _name(k) or not _name(v) for k, v in selector.items())
                or type(core) is not list or not core
                or any(not _name(v) for v in core) or len(core) != len(set(core))
                or rule["status_field"] not in core
                or field in {*core, *selector, rule["primary_key"]}
                or not set(selector).issubset({*core, rule["primary_key"]})):
            raise StateCompatibilityError("Invalid state selector/core fields")
        formats = {"date": "%Y-%m-%d", "local_datetime": "%Y-%m-%dT%H:%M:%S"}
        fmt = formats.get(rule["value_kind"])
        if fmt is None or type(rule["default_value"]) is not str:
            raise StateCompatibilityError("Unsupported state value kind")
        try:
            parsed = datetime.strptime(rule["default_value"], fmt)
        except ValueError as exc:
            raise StateCompatibilityError("Invalid state default value") from exc
        if parsed.strftime(fmt) != rule["default_value"]:
            raise StateCompatibilityError("State default must use its exact declared format")
        key = rule["tool"], rule["table"], field
        if key in seen:
            raise StateCompatibilityError("Duplicate state rule ownership")
        seen.add(key)
        spec = rule["primary_key"], tuple(core)
        if rule["table"] in table_specs and table_specs[rule["table"]] != spec:
            raise StateCompatibilityError("Conflicting table identity/core fields")
        table_specs[rule["table"]] = spec


def validate_state_compatibility_schema(value):
    if (type(value) is not dict or set(value) != ({"schema_version", "revision", "domains", "profiles"}
                                  if value.get("schema_version") == 2 else {"schema_version", "revision", "domains"})
            or type(value["schema_version"]) is not int or value["schema_version"] not in {1, 2}
            or type(value["revision"]) is not str or not value["revision"]
            or type(value["domains"]) is not dict):
        raise StateCompatibilityError("Invalid state compatibility schema")
    ids = set()
    for domain, versions in value["domains"].items():
        if not _name(domain) or type(versions) is not dict or set(versions) != _VERSIONS:
            raise StateCompatibilityError("State domain must declare all supported versions")
        for rules in versions.values():
            _validate_rules(rules)
            for rule in rules:
                if rule["id"] in ids:
                    raise StateCompatibilityError("Duplicate state rule ID")
                ids.add(rule["id"])
    if value["schema_version"] == 2:
        if type(value["profiles"]) is not dict:
            raise StateCompatibilityError("Audited profiles must be an object")
        table_keys = {}
        for name, profile in value["profiles"].items():
            if not _name(name):
                raise StateCompatibilityError("Invalid audited profile identifier")
            _validate_profile(profile)
            table = profile["domain"], profile["table"]
            if table in table_keys and table_keys[table] != profile["primary_key"]:
                raise StateCompatibilityError("Conflicting audited table identities")
            table_keys[table] = profile["primary_key"]
    return value


@lru_cache(maxsize=8)
def _parse_schema(content: bytes):
    return validate_state_compatibility_schema(json.loads(content))


def load_state_compatibility_schema(data_dir: Path):
    path = Path(data_dir) / "meta" / "state_compatibility_schema.json"
    try:
        # Cache by exact content, never by domain alone or by a task's gold.
        return deepcopy(_parse_schema(path.read_bytes()))
    except (OSError, ValueError) as exc:
        raise StateCompatibilityError(f"Cannot load state compatibility schema: {path}: {exc}") from exc


def get_state_contracts(schema, domain, version, admission=None):
    if version not in _VERSIONS:
        raise StateCompatibilityError(f"Unsupported state version: {domain}/{version}")
    rules = schema["domains"].get(domain, {}).get(version, [])
    audited = []
    if admission is not None:
        _validate_admission(admission)
        for item in admission["rules"]:
            spec = schema.get("profiles", {}).get(item["profile"])
            if spec is None or spec["domain"] != domain:
                raise StateCompatibilityError("Unknown or wrong-domain admitted state profile")
            audited.append({"profile": item["profile"], "step_index": item["target"]["step_index"],
                            "spec": deepcopy(spec)})
    if not rules and not audited:
        return {}
    result = {"domain": domain, "version": version, "revision": schema["revision"],
              "rules": deepcopy(rules)}
    if audited:
        result.update(audited=audited, semantic_fingerprint=admission["semantic_fingerprint"])
    return result


def _context(contracts):
    fields = {"domain", "version", "revision", "rules"}
    audited = type(contracts) is dict and "audited" in contracts
    if audited:
        fields |= {"audited", "semantic_fingerprint"}
    if (type(contracts) is not dict or set(contracts) != fields
            or not _name(contracts["domain"]) or contracts["version"] not in _VERSIONS
            or type(contracts["revision"]) is not str or not contracts["revision"]):
        raise StateCompatibilityError("Invalid resolved state contracts")
    _validate_rules(contracts["rules"])
    context = {key: contracts[key] for key in ("domain", "version", "revision")}
    if audited:
        if (type(contracts["audited"]) is not list or not contracts["audited"]
                or not _fingerprint(contracts["semantic_fingerprint"])):
            raise StateCompatibilityError("Invalid resolved audited contracts")
        seen, keys = set(), {}
        for entry in contracts["audited"]:
            mirror = (type(entry) is dict and type(entry.get("spec")) is dict
                      and entry["spec"].get("kind") == "auxiliary_parameter_mirror")
            entry_fields = {"profile", "step_index", "spec", "gold_id", "gold_row_digest"}
            if mirror:
                entry_fields |= {"expected_value", "relation_bindings", "relation_row_digest"}
            if (type(entry) is not dict or set(entry) != entry_fields
                    or not _name(entry["profile"]) or type(entry["step_index"]) is not int
                    or entry["step_index"] < 0 or type(entry["gold_id"]) is not str
                    or not entry["gold_id"] or not _fingerprint("sha256:" + str(entry["gold_row_digest"]))):
                raise StateCompatibilityError("Unbound or invalid audited gold target")
            _validate_profile(entry["spec"])
            spec = entry["spec"]
            if mirror:
                allowed = {v for writers in spec["writers"].values()
                           for writer in writers for v in writer["explicit_values"]}
                bindings = entry["relation_bindings"]
                if (type(entry["expected_value"]) is not str or entry["expected_value"] not in allowed
                        or type(bindings) is not dict
                        or set(bindings) != set(spec["relation_source"]["bindings"])
                        or any(type(v) is not str or not v for v in bindings.values())
                        or not _fingerprint("sha256:" + str(entry["relation_row_digest"]))):
                    raise StateCompatibilityError("Invalid bound parameter mirror value or relation")
            if spec["domain"] != contracts["domain"]:
                raise StateCompatibilityError("Resolved audited domain mismatch")
            marker = entry["profile"], entry["step_index"]
            if marker in seen or (spec["table"] in keys and keys[spec["table"]] != spec["primary_key"]):
                raise StateCompatibilityError("Duplicate or conflicting resolved audited target")
            seen.add(marker); keys[spec["table"]] = spec["primary_key"]
        context.update(semantic_fingerprint=contracts["semantic_fingerprint"],
                       contract_digest=_digest(contracts))
    return context


def _tables(state):
    if state is None or state.entities.keys() & state.relationships.keys():
        raise StateCompatibilityError("Missing or ambiguous database snapshot")
    return {**state.entities, **state.relationships}


def _watched(tables, contracts):
    indexes = {}
    for rule in _observation_rules(contracts):
        table, key = rule["table"], rule["primary_key"]
        if table in indexes:
            continue
        rows = tables.get(table)
        if type(rows) is not list:
            raise StateCompatibilityError(f"Missing watched table: {table}")
        index = {}
        for row in rows:
            identifier = _row_identity(row, key)
            if identifier in index:
                raise StateCompatibilityError(f"Duplicate watched identity: {table}.{key}")
            index[identifier] = deepcopy(row)
        indexes[table] = index
    return indexes


class StateCompatibilityObserver:
    """Read only the contracted tables after each actual tool invocation."""

    def __init__(self, contracts):
        self.contracts = deepcopy(contracts)
        self.context = _context(contracts)
        self.events = []

    def start(self, server, session_id):
        self.server, self.session_id = server, session_id
        self.current = _watched(strict_database(server, session_id), self.contracts)
        self.initial_digest = _digest(self.current)

    def after_call(self, step_index, tool_name, arguments, result):
        current = _watched(strict_database(self.server, self.session_id), self.contracts)
        changes = []
        for table, before in self.current.items():
            after = current[table]
            for identifier in sorted(before.keys() | after.keys()):
                old, new = before.get(identifier), after.get(identifier)
                if not _equal(old, new):
                    changes.append({"table": table, "id": identifier,
                                    "before": old, "after": new})
        self.events.append({"step_index": step_index, "tool_name": tool_name,
                            "arguments": deepcopy(arguments), "outcome": result_outcome(result),
                            "success": result.get("success") is True, "changes": changes})
        if self.contracts.get("audited"):
            self.events[-1]["result"] = deepcopy(result)
            self.events[-1]["success"] = _audited_success(result)
        self.current = current

    def finish(self):
        current = _watched(strict_database(self.server, self.session_id), self.contracts)
        if not _equal(self.current, current):
            raise StateCompatibilityError("Unobserved mutation after the final tool call")
        return {"context": deepcopy(self.context), "initial_digest": self.initial_digest,
                "final_digest": _digest(current), "events": deepcopy(self.events)}


def _identity_fields(identity_schema, domain):
    fields = set()
    section = (identity_schema or {}).get("domains", {}).get(domain, {})
    for view in [section, *section.get("strategy_extensions", {}).values()]:
        for name in ("entities", "relationship_primary_keys"):
            fields.update((table, rule["primary_key"]) for table, rule in view.get(name, {}).items())
        fields.update((edge["source_table"], edge["field"]) for edge in view.get("foreign_keys", []))
        for rule in view.get("derived_fields", []):
            fields.add((rule["table"], rule["field"]))
    return fields


class PreparedStateCompatibility:
    """Validate provenance, then construct a disposable comparison state."""

    def __init__(self, *, contracts, evidence, initial_state, gold_state, agent_state,
                 agent_initial_state, identity_schema=None):
        context = _context(contracts)
        initial_tables, agent_initial_tables = _tables(initial_state), _tables(agent_initial_state)
        if (not _equal(initial_tables, agent_initial_tables)
                or not _equal(initial_state.entities, agent_initial_state.entities)
                or not _equal(initial_state.relationships, agent_initial_state.relationships)):
            raise StateCompatibilityError("Gold and agent raw initial databases differ")
        protected = _identity_fields(identity_schema, context["domain"])
        if any((r["table"], r["field"]) in protected for r in contracts["rules"]):
            raise StateCompatibilityError("State rule overlaps a declared identity/derived field")
        self.contracts, self.identity_schema = contracts, identity_schema
        self.initial = _watched(initial_tables, contracts)
        self.gold = _watched(_tables(gold_state), contracts)
        self.agent = _watched(_tables(agent_state), contracts)
        self.diagnostic = {**context, "passed": True, "applied": [], "unapplied": [],
                           "business_checks": [], "failures": []}
        if (type(evidence) is not dict or evidence.get("context") != context
                or evidence.get("initial_digest") != _digest(self.initial)
                or evidence.get("final_digest") != _digest(self.agent)
                or type(evidence.get("events")) is not list):
            raise StateCompatibilityError("Missing or mismatched actual replay evidence")
        current, writers = deepcopy(self.initial), {}
        fields = {(r["table"], r["field"]) for r in contracts["rules"]}
        for position, event in enumerate(evidence["events"]):
            if (type(event) is not dict or event.get("step_index") != position
                    or type(event.get("arguments")) is not dict
                    or type(event.get("changes")) is not list):
                raise StateCompatibilityError("Invalid replay event sequence")
            changed = set()
            for change in event["changes"]:
                table, identifier = change["table"], change["id"]
                if (table not in current or (table, identifier) in changed
                        or not _equal(current[table].get(identifier), change["before"])):
                    raise StateCompatibilityError("Replay event state chain is inconsistent")
                changed.add((table, identifier))
                before, after = change["before"], change["after"]
                key = next(r["primary_key"] for r in contracts["rules"] if r["table"] == table)
                if after is not None and (type(after) is not dict or after.get(key) != identifier):
                    raise StateCompatibilityError("Replay event changed a literal row identity")
                for t, field in fields:
                    if t == table and not _equal(
                            {field: before[field]} if before and field in before else {},
                            {field: after[field]} if after and field in after else {}):
                        writers[table, identifier, field] = {
                            "valid": False, "step_index": position, "reason": "unreviewed_writer"}
                if after is None:
                    current[table].pop(identifier, None)
                else:
                    current[table][identifier] = deepcopy(after)
            for rule in contracts["rules"]:
                if event["tool_name"] != rule["tool"]:
                    continue
                args = event["arguments"]
                if not (event.get("success") is True and event.get("outcome") == "success"
                        and args.get("confirm") is True):
                    continue
                selected = [row for row in current[rule["table"]].values()
                            if all(arg in args and field in row and _equal(row[field], args[arg])
                                   for field, arg in rule["selector"].items())]
                if len(selected) != 1:
                    raise StateCompatibilityError("Successful status call has no unique persisted target")
                row = selected[0]
                value = args.get(rule["parameter"])
                eligible = value is None or _equal(value, rule["default_value"])
                valid = (eligible and rule["field"] in row
                         and _equal(row[rule["field"]], rule["default_value"])
                         and rule["status_parameter"] in args
                         and rule["status_field"] in row
                         and _equal(row[rule["status_field"]], args[rule["status_parameter"]]))
                writers[rule["table"], row[rule["primary_key"]], rule["field"]] = {
                    "valid": valid, "rule_id": rule["id"], "step_index": position,
                    "branch": "default" if value is None else ("explicit_default" if eligible else "explicit_unreviewed"),
                    "reason": "" if valid else ("unsupported_explicit_value" if not eligible
                                                  else "expected_status_time_not_persisted")}
        if not _equal(current, self.agent):
            raise StateCompatibilityError("Replay events do not reconstruct the final state")
        self.writers, self.rejected = writers, set()
        self.agent_state = deepcopy(agent_state)
        copied = _tables(self.agent_state)
        done = set()
        for rule in contracts["rules"]:
            table, field, key = rule["table"], rule["field"], rule["primary_key"]
            for row in copied[table]:
                identifier = row[key]
                slot = table, identifier, field
                if slot in done:
                    continue
                done.add(slot)
                writer = writers.get(slot)
                if field not in row and writer is None:
                    continue
                old = self.initial[table].get(identifier, {})
                expected = self.gold[table].get(identifier, {})
                eligible = field not in old and field not in expected
                record = {"table": table, "primary_key": key, "row_id": identifier,
                          "field": field, "present": field in row,
                          "raw_value": row.get(field), **(writer or {"reason": "no_actual_writer"})}
                if eligible and writer and writer["valid"] and field in row:
                    del row[field]
                    self.diagnostic["applied"].append(record)
                else:
                    # Keep unproven values literal; only required rows make this
                    # an additional failure. Extra rows retain subset semantics.
                    record["reason"] = ("field_already_in_common_state" if not eligible
                                        else record.get("reason") or "unproven_auxiliary_value")
                    self.diagnostic["unapplied"].append(record)
                    if eligible:
                        self.rejected.add((table, identifier))

    def finish(self, result, evaluation):
        mapping = {}
        identity = result.get("identity_comparison", {})
        entries = identity.get("mapping")
        if type(entries) is list:
            mapping = {(e["table"], e["gold_id"]): e["agent_id"] for e in entries}
        elif type(entries) is dict and evaluation["strategy"] == "new_entity_pk":
            mapping = {(evaluation["targets"][0], old): new for old, new in entries.items()}
        section = (self.identity_schema or {}).get("domains", {}).get(self.contracts["domain"], {})
        extension = section.get("strategy_extensions", {}).get(evaluation["strategy"], {})
        references = {(e["source_table"], e["field"]): e["target_entity"]
                      for view in (section, extension) for e in view.get("foreign_keys", [])}
        seen_tables = set()
        for rule in self.contracts["rules"]:
            table = rule["table"]
            if table in seen_tables:
                continue
            seen_tables.add(table)
            used = set()
            for identifier, expected in self.gold[table].items():
                old = self.initial[table].get(identifier)
                if _equal(old, expected):
                    continue
                if any(field not in expected for field in rule["core_fields"]):
                    raise StateCompatibilityError(f"Gold lacks required core fields: {table}")
                values = {f: mapping.get((references.get((table, f)), expected[f]), expected[f])
                          if type(expected[f]) is str else expected[f] for f in rule["core_fields"]}
                candidates = [identifier] if old is not None else sorted(
                    self.agent[table].keys() - self.initial[table].keys())
                matches = [i for i in candidates if i not in used and i in self.agent[table]
                           and (table, i) not in self.rejected
                           and all(f in self.agent[table][i] and _equal(self.agent[table][i][f], value)
                                   for f, value in values.items())]
                check = {"table": table, "gold_id": identifier, "core_fields": list(values),
                         "passed": bool(matches), "agent_id": matches[0] if matches else None}
                self.diagnostic["business_checks"].append(check)
                if matches:
                    used.add(matches[0])
                else:
                    self.diagnostic["failures"].append({**check, "reason": "required_business_state_or_auxiliary_provenance"})
        self.diagnostic["passed"] = not self.diagnostic["failures"]
        if not self.diagnostic["passed"]:
            result.update(passed=False, reason="State compatibility: required business state or provenance failed")
        result["state_compatibility"] = self.diagnostic
        return result



_PROFILE_FIELDS = {"domain", "table", "primary_key", "field", "kind", "default_value",
                   "core_fields", "gold", "writers"}
_WRITER_FIELDS = {"tool", "parameter", "allow_omitted", "allow_null", "explicit_values",
                  "selector", "bindings", "mode"}
_MAX_BINDING_STATES = 10000


def _key_fields(key):
    fields = [key] if type(key) is str else key
    if (type(fields) is not list or not fields or any(not _name(k) for k in fields)
            or len(set(fields)) != len(fields) or (type(key) is list and len(fields) < 2)):
        raise StateCompatibilityError("Invalid watched primary key")
    return fields


def _row_identity(row, key):
    fields = _key_fields(key)
    if type(row) is not dict or any(type(row.get(k)) is not str or not row[k] for k in fields):
        raise StateCompatibilityError("Missing or invalid watched identity")
    # Composite keys are observer indexes only; no synthetic DB field is added.
    return row[key] if type(key) is str else _encoded([row[k] for k in fields])


def _audited_success(result):
    return (result_outcome(result) == "success"
            and (result.get("success") is True or result.get("permitted") is True))


def _fingerprint(value):
    return (type(value) is str and value.startswith("sha256:") and len(value) == 71
            and all(c in "0123456789abcdef" for c in value[7:]))


def _validate_admission(value):
    if (type(value) is not dict
            or set(value) != {"schema_version", "semantic_fingerprint", "rules"}
            or type(value["schema_version"]) is not int or value["schema_version"] != 1
            or not _fingerprint(value["semantic_fingerprint"])
            or type(value["rules"]) is not list or not value["rules"]):
        raise StateCompatibilityError("Invalid canonical state comparison admission")
    seen = set()
    for item in value["rules"]:
        if type(item) is not dict or set(item) != {"profile", "target"} or not _name(item["profile"]):
            raise StateCompatibilityError("Invalid admitted state profile")
        target = item["target"]
        if (type(target) is not dict or set(target) != {"kind", "step_index"}
                or target["kind"] != "base_step_rows" or type(target["step_index"]) is not int
                or target["step_index"] < 0):
            raise StateCompatibilityError("Invalid admitted base target")
        key = item["profile"], target["step_index"]
        if key in seen:
            raise StateCompatibilityError("Duplicate admitted target/profile")
        seen.add(key)


def semantic_fingerprint(task):
    """Detect stale audited intent without incorporating the admission itself."""
    values = {k: task.get(k) for k in ("domain", "instruction", "user_system_prompt",
                                     "gold_trajectory", "version_trajectories")}
    values["reason_for_call"] = task.get("metadata", {}).get("reason_for_call")
    return "sha256:" + _digest(values)


def parse_state_comparison(task):
    metadata = task.get("metadata", {})
    if type(metadata) is not dict:
        raise StateCompatibilityError("Canonical metadata must be an object")
    if "state_comparison" not in metadata:
        return None
    value = metadata["state_comparison"]
    _validate_admission(value)
    if value["semantic_fingerprint"] != semantic_fingerprint(task):
        raise StateCompatibilityError("Stale canonical state comparison admission")
    tools = task.get("gold_trajectory", {}).get("tools")
    if type(tools) is not list or any(item["target"]["step_index"] >= len(tools) for item in value["rules"]):
        raise StateCompatibilityError("Admitted target is outside the base trajectory")
    return deepcopy(value)


def _validate_selector(selector, *, required=True):
    if type(selector) is not dict or (required and not selector):
        raise StateCompatibilityError("A target needs an explicit selector")
    for field, ref in selector.items():
        if (not _name(field) or type(ref) is not dict or set(ref) != {"source", "path"}
                or type(ref["source"]) is not str or ref["source"] not in {"arguments", "result"}
                or type(ref["path"]) is not list or not ref["path"]
                or any(not _name(k) for k in ref["path"])):
            raise StateCompatibilityError("Invalid target input/result path")


def _validate_profile(profile):
    mirror = type(profile) is dict and profile.get("kind") == "auxiliary_parameter_mirror"
    fields = _PROFILE_FIELDS | ({"relation_source"} if mirror else set())
    if type(profile) is not dict or set(profile) != fields:
        raise StateCompatibilityError("Unknown or missing audited profile fields")
    if any(not _name(profile[k]) for k in ("domain", "table", "field")):
        raise StateCompatibilityError("Invalid audited profile name")
    key_fields = _key_fields(profile["primary_key"])
    field, core = profile["field"], profile["core_fields"]
    if (field in key_fields or field == "id" or field.endswith("_id")
            or type(core) is not list or not core or len(set(core)) != len(core)
            or any(not _name(k) for k in core) or field in core):
        raise StateCompatibilityError("Invalid identity/core fields in audited profile")
    if type(profile["kind"]) is not str or profile["kind"] not in {"auxiliary_literal", "calendar_date", "auxiliary_parameter_mirror"}:
        raise StateCompatibilityError("Unknown audited comparison kind")
    if profile["kind"] == "auxiliary_literal" and type(profile["default_value"]) is not str:
        raise StateCompatibilityError("Audited auxiliary values must be explicit strings")
    if profile["kind"] == "calendar_date" and profile["default_value"] is not None:
        raise StateCompatibilityError("Calendar rules cannot invent a default date")
    if mirror and (profile["default_value"] is not None or type(profile["primary_key"]) is not str):
        raise StateCompatibilityError("Parameter mirrors require a scalar target key and a bound value")
    gold = profile["gold"]
    gold_fields = {"tool", "selector"} | ({"value_source"} if mirror else set())
    if type(gold) is not dict or set(gold) != gold_fields or not _name(gold["tool"]):
        raise StateCompatibilityError("Invalid gold target binding")
    _validate_selector(gold["selector"])
    if mirror:
        _validate_selector({"value": gold["value_source"]})
        if gold["value_source"]["source"] != "arguments":
            raise StateCompatibilityError("Parameter mirror values must come from actual gold arguments")
        source = profile["relation_source"]
        if (type(source) is not dict or set(source) != {
                "table", "field", "target_foreign_key", "mode", "bindings"}
                or any(not _name(source[k]) for k in ("table", "field", "target_foreign_key"))
                or source["mode"] != "unique_new_row" or source["table"] == profile["table"]
                or source["field"] == "id" or source["field"].endswith("_id")):
            raise StateCompatibilityError("Invalid parameter mirror relation source")
        _validate_selector(source["bindings"])
        if (source["target_foreign_key"] in source["bindings"]
                or source["field"] in source["bindings"]
                or any(ref["source"] != "arguments" for ref in source["bindings"].values())):
            raise StateCompatibilityError("Relation scope cannot override its target or mirror value")
    versions = profile["writers"]
    if type(versions) is not dict or set(versions) != _VERSIONS or versions["base"]:
        raise StateCompatibilityError("Audited profiles must declare versions and leave base literal")
    for writers in versions.values():
        if type(writers) is not list:
            raise StateCompatibilityError("Audited writers must be a list")
        tools = set()
        for writer in writers:
            writer_fields = _WRITER_FIELDS | ({"omitted_value"} if mirror else set())
            if (type(writer) is not dict or set(writer) != writer_fields
                    or not _name(writer["tool"]) or not _name(writer["parameter"])
                    or type(writer["mode"]) is not str or writer["mode"] not in {"create", "update", "either"}
                    or type(writer["allow_omitted"]) is not bool or type(writer["allow_null"]) is not bool
                    or type(writer["explicit_values"]) is not list
                    or any(type(v) is not str for v in writer["explicit_values"])):
                raise StateCompatibilityError("Invalid audited writer declaration")
            if writer["tool"] in tools:
                raise StateCompatibilityError("Duplicate audited writer")
            tools.add(writer["tool"])
            _validate_selector(writer["selector"])
            _validate_selector(writer["bindings"], required=False)
            if field in writer["selector"]:
                raise StateCompatibilityError("An auxiliary value cannot select its own target")
            if profile["kind"] == "calendar_date" and (
                    writer["allow_omitted"] or writer["allow_null"] or writer["explicit_values"]):
                raise StateCompatibilityError("Calendar writer requires the actual explicit date")
            if mirror:
                omitted = writer["omitted_value"]
                if (writer["mode"] != "create" or writer["allow_null"]
                        or not writer["explicit_values"]
                        or len(set(writer["explicit_values"])) != len(writer["explicit_values"])
                        or any(not value for value in writer["explicit_values"])
                        or (writer["allow_omitted"] and (
                            type(omitted) is not str or omitted not in writer["explicit_values"]))
                        or (not writer["allow_omitted"] and omitted is not None)):
                    raise StateCompatibilityError("Invalid explicit/default parameter mirror writer")
    if mirror and not any(profile["writers"].values()):
        raise StateCompatibilityError("Parameter mirror has no reviewed writer")


def _observation_rules(contracts):
    return [*contracts["rules"], *[entry["spec"] for entry in contracts.get("audited", [])]]


def _reference_value(reference, arguments, result):
    current = arguments if reference["source"] == "arguments" else result
    for key in reference["path"]:
        if type(current) is not dict or key not in current:
            raise StateCompatibilityError("Missing actual target binding value")
        current = current[key]
    return current


def _select_values(selector, arguments, result):
    return {field: _reference_value(ref, arguments, result) for field, ref in selector.items()}


def _matches_values(row, values):
    return all(field in row and _equal(row[field], value) for field, value in values.items())


def _mirror_relation(tables, initial_tables, spec, identifier, bindings, expected):
    """Locate one new scoped raw relationship; never project relationship data."""
    source = spec["relation_source"]
    table = source["table"]
    rows, before = tables.get(table, []), initial_tables.get(table, [])
    if (type(rows) is not list or type(before) is not list
            or any(type(row) is not dict for row in [*rows, *before])):
        raise StateCompatibilityError("Invalid parameter mirror relationship table")
    selector = {source["target_foreign_key"]: identifier, **bindings}
    matches = [row for row in rows if _matches_values(row, selector)]
    if len(matches) != 1 or any(_matches_values(row, selector) for row in before):
        raise StateCompatibilityError("Parameter mirror needs one unique new scoped relationship")
    row = matches[0]
    if source["field"] not in row or not _equal(row[source["field"]], expected):
        raise StateCompatibilityError("Parameter mirror relationship value differs from the bound value")
    return row


def bind_state_targets(contracts, gold_steps, initial_state, gold_state):
    """Resolve admitted semantic targets from internal base replay, never agent text."""
    if not contracts.get("audited"):
        return contracts
    out = deepcopy(contracts)
    initial, gold = _watched(_tables(initial_state), out), _watched(_tables(gold_state), out)
    for entry in out["audited"]:
        spec, index = entry["spec"], entry["step_index"]
        if not 0 <= index < len(gold_steps):
            raise StateCompatibilityError("Missing admitted gold operation")
        step = gold_steps[index]
        if (step.get("step_index") != index or step.get("tool_name") != spec["gold"]["tool"]
                or step.get("success") is not True or result_outcome(step.get("result", {})) != "success"):
            raise StateCompatibilityError("Admitted gold operation did not complete")
        values = _select_values(spec["gold"]["selector"], step["arguments"], step["result"])
        rows = [r for r in gold[spec["table"]].values() if _matches_values(r, values)]
        if len(rows) > 1:
            rows = [r for r in rows if not _equal(r, initial[spec["table"]].get(_row_identity(r, spec["primary_key"]))) ]
        if len(rows) != 1:
            raise StateCompatibilityError("Gold operation has no unique persisted target")
        row = rows[0]
        if any(field not in row for field in spec["core_fields"]):
            raise StateCompatibilityError("Gold target lacks audited core business fields")
        if spec["kind"] == "calendar_date" and not _calendar_day(row.get(spec["field"]), midnight=False):
            raise StateCompatibilityError("Admitted gold target is not a literal calendar date")
        entry.update(gold_id=_row_identity(row, spec["primary_key"]), gold_row_digest=_digest(row))
        if spec["kind"] == "auxiliary_parameter_mirror":
            identifier = entry["gold_id"]
            if identifier in initial[spec["table"]] or spec["field"] in row:
                raise StateCompatibilityError("Parameter mirrors require a new gold target without the extra field")
            expected = _reference_value(spec["gold"]["value_source"], step["arguments"], step["result"])
            allowed = {v for writers in spec["writers"].values()
                       for writer in writers for v in writer["explicit_values"]}
            if type(expected) is not str or expected not in allowed:
                raise StateCompatibilityError("Gold parameter mirror value is not an admitted string")
            bindings = _select_values(spec["relation_source"]["bindings"], step["arguments"], step["result"])
            if any(type(value) is not str or not value for value in bindings.values()):
                raise StateCompatibilityError("Gold parameter mirror relation scope is invalid")
            relation = _mirror_relation(_tables(gold_state), _tables(initial_state), spec,
                                        identifier, bindings, expected)
            entry.update(expected_value=expected, relation_bindings=bindings,
                         relation_row_digest=_digest(relation))
    return out


def _calendar_day(value, *, midnight):
    if type(value) is not str:
        return None
    fmt = "%Y-%m-%dT00:00:00Z" if midnight else "%Y-%m-%d"
    try:
        parsed = datetime.strptime(value, fmt)
    except ValueError:
        return None
    day = parsed.date().isoformat()
    if (day + "T00:00:00Z" if midnight else day) != value:
        return None
    return day


class PreparedAuditedStateCompatibility:
    """Try bounded target correspondences; project only a proven row/field pair.

    The existing identity strategy independently proves each proposed mapping.
    No task ID, unscoped row deletion, or nearest-candidate heuristic is used.
    """

    def __init__(self, *, contracts, evidence, initial_state, gold_state, agent_state,
                 agent_initial_state, identity_schema=None):
        self.context = _context(contracts)
        if contracts["rules"]:
            raise StateCompatibilityError("Mixed legacy/audited domains require a separately reviewed contract")
        if (not _equal(initial_state.entities, agent_initial_state.entities)
                or not _equal(initial_state.relationships, agent_initial_state.relationships)):
            raise StateCompatibilityError("Gold and agent raw initial databases differ")
        self.contracts, self.identity_schema = deepcopy(contracts), identity_schema or {}
        self.initial = _watched(_tables(initial_state), contracts)
        self.gold = _watched(_tables(gold_state), contracts)
        self.agent = _watched(_tables(agent_state), contracts)
        self.agent_state, self.initial_state = agent_state, initial_state
        self.all_gold, self.all_agent = _tables(gold_state), _tables(agent_state)
        protected = _identity_fields(identity_schema, self.context["domain"])
        for entry in contracts["audited"]:
            spec = entry["spec"]
            if (spec["table"], spec["field"]) in protected:
                raise StateCompatibilityError("Audited field overlaps identity or derived structure")
            row = self.gold[spec["table"]].get(entry["gold_id"])
            if row is None or _digest(row) != entry["gold_row_digest"]:
                raise StateCompatibilityError("Bound gold target does not match the current raw state")
            if spec["kind"] == "auxiliary_parameter_mirror":
                if entry["gold_id"] in self.initial[spec["table"]] or spec["field"] in row:
                    raise StateCompatibilityError("Bound mirror target is not a new absent-field gold entity")
                relation = _mirror_relation(self.all_gold, _tables(initial_state), spec,
                                            entry["gold_id"], entry["relation_bindings"], entry["expected_value"])
                if _digest(relation) != entry["relation_row_digest"]:
                    raise StateCompatibilityError("Bound gold mirror relationship no longer matches")
        self.diagnostic = {**self.context, "passed": False, "applied": [], "unapplied": [],
                           "business_checks": [], "failures": [], "binding_search_states": 0}
        self.writers = self._validate_evidence(evidence)
        section = self.identity_schema.get("domains", {}).get(self.context["domain"], {})
        views = [section, *section.get("strategy_extensions", {}).values()]
        self.references = {(e["source_table"], e["field"]): e["target_entity"]
                           for view in views for e in view.get("foreign_keys", [])}
        self.entity_keys = {t: r["primary_key"] for view in views for t, r in view.get("entities", {}).items()}
        self.generated = {}
        initial_tables = _tables(initial_state)
        for table, key in self.entity_keys.items():
            before = {r[key] for r in initial_tables.get(table, [])}
            self.generated[table] = {r[key] for r in self.all_gold.get(table, [])} - before

    def _validate_evidence(self, evidence):
        if (type(evidence) is not dict or evidence.get("context") != self.context
                or evidence.get("initial_digest") != _digest(self.initial)
                or evidence.get("final_digest") != _digest(self.agent)
                or type(evidence.get("events")) is not list):
            raise StateCompatibilityError("Missing or mismatched internal state evidence")
        current, writers = deepcopy(self.initial), {}
        specs = {e["profile"]: e["spec"] for e in self.contracts["audited"]}
        table_keys = {s["table"]: s["primary_key"] for s in specs.values()}
        fields = {(s["table"], s["field"]) for s in specs.values()}
        for index, event in enumerate(evidence["events"]):
            if (type(event) is not dict or event.get("step_index") != index
                    or type(event.get("arguments")) is not dict or type(event.get("result")) is not dict
                    or event.get("outcome") != result_outcome(event["result"])
                    or event.get("success") != (_audited_success(event["result"]))
                    or type(event.get("changes")) is not list):
                raise StateCompatibilityError("Invalid actual replay event")
            changed, field_changes = set(), set()
            for change in event["changes"]:
                table, identifier = change["table"], change["id"]
                if (table not in current or (table, identifier) in changed
                        or not _equal(current[table].get(identifier), change["before"])):
                    raise StateCompatibilityError("Inconsistent replay state chain")
                changed.add((table, identifier))
                old, new = change["before"], change["after"]
                if new is not None and (type(new) is not dict or _row_identity(new, table_keys[table]) != identifier):
                    raise StateCompatibilityError("Actual event renamed a literal target identity")
                for t, field in fields:
                    if t == table and not _equal(
                            {field: old[field]} if old is not None and field in old else {},
                            {field: new[field]} if new is not None and field in new else {}):
                        slot = table, identifier, field
                        field_changes.add(slot)
                        writers[slot] = {"valid": False, "reason": "unreviewed_actual_writer", "step_index": index}
                if new is None:
                    current[table].pop(identifier, None)
                else:
                    current[table][identifier] = deepcopy(new)
            if not (event["success"] and event["outcome"] == "success"
                    and event["arguments"].get("confirm") is True):
                continue
            for profile, spec in specs.items():
                for writer in spec["writers"][self.contracts["version"]]:
                    if writer["tool"] != event["tool_name"]:
                        continue
                    args, result = event["arguments"], event["result"]
                    try:
                        selected = _select_values(writer["selector"], args, result)
                        bindings = _select_values(writer["bindings"], args, result)
                    except StateCompatibilityError:
                        continue
                    rows = [r for identifier, r in current[spec["table"]].items()
                            if (spec["table"], identifier) in changed and _matches_values(r, selected)]
                    if len(rows) != 1:
                        continue
                    row = rows[0]; identifier = _row_identity(row, spec["primary_key"])
                    slot = spec["table"], identifier, spec["field"]
                    if slot not in field_changes or not _matches_values(row, bindings):
                        continue
                    change = next(c for c in event["changes"] if (c["table"],c["id"]) == (spec["table"],identifier))
                    is_new = change["before"] is None
                    if (writer["mode"] == "create" and not is_new) or (writer["mode"] == "update" and is_new):
                        continue
                    param, field = writer["parameter"], spec["field"]
                    branch = "omitted" if param not in args else "null" if args[param] is None else "explicit"
                    mirror_evidence = {}
                    if spec["kind"] == "auxiliary_literal":
                        allowed = (writer["allow_omitted"] if branch == "omitted" else writer["allow_null"] if branch == "null"
                                   else any(_equal(args[param], v) for v in writer["explicit_values"]))
                        valid = allowed and field in row and _equal(row[field], spec["default_value"])
                    elif spec["kind"] == "calendar_date":
                        valid = (branch == "explicit" and field in row and _equal(row[field], args[param])
                                 and _calendar_day(row[field], midnight=True) is not None)
                    else:
                        effective = writer["omitted_value"] if branch == "omitted" else args[param]
                        allowed = (writer["allow_omitted"] if branch == "omitted" else
                                   branch == "explicit" and type(effective) is str
                                   and effective in writer["explicit_values"])
                        try:
                            relation_bindings = _select_values(spec["relation_source"]["bindings"], args, result)
                        except StateCompatibilityError:
                            continue
                        valid = (allowed and type(effective) is str and field in row
                                 and _equal(row[field], effective)
                                 and all(type(v) is str and v for v in relation_bindings.values()))
                        mirror_evidence = {"effective_value": effective, "relation_bindings": relation_bindings}
                    if valid:
                        writers[slot] = {"valid": True, "profile": profile, "tool": writer["tool"],
                                         "step_index": index, "argument_branch": branch, **mirror_evidence}
        if not _equal(current, self.agent):
            raise StateCompatibilityError("Actual events do not reconstruct the final raw state")
        return writers

    def _core_matches(self, table, expected, actual, fields, mapping=None):
        for field in fields:
            if field not in expected or field not in actual:
                return False
            value = expected[field]
            target = self.references.get((table, field))
            if target and type(value) is str and value in self.generated.get(target, set()):
                if mapping is None:
                    key = self.entity_keys[target]
                    before = {r[key] for r in _tables(self.initial_state).get(target, [])}
                    available = {r[key] for r in self.all_agent.get(target, [])} - before
                    if type(actual[field]) is not str or actual[field] not in available:
                        return False
                    continue
                if (target, value) not in mapping:
                    return False
                value = mapping[target, value]
            if not _equal(value, actual[field]):
                return False
        return True

    def _reject_pair(self, table, gold_id, agent_id, reason, **details):
        if len(self.diagnostic["unapplied"]) < 32:
            self.diagnostic["unapplied"].append({"table": table, "gold_id": gold_id,
                "agent_id": agent_id, "reason": reason, **details})
        return None

    def _mirror_pair_issue(self, entry, identifier):
        spec, expected = entry["spec"], entry["expected_value"]
        table, field = spec["table"], spec["field"]
        if identifier in self.initial[table]:
            return "mirror_target_is_not_new"
        actual = self.agent[table][identifier]
        if field in actual:
            writer = self.writers.get((table, identifier, field), {})
            if not writer.get("valid") or writer.get("profile") != entry["profile"]:
                return "unproven_mirror_writer"
            if not _equal(writer.get("effective_value"), expected) or not _equal(actual[field], expected):
                return "mirror_parameter_or_entity_value_mismatch"
            if not _equal(writer.get("relation_bindings"), entry["relation_bindings"]):
                return "mirror_actor_binding_mismatch"
        # Absence needs no projection, but the admitted target's relationship
        # must still be unambiguous and match its actual base creation scope.
        try:
            _mirror_relation(self.all_agent, _tables(self.initial_state), spec, identifier,
                             entry["relation_bindings"], expected)
        except StateCompatibilityError:
            return "mirror_relationship_not_unique_new_or_value_mismatch"
        return None

    def _pair(self, entries, identifier):
        first = entries[0]; spec = first["spec"]; table, gold_id = spec["table"], first["gold_id"]
        expected, actual = self.gold[table][gold_id], self.agent[table][identifier]
        core = sorted({f for entry in entries for f in entry["spec"]["core_fields"]})
        calendar_fields = {e["spec"]["field"] for e in entries if e["spec"]["kind"] == "calendar_date"}
        ordinary_core = [f for f in core if f not in calendar_fields]
        if not self._core_matches(table, expected, actual, ordinary_core):
            differing = [f for f in ordinary_core if not self._core_matches(table, expected, actual, [f])]
            return self._reject_pair(table, gold_id, identifier, "raw_core_fields_differ", fields=differing)
        projected, applied = deepcopy(actual), []
        for entry in entries:
            spec, profile = entry["spec"], entry["profile"]
            field = spec["field"]
            if spec["kind"] == "auxiliary_parameter_mirror":
                issue = self._mirror_pair_issue(entry, identifier)
                if issue:
                    return self._reject_pair(table, gold_id, identifier, issue, profile=profile, field=field)
            left = {field: expected[field]} if field in expected else {}
            right = {field: actual[field]} if field in actual else {}
            if _equal(left, right):
                continue
            writer = self.writers.get((table, identifier, field), {})
            if not writer.get("valid") or writer.get("profile") != profile:
                return self._reject_pair(table, gold_id, identifier, "unproven_actual_writer",
                                         profile=profile, field=field, writer=writer)
            if spec["kind"] in {"auxiliary_literal", "auxiliary_parameter_mirror"}:
                permitted = entry["expected_value"] if spec["kind"] == "auxiliary_parameter_mirror" else spec["default_value"]
                if (field in expected or field in self.initial[table].get(gold_id, {})
                        or field in self.initial[table].get(identifier, {})
                        or field not in actual or not _equal(actual[field], permitted)):
                    return self._reject_pair(table, gold_id, identifier, "not_an_admitted_absent_auxiliary_field",
                                             profile=profile, field=field)
                projected.pop(field)
                compared = {"present": False}
            else:
                day = _calendar_day(actual.get(field), midnight=True)
                if not day or day != _calendar_day(expected.get(field), midnight=False):
                    return self._reject_pair(table, gold_id, identifier, "different_or_unsupported_calendar_date",
                                             profile=profile, field=field)
                projected[field] = day
                compared = {"present": True, "value": day}
            applied.append({"profile": profile, "table": table, "field": field,
                            "gold_id": gold_id, "agent_id": identifier, "raw_value": actual[field],
                            "comparison_value": compared, **writer})
            if spec["kind"] == "auxiliary_parameter_mirror":
                applied[-1]["expected_value"] = entry["expected_value"]
        if not self._core_matches(table, expected, projected, core):
            return self._reject_pair(table, gold_id, identifier, "proven_calendar_core_fields_differ")
        return {"table": table, "gold_id": gold_id, "agent_id": identifier,
                "core_fields": core, "projected": projected, "applied": applied}

    def variants(self):
        groups = {}
        for entry in self.contracts["audited"]:
            key = entry["spec"]["table"], entry["gold_id"]
            groups.setdefault(key, {})[entry["profile"]] = entry
        options = []
        for (table, gold_id), items in groups.items():
            candidates = [gold_id] if gold_id in self.initial[table] else sorted(self.agent[table].keys() - self.initial[table].keys())
            choices = [pair for identifier in candidates if identifier in self.agent[table]
                       if (pair := self._pair(list(items.values()), identifier)) is not None]
            if not choices:
                self.diagnostic["failures"].append({"table": table, "gold_id": gold_id,
                    "reason": "no_target_with_correct_business_values_and_audited_field_provenance"})
                return
            options.append(choices)
        options.sort(key=len)
        selected, used = [], set()

        def search(depth):
            self.diagnostic["binding_search_states"] += 1
            if self.diagnostic["binding_search_states"] > _MAX_BINDING_STATES:
                raise StateCompatibilityError("Audited target search limit exceeded; comparison incomplete")
            if depth == len(options):
                state = deepcopy(self.agent_state); tables = _tables(state)
                for pair in selected:
                    table, identifier = pair["table"], pair["agent_id"]
                    key = next(e["spec"]["primary_key"] for e in self.contracts["audited"] if e["spec"]["table"] == table)
                    for i, row in enumerate(tables[table]):
                        if _row_identity(row, key) == identifier:
                            tables[table][i] = deepcopy(pair["projected"])
                            break
                yield state, deepcopy(selected)
                return
            for choice in options[depth]:
                slot = choice["table"], choice["agent_id"]
                if slot in used:
                    continue
                used.add(slot); selected.append(choice)
                yield from search(depth + 1)
                selected.pop(); used.remove(slot)
        yield from search(0)

    def finish(self, result, evaluation, pairs):
        diagnostic = deepcopy(self.diagnostic)
        mapping = {}
        identity = result.get("identity_comparison", {})
        entries = identity.get("mapping")
        if type(entries) is list:
            mapping.update({(e["table"], e["gold_id"]): e["agent_id"] for e in entries})
        elif type(entries) is dict and evaluation["strategy"] == "new_entity_pk":
            mapping.update({(evaluation["targets"][0], k): v for k, v in entries.items()})
        mapping.update({(e["table"], e["gold_id"]): e["agent_id"]
                        for e in identity.get("matched_relationships", [])})
        for pair in pairs:
            table, gold_id, agent_id = pair["table"], pair["gold_id"], pair["agent_id"]
            # Only proven calendar fields differ in this core-field view; every
            # other business value still comes from the original observed row.
            actual = deepcopy(self.agent[table][agent_id])
            for applied in pair["applied"]:
                if applied["comparison_value"]["present"]:
                    actual[applied["field"]] = applied["comparison_value"]["value"]
            valid = self._core_matches(table, self.gold[table][gold_id], actual, pair["core_fields"], mapping)
            if (table in evaluation["targets"] or (table, gold_id) in mapping) and gold_id not in self.initial[table]:
                valid = valid and mapping.get((table, gold_id)) == agent_id
            mirror_checks = []
            for entry in self.contracts["audited"]:
                spec = entry["spec"]
                if (spec["kind"] == "auxiliary_parameter_mirror"
                        and (spec["table"], entry["gold_id"]) == (table, gold_id)):
                    issue = self._mirror_pair_issue(entry, agent_id)
                    mirror_checks.append({"profile": entry["profile"], "expected_value": entry["expected_value"],
                                          "relation_bindings": deepcopy(entry["relation_bindings"]),
                                          "passed": issue is None, "reason": issue})
                    valid = valid and issue is None
            check = {"table": table, "gold_id": gold_id, "agent_id": agent_id, "passed": valid}
            if mirror_checks:
                check["mirror_checks"] = mirror_checks
            diagnostic["business_checks"].append(check)
            diagnostic["applied"].extend(pair["applied"])
            if not valid:
                diagnostic["failures"].append({"table": table, "gold_id": gold_id, "reason": "raw_business_or_identity_binding_failed"})
        diagnostic["passed"] = not diagnostic["failures"] and bool(pairs)
        if not diagnostic["passed"]:
            result.update(passed=False, reason="Audited state compatibility: target business state or provenance failed")
        result["state_compatibility"] = diagnostic
        return result
