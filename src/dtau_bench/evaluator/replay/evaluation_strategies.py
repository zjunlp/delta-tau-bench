"""Task-category routing for state comparison.

Category declarations come from canonical task metadata. Identity structure is
shared by domain and table; no strategy depends on a task ID. The first identity
strategy supports one newly generated UUID primary key without gold references.
All other required state changes still use the original subset comparator.
"""
from __future__ import annotations

import copy
from collections import Counter, defaultdict
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Optional

from dtau_bench.evaluator.replay.diffing import (
    DiffMetadata,
    evaluate_gold_subset_relation,
)
from dtau_bench.evaluator.replay.executor import DatabaseState
from dtau_bench.evaluator.replay.state_compatibility import (
    PreparedStateCompatibility, PreparedAuditedStateCompatibility, StateCompatibilityError,
)


class EvaluationStrategyError(ValueError):
    """Invalid configuration or a state outside the declared strategy's scope."""


_UUID_TEXT = r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}"
_UUID = re.compile(_UUID_TEXT, re.IGNORECASE)
_UUID_IN_TEXT = re.compile(_UUID_TEXT, re.IGNORECASE)


def parse_evaluation_config(value: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Validate a declaration; None is the legacy, undeclared default route."""
    if value is None:
        return {"schema_version": 1, "strategy": "default", "targets": []}
    if type(value) is not dict:
        raise EvaluationStrategyError("metadata.evaluation must be an object")
    if set(value) - {"schema_version", "strategy", "targets"}:
        raise EvaluationStrategyError("Unknown metadata.evaluation fields")
    if type(value.get("schema_version")) is not int or value["schema_version"] != 1:
        raise EvaluationStrategyError("Unsupported evaluation schema_version")
    strategy = value.get("strategy")
    if type(strategy) is not str or strategy not in _STRATEGIES:
        raise EvaluationStrategyError(f"Unknown or unimplemented evaluation strategy: {strategy!r}")
    targets = value.get("targets", [])
    if type(targets) is not list or any(type(t) is not str or not t for t in targets):
        raise EvaluationStrategyError("evaluation.targets must be a list of table names")
    if strategy == "default" and targets:
        raise EvaluationStrategyError("default strategy cannot declare identity targets")
    if strategy == "new_entity_pk" and len(targets) != 1:
        raise EvaluationStrategyError("new_entity_pk requires exactly one target table")
    if strategy == "new_entity_pk_fk" and (not targets or len(set(targets)) != len(targets)):
        raise EvaluationStrategyError("new_entity_pk_fk requires distinct, nonempty targets")
    if strategy == "new_entity_pk_fk_custom_id" and (not targets or len(set(targets)) != len(targets)):
        raise EvaluationStrategyError("new_entity_pk_fk_custom_id requires distinct, nonempty targets")
    if strategy == "new_entity_pk_fk_derived" and (not targets or len(set(targets)) != len(targets)):
        raise EvaluationStrategyError("new_entity_pk_fk_derived requires distinct, nonempty targets")
    if strategy == "new_relationship_pk_derived" and (not targets or len(set(targets)) != len(targets)):
        raise EvaluationStrategyError("new_relationship_pk_derived requires distinct, nonempty targets")
    return {"schema_version": 1, "strategy": strategy, "targets": list(targets)}


def load_identity_schema(data_dir: Path) -> Dict[str, Any]:
    """Load shared identity structure only when an identity strategy is used."""
    path = Path(data_dir) / "meta" / "eval_identity_schema.json"
    try:
        with path.open(encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, ValueError) as exc:
        raise EvaluationStrategyError(f"Cannot load identity schema: {path}: {exc}") from exc
    return value


def validate_strategy_schema(
    evaluation: Optional[Dict[str, Any]],
    domain: str,
    identity_schema: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Resolve declared table structure before starting either replay."""
    config = parse_evaluation_config(evaluation)
    if config["strategy"] == "default":
        return {}
    if type(identity_schema) is not dict:
        raise EvaluationStrategyError("Identity strategy requires a shared identity schema")
    if (type(identity_schema.get("schema_version")) is not int
            or identity_schema["schema_version"] != 1):
        raise EvaluationStrategyError("Unsupported identity schema_version")
    domains = identity_schema.get("domains")
    if type(domains) is not dict or type(domains.get(domain)) is not dict:
        raise EvaluationStrategyError(f"Identity schema missing domain: {domain}")
    entities = domains[domain].get("entities")
    if config["strategy"] == "new_relationship_pk_derived":
        return _validate_relationship_derived_schema(config, domains[domain])
    if config["strategy"] == "new_entity_pk_fk_derived":
        return _validate_derived_schema(config, domains[domain])
    if config["strategy"] == "new_entity_pk_fk_custom_id":
        return _validate_custom_id_schema(config, domains[domain])
    if config["strategy"] == "new_entity_pk_fk":
        return _validate_pk_fk_schema(config, domains[domain])
    table = config["targets"][0]
    if type(entities) is not dict or type(entities.get(table)) is not dict:
        raise EvaluationStrategyError(f"Identity schema missing entity: {domain}.{table}")
    rule = entities[table]
    if set(rule) != {"primary_key", "id_kind"}:
        raise EvaluationStrategyError(f"Unsupported identity fields for {domain}.{table}")
    if type(rule["primary_key"]) is not str or not rule["primary_key"]:
        raise EvaluationStrategyError(f"Invalid primary key for {domain}.{table}")
    if rule["id_kind"] != "uuid":
        raise EvaluationStrategyError("new_entity_pk only supports declared UUID identities")
    return {"table": table, "primary_key": rule["primary_key"]}


def _strict_equal(left: Any, right: Any) -> bool:
    """JSON equality preserving types, missing fields and list order."""
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        return (left.keys() == right.keys()
                and all(_strict_equal(v, right[k]) for k, v in left.items()))
    if type(left) is list:
        return len(left) == len(right) and all(_strict_equal(a, b) for a, b in zip(left, right))
    if type(left) is float:
        return math.isfinite(left) and math.isfinite(right) and left == right
    if type(left) in (str, int, bool, type(None)):
        return left == right
    raise EvaluationStrategyError(f"Unsupported state value type: {type(left).__name__}")


def _tables(state: DatabaseState) -> Dict[str, Any]:
    if not isinstance(state, DatabaseState):
        raise EvaluationStrategyError("Identity comparison requires captured database states")
    if type(state.entities) is not dict or type(state.relationships) is not dict:
        raise EvaluationStrategyError("Invalid captured table containers")
    overlap = state.entities.keys() & state.relationships.keys()
    if overlap:
        raise EvaluationStrategyError(f"Ambiguous table capture containers: {sorted(overlap)}")
    tables = {**state.entities, **state.relationships}
    for table, rows in tables.items():
        if type(table) is not str or type(rows) is not list or any(type(r) is not dict for r in rows):
            raise EvaluationStrategyError("Captured tables must contain lists of records")
    return tables


def _uuid_key(value: Any, label: str) -> str:
    if type(value) is not str or not _UUID.fullmatch(value):
        raise EvaluationStrategyError(f"Missing or invalid UUID primary key: {label}")
    return value.lower()


def _index(rows: Any, table: str, key: str) -> Dict[str, Any]:
    result = {}
    for row in rows:
        identifier = _uuid_key(row.get(key), f"{table}.{key}")
        if identifier in result:
            raise EvaluationStrategyError(f"Duplicate primary key: {table}.{key}={identifier}")
        result[identifier] = row
    return result


def _uuid_mentions(value: Any, path: tuple = ()):
    """Include embedded references and dictionary keys; never rewrite them."""
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise EvaluationStrategyError("State object keys must be strings")
            for match in _UUID_IN_TEXT.finditer(key):
                yield match.group().lower(), path + ("<key>", key)
            yield from _uuid_mentions(item, path + (key,))
    elif type(value) is list:
        for index, item in enumerate(value):
            yield from _uuid_mentions(item, path + (index,))
    elif type(value) is str:
        for match in _UUID_IN_TEXT.finditer(value):
            yield match.group().lower(), path


def _mapping_allowed(constraints, table, gold_id, agent_id):
    if not constraints:
        return True
    required = constraints.get((table, gold_id))
    if required is not None:
        return required == agent_id
    # Reserve constrained rows so another gold row cannot consume them first.
    return all(t != table or actual != agent_id for (t, _), actual in constraints.items())


def _evaluate_default(*, initial_state, gold_state, agent_state, diff_meta, **unused):
    # Preserve the legacy result, including its field filtering and statistics.
    return evaluate_gold_subset_relation(initial_state, gold_state, agent_state, diff_meta)


def _evaluate_new_entity_pk(
    *, initial_state, gold_state, agent_state, diff_meta,
    agent_initial_state, evaluation, identity_schema,
    mapping_constraints=None,
):
    rule = validate_strategy_schema(evaluation, diff_meta.domain, identity_schema)
    table, key = rule["table"], rule["primary_key"]
    before = _tables(initial_state)
    agent_before = _tables(agent_initial_state)
    gold_tables, agent_tables = _tables(gold_state), _tables(agent_state)
    if not _strict_equal(before, agent_before):
        raise EvaluationStrategyError("Gold and agent initial databases differ")

    initial_index = _index(before.get(table, []), table, key)
    gold_index = _index(gold_tables.get(table, []), table, key)
    agent_index = _index(agent_tables.get(table, []), table, key)
    for name, index in (("gold", gold_index), ("agent", agent_index)):
        if initial_index.keys() - index.keys():
            raise EvaluationStrategyError(f"{name} removed or renamed an initial {table} identity")
        if any(index[identifier][key] != row[key] for identifier, row in initial_index.items()):
            raise EvaluationStrategyError(f"{name} changed the literal value of an initial {table} identity")
    gold_new = gold_index.keys() - initial_index.keys()
    agent_new = agent_index.keys() - initial_index.keys()
    if len(gold_new) != 1:
        raise EvaluationStrategyError("new_entity_pk requires exactly one new gold entity")
    gold_id = next(iter(gold_new))
    gold_row = gold_index[gold_id]
    initial_mentions = {identifier for identifier, _ in _uuid_mentions(before)}
    if (gold_new | agent_new) & initial_mentions:
        raise EvaluationStrategyError("New identity collides with an initial identity or reference")

    gold_position = next(i for i, row in enumerate(gold_tables[table]) if row is gold_row)
    primary_path = (table, gold_position, key)
    references = [list(path) for identifier, path in _uuid_mentions(gold_tables)
                  if identifier == gold_id and path != primary_path]
    if references:
        raise EvaluationStrategyError(
            f"new_entity_pk does not support gold identity references or derived fields: {references[:5]}"
        )

    expected = {k: v for k, v in gold_row.items() if k != key}
    candidates, rejected = [], []
    for identifier in sorted(agent_new):
        if not _mapping_allowed(mapping_constraints, table, gold_row[key], agent_index[identifier][key]):
            continue
        actual = {k: v for k, v in agent_index[identifier].items() if k != key}
        if _strict_equal(expected, actual):
            candidates.append(identifier)
        else:
            fields = sorted(k for k in expected.keys() | actual.keys()
                            if k not in expected or k not in actual
                            or not _strict_equal(expected[k], actual[k]))
            rejected.append({"agent_id": agent_index[identifier][key], "differing_fields": fields})

    diagnostic = {
        "strategy": "new_entity_pk", "table": table, "primary_key": key,
        "gold_new_count": len(gold_new), "agent_new_count": len(agent_new),
        "candidate_count": len(candidates), "rejected_candidates": rejected,
        "mapping": None,
    }
    if not candidates:
        result = evaluate_gold_subset_relation(initial_state, gold_state, agent_state, diff_meta)
        result.update(passed=False, reason="No newly created entity matches all non-primary-key fields")
        result["identity_comparison"] = diagnostic
        return result

    first_failure = None
    for identifier in candidates:
        # Copy only the comparison input; the database and replay snapshots stay intact.
        aligned_gold = copy.deepcopy(gold_state)
        container = aligned_gold.entities if table in aligned_gold.entities else aligned_gold.relationships
        container[table][gold_position][key] = agent_index[identifier][key]
        result = evaluate_gold_subset_relation(initial_state, aligned_gold, agent_state, diff_meta)
        result["identity_comparison"] = dict(
            diagnostic, mapping={gold_row[key]: agent_index[identifier][key]},
        )
        if result["passed"]:
            return result
        if first_failure is None:
            first_failure = result
    return first_failure


def _validate_pk_fk_schema(config, domain_schema):
    """Resolve typed edges; table containers do not establish identity semantics."""
    entities = domain_schema.get("entities")
    if type(entities) is not dict:
        raise EvaluationStrategyError("Missing entity identity rules")
    keys = {}
    for table in config["targets"]:
        rule = entities.get(table)
        if (type(rule) is not dict or set(rule) != {"primary_key", "id_kind"}
                or type(rule["primary_key"]) is not str or not rule["primary_key"]
                or rule["id_kind"] != "uuid"):
            raise EvaluationStrategyError(f"Invalid UUID identity rule: {table}")
        keys[table] = rule["primary_key"]
    references = domain_schema.get("foreign_keys")
    if type(references) is not list:
        raise EvaluationStrategyError("new_entity_pk_fk requires typed foreign_keys")
    selected, seen = [], set()
    for rule in references:
        if (type(rule) is not dict or set(rule) != {"source_table", "field", "target_entity"}
                or any(type(value) is not str or not value for value in rule.values())):
            raise EvaluationStrategyError("Invalid typed foreign key rule")
        source, field, target = rule["source_table"], rule["field"], rule["target_entity"]
        if (source, field) in seen or target not in entities:
            raise EvaluationStrategyError("Duplicate foreign key or unknown target entity")
        seen.add((source, field))
        if target in keys:
            if source in keys and keys[source] == field:
                raise EvaluationStrategyError("An entity primary key cannot also be a foreign key")
            selected.append(dict(rule))
    if not selected:
        raise EvaluationStrategyError("new_entity_pk_fk has no declared references to its targets")
    declared = domain_schema.get("relationship_primary_keys", {})
    if type(declared) is not dict:
        raise EvaluationStrategyError("Invalid relationship_primary_keys")
    relation_keys = {}
    for table in {r["source_table"] for r in selected} - keys.keys():
        rule = declared.get(table)
        if (type(rule) is not dict
                or set(rule) != {"primary_key", "id_kind", "new_rows_only", "basis"}
                or type(rule["primary_key"]) is not str or not rule["primary_key"]
                or rule["id_kind"] != "uuid" or rule["new_rows_only"] is not True
                or rule["basis"] != "generated_key_already_ignored_by_legacy_comparator"):
            raise EvaluationStrategyError(f"Invalid generated relationship key rule: {table}")
        if any(r["source_table"] == table and r["field"] == rule["primary_key"] for r in selected):
            raise EvaluationStrategyError("A relationship key cannot also be an entity reference")
        relation_keys[table] = rule["primary_key"]
    return {"entity_keys": keys, "references": selected, "relation_keys": relation_keys}


def _index_b_entities(rows, table, key, initial_index=None):
    """Keep opaque preloaded identities fixed; only newly added identities map.

    Some tables contain legacy business keys but create new records with UUIDs.
    UUID spelling/uniqueness rules still apply; an opaque key is permitted after
    replay only if that exact key was present in the initial table.
    """
    result = {}
    for row in rows:
        value = row.get(key)
        if type(value) is not str or not value.strip():
            raise EvaluationStrategyError(f"Missing or invalid primary key: {table}.{key}")
        identifier = value.lower() if _UUID.fullmatch(value) else value
        if initial_index is not None and identifier not in initial_index:
            identifier = _uuid_key(value, f"{table}.{key}")
        if identifier in result:
            raise EvaluationStrategyError(f"Duplicate primary key: {table}.{key}={identifier}")
        result[identifier] = row
    return result


def _strict_fingerprint(value):
    """Hashable JSON value, preserving the same types and equality as _strict_equal."""
    kind = type(value)
    if kind is dict:
        if any(type(k) is not str for k in value):
            raise EvaluationStrategyError("State object keys must be strings")
        return ("dict", tuple((k, _strict_fingerprint(v)) for k, v in sorted(value.items())))
    if kind is list:
        return ("list", tuple(_strict_fingerprint(v) for v in value))
    if kind is float and not math.isfinite(value):
        raise EvaluationStrategyError("Nonfinite state number")
    if kind in (str, int, float, bool, type(None)):
        return (kind.__name__, value)
    raise EvaluationStrategyError(f"Unsupported state value type: {kind.__name__}")


_B_MAX_SEARCH_STATES = 10000


def _evaluate_new_entity_pk_fk(
    *, initial_state, gold_state, agent_state, diff_meta,
    agent_initial_state, evaluation, identity_schema,
    mapping_constraints=None,
):
    """Find one injective identity mapping covering all strictly required records.

    Only new UUID entity keys and their declared scalar foreign keys are aligned.
    Independently generated new relationship keys retain their existing exemption;
    the raw business values and multiplicity of required rows are checked first.
    """
    rule = validate_strategy_schema(evaluation, diff_meta.domain, identity_schema)
    entity_keys, relation_keys = rule["entity_keys"], rule["relation_keys"]
    references = rule["references"]
    initial, gold, agent = _tables(initial_state), _tables(gold_state), _tables(agent_state)
    if not _strict_equal(initial, _tables(agent_initial_state)):
        raise EvaluationStrategyError("Gold and agent initial databases differ")
    for table, key in relation_keys.items():
        if not (key in diff_meta.ignored_fields
                or (key.lower().endswith("_id") and key.lower() not in diff_meta.entity_id_names)):
            raise EvaluationStrategyError(f"Relationship key is not ignored by the existing comparator: {table}.{key}")

    all_keys = {**entity_keys, **relation_keys}
    before_index = {
        table: _index_b_entities(initial.get(table, []), table, key)
        for table, key in all_keys.items()
    }
    gold_index, agent_index = [
        {table: _index_b_entities(state.get(table, []), table, key, before_index[table])
         for table, key in all_keys.items()}
        for state in (gold, agent)
    ]
    new = [{table: index[table].keys() - before_index[table].keys() for table in all_keys}
           for index in (gold_index, agent_index)]
    gold_new, agent_new = new
    initial_mentions = {identifier for identifier, _ in _uuid_mentions(initial)}
    for name, index, added in (("gold", gold_index, gold_new), ("agent", agent_index, agent_new)):
        for table, key in all_keys.items():
            if table in entity_keys and before_index[table].keys() - index[table].keys():
                raise EvaluationStrategyError(f"{name} removed or renamed an initial {table} identity")
            for identifier in before_index[table].keys() & index[table].keys():
                if index[table][identifier][key] != before_index[table][identifier][key]:
                    raise EvaluationStrategyError(f"{name} changed the literal value of an initial {table} identity")
            if added[table] & initial_mentions:
                raise EvaluationStrategyError(f"New {table} identity collides with an initial identity or reference")
    if any(not gold_new[table] for table in entity_keys):
        raise EvaluationStrategyError("Every declared B target must have a new gold entity")
    for table in relation_keys:
        if before_index[table].keys() - gold_index[table].keys():
            raise EvaluationStrategyError("B does not support required deletion of initial relationship identities")

    gold_nodes = [(table, identifier) for table in sorted(entity_keys) for identifier in sorted(gold_new[table])]
    gold_exact = {table: {gold_index[table][i][key]: (table, i) for i in gold_new[table]}
                  for table, key in entity_keys.items()}
    agent_exact = {table: {agent_index[table][i][key] for i in agent_new[table]}
                   for table, key in entity_keys.items()}
    fk_by_table = defaultdict(dict)
    for edge in references:
        fk_by_table[edge["source_table"]][edge["field"]] = edge["target_entity"]

    # Prove every gold occurrence of a new identity is an exact declared position.
    allowed, own_relation_paths = defaultdict(set), defaultdict(set)
    positions = {}
    for table, key in all_keys.items():
        positions[table] = {(row[key].lower() if _UUID.fullmatch(row[key]) else row[key]): i
                            for i, row in enumerate(gold.get(table, []))}
        for identifier in gold_new[table]:
            path = (table, positions[table][identifier], key)
            (allowed if table in entity_keys else own_relation_paths)[identifier].add(path)
    for edge in references:
        table, field, target = edge["source_table"], edge["field"], edge["target_entity"]
        for i, row in enumerate(gold.get(table, [])):
            value = row.get(field)
            if type(value) is str and value in gold_exact[target]:
                allowed[value.lower()].add((table, i, field))
    for identifier, path in _uuid_mentions(gold):
        if identifier in allowed and path not in allowed[identifier]:
            raise EvaluationStrategyError(f"Undeclared new entity reference or derived field: {path}")
        if identifier in own_relation_paths and path not in own_relation_paths[identifier]:
            raise EvaluationStrategyError(f"Generated relationship key has a downstream reference: {path}")
    if any(len(paths) != 1 for paths in own_relation_paths.values()):
        raise EvaluationStrategyError("Generated relationship identity is shared across tables")

    def mapped_row(table, row, mapping, drop_key=None):
        result = {k: v for k, v in row.items() if k != drop_key}
        for field, target in fk_by_table.get(table, {}).items():
            value = result.get(field)
            node = gold_exact[target].get(value) if type(value) is str else None
            if node is not None:
                target_id = mapping[node]
                result[field] = agent_index[target][target_id][entity_keys[target]]
        return result

    candidates, rejected = {}, []
    for table, identifier in gold_nodes:
        key = entity_keys[table]
        expected = {k: v for k, v in gold_index[table][identifier].items() if k != key}
        options = []
        for actual_id in sorted(agent_new[table]):
            if not _mapping_allowed(mapping_constraints, table, gold_index[table][identifier][key],
                                    agent_index[table][actual_id][key]):
                continue
            actual = {k: v for k, v in agent_index[table][actual_id].items() if k != key}
            fields = []
            for field in expected.keys() | actual.keys():
                if field not in expected or field not in actual:
                    fields.append(field)
                    continue
                target = fk_by_table.get(table, {}).get(field)
                value = expected[field]
                if target and type(value) is str and value in gold_exact[target]:
                    if type(actual[field]) is not str or actual[field] not in agent_exact[target]:
                        fields.append(field)
                elif not _strict_equal(value, actual[field]):
                    fields.append(field)
            if not fields:
                options.append(actual_id)
            elif len(rejected) < 30:
                rejected.append({"table": table, "gold_id": gold_index[table][identifier][key],
                                 "agent_id": agent_index[table][actual_id][key], "differing_fields": sorted(fields)})
        candidates[(table, identifier)] = options

    # New rows can match only new rows. Old-row updates retain their exact key.
    required_new, required_old = [], []
    buckets = {}
    for table in fk_by_table:
        key = all_keys[table]
        for identifier, row in gold_index[table].items():
            old = before_index[table].get(identifier)
            if old is None or not _strict_equal(old, row):
                for field, target in fk_by_table[table].items():
                    value = row.get(field)
                    if value is not None and (type(value) is not str or not any(
                            r[entity_keys[target]] == value for r in gold_index[target].values())):
                        raise EvaluationStrategyError(f"Invalid required gold foreign key: {table}.{field}")
                if table not in entity_keys and old is None:
                    required_new.append((table, identifier, row))
                elif old is not None:
                    required_old.append((table, identifier, row))
        if table in relation_keys:
            grouped = defaultdict(list)
            for identifier in sorted(agent_new[table]):
                row = {k: v for k, v in agent_index[table][identifier].items() if k != key}
                grouped[_strict_fingerprint(row)].append(identifier)
            buckets[table] = grouped

    diagnostic = {"strategy": "new_entity_pk_fk", "targets": sorted(entity_keys),
                  "gold_new_counts": {t: len(gold_new[t]) for t in entity_keys},
                  "agent_new_counts": {t: len(agent_new[t]) for t in entity_keys},
                  "candidate_counts": [{"table": t, "gold_id": gold_index[t][i][entity_keys[t]],
                                        "count": len(candidates[t, i])} for t, i in gold_nodes],
                  "rejected_candidates": rejected, "mapping": None, "matched_relationships": [],
                  "required_new_relationships": len(required_new), "required_old_updates": len(required_old),
                  "search_states": 0, "constraint_failures": []}

    def reject(reason, details):
        if len(diagnostic["constraint_failures"]) < 20:
            diagnostic["constraint_failures"].append(dict(reason=reason, **details))
        return None

    def strict_witness(mapping):
        for table, identifier in gold_nodes:
            expected = mapped_row(table, gold_index[table][identifier], mapping, entity_keys[table])
            actual = {k: v for k, v in agent_index[table][mapping[table, identifier]].items()
                      if k != entity_keys[table]}
            if not _strict_equal(expected, actual):
                return reject("inconsistent_entity_foreign_key", {"table": table, "gold_id": identifier})
        matches, consumed = [], set()
        for table, identifier, row in required_old:
            expected = mapped_row(table, row, mapping)
            if not _strict_equal(expected, agent_index[table].get(identifier)):
                return reject("missing_or_incorrect_old_record_update", {"table": table, "gold_id": identifier})
            matches.append({"table": table, "gold_id": row[all_keys[table]], "agent_id": expected[all_keys[table]], "new": False})
        for table, identifier, row in required_new:
            expected = mapped_row(table, row, mapping, relation_keys[table])
            fingerprint = _strict_fingerprint(expected)
            options = buckets[table].get(fingerprint, [])
            options = [i for i in options if (table, i) not in consumed
                       and _mapping_allowed(mapping_constraints, table, row[relation_keys[table]],
                                            agent_index[table][i][relation_keys[table]])]
            if not options:
                return reject("missing_or_incorrect_required_relationship", {"table": table, "gold_id": row[relation_keys[table]]})
            actual_id = options[0]
            consumed.add((table, actual_id))
            matches.append({"table": table, "gold_id": row[relation_keys[table]],
                            "agent_id": agent_index[table][actual_id][relation_keys[table]], "new": True})
        return matches

    ordered = sorted(gold_nodes, key=lambda node: (len(candidates[node]), node))
    mapping, used = {}, set()

    def assignments(depth=0):
        diagnostic["search_states"] += 1
        if diagnostic["search_states"] > _B_MAX_SEARCH_STATES:
            raise EvaluationStrategyError("B identity mapping search budget exceeded; comparison incomplete")
        if depth == len(ordered):
            yield dict(mapping)
            return
        table, identifier = node = ordered[depth]
        for actual_id in candidates[node]:
            agent_node = (table, actual_id)
            if agent_node in used:
                continue
            mapping[node] = actual_id
            used.add(agent_node)
            yield from assignments(depth + 1)
            used.remove(agent_node)
            del mapping[node]

    first_failure = None
    if all(candidates.values()):
        for assignment in assignments():
            witness = strict_witness(assignment)
            if witness is None:
                continue
            aligned_gold = copy.deepcopy(gold_state)
            aligned = _tables(aligned_gold)
            for (table, identifier), actual_id in assignment.items():
                aligned[table][positions[table][identifier]][entity_keys[table]] = agent_index[table][actual_id][entity_keys[table]]
            for edge in references:
                table, field, target = edge["source_table"], edge["field"], edge["target_entity"]
                for row in aligned.get(table, []):
                    value = row.get(field)
                    node = gold_exact[target].get(value) if type(value) is str else None
                    if node is not None:
                        row[field] = agent_index[target][assignment[node]][entity_keys[target]]
            result = evaluate_gold_subset_relation(initial_state, aligned_gold, agent_state, diff_meta)
            result["identity_comparison"] = dict(diagnostic,
                mapping=[{"table": t, "gold_id": gold_index[t][i][entity_keys[t]],
                          "agent_id": agent_index[t][a][entity_keys[t]]} for (t, i), a in assignment.items()],
                matched_relationships=witness)
            if result["passed"]:
                return result
            if first_failure is None:
                first_failure = result
    if first_failure is not None:
        first_failure["identity_comparison"]["search_states"] = diagnostic["search_states"]
        return first_failure
    result = evaluate_gold_subset_relation(initial_state, gold_state, agent_state, diff_meta)
    result.update(passed=False, reason="No consistent identity mapping covers all required entities and relationships")
    result["identity_comparison"] = diagnostic
    return result


def _validate_custom_id_schema(config, domain_schema):
    """Compare declared prefixed or UUID identities with keyless reference rows."""
    if set(domain_schema) - {"entities", "foreign_keys", "row_matching"}:
        raise EvaluationStrategyError("Unknown custom identity schema fields")
    entities = domain_schema.get("entities")
    if type(entities) is not dict:
        raise EvaluationStrategyError("Missing custom identity rules")
    keys, patterns = {}, {}
    for table in config["targets"]:
        rule = entities.get(table)
        if type(rule) is dict and rule.get("id_kind") == "uuid":
            if (set(rule) != {"primary_key", "id_kind"}
                    or type(rule.get("primary_key")) is not str or not rule["primary_key"]):
                raise EvaluationStrategyError(f"Invalid UUID identity rule: {table}")
            patterns[table] = _UUID
            keys[table] = rule["primary_key"]
            continue
        if (type(rule) is not dict
                or set(rule) != {"primary_key", "id_kind", "prefix", "hex_digits", "letter_case"}
                or type(rule["primary_key"]) is not str or not rule["primary_key"]
                or rule["id_kind"] != "prefixed_hex"
                or type(rule["prefix"]) is not str or not rule["prefix"]
                or len(rule["prefix"]) > 64
                or type(rule["hex_digits"]) is not int or not 1 <= rule["hex_digits"] <= 64
                or rule["letter_case"] not in ("upper", "lower")):
            raise EvaluationStrategyError(f"Invalid custom identity rule: {table}")
        alphabet = "[0-9A-F]" if rule["letter_case"] == "upper" else "[0-9a-f]"
        patterns[table] = re.compile(re.escape(rule["prefix"]) + alphabet + "{" + str(rule["hex_digits"]) + "}")
        keys[table] = rule["primary_key"]
    references = domain_schema.get("foreign_keys")
    if type(references) is not list:
        raise EvaluationStrategyError("Custom identities require typed foreign_keys")
    selected, seen = [], set()
    for edge in references:
        if (type(edge) is not dict or set(edge) != {"source_table", "field", "target_entity"}
                or any(type(v) is not str or not v for v in edge.values())):
            raise EvaluationStrategyError("Invalid custom identity foreign key")
        source, field, target = edge["source_table"], edge["field"], edge["target_entity"]
        if (source, field) in seen or target not in entities:
            raise EvaluationStrategyError("Duplicate foreign key or unknown custom identity target")
        seen.add((source, field))
        if target in keys:
            if source in keys and keys[source] == field:
                raise EvaluationStrategyError("An entity primary key cannot be a foreign key")
            selected.append(dict(edge))
    if not selected:
        raise EvaluationStrategyError("Custom identity strategy requires declared references")
    row_matching = domain_schema.get("row_matching")
    if type(row_matching) is not dict:
        raise EvaluationStrategyError("Missing keyless relationship matching rules")
    relation_tables = {edge["source_table"] for edge in selected} - keys.keys()
    for table, matching in row_matching.items():
        if (type(table) is not str or not table or type(matching) is not dict
                or matching != {"mode": "multiset"}):
            raise EvaluationStrategyError("Invalid keyless relationship matching rule")
    if any(table not in row_matching for table in relation_tables):
        raise EvaluationStrategyError("Every keyless reference table requires multiset matching")
    return {"entity_keys": keys, "patterns": patterns, "references": selected,
            "relation_tables": sorted(relation_tables)}


def _index_custom_entities(rows, table, key, pattern, initial_index=None):
    result = {}
    for row in rows:
        value = row.get(key)
        if type(value) is not str or not value.strip():
            raise EvaluationStrategyError(f"Missing or invalid primary key: {table}.{key}")
        # Existing identities retain their exact spelling, regardless of format.
        if initial_index is not None and value not in initial_index and not pattern.fullmatch(value):
            raise EvaluationStrategyError(f"Invalid newly generated identity format: {table}.{key}")
        if value in result:
            raise EvaluationStrategyError(f"Duplicate primary key: {table}.{key}={value}")
        result[value] = row
    return result


def _custom_identity_mentions(value, identifiers, path=()):
    """Locate actual generated values, including embedded or key occurrences.

    This detects unsupported positions; it never performs text substitution.
    Format matching alone must not turn arbitrary business strings into IDs.
    """
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise EvaluationStrategyError("State object keys must be strings")
            for identifier in identifiers:
                if identifier in key:
                    yield identifier, path + ("<key>", key)
            yield from _custom_identity_mentions(item, identifiers, path + (key,))
    elif type(value) is list:
        for index, item in enumerate(value):
            yield from _custom_identity_mentions(item, identifiers, path + (index,))
    elif type(value) is str:
        for identifier in identifiers:
            if identifier in value:
                yield identifier, path


_C_MAX_SEARCH_STATES = 10000


def _evaluate_new_entity_pk_fk_custom_id(
    *, initial_state, gold_state, agent_state, diff_meta,
    agent_initial_state, evaluation, identity_schema,
    mapping_constraints=None,
):
    """Compare generated custom IDs and strictly required keyless relations.

    IDs map only between newly added entities. Relationships are multisets of
    complete typed rows; initial rows cannot supply a required new occurrence.
    Required deletion/update of a keyless initial row is deliberately unsupported
    until that table declares a reliable record identity.
    """
    rule = validate_strategy_schema(evaluation, diff_meta.domain, identity_schema)
    keys, patterns = rule["entity_keys"], rule["patterns"]
    references, relation_tables = rule["references"], rule["relation_tables"]
    initial, gold, agent = _tables(initial_state), _tables(gold_state), _tables(agent_state)
    if not _strict_equal(initial, _tables(agent_initial_state)):
        raise EvaluationStrategyError("Gold and agent initial databases differ")
    before = {t: _index_custom_entities(initial.get(t, []), t, k, patterns[t]) for t, k in keys.items()}
    gold_index, agent_index = [
        {t: _index_custom_entities(state.get(t, []), t, k, patterns[t], before[t]) for t, k in keys.items()}
        for state in (gold, agent)
    ]
    gold_new, agent_new = [{t: index[t].keys() - before[t].keys() for t in keys}
                           for index in (gold_index, agent_index)]
    for name, index, added in (("gold", gold_index, gold_new), ("agent", agent_index, agent_new)):
        for table in keys:
            if before[table].keys() - index[table].keys():
                raise EvaluationStrategyError(f"{name} removed or renamed an initial {table} identity")
        identifiers = set().union(*added.values())
        if next(_custom_identity_mentions(initial, identifiers), None) is not None:
            raise EvaluationStrategyError("New custom identity collides with an initial value or reference")
    if any(not gold_new[t] for t in keys):
        raise EvaluationStrategyError("Every declared C target must have a new gold entity")

    fk_by_table = defaultdict(dict)
    for edge in references:
        fk_by_table[edge["source_table"]][edge["field"]] = edge["target_entity"]
    gold_nodes = [(t, identifier) for t in sorted(keys) for identifier in sorted(gold_new[t])]
    gold_exact = {t: {identifier: (t, identifier) for identifier in gold_new[t]} for t in keys}

    # Gold is the only state rewritten. Its uses must be declared; unrelated
    # agent-only fields retain the original subset comparator's extra-op policy.
    for name, state, added in (("gold", gold, gold_new),):
        allowed = defaultdict(set)
        for table, key in keys.items():
            for i, row in enumerate(state.get(table, [])):
                if row[key] in added[table]:
                    allowed[row[key]].add((table, i, key))
        for edge in references:
            table, field, target = edge["source_table"], edge["field"], edge["target_entity"]
            for i, row in enumerate(state.get(table, [])):
                value = row.get(field)
                if type(value) is str and value in added[target]:
                    allowed[value].add((table, i, field))
        for identifier, path in _custom_identity_mentions(state, set().union(*added.values())):
            if path not in allowed[identifier]:
                raise EvaluationStrategyError(f"Undeclared {name} custom identity reference or derived field: {path}")

    def mapped_row(table, row, mapping, drop_key=None):
        result = {k: v for k, v in row.items() if k != drop_key}
        for field, target in fk_by_table.get(table, {}).items():
            value = result.get(field)
            node = gold_exact[target].get(value) if type(value) is str else None
            if node is not None:
                result[field] = mapping[node]
        return result

    candidates, rejected = {}, []
    for table, identifier in gold_nodes:
        key = keys[table]
        expected = {k: v for k, v in gold_index[table][identifier].items() if k != key}
        options = []
        for actual_id in sorted(agent_new[table]):
            if not _mapping_allowed(mapping_constraints, table, identifier, actual_id):
                continue
            actual = {k: v for k, v in agent_index[table][actual_id].items() if k != key}
            fields = []
            for field in expected.keys() | actual.keys():
                if field not in expected or field not in actual:
                    fields.append(field)
                    continue
                target, value = fk_by_table.get(table, {}).get(field), expected[field]
                if target and type(value) is str and value in gold_exact[target]:
                    if type(actual[field]) is not str or actual[field] not in agent_new[target]:
                        fields.append(field)
                elif not _strict_equal(value, actual[field]):
                    fields.append(field)
            if not fields:
                options.append(actual_id)
            elif len(rejected) < 30:
                rejected.append({"table": table, "gold_id": identifier, "agent_id": actual_id,
                                 "differing_fields": sorted(fields)})
        candidates[table, identifier] = options

    required_old = []
    for table in keys:
        for identifier, old in before[table].items():
            row = gold_index[table][identifier]
            if not _strict_equal(old, row):
                required_old.append((table, identifier, row))

    required_relations, available_relations = {}, {}
    for table in relation_tables:
        old_counts = Counter(_strict_fingerprint(row) for row in initial.get(table, []))
        gold_rows = {_strict_fingerprint(row): row for row in gold.get(table, [])}
        gold_counts = Counter(_strict_fingerprint(row) for row in gold.get(table, []))
        if old_counts - gold_counts:
            raise EvaluationStrategyError(f"C cannot identify required deletion/update of keyless initial rows: {table}")
        needed = gold_counts - old_counts
        required_relations[table] = [(gold_rows[fingerprint], count) for fingerprint, count in needed.items()]
        available_relations[table] = Counter(_strict_fingerprint(row) for row in agent.get(table, [])) - old_counts
        for row, _ in required_relations[table]:
            for field, target in fk_by_table[table].items():
                value = row.get(field)
                if type(value) is not str or value not in gold_index[target]:
                    raise EvaluationStrategyError(f"Invalid required gold foreign key: {table}.{field}")

    diagnostic = {"strategy": "new_entity_pk_fk_custom_id", "targets": sorted(keys),
                  "gold_new_counts": {t: len(gold_new[t]) for t in keys},
                  "agent_new_counts": {t: len(agent_new[t]) for t in keys},
                  "candidate_counts": [{"table": t, "gold_id": i, "count": len(candidates[t, i])} for t, i in gold_nodes],
                  "rejected_candidates": rejected, "mapping": None,
                  "required_relationship_rows": {t: sum(n for _, n in rows) for t, rows in required_relations.items()},
                  "required_old_updates": len(required_old), "constraint_failures": [], "search_states": 0}

    def strict_witness(mapping):
        for table, identifier in gold_nodes:
            if not _strict_equal(mapped_row(table, gold_index[table][identifier], mapping, keys[table]),
                                 {k: v for k, v in agent_index[table][mapping[table, identifier]].items() if k != keys[table]}):
                return "inconsistent_entity_foreign_key", {"table": table, "gold_id": identifier}
        for table, identifier, row in required_old:
            if not _strict_equal(mapped_row(table, row, mapping), agent_index[table][identifier]):
                return "missing_or_incorrect_old_record_update", {"table": table, "gold_id": identifier}
        for table, rows in required_relations.items():
            needed = Counter()
            for row, count in rows:
                needed[_strict_fingerprint(mapped_row(table, row, mapping))] += count
            missing = needed - available_relations[table]
            if missing:
                return "missing_or_incorrect_required_relationship", {"table": table, "missing_count": sum(missing.values())}
        return None

    ordered = sorted(gold_nodes, key=lambda node: (len(candidates[node]), node))
    mapping, used = {}, set()

    def assignments(depth=0):
        diagnostic["search_states"] += 1
        if diagnostic["search_states"] > _C_MAX_SEARCH_STATES:
            raise EvaluationStrategyError("C identity mapping search budget exceeded; comparison incomplete")
        if depth == len(ordered):
            yield dict(mapping)
            return
        table, identifier = node = ordered[depth]
        for actual_id in candidates[node]:
            actual_node = (table, actual_id)
            if actual_node in used:
                continue
            mapping[node] = actual_id
            used.add(actual_node)
            yield from assignments(depth + 1)
            used.remove(actual_node)
            del mapping[node]

    first_failure = None
    if all(candidates.values()):
        for assignment in assignments():
            failure = strict_witness(assignment)
            if failure is not None:
                if len(diagnostic["constraint_failures"]) < 20:
                    diagnostic["constraint_failures"].append({"reason": failure[0], **failure[1]})
                continue
            aligned_gold = copy.deepcopy(gold_state)
            aligned = _tables(aligned_gold)
            for table, key in keys.items():
                for row in aligned.get(table, []):
                    node = gold_exact[table].get(row[key])
                    if node is not None:
                        row[key] = assignment[node]
            for edge in references:
                table, field, target = edge["source_table"], edge["field"], edge["target_entity"]
                for row in aligned.get(table, []):
                    value = row.get(field)
                    node = gold_exact[target].get(value) if type(value) is str else None
                    if node is not None:
                        row[field] = assignment[node]
            result = evaluate_gold_subset_relation(initial_state, aligned_gold, agent_state, diff_meta)
            result["identity_comparison"] = dict(diagnostic,
                mapping=[{"table": t, "gold_id": i, "agent_id": a} for (t, i), a in assignment.items()])
            if result["passed"]:
                return result
            if first_failure is None:
                first_failure = result
    if first_failure is not None:
        first_failure["identity_comparison"]["search_states"] = diagnostic["search_states"]
        return first_failure
    result = evaluate_gold_subset_relation(initial_state, gold_state, agent_state, diff_meta)
    result.update(passed=False, reason="No consistent custom identity mapping covers all required entities and relationships")
    result["identity_comparison"] = diagnostic
    return result


def _validate_derived_schema(config, domain_schema):
    """Extend a domain only for D; existing category schema views stay unchanged."""
    extensions = domain_schema.get("strategy_extensions")
    extension = extensions.get(config["strategy"]) if type(extensions) is dict else None
    if (type(extension) is not dict or set(extension) != {
            "entities", "foreign_keys", "relationship_primary_keys", "derived_fields"}):
        raise EvaluationStrategyError("Missing or invalid derived identity extension")
    view = {}
    for section in ("entities", "relationship_primary_keys"):
        original, additions = domain_schema.get(section), extension[section]
        if (type(original) is not dict or type(additions) is not dict
                or original.keys() & additions.keys()
                or any(type(table) is not str or not table for table in additions)):
            raise EvaluationStrategyError(f"Invalid or overlapping derived extension: {section}")
        view[section] = copy.deepcopy({**original, **additions})
    original, additions = domain_schema.get("foreign_keys"), extension["foreign_keys"]
    if type(original) is not list or type(additions) is not list:
        raise EvaluationStrategyError("Derived extension requires typed foreign_keys")
    view["foreign_keys"] = copy.deepcopy(original + additions)
    resolved = _validate_pk_fk_schema(config, view)
    derived = extension["derived_fields"]
    if type(derived) is not list or not derived:
        raise EvaluationStrategyError("Derived strategy requires nonempty derived_fields")
    seen = set()
    references = {(edge["source_table"], edge["field"]) for edge in view["foreign_keys"]}
    for rule in derived:
        if (type(rule) is not dict or set(rule) != {
                "table", "field", "operation", "source_field", "prefix", "new_rows_only"}
                or any(type(rule[name]) is not str or not rule[name]
                       for name in ("table", "field", "operation", "source_field", "prefix"))
                or rule["operation"] != "prefix_own_primary_key"
                or rule["new_rows_only"] is not True):
            raise EvaluationStrategyError("Invalid derived field rule")
        table, field = rule["table"], rule["field"]
        if (table not in resolved["entity_keys"]
                or rule["source_field"] != resolved["entity_keys"][table]
                or field == rule["source_field"] or (table, field) in references
                or (table, field) in seen or _UUID_IN_TEXT.search(rule["prefix"])):
            raise EvaluationStrategyError(f"Unsupported or conflicting derived field: {table}.{field}")
        seen.add((table, field))
    return {**resolved, "schema_view": view, "derived_fields": copy.deepcopy(derived)}


def _evaluate_new_entity_pk_fk_derived(
    *, initial_state, gold_state, agent_state, diff_meta,
    agent_initial_state, evaluation, identity_schema,
):
    """Prove own-key derivations before using B's unchanged identity constraints.

    Only validated new-row fields are normalized on disposable copies. After B
    finds a consistent witness, rebuild complete paths on the original gold copy
    and compare against the unmodified agent state. Old paths remain literal.
    """
    rule = validate_strategy_schema(evaluation, diff_meta.domain, identity_schema)
    initial = _tables(initial_state)
    if not _strict_equal(initial, _tables(agent_initial_state)):
        raise EvaluationStrategyError("Gold and agent initial databases differ")
    normalized = [copy.deepcopy(gold_state), copy.deepcopy(agent_state)]
    tables = [_tables(state) for state in normalized]
    derived_diagnostic = []
    for derivation in rule["derived_fields"]:
        table, field = derivation["table"], derivation["field"]
        key, prefix = derivation["source_field"], derivation["prefix"]
        before = _index_b_entities(initial.get(table, []), table, key)
        counts = {}
        for side, state in zip(("gold", "agent"), tables):
            index = _index_b_entities(state.get(table, []), table, key, before)
            added = index.keys() - before.keys()
            counts[side + "_validated_new_rows"] = len(added)
            for identifier in sorted(added):
                row = index[identifier]
                expected = prefix + row[key]
                if type(row.get(field)) is not str or row[field] != expected:
                    raise EvaluationStrategyError(
                        f"{side} invalid derived field: {table}.{field} for {row[key]!r}; "
                        f"expected exact prefix + own primary key {expected!r}")
                # This literal prefix carries no generated identity. Its formula
                # has already been checked; it is never stored or returned as a
                # replacement business value.
                row[field] = prefix
        derived_diagnostic.append({**derivation, **counts})

    inner_config = {**evaluation, "strategy": "new_entity_pk_fk"}
    inner_schema = {"schema_version": 1, "domains": {diff_meta.domain: rule["schema_view"]}}
    inner = _evaluate_new_entity_pk_fk(
        initial_state=initial_state, gold_state=normalized[0], agent_state=normalized[1],
        diff_meta=diff_meta, agent_initial_state=agent_initial_state,
        evaluation=inner_config, identity_schema=inner_schema,
    )
    diagnostic = copy.deepcopy(inner["identity_comparison"])
    diagnostic.update(strategy=evaluation["strategy"], derived_fields=derived_diagnostic,
                      derived_mappings=[])
    aligned_gold = copy.deepcopy(gold_state)
    if diagnostic["mapping"] is not None:
        aligned = _tables(aligned_gold)
        mapping = {(item["table"], item["gold_id"]): item["agent_id"]
                   for item in diagnostic["mapping"]}
        for table, key in rule["entity_keys"].items():
            for row in aligned.get(table, []):
                if (table, row[key]) in mapping:
                    row[key] = mapping[table, row[key]]
        for edge in rule["references"]:
            table, field, target = edge["source_table"], edge["field"], edge["target_entity"]
            for row in aligned.get(table, []):
                value = row.get(field)
                if type(value) is str and (target, value) in mapping:
                    row[field] = mapping[target, value]
        original = _tables(gold_state)
        for derivation in rule["derived_fields"]:
            table, field = derivation["table"], derivation["field"]
            key, prefix = derivation["source_field"], derivation["prefix"]
            for old, row in zip(original.get(table, []), aligned.get(table, [])):
                if (table, old[key]) in mapping:
                    row[field] = prefix + row[key]
                    diagnostic["derived_mappings"].append({
                        "table": table, "field": field, "gold_id": old[key], "agent_id": row[key],
                        "gold_value": old[field], "aligned_gold_value": row[field],
                    })
    result = evaluate_gold_subset_relation(initial_state, aligned_gold, agent_state, diff_meta)
    if not inner["passed"]:
        # B's raw business/multiplicity checks may be stricter than the legacy
        # subset comparator. Never let a legacy-filtered result override them.
        result.update(passed=False, reason=inner["reason"])
    result["identity_comparison"] = diagnostic
    return result


def _validate_relationship_derived_schema(config, domain_schema):
    """Resolve generated relationship keys and bounded, declarative derivations."""
    extensions = domain_schema.get("strategy_extensions")
    extension = extensions.get(config["strategy"]) if type(extensions) is dict else None
    if type(extension) is not dict or set(extension) != {"relationships"}:
        raise EvaluationStrategyError("Missing or invalid relationship derivation extension")
    declared = extension["relationships"]
    if type(declared) is not dict:
        raise EvaluationStrategyError("Invalid relationship derivation tables")
    selected = {}
    valid_name = lambda value: type(value) is str and re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", value) is not None
    for table in config["targets"]:
        rule = declared.get(table)
        if (not valid_name(table) or type(rule) is not dict or set(rule) != {
                "primary_key", "id_kind", "new_rows_only", "fixed_foreign_keys", "derived_fields"}
                or not valid_name(rule["primary_key"]) or rule["id_kind"] != "uuid"
                or rule["new_rows_only"] is not True):
            raise EvaluationStrategyError(f"Invalid generated relationship rule: {table}")
        key, foreign, derived = rule["primary_key"], rule["fixed_foreign_keys"], rule["derived_fields"]
        if type(foreign) is not dict or not 1 <= len(foreign) <= 8:
            raise EvaluationStrategyError(f"Invalid fixed relationship references: {table}")
        for field, target in foreign.items():
            if (not valid_name(field) or field == key or type(target) is not dict
                    or set(target) != {"table", "primary_key"}
                    or not all(valid_name(v) for v in target.values())
                    or target["table"] in config["targets"]):
                raise EvaluationStrategyError(f"Invalid fixed foreign key: {table}.{field}")
        if type(derived) is not dict or not 1 <= len(derived) <= 8:
            raise EvaluationStrategyError(f"Invalid relationship derived fields: {table}")
        for field, expression in derived.items():
            if (not valid_name(field) or field == key or field in foreign
                    or type(expression) is not dict or set(expression) != {"operation", "parts"}
                    or expression["operation"] != "concat" or type(expression["parts"]) is not list
                    or not 1 <= len(expression["parts"]) <= 8):
                raise EvaluationStrategyError(f"Invalid relationship derivation: {table}.{field}")
            uses_key = False
            for part in expression["parts"]:
                if type(part) is not dict:
                    raise EvaluationStrategyError("Derivation parts must be objects")
                if set(part) == {"literal"}:
                    if (type(part["literal"]) is not str or not 1 <= len(part["literal"]) <= 64
                            or _UUID_IN_TEXT.search(part["literal"])):
                        raise EvaluationStrategyError("Invalid derivation literal")
                elif set(part) == {"field", "start", "stop", "upper"}:
                    if (type(part["field"]) is not str or part["field"] not in {key, *foreign}
                            or type(part["start"]) is not int or type(part["stop"]) is not int
                            or not 0 <= part["start"] < part["stop"] <= 64
                            or type(part["upper"]) is not bool
                            or (part["field"] == key and part["stop"] > 36)):
                        raise EvaluationStrategyError("Invalid derivation field or bounded slice")
                    uses_key |= part["field"] == key
                else:
                    raise EvaluationStrategyError("Unknown derivation operation")
            if not uses_key:
                raise EvaluationStrategyError("Derived fields must depend on their own generated key")
        selected[table] = copy.deepcopy(rule)
    return {"relationships": selected}


def _derive_relationship_value(row, expression, label):
    parts = []
    for part in expression["parts"]:
        if "literal" in part:
            parts.append(part["literal"])
        else:
            value = row.get(part["field"])
            if type(value) is not str or len(value) < part["stop"]:
                raise EvaluationStrategyError(f"Missing or short derivation source: {label}.{part['field']}")
            value = value[part["start"]:part["stop"]]
            parts.append(value.upper() if part["upper"] else value)
    return "".join(parts)


def _evaluate_new_relationship_pk_derived(
    *, initial_state, gold_state, agent_state, diff_meta,
    agent_initial_state, evaluation, identity_schema,
):
    """Match new relationships after proving own-key derived values on both sides.

    All other fields and fixed entity references remain exact. Equal business
    rows retain their multiplicity; account prefixes are never identity keys.
    Only a disposable gold copy is aligned before the legacy subset comparison.
    """
    rules = validate_strategy_schema(evaluation, diff_meta.domain, identity_schema)["relationships"]
    states = (initial_state, gold_state, agent_state)
    initial, gold, agent = [_tables(state) for state in states]
    if not _strict_equal(initial, _tables(agent_initial_state)):
        raise EvaluationStrategyError("Gold and agent initial databases differ")
    initial_mentions = {identifier for identifier, _ in _uuid_mentions(initial)}
    before, indexes, added = {}, [{}, {}], [{}, {}]
    allowed = [defaultdict(set), defaultdict(set)]
    owners = [defaultdict(set), defaultdict(set)]
    diagnostic = {"strategy": evaluation["strategy"], "targets": sorted(rules),
                  "gold_new_counts": {}, "agent_new_counts": {}, "formula_checks": [],
                  "mapping": None, "derived_mappings": [], "constraint_failures": [],
                  "rejected_candidates": []}
    for table, rule in rules.items():
        key = rule["primary_key"]
        if any(table in state.entities for state in states):
            raise EvaluationStrategyError(f"Declared relationship captured as an entity: {table}")
        if not (key in diff_meta.ignored_fields
                or (key.lower().endswith("_id") and key.lower() not in diff_meta.entity_id_names)):
            raise EvaluationStrategyError(f"Relationship key is not ignored by the existing comparator: {table}.{key}")
        before[table] = _index_b_entities(initial.get(table, []), table, key)
        fixed = {}
        for field, target in rule["fixed_foreign_keys"].items():
            target_table, target_key = target["table"], target["primary_key"]
            if any(target_table not in state.entities for state in states):
                raise EvaluationStrategyError(f"Fixed reference target is not a captured entity: {target_table}")
            original = _index_b_entities(initial[target_table], target_table, target_key)
            original_values = {row[target_key] for row in original.values()}
            final_values = []
            for state in (gold, agent):
                current = _index_b_entities(state[target_table], target_table, target_key, original)
                final_values.append({row[target_key] for row in current.values()})
            fixed[field] = (original_values, final_values)
        for side_number, (side, state) in enumerate((("gold", gold), ("agent", agent))):
            index = _index_b_entities(state.get(table, []), table, key, before[table])
            indexes[side_number][table] = index
            new_ids = index.keys() - before[table].keys()
            added[side_number][table] = new_ids
            diagnostic[side + "_new_counts"][table] = len(new_ids)
            # This route supports creations and old-row updates, not deleting or
            # renaming initial relationship identities to manufacture new rows.
            if before[table].keys() - index.keys():
                raise EvaluationStrategyError(f"{side} removed or renamed an initial {table} identity")
            for identifier, old in before[table].items():
                if index[identifier][key] != old[key]:
                    raise EvaluationStrategyError(f"{side} changed the literal value of an initial {table} identity")
            if new_ids & initial_mentions:
                raise EvaluationStrategyError(f"{side} new relationship key collides with an initial identity or reference")
            for position, row in enumerate(state.get(table, [])):
                identifier = row[key].lower() if _UUID.fullmatch(row[key]) else row[key]
                if identifier not in new_ids:
                    continue
                allowed[side_number][identifier].add((table, position, key))
                owners[side_number][identifier].add((table, position, key))
                for field, (original_values, final_values) in fixed.items():
                    value = row.get(field)
                    if type(value) is not str or value not in original_values or value not in final_values[side_number]:
                        raise EvaluationStrategyError(f"{side} invalid fixed foreign key: {table}.{field}")
                for field, expression in rule["derived_fields"].items():
                    expected = _derive_relationship_value(row, expression, table)
                    if type(row.get(field)) is not str or row[field] != expected:
                        raise EvaluationStrategyError(f"{side} invalid derived field: {table}.{field} for {row[key]!r}")
                    allowed[side_number][identifier].add((table, position, field))
            diagnostic["formula_checks"].append({"side": side, "table": table,
                                                 "validated_new_rows": len(new_ids),
                                                 "fields": sorted(rule["derived_fields"])})
        if not added[0][table]:
            raise EvaluationStrategyError(f"Declared relationship target has no new gold row: {table}")
    for side_number, (side, state) in enumerate((("gold", gold), ("agent", agent))):
        if any(len(paths) != 1 for paths in owners[side_number].values()):
            raise EvaluationStrategyError(f"{side} generated relationship key is shared across tables")
        for identifier, path in _uuid_mentions(state):
            if identifier in allowed[side_number] and path not in allowed[side_number][identifier]:
                raise EvaluationStrategyError(f"{side} undeclared new relationship reference or derivation: {path}")

    mapping, failures = {}, diagnostic["constraint_failures"]
    for table, rule in rules.items():
        key, derived = rule["primary_key"], rule["derived_fields"]
        excluded = {key, *derived}
        fingerprint = lambda row: _strict_fingerprint({k: v for k, v in row.items() if k not in excluded})
        buckets = defaultdict(list)
        for identifier in sorted(added[1][table]):
            buckets[fingerprint(indexes[1][table][identifier])].append(identifier)
        consumed = Counter()
        for identifier in sorted(added[0][table]):
            row = indexes[0][table][identifier]
            signature = fingerprint(row)
            options = buckets[signature]
            offset = consumed[signature]
            if offset >= len(options):
                failures.append({"reason": "missing_or_incorrect_required_relationship", "table": table, "gold_id": row[key]})
                if not options:
                    for actual_id in sorted(added[1][table]):
                        if len(diagnostic["rejected_candidates"]) >= 30:
                            break
                        actual = indexes[1][table][actual_id]
                        fields = [field for field in (row.keys() | actual.keys()) - excluded
                                  if field not in row or field not in actual
                                  or not _strict_equal(row[field], actual[field])]
                        diagnostic["rejected_candidates"].append({"table": table, "gold_id": row[key],
                            "agent_id": actual[key], "differing_fields": sorted(fields)})
            else:
                actual = indexes[1][table][options[offset]]
                mapping[table, row[key]] = actual[key]
                consumed[signature] += 1
        for identifier, old in before[table].items():
            expected = indexes[0][table][identifier]
            if not _strict_equal(old, expected) and not _strict_equal(expected, indexes[1][table][identifier]):
                failures.append({"reason": "missing_or_incorrect_old_record_update", "table": table, "gold_id": old[key]})

    aligned_gold = copy.deepcopy(gold_state)
    if not failures:
        diagnostic["mapping"] = [{"table": table, "gold_id": old, "agent_id": new}
                                  for (table, old), new in sorted(mapping.items())]
        for table, rule in rules.items():
            key = rule["primary_key"]
            for row in aligned_gold.relationships.get(table, []):
                original_id = row[key]
                if (table, original_id) not in mapping:
                    continue
                row[key] = mapping[table, original_id]
                for field, expression in rule["derived_fields"].items():
                    old_value = row[field]
                    row[field] = _derive_relationship_value(row, expression, table)
                    diagnostic["derived_mappings"].append({"table": table, "field": field,
                        "gold_id": original_id, "agent_id": row[key],
                        "gold_value": old_value, "aligned_gold_value": row[field]})
    result = evaluate_gold_subset_relation(initial_state, aligned_gold, agent_state, diff_meta)
    if failures:
        result.update(passed=False, reason="No injective relationship mapping covers all required business records")
    result["identity_comparison"] = diagnostic
    return result
_STRATEGIES = {
    "new_relationship_pk_derived": _evaluate_new_relationship_pk_derived,
    "default": _evaluate_default,
    "new_entity_pk": _evaluate_new_entity_pk,
    "new_entity_pk_fk": _evaluate_new_entity_pk_fk,
    "new_entity_pk_fk_custom_id": _evaluate_new_entity_pk_fk_custom_id,
    "new_entity_pk_fk_derived": _evaluate_new_entity_pk_fk_derived,
}


def evaluate_by_strategy(
    initial_state: DatabaseState,
    gold_state: DatabaseState,
    agent_state: DatabaseState,
    diff_meta: DiffMetadata,
    *,
    evaluation: Optional[Dict[str, Any]] = None,
    identity_schema: Optional[Dict[str, Any]] = None,
    agent_initial_state: Optional[DatabaseState] = None,
    state_contracts: Optional[Dict[str, Any]] = None,
    state_evidence: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Dispatch by trusted category, with optional evidenced state compatibility."""
    config = parse_evaluation_config(evaluation)
    if state_contracts and state_contracts.get("audited"):
        if state_contracts.get("domain") != diff_meta.domain:
            raise StateCompatibilityError("State contract domain does not match evaluation")
        if config["strategy"] not in {"default", "new_entity_pk", "new_entity_pk_fk", "new_entity_pk_fk_custom_id"}:
            raise StateCompatibilityError("Audited state contracts require a reviewed identity strategy")
        rule = (validate_strategy_schema(config, diff_meta.domain, identity_schema)
                if config["strategy"] == "new_entity_pk_fk_custom_id" else None)
        prepared = PreparedAuditedStateCompatibility(
            contracts=state_contracts, evidence=state_evidence,
            initial_state=initial_state, gold_state=gold_state, agent_state=agent_state,
            agent_initial_state=agent_initial_state, identity_schema=identity_schema,
        )
        mirror_specs = [entry["spec"] for entry in prepared.contracts["audited"]
                        if entry["spec"].get("kind") == "auxiliary_parameter_mirror"]
        if mirror_specs and config["strategy"] != "new_entity_pk_fk_custom_id":
            raise StateCompatibilityError("Parameter mirrors require a reviewed custom identity strategy")
        if config["strategy"] == "new_entity_pk_fk_custom_id":
            for entry in prepared.contracts["audited"]:
                spec = entry["spec"]
                if (spec["table"] not in config["targets"]
                        or spec["primary_key"] != rule["entity_keys"].get(spec["table"])
                        or entry["gold_id"] in prepared.initial[spec["table"]]):
                    raise StateCompatibilityError("C state projection must bind a declared entity target and key")
                if spec["kind"] == "auxiliary_parameter_mirror":
                    source = spec["relation_source"]
                    edge = {"source_table": source["table"], "field": source["target_foreign_key"],
                            "target_entity": spec["table"]}
                    if source["table"] not in rule["relation_tables"] or edge not in rule["references"]:
                        raise StateCompatibilityError("Parameter mirror source must use a declared C relationship foreign key")
        first_failure = None
        for projected, pairs in prepared.variants():
            constraints = {(p["table"], p["gold_id"]): p["agent_id"] for p in pairs}
            result = _STRATEGIES[config["strategy"]](
                initial_state=initial_state, gold_state=gold_state, agent_state=projected,
                diff_meta=diff_meta, evaluation=config, identity_schema=identity_schema,
                agent_initial_state=agent_initial_state, mapping_constraints=constraints,
            )
            result = prepared.finish(result, config, pairs)
            if result.get("passed") is True:
                return result
            if first_failure is None:
                first_failure = result
        if first_failure is not None:
            first_failure["state_compatibility"]["binding_search_states"] = prepared.diagnostic["binding_search_states"]
            return first_failure
        result = _STRATEGIES[config["strategy"]](
            initial_state=initial_state, gold_state=gold_state, agent_state=agent_state,
            diff_meta=diff_meta, evaluation=config, identity_schema=identity_schema,
            agent_initial_state=agent_initial_state,
        )
        return prepared.finish(result, config, [])
    compatibility = None
    if state_contracts:
        if state_contracts.get("domain") != diff_meta.domain:
            raise StateCompatibilityError("State contract domain does not match evaluation")
        compatibility = PreparedStateCompatibility(
            contracts=state_contracts, evidence=state_evidence,
            initial_state=initial_state, gold_state=gold_state, agent_state=agent_state,
            agent_initial_state=agent_initial_state, identity_schema=identity_schema,
        )
    result = _STRATEGIES[config["strategy"]](
        initial_state=initial_state, gold_state=gold_state,
        agent_state=compatibility.agent_state if compatibility else agent_state,
        diff_meta=diff_meta, evaluation=config, identity_schema=identity_schema,
        agent_initial_state=agent_initial_state,
    )
    return compatibility.finish(result, config) if compatibility else result
