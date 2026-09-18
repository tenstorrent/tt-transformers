#!/usr/bin/env python3
"""Validate runner evidence and emit a deterministic canonical evidence index."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import string
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qualification.tools import run_hardware_matrix as runner

DEFAULT_MATRIX = ROOT / "tests/hardware/hardware-matrix.json"
DEFAULT_SCHEMA = ROOT / "qualification/schemas/hardware-evidence.schema.json"
FINAL_CLASSIFICATIONS = {
    "passed",
    "functional_failure",
    "hardware_lifecycle_failure",
    "missing_acceptance_data",
    "no_passing_tests",
}
OUTCOMES = (
    "passed",
    "pre_device_failure",
    "functional_failure",
    "hardware_lifecycle_failure",
    "missing_acceptance_data",
)
PRE_DEVICE_PATTERNS = (
    re.compile(r"^ERROR collecting ", re.MULTILINE),
    re.compile(r"^ERROR: not found:", re.MULTILINE),
    re.compile(r"collected 0 items(?:\s*/\s*\d+ errors?)?", re.IGNORECASE),
    re.compile(r"interrupted:\s*\d+ errors? during collection", re.IGNORECASE),
    re.compile(r"no tests ran", re.IGNORECASE),
)
SHA1_PATTERN = re.compile(r"^[0-9a-f]{40}$")


class EvidenceError(ValueError):
    """Hardware evidence is incomplete, contradictory, or not attributable."""


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EvidenceError(f"cannot read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise EvidenceError(f"JSON root must be an object: {path}")
    return value


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def file_identity(path: Path, root: Path) -> dict[str, Any]:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(root.resolve()).as_posix()
    except ValueError as error:
        raise EvidenceError(f"evidence file escapes evidence root: {path}") from error
    size = resolved.stat().st_size
    if size <= 0:
        raise EvidenceError(f"evidence file is empty: {path}")
    return {"path": relative, "sha256": sha256(resolved), "size_bytes": size}


def parse_utc(value: Any, *, field: str, path: Path) -> dt.datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise EvidenceError(f"{path}: {field} must be a UTC ISO-8601 string ending in Z")
    try:
        parsed = dt.datetime.fromisoformat(value.removesuffix("Z") + "+00:00")
    except ValueError as error:
        raise EvidenceError(f"{path}: invalid {field}: {value!r}") from error
    return parsed


def matrix_path_identity(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def discover_records(evidence_root: Path) -> list[tuple[Path, dict[str, Any]]]:
    if not evidence_root.is_dir():
        raise EvidenceError(f"evidence root is not a directory: {evidence_root}")
    records = []
    for path in sorted(evidence_root.rglob("*.json")):
        value = load_json(path)
        # Physical-inventory and generated-index JSON may share the directory.
        # A runner record is unambiguously identified by its node field.
        if "node" in value:
            records.append((path, value))
    if not records:
        raise EvidenceError(f"no runner evidence JSON found under {evidence_root}")
    return records


def resolve_machine(matrix: dict[str, Any], node: dict[str, Any], declared: Any, path: Path) -> str:
    if not isinstance(declared, str) or not declared:
        raise EvidenceError(f"{path}: machine_class must be non-empty")
    machines = matrix.get("machines")
    if not isinstance(machines, dict):
        raise EvidenceError("matrix machines must be an object")
    if declared not in machines:
        raise EvidenceError(f"{path}: unknown machine class {declared!r}")
    if declared not in node.get("machine_pool", []):
        raise EvidenceError(f"{path}: machine class {declared!r} is not in the pool for node {node.get('id')!r}")
    return declared


def template_matches(template: Any, actual: Any, *, node: dict[str, Any]) -> bool:
    if template is None:
        return actual is None
    if not isinstance(template, str) or not isinstance(actual, str):
        return template == actual
    pieces = []
    for literal, field_name, format_spec, conversion in string.Formatter().parse(template):
        pieces.append(re.escape(literal))
        if field_name is None:
            continue
        if format_spec or conversion or field_name not in {"checkout", "run_root", "hf_home", "node_id", "mesh_device"}:
            return False
        if field_name in {"checkout", "run_root", "hf_home"}:
            pieces.append(r".+")
        elif field_name == "node_id":
            pieces.append(re.escape(node["id"]))
        else:
            pieces.append(re.escape(node["mesh_device"]))
    return re.fullmatch("".join(pieces), actual) is not None


def validate_environment(
    node: dict[str, Any],
    machine: dict[str, Any],
    evidence: dict[str, Any],
    *,
    path: Path,
) -> None:
    expected = dict(node.get("environment", {}))
    # Mirrors runner.environment_for: the class declares the device-selection
    # policy, and the recorded inventory supplies the caller's own BDFs.
    policy = machine.get("tt_visible_devices")
    if policy == "must_be_unset":
        expected["TT_VISIBLE_DEVICES"] = None
    elif policy == "from_selected_bdfs":
        selected = (evidence.get("physical_inventory") or {}).get("selected_bdfs") or []
        expected["TT_VISIBLE_DEVICES"] = ",".join(selected) if selected else None
    cache = node.get("cache_requirement", {})
    if cache.get("kind") == "writable_node_local":
        expected["TT_CACHE_PATH"] = cache.get("path")
    actual = evidence.get("environment")
    if not isinstance(actual, dict) or set(actual) != set(expected):
        actual_keys = sorted(actual) if isinstance(actual, dict) else actual
        raise EvidenceError(
            f"{path}: environment keys differ from matrix: expected={sorted(expected)} actual={actual_keys}"
        )
    for key, template in expected.items():
        if not template_matches(template, actual[key], node=node):
            raise EvidenceError(
                f"{path}: environment {key} does not match matrix template {template!r}: {actual[key]!r}"
            )
    if evidence.get("tt_visible_devices") != actual.get("TT_VISIBLE_DEVICES"):
        raise EvidenceError(f"{path}: tt_visible_devices differs from recorded environment")
    cache_paths = evidence.get("cache_paths")
    if not isinstance(cache_paths, dict) or cache_paths.get("TT_CACHE_PATH") != actual.get("TT_CACHE_PATH"):
        raise EvidenceError(f"{path}: cache_paths TT_CACHE_PATH differs from recorded environment")


def validate_command(node: dict[str, Any], evidence: dict[str, Any], *, path: Path) -> None:
    command = evidence.get("command")
    if not isinstance(command, list) or not command or not all(isinstance(value, str) for value in command):
        raise EvidenceError(f"{path}: command must be a non-empty string array")
    expected = runner.build_command(node, command[0])
    if command != expected:
        raise EvidenceError(f"{path}: command differs from matrix selector/timeout")


def validate_final_result(evidence: dict[str, Any], *, path: Path) -> None:
    classification = evidence.get("failure_classification")
    if classification not in FINAL_CLASSIFICATIONS:
        raise EvidenceError(f"{path}: non-final failure_classification {classification!r}")
    exit_code = evidence.get("exit_code")
    if isinstance(exit_code, bool) or not isinstance(exit_code, int):
        raise EvidenceError(f"{path}: final exit_code must be an integer")
    if classification == "passed" and exit_code != 0:
        raise EvidenceError(f"{path}: passed evidence must have exit_code 0")
    if classification in {"missing_acceptance_data", "no_passing_tests"} and exit_code != 0:
        raise EvidenceError(f"{path}: {classification} must have exit_code 0")
    if classification in {"functional_failure", "hardware_lifecycle_failure"} and exit_code == 0:
        raise EvidenceError(f"{path}: failed evidence cannot have exit_code 0")
    teardown = evidence.get("teardown_status")
    if not isinstance(teardown, str) or not teardown or teardown == "not_started":
        raise EvidenceError(f"{path}: final teardown_status is missing")
    reset = evidence.get("reset")
    if (
        not isinstance(reset, dict)
        or not isinstance(reset.get("performed"), bool)
        or reset.get("automatic") is not False
        or not isinstance(reset.get("reason"), str)
        or not reset["reason"]
    ):
        raise EvidenceError(f"{path}: reset record is incomplete or permits automatic reset")
    metrics = evidence.get("metrics")
    if not isinstance(metrics, list) or not all(isinstance(value, str) for value in metrics):
        raise EvidenceError(f"{path}: metrics must be a string array")


def classify_outcome(evidence: dict[str, Any], log_text: str) -> str:
    classification = evidence["failure_classification"]
    if classification == "no_passing_tests":
        return "pre_device_failure"
    if classification == "functional_failure" and any(pattern.search(log_text) for pattern in PRE_DEVICE_PATTERNS):
        return "pre_device_failure"
    return classification


def validate_log_result(evidence: dict[str, Any], log_text: str, *, path: Path) -> None:
    pass_count = runner.pytest_pass_count(log_text)
    classification = evidence["failure_classification"]
    if classification in {"passed", "missing_acceptance_data"} and (pass_count is None or pass_count <= 0):
        raise EvidenceError(f"{path}: {classification} record has no positive pytest pass summary")
    if classification == "no_passing_tests" and pass_count not in {None, 0}:
        raise EvidenceError(f"{path}: no_passing_tests record contains {pass_count} passing tests")


def validate_record(
    *,
    evidence_path: Path,
    evidence: dict[str, Any],
    evidence_root: Path,
    candidate_sha: str,
    matrix: dict[str, Any],
    nodes: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    required = matrix.get("required_evidence_fields")
    if not isinstance(required, list) or not required or not all(isinstance(value, str) for value in required):
        raise EvidenceError("matrix required_evidence_fields must be a non-empty string array")
    missing = (set(required) | {"schema_version", "node", "command"}) - set(evidence)
    if missing:
        raise EvidenceError(f"{evidence_path}: missing evidence fields: {sorted(missing)}")
    if evidence.get("schema_version") != 1:
        raise EvidenceError(f"{evidence_path}: unsupported evidence schema_version")
    node_id = evidence.get("node")
    if node_id not in nodes:
        raise EvidenceError(f"{evidence_path}: unknown matrix node {node_id!r}")
    node = nodes[node_id]
    if evidence.get("full_sha") != candidate_sha:
        raise EvidenceError(
            f"{evidence_path}: full_sha {evidence.get('full_sha')!r} does not match candidate {candidate_sha}"
        )
    if not isinstance(evidence.get("branch"), str) or not evidence["branch"]:
        raise EvidenceError(f"{evidence_path}: branch must be non-empty")
    started = parse_utc(evidence.get("started_utc"), field="started_utc", path=evidence_path)
    finished = parse_utc(evidence.get("finished_utc"), field="finished_utc", path=evidence_path)
    if finished < started:
        raise EvidenceError(f"{evidence_path}: finished_utc precedes started_utc")
    for field in ("architecture", "mesh_device", "selector", "acceptance_types"):
        if evidence.get(field) != node.get(field):
            raise EvidenceError(f"{evidence_path}: {field} differs from matrix node {node_id}")
    if not isinstance(evidence.get("acceptance_types"), list) or not evidence["acceptance_types"]:
        raise EvidenceError(f"{evidence_path}: acceptance_types must be non-empty")

    machine_name = resolve_machine(matrix, node, evidence.get("machine_class"), evidence_path)
    inventory = evidence.get("physical_inventory")
    if not isinstance(inventory, dict):
        raise EvidenceError(f"{evidence_path}: physical_inventory must be an object")
    try:
        runner.validate_physical_inventory(
            matrix,
            node,
            machine_name,
            inventory,
        )
    except runner.MatrixError as error:
        raise EvidenceError(f"{evidence_path}: invalid physical inventory: {error}") from error
    validate_environment(node, matrix["machines"][machine_name], evidence, path=evidence_path)
    validate_command(node, evidence, path=evidence_path)
    validate_final_result(evidence, path=evidence_path)

    log_path = evidence_path.with_suffix(".log")
    if not log_path.is_file():
        raise EvidenceError(f"{evidence_path}: paired stdout log is missing: {log_path.name}")
    if Path(str(evidence["stdout_log_path"])).name != log_path.name:
        raise EvidenceError(f"{evidence_path}: stdout_log_path does not name paired log {log_path.name}")
    if Path(str(evidence["evidence_json_path"])).name != evidence_path.name:
        raise EvidenceError(f"{evidence_path}: evidence_json_path does not name its JSON file")
    evidence_identity = file_identity(evidence_path, evidence_root)
    log_identity = file_identity(log_path, evidence_root)
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    validate_log_result(evidence, log_text, path=evidence_path)
    outcome = classify_outcome(evidence, log_text)
    pair_hash = hashlib.sha256(
        (evidence_identity["sha256"] + "\0" + log_identity["sha256"]).encode("ascii")
    ).hexdigest()
    return {
        "node": node_id,
        "priority": node["priority"],
        "stage": node["stage"],
        "architecture": node["architecture"],
        "mesh_device": node["mesh_device"],
        "machine_pool_entry": machine_name,
        "branch": evidence["branch"],
        "full_sha": evidence["full_sha"],
        "started_utc": evidence["started_utc"],
        "finished_utc": evidence["finished_utc"],
        "exit_code": evidence["exit_code"],
        "runner_classification": evidence["failure_classification"],
        "outcome": outcome,
        "acceptance_types": evidence["acceptance_types"],
        "metrics_count": len(evidence["metrics"]),
        "teardown_status": evidence["teardown_status"],
        "reset": evidence["reset"],
        "source": {
            "evidence_json": evidence_identity,
            "stdout_log": log_identity,
            "pair_sha256": pair_hash,
        },
    }


def outcome_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(record["outcome"] for record in records)
    return {outcome: counts[outcome] for outcome in OUTCOMES}


def group_summary(records: list[dict[str, Any]], field: str) -> list[dict[str, Any]]:
    groups = []
    for name in sorted({record[field] for record in records}):
        selected = [record for record in records if record[field] == name]
        groups.append({"name": name, "total": len(selected), "outcomes": outcome_counts(selected)})
    return groups


def validate_canonical_schema(index: dict[str, Any], schema_path: Path) -> None:
    schema = load_json(schema_path)
    try:
        Draft202012Validator.check_schema(schema)
    except Exception as error:
        raise EvidenceError(f"canonical evidence schema is invalid: {error}") from error
    errors = sorted(
        Draft202012Validator(schema).iter_errors(index),
        key=lambda error: tuple(str(part) for part in error.absolute_path),
    )
    if errors:
        details = "; ".join(
            f"{'/'.join(map(str, error.absolute_path)) or '<root>'}: {error.message}" for error in errors
        )
        raise EvidenceError(f"canonical evidence schema validation failed: {details}")


def build_index(
    *,
    candidate_sha: str,
    matrix_path: Path,
    evidence_root: Path,
    schema_path: Path = DEFAULT_SCHEMA,
) -> dict[str, Any]:
    if SHA1_PATTERN.fullmatch(candidate_sha) is None:
        raise EvidenceError("candidate SHA must be 40 lowercase hexadecimal characters")
    matrix = load_json(matrix_path)
    if not isinstance(matrix.get("schema_version"), int):
        raise EvidenceError("matrix schema_version must be an integer")
    matrix_nodes = matrix.get("nodes")
    if not isinstance(matrix_nodes, list) or not matrix_nodes:
        raise EvidenceError("matrix nodes must be a non-empty array")
    nodes: dict[str, dict[str, Any]] = {}
    for node in matrix_nodes:
        if not isinstance(node, dict) or not isinstance(node.get("id"), str) or not node["id"]:
            raise EvidenceError("matrix contains an invalid node")
        if node["id"] in nodes:
            raise EvidenceError(f"matrix contains duplicate node {node['id']}")
        nodes[node["id"]] = node

    records = []
    seen_nodes: dict[str, Path] = {}
    for evidence_path, evidence in discover_records(evidence_root):
        node_id = evidence.get("node")
        if isinstance(node_id, str) and node_id in seen_nodes:
            raise EvidenceError(f"duplicate evidence for node {node_id}: {seen_nodes[node_id]} and {evidence_path}")
        if isinstance(node_id, str):
            seen_nodes[node_id] = evidence_path
        records.append(
            validate_record(
                evidence_path=evidence_path,
                evidence=evidence,
                evidence_root=evidence_root,
                candidate_sha=candidate_sha,
                matrix=matrix,
                nodes=nodes,
            )
        )
    records.sort(key=lambda record: (record["priority"], record["node"]))
    index = {
        "schema_version": 1,
        "candidate_sha": candidate_sha,
        "matrix": {
            "path": matrix_path_identity(matrix_path),
            "schema_version": matrix["schema_version"],
            "sha256": sha256(matrix_path),
            "node_count": len(matrix_nodes),
        },
        "summary": {
            "total": len(records),
            "outcomes": outcome_counts(records),
            "by_stage": group_summary(records, "stage"),
            "by_architecture": group_summary(records, "architecture"),
            "by_mesh_device": group_summary(records, "mesh_device"),
        },
        "records": records,
    }
    validate_canonical_schema(index, schema_path)
    return index


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--candidate-sha", required=True)
    result.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    result.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    result.add_argument("--evidence-root", type=Path, required=True)
    result.add_argument("--output", type=Path)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        index = build_index(
            candidate_sha=args.candidate_sha,
            matrix_path=args.matrix,
            evidence_root=args.evidence_root,
            schema_path=args.schema,
        )
    except EvidenceError as error:
        print(f"hardware evidence validation FAILED: {error}", file=sys.stderr)
        return 1
    rendered = json.dumps(index, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        print(
            f"Validated {index['summary']['total']} hardware evidence records for "
            f"{index['candidate_sha']}; wrote {args.output}"
        )
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
