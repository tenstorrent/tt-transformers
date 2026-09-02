#!/usr/bin/env python3
"""Validate, preview, or serially execute one hardware-matrix node."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import signal
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MATRIX = ROOT / "qualification/manifests/hardware-matrix.json"
ALLOWED_STAGES = {"module", "runtime", "smoke", "e2e"}
ALLOWED_ARCHITECTURES = {"wormhole", "blackhole"}
ALLOWED_MESH_DEVICES = {"N150", "N300", "T3K", "P150", "P150x4"}
PARALLEL_ARGUMENTS = {"-n", "--numprocesses", "--dist", "--tx"}
HARDWARE_FAILURE_PATTERNS = (
    r"watcher.*(?:error|fatal)",
    r"(?:device|card).*(?:hang|unhealthy|unresponsive)",
    r"(?:firmware|driver).*(?:mismatch|error|fatal)",
    r"metal.*(?:fatal|timeout)",
    r"reset required",
)
METRIC_PATTERN = re.compile(
    r"(?:\bPCC\b|top[- ]?[15]|cache|TTFT|TPOT|tok(?:ens)?/s|throughput)",
    re.IGNORECASE,
)


class MatrixError(RuntimeError):
    """Refusal or validation failure before a hardware process starts."""


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise MatrixError(f"cannot read JSON {path}: {error}") from error


def selector_argv(node: dict[str, Any]) -> list[str]:
    selector = node["selector"]
    argv = [selector["target"]]
    if selector.get("keyword"):
        argv.extend(["-k", selector["keyword"]])
    return argv


def contains_parallel_pytest_args(argv: list[str]) -> bool:
    for index, argument in enumerate(argv):
        if argument in PARALLEL_ARGUMENTS:
            return True
        if argument.startswith(("--numprocesses=", "--dist=", "-n=")):
            return True
        if argument == "-n" and index + 1 < len(argv):
            return True
    return False


def validate_matrix(matrix: dict[str, Any], root: Path = ROOT) -> dict[str, int]:
    if matrix.get("schema_version") != 1:
        raise MatrixError("hardware matrix schema_version must be 1")
    serialization = matrix.get("serialization", {})
    if serialization.get("max_concurrent_processes") != 1:
        raise MatrixError("hardware matrix must allow exactly one concurrent process")
    if serialization.get("automatic_reset") is not False:
        raise MatrixError("automatic reset must be disabled")

    required_evidence = matrix.get("required_evidence_fields")
    if not isinstance(required_evidence, list) or not required_evidence:
        raise MatrixError("required_evidence_fields must be a non-empty list")
    machines = matrix.get("machines")
    nodes = matrix.get("nodes")
    if not isinstance(machines, dict) or not isinstance(nodes, list) or not nodes:
        raise MatrixError("matrix requires machines and nodes")

    ids: set[str] = set()
    priorities: set[int] = set()
    mesh_counts: dict[str, int] = {name: 0 for name in ALLOWED_MESH_DEVICES}
    for node in nodes:
        node_id = node.get("id")
        if not isinstance(node_id, str) or not node_id or node_id in ids:
            raise MatrixError(f"duplicate/invalid node id: {node_id!r}")
        ids.add(node_id)
        priority = node.get("priority")
        if not isinstance(priority, int) or priority <= 0 or priority in priorities:
            raise MatrixError(f"duplicate/invalid priority for {node_id}: {priority!r}")
        priorities.add(priority)
        if node.get("stage") not in ALLOWED_STAGES:
            raise MatrixError(f"invalid stage for {node_id}")
        if node.get("architecture") not in ALLOWED_ARCHITECTURES:
            raise MatrixError(f"invalid architecture for {node_id}")
        mesh = node.get("mesh_device")
        if mesh not in ALLOWED_MESH_DEVICES:
            raise MatrixError(f"invalid MESH_DEVICE for {node_id}: {mesh!r}")
        mesh_counts[mesh] += 1
        if node.get("environment", {}).get("MESH_DEVICE") != mesh:
            raise MatrixError(f"environment MESH_DEVICE mismatch for {node_id}")
        if not 60 <= node.get("timeout_seconds", 0) <= 7200:
            raise MatrixError(f"invalid timeout for {node_id}")
        if not node.get("acceptance_types"):
            raise MatrixError(f"missing acceptance_types for {node_id}")
        if node.get("required_evidence_fields") != required_evidence:
            raise MatrixError(f"node evidence schema drift for {node_id}")
        pools = node.get("machine_pool")
        if not pools or any(machine not in machines for machine in pools):
            raise MatrixError(f"invalid machine pool for {node_id}")
        if any(machines[machine]["architecture"] != node["architecture"] for machine in pools):
            raise MatrixError(f"machine architecture mismatch for {node_id}")
        argv = selector_argv(node)
        if contains_parallel_pytest_args(argv):
            raise MatrixError(f"parallel pytest option forbidden for {node_id}")
        target = node["selector"].get("target", "")
        test_path, separator, test_name = target.partition("::")
        path = root / test_path
        if not path.is_file():
            raise MatrixError(f"selector path does not exist for {node_id}: {test_path}")
        if separator:
            function = test_name.split("[", 1)[0]
            if f"def {function}(" not in path.read_text(encoding="utf-8"):
                raise MatrixError(f"selector function not found for {node_id}: {function}")
    if any(count == 0 for count in mesh_counts.values()):
        raise MatrixError(f"matrix does not cover every required mesh: {mesh_counts}")
    return mesh_counts


def select_node(matrix: dict[str, Any], node_id: str) -> dict[str, Any]:
    matches = [node for node in matrix["nodes"] if node["id"] == node_id]
    if len(matches) != 1:
        raise MatrixError(f"unknown or duplicate node id: {node_id}")
    node = matches[0]
    if not node.get("enabled", True):
        raise MatrixError(
            f"node {node_id} is disabled: {node.get('disabled_classification') or 'unimplemented_gate'}"
        )
    return node


def validate_attestation(
    matrix: dict[str, Any],
    node: dict[str, Any],
    *,
    common_sha: str,
    branch: str,
    machine_identity: str,
    sync_gate_passed: bool,
) -> str:
    if not sync_gate_passed:
        raise MatrixError("external synchronized-checkout gate was not attested")
    if not re.fullmatch(r"[0-9a-f]{40}", common_sha):
        raise MatrixError("common SHA must be 40 lowercase hexadecimal characters")
    if not branch.strip():
        raise MatrixError("branch identity is required")
    candidate_machines = node["machine_pool"]
    matched = [
        name
        for name in candidate_machines
        if machine_identity in matrix["machines"][name]["allowed_identities"]
    ]
    if len(matched) != 1:
        raise MatrixError(
            f"machine identity {machine_identity!r} is not valid for node {node['id']}"
        )
    return matched[0]


def validate_physical_inventory(
    matrix: dict[str, Any],
    node: dict[str, Any],
    machine_name: str,
    machine_identity: str,
    inventory: dict[str, Any],
) -> None:
    fields = {
        "captured_utc",
        "machine_identity",
        "architecture",
        "physical_sku",
        "device_count",
        "board_types",
        "cluster_type",
        "system_mesh",
        "tt_visible_devices",
        "source_command",
    }
    missing = fields - set(inventory)
    if missing:
        raise MatrixError(f"physical inventory lacks fields: {sorted(missing)}")
    if inventory["machine_identity"] != machine_identity:
        raise MatrixError("physical inventory machine identity does not match caller")
    if inventory["architecture"] != node["architecture"]:
        raise MatrixError("physical inventory architecture does not match node")
    if not isinstance(inventory["device_count"], int) or inventory["device_count"] <= 0:
        raise MatrixError("physical inventory device_count must be positive")
    if not isinstance(inventory["board_types"], list) or not inventory["board_types"]:
        raise MatrixError("physical inventory board_types must be non-empty")
    if inventory["source_command"] != "tt-smi -s":
        raise MatrixError("physical inventory must identify the external tt-smi -s source")
    try:
        dt.datetime.fromisoformat(inventory["captured_utc"].replace("Z", "+00:00"))
    except (AttributeError, ValueError) as error:
        raise MatrixError("physical inventory captured_utc must be ISO-8601") from error

    machine = matrix["machines"][machine_name]
    expected_inventory = machine.get("expected_inventory", {})
    expected_count = expected_inventory.get("device_count")
    if expected_count is not None and inventory["device_count"] != expected_count:
        raise MatrixError(
            f"physical device count mismatch: {inventory['device_count']} != {expected_count}"
        )
    observed_boards = {str(value).lower() for value in inventory["board_types"]}
    expected_boards = {
        str(value).lower() for value in expected_inventory.get("board_types", [])
    }
    if not expected_boards <= observed_boards:
        raise MatrixError(
            f"physical board types mismatch: missing {sorted(expected_boards - observed_boards)}"
        )
    if node["mesh_device"] == "P150x4":
        allowed_clusters = node["physical_sku_provenance"]["required_cluster_types"]
        if inventory["cluster_type"] not in allowed_clusters:
            raise MatrixError(
                f"P150x4 requires physical cluster provenance in {allowed_clusters}"
            )
        if machine_name == "bh-lb-11":
            required = {
                "0000:01:00.0",
                "0000:41:00.0",
                "0000:81:00.0",
                "0000:c1:00.0",
            }
            if set(inventory.get("selected_bdfs", [])) != required:
                raise MatrixError("bh-lb-11 P150x4 requires the revalidated four-BDF Ring")
        elif inventory["tt_visible_devices"] not in {None, ""}:
            raise MatrixError(f"{machine_name} full-host P150x4 requires TT_VISIBLE_DEVICES unset")


def expand(value: str, *, checkout: Path, run_root: Path, node: dict[str, Any]) -> str:
    return (
        value.replace("{checkout}", str(checkout))
        .replace("{run_root}", str(run_root))
        .replace("{node_id}", node["id"])
        .replace("{mesh_device}", node["mesh_device"])
    )


def environment_for(
    node: dict[str, Any],
    machine_name: str,
    *,
    checkout: Path,
    run_root: Path,
) -> dict[str, str | None]:
    values: dict[str, str | None] = dict(node["environment"])
    values.update(node.get("machine_environment_overrides", {}).get(machine_name, {}))
    cache = node["cache_requirement"]
    if cache["kind"] == "writable_node_local":
        values["TT_CACHE_PATH"] = cache["path"]
    return {
        key: expand(value, checkout=checkout, run_root=run_root, node=node)
        if isinstance(value, str)
        else None
        for key, value in values.items()
    }


def build_command(node: dict[str, Any], python: str) -> list[str]:
    command = [
        python,
        "-m",
        "pytest",
        "-vv",
        f"--timeout={node['timeout_seconds']}",
        *selector_argv(node),
    ]
    if contains_parallel_pytest_args(command):
        raise MatrixError("parallel pytest execution is forbidden")
    return command


def refuse_parallel_environment(environ: dict[str, str]) -> None:
    if environ.get("PYTEST_XDIST_WORKER"):
        raise MatrixError("existing pytest-xdist worker environment is forbidden")
    addopts = environ.get("PYTEST_ADDOPTS", "").split()
    if contains_parallel_pytest_args(addopts):
        raise MatrixError("parallel option in PYTEST_ADDOPTS is forbidden")


class ProcessLock:
    def __init__(self, path: Path):
        self.path = path
        self.fd: int | None = None

    def __enter__(self):
        try:
            self.fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError as error:
            raise MatrixError(f"hardware runner lock already exists: {self.path}") from error
        os.write(self.fd, f"{os.getpid()}\n".encode())
        return self

    def __exit__(self, exc_type, exc, traceback):
        if self.fd is not None:
            os.close(self.fd)
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass


def classify_failure(exit_code: int | None, output: str, timed_out: bool = False) -> str:
    if timed_out:
        return "hardware_lifecycle_failure"
    if exit_code == 0:
        return "passed"
    if any(re.search(pattern, output, re.IGNORECASE | re.DOTALL) for pattern in HARDWARE_FAILURE_PATTERNS):
        return "hardware_lifecycle_failure"
    return "functional_failure"


def collect_metrics(output: str) -> list[str]:
    return [line for line in output.splitlines() if METRIC_PATTERN.search(line)][:200]


def local_checkout_identity(checkout: Path) -> tuple[str, str]:
    try:
        branch = subprocess.check_output(
            ["git", "-C", str(checkout), "branch", "--show-current"], text=True
        ).strip()
        sha = subprocess.check_output(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
        ).strip()
    except subprocess.CalledProcessError as error:
        raise MatrixError(f"cannot read local checkout identity: {error}") from error
    return branch, sha


def preview(
    matrix: dict[str, Any],
    node: dict[str, Any],
    args: argparse.Namespace,
    inventory: dict[str, Any],
) -> dict[str, Any]:
    machine_name = validate_attestation(
        matrix,
        node,
        common_sha=args.common_sha,
        branch=args.branch,
        machine_identity=args.machine_identity,
        sync_gate_passed=args.sync_gate_passed,
    )
    validate_physical_inventory(
        matrix, node, machine_name, args.machine_identity, inventory
    )
    run_root = args.output_dir.resolve()
    environment = environment_for(
        node, machine_name, checkout=args.checkout.resolve(), run_root=run_root
    )
    return {
        "classification": "not_executed_dry_run",
        "node": node["id"],
        "priority": node["priority"],
        "machine_pool_entry": machine_name,
        "caller_machine_identity": args.machine_identity,
        "common_sha": args.common_sha,
        "branch": args.branch,
        "command": build_command(node, args.python),
        "environment": environment,
        "physical_inventory": inventory,
        "automatic_reset": False,
    }


def execute(
    matrix: dict[str, Any],
    node: dict[str, Any],
    args: argparse.Namespace,
    inventory: dict[str, Any],
) -> dict[str, Any]:
    dry = preview(matrix, node, args, inventory)
    refuse_parallel_environment(dict(os.environ))
    actual_branch, actual_sha = local_checkout_identity(args.checkout.resolve())
    if (actual_branch, actual_sha) != (args.branch, args.common_sha):
        raise MatrixError(
            "local checkout does not match caller-attested branch/common SHA"
        )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    started = utc_now()
    stem = f"{started.replace(':', '').replace('-', '')}-{node['id']}"
    log_path = output_dir / f"{stem}.log"
    evidence_path = output_dir / f"{stem}.json"
    environment = os.environ.copy()
    for key, value in dry["environment"].items():
        if value is None:
            environment.pop(key, None)
        else:
            environment[key] = value
    command = dry["command"]

    evidence: dict[str, Any] = {
        "schema_version": 1,
        "started_utc": started,
        "finished_utc": None,
        "branch": actual_branch,
        "full_sha": actual_sha,
        "caller_machine_identity": args.machine_identity,
        "actual_fqdn": socket.getfqdn(),
        "architecture": node["architecture"],
        "physical_inventory": inventory,
        "mesh_device": node["mesh_device"],
        "tt_visible_devices": environment.get("TT_VISIBLE_DEVICES"),
        "selector": node["selector"],
        "command": command,
        "environment": dry["environment"],
        "cache_paths": {"TT_CACHE_PATH": environment.get("TT_CACHE_PATH")},
        "node": node["id"],
        "exit_code": None,
        "failure_classification": None,
        "acceptance_types": node["acceptance_types"],
        "metrics": [],
        "teardown_status": "not_started",
        "reset": {
            "performed": False,
            "automatic": False,
            "reason": "runner never resets hardware",
        },
        "stdout_log_path": str(log_path),
        "evidence_json_path": str(evidence_path),
    }
    evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")

    timed_out = False
    with ProcessLock(args.lock_file):
        process = subprocess.Popen(
            command,
            cwd=args.checkout,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        try:
            output, _ = process.communicate(timeout=node["timeout_seconds"] + 30)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGTERM)
            try:
                output, _ = process.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                output, _ = process.communicate()
        log_path.write_text(output, encoding="utf-8")

    exit_code = process.returncode
    classification = classify_failure(exit_code, output, timed_out)
    metrics = collect_metrics(output)
    if (
        classification == "passed"
        and node["stage"] == "e2e"
        and not metrics
    ):
        classification = "missing_acceptance_data"
    evidence.update(
        {
            "finished_utc": utc_now(),
            "exit_code": exit_code,
            "failure_classification": classification,
            "metrics": metrics,
            "teardown_status": (
                "process_exited; fixture teardown not independently hardware-verified"
            ),
        }
    )
    evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    return evidence


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    modes = result.add_mutually_exclusive_group(required=True)
    modes.add_argument("--validate", action="store_true")
    modes.add_argument("--list", action="store_true")
    modes.add_argument("--dry-run", action="store_true")
    modes.add_argument("--execute", action="store_true")
    modes.add_argument("--classify-log", type=Path)
    result.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    result.add_argument("--node")
    result.add_argument("--common-sha")
    result.add_argument("--branch")
    result.add_argument("--machine-identity")
    result.add_argument("--sync-gate-passed", action="store_true")
    result.add_argument("--physical-inventory", type=Path)
    result.add_argument("--checkout", type=Path, default=ROOT)
    result.add_argument("--output-dir", type=Path, default=Path("hardware-results"))
    result.add_argument(
        "--lock-file", type=Path, default=Path("/tmp/tt-transformers-hardware.lock")
    )
    result.add_argument("--python", default=sys.executable)
    result.add_argument("--exit-code", type=int)
    result.add_argument("--timed-out", action="store_true")
    return result


def main() -> int:
    args = parser().parse_args()
    if args.classify_log:
        if args.exit_code is None:
            raise MatrixError("--classify-log requires --exit-code")
        output = args.classify_log.read_text(encoding="utf-8")
        print(classify_failure(args.exit_code, output, args.timed_out))
        return 0

    matrix = load_json(args.matrix)
    counts = validate_matrix(matrix)
    if args.validate:
        print(json.dumps({"valid": True, "mesh_counts": counts}, sort_keys=True))
        return 0
    if args.list:
        for node in sorted(matrix["nodes"], key=lambda item: item["priority"]):
            print(
                f"{node['priority']:02d} {node['id']} "
                f"{node['architecture']} {node['mesh_device']} {node['stage']}"
            )
        return 0

    required = {
        "--node": args.node,
        "--common-sha": args.common_sha,
        "--branch": args.branch,
        "--machine-identity": args.machine_identity,
        "--physical-inventory": args.physical_inventory,
    }
    missing = [flag for flag, value in required.items() if not value]
    if missing:
        raise MatrixError(f"missing required execution arguments: {missing}")
    node = select_node(matrix, args.node)
    inventory = load_json(args.physical_inventory)
    result = (
        preview(matrix, node, args, inventory)
        if args.dry_run
        else execute(matrix, node, args, inventory)
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.execute:
        return int(result["failure_classification"] != "passed")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except MatrixError as error:
        print(f"hardware-matrix refusal: {error}", file=sys.stderr)
        raise SystemExit(2)
