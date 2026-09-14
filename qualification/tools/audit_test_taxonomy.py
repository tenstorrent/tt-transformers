#!/usr/bin/env python3
"""Audit explicit pytest taxonomy and optionally collect only host suites."""

from __future__ import annotations

import argparse
import ast
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests"
TAXONOMY = {
    "host",
    "device",
    "slow",
    "model",
    "wormhole",
    "blackhole",
    "n150",
    "n300",
    "t3k",
    "p150",
    "p300",
    "p150x4",
    "galaxy_wh",
    "galaxy_bh",
}
#: `galaxy` was one marker treated as an arch-implying SKU, which made a Galaxy
#: test on Blackhole unexpressible: the rule below enforces SKU => arch, so
#: `galaxy` + `blackhole` failed this gate. Galaxy is a *topology*, and both
#: architectures ship one, so the marker splits per arch rather than dropping
#: its arch implication -- that implication is what keeps the gate useful.
SKU_ARCH = {
    "n150": "wormhole",
    "n300": "wormhole",
    "t3k": "wormhole",
    "p150": "blackhole",
    "p300": "blackhole",
    "p150x4": "blackhole",
    "galaxy_wh": "wormhole",
    "galaxy_bh": "blackhole",
}
PROVEN_DEVICE_MARKS = {
    ("tests/modules/mlp/test_mlp_2d.py", "test_ttnn_linear_2d_mesh_topology_bug"): {"device", "wormhole"},
}


def test_functions(tree: ast.Module):
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_"):
            yield node
        elif isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name.startswith("test_"):
                    yield child


def marks(node) -> set[str]:
    result = set()
    for decorator in node.decorator_list:
        value = ast.unparse(decorator)
        if value.startswith("pytest.mark."):
            result.add(value.split(".", 2)[2].split("(", 1)[0])
    return result


def configured_markers() -> set[str]:
    text = (ROOT / "pyproject.toml").read_text()
    tree = ast.parse("value = " + text.split("markers =", 1)[1].split("]", 1)[0] + "]")
    values = ast.literal_eval(tree.body[0].value)
    return {value.split(":", 1)[0].strip() for value in values}


def audit() -> dict:
    errors = []
    counts = Counter()
    functions = 0
    for path in sorted(TESTS.rglob("test_*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        relative = path.relative_to(ROOT).as_posix()
        if relative.startswith("tests/hardware/models/") and "selected_topology_marks" not in path.read_text():
            errors.append(f"{relative}: hardware wrapper lacks explicit MESH_DEVICE topology augmentation")
        for node in test_functions(tree):
            functions += 1
            node_marks = marks(node)
            counts.update(node_marks & TAXONOMY)
            lane = node_marks & {"host", "device"}
            if len(lane) != 1:
                errors.append(
                    f"{relative}:{node.lineno}:{node.name}: expected exactly one of host/device, got {sorted(lane)}"
                )
            if relative.startswith(("tests/models/", "tests/hardware/models/")) and "model" not in node_marks:
                errors.append(f"{relative}:{node.lineno}:{node.name}: concrete model test lacks model mark")
            if relative.startswith("tests/hardware/models/") and "slow" not in node_marks:
                errors.append(f"{relative}:{node.lineno}:{node.name}: e2e hardware gate lacks slow mark")
            for sku, arch in SKU_ARCH.items():
                if sku in node_marks and not {"device", arch} <= node_marks:
                    errors.append(f"{relative}:{node.lineno}:{node.name}: {sku} must imply device+{arch}")
            if "host" in node_marks and (node_marks & ({"wormhole", "blackhole"} | set(SKU_ARCH))):
                errors.append(f"{relative}:{node.lineno}:{node.name}: host test carries hardware topology marks")
            required = PROVEN_DEVICE_MARKS.get((relative, node.name), set())
            if not required <= node_marks:
                errors.append(
                    f"{relative}:{node.lineno}:{node.name}: proven hardware fixture requires {sorted(required)}"
                )
    missing_config = TAXONOMY - configured_markers()
    if missing_config:
        errors.append(f"pyproject marker registry is missing {sorted(missing_config)}")
    return {
        "functions": functions,
        "counts": dict(sorted(counts.items())),
        "errors": errors,
    }


def collect_host() -> tuple[int, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = "src"
    command = [
        sys.executable,
        "-m",
        "pytest",
        "--collect-only",
        "-q",
        "-m",
        "host",
        "tests/host",
        "tests/qualification/readiness",
    ]
    result = subprocess.run(command, cwd=ROOT, env=env, text=True, capture_output=True, check=False)
    return result.returncode, result.stdout + result.stderr


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit the audit result as JSON")
    parser.add_argument("--collect-host", action="store_true", help="also collect only the explicit host suites")
    args = parser.parse_args(argv)
    result = audit()
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"Audited {result['functions']} explicit test functions: {result['counts']}")
        for error in result["errors"]:
            print(f"ERROR: {error}")
    if result["errors"]:
        return 1
    if args.collect_host:
        returncode, output = collect_host()
        print(output, end="")
        return returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
