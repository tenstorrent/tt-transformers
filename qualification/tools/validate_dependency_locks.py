#!/usr/bin/env python3
"""Validate dependency lock syntax, provenance, hashes, schema, and install evidence."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from urllib.parse import urlparse

from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "qualification/reports/dependency-lock-evidence.json"
SCHEMA = ROOT / "qualification/schemas/dependency-lock-evidence.schema.json"
TARGETS = {
    "base-py310",
    "host-py310",
    "qualification-py310",
    "build-dev-py310",
    "base-py312",
    "host-py312",
    "qualification-py312",
    "build-dev-py312",
}
EXPECTED_COUNTS = {
    "base-py310": 31,
    "host-py310": 64,
    "qualification-py310": 20,
    "build-dev-py310": 44,
    "base-py312": 29,
    "host-py312": 61,
    "qualification-py312": 19,
    "build-dev-py312": 40,
}


def canonical(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_lock(path: Path) -> dict[str, tuple[str, str]]:
    text = path.read_text()
    required_options = (
        "--index-url https://pypi.org/simple",
        "--extra-index-url https://download.pytorch.org/whl/cpu",
        "--only-binary :all:",
        "--require-hashes",
    )
    for option in required_options:
        if text.count(option) != 1:
            raise ValueError(f"{path}: expected exactly one {option!r}")
    forbidden = re.compile(r"(^|\s)(-e|--editable|git\+|hg\+|svn\+|bzr\+|file:|\.\.?/|/)", re.MULTILINE)
    if forbidden.search("\n".join(line for line in text.splitlines() if not line.startswith("#"))):
        raise ValueError(f"{path}: editable/VCS/path requirement found")
    rows = {}
    pattern = re.compile(
        r"^([a-z0-9][a-z0-9-]*)==([^\s\\]+) \\\n    --hash=sha256:([0-9a-f]{64})$",
        re.MULTILINE,
    )
    for name, version, digest in pattern.findall(text):
        name = canonical(name)
        if name in rows:
            raise ValueError(f"{path}: duplicate {name}")
        rows[name] = (version, digest)
    requirement_lines = [line for line in text.splitlines() if line and not line.startswith(("#", "--", " "))]
    if len(rows) != len(requirement_lines):
        raise ValueError(f"{path}: unparsed or unhashed requirement")
    return rows


def freeze_map(lines: list[str]) -> dict[str, str]:
    result = {}
    for line in lines:
        if "==" not in line:
            raise ValueError(f"non-pinned installed distribution: {line}")
        name, version = line.split("==", 1)
        result[canonical(name)] = version
    result.pop("pip", None)
    return result


def main() -> int:
    errors = []
    schema = json.loads(SCHEMA.read_text())
    evidence = json.loads(EVIDENCE.read_text())
    for error in Draft202012Validator(schema).iter_errors(evidence):
        errors.append(f"schema:{'/'.join(map(str, error.absolute_path))}: {error.message}")
    if evidence.get("status") != "strict_installs_pass":
        errors.append(f"evidence status is not strict_installs_pass: {evidence.get('status')}")
    locks = {lock["target"]: lock for lock in evidence.get("locks", [])}
    if set(locks) != TARGETS:
        errors.append(f"lock target set drifted: {sorted(locks)}")
    by_python_group = {}
    for target, lock in locks.items():
        path = ROOT / lock["lock_path"]
        try:
            parsed = parse_lock(path)
        except Exception as error:
            errors.append(str(error))
            continue
        if sha256(path) != lock["lock_sha256"]:
            errors.append(f"{target}: lock digest drifted")
        packages = {package["name"]: package for package in lock["packages"]}
        if len(packages) != lock["package_count"] or len(parsed) != lock["package_count"]:
            errors.append(f"{target}: package count drifted")
        if "tt-transformers" in packages or "tt-transformers" in parsed:
            errors.append(f"{target}: contains tt-transformers")
        for name, package in packages.items():
            if parsed.get(name) != (package["version"], package["sha256"]):
                errors.append(f"{target}:{name}: lock/evidence mismatch")
            url = urlparse(package["url"])
            expected_index = "pytorch_cpu" if url.hostname in {"download.pytorch.org", "download-r2.pytorch.org"} else "pypi"
            if url.scheme != "https" or package["index"] != expected_index:
                errors.append(f"{target}:{name}: URL/index provenance mismatch")
            if not package["filename"].endswith(".whl"):
                errors.append(f"{target}:{name}: non-wheel archive")
        requires_torch = lock["group"] in {"base", "host"}
        torch = packages.get("torch", {})
        if requires_torch and (torch.get("version") != "2.11.0+cpu" or torch.get("index") != "pytorch_cpu"):
            errors.append(f"{target}: wrong torch artifact")
        if not requires_torch and torch:
            errors.append(f"{target}: auxiliary lock unexpectedly contains torch")
        validation = lock.get("strict_install_validation") or {}
        if not (
            validation.get("passed") is True
            and validation.get("strict") is True
            and validation.get("require_hashes") is True
            and validation.get("clean_venv") is True
            and validation.get("install_exit_code") == 0
            and validation.get("pip_check_exit_code") == 0
            and (
                (requires_torch and validation.get("torch_version") == "2.11.0+cpu")
                or (not requires_torch and validation.get("torch_absent") is True)
            )
            and validation.get("tt_transformers_absent") is True
        ):
            errors.append(f"{target}: strict-install evidence is incomplete")
        try:
            installed = freeze_map(validation.get("freeze", []))
            expected = {name: package["version"] for name, package in packages.items()}
            if "setuptools" not in expected:
                installed.pop("setuptools", None)  # CPython 3.10 venv bootstrap, not a selected dependency.
            if installed != expected:
                errors.append(f"{target}: installed freeze differs from lock")
        except Exception as error:
            errors.append(f"{target}: {error}")
        by_python_group[(lock["python"], lock["group"])] = packages
        requested_names = {name for name, package in packages.items() if package["requested"]}
        if lock["group"] == "qualification" and requested_names != {"openai", "requests"}:
            errors.append(f"{target}: qualification direct requirements drifted: {sorted(requested_names)}")
        if lock["group"] == "build-dev" and requested_names != {
            "setuptools",
            "wheel",
            "build",
            "black",
            "mypy",
            "ruff",
            "twine",
            "readme-renderer",
        }:
            errors.append(f"{target}: build/dev direct requirements drifted: {sorted(requested_names)}")
    for python in ("3.10.19", "3.12.13"):
        base = by_python_group.get((python, "base"), {})
        host = by_python_group.get((python, "host"), {})
        for name, package in base.items():
            comparable = ("version", "filename", "url", "index", "sha256")
            if host.get(name) is None or any(host[name][field] != package[field] for field in comparable):
                errors.append(f"Python {python}: base artifact {name} differs in host lock")
        if base.get("pyyaml", {}).get("version") != "6.0.3":
            errors.append(f"Python {python}: base lock does not provide PyYAML 6.0.3")
    actual_counts = {target: lock["package_count"] for target, lock in locks.items()}
    if actual_counts != EXPECTED_COUNTS:
        errors.append(f"package counts drifted: {actual_counts}")
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    counts = ", ".join(f"{target}={locks[target]['package_count']}" for target in sorted(locks))
    print(f"validated eight hash-complete locks and strict clean installs: {counts}; YAML supplied by both base locks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
