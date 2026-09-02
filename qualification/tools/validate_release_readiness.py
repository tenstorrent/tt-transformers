#!/usr/bin/env python3
"""Validate the release-readiness checklist against current repository facts."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CHECKLIST = ROOT / "qualification/reports/release-readiness-checklist.csv"
MODELS = {
    "deepseek_r1_distill_qwen_14b",
    "llama32_1b",
    "llama32_3b",
    "llama33_70b",
    "llama3_8b",
    "mistral_7b",
    "phi4",
    "qwen25_72b",
    "qwen25_7b",
    "qwen25_coder_32b",
    "qwen2_7b",
    "qwen3_32b",
}
EXPECTED_MISSING_DESTINATIONS: set[str] = set()


def require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    errors: list[str] = []
    with CHECKLIST.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    expected_fields = [
        "id",
        "phase",
        "criterion",
        "status",
        "evidence_class",
        "release_blocking",
        "evidence_paths",
        "evidence_summary",
        "owner",
        "next_gate",
    ]
    require(list(rows[0]) == expected_fields, "checklist columns drifted", errors)
    ids = [row["id"] for row in rows]
    require(len(ids) == len(set(ids)), "checklist IDs are not unique", errors)
    require({row["phase"] for row in rows} >= {*(str(value) for value in range(10)), "overall"}, "not every phase and overall criteria are represented", errors)
    for row in rows:
        require(row["status"] in {"pass", "partial", "blocked"}, f"{row['id']}: invalid status", errors)
        require(row["release_blocking"] in {"true", "false"}, f"{row['id']}: invalid release_blocking", errors)
        require(bool(row["owner"].strip()), f"{row['id']}: owner is empty", errors)
        require(bool(row["next_gate"].strip()), f"{row['id']}: next gate is empty", errors)
        for relative in row["evidence_paths"].split("|"):
            require((ROOT / relative).exists(), f"{row['id']}: missing evidence path {relative}", errors)
    status_counts = Counter(row["status"] for row in rows)
    require(status_counts == {"pass": 45, "partial": 10, "blocked": 24}, f"unexpected status counts: {dict(status_counts)}", errors)
    require(any(row["release_blocking"] == "true" and row["status"] != "pass" for row in rows), "checklist has no release blocker", errors)

    model_rows = {row["id"].removeprefix("model."): row for row in rows if row["id"].startswith("model.")}
    require(set(model_rows) == MODELS, "model checklist does not cover exactly twelve packages", errors)
    require(all(row["status"] == "blocked" for row in model_rows.values()), "a model is incorrectly release-ready", errors)

    support = {}
    for path in sorted((ROOT / "examples").glob("*/support.json")):
        support[path.parent.name] = json.loads(path.read_text(encoding="utf-8"))
    require(set(support) == MODELS, "support manifests do not cover exactly twelve models", errors)
    require(all(value["status"] == "experimental" for value in support.values()), "a support manifest is not experimental", errors)
    require(all(not value["validation"]["evidence"] for value in support.values()), "a support manifest claims evidence", errors)
    revisions = [value["model"]["hf_revision"] for value in support.values()]
    require(
        all(re.fullmatch(r"[0-9a-f]{40}", revision or "") for revision in revisions),
        f"not all model revisions are pinned: {revisions}",
        errors,
    )

    with (ROOT / "qualification/provenance/source_inventory.csv").open(newline="", encoding="utf-8") as stream:
        inventory = list(csv.DictReader(stream))
    excluded = [row for row in inventory if row["disposition"] == "excluded"]
    require(len(inventory) == 365, f"expected 365 provenance rows; got {len(inventory)}", errors)
    require(len(excluded) == 26 and all("/moe/" in row["source_path"] for row in excluded), "MoE-only exclusion invariant failed", errors)
    missing = {
        destination
        for row in inventory
        if row["disposition"] != "excluded"
        for destination in row["destination_path"].split(";")
        if destination and not (ROOT / destination).exists()
    }
    require(missing == EXPECTED_MISSING_DESTINATIONS, f"provenance destination drift: {sorted(missing)}", errors)

    hardware = json.loads((ROOT / "qualification/manifests/hardware-matrix.json").read_text(encoding="utf-8"))
    require(hardware["status"] == "readiness_only_no_hardware_executed", "hardware matrix status changed; refresh audit", errors)
    require(len(hardware["nodes"]) == 42, "hardware readiness node count changed", errors)

    with (ROOT / "qualification/reports/consumer_import_sites.csv").open(newline="", encoding="utf-8") as stream:
        consumers = list(csv.DictReader(stream))
    require(len(consumers) == 209, f"consumer site count changed: {len(consumers)}", errors)
    require(all(row["external_status"] == "not_cut_over" for row in consumers), "consumer cutover status changed; refresh audit", errors)

    trust_remote_code_sites = []
    for path in (ROOT / "src/tt_transformers/models").rglob("*.py"):
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if re.search(r'trust_remote_code\s*["\']?\s*[:=]\s*True', line):
                trust_remote_code_sites.append((path, line_number))
    require(not trust_remote_code_sites, f"forced trust_remote_code sites remain: {trust_remote_code_sites}", errors)

    example_policy = json.loads(
        (ROOT / "qualification/extraction/example_runtime_policy_manifest.json").read_text(encoding="utf-8")
    )
    require(
        example_policy["policy"]
        == {
            "forced_trust_remote_code_true": 0,
            "cwd_relative_model_cache": 0,
            "unscoped_default_device_access": 0,
            "scoped_smoke_functions": 12,
        },
        f"example runtime policy drifted: {example_policy['policy']}",
        errors,
    )

    artifact_data = json.loads((ROOT / "qualification/reports/package-artifact-hashes.json").read_text(encoding="utf-8"))
    for artifact in artifact_data["artifacts"]:
        path = ROOT / "dist" / artifact["filename"]
        require(path.is_file(), f"missing built artifact {path.name}", errors)
        if path.is_file():
            require(path.stat().st_size == artifact["size"], f"artifact size drift: {path.name}", errors)
            require(sha256(path) == artifact["sha256"], f"artifact hash drift: {path.name}", errors)
    require(
        artifact_data["artifacts"][0]["sha256"]
        == "a8837a3803930b97d527ef569190ec69b9dddc7b0c7537d183c1ad1268a0ae51",
        "final wheel identity drifted",
        errors,
    )
    require(
        artifact_data["artifacts"][1]["sha256"]
        == "4f62cb8df64da31af6076d86bbb4fb7528de4ae8ff5eddeab0f14170e40dc407",
        "final sdist identity drifted",
        errors,
    )
    require(
        sha256(ROOT / "pyproject.toml") == artifact_data["pyproject_sha256"],
        "artifact pyproject identity is stale",
        errors,
    )
    require(
        artifact_data["source_tree"]
        == {
            "algorithm": "sha256(path + NUL + content + NUL, sorted by path)",
            "python_files": 132,
            "sha256": "3814705a56bcf5bdd3ed88578e203d6d9e428f329289facb9bc2215366fb4e5f",
        },
        "artifact source identity drifted",
        errors,
    )

    require((ROOT / ".github/workflows/host.yml").is_file(), "consolidated host workflow is missing", errors)
    require(not (ROOT / ".github/workflows/host-tests.yml").exists(), "superseded host workflow still exists", errors)
    constraints = {
        "constraints/host-py310.txt": "b18a42fa787a81df5cca8bc4a6a56953016f14d4a654cb2d56e079d120f1ec44",
        "constraints/host-py312.txt": "f1aeb029b428de852ff3e10fef19f75e9c9b8adf76ea21fb2bc7c843ee7fe7b5",
    }
    for relative, digest in constraints.items():
        require(sha256(ROOT / relative) == digest, f"constraint identity drifted: {relative}", errors)

    dependency_locks = json.loads(
        (ROOT / "qualification/reports/dependency-lock-evidence.json").read_text(encoding="utf-8")
    )
    lock_counts = {lock["target"]: lock["package_count"] for lock in dependency_locks.get("locks", [])}
    require(dependency_locks.get("status") == "strict_installs_pass", "strict dependency-lock installs are not passing", errors)
    require(
        lock_counts
        == {
            "base-py310": 31,
            "host-py310": 64,
            "qualification-py310": 20,
            "build-dev-py310": 44,
            "base-py312": 29,
            "host-py312": 61,
            "qualification-py312": 19,
            "build-dev-py312": 40,
        },
        f"dependency-lock package counts drifted: {lock_counts}",
        errors,
    )
    require(
        all(
            lock.get("strict_install_validation", {}).get("passed") is True
            and lock.get("strict_install_validation", {}).get("tt_transformers_absent") is True
            and (
                (
                    lock["group"] in {"base", "host"}
                    and lock.get("strict_install_validation", {}).get("torch_version") == "2.11.0+cpu"
                )
                or (
                    lock["group"] in {"qualification", "build-dev"}
                    and lock.get("strict_install_validation", {}).get("torch_absent") is True
                )
            )
            for lock in dependency_locks.get("locks", [])
        ),
        "strict dependency-lock validation evidence is incomplete",
        errors,
    )
    base_locks = [lock for lock in dependency_locks.get("locks", []) if lock["group"] == "base"]
    require(
        len(base_locks) == 2
        and all(any(package["name"] == "pyyaml" and package["version"] == "6.0.3" for package in lock["packages"]) for lock in base_locks),
        "PyYAML is not proven by both base locks",
        errors,
    )

    workflow = (ROOT / ".github/workflows/host.yml").read_text(encoding="utf-8")
    for lock_prefix in ("host-py", "base-py", "build-dev-py"):
        require(f'constraints/locks/{lock_prefix}' in workflow, f"workflow does not consume {lock_prefix} lock", errors)

    static_quality = json.loads(
        (ROOT / "qualification/reports/static-quality-baseline.json").read_text(encoding="utf-8")
    )
    require(static_quality["ruff_check"]["findings"] == 2148, "Ruff baseline drifted", errors)
    require(static_quality["ruff_format"]["would_reformat"] == 163, "Ruff format baseline drifted", errors)
    require(
        static_quality["mypy"]["errors"] == 489 and static_quality["mypy"]["files_with_errors"] == 99,
        "mypy baseline drifted",
        errors,
    )
    require(static_quality["black"]["status"] == "timed_out", "Black bounded baseline drifted", errors)

    require((ROOT / "qualification/reports/ttnn-0.77.0.md").is_file(), "primary TTNN verdict is missing", errors)
    require(
        (ROOT / "qualification/reports/ttnn-0.77.0-compatibility-matrix.json").is_file(),
        "machine TTNN compatibility verdict is missing",
        errors,
    )
    host_report = (ROOT / "qualification/reports/host-ttnn-0.77.md").read_text(encoding="utf-8")
    require("| 3.10.19 | 2,115 | 28 | 6,791 | 81 | 0 | 0 |" in host_report, "Python 3.10 host result drifted", errors)
    require("| 3.12.13 | 2,115 | 28 | 6,791 | 81 | 0 | 0 |" in host_report, "Python 3.12 host result drifted", errors)
    require("nanobind: leaked 8 instances!" in host_report, "Python 3.12 nanobind diagnostic is missing", errors)

    pyramid = (ROOT / "qualification/reports/test-pyramid.md").read_text(encoding="utf-8")
    require("1,430 source-level test functions" in pyramid, "test-pyramid total is stale", errors)
    require(
        "1,176 explicitly `host`" in pyramid and "254 explicitly `device`" in pyramid,
        "test-pyramid lanes are stale",
        errors,
    )

    package_report = (ROOT / "qualification/reports/package-wheel.md").read_text(encoding="utf-8")
    require("twine==7.0.0" in package_report and "readme-renderer==46.0" in package_report, "Twine tool identity is missing", errors)
    require(
        "Checking dist/tt_transformers-0.1.0.dev0-py3-none-any.whl: PASSED" in package_report
        and "Checking dist/tt_transformers-0.1.0.dev0.tar.gz: PASSED" in package_report,
        "strict Twine results are missing",
        errors,
    )

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print(
        f"Validated {len(rows)} release-readiness criteria: "
        f"{status_counts['pass']} pass, {status_counts['partial']} partial, {status_counts['blocked']} blocked; "
        "12/12 models remain experimental and not release-qualified."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
