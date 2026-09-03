#!/usr/bin/env python3
"""Generate and validate the authoritative TTNN 0.77 compatibility matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "qualification/reports/ttnn-0.77.0-compatibility-matrix.json"
REPORT = ROOT / "qualification/reports/ttnn-0.77.0.md"
HARDWARE_CANDIDATE_SHA = "73d414f8b826a7da982df8c8229d4ac41ed8ba33"
HARDWARE_INDEX_RELATIVE = f"qualification/evidence/hardware/{HARDWARE_CANDIDATE_SHA}/index.json"
HARDWARE_MATRIX_RELATIVE = "qualification/manifests/hardware-matrix.json"
HARDWARE_INDEX_ROOT = ROOT / Path(HARDWARE_INDEX_RELATIVE).parent
EXPECTED_HARDWARE_INDEX_SHA256 = "4e98b62c34fb9f8744e00624c091dc9de18b3f32c5cd74f2ad2be1ad26274d7a"
EXPECTED_HARDWARE_MATRIX_SHA256 = "1d04716dd79cf3c5ab6a7a2ac251224fe326ad64ef19fa440c06187118b9d7de"
DIAGNOSTIC_CANDIDATE_SHA = "2883a949860d749adc2ed1af5525b27a9a547505"
TTFT_DIAGNOSTIC_RELATIVE = (
    f"qualification/evidence/diagnostics/{DIAGNOSTIC_CANDIDATE_SHA}/wh-lb-42/accuracy-ttft/summary.json"
)
EXPECTED_HARDWARE_OUTCOMES = {
    "passed": 42,
    "functional_failure": 0,
    "hardware_lifecycle_failure": 0,
    "missing_acceptance_data": 0,
    "pre_device_failure": 0,
}
EXPECTED_STAGE_COVERAGE = {
    "module": (30, 30, 30, 0, 0),
    "runtime": (1, 1, 1, 0, 0),
    "smoke": (3, 3, 3, 0, 0),
    "e2e": (8, 8, 8, 0, 0),
}
EXPECTED_W6_NODE = "wh-t3k-runtime-trace-order"
EXPECTED_TEARDOWN_STATUS = "process_exited; fixture teardown not independently hardware-verified"
EXPECTED_SURFACES = {
    "package_artifact",
    "package_root",
    "foundation_helpers",
    "reusable_modules",
    "sampling_foundation",
    "llm_runtime",
    "shared_executors",
    "concrete_model_cores",
}
SURFACE_HARDWARE_VERDICTS = {
    "package_artifact": "not_applicable",
    "package_root": "not_applicable",
    "foundation_helpers": "partial_exact_sha_matrix_evidence_not_surface_complete",
    "reusable_modules": "pass_executed_scope_30_of_30",
    "sampling_foundation": "partial_exact_sha_matrix_evidence_not_surface_complete",
    "llm_runtime": "pass_executed_w6_trace_order_scope_not_runtime_complete",
    "shared_executors": "partial_exact_sha_runtime_smoke_and_e2e_pass",
    "concrete_model_cores": "pass_executed_smoke_and_e2e_scope_not_manifest_qualified",
}
SURFACE_DEFINITIONS = (
    (
        "package_artifact",
        "base",
        "pass_build_metadata_archive_install_and_pip_check",
        "Built wheel/sdist, exact base metadata, archive policy, non-editable installs, and pip consistency.",
        ["qualification/reports/package-wheel.md", "qualification/reports/package-artifact-hashes.json"],
    ),
    (
        "package_root",
        "base",
        "pass_import_py310_py312",
        "Root package imports from site-packages with optional model dependencies absent and blocked.",
        ["qualification/reports/package-wheel.md", "qualification/reports/host-ttnn-0.77.md"],
    ),
    (
        "foundation_helpers",
        "base",
        "pass_import_both_and_host_semantics_py310_py312",
        (
            "Device/tensor/mesh helpers, typed environment/cache policy, "
            "program-config serialization, and ownership logic."
        ),
        [
            "qualification/reports/package-wheel.md",
            "qualification/reports/ttnn-0.77.0-host-triage.md",
            "qualification/extraction/cache_environment.md",
            "qualification/extraction/device_ownership.md",
        ],
    ),
    (
        "reusable_modules",
        "modules",
        "pass_import_both_and_host_semantics_py310_py312",
        "Attention, embedding, lazy tensor owners, LM head, MLP, RMSNorm, RoPE, and CCL host contracts.",
        ["qualification/reports/package-wheel.md", "qualification/reports/host-ttnn-0.77.md"],
    ),
    (
        "sampling_foundation",
        "modules",
        "pass_import_both_and_host_semantics_py310_py312",
        "Canonical SamplingParams, preparation, penalties, seed/state, log-probability, and ownership contracts.",
        ["qualification/reports/package-wheel.md", "qualification/reports/host-ttnn-0.77.md"],
    ),
    (
        "llm_runtime",
        "runtime",
        "pass_import_both_and_host_semantics_py310_py312",
        (
            "Prefill/decode planning, program/trace identity, tensor cleanup, "
            "output reading, cache geometry, and adapters with fakes/mocks."
        ),
        ["qualification/reports/package-wheel.md", "qualification/reports/host-ttnn-0.77.md"],
    ),
    (
        "shared_executors",
        "runtime",
        "pass_import_both_and_host_semantics_py310_py312",
        "Family-neutral, Llama-family, and Qwen-family executor configuration/lifecycle contracts.",
        ["qualification/reports/package-wheel.md", "qualification/reports/host-ttnn-0.77.md"],
    ),
    (
        "concrete_model_cores",
        "models",
        "pass_import_both_and_host_config_contracts_py310_py312",
        "All twelve package/core modules and host-only profiles/configuration/contracts pass on both Pythons.",
        [
            "qualification/reports/package-wheel.md",
            "qualification/reports/host-ttnn-0.77.md",
            "qualification/reports/ttnn-0.77.0-host-triage.md",
        ],
    ),
)


def read_json(relative: str) -> Any:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def outcome_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    return {outcome: sum(record["outcome"] == outcome for record in records) for outcome in EXPECTED_HARDWARE_OUTCOMES}


def coverage_rows(
    matrix_nodes: list[dict[str, Any]],
    records: list[dict[str, Any]],
    *,
    key: str,
    order: list[str],
) -> list[dict[str, Any]]:
    executed_ids = {record["node"] for record in records}
    rows = []
    for name in order:
        selected_nodes = [node for node in matrix_nodes if node[key] == name]
        selected_records = [record for record in records if record[key] == name]
        rows.append(
            {
                "name": name,
                "matrix_total": len(selected_nodes),
                "executed": len(selected_records),
                "passed": sum(record["outcome"] == "passed" for record in selected_records),
                "functional_failure": sum(record["outcome"] == "functional_failure" for record in selected_records),
                "hardware_lifecycle_failure": sum(
                    record["outcome"] == "hardware_lifecycle_failure" for record in selected_records
                ),
                "deferred": sum(node["id"] not in executed_ids for node in selected_nodes),
            }
        )
    return rows


def build_hardware_qualification() -> dict[str, Any]:
    index_path = ROOT / HARDWARE_INDEX_RELATIVE
    matrix_path = ROOT / HARDWARE_MATRIX_RELATIVE
    index = read_json(HARDWARE_INDEX_RELATIVE)
    hardware_matrix = read_json(HARDWARE_MATRIX_RELATIVE)
    records = index["records"]
    matrix_nodes = hardware_matrix["nodes"]
    matrix_by_id = {node["id"]: node for node in matrix_nodes}

    if sha256_file(index_path) != EXPECTED_HARDWARE_INDEX_SHA256:
        raise ValueError("canonical hardware index digest drift")
    if index["candidate_sha"] != HARDWARE_CANDIDATE_SHA:
        raise ValueError(f"unexpected hardware candidate SHA: {index['candidate_sha']}")
    if sha256_file(matrix_path) != EXPECTED_HARDWARE_MATRIX_SHA256:
        raise ValueError("hardware matrix digest drift")
    if index["matrix"] != {
        "node_count": 42,
        "path": HARDWARE_MATRIX_RELATIVE,
        "schema_version": hardware_matrix["schema_version"],
        "sha256": sha256_file(matrix_path),
    }:
        raise ValueError("canonical hardware index matrix identity drift")
    if len(matrix_by_id) != len(matrix_nodes) or len(matrix_nodes) != 42:
        raise ValueError("hardware matrix must contain 42 unique nodes")
    if len(records) != 42 or len({record["node"] for record in records}) != 42:
        raise ValueError("canonical hardware evidence must contain 42 unique executed nodes")

    for record in records:
        if record["full_sha"] != HARDWARE_CANDIDATE_SHA:
            raise ValueError(f"mixed-SHA hardware record: {record['node']}")
        if record["branch"] != "tttv2-standalone-migration":
            raise ValueError(f"unexpected hardware branch: {record['node']}")
        node = matrix_by_id.get(record["node"])
        if node is None:
            raise ValueError(f"hardware record is absent from the matrix: {record['node']}")
        for record_key, node_key in (
            ("priority", "priority"),
            ("stage", "stage"),
            ("architecture", "architecture"),
            ("mesh_device", "mesh_device"),
        ):
            if record[record_key] != node[node_key]:
                raise ValueError(f"hardware record/matrix {record_key} mismatch: {record['node']}")
        if record["machine_pool_entry"] not in node["machine_pool"]:
            raise ValueError(f"hardware record uses an invalid machine: {record['node']}")
        if record["reset"] != {
            "automatic": False,
            "performed": False,
            "reason": "runner never resets hardware",
        }:
            raise ValueError(f"hardware reset evidence drift: {record['node']}")
        if record["teardown_status"] != EXPECTED_TEARDOWN_STATUS:
            raise ValueError(f"hardware teardown evidence drift: {record['node']}")
        for artifact_name in ("evidence_json", "stdout_log"):
            artifact = record["source"][artifact_name]
            artifact_path = HARDWARE_INDEX_ROOT / artifact["path"]
            if not artifact_path.is_file():
                raise ValueError(f"missing hardware evidence artifact: {artifact_path}")
            if artifact_path.stat().st_size != artifact["size_bytes"]:
                raise ValueError(f"hardware evidence size drift: {artifact_path}")
            if sha256_file(artifact_path) != artifact["sha256"]:
                raise ValueError(f"hardware evidence digest drift: {artifact_path}")
        if record["mesh_device"] == "P150":
            source_evidence = json.loads(
                (HARDWARE_INDEX_ROOT / record["source"]["evidence_json"]["path"]).read_text(encoding="utf-8")
            )
            if (
                record["machine_pool_entry"] != "bh-qb-05"
                or source_evidence["environment"].get("MESH_DEVICE") != "P150"
                or source_evidence["environment"].get("TT_VISIBLE_DEVICES") is not None
                or source_evidence["physical_inventory"].get("cluster_type") != "P150_X4"
                or source_evidence["physical_inventory"].get("system_mesh") != "2x2"
                or source_evidence["physical_inventory"].get("device_count") != 4
            ):
                raise ValueError(f"logical single-P150 physical provenance drift: {record['node']}")

    actual_outcomes = outcome_counts(records)
    if index["summary"]["total"] != 42 or actual_outcomes != EXPECTED_HARDWARE_OUTCOMES:
        raise ValueError(f"hardware outcome drift: {actual_outcomes}")
    if index["summary"]["outcomes"] != actual_outcomes:
        raise ValueError("canonical hardware summary does not match its records")

    stage_coverage = coverage_rows(
        matrix_nodes,
        records,
        key="stage",
        order=["module", "runtime", "smoke", "e2e"],
    )
    actual_stage_coverage = {
        row["name"]: (
            row["matrix_total"],
            row["executed"],
            row["passed"],
            row["functional_failure"],
            row["deferred"],
        )
        for row in stage_coverage
    }
    if actual_stage_coverage != EXPECTED_STAGE_COVERAGE:
        raise ValueError(f"hardware stage coverage drift: {actual_stage_coverage}")

    executed_ids = {record["node"] for record in records}
    deferred_nodes = [node for node in matrix_nodes if node["id"] not in executed_ids]
    if deferred_nodes:
        raise ValueError(f"canonical hardware matrix is not fully executed: {[node['id'] for node in deferred_nodes]}")

    blockers = [record for record in records if record["outcome"] != "passed"]
    if blockers:
        raise ValueError(f"unexpected hardware blockers: {[record['node'] for record in blockers]}")
    w6_records = [record for record in records if record["node"] == EXPECTED_W6_NODE]
    if len(w6_records) != 1 or w6_records[0]["outcome"] != "passed" or w6_records[0]["exit_code"] != 0:
        raise ValueError("W6 exact-SHA hardware pass is missing")
    w6_log = HARDWARE_INDEX_ROOT / w6_records[0]["source"]["stdout_log"]["path"]
    if "4 passed in 490.54s" not in w6_log.read_text(encoding="utf-8"):
        raise ValueError("W6 four-case duration observation drift")

    executed_records = []
    for record in records:
        executed_records.append(
            {
                "priority": record["priority"],
                "node": record["node"],
                "stage": record["stage"],
                "architecture": record["architecture"],
                "mesh_device": record["mesh_device"],
                "machine_pool_entry": record["machine_pool_entry"],
                "outcome": record["outcome"],
                "metrics_count": record["metrics_count"],
                "teardown_status": record["teardown_status"],
                "reset_performed": record["reset"]["performed"],
                "evidence_json": str(Path(HARDWARE_INDEX_RELATIVE).parent / record["source"]["evidence_json"]["path"]),
                "stdout_log": str(Path(HARDWARE_INDEX_RELATIVE).parent / record["source"]["stdout_log"]["path"]),
            }
        )

    return {
        "verdict": "pass_42_of_42_matrix_nodes",
        "candidate_sha": HARDWARE_CANDIDATE_SHA,
        "branch": "tttv2-standalone-migration",
        "canonical_index": {
            "path": HARDWARE_INDEX_RELATIVE,
            "sha256": sha256_file(index_path),
            "record_count": len(records),
        },
        "matrix": index["matrix"],
        "outcomes": actual_outcomes,
        "stage_coverage": stage_coverage,
        "mesh_coverage": coverage_rows(
            matrix_nodes,
            records,
            key="mesh_device",
            order=["N150", "N300", "T3K", "P150", "P150x4"],
        ),
        "architecture_coverage": coverage_rows(
            matrix_nodes,
            records,
            key="architecture",
            order=["wormhole", "blackhole"],
        ),
        "lifecycle": {
            "hardware_lifecycle_failures": 0,
            "resets_performed": 0,
            "automatic_resets": 0,
            "teardown_status": EXPECTED_TEARDOWN_STATUS,
        },
        "w6_observation": {
            "cases_passed": 4,
            "pytest_duration_seconds": 490.54,
            "configured_performance_floor": False,
            "interpretation": "correctness pass; slower timing observation is not a configured performance failure",
        },
        "p150_logical_provenance": {
            "records": 8,
            "logical_mesh": "1x1",
            "machine_pool_entry": "bh-qb-05",
            "physical_cluster_type": "P150_X4",
            "physical_system_mesh": "2x2",
            "selection_environment": {"MESH_DEVICE": "P150", "TT_VISIBLE_DEVICES": None},
            "claim_limit": "logical single-P150 execution; not standalone-product or full-host P150_X4 evidence",
        },
        "blocking_records": [],
        "deferred": {"count": 0, "classification": "none", "nodes": []},
        "executed_records": executed_records,
        "evidence": evidence_record([HARDWARE_INDEX_RELATIVE, HARDWARE_MATRIX_RELATIVE]),
    }


def build_noncanonical_performance_feedback() -> dict[str, Any]:
    diagnostic_path = ROOT / TTFT_DIAGNOSTIC_RELATIVE
    diagnostic = read_json(TTFT_DIAGNOSTIC_RELATIVE)
    expected_summary = {
        "cells": 4,
        "hardware_lifecycle_failures": 0,
        "overall_target_passed": 0,
        "pytest_failed": 4,
        "pytest_passed": 0,
        "resets": 0,
        "throughput_target_passed": 0,
        "ttft_target_passed": 4,
    }
    if (
        diagnostic.get("candidate_sha") != DIAGNOSTIC_CANDIDATE_SHA
        or diagnostic.get("classification") != "non-canonical performance diagnostic"
        or diagnostic.get("canonical_hardware_evidence_modified") is not False
        or diagnostic.get("summary") != expected_summary
    ):
        raise ValueError("Llama33 accuracy TTFT diagnostic summary drift")
    cells = diagnostic.get("cells")
    if not isinstance(cells, list) or len(cells) != 4:
        raise ValueError("Llama33 accuracy TTFT diagnostics must contain four cells")
    expected_cells = {
        ("accuracy-batch-32-T3K", "host"): (87.1, 7.9, 9.3),
        ("accuracy-batch-32-T3K", "on_device_topk"): (86.7, 12.2, 14.4),
        ("accuracy-batch-32-ci-T3K", "host"): (86.9, 7.7, 8.9),
        ("accuracy-batch-32-ci-T3K", "on_device_topk"): (87.2, 11.9, 14.2),
    }
    projected_cells = []
    evidence_paths = [TTFT_DIAGNOSTIC_RELATIVE]
    seen = set()
    for cell in cells:
        parameter_id = cell["node_id"].rsplit("[", 1)[-1].removesuffix("]")
        key = (parameter_id, cell["sampling_mode"])
        if key not in expected_cells or key in seen:
            raise ValueError(f"unexpected Llama33 accuracy TTFT diagnostic cell: {key}")
        seen.add(key)
        expected_ttft, expected_tok_s_u, expected_floor = expected_cells[key]
        metrics = cell["printed_metrics"]
        targets = cell["targets"]
        if (
            metrics["ttft_ms"] != expected_ttft
            or metrics["tok_s_u"] != expected_tok_s_u
            or targets["ttft_ms"] != 100.0
            or targets["adjusted_ttft_max_ms"] != 105.0
            or targets["tok_s_u"] != expected_floor
            or targets["adjusted_tok_s_u_min"] != expected_floor * 0.95
            or cell["target_adjusted"] != {"overall": False, "tok_s_u": False, "ttft": True}
            or cell["pytest_exit_code"] != 1
            or cell["pytest_result"] != "failed: throughput target assertion"
            or cell["reset_performed"] is not False
        ):
            raise ValueError(f"Llama33 accuracy TTFT diagnostic result drift: {key}")
        copied_log = diagnostic_path.parent / Path(cell["log_path"]).name
        if (
            not copied_log.is_file()
            or copied_log.stat().st_size != cell["log_size_bytes"]
            or sha256_file(copied_log) != cell["log_sha256"]
        ):
            raise ValueError(f"Llama33 accuracy TTFT diagnostic log drift: {copied_log}")
        copied_log_relative = str(copied_log.relative_to(ROOT))
        evidence_paths.append(copied_log_relative)
        projected_cells.append(
            {
                "parameter_id": parameter_id,
                "sampling_mode": cell["sampling_mode"],
                "ttft_ms": metrics["ttft_ms"],
                "ttft_target_ms": targets["ttft_ms"],
                "ttft_adjusted_max_ms": targets["adjusted_ttft_max_ms"],
                "ttft_target_passed": True,
                "tok_s_u": metrics["tok_s_u"],
                "tok_s_u_target": targets["tok_s_u"],
                "tok_s_u_adjusted_min": targets["adjusted_tok_s_u_min"],
                "throughput_target_passed": False,
                "overall_target_passed": False,
                "pytest_exit_code": cell["pytest_exit_code"],
                "teardown_status": cell["teardown_status"],
                "reset_performed": cell["reset_performed"],
                "log": copied_log_relative,
            }
        )
    if seen != set(expected_cells):
        raise ValueError("Llama33 accuracy TTFT diagnostic coverage is incomplete")
    return {
        "classification": "non_canonical_performance_diagnostic",
        "candidate_sha": DIAGNOSTIC_CANDIDATE_SHA,
        "current_hardware_candidate_sha": HARDWARE_CANDIDATE_SHA,
        "rerun_at_current_hardware_candidate": False,
        "canonical_hardware_record_count_impact": 0,
        "verdict": "ttft_4_of_4_pass_throughput_0_of_4_pass_overall_0_of_4",
        "summary": expected_summary,
        "cells": sorted(projected_cells, key=lambda row: (row["parameter_id"], row["sampling_mode"])),
        "evidence": evidence_record(evidence_paths),
    }


def evidence_record(paths: list[str]) -> list[dict[str, str]]:
    return [{"path": path} for path in paths]


def environment_record(label: str, dependency: dict[str, Any]) -> dict[str, Any]:
    imports = dependency["import_probe"]["imports"]
    return {
        "label": label,
        "python": dependency["python"],
        "implementation": dependency["implementation"],
        "platform": dependency["platform"],
        "dependency_group_versions": {
            name: imports[name]["distribution_version"]
            for name in ("ttnn", "torch", "loguru", "transformers", "tqdm", "pytest")
        },
        "direct_host_test_dependencies": {
            "jsonschema": "4.26.0",
            "pytz": "2026.3.post1",
        },
        "dependency_groups_pip_check": dependency["pip_check"]["passed"],
        "dependency_groups_isolated": (
            dependency["isolation"]["pythonpath_unset"] and not dependency["isolation"]["repo_checkout_on_sys_path"]
        ),
        "installed_base_wheel_probe": {
            "tt_transformers": "0.1.0.dev0",
            "ttnn": "0.77.0",
            "torch": "2.11.0",
            "loguru": "0.6.0",
            "optional_transformers_tqdm_pytest": "absent_and_blocked",
            "pip_check": "pass",
        },
        "evidence": evidence_record(
            [
                f"qualification/reports/dependencies-py{label}.md",
                "qualification/reports/package-wheel.md",
                "qualification/reports/host-ttnn-0.77.md",
            ]
        ),
    }


def model_record(manifest_path: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    model = manifest["model"]
    validation = manifest["validation"]
    stale_hardware_gap = "No hardware evidence is attributable to the pinned extraction revision"
    current_known_gaps = [
        "Final-SHA hardware evidence covers the 42-node matrix but does not qualify the complete model contract",
        *(gap for gap in manifest["known_gaps"] if gap != stale_hardware_gap),
    ]
    return {
        "package": model["package"],
        "hf_id": model["hf_id"],
        "hf_revision": model["hf_revision"],
        "lifecycle_status": manifest["status"],
        "host_verdict": "pass_import_py310_py312_and_host_contracts_py310_py312",
        "host_import_pythons": ["3.10.19", "3.12.13"],
        "host_semantics_pythons": ["3.10.19", "3.12.13"],
        "hardware_verdict": "not_qualified",
        "firmware_driver_verdict": "partial_exact_sha_stacks_recorded_not_target_complete",
        "geometry_verdict": "source_declared_with_full_exact_sha_matrix_evidence_not_promoted",
        "declared_hardware_targets": manifest["hardware"],
        "known_gaps": current_known_gaps,
        "manifest_validation": validation,
        "evidence": evidence_record(
            [
                str(manifest_path.relative_to(ROOT)),
                "qualification/reports/package-wheel.md",
                "qualification/reports/host-ttnn-0.77.md",
            ]
        ),
    }


def build_matrix() -> dict[str, Any]:
    api = read_json("qualification/reports/ttnn-0.77.0-api-availability.json")
    dependencies310 = read_json("qualification/reports/dependencies-py310.json")
    dependencies312 = read_json("qualification/reports/dependencies-py312.json")
    artifacts = read_json("qualification/reports/package-artifact-hashes.json")
    hardware_qualification = build_hardware_qualification()
    package_artifacts = {artifact["filename"]: artifact for artifact in artifacts["artifacts"]}
    expected_package_artifacts = {
        "tt_transformers-0.1.0.dev0-py3-none-any.whl": (
            "8c55fac0a764fb9ae4d6ca514062ef2cb6cfe3097306a2f877bcad41269c50c2"
        ),
        "tt_transformers-0.1.0.dev0.tar.gz": ("f161e13dedc5ce076d9553b677f0a1a4785996f932316f2325de9217376da644"),
    }
    if set(package_artifacts) != set(expected_package_artifacts):
        raise ValueError(f"package artifact set drift: {sorted(package_artifacts)}")
    for filename, expected_sha256 in expected_package_artifacts.items():
        if package_artifacts[filename]["sha256"] != expected_sha256:
            raise ValueError(f"package artifact digest drift: {filename}")
    if api["source_inventory"]["source_sha256"] != artifacts["source_tree"]["sha256"]:
        raise ValueError("API inventory and final package source-tree identities differ")
    expected_dependency_versions = {
        "ttnn": "0.77.0",
        "torch": "2.11.0+cpu",
        "loguru": "0.6.0",
        "transformers": "5.12.1",
        "tqdm": "4.66.3",
        "pytest": "9.0.3",
    }
    for dependency in (dependencies310, dependencies312):
        actual = {
            name: dependency["import_probe"]["imports"][name]["distribution_version"]
            for name in expected_dependency_versions
        }
        if actual != expected_dependency_versions or not dependency["pip_check"]["passed"]:
            raise ValueError(f"dependency matrix drift for Python {dependency['python']}: {actual}")
    host_text = (ROOT / "qualification/reports/host-ttnn-0.77.md").read_text(encoding="utf-8")
    for required in (
        "| 3.10.19 | 2,170 | 28 | 6,791 | 81 | 0 | 0 |",
        "| 3.12.13 | 2,170 | 28 | 6,791 | 81 | 0 | 0 |",
        "nanobind: leaked 10 instances!",
        "nanobind: leaked 36 types!",
        "nanobind: leaked 330 functions!",
        "not counted as a test failure or collection error",
        "jsonschema 4.26.0",
        "pytz 2026.3.post1",
    ):
        if required not in host_text:
            raise ValueError(f"host support report is missing final evidence {required!r}")
    pyramid_text = (ROOT / "qualification/reports/test-pyramid.md").read_text(encoding="utf-8")
    for required in (
        "1,465 source-level test functions",
        "1,211 explicitly `host`",
        "254 explicitly `device`",
        "393 concrete `model` surfaces",
    ):
        if required not in pyramid_text:
            raise ValueError(f"test-pyramid report is missing final taxonomy {required!r}")
    package_text = (ROOT / "qualification/reports/package-wheel.md").read_text(encoding="utf-8")
    for required in (
        artifacts["source_tree"]["sha256"],
        *(artifact["sha256"] for artifact in artifacts["artifacts"]),
        "ttnn==0.77.0",
        "torch==2.11.0",
        "loguru==0.6.0",
        "132 source-identical Python files",
    ):
        if required not in package_text:
            raise ValueError(f"package-wheel evidence is missing {required!r}")
    manifests = []
    for path in sorted((ROOT / "examples").glob("*/support.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        software = manifest["software"]
        if software["tt_transformers"] != "0.1.0.dev0" or software["ttnn"] != "0.77.0":
            raise ValueError(f"support manifest software drift: {path}")
        if software["python"] != ["3.10", "3.12"] or software["torch"] != "2.11.0":
            raise ValueError(f"support manifest interpreter/Torch drift: {path}")
        manifests.append(model_record(path, manifest))
    surfaces = [
        {
            "id": identifier,
            "category": category,
            "host_verdict": verdict,
            "host_scope": scope,
            "host_import_pythons": ["3.10.19", "3.12.13"],
            "host_semantics_pythons": ["3.10.19", "3.12.13"],
            "hardware_verdict": SURFACE_HARDWARE_VERDICTS[identifier],
            "evidence": evidence_record(
                [
                    *evidence,
                    *(
                        [HARDWARE_INDEX_RELATIVE, HARDWARE_MATRIX_RELATIVE]
                        if identifier not in {"package_artifact", "package_root"}
                        else []
                    ),
                ]
            ),
        }
        for identifier, category, verdict, scope, evidence in SURFACE_DEFINITIONS
    ]
    return {
        "schema_version": 3,
        "verdict": "host_compatible_with_ttnn_0_77_partial_release_qualification",
        "validation_date": "2026-09-03",
        "tt_transformers": "0.1.0.dev0",
        "ttnn": "0.77.0",
        "source_tree": artifacts["source_tree"],
        "package_artifacts": artifacts["artifacts"],
        "software_matrix": [
            environment_record("310", dependencies310),
            environment_record("312", dependencies312),
        ],
        "api_availability": {
            "unique_module_paths": api["source_inventory"]["unique_module_api_paths"],
            "occurrences": api["source_inventory"]["module_api_occurrences"],
            "available": {"3.10.19": 160, "3.12.13": 160},
            "missing": {"3.10.19": 0, "3.12.13": 0},
            "tiers": api["summaries"]["3.10"]["tiers"],
            "unstable_paths": api["unstable_api_cross_reference"],
            "unstable_presence_verdict": "14_of_14_present_both_pythons_semantics_unqualified",
            "signature_evidence": {
                "3.10.19": api["summaries"]["3.10"]["signature_quality"],
                "3.12.13": api["summaries"]["3.12"]["signature_quality"],
                "cross_interpreter_differences": len(api["cross_interpreter_signature_differences"]),
            },
            "evidence": evidence_record(
                [
                    "qualification/reports/ttnn-0.77.0-api-availability.md",
                    "qualification/reports/ttnn-0.77.0-api-availability.json",
                ]
            ),
        },
        "host_suites": [
            {
                "python": "3.10.19",
                "passed": 2170,
                "skipped": 28,
                "deselected": 6791,
                "warnings": 5,
                "subtests_passed": 81,
                "failures": 0,
                "errors": 0,
                "process_exit": 0,
                "shutdown_diagnostics": None,
                "evidence": evidence_record(
                    [
                        "qualification/reports/host-ttnn-0.77.md",
                        "qualification/reports/ttnn-0.77.0-host-triage.md",
                    ]
                ),
            },
            {
                "python": "3.12.13",
                "passed": 2170,
                "skipped": 28,
                "deselected": 6791,
                "warnings": 5,
                "subtests_passed": 81,
                "failures": 0,
                "errors": 0,
                "process_exit": 0,
                "shutdown_diagnostics": {
                    "classification": "ttnn_nanobind_binding_lifecycle_feedback",
                    "phase": "interpreter_shutdown_after_pytest_success",
                    "leaked_instances": 10,
                    "leaked_types": 36,
                    "leaked_functions": 330,
                    "test_failure": False,
                    "message": "this is likely caused by a reference counting issue in the binding code.",
                    "retained_output": "/tmp/gwang/tttv2-73d-py312-host-final.log",
                },
                "evidence": evidence_record(
                    [
                        "qualification/reports/host-ttnn-0.77.md",
                        "qualification/reports/ttnn-0.77.0-host-triage.md",
                    ]
                ),
            },
        ],
        "test_taxonomy": {
            "source_level_test_functions": 1465,
            "host": 1211,
            "device": 254,
            "model": 393,
            "evidence": evidence_record(["qualification/reports/test-pyramid.md"]),
        },
        "hardware_qualification": hardware_qualification,
        "noncanonical_performance_feedback": build_noncanonical_performance_feedback(),
        "surfaces": surfaces,
        "models": manifests,
        "closed_findings": [
            {
                "classification": "tttv2_packaging_defect",
                "finding": "missing pinned LMHead _nearest_32 alias",
                "status": "fixed",
                "evidence": evidence_record(["qualification/reports/package-wheel.md"]),
            },
            {
                "classification": "tttv2_packaging_defect",
                "finding": "guarded PyYAML feature lacked a declared optional extra",
                "status": "fixed",
                "evidence": evidence_record(["qualification/reports/package-wheel.md"]),
            },
            {
                "classification": "semantic_or_signature_difference",
                "finding": "TTNN 0.77 constructor lacks later num_workers_per_dram_bank argument",
                "status": "test_expectation_corrected_no_production_usage",
                "evidence": evidence_record(["qualification/reports/ttnn-0.77.0-host-triage.md"]),
            },
            {
                "classification": "tttv2_implementation_defect",
                "finding": "repr-only fallback failed for TTNN 0.77 SDPAProgramConfig binding",
                "status": "fixed_using_supported_public_attributes",
                "evidence": evidence_record(["qualification/reports/ttnn-0.77.0-host-triage.md"]),
            },
            {
                "classification": "obsolete_characterization",
                "finding": "stale old-namespace/model_params and retired TTTv1 expectations",
                "status": "migrated_or_explicitly_retired",
                "evidence": evidence_record(["qualification/reports/host-ttnn-0.77.md"]),
            },
        ],
        "open_binding_feedback": [
            {
                "classification": "ttnn_nanobind_binding_lifecycle_feedback",
                "finding": (
                    "Python 3.12 reports leaked TTNN nanobind instances, types, and "
                    "functions during interpreter shutdown"
                ),
                "status": "open_non_failing_upstream_binding_feedback",
                "test_failure": False,
                "evidence": evidence_record(["qualification/reports/host-ttnn-0.77.md"]),
            }
        ],
        "host_semantics_exercised": [
            "dependency resolution, metadata, wheel installation, and import isolation",
            "public TTNN symbol/descriptor presence on both interpreters",
            "host-safe TTNN config constructors and program-config serialization",
            "pure configuration, profile, topology, planning, signature, and cache-geometry logic",
            "runtime ownership, cleanup, trace/program identity, and adapter behavior through fakes/mocks",
            "optional dependency laziness and secret-redacted cache/device preflight",
            "all twelve model package/core imports and host-only model configuration/contracts",
        ],
        "not_exercised": [
            "experimental/private TTNN operations not selected by the 42 executed matrix records",
            "hardware behavior outside the exact recorded N150, N300, T3K, P150, and P150x4 selectors",
            (
                "performance qualification beyond the recorded correctness and token-accuracy acceptance data; "
                "four Llama33 accuracy TTFT diagnostics are noncanonical feedback only"
            ),
            "complete model-manifest geometry, TP/DP, batch/context, and lifecycle qualification",
            "independent post-fixture hardware teardown verification",
        ],
        "remaining_hardware_gaps": {
            "overall": "42_of_42_matrix_nodes_pass; complete model contracts remain unqualified",
            "firmware_driver": (
                "exact stacks recorded for executed wh-lb-42 and bh-qb-05 evidence; not qualified beyond those records"
            ),
            "unstable_api_semantics": ("14 paths present; only behavior selected by the executed matrix is evidenced"),
            "model_geometry": (
                "full exact-SHA matrix evidence exists; manifests remain experimental and unpromoted"
            ),
            "required_next_evidence": [
                "extend exact-SHA evidence to remaining model-manifest geometry and context buckets",
                "close the throughput regression measured only by the older noncanonical diagnostic SHA",
                "independently verify post-fixture hardware teardown where a clean-teardown claim is required",
            ],
            "evidence": evidence_record(
                [
                    "qualification/reports/host-ttnn-0.77.md",
                    HARDWARE_INDEX_RELATIVE,
                    HARDWARE_MATRIX_RELATIVE,
                ]
            ),
        },
        "authoritative_inputs": evidence_record(
            [
                "qualification/reports/dependencies-py310.json",
                "qualification/reports/dependencies-py312.json",
                "qualification/reports/ttnn-0.77.0-api-availability.json",
                "qualification/reports/package-artifact-hashes.json",
                "qualification/reports/package-wheel.md",
                "qualification/reports/ttnn-0.77.0-host-triage.md",
                "qualification/reports/host-ttnn-0.77.md",
                "qualification/reports/test-pyramid.md",
                HARDWARE_INDEX_RELATIVE,
                HARDWARE_MATRIX_RELATIVE,
            ]
        ),
    }


def validate(matrix: dict[str, Any]) -> None:
    generated = build_matrix()
    if matrix != generated:
        raise ValueError("compatibility matrix differs from deterministic authoritative-input projection")
    if (
        matrix["schema_version"] != 3
        or matrix["verdict"] != "host_compatible_with_ttnn_0_77_partial_release_qualification"
    ):
        raise ValueError("top-level TTNN 0.77 compatibility verdict drift")
    if {surface["id"] for surface in matrix["surfaces"]} != EXPECTED_SURFACES:
        raise ValueError("base/module/runtime surface verdict coverage is incomplete")
    for surface in matrix["surfaces"]:
        if not surface["host_verdict"] or not surface["evidence"]:
            raise ValueError(f"surface lacks verdict/evidence: {surface['id']}")
        if surface["host_semantics_pythons"] != ["3.10.19", "3.12.13"]:
            raise ValueError(f"surface lacks two-Python host verdict: {surface['id']}")
        if surface["hardware_verdict"] != SURFACE_HARDWARE_VERDICTS[surface["id"]]:
            raise ValueError(f"surface hardware verdict drift: {surface['id']}")

    hardware = matrix["hardware_qualification"]
    if (
        hardware["candidate_sha"] != HARDWARE_CANDIDATE_SHA
        or hardware["canonical_index"]["record_count"] != 42
        or hardware["outcomes"] != EXPECTED_HARDWARE_OUTCOMES
        or len(hardware["executed_records"]) != 42
        or hardware["deferred"] != {"count": 0, "classification": "none", "nodes": []}
    ):
        raise ValueError("hardware qualification totals drift")
    if hardware["canonical_index"]["sha256"] != EXPECTED_HARDWARE_INDEX_SHA256:
        raise ValueError("canonical hardware index digest drift")
    if {
        row["name"]: (
            row["matrix_total"],
            row["executed"],
            row["passed"],
            row["functional_failure"],
            row["deferred"],
        )
        for row in hardware["stage_coverage"]
    } != EXPECTED_STAGE_COVERAGE:
        raise ValueError("hardware stage verdict drift")
    if hardware["lifecycle"] != {
        "hardware_lifecycle_failures": 0,
        "resets_performed": 0,
        "automatic_resets": 0,
        "teardown_status": EXPECTED_TEARDOWN_STATUS,
    }:
        raise ValueError("hardware lifecycle/reset verdict drift")
    if hardware["p150_logical_provenance"] != {
        "records": 8,
        "logical_mesh": "1x1",
        "machine_pool_entry": "bh-qb-05",
        "physical_cluster_type": "P150_X4",
        "physical_system_mesh": "2x2",
        "selection_environment": {"MESH_DEVICE": "P150", "TT_VISIBLE_DEVICES": None},
        "claim_limit": "logical single-P150 execution; not standalone-product or full-host P150_X4 evidence",
    }:
        raise ValueError("logical single-P150 qualification provenance drift")
    if hardware["blocking_records"]:
        raise ValueError("canonical hardware evidence must have no blocking record")
    w6_records = [record for record in hardware["executed_records"] if record["node"] == EXPECTED_W6_NODE]
    if len(w6_records) != 1 or w6_records[0]["outcome"] != "passed":
        raise ValueError("W6 hardware pass verdict drift")
    source_models = {path.parent.name for path in (ROOT / "src/tt_transformers/models").glob("*/model.py")}
    manifest_models = {
        json.loads(path.read_text(encoding="utf-8"))["model"]["package"]
        for path in (ROOT / "examples").glob("*/support.json")
    }
    matrix_models = {model["package"] for model in matrix["models"]}
    if not (source_models == manifest_models == matrix_models) or len(matrix_models) != 12:
        raise ValueError(
            f"model coverage mismatch: source={sorted(source_models)}, "
            f"manifest={sorted(manifest_models)}, matrix={sorted(matrix_models)}"
        )
    for model in matrix["models"]:
        if model["lifecycle_status"] != "experimental":
            raise ValueError(f"unexpected model lifecycle claim: {model['package']}")
        if model["hardware_verdict"] != "not_qualified":
            raise ValueError(f"unsupported hardware claim: {model['package']}")
        if model["manifest_validation"]["evidence"]:
            raise ValueError(f"model has hardware evidence but matrix says unqualified: {model['package']}")
        if not model["known_gaps"] or not model["evidence"]:
            raise ValueError(f"model lacks gaps/evidence: {model['package']}")
        if model["known_gaps"][0] != (
            "Final-SHA hardware evidence covers the 42-node matrix but does not qualify the complete model contract"
        ):
            raise ValueError(f"model hardware gap verdict drift: {model['package']}")
        if model["host_semantics_pythons"] != ["3.10.19", "3.12.13"]:
            raise ValueError(f"model lacks two-Python host verdict: {model['package']}")
    api = matrix["api_availability"]
    if api["unique_module_paths"] != 160 or len(api["unstable_paths"]) != 14:
        raise ValueError("TTNN API coverage must remain 160 total and 14 unstable")
    if any(not item["py310_available"] or not item["py312_available"] for item in api["unstable_paths"]):
        raise ValueError("an unstable API is missing on a supported interpreter")
    if [suite["python"] for suite in matrix["host_suites"]] != ["3.10.19", "3.12.13"]:
        raise ValueError("host suite interpreter coverage is incomplete")
    for suite in matrix["host_suites"]:
        expected = (2170, 28, 6791, 5, 81, 0, 0, 0)
        actual = tuple(
            suite[field]
            for field in (
                "passed",
                "skipped",
                "deselected",
                "warnings",
                "subtests_passed",
                "failures",
                "errors",
                "process_exit",
            )
        )
        if actual != expected:
            raise ValueError(f"host suite result drift for Python {suite['python']}: {actual}")
    if matrix["test_taxonomy"] != {
        "source_level_test_functions": 1465,
        "host": 1211,
        "device": 254,
        "model": 393,
        "evidence": evidence_record(["qualification/reports/test-pyramid.md"]),
    }:
        raise ValueError("test taxonomy drift")
    performance_feedback = matrix["noncanonical_performance_feedback"]
    if (
        performance_feedback["candidate_sha"] != DIAGNOSTIC_CANDIDATE_SHA
        or performance_feedback["current_hardware_candidate_sha"] != HARDWARE_CANDIDATE_SHA
        or performance_feedback["rerun_at_current_hardware_candidate"] is not False
        or performance_feedback["canonical_hardware_record_count_impact"] != 0
        or performance_feedback["summary"]["ttft_target_passed"] != 4
        or performance_feedback["summary"]["throughput_target_passed"] != 0
        or performance_feedback["summary"]["overall_target_passed"] != 0
        or len(performance_feedback["cells"]) != 4
    ):
        raise ValueError("noncanonical Llama33 performance feedback drift")
    feedback = matrix["host_suites"][1]["shutdown_diagnostics"]
    if (
        feedback["classification"] != "ttnn_nanobind_binding_lifecycle_feedback"
        or feedback["test_failure"]
        or (feedback["leaked_instances"], feedback["leaked_types"], feedback["leaked_functions"]) != (10, 36, 330)
    ):
        raise ValueError("Python 3.12 TTNN binding feedback is missing or misclassified")
    for section in (
        matrix["surfaces"],
        matrix["models"],
        matrix["closed_findings"],
        matrix["open_binding_feedback"],
    ):
        for row in section:
            for evidence in row["evidence"]:
                if not (ROOT / evidence["path"]).is_file():
                    raise ValueError(f"missing evidence: {evidence['path']}")
    for environment in matrix["software_matrix"]:
        if environment["direct_host_test_dependencies"] != {
            "jsonschema": "4.26.0",
            "pytz": "2026.3.post1",
        }:
            raise ValueError(f"direct host-test dependency drift: {environment['label']}")
        for evidence in environment["evidence"]:
            if not (ROOT / evidence["path"]).is_file():
                raise ValueError(f"missing environment evidence: {evidence['path']}")
    for suite in matrix["host_suites"]:
        for evidence in suite["evidence"]:
            if not (ROOT / evidence["path"]).is_file():
                raise ValueError(f"missing host-suite evidence: {evidence['path']}")
    for evidence in matrix["test_taxonomy"]["evidence"]:
        if not (ROOT / evidence["path"]).is_file():
            raise ValueError(f"missing test-taxonomy evidence: {evidence['path']}")
    for evidence in hardware["evidence"]:
        if not (ROOT / evidence["path"]).is_file():
            raise ValueError(f"missing hardware-qualification evidence: {evidence['path']}")
    for evidence in performance_feedback["evidence"]:
        if not (ROOT / evidence["path"]).is_file():
            raise ValueError(f"missing noncanonical performance evidence: {evidence['path']}")
    for record in hardware["executed_records"]:
        for key in ("evidence_json", "stdout_log"):
            if not (ROOT / record[key]).is_file():
                raise ValueError(f"missing executed hardware evidence: {record[key]}")
    for evidence in matrix["authoritative_inputs"] + matrix["remaining_hardware_gaps"]["evidence"]:
        if not (ROOT / evidence["path"]).is_file():
            raise ValueError(f"missing evidence: {evidence['path']}")
    report_text = REPORT.read_text(encoding="utf-8")
    for required in (
        HARDWARE_CANDIDATE_SHA,
        DIAGNOSTIC_CANDIDATE_SHA,
        "2,170 passed",
        "1,465 source-level test functions",
        "1,211 host, 254 device, and 393 model-marked surfaces",
        "42/42 matrix nodes passed",
        "modules 30/30",
        "smoke 3/3",
        "e2e 8/8",
        "W6 passes all four",
        "490.54 seconds",
        "TTFT passed 4/4",
        "throughput failed 4/4",
        "logical 1x1 execution on the physical P150_X4",
        "zero hardware-lifecycle failures and zero resets",
        "All twelve model manifests remain `experimental`",
    ):
        if required not in report_text:
            raise ValueError(f"compatibility report is missing hardware verdict {required!r}")
    for artifact in matrix["package_artifacts"]:
        if artifact["sha256"] not in report_text:
            raise ValueError(f"compatibility report is missing package digest: {artifact['filename']}")
    for model in matrix["models"]:
        if f"`{model['hf_revision']}`" not in report_text:
            raise ValueError(f"compatibility report is missing model revision: {model['package']}")
    for target in re.findall(r"\]\(([^)]+)\)", report_text):
        local_target = target.split("#", 1)[0]
        if not local_target or local_target.startswith(("http://", "https://")):
            continue
        if not (REPORT.parent / local_target).resolve().is_file():
            raise ValueError(f"missing compatibility-report link: {target}")


def render(matrix: dict[str, Any]) -> str:
    return json.dumps(matrix, indent=2, sort_keys=True) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    generated = build_matrix()
    if args.write:
        OUTPUT.write_text(render(generated), encoding="utf-8")
    if not OUTPUT.is_file():
        raise FileNotFoundError(f"missing compatibility matrix: {OUTPUT}")
    matrix = json.loads(OUTPUT.read_text(encoding="utf-8"))
    validate(matrix)
    if OUTPUT.read_text(encoding="utf-8") != render(matrix):
        raise ValueError("compatibility matrix JSON is not deterministically formatted")
    print(
        f"validated {len(matrix['surfaces'])} surfaces, {len(matrix['models'])} models, "
        f"{matrix['api_availability']['unique_module_paths']} TTNN APIs, "
        f"{len(matrix['api_availability']['unstable_paths'])} unstable APIs, and "
        f"{matrix['hardware_qualification']['canonical_index']['record_count']} "
        "same-SHA hardware records"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
