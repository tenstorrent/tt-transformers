#!/usr/bin/env python3
"""Generate and validate the authoritative TTNN 0.77 compatibility matrix."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "qualification/reports/ttnn-0.77.0-compatibility-matrix.json"
REPORT = ROOT / "qualification/reports/ttnn-0.77.0.md"
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
        "Device/tensor/mesh helpers, typed environment/cache policy, program-config serialization, and ownership logic.",
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
        "Prefill/decode planning, program/trace identity, tensor cleanup, output reading, cache geometry, and adapters with fakes/mocks.",
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
            dependency["isolation"]["pythonpath_unset"]
            and not dependency["isolation"]["repo_checkout_on_sys_path"]
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
    return {
        "package": model["package"],
        "hf_id": model["hf_id"],
        "hf_revision": model["hf_revision"],
        "lifecycle_status": manifest["status"],
        "host_verdict": "pass_import_py310_py312_and_host_contracts_py310_py312",
        "host_import_pythons": ["3.10.19", "3.12.13"],
        "host_semantics_pythons": ["3.10.19", "3.12.13"],
        "hardware_verdict": "not_qualified",
        "firmware_driver_verdict": "not_evaluated",
        "geometry_verdict": "source_declared_not_hardware_validated",
        "declared_hardware_targets": manifest["hardware"],
        "known_gaps": manifest["known_gaps"],
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
        "| 3.10.19 | 2,115 | 28 | 6,791 | 81 | 0 | 0 |",
        "| 3.12.13 | 2,115 | 28 | 6,791 | 81 | 0 | 0 |",
        "nanobind: leaked 8 instances!",
        "nanobind: leaked 36 types!",
        "nanobind: leaked 330 functions!",
        "not counted as a test failure or collection error",
        "jsonschema 4.26.0",
        "pytz 2026.3.post1",
    ):
        if required not in host_text:
            raise ValueError(f"host support report is missing final evidence {required!r}")
    package_text = (ROOT / "qualification/reports/package-wheel.md").read_text(encoding="utf-8")
    for required in (
        artifacts["source_tree"]["sha256"],
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
            "hardware_verdict": (
                "not_applicable" if identifier in {"package_artifact", "package_root"} else "not_qualified"
            ),
            "evidence": evidence_record(evidence),
        }
        for identifier, category, verdict, scope, evidence in SURFACE_DEFINITIONS
    ]
    return {
        "schema_version": 2,
        "verdict": "host_compatible_with_ttnn_0_77_hardware_unqualified",
        "validation_date": "2026-09-02",
        "tt_transformers": "0.1.0.dev0",
        "ttnn": "0.77.0",
        "source_tree": artifacts["source_tree"],
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
                "passed": 2115,
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
                "passed": 2115,
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
                    "leaked_instances": 8,
                    "leaked_types": 36,
                    "leaked_functions": 330,
                    "test_failure": False,
                    "message": "this is likely caused by a reference counting issue in the binding code.",
                    "retained_output": "/tmp/gwang/tttv2-py312-host-final.log",
                },
                "evidence": evidence_record(
                    [
                        "qualification/reports/host-ttnn-0.77.md",
                        "qualification/reports/ttnn-0.77.0-host-triage.md",
                    ]
                ),
            },
        ],
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
                "finding": "Python 3.12 reports leaked TTNN nanobind instances, types, and functions during interpreter shutdown",
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
            "TT tensor kernels or numerical tensor results on a Tenstorrent device",
            "experimental/private collective, RoPE, paged-cache, minimal-matmul, or flatbuffer operation semantics",
            "trace capture/replay, command queues, asynchronous reads, or fabric on hardware",
            "device sampling, paged KV mutation, tensor-cache warm/cold behavior, or cleanup of real TT resources",
            "HF checkpoint download/load, weight conversion/materialization, token/text correctness, accuracy, or performance",
            "firmware, driver, architecture, SKU, mesh, TP/DP, batch/context, or geometry compatibility",
        ],
        "remaining_hardware_gaps": {
            "overall": "no_hardware_qualification_for_pinned_standalone_revision",
            "firmware_driver": "versions_and_compatibility_not_evaluated",
            "unstable_api_semantics": "14_paths_present_but_device_behavior_not_evaluated",
            "model_geometry": "manifest_targets_are_source_declared_only; see each model known_gaps",
            "required_next_evidence": [
                "serialized reusable-module hardware gates on representative Wormhole and Blackhole",
                "runtime eager/trace/cache/sampling/cleanup gates with real devices",
                "per-model correctness on each claimed SKU/mesh/TP/DP and context bucket",
                "recorded firmware/driver/TTNN/wheel/checkpoint/cache identity for every run",
            ],
            "evidence": evidence_record(
                ["qualification/reports/host-ttnn-0.77.md", "qualification/analysis/support/support_baseline.md"]
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
            ]
        ),
    }


def validate(matrix: dict[str, Any]) -> None:
    generated = build_matrix()
    if matrix != generated:
        raise ValueError("compatibility matrix differs from deterministic authoritative-input projection")
    if {surface["id"] for surface in matrix["surfaces"]} != EXPECTED_SURFACES:
        raise ValueError("base/module/runtime surface verdict coverage is incomplete")
    for surface in matrix["surfaces"]:
        if not surface["host_verdict"] or not surface["evidence"]:
            raise ValueError(f"surface lacks verdict/evidence: {surface['id']}")
        if surface["host_semantics_pythons"] != ["3.10.19", "3.12.13"]:
            raise ValueError(f"surface lacks two-Python host verdict: {surface['id']}")
    source_models = {
        path.parent.name
        for path in (ROOT / "src/tt_transformers/models").glob("*/model.py")
    }
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
        expected = (2115, 28, 6791, 5, 81, 0, 0, 0)
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
    feedback = matrix["host_suites"][1]["shutdown_diagnostics"]
    if (
        feedback["classification"] != "ttnn_nanobind_binding_lifecycle_feedback"
        or feedback["test_failure"]
        or (feedback["leaked_instances"], feedback["leaked_types"], feedback["leaked_functions"])
        != (8, 36, 330)
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
    for evidence in matrix["authoritative_inputs"] + matrix["remaining_hardware_gaps"]["evidence"]:
        if not (ROOT / evidence["path"]).is_file():
            raise ValueError(f"missing evidence: {evidence['path']}")
    report_text = REPORT.read_text(encoding="utf-8")
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
        f"and {len(matrix['api_availability']['unstable_paths'])} unstable APIs"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
