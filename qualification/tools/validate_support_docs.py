#!/usr/bin/env python3
"""Validate experimental example manifests and generated support documentation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[2]
MODELS = (
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
)
PINNED_SHA = "00748e6ac7b65f50e5c2af07f6e7c1c535c7f4c0"
FINAL_HARDWARE_SHA = "73d414f8b826a7da982df8c8229d4ac41ed8ba33"
FINAL_HARDWARE_INDEX = ROOT / "qualification/evidence/hardware" / FINAL_HARDWARE_SHA / "index.json"
FINAL_HARDWARE_INDEX_SHA256 = "4e98b62c34fb9f8744e00624c091dc9de18b3f32c5cd74f2ad2be1ad26274d7a"
PREVIOUS_HARDWARE_SHA = "2883a949860d749adc2ed1af5525b27a9a547505"
PREVIOUS_HARDWARE_INDEX = ROOT / "qualification/evidence/hardware" / PREVIOUS_HARDWARE_SHA / "index.json"
SUPERSEDED_HARDWARE_SHA = "b24eabe35c8f2c73f45493da40e5a6351eb0ec2d"
SUPERSEDED_HARDWARE_INDEX = ROOT / "qualification/evidence/hardware" / SUPERSEDED_HARDWARE_SHA / "index.json"
HISTORICAL_HARDWARE_SHA = "ba7abefba4484689c953ac53fe8810322db1d184"
HISTORICAL_HARDWARE_INDEX = ROOT / "qualification/evidence/hardware" / HISTORICAL_HARDWARE_SHA / "index.json"
HARDWARE_MATRIX = ROOT / "qualification/manifests/hardware-matrix.json"
HARDWARE_EVIDENCE_CSV = ROOT / "qualification/analysis/support/hardware_evidence.csv"
ACCURACY_TTFT_DIAGNOSTIC = (
    ROOT / "qualification/evidence/diagnostics" / PREVIOUS_HARDWARE_SHA / "accuracy-ttft/summary.json"
)
RELAXED_W6_DIAGNOSTIC_SHA = "d7677f822356e839f707a6447fd0abc89e620d56"
EXPECTED_TEARDOWN = "process_exited; fixture teardown not independently hardware-verified"
REQUIRED_HEADINGS = (
    "### Purpose",
    "### Status and checkpoint",
    "### Candidate software tuple",
    "### Declared hardware geometry",
    "### Proven limits and features",
    "### Checkpoint, cache, and offline requirements",
    "### Install, run, and collect",
    "### Correctness criterion",
    "### Validation and evidence",
    "### Known and unsupported gaps",
)


def _validate_hardware_authorities(errors: list[str]) -> None:
    if not FINAL_HARDWARE_INDEX.is_file():
        errors.append(f"centralized final-SHA hardware index is missing: {FINAL_HARDWARE_INDEX}")
        return
    if not HARDWARE_MATRIX.is_file():
        errors.append(f"hardware matrix is missing: {HARDWARE_MATRIX}")
        return

    hardware_index = json.loads(FINAL_HARDWARE_INDEX.read_text())
    matrix = json.loads(HARDWARE_MATRIX.read_text())
    records = hardware_index.get("records", [])
    matrix_nodes = matrix.get("nodes", [])
    if hardware_index.get("candidate_sha") != FINAL_HARDWARE_SHA:
        errors.append(f"centralized hardware candidate SHA drifted: {hardware_index.get('candidate_sha')}")
    index_sha256 = hashlib.sha256(FINAL_HARDWARE_INDEX.read_bytes()).hexdigest()
    if index_sha256 != FINAL_HARDWARE_INDEX_SHA256:
        errors.append(f"centralized hardware index SHA-256 drifted: {index_sha256}")
    if len(records) != 42 or len({record.get("node") for record in records}) != 42:
        errors.append(f"centralized hardware record count/identity drifted: {len(records)}")
    if any(record.get("full_sha") != FINAL_HARDWARE_SHA for record in records):
        errors.append("centralized hardware bundle contains a mixed-SHA record")

    outcomes = Counter(record.get("outcome") for record in records)
    expected_outcomes = Counter({"passed": 42})
    if outcomes != expected_outcomes:
        errors.append(f"centralized hardware outcome count drifted: {dict(outcomes)}")
    stage_outcomes = Counter((record.get("stage"), record.get("outcome")) for record in records)
    expected_stage_outcomes = Counter(
        {
            ("module", "passed"): 30,
            ("smoke", "passed"): 3,
            ("e2e", "passed"): 8,
            ("runtime", "passed"): 1,
        }
    )
    if stage_outcomes != expected_stage_outcomes:
        errors.append(f"centralized hardware stage outcomes drifted: {dict(stage_outcomes)}")
    mesh_outcomes = Counter((record.get("mesh_device"), record.get("outcome")) for record in records)
    expected_mesh_outcomes = Counter(
        {
            ("N150", "passed"): 9,
            ("N300", "passed"): 6,
            ("T3K", "passed"): 8,
            ("P150", "passed"): 8,
            ("P150x4", "passed"): 11,
        }
    )
    if mesh_outcomes != expected_mesh_outcomes:
        errors.append(f"centralized hardware mesh outcomes drifted: {dict(mesh_outcomes)}")
    machine_outcomes = Counter((record.get("machine_pool_entry"), record.get("outcome")) for record in records)
    expected_machine_outcomes = Counter({("wh-lb-42", "passed"): 23, ("bh-qb-05", "passed"): 19})
    if machine_outcomes != expected_machine_outcomes:
        errors.append(f"centralized hardware host outcomes drifted: {dict(machine_outcomes)}")

    w6_records = [record for record in records if record.get("node") == "wh-t3k-runtime-trace-order"]
    if len(w6_records) != 1 or w6_records[0].get("outcome") != "passed":
        errors.append(f"centralized strict W6 pass drifted: {w6_records}")
    else:
        raw_path = FINAL_HARDWARE_INDEX.parent / w6_records[0]["source"]["evidence_json"]["path"]
        if not raw_path.is_file():
            errors.append(f"strict W6 evidence is missing: {raw_path}")
        else:
            raw = json.loads(raw_path.read_text())
            expected_selector = {
                "target": (
                    "tests/models/llama33_70b/test_t3k_batched_prefill_correctness.py::"
                    "test_w6_active15_padded16_trace_correctness"
                )
            }
            if raw.get("selector") != expected_selector:
                errors.append(f"canonical W6 selector is not the official strict node: {raw.get('selector')}")
            if raw.get("failure_classification") != "passed" or raw.get("exit_code") != 0:
                errors.append("canonical W6 result is not the recorded strict pass")
            environment = raw.get("environment", {})
            forbidden = [
                key
                for key in environment
                if key == "DISABLE_MINIMAL_MATMUL" or key.startswith("W6_LOGITS_") or key.startswith("W6_DECODE_")
            ]
            if forbidden:
                errors.append(f"canonical W6 pass contains forbidden diagnostic overrides: {forbidden}")
            log_path = FINAL_HARDWARE_INDEX.parent / w6_records[0]["source"]["stdout_log"]["path"]
            if log_path.is_file() and "4 passed in 490.54s" not in log_path.read_text():
                errors.append("canonical W6 log does not contain the four-case strict pass")

    for record in records:
        if record.get("reset") != {
            "automatic": False,
            "performed": False,
            "reason": "runner never resets hardware",
        }:
            errors.append(f"hardware reset record drifted: {record.get('node')}")
        if record.get("teardown_status") != EXPECTED_TEARDOWN:
            errors.append(f"hardware teardown record drifted: {record.get('node')}")
        for artifact in ("evidence_json", "stdout_log"):
            path = FINAL_HARDWARE_INDEX.parent / record["source"][artifact]["path"]
            if not path.is_file():
                errors.append(f"hardware evidence artifact is missing: {path}")

    executed = {record.get("node") for record in records}
    matrix_ids = {node.get("id") for node in matrix_nodes}
    if executed != matrix_ids:
        errors.append(f"current hardware execution does not cover the full matrix: {sorted(matrix_ids - executed)}")

    p150_records = [record for record in records if record.get("mesh_device") == "P150"]
    if len(p150_records) != 8 or any(record.get("machine_pool_entry") != "bh-qb-05" for record in p150_records):
        errors.append("logical single-P150 record provenance drifted")
    for record in p150_records:
        raw_path = FINAL_HARDWARE_INDEX.parent / record["source"]["evidence_json"]["path"]
        if not raw_path.is_file():
            continue
        raw = json.loads(raw_path.read_text())
        inventory = raw.get("physical_inventory", {})
        environment = raw.get("environment", {})
        if (
            environment.get("MESH_DEVICE") != "P150"
            or environment.get("TT_VISIBLE_DEVICES") is not None
            or inventory.get("cluster_type") != "P150_X4"
            or inventory.get("system_mesh") != "2x2"
            or inventory.get("device_count") != 4
        ):
            errors.append(f"logical single-P150 physical boundary drifted: {record.get('node')}")

    summary = hardware_index.get("summary", {})
    if summary.get("total") != 42 or summary.get("outcomes") != {
        "functional_failure": 0,
        "hardware_lifecycle_failure": 0,
        "missing_acceptance_data": 0,
        "passed": 42,
        "pre_device_failure": 0,
    }:
        errors.append(f"canonical hardware summary drifted: {summary.get('outcomes')}")

    if not PREVIOUS_HARDWARE_INDEX.is_file():
        errors.append(f"superseded 2883 hardware index is missing: {PREVIOUS_HARDWARE_INDEX}")
    else:
        previous = json.loads(PREVIOUS_HARDWARE_INDEX.read_text())
        if (
            previous.get("candidate_sha") != PREVIOUS_HARDWARE_SHA
            or previous.get("summary", {}).get("total") != 34
            or previous.get("summary", {}).get("outcomes", {}).get("passed") != 34
        ):
            errors.append("superseded 2883 hardware evidence drifted")

    if not SUPERSEDED_HARDWARE_INDEX.is_file():
        errors.append(f"superseded b24 hardware index is missing: {SUPERSEDED_HARDWARE_INDEX}")
    else:
        superseded = json.loads(SUPERSEDED_HARDWARE_INDEX.read_text())
        if (
            superseded.get("candidate_sha") != SUPERSEDED_HARDWARE_SHA
            or superseded.get("summary", {}).get("total") != 34
            or superseded.get("summary", {}).get("outcomes", {}).get("passed") != 33
            or superseded.get("summary", {}).get("outcomes", {}).get("functional_failure") != 1
        ):
            errors.append("superseded b24 hardware evidence drifted")

    if not HISTORICAL_HARDWARE_INDEX.is_file():
        errors.append(f"historical ba7 hardware index is missing: {HISTORICAL_HARDWARE_INDEX}")
    else:
        historical = json.loads(HISTORICAL_HARDWARE_INDEX.read_text())
        if (
            historical.get("candidate_sha") != HISTORICAL_HARDWARE_SHA
            or historical.get("summary", {}).get("total") != 34
        ):
            errors.append("historical ba7 hardware evidence drifted")

    if not HARDWARE_EVIDENCE_CSV.is_file():
        errors.append(f"hardware evidence CSV is missing: {HARDWARE_EVIDENCE_CSV}")
        return
    with HARDWARE_EVIDENCE_CSV.open(newline="", encoding="utf-8") as handle:
        ledger = list(csv.DictReader(handle))
    current = [row for row in ledger if row["evidence_sha"] == FINAL_HARDWARE_SHA]
    current_executed = [row for row in current if row["eligible_for_pinned_baseline"] == "true"]
    if len(current_executed) != 42 or {row["scope"] for row in current_executed} != executed:
        errors.append("hardware evidence CSV does not contain the 42 current execution records")
    if any(row["result"] != "passed" for row in current_executed):
        errors.append("current eligible hardware evidence contains a non-pass result")
    indexed_sources = {
        record["node"]: str(
            FINAL_HARDWARE_INDEX.parent.joinpath(record["source"]["evidence_json"]["path"]).relative_to(ROOT)
        )
        for record in records
    }
    if any(row["source"] != indexed_sources.get(row["scope"]) for row in current_executed):
        errors.append("current hardware evidence CSV source paths drifted from the canonical index")
    if len(current) != 42:
        errors.append(f"hardware evidence CSV current-candidate row count drifted: {len(current)}")
    previous_rows = [row for row in ledger if row["evidence_sha"] == PREVIOUS_HARDWARE_SHA]
    if len(previous_rows) != 42 or any(row["eligible_for_pinned_baseline"] != "false" for row in previous_rows):
        errors.append("hardware evidence CSV does not preserve 2883 as 42 ineligible superseded rows")
    superseded_rows = [row for row in ledger if row["evidence_sha"] == SUPERSEDED_HARDWARE_SHA]
    if len(superseded_rows) != 42 or any(row["eligible_for_pinned_baseline"] != "false" for row in superseded_rows):
        errors.append("hardware evidence CSV does not preserve b24 as 42 ineligible superseded rows")
    historical_rows = [row for row in ledger if row["evidence_sha"] == HISTORICAL_HARDWARE_SHA]
    if len(historical_rows) != 42 or any(row["eligible_for_pinned_baseline"] != "false" for row in historical_rows):
        errors.append("hardware evidence CSV does not preserve ba7 as 42 ineligible historical rows")
    if any(row["evidence_sha"] == RELAXED_W6_DIAGNOSTIC_SHA for row in ledger):
        errors.append("non-qualifying relaxed W6 diagnostic must not enter the hardware evidence CSV")

    if not ACCURACY_TTFT_DIAGNOSTIC.is_file():
        errors.append(f"accuracy TTFT diagnostic summary is missing: {ACCURACY_TTFT_DIAGNOSTIC}")
        return
    diagnostic = json.loads(ACCURACY_TTFT_DIAGNOSTIC.read_text())
    cells = diagnostic.get("cells", [])
    if (
        diagnostic.get("candidate_sha") != PREVIOUS_HARDWARE_SHA
        or diagnostic.get("canonical_hardware_evidence") is not False
        or diagnostic.get("classification") != "noncanonical_performance_diagnostic"
        or len(cells) != 4
    ):
        errors.append("accuracy TTFT diagnostic identity/classification drifted")
    ttft_values = sorted(cell.get("ttft_ms") for cell in cells)
    if ttft_values != [86.7, 86.9, 87.1, 87.2] or any(cell.get("ttft_result") != "passed" for cell in cells):
        errors.append(f"accuracy TTFT diagnostic pass/range drifted: {ttft_values}")
    if any(cell.get("throughput_result") != "performance_floor_failure" for cell in cells):
        errors.append("accuracy TTFT diagnostic throughput disposition drifted")
    if diagnostic.get("summary") != {
        "cell_count": 4,
        "throughput": {"failed": 4, "passed": 0, "result": "performance_floor_failure"},
        "ttft": {"failed": 0, "passed": 4, "range_ms": [86.7, 87.2], "result": "passed"},
    }:
        errors.append("accuracy TTFT diagnostic summary counts drifted")


def validate() -> list[str]:
    errors: list[str] = []
    schema_path = ROOT / "qualification/schemas/support-manifest.schema.json"
    schema = json.loads(schema_path.read_text())
    try:
        Draft202012Validator.check_schema(schema)
    except Exception as error:
        errors.append(f"support schema is invalid: {error}")
        return errors
    validator = Draft202012Validator(schema)

    _validate_hardware_authorities(errors)

    manifests = {}
    paths = sorted((ROOT / "examples").glob("*/support.json"))
    found = {path.parent.name for path in paths}
    if found != set(MODELS):
        errors.append(f"support manifest packages differ: expected={sorted(MODELS)} found={sorted(found)}")

    for path in paths:
        model = path.parent.name
        manifest = json.loads(path.read_text())
        manifests[model] = manifest
        for error in validator.iter_errors(manifest):
            errors.append(f"{path}:{'/'.join(map(str, error.absolute_path))}: {error.message}")
        if manifest.get("status") != "experimental":
            errors.append(f"{path}: active extraction status must remain experimental")
        validation = manifest.get("validation")
        if validation != {"date": None, "git_sha": None, "evidence": []}:
            errors.append(f"{path}: unsupported validation/evidence claim: {validation}")
        revision = manifest.get("model", {}).get("hf_revision")
        if revision == "UNPINNED":
            if "HF revision is not pinned" not in manifest.get("known_gaps", []):
                errors.append(f"{path}: UNPINNED revision lacks explicit known gap")
        elif not isinstance(revision, str) or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
            errors.append(f"{path}: revision must be UNPINNED or a 40-character lowercase SHA")
        if manifest.get("software") != {
            "tt_transformers": "0.1.0.dev0",
            "ttnn": "0.77.0",
            "python": ["3.10", "3.12"],
            "torch": "2.11.0",
            "transformers": "5.12.1",
        }:
            errors.append(f"{path}: candidate software tuple drifted")
        if not manifest.get("hardware"):
            errors.append(f"{path}: no declared candidate hardware rows")
        assets = manifest.get("assets", {})
        if assets.get("remote_code_default") is not False:
            errors.append(f"{path}: remote-code default must be false")
        if assets.get("cwd_relative_cache_default") is not False:
            errors.append(f"{path}: cwd-relative cache default must be false")
        if assets.get("cache_resolver") != "tt_transformers.cache_environment.resolve_model_cache_path":
            errors.append(f"{path}: standalone cache resolver is missing")
        if assets.get("tt_cache_path_semantics") != "append topology exactly once":
            errors.append(f"{path}: TT_CACHE_PATH topology semantics drifted")

        readme_path = path.with_name("README.md")
        readme = readme_path.read_text()
        if readme.count("<!-- BEGIN GENERATED SUPPORT -->") != 1 or readme.count("<!-- END GENERATED SUPPORT -->") != 1:
            errors.append(f"{readme_path}: generated support markers must occur exactly once")
        for heading in REQUIRED_HEADINGS:
            if heading not in readme:
                errors.append(f"{readme_path}: missing {heading}")
        required_text = (
            manifest["model"]["hf_id"],
            "experimental",
            "concrete and runnable",
            "ttnn==0.77.0",
            "torch==2.11.0",
            "transformers==5.12.1",
            "HF_HUB_OFFLINE=1",
            "TRANSFORMERS_OFFLINE=1",
            "TT_TRANSFORMERS_CACHE",
            "trust_remote_code` is not enabled by default",
            f"python -m examples.{model}.demo",
            f"tests/hardware/models/{model}/test_demo.py",
            "Correctness criterion",
            f"pinned source revision `{PINNED_SHA}`",
            "Last standalone hardware validation date: **none**",
        )
        for token in required_text:
            if token not in readme:
                errors.append(f"{readme_path}: missing required token {token!r}")
        if revision == "UNPINNED" and "UNPINNED — release blocker" not in readme:
            errors.append(f"{readme_path}: unpinned blocker is not prominent")
        if revision != "UNPINNED" and revision not in readme:
            errors.append(f"{readme_path}: pinned revision missing")
        for hardware in manifest.get("hardware", []):
            row_tokens = (
                hardware["architecture"],
                hardware["sku"],
                hardware["mesh"],
                str(hardware["tp"]),
                str(hardware["dp"]),
            )
            if not all(token in readme for token in row_tokens):
                errors.append(f"{readme_path}: hardware row is not rendered: {hardware}")
        for limit in manifest.get("limits", {}).get("source_proven", []):
            if limit not in readme:
                errors.append(f"{readme_path}: source-proven limit missing: {limit}")
        for target in (
            path,
            ROOT / "tests/hardware/models" / model / "test_demo.py",
            ROOT / "qualification/analysis/support/support_baseline.md",
            ROOT / "qualification/analysis/support/hardware_evidence.csv",
            FINAL_HARDWARE_INDEX,
        ):
            if not target.exists():
                errors.append(f"{readme_path}: linked target missing: {target}")

    root_matrix = (ROOT / "SUPPORT.md").read_text()
    example_matrix = (ROOT / "examples/README.md").read_text()
    for matrix_name, matrix in (("SUPPORT.md", root_matrix), ("examples/README.md", example_matrix)):
        for token in (
            "Current final-SHA subset evidence",
            FINAL_HARDWARE_SHA,
            PREVIOUS_HARDWARE_SHA,
            "42 passing nodes",
            "official strict W6",
            "30 module nodes",
            "all three smoke nodes",
            "all eight end-to-end/token-accuracy nodes",
            "MESH_DEVICE=P150",
            "logical 1x1",
            "not standalone-P150 product evidence",
            "P150_X4 evidence",
            "accuracy-TTFT diagnostic",
            "TTFT passed 4/4",
            "throughput produced `performance_floor_failure` in all 4/4 cells",
            RELAXED_W6_DIAGNOSTIC_SHA,
            "non-qualifying",
            "excluded from the canonical pass count",
            SUPERSEDED_HARDWARE_SHA,
            HISTORICAL_HARDWARE_SHA,
            "do not qualify any model or geometry"
            if matrix_name == "SUPPORT.md"
            else "not qualification of an example",
        ):
            if token not in matrix:
                errors.append(f"{matrix_name}: missing centralized evidence token {token!r}")

    authority_docs = {
        "qualification/evidence/hardware/README.md": (
            "Current candidate",
            "Superseded candidate",
            "Historical candidate",
            FINAL_HARDWARE_SHA,
            FINAL_HARDWARE_INDEX_SHA256,
            PREVIOUS_HARDWARE_SHA,
            SUPERSEDED_HARDWARE_SHA,
            HISTORICAL_HARDWARE_SHA,
            RELAXED_W6_DIAGNOSTIC_SHA,
            "not a qualification pass",
            "official strict",
            "TTFT passed 4/4",
            "performance_floor_failure",
            "excluded from the current 42",
            "current canonical index",
            "MESH_DEVICE=P150",
            "standalone-P150 product",
        ),
        "qualification/reports/hardware-readiness.md": (
            FINAL_HARDWARE_SHA,
            FINAL_HARDWARE_INDEX_SHA256,
            PREVIOUS_HARDWARE_SHA,
            SUPERSEDED_HARDWARE_SHA,
            HISTORICAL_HARDWARE_SHA,
            RELAXED_W6_DIAGNOSTIC_SHA,
            "42/42 passed",
            "30/30",
            "1/1",
            "3/3",
            "8/8",
            "Noncanonical accuracy-TTFT diagnostic",
            "TTFT passed 4/4",
            "performance_floor_failure",
            "42-record index",
            "MESH_DEVICE=P150",
            "standalone-P150 product",
            "excluded from the",
            "current canonical index and pass count",
        ),
    }
    for relative_path, tokens in authority_docs.items():
        path = ROOT / relative_path
        if not path.is_file():
            errors.append(f"hardware authority is missing: {relative_path}")
            continue
        text = path.read_text()
        for token in tokens:
            if token not in text:
                errors.append(f"{relative_path}: missing final hardware token {token!r}")
    for model in MODELS:
        manifest = manifests.get(model)
        if not manifest:
            continue
        for matrix_name, matrix in (("SUPPORT.md", root_matrix), ("examples/README.md", example_matrix)):
            if matrix.count(f"[{model}]") != 1:
                errors.append(f"{matrix_name}: {model} row must occur exactly once")
            if manifest["model"]["hf_id"] not in matrix:
                errors.append(f"{matrix_name}: missing HF ID for {model}")
        if "none at pinned SHA" not in root_matrix or "none at pinned SHA" not in example_matrix:
            errors.append("support matrices must state evidence absence explicitly")
    if "| qualified |" in root_matrix.lower() or "| qualified |" in example_matrix.lower():
        errors.append("support matrix contains an unsupported qualified row")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    errors = validate()
    if errors:
        print("Support documentation validation FAILED:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print("Validated 12 experimental support manifests, READMEs, and both support matrices.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
