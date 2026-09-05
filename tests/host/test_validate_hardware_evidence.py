# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Host-only tests for canonical hardware-evidence ingestion."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from qualification.tools import run_hardware_matrix as runner
from qualification.tools import validate_hardware_evidence as evidence_validator

pytestmark = pytest.mark.host
CANDIDATE_SHA = "a" * 40


def _matrix():
    return runner.load_json(runner.DEFAULT_MATRIX)


def _inventory():
    return {
        "captured_utc": "2026-09-02T00:00:00Z",
        "machine_identity": "wh-lb-42.yyz2.tenstorrent.com",
        "architecture": "wormhole",
        "physical_sku": "T3K",
        "device_count": 8,
        "board_types": ["n300 L", "n300 R"],
        "cluster_type": "T3K",
        "system_mesh": "2x4",
        "tt_visible_devices": None,
        "source_command": "tt-smi -s",
    }


def _expanded_environment(node, root):
    return runner.environment_for(
        node,
        "wh-lb-42",
        checkout=Path("/qualified/checkout"),
        run_root=root,
    )


def _write_record(
    root: Path,
    *,
    node_id: str = "wh-n150-rmsnorm-prefill",
    classification: str = "passed",
    exit_code: int = 0,
    log_text: str = "PCC 0.999 passed\n================ 1 passed in 1.00s ================\n",
    suffix: str = "",
):
    matrix = _matrix()
    node = next(item for item in matrix["nodes"] if item["id"] == node_id)
    stem = f"20260902T000000.000000Z-{node_id}{suffix}"
    evidence_path = root / f"{stem}.json"
    log_path = root / f"{stem}.log"
    environment = _expanded_environment(node, root)
    evidence = {
        "schema_version": 1,
        "started_utc": "2026-09-02T00:00:00Z",
        "finished_utc": "2026-09-02T00:00:01Z",
        "branch": "qualified-branch",
        "full_sha": CANDIDATE_SHA,
        "caller_machine_identity": "wh-lb-42.yyz2.tenstorrent.com",
        "actual_fqdn": "wh-lb-42-special-test-reservation",
        "architecture": node["architecture"],
        "physical_inventory": _inventory(),
        "mesh_device": node["mesh_device"],
        "tt_visible_devices": environment.get("TT_VISIBLE_DEVICES"),
        "selector": node["selector"],
        "command": runner.build_command(node, "/qualified/python"),
        "environment": environment,
        "cache_paths": {"TT_CACHE_PATH": environment.get("TT_CACHE_PATH")},
        "node": node_id,
        "exit_code": exit_code,
        "failure_classification": classification,
        "acceptance_types": node["acceptance_types"],
        "metrics": ["PCC 0.999"] if "PCC" in log_text else [],
        "teardown_status": "process_exited; fixture teardown not independently hardware-verified",
        "reset": {"performed": False, "automatic": False, "reason": "runner never resets hardware"},
        "stdout_log_path": f"/remote/hardware-evidence/{log_path.name}",
        "evidence_json_path": f"/remote/hardware-evidence/{evidence_path.name}",
    }
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
    log_path.write_text(log_text, encoding="utf-8")
    return evidence_path, log_path, evidence


def _build(root: Path):
    return evidence_validator.build_index(
        candidate_sha=CANDIDATE_SHA,
        matrix_path=runner.DEFAULT_MATRIX,
        evidence_root=root,
    )


@pytest.mark.host
def test_builds_schema_valid_index_with_hash_bound_pair_and_summaries(tmp_path):
    evidence_path, log_path, _ = _write_record(tmp_path)

    index = _build(tmp_path)

    Draft202012Validator(json.loads(evidence_validator.DEFAULT_SCHEMA.read_text(encoding="utf-8"))).validate(index)
    assert index["summary"]["total"] == 1
    assert index["summary"]["outcomes"]["passed"] == 1
    assert index["summary"]["by_stage"][0]["name"] == "module"
    assert index["summary"]["by_architecture"][0]["name"] == "wormhole"
    assert index["summary"]["by_mesh_device"][0]["name"] == "N150"
    source = index["records"][0]["source"]
    evidence_hash = hashlib.sha256(evidence_path.read_bytes()).hexdigest()
    log_hash = hashlib.sha256(log_path.read_bytes()).hexdigest()
    assert source["evidence_json"]["sha256"] == evidence_hash
    assert source["stdout_log"]["sha256"] == log_hash
    assert source["pair_sha256"] == hashlib.sha256((evidence_hash + "\0" + log_hash).encode("ascii")).hexdigest()


@pytest.mark.parametrize(
    "classification,exit_code,log_text,expected",
    [
        (
            "functional_failure",
            2,
            "ERROR collecting tests/test_model.py\nInterrupted: 1 error during collection\n",
            "pre_device_failure",
        ),
        ("functional_failure", 1, "FAILED tests/test_model.py::test_model - assert False\n", "functional_failure"),
        ("hardware_lifecycle_failure", -15, "device unresponsive; reset required\n", "hardware_lifecycle_failure"),
        (
            "missing_acceptance_data",
            0,
            "================ 1 passed in 1.00s ================\n",
            "missing_acceptance_data",
        ),
        (
            "no_passing_tests",
            0,
            "================ 1 skipped in 0.01s ================\n",
            "pre_device_failure",
        ),
    ],
)
@pytest.mark.host
def test_distinguishes_pre_device_functional_hardware_and_acceptance_failures(
    tmp_path, classification, exit_code, log_text, expected
):
    _write_record(
        tmp_path,
        classification=classification,
        exit_code=exit_code,
        log_text=log_text,
    )
    index = _build(tmp_path)
    assert index["records"][0]["outcome"] == expected
    assert index["summary"]["outcomes"][expected] == 1


@pytest.mark.host
def test_rejects_candidate_sha_and_matrix_binding_mismatches(tmp_path):
    evidence_path, _, evidence = _write_record(tmp_path)
    evidence["full_sha"] = "b" * 40
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
    with pytest.raises(evidence_validator.EvidenceError, match="does not match candidate"):
        _build(tmp_path)

    evidence["full_sha"] = CANDIDATE_SHA
    evidence["selector"] = {"target": "tests/not-the-matrix.py"}
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
    with pytest.raises(evidence_validator.EvidenceError, match="selector differs from matrix"):
        _build(tmp_path)


@pytest.mark.host
def test_requires_runner_derived_node_local_cache_path(tmp_path):
    evidence_path, _, evidence = _write_record(tmp_path)
    assert evidence["environment"]["TT_CACHE_PATH"] == str(tmp_path / "cache/wh-n150-rmsnorm-prefill")
    evidence["environment"].pop("TT_CACHE_PATH")
    evidence["cache_paths"]["TT_CACHE_PATH"] = None
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")

    with pytest.raises(evidence_validator.EvidenceError, match="environment keys differ from matrix"):
        _build(tmp_path)


@pytest.mark.host
def test_rejects_node_local_cache_path_template_mismatch(tmp_path):
    evidence_path, _, evidence = _write_record(tmp_path)
    wrong_cache_path = tmp_path / "cache/not-the-matrix-node"
    evidence["environment"]["TT_CACHE_PATH"] = str(wrong_cache_path)
    evidence["cache_paths"]["TT_CACHE_PATH"] = str(wrong_cache_path)
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")

    with pytest.raises(
        evidence_validator.EvidenceError,
        match="environment TT_CACHE_PATH does not match matrix template",
    ):
        _build(tmp_path)


@pytest.mark.host
def test_rejects_duplicate_nodes_even_when_records_have_different_names(tmp_path):
    _write_record(tmp_path)
    _write_record(tmp_path, suffix="-retry")
    with pytest.raises(evidence_validator.EvidenceError, match="duplicate evidence for node"):
        _build(tmp_path)


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda evidence: evidence.update(finished_utc=None), "finished_utc must be"),
        (lambda evidence: evidence.update(exit_code=None), "exit_code must be an integer"),
        (lambda evidence: evidence.update(failure_classification=None), "non-final failure_classification"),
        (lambda evidence: evidence.update(teardown_status="not_started"), "teardown_status is missing"),
    ],
)
@pytest.mark.host
def test_rejects_incomplete_runner_records(tmp_path, mutation, match):
    evidence_path, _, evidence = _write_record(tmp_path)
    mutation(evidence)
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
    with pytest.raises(evidence_validator.EvidenceError, match=match):
        _build(tmp_path)


@pytest.mark.host
def test_rejects_missing_or_mispaired_log(tmp_path):
    evidence_path, log_path, evidence = _write_record(tmp_path)
    log_path.unlink()
    with pytest.raises(evidence_validator.EvidenceError, match="paired stdout log is missing"):
        _build(tmp_path)

    log_path.write_text("1 passed\n", encoding="utf-8")
    evidence["stdout_log_path"] = "/remote/hardware-evidence/different.log"
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
    with pytest.raises(evidence_validator.EvidenceError, match="does not name paired log"):
        _build(tmp_path)


@pytest.mark.host
def test_rejects_pass_without_positive_pytest_summary(tmp_path):
    _write_record(tmp_path, log_text="================ 1 skipped in 0.01s ================\n")
    with pytest.raises(evidence_validator.EvidenceError, match="no positive pytest pass summary"):
        _build(tmp_path)


@pytest.mark.host
def test_cli_writes_deterministic_index(tmp_path, capsys):
    _write_record(tmp_path)
    output = tmp_path.parent / "index.json"
    assert (
        evidence_validator.main(
            [
                "--candidate-sha",
                CANDIDATE_SHA,
                "--evidence-root",
                str(tmp_path),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert json.loads(output.read_text(encoding="utf-8"))["summary"]["total"] == 1
    assert "Validated 1 hardware evidence records" in capsys.readouterr().out
