# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Host-only validation for serialized hardware qualification readiness."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest

from qualification.tools import run_hardware_matrix as runner

pytestmark = pytest.mark.host


def _matrix():
    return runner.load_json(runner.DEFAULT_MATRIX)


def _wh_inventory():
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


def _dry_args(tmp_path):
    return SimpleNamespace(
        common_sha="a" * 40,
        branch="qualified-branch",
        machine_identity="wh-lb-42.yyz2.tenstorrent.com",
        sync_gate_passed=True,
        output_dir=tmp_path / "results",
        checkout=runner.ROOT,
        python="/qualified/python",
    )


@pytest.mark.host
def test_checked_in_matrix_validates_and_covers_every_required_mesh():
    counts = runner.validate_matrix(_matrix())
    assert counts == {"N150": 9, "N300": 6, "T3K": 8, "P150": 8, "P150x4": 11}
    assert sum(counts.values()) == 42


@pytest.mark.host
def test_matrix_requires_serial_execution_and_forbids_automatic_reset():
    matrix = _matrix()
    assert matrix["serialization"]["max_concurrent_processes"] == 1
    assert matrix["serialization"]["automatic_reset"] is False
    assert "tt-smi -r" not in runner.Path(runner.__file__).read_text()

    parallel = copy.deepcopy(matrix)
    parallel["serialization"]["max_concurrent_processes"] = 2
    with pytest.raises(runner.MatrixError, match="exactly one"):
        runner.validate_matrix(parallel)

    resetting = copy.deepcopy(matrix)
    resetting["serialization"]["automatic_reset"] = True
    with pytest.raises(runner.MatrixError, match="automatic reset"):
        runner.validate_matrix(resetting)


@pytest.mark.host
def test_dry_run_serializes_one_exact_node_without_starting_a_process(tmp_path):
    matrix = _matrix()
    node = runner.select_node(matrix, "wh-n150-rmsnorm-prefill")
    result = runner.preview(matrix, node, _dry_args(tmp_path), _wh_inventory())

    assert result["classification"] == "not_executed_dry_run"
    assert result["node"] == node["id"]
    assert result["command"][-1] == node["selector"]["target"]
    assert result["environment"]["MESH_DEVICE"] == "N150"
    assert result["environment"]["TT_CACHE_PATH"].endswith(node["id"])
    assert result["automatic_reset"] is False
    assert not (tmp_path / "results").exists()


@pytest.mark.host
def test_attestation_requires_full_sha_external_gate_and_allowed_machine():
    matrix = _matrix()
    node = runner.select_node(matrix, "wh-n150-rmsnorm-prefill")
    with pytest.raises(runner.MatrixError, match="not attested"):
        runner.validate_attestation(
            matrix,
            node,
            common_sha="a" * 40,
            branch="branch",
            machine_identity="wh-lb-42.yyz2.tenstorrent.com",
            sync_gate_passed=False,
        )
    with pytest.raises(runner.MatrixError, match="40 lowercase"):
        runner.validate_attestation(
            matrix,
            node,
            common_sha="short",
            branch="branch",
            machine_identity="wh-lb-42.yyz2.tenstorrent.com",
            sync_gate_passed=True,
        )
    with pytest.raises(runner.MatrixError, match="not valid"):
        runner.validate_attestation(
            matrix,
            node,
            common_sha="a" * 40,
            branch="branch",
            machine_identity="bh-lb-11.yyz2.tenstorrent.com",
            sync_gate_passed=True,
        )


@pytest.mark.host
def test_existing_lock_refuses_a_second_process(tmp_path):
    lock = tmp_path / "hardware.lock"
    lock.write_text("another-pid\n")
    with pytest.raises(runner.MatrixError, match="already exists"):
        with runner.ProcessLock(lock):
            pass
    assert lock.read_text() == "another-pid\n"


@pytest.mark.host
def test_process_lock_is_removed_after_serial_owner_exits(tmp_path):
    lock = tmp_path / "hardware.lock"
    with runner.ProcessLock(lock):
        assert lock.exists()
    assert not lock.exists()


@pytest.mark.host
@pytest.mark.parametrize(
    "argv",
    [
        ["pytest", "-n", "2", "test.py"],
        ["pytest", "--numprocesses=auto", "test.py"],
        ["pytest", "--dist=load", "test.py"],
    ],
)
def test_parallel_pytest_arguments_are_refused(argv):
    assert runner.contains_parallel_pytest_args(argv)


@pytest.mark.host
@pytest.mark.parametrize(
    "exit_code,output,timed_out,expected",
    [
        (0, "passed", False, "passed"),
        (1, "assert PCC failed", False, "functional_failure"),
        (1, "device unresponsive; reset required", False, "hardware_lifecycle_failure"),
        (None, "", True, "hardware_lifecycle_failure"),
    ],
)
def test_failure_classification_is_separate(exit_code, output, timed_out, expected):
    assert runner.classify_failure(exit_code, output, timed_out) == expected


@pytest.mark.host
def test_p150x4_inventory_rejects_nonphysical_cluster_and_missing_bdfs():
    matrix = _matrix()
    node = runner.select_node(matrix, "bh-p150x4-rmsnorm")
    inventory = {
        "captured_utc": "2026-09-02T00:00:00Z",
        "machine_identity": "bh-lb-11.yyz2.tenstorrent.com",
        "architecture": "blackhole",
        "physical_sku": "arbitrary-submesh",
        "device_count": 8,
        "board_types": ["P150"],
        "cluster_type": "CUSTOM",
        "system_mesh": "2x4",
        "tt_visible_devices": None,
        "source_command": "tt-smi -s",
        "selected_bdfs": [],
    }
    with pytest.raises(runner.MatrixError, match="physical cluster provenance"):
        runner.validate_physical_inventory(
            matrix,
            node,
            "bh-lb-11",
            inventory["machine_identity"],
            inventory,
        )
    inventory["cluster_type"] = "P150_X4"
    with pytest.raises(runner.MatrixError, match="four-BDF Ring"):
        runner.validate_physical_inventory(
            matrix,
            node,
            "bh-lb-11",
            inventory["machine_identity"],
            inventory,
        )


@pytest.mark.host
def test_every_node_uses_the_complete_evidence_schema():
    matrix = _matrix()
    required = matrix["required_evidence_fields"]
    assert {
        "started_utc",
        "branch",
        "full_sha",
        "actual_fqdn",
        "physical_inventory",
        "exit_code",
        "metrics",
        "teardown_status",
        "reset",
    } <= set(required)
    assert all(node["required_evidence_fields"] == required for node in matrix["nodes"])
