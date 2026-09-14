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


def _bh_qb_inventory():
    return {
        "captured_utc": "2026-09-03T00:00:00Z",
        "machine_identity": "bh-qb-05.yyz2.tenstorrent.com",
        "architecture": "blackhole",
        "physical_sku": "P150_X4 quietbox (four physical P150B boards)",
        "device_count": 4,
        "board_types": ["p150b"],
        "cluster_type": "P150_X4",
        "system_mesh": "2x2",
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
    assert counts == {"N150": 9, "N300": 6, "T3K": 8, "P150": 8, "P150x4": 11, "TG": 1, "BHGLX": 1}
    assert sum(counts.values()) == 44


@pytest.mark.host
def test_the_tg_galaxy_node_is_a_disabled_proposal_and_cannot_be_selected():
    """The one TG node is a proposal, not a gate, and must stay unselectable.

    A 32-board Galaxy is an infrastructure commitment the port cannot make on its
    own: it means claiming a machine pool and extending the serial-reservation
    policy from an 8-device host to a 32-board chassis, which is the one place the
    existing `scope: physical_host` policy may genuinely not stretch. The node is
    checked in `enabled: false` so the shape is reviewable as a diff rather than
    negotiated in the abstract. Whoever enables it should have settled the
    serialization question first -- and will have to delete this test to do it,
    which is the point.
    """

    matrix = _matrix()
    node = next(entry for entry in matrix["nodes"] if entry["mesh_device"] == "TG")
    assert node["id"] == "wh-tg-rmsnorm-2d-qk-norm"
    assert node["enabled"] is False
    assert node["disabled_classification"] == "different_hardware_deferred"
    assert node["machine_pool"] == ["wh-glx6u-05"]

    machine = matrix["machines"]["wh-glx6u-05"]
    assert machine["expected_inventory"]["device_count"] == 32
    assert machine["supported_mesh_devices"] == ["TG"]

    # The Blackhole Galaxy proposal is the only other deferral in the matrix.
    assert sorted(entry["id"] for entry in matrix["nodes"] if not entry["enabled"]) == [
        "bh-glx-topology-probe",
        node["id"],
    ]

    with pytest.raises(runner.MatrixError, match="different_hardware_deferred"):
        runner.select_node(matrix, node["id"])


@pytest.mark.host
def test_all_single_p150_nodes_admit_bh_qb_05_with_only_mesh_selection(tmp_path):
    matrix = _matrix()
    machine = matrix["machines"]["bh-qb-05"]
    assert "P150" in machine["supported_mesh_devices"]

    nodes = [node for node in matrix["nodes"] if node["mesh_device"] == "P150"]
    assert {node["priority"] for node in nodes} == {*range(24, 31), 38}
    for node in nodes:
        assert node["machine_pool"] == ["bh-lb-11", "bh-qb-05"]
        assert node["environment"]["MESH_DEVICE"] == "P150"
        assert node["machine_environment_overrides"]["bh-qb-05"] == {"TT_VISIBLE_DEVICES": None}
        assert node["physical_sku_provenance"]["selection_environment"] == {"MESH_DEVICE": "P150"}

    node = runner.select_node(matrix, "bh-p150-rmsnorm-decode")
    args = _dry_args(tmp_path)
    args.machine_identity = "bh-qb-05.yyz2.tenstorrent.com"
    result = runner.preview(matrix, node, args, _bh_qb_inventory())

    assert result["machine_pool_entry"] == "bh-qb-05"
    assert result["environment"]["MESH_DEVICE"] == "P150"
    assert result["environment"]["TT_VISIBLE_DEVICES"] is None


@pytest.mark.host
def test_matrix_refuses_a_pool_entry_that_does_not_support_the_requested_mesh():
    matrix = copy.deepcopy(_matrix())
    matrix["machines"]["bh-qb-05"]["supported_mesh_devices"].remove("P150")

    with pytest.raises(runner.MatrixError, match="machine mesh support mismatch"):
        runner.validate_matrix(matrix)


@pytest.mark.host
@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("cluster_type", "CUSTOM", "physical cluster provenance"),
        ("system_mesh", "1x1", "physical system_mesh mismatch"),
        (
            "tt_visible_devices",
            "0000:01:00.0",
            "sole topology selector",
        ),
    ],
)
def test_bh_qb_single_p150_rejects_wrong_physical_provenance(field, value, message):
    matrix = _matrix()
    node = runner.select_node(matrix, "bh-p150-rmsnorm-decode")
    inventory = _bh_qb_inventory()
    inventory[field] = value

    with pytest.raises(runner.MatrixError, match=message):
        runner.validate_physical_inventory(
            matrix,
            node,
            "bh-qb-05",
            inventory["machine_identity"],
            inventory,
        )


@pytest.mark.host
def test_matrix_requires_serial_execution_and_forbids_automatic_reset():
    matrix = _matrix()
    assert matrix["serialization"]["max_concurrent_processes"] == 1
    assert matrix["serialization"]["scope"] == "physical_host"
    assert matrix["serialization"]["automatic_reset"] is False
    assert "tt-smi -r" not in runner.Path(runner.__file__).read_text()

    parallel = copy.deepcopy(matrix)
    parallel["serialization"]["max_concurrent_processes"] = 2
    with pytest.raises(runner.MatrixError, match="exactly one"):
        runner.validate_matrix(parallel)

    global_scope = copy.deepcopy(matrix)
    global_scope["serialization"]["scope"] = "global"
    with pytest.raises(runner.MatrixError, match="physical_host"):
        runner.validate_matrix(global_scope)

    resetting = copy.deepcopy(matrix)
    resetting["serialization"]["automatic_reset"] = True
    with pytest.raises(runner.MatrixError, match="automatic reset"):
        runner.validate_matrix(resetting)

    missing_classification = copy.deepcopy(matrix)
    missing_classification["failure_classifications"].remove("no_passing_tests")
    with pytest.raises(runner.MatrixError, match="failure classifications"):
        runner.validate_matrix(missing_classification)


@pytest.mark.host
def test_model_cache_roots_are_namespaced_by_model_family():
    seen = {}
    for node in _matrix()["nodes"]:
        cache = node["cache_requirement"]
        if cache["kind"] != "established_warm_model_cache":
            continue
        model = cache["model"]
        root = node["environment"]["TT_CACHE_PATH"]
        assert root.endswith(f"/{model}"), node["id"]
        assert seen.setdefault(root, model) == model


@pytest.mark.host
def test_llama_p150x4_smoke_uses_the_collected_parameter_id():
    node = runner.select_node(_matrix(), "bh-p150x4-llama33-one-layer-smoke")
    assert node["selector"]["target"].endswith("[physical-BH-TP4-ring]")


@pytest.mark.host
def test_dry_run_serializes_one_exact_node_without_starting_a_process(tmp_path):
    matrix = _matrix()
    node = runner.select_node(matrix, "wh-n150-rmsnorm-prefill")
    result = runner.preview(matrix, node, _dry_args(tmp_path), _wh_inventory())

    assert result["classification"] == "not_executed_dry_run"
    assert result["node"] == node["id"]
    assert result["command"][-1] == node["selector"]["target"]
    assert "--color=no" in result["command"]
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
        (0, "===================== 1 passed in 0.12s =====================", False, "passed"),
        (
            0,
            "================ 2 passed, 1 skipped in 0.12s ================",
            False,
            "passed",
        ),
        (0, "===================== 1 skipped in 0.12s ====================", False, "no_passing_tests"),
        (0, "==================== no tests ran in 0.12s ==================", False, "no_passing_tests"),
        (0, "test_gate.py::test_gate PASSED", False, "no_passing_tests"),
        (
            0,
            "\n".join(
                (
                    "===================== 1 passed in 0.12s =====================",
                    "===================== 1 skipped in 0.13s ====================",
                )
            ),
            False,
            "no_passing_tests",
        ),
        (1, "assert PCC failed", False, "functional_failure"),
        (1, "device unresponsive; reset required", False, "hardware_lifecycle_failure"),
        (None, "", True, "hardware_lifecycle_failure"),
    ],
)
def test_failure_classification_is_separate(exit_code, output, timed_out, expected):
    assert runner.classify_failure(exit_code, output, timed_out) == expected


@pytest.mark.host
def test_unrelated_hardware_terms_across_log_are_a_functional_failure():
    output = "\n".join(
        (
            "Metal | Initializing Fabric (fabric_firmware_initializer.cpp:296)",
            "This will become a hard error in a future release",
            "tests/test_attention.py::test_attention FAILED",
            "E   AssertionError: timeout value did not meet the expected result",
            "===================== 1 failed in 2.14s =====================",
        )
    )

    assert runner.classify_failure(1, output) == "functional_failure"


@pytest.mark.parametrize(
    "signature",
    [
        "Watcher reported a fatal device event",
        "Device 0 is unresponsive",
        "Metal fatal: dispatch core halted",
        "Reset required before the next test",
    ],
)
@pytest.mark.host
def test_local_hardware_fatal_signatures_remain_lifecycle_failures(signature):
    output = f"pytest setup completed\n{signature}\npytest session aborted"

    assert runner.classify_failure(1, output) == "hardware_lifecycle_failure"


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


@pytest.mark.host
def test_the_bh_galaxy_node_is_a_disabled_proposal_and_cannot_be_selected():
    """The Blackhole Galaxy node is a proposal, not a gate, and must stay unselectable.

    Same discipline as the TG node above, for a stronger reason: **nothing in
    this node has been measured.** Its geometry is declared from `tt-metal`'s
    mesh graph descriptors and read off a third-party reference port, and no
    Blackhole Galaxy run has ever been taken by this repository. Checking it in
    disabled makes that shape reviewable as a diff instead of negotiated in the
    abstract, and whoever enables it has to delete this test -- which is the
    conversation, made unavoidable.

    Two things must be settled before that happens, and neither is code:

    * the machine identity is an **example**, not an allocation. Blackhole Galaxy
      nodes are scheduled per window, so `allowed_identities` has to name the
      node the allocation sheet actually granted;
    * the 2D module gates still accept Wormhole only, so the modules this node
      would eventually exercise reject a Blackhole mesh by design. The probe this
      node selects deliberately needs none of them.
    """

    matrix = _matrix()
    node = next(entry for entry in matrix["nodes"] if entry["mesh_device"] == "BHGLX")
    assert node["id"] == "bh-glx-topology-probe"
    assert node["enabled"] is False
    assert node["disabled_classification"] == "different_hardware_deferred"
    assert node["machine_pool"] == ["bh-glx-32"]

    machine = matrix["machines"]["bh-glx-32"]
    assert machine["architecture"] == "blackhole"
    assert machine["expected_inventory"]["device_count"] == 32
    assert machine["expected_inventory"]["cluster_type"] == "BLACKHOLE_GALAXY"
    assert machine["supported_mesh_devices"] == ["BHGLX"]

    # The provenance must not claim a measurement nobody took.
    assert any("NOT measured" in basis for basis in node["source_basis"])

    # Both Galaxy nodes are deferrals, and they are the only ones in the matrix.
    assert sorted(entry["id"] for entry in matrix["nodes"] if not entry["enabled"]) == [
        "bh-glx-topology-probe",
        "wh-tg-rmsnorm-2d-qk-norm",
    ]

    with pytest.raises(runner.MatrixError, match="different_hardware_deferred"):
        runner.select_node(matrix, node["id"])
