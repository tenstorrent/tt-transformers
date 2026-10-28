# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Host-only contracts for the opt-in multi-host (cross-box) mesh primitives.

These exercise the rank-aware path selection, the per-step cross-host
rendezvous, and the mesh-open retry without opening a device: ``ttnn`` is
replaced with a fake namespace so the behaviour is asserted against the
primitives' logic rather than against a particular ttnn build.
"""

from types import SimpleNamespace

import pytest

from tt_transformers import distributed


@pytest.mark.host
def test_plain_process_is_single_host(monkeypatch):
    monkeypatch.setattr(distributed, "ttnn", SimpleNamespace())
    monkeypatch.delenv("OMPI_COMM_WORLD_SIZE", raising=False)
    monkeypatch.delenv("OMPI_COMM_WORLD_RANK", raising=False)

    assert distributed.num_host_ranks() == 1
    assert distributed.host_rank() == 0
    assert distributed.is_multi_rank() is False


@pytest.mark.host
def test_mpi_env_sets_rank_count_when_no_context(monkeypatch):
    monkeypatch.setattr(distributed, "ttnn", SimpleNamespace())
    monkeypatch.setenv("OMPI_COMM_WORLD_SIZE", "4")
    monkeypatch.setenv("OMPI_COMM_WORLD_RANK", "2")

    assert distributed.num_host_ranks() == 4
    assert distributed.host_rank() == 2
    assert distributed.is_multi_rank() is True


@pytest.mark.host
def test_context_world_size_takes_precedence_over_env(monkeypatch):
    fake = SimpleNamespace(
        distributed_context_is_initialized=lambda: True,
        distributed_context_world_size=lambda: 2,
        distributed_context_world_rank=lambda: 1,
    )
    monkeypatch.setattr(distributed, "ttnn", fake)
    monkeypatch.setenv("OMPI_COMM_WORLD_SIZE", "8")
    monkeypatch.setenv("OMPI_COMM_WORLD_RANK", "7")

    assert distributed.num_host_ranks() == 2
    assert distributed.host_rank() == 1
    assert distributed.is_multi_rank() is True


@pytest.mark.host
def test_uninitialized_context_falls_back_to_single_host(monkeypatch):
    fake = SimpleNamespace(
        distributed_context_is_initialized=lambda: False,
        distributed_context_world_size=lambda: 4,
    )
    monkeypatch.setattr(distributed, "ttnn", fake)
    monkeypatch.delenv("OMPI_COMM_WORLD_SIZE", raising=False)

    assert distributed.num_host_ranks() == 1
    assert distributed.is_multi_rank() is False


@pytest.mark.host
def test_cross_host_rendezvous_single_host_drains_without_barrier(monkeypatch):
    calls = []
    fake = SimpleNamespace(
        synchronize_device=lambda mesh: calls.append(("sync", mesh)),
        distributed_context_barrier=lambda: calls.append(("barrier", None)),
    )
    monkeypatch.setattr(distributed, "ttnn", fake)
    monkeypatch.setattr(distributed, "is_multi_rank", lambda: False)

    mesh = object()
    distributed.cross_host_rendezvous(mesh)

    assert calls == [("sync", mesh)]


@pytest.mark.host
def test_cross_host_rendezvous_multi_host_drains_then_barriers(monkeypatch):
    calls = []
    fake = SimpleNamespace(
        synchronize_device=lambda mesh: calls.append("sync"),
        distributed_context_barrier=lambda: calls.append("barrier"),
    )
    monkeypatch.setattr(distributed, "ttnn", fake)
    monkeypatch.setattr(distributed, "is_multi_rank", lambda: True)

    distributed.cross_host_rendezvous(object())

    # The drain must happen before the barrier.
    assert calls == ["sync", "barrier"]


@pytest.mark.host
def test_cross_host_rendezvous_requires_barrier_on_multi_host(monkeypatch):
    fake = SimpleNamespace(synchronize_device=lambda mesh: None)
    monkeypatch.setattr(distributed, "ttnn", fake)
    monkeypatch.setattr(distributed, "is_multi_rank", lambda: True)

    with pytest.raises(RuntimeError, match="distributed_context_barrier"):
        distributed.cross_host_rendezvous(object())


@pytest.mark.host
def test_open_retry_returns_on_first_success_without_sleeping(monkeypatch):
    def fail_sleep(_seconds):
        raise AssertionError("a first-try success must not sleep")

    monkeypatch.setattr(distributed.time, "sleep", fail_sleep)

    assert distributed.open_mesh_device_with_retry(lambda: "mesh") == "mesh"


@pytest.mark.host
def test_open_retry_recovers_from_transient_fabric_timeout(monkeypatch):
    monkeypatch.setattr(distributed.time, "sleep", lambda _seconds: None)
    attempts = {"n": 0}

    def opener():
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise RuntimeError("Fabric router sync timeout while opening mesh")
        return "mesh"

    result = distributed.open_mesh_device_with_retry(opener, attempts=5, backoff_seconds=0)

    assert result == "mesh"
    assert attempts["n"] == 3


@pytest.mark.host
def test_open_retry_reraises_non_transient_error_immediately(monkeypatch):
    monkeypatch.setattr(distributed.time, "sleep", lambda _seconds: None)
    attempts = {"n": 0}

    def opener():
        attempts["n"] += 1
        raise RuntimeError("out of DRAM")

    with pytest.raises(RuntimeError, match="out of DRAM"):
        distributed.open_mesh_device_with_retry(opener)

    assert attempts["n"] == 1


@pytest.mark.host
def test_open_retry_exhausts_attempts_then_raises(monkeypatch):
    monkeypatch.setattr(distributed.time, "sleep", lambda _seconds: None)
    attempts = {"n": 0}

    def opener():
        attempts["n"] += 1
        raise RuntimeError("fabric router sync timeout")

    with pytest.raises(RuntimeError, match="fabric router"):
        distributed.open_mesh_device_with_retry(opener, attempts=3, backoff_seconds=0)

    assert attempts["n"] == 3
