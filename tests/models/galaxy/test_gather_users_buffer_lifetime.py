# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The decode user gather's fabric-write destination is plan-owned and outlives
the graph body.

`ttnn.all_gather` documents its own `output_tensor` argument as *"Pre-allocated
output tensor... **This must be allocated before invoking any op to avoid
races.**"* The destination is what the three peer column devices fabric-write
into, so — exactly as for the LM head's `buffer_tensor`
(`test_lm_head_buffer_lifetime.py`) — the only defence available to it is
address exclusivity, and address exclusivity is a **lifetime** property: the
address has to stay owned for as long as any captured program can write to it.

Before this change `GalaxyAttentionCollectives.gather_users` let the op allocate
a fresh destination on every one of the model's decode layers and
`Attention2D._transition` freed it in the same graph body, while the decode plan
already provisioned a persistent buffer for exactly that key
(`plans.py::gather_users`) that no caller borrowed.

Three assertions, all host-only, driven against fakes the way
`test_collectives.py` does. The first two fail without the change and pass with
it; the third pins the plan spec the other two depend on and passes either way, which
is the point of it — the day that spec moves, the fix stops being free.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
import ttnn

from tt_transformers.models.galaxy import collectives as galaxy_collectives
from tt_transformers.models.galaxy.collectives import GalaxyAttentionCollectives
from tt_transformers.models.galaxy.plans import build_galaxy_resources_config
from tt_transformers.models.galaxy.recipes import (
    GALAXY_MESH_SHAPE,
    GALAXY_PHYSICAL_BATCH,
    GalaxyDenseGeometry,
    resolve_galaxy_decode_placements,
)

LLAMA = dict(dim=8192, hidden_dim=28672, n_heads=64, n_kv_heads=8, head_dim=128, vocab_size=128256)
QWEN = dict(dim=5120, hidden_dim=25600, n_heads=64, n_kv_heads=8, head_dim=128, vocab_size=151936)


class _FakeTensor:
    def __init__(self, name: str, shape: tuple[int, ...], address: int = 0):
        self.name = name
        self.shape = shape
        self.dtype = ttnn.bfloat16
        self.deallocations = 0
        self._address = address

    def buffer_address(self) -> int:
        return self._address

    def memory_config(self) -> Any:
        return SimpleNamespace(is_sharded=lambda: True)

    def is_allocated(self) -> bool:
        return True

    def deallocate(self, force: bool = False) -> None:
        self.deallocations += 1

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return f"_FakeTensor({self.name})"


def _mesh():
    mesh = MagicMock(spec=ttnn.MeshDevice)
    mesh.shape = GALAXY_MESH_SHAPE
    mesh.get_num_devices.return_value = 32
    mesh.arch.return_value = ttnn.device.Arch.WORMHOLE_B0
    mesh.dram_grid_size.return_value = SimpleNamespace(x=12, y=1)
    mesh.compute_with_storage_grid_size.return_value = SimpleNamespace(x=7, y=10)
    return mesh


def _drive(monkeypatch) -> tuple[_FakeTensor, list[Any], Any]:
    """Run one decode user gather against fakes; return the plan's destination."""

    destination = _FakeTensor("plan-gather-users", (1, 32, 8, 128), address=0x1000)
    resource = SimpleNamespace(
        key=SimpleNamespace(operation="all_gather", cluster_axis=1, geometry=None, sequence_key=None),
        topology=ttnn.Topology.Ring,
        num_links=1,
        persistent_output_buffers=(destination,),
        intermediate_output_buffers=(),
    )
    context = SimpleNamespace(
        worker_sub_device_id="worker",
        # `is_borrowed_output` walks the owner's own registry rather than trusting
        # the caller, so the fake context has to model both halves of that seam.
        resource_keys=(resource.key,),
        resources=lambda *args, **kwargs: resource,
        next_semaphore_handles=lambda *args, **kwargs: "semaphore",
    )
    resources = SimpleNamespace(
        context=lambda mode: context,
        synchronize=lambda mode: None,
    )
    seen: list[Any] = []

    monkeypatch.setattr(galaxy_collectives, "select_galaxy_resource", lambda *args, **kwargs: resource)

    def all_gather(tensor, dim, **kwargs):
        seen.append(kwargs.get("output_tensor"))
        # The op returns the borrowed destination when one is supplied, which is
        # the contract that makes the borrow check below load-bearing.
        return kwargs.get("output_tensor") or _FakeTensor("fresh", (1, 32, 8, 128), address=0x2000)

    monkeypatch.setattr(galaxy_collectives.ttnn, "all_gather", all_gather)

    collective = GalaxyAttentionCollectives(
        resources=resources,
        mesh_device=_mesh(),
        geometry=GalaxyDenseGeometry(**LLAMA, max_seq_len=2048, prefill_sequence_lengths=(128,)),
        decode_placements=SimpleNamespace(attention_gather_users_memcfg="gather-users-memcfg"),
    )
    output = collective.gather_users(_FakeTensor("attention", (1, 8, 8, 128), address=0x3000), mode="decode")
    return destination, seen, (collective, output)


@pytest.mark.host
@pytest.mark.model
def test_the_decode_user_gather_writes_into_the_plans_own_buffer(monkeypatch):
    destination, seen, _ = _drive(monkeypatch)

    assert seen == [destination], seen


@pytest.mark.host
@pytest.mark.model
def test_the_decode_user_gather_result_reports_itself_as_borrowed(monkeypatch):
    """`Attention2D` frees whatever it is not told is borrowed."""

    _destination, _seen, (collective, output) = _drive(monkeypatch)

    assert collective.is_borrowed_output(output) is True, (
        "the user gather handed back the plan's own buffer and the ownership seam said it was not "
        "borrowed, so Attention2D frees the tensor the peer devices fabric-write into"
    )


@pytest.fixture
def _host_mesh_mappers(monkeypatch):
    monkeypatch.setattr(ttnn, "ShardTensor2dMesh", lambda *args, **kwargs: "shard-2d-mapper")
    monkeypatch.setattr(ttnn, "ReplicateTensorToMesh", lambda *args, **kwargs: "replicate-mapper")


@pytest.mark.host
@pytest.mark.model
@pytest.mark.parametrize("model", [LLAMA, QWEN], ids=["llama-3.3-70b", "qwen3-32b"])
def test_the_plan_provisions_a_user_gather_destination_of_the_gathered_shape(model, _host_mesh_mappers):
    """The buffer the collective borrows is the *output* shape, not the input's."""

    mesh = _mesh()
    geometry = GalaxyDenseGeometry(**model, max_seq_len=2048, prefill_sequence_lengths=(128, 2048))
    placements = resolve_galaxy_decode_placements(geometry, mesh)
    config = build_galaxy_resources_config(mesh, geometry, placements)

    gathers = [
        plan
        for plan in config.decode.collectives
        if plan.key.operation == "all_gather"
        and plan.persistent_output_specs[0].shape == (1, GALAXY_PHYSICAL_BATCH, geometry.local_heads, geometry.head_dim)
    ]
    assert len(gathers) == 1, [plan.key for plan in config.decode.collectives]
