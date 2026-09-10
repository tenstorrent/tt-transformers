# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The decode LM head's fabric-write buffer is plan-owned and outlives the graph.

`GalaxyColumnAllReduce._persistent_all_reduce` hands `all_reduce_async` a
`buffer_tensor` - the tensor the three peer column devices fabric-write into.
That overload takes no barrier semaphore (`all_reduce_async.hpp:51-64`), so the
only defence the CCL best-practices note sanctions for it is address
exclusivity, and address exclusivity is a **lifetime** property: the address has
to stay owned for as long as any captured program can write to it.

It used to be built inside the graph body and freed in the same `finally`. Under
a trace that free is host-side allocator bookkeeping the captured program cannot
see, and the traced decode stopped reproducing itself - see the plan's own
comment at `lm_head_all_reduce` for the silicon behind this, and
`tttv2_milestone_c_evidence/trace/logs/p8_L80_lm_head_hoist_headroom120.log` for
the run in which the fix takes the traced decode to fourteen of fourteen exact.

Both assertions fail without the change and pass with it. They are host-only:
the collective is driven against fakes, exactly as `test_collectives.py` does.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
import ttnn

from tt_transformers.models.galaxy.collectives import GalaxyColumnAllReduce
from tt_transformers.models.galaxy.plans import build_galaxy_resources_config
from tt_transformers.models.galaxy.recipes import (
    GALAXY_MESH_SHAPE,
    GalaxyDenseGeometry,
    resolve_galaxy_decode_placements,
)

LLAMA = dict(dim=8192, hidden_dim=28672, n_heads=64, n_kv_heads=8, head_dim=128, vocab_size=128256)
QWEN = dict(dim=5120, hidden_dim=25600, n_heads=64, n_kv_heads=8, head_dim=128, vocab_size=151936)


class _FakeMemoryConfig:
    """Enough of a memory config for the collective's own trace messages."""

    shard_spec = SimpleNamespace(grid=SimpleNamespace(num_cores=lambda: 42))


class _FakeTensor:
    def __init__(self, name: str, shape: tuple[int, ...], memory_config: Any = None):
        self.name = name
        self.shape = shape
        self.dtype = ttnn.bfloat8_b
        self.deallocations = 0
        self._memory_config = memory_config or _FakeMemoryConfig()

    def memory_config(self) -> Any:
        return self._memory_config

    def deallocate(self, force: bool = False) -> None:
        self.deallocations += 1

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return f"_FakeTensor({self.name})"


def _drive(monkeypatch) -> tuple[_FakeTensor, list[Any]]:
    """Run one decode LM-head reduction against fakes; return the plan buffer."""

    plan_buffer = _FakeTensor("plan-lm-head-l1", (1, 1, 32, 64512))
    resource = SimpleNamespace(
        key=SimpleNamespace(operation="all_reduce", cluster_axis=1, geometry=None, sequence_key=None),
        topology=ttnn.Topology.Ring,
        num_links=4,
        persistent_output_buffers=(_FakeTensor("plan-lm-head-dram", (1, 1, 32, 64512)),),
        intermediate_output_buffers=(plan_buffer,),
    )
    context = SimpleNamespace(next_semaphore_handles=lambda *args, **kwargs: "semaphore")
    resources = SimpleNamespace(
        context=lambda mode: context,
        synchronize=lambda mode: None,
    )
    placements = SimpleNamespace(
        lm_head_all_reduce_input_memcfg="input-memcfg",
        lm_head_all_reduce_buffer_memcfg="buffer-memcfg",
    )
    seen: list[Any] = []

    monkeypatch.setattr(
        "tt_transformers.models.galaxy.collectives.select_galaxy_resource",
        lambda *args, **kwargs: resource,
    )
    # A distinct tensor each time, because the collective's own contract check
    # rejects a reduction that hands back its input placement.
    monkeypatch.setattr(
        "tt_transformers.models.galaxy.collectives._relocate_sharded",
        lambda tensor, memory_config: _FakeTensor(f"{tensor.name}-relocated", tensor.shape),
    )

    def all_reduce_async(input_tensor, buffer_tensor, **kwargs):
        seen.append(buffer_tensor)
        return _FakeTensor("reduced", input_tensor.shape)

    monkeypatch.setattr(ttnn.experimental, "all_reduce_async", all_reduce_async)
    monkeypatch.setattr(
        ttnn,
        "interleaved_to_sharded",
        lambda *args, **kwargs: pytest.fail("the LM head must not build its fabric-write buffer in the graph body"),
    )

    collective = GalaxyColumnAllReduce(
        SimpleNamespace(),
        resources=resources,
        placements=placements,
        dtype=ttnn.bfloat8_b,
    )
    collective(_FakeTensor("logits", (1, 1, 32, 16128)))
    return plan_buffer, seen


@pytest.mark.host
@pytest.mark.model
def test_the_decode_lm_head_reduces_into_the_plans_own_l1_buffer(monkeypatch):
    plan_buffer, seen = _drive(monkeypatch)

    assert seen == [plan_buffer], seen


@pytest.mark.host
@pytest.mark.model
def test_the_decode_lm_head_never_deallocates_its_fabric_write_buffer(monkeypatch):
    plan_buffer, _ = _drive(monkeypatch)

    assert plan_buffer.deallocations == 0, (
        "the LM head freed the tensor the peer devices fabric-write into; a captured "
        "program keeps that address baked in and every replay lets skewed peers write "
        "over whatever the allocator puts there next"
    )


@pytest.fixture
def _host_mesh_mappers(monkeypatch):
    """Mesh mappers require a live device; plans only need placement identity."""

    monkeypatch.setattr(ttnn, "ShardTensor2dMesh", lambda *args, **kwargs: "shard-2d-mapper")
    monkeypatch.setattr(ttnn, "ReplicateTensorToMesh", lambda *args, **kwargs: "replicate-mapper")


def _mesh():
    mesh = MagicMock(spec=ttnn.MeshDevice)
    mesh.shape = GALAXY_MESH_SHAPE
    mesh.get_num_devices.return_value = 32
    mesh.arch.return_value = ttnn.device.Arch.WORMHOLE_B0
    mesh.dram_grid_size.return_value = SimpleNamespace(x=12, y=1)
    mesh.compute_with_storage_grid_size.return_value = SimpleNamespace(x=7, y=10)
    return mesh


@pytest.mark.host
@pytest.mark.model
@pytest.mark.parametrize("model", [LLAMA, QWEN], ids=["llama-3.3-70b", "qwen3-32b"])
def test_only_the_decode_lm_head_all_reduce_provisions_an_l1_intermediate(model, _host_mesh_mappers):
    """The LM head's fabric-write buffer is plan-owned; nothing else takes one.

    This pins the cost as well as the shape. The buffer is resident for the life
    of the process - 52 224 B per worker core on Llama-3.3-70B, 65 280 on
    Qwen3-32B - and the reserve above the prefetcher's global circular buffer has
    room for exactly one of them, measured at
    `tttv2_milestone_c_evidence/trace/logs/p4_L80_baseline_l1_map.log`. A second
    key acquiring an L1 intermediate would be silently unaffordable, and the
    decode graph would stop placing rather than say so.
    """

    mesh = _mesh()
    geometry = GalaxyDenseGeometry(**model, max_seq_len=2048, prefill_sequence_lengths=(128, 2048))
    placements = resolve_galaxy_decode_placements(geometry, mesh)
    config = build_galaxy_resources_config(mesh, geometry, placements)

    intermediates = {
        (plan.key.operation, plan.key.cluster_axis): len(plan.intermediate_output_specs)
        for plan in config.decode.collectives
        if plan.intermediate_output_specs
    }

    assert intermediates[("all_reduce", 1)] == 1
    assert set(intermediates) == {("all_reduce", 1), ("reduce_scatter", 1)}
    lm_head = next(
        plan for plan in config.decode.collectives if plan.key.operation == "all_reduce" and plan.key.cluster_axis == 1
    )
    assert lm_head.intermediate_output_specs[0].memory_config == placements.lm_head_all_reduce_buffer_memcfg
