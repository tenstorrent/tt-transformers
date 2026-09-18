# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Do the collectives the Blackhole Galaxy decode path issues actually move data?

**The most dangerous item in the port**, because the failure mode moves no data
and raises nothing, so the collective *appears* to run.

Blackhole Galaxy needs a 2D-torus fabric, a 2D fabric uses a different packet
header, and every CCL kernel written only for the 1D header is therefore
unavailable. That manifests in two ways, and **they are not the same**:

* **Safe.** The fused ``all_gather_minimal_matmul_async`` *fails to compile*
  against the 2D-torus ``HybridMeshPacketHeader``. Loud, and nothing to do
  about it.
* **Dangerous.** Five galaxy-specific fused CCLs -- ``fused_rms_minimal``,
  ``llama_rs_create_heads``, ``all_gather_concat``, ``llama_rs_matmul``,
  ``llama_reduce_scatter`` -- use 1D-multicast writers that **no-op**. They move
  no data and raise nothing. The 2D fabric does *not* reject what it cannot run.

**Do not generalize the compile error into an expectation of loud failure.**
That inference is the trap, and it is named here because it is easy to make.

Milestone 1 mostly dodges this: the prefetcher-free path already uses the
standard collectives -- the ones the reference falls back *to* -- and
``GalaxyCapabilities.has_fused_ccl`` is ``False`` on the Blackhole descriptor.
This file confirms the dodge rather than assuming it.

## ``all_reduce_create_qkv_heads`` is deliberately *not* one of the five

``ttnn.experimental.all_reduce_create_qkv_heads`` is easy to mistake for a sixth
member of the list above, and it is not one. The fused head-creating collective
that no-ops is ``llama_rs_create_heads`` (``llama_reduce_scatter_create_heads``
in tt-metal), which is **a different operation**:
``all_reduce_create_qkv_heads`` lives under
``operations/experimental/transformer/`` and is classed with the generic async
CCLs -- ``all_gather_async``, ``reduce_scatter_minimal_async``,
``all_reduce_async`` -- that choose sender worker cores from the worker
sub-device (see ``tests/modules/_wh_galaxy_hardware.py``'s note on
``semaphore_cores``).

That leaves it in the worst position available: **not known to no-op, and not
known to be safe** on the 2D-torus fabric. Nothing established either, and it
matters concretely -- ``Attention2D``'s decode path cannot express the 32-to-8
column-user slice without it, because ``nlp_create_qkv_heads_decode`` takes no
``batch_offset``/``slice_size`` and the slice exists only inside the fused op. So
the last node id in this file measures it, and it runs last because it is the one
node here whose op might hang rather than fail. Do not re-derive the distinction:
the five are named in ``_FUSED_CCLS_THAT_SILENTLY_NOOP`` below and this op is not
among them, which is why the enumeration test's assertion correctly does not fire
on it.

## Why this is a positive-control suite and not a PCC suite

The failure mode is silence, so the usual shape -- run the op, check PCC -- is
insufficient: a no-op writer leaves the output buffer holding whatever was
there, and if that happens to be a zeroed buffer or a plausible-looking prior
value, PCC against a reference may not be the first thing to fail. So every
collective here:

1. has its **output buffer pre-filled with a poison pattern** -- not zeros,
   which are too easy to confuse with a legitimate result, but a repeating
   four-value pattern at magnitude ``2 ** 100``, which no sum of the inputs can
   produce and which is exactly representable in bfloat16 so the comparison is
   equality and not a tolerance;
2. runs;
3. is asserted to retain **zero poison elements**, per device, *before* any PCC
   is computed -- so a no-op reads as a no-op instead of being debugged as a
   numerics problem. ``not torch.equal(actual, poison)`` would not do: it passes
   as soon as a single device moves data. The detector is a count, and it is
   reported per mesh row and per mesh column so a partial no-op localizes to the
   devices that produced it;
4. and only then is correlated against a CPU reference, per device.

``ttnn.all_reduce`` is the one exception and it is a real limitation, not an
omission: its signature takes ``input_tensor, cluster_axis, subdevice_id,
memory_config, num_links, topology`` and **no caller-allocated output tensor**,
so a poison pattern cannot be placed where its result will land. Its positive
control is structural instead -- the reduction of ``N`` independent random
contributions has a norm about ``sqrt(N)`` times a single contribution's, and is
bit-equal to none of them, so "the peers' data arrived" is asserted directly.
The two halves an all-reduce decomposes into (``reduce_scatter`` then
``all_gather``, which is what the prefetcher-free *prefill* all-reduce literally
issues) are poison-checked in their own tests above it.

``all_reduce_create_qkv_heads`` allocates all four of its results too, but it
*does* take a caller-allocated ``buffer_tensor`` -- the L1 scratch its column
peers fabric-write into -- so the poison form **is** expressible for it, on the
write target rather than on the output. That is the stronger place for it: the
poison sits exactly where a 1D-multicast writer would have written and did not.
Its detector is per width slot and per device, and it is dtype-robust rather
than bit-exact, because that buffer is ``bfloat8_b`` block float where a host
round-trip is not bit-preserving -- see ``_assert_the_peers_overwrote_the_scratch``.

## The two ways this file could hang instead of failing

* **``num_links`` must be 2 on Blackhole** (Wormhole: 4). An over-requested link
  count **deadlocks with no traceback**, and a ``TT_FATAL`` out of
  ``enqueue_mesh_workload`` leaves the mesh un-drainable -- SIGKILL plus a reset
  is the only recovery. It is therefore read from the descriptor
  (``galaxy_fabric_links``) and asserted, never written down here.
* **``Topology.Ring`` is still correct, but it pairs with
  ``FABRIC_2D_TORUS_XY``**, not Wormhole's ``FABRIC_1D_RING``. Carrying the
  Wormhole pairing reproduces the class of opaque hang that caused most of the
  observed hangs across the Wormhole corpus, so the pairing is asserted on the
  descriptor before a single collective is enqueued.

Both mesh axes are covered for all three standard operations, because
``_CANONICAL_AXES`` in ``models/galaxy/ccl.py`` admits ``{0, 1}`` for each of
``reduce_scatter``, ``all_gather`` and ``all_reduce``. It admits ``{1}`` only for
``all_reduce_create_qkv_heads``, so that one is column-axis only and the test
proves axis 0 is *illegal* rather than merely untried. **``cluster_axis=1`` is
the axis that requires the 2D fabric** -- ``FABRIC_1D``/``FABRIC_1D_RING`` throw
``IndexError: map::at`` on the cross-column route -- so every parametrization
lists axis 1 first and the file is ordered cheapest-first, so a run cut short
still banks the column-axis results.

Needs no checkpoint and no model: ``torch.randn`` inputs, a CPU reference, and
the shared Blackhole plumbing in ``tests/modules/_bh_galaxy_hardware.py``.
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any, get_args

import pytest
import torch
import ttnn
from tests.models.galaxy.galaxy_hardware import BHGLX_DEVICE_PARAMS, GALAXY_MESH_SHAPE
from tests.modules._bh_galaxy_hardware import (
    bh_galaxy_mode_plan,
    bh_galaxy_resources_config,
    bh_galaxy_topology,
    deallocate_tensor,
    exact_tensor_resource,
    record_module_output,
    require_bh_galaxy_ccl_resources,
)
from tests.support.comparison import comp_pcc

from tt_transformers.device_utils import has_l1_small_region
from tt_transformers.models.galaxy import (
    GalaxyCollectivePlan,
    GalaxyResourceKey,
    GalaxyTensorSpec,
    build_galaxy_decode_collectives,
    build_galaxy_prefill_collectives,
)
from tt_transformers.models.galaxy.ccl import CollectiveName
from tt_transformers.models.galaxy.recipes import (
    GALAXY_COLUMNS,
    GALAXY_PHYSICAL_BATCH,
    GALAXY_ROWS,
    GalaxyDenseGeometry,
    distributed_norm_decode_memory_config,
    galaxy_fabric_links,
    height_sharded_memory_config,
    width_sharded_memory_config,
    worker_cores,
)
from tt_transformers.models.galaxy.topology import (
    ccl_offset_placeable_worker_cores,
    reduce_scatter_worker_cores,
    reduce_scatter_workers_per_link,
)

#: The two Galaxy model geometries, as the host plan suites already write them.
#: Taken from `tests/models/galaxy/test_plans.py` rather than re-derived, so the
#: widths here are the same widths the resource keys are built from elsewhere.
LLAMA = dict(dim=8192, hidden_dim=28672, n_heads=64, n_kv_heads=8, head_dim=128, vocab_size=128256)
QWEN = dict(dim=5120, hidden_dim=25600, n_heads=64, n_kv_heads=8, head_dim=128, vocab_size=151936)
_MODELS = [LLAMA, QWEN]
_MODEL_IDS = ["llama-8192", "qwen-5120"]

#: Axis 1 (column) first, deliberately. It is the axis that needs the 2D-torus
#: fabric, so it is the result worth banking if a run is cut short.
_AXES = [1, 0]
_AXIS_IDS = ["axis1-column", "axis0-row"]

_PCC = 0.99
_MAX_SEQ_LEN = 2048

#: The five galaxy-specific fused CCLs whose 1D-multicast writers no-op on the
#: 2D-torus fabric. Four of them are not even expressible as a
#: `GalaxyResourceKey` operation; `all_gather_concat` is, which is what makes
#: the assertion in the first test a real one rather than a tautology.
_FUSED_CCLS_THAT_SILENTLY_NOOP = (
    "fused_rms_minimal",
    "llama_rs_create_heads",
    "all_gather_concat",
    "llama_rs_matmul",
    "llama_reduce_scatter",
)

#: Poison magnitude. `2 ** 100` and the multipliers below need two mantissa bits,
#: so every poison value is exact in bfloat16 and the survival check is equality
#: rather than a tolerance. It is ~1.3e30 against inputs of order 1, so a poison
#: value that reaches the PCC stage also blows the correlation wide open -- but
#: the point is that it never reaches the PCC stage.
_POISON_MAGNITUDE = float(2**100)

#: Everything above this magnitude is poison, nothing a sum of the inputs can
#: reach is anywhere near it. Used where the buffer's dtype makes bit-equality
#: unavailable -- `bfloat8_b` is block float, so a host round-trip through
#: `from_torch`/`to_torch` is not bit-preserving and the poison has to be
#: recognised by size instead of by pattern. Inputs are order 1; this is ~1e12.
_POISON_DETECTION_FLOOR = float(2**40)

#: The fused create-QKV-heads geometry. Every number is a property of the mesh
#: split, not of the model: both Galaxy models have `n_heads=64`, `n_kv_heads=8`
#: and `head_dim=128`, so `local_qkv_size` is 1280 on both and the fused
#: collective's program is byte-identical between them. `test_all_reduce_...`
#: asserts that equality rather than asserting it in prose, which is why that
#: node is not parametrized over the two geometries.
_HEAD_DIM = 128
_LOCAL_HEADS = LLAMA["n_heads"] // GALAXY_ROWS
_LOCAL_KV_HEADS = LLAMA["n_kv_heads"] // GALAXY_ROWS
_USERS_PER_COLUMN = GALAXY_PHYSICAL_BATCH // GALAXY_COLUMNS
#: `(0, 8, 16, 24)`, i.e. `Attention2DConfig.batch_offsets` for batch 32.
_BATCH_OFFSETS = tuple(range(0, GALAXY_PHYSICAL_BATCH, _USERS_PER_COLUMN))

galaxy = pytest.mark.parametrize("mesh_device", [pytest.param(GALAXY_MESH_SHAPE, id="8x4")], indirect=True)


def _local_qkv_size(model: dict) -> int:
    """Row-local fused QKV width, the way `GalaxyDenseGeometry` derives it."""

    return _HEAD_DIM * (model["n_heads"] + 2 * model["n_kv_heads"]) // GALAXY_ROWS


galaxy_params = pytest.mark.parametrize("device_params", [BHGLX_DEVICE_PARAMS], indirect=True)


def _report(title: str, value: object) -> None:
    print(f"[e07] {title}: {value}", flush=True)


# ---------------------------------------------------------------------------
# Host-side construction
# ---------------------------------------------------------------------------


def _mesh_source(width: int, *, seed: int) -> torch.Tensor:
    """Return `(8, 4, 32, width)` of distinct per-device data, in bfloat16.

    One block per device, so every device's contribution is independent and a
    result that carries only the local block is distinguishable from one that
    carries the axis. Held in bfloat16 because that is what the device receives:
    the CPU reference is then computed from exactly the values the collective
    saw, which is the only way a PCC number means anything at this width.
    """

    generator = torch.Generator().manual_seed(seed)
    values = torch.randn(GALAXY_ROWS, GALAXY_COLUMNS, GALAXY_PHYSICAL_BATCH, width, generator=generator)
    return (values * 0.5).to(torch.bfloat16)


def _poison_pattern(width: int) -> torch.Tensor:
    """Return the `(8, 4, 32, width)` poison the output buffer is pre-filled with.

    Four repeating values at magnitude `2 ** 100`, not zeros. Zeros are what a
    freshly allocated buffer already holds, so a zero-filled "poison" cannot
    distinguish *the writer did nothing* from *the op allocated a clean buffer
    and the writer did nothing*, which is the whole question. A non-uniform
    pattern additionally means a partial overwrite cannot coincidentally
    reproduce it.
    """

    count = GALAXY_ROWS * GALAXY_COLUMNS * GALAXY_PHYSICAL_BATCH * width
    index = torch.arange(count, dtype=torch.float32)
    values = -_POISON_MAGNITUDE * (1.0 + 0.25 * (index % 4))
    return values.reshape(GALAXY_ROWS, GALAXY_COLUMNS, GALAXY_PHYSICAL_BATCH, width).to(torch.bfloat16)


def _to_mesh(
    host: torch.Tensor,
    mesh_device: ttnn.MeshDevice,
    *,
    dtype: Any = ttnn.bfloat16,
    memory_config: Any = None,
) -> ttnn.Tensor:
    """Place `(8, 4, 32, W)` so device `(row, column)` owns `host[row, column]`.

    The same `dims=(0, 1)` row/column shard the decode plans use for their own
    per-device buffers (`plans.py::build_galaxy_decode_collectives`), so each
    device's tensor is `(1, 1, 32, W)` -- the shape the collective and its
    resource key are both written against.

    `memory_config` may be a *sharded* L1 placement, which is how the fused
    create-QKV-heads node gets a poisoned width-sharded scratch: it is the same
    `from_torch(..., memory_config=spec.memory_config, mesh_mapper=...)` call
    `resources.py::_allocate_tensor` makes for the plan's own buffers.
    """

    return ttnn.from_torch(
        host,
        device=mesh_device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=memory_config or ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(0, 1), mesh_shape=GALAXY_MESH_SHAPE),
    )


def _worker_subgrid(topology: Any, count: int, *, row_wise: bool) -> ttnn.CoreRangeSet:
    """Take `count` cores from the descriptor's worker envelope, from its origin.

    Deliberately the same construction as `recipes._subgrid_cores` -- same
    origin (`sampling_start_core`), same envelope, same ttnn helper -- called
    publicly because `_subgrid_cores` is private and
    `resolve_galaxy_decode_placements` now (correctly) refuses to run on
    Blackhole at all. It is therefore not a second *geometry*, only a second
    caller of the one geometry, which matters because `CoreRangeSet` preserves
    constructor sequence and a differently-ordered set is a different object.
    """

    return ttnn.num_cores_to_corerangeset_in_subcoregrids(
        ttnn.CoreCoord(*topology.sampling_start_core), count, worker_cores(topology), row_wise=row_wise
    )


def _compose(tensor: ttnn.Tensor, mesh_device: ttnn.MeshDevice) -> torch.Tensor:
    """Read a per-device `(1, 1, 32, W)` result back as `(8, 4, 32, W)`.

    Composed by *distribution* and not by `to_torch_auto_compose`, for the reason
    `collectives.py::compose_galaxy_logits` documents at length: an auto-composer
    infers its composer from the tensor's own `tensor_topology()`, which an op's
    output inherits from its activation rather than from what the collective
    actually did to it.
    """

    return ttnn.to_torch(
        tensor,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(0, 1), mesh_shape=GALAXY_MESH_SHAPE),
    )


def _buffer_address(tensor: Any) -> int | None:
    getter = getattr(tensor, "buffer_address", None)
    if not callable(getter):
        return None
    try:
        return int(getter())
    except BaseException:
        return None


def _plan(
    operation: CollectiveName,
    cluster_axis: int,
    input_shape: tuple[int, ...],
    output_shape: tuple[int, ...],
    links: int,
) -> GalaxyCollectivePlan:
    """Register one keyed resource for the exact geometry the op will observe.

    The plans are the real mechanism, not decoration: `exact_tensor_resource`
    resolves `num_links` and `topology` out of this registry the way every
    production Galaxy call site does, so a shape this suite got wrong raises a
    `KeyError` on the host instead of reaching a kernel. `Topology.Ring` is the
    decode value `plans.py` uses for every Blackhole collective, and on this
    architecture it pairs with `FABRIC_2D_TORUS_XY`.

    The persistent buffers themselves go unused -- the poison targets are
    allocated per call, because a poison pattern has to be *written* and a
    plan-owned buffer is allocated zeroed and never rewritten from host. They are
    sized to the op's output anyway so that nothing about the registration is a
    lie, and they are small: at most 128 kB per device.
    """

    def spec(shape: tuple[int, ...]) -> GalaxyTensorSpec:
        return GalaxyTensorSpec(shape, ttnn.bfloat16, ttnn.TILE_LAYOUT, ttnn.DRAM_MEMORY_CONFIG)

    return GalaxyCollectivePlan(
        key=GalaxyResourceKey(operation, cluster_axis, input_shape, math.prod(input_shape[:-1])),
        topology=ttnn.Topology.Ring,
        num_links=links,
        persistent_output_specs=(spec(output_shape),),
        # `GalaxyCollectivePlan` requires an intermediate for reduce_scatter; the
        # ring op stages through one whether or not this suite passes it on.
        intermediate_output_specs=(spec(input_shape),) if operation == "reduce_scatter" else (),
    )


def _resources(mesh_device: ttnn.MeshDevice, topology: Any, plans: tuple[GalaxyCollectivePlan, ...]) -> Any:
    """Open CCL-only resources over the prefetcher-free Blackhole envelope.

    `require_bh_galaxy_ccl_resources` starts no prefetch producer, so there is
    nothing to drain and `ttnn.close_mesh_device` cannot hang in the fixture
    after the test's own assertions have passed. `bh_galaxy_mode_plan` builds the
    single worker sub-device from the descriptor's envelope, which is what keeps
    kernels off Blackhole's column-11 dispatch cores, and `semaphore_cores` stays
    at its default: the generic async CCLs choose sender worker cores from the
    worker sub-device, and a narrower semaphore leaves a sender polling an L1
    address its own core never had reserved, which hangs rather than fails.

    Prefill gets one placeholder key because `GalaxyModePlan` requires at least
    one collective per mode and this suite only ever activates decode.
    """

    placeholder = _plan(
        "all_gather",
        1,
        (1, 1, GALAXY_PHYSICAL_BATCH, ttnn.TILE_SIZE),
        (1, 1, GALAXY_PHYSICAL_BATCH, ttnn.TILE_SIZE * GALAXY_COLUMNS),
        1,
    )
    return require_bh_galaxy_ccl_resources(
        mesh_device,
        config=bh_galaxy_resources_config(
            mesh_device,
            prefill=bh_galaxy_mode_plan("prefill", (placeholder,), mesh_device, topology=topology),
            decode=bh_galaxy_mode_plan("decode", plans, mesh_device, topology=topology),
        ),
    )


def _key_only_decode_placements(geometry: GalaxyDenseGeometry, topology: Any) -> Any:
    """Supply the seven placement fields the decode plan builder reads, no more.

    `build_galaxy_decode_collectives` reads exactly `residual_memcfg`,
    `attention_qkv_scratch_memcfg`, `attention_gather_users_memcfg`,
    `mlp_reduce_scatter_memcfg`, `mlp_w2_input_memcfg`, `all_reduce_buffer_memcfg`
    and `lm_head_all_reduce_buffer_memcfg`, and six of the seven only ever become
    the `memory_config` of a `GalaxyTensorSpec`. The enumeration test allocates
    none of those buffers, so DRAM stands in for the six.

    `residual_memcfg` is the exception: `distributed_norm_stats_memory_config`
    reads its shard grid to place the fused statistics, so it has to be a real
    L1 width-sharded placement -- and on Blackhole it is a real one,
    `distributed_norm_decode_memory_config` having no ring dependency and
    resolving correctly from the descriptor.
    """

    return SimpleNamespace(
        residual_memcfg=distributed_norm_decode_memory_config(geometry, topology),
        attention_qkv_scratch_memcfg=ttnn.DRAM_MEMORY_CONFIG,
        attention_gather_users_memcfg=ttnn.DRAM_MEMORY_CONFIG,
        mlp_reduce_scatter_memcfg=ttnn.DRAM_MEMORY_CONFIG,
        mlp_w2_input_memcfg=ttnn.DRAM_MEMORY_CONFIG,
        all_reduce_buffer_memcfg=ttnn.DRAM_MEMORY_CONFIG,
        lm_head_all_reduce_buffer_memcfg=ttnn.DRAM_MEMORY_CONFIG,
    )


def _pin_the_fabric_pairing(mesh_device: ttnn.MeshDevice, topology: Any) -> int:
    """Assert the Blackhole link budget and the Ring/2D-torus pairing, and report both.

    Both numbers are hang-shaped rather than failure-shaped, so they are checked
    on the descriptor before anything is enqueued: an over-requested link count
    overruns the ethernet channel array and deadlocks with no traceback, and a
    topology paired with the wrong fabric is an opaque hang.
    """

    links = galaxy_fabric_links(mesh_device)
    _report("fabric links per direction", links)
    _report("fabric config", topology.fabric_config)
    _report("has_fused_ccl", topology.capabilities.has_fused_ccl)
    assert links == 2, (
        f"Blackhole Galaxy has 2 fabric links per direction; the descriptor says {links}. "
        "Over-requesting links deadlocks with no traceback and leaves the mesh un-drainable."
    )
    assert topology.fabric_config == ttnn.FabricConfig.FABRIC_2D_TORUS_XY, (
        f"Topology.Ring pairs with FABRIC_2D_TORUS_XY on Blackhole, not {topology.fabric_config}. "
        "Carrying the Wormhole Ring/FABRIC_1D_RING pairing is an opaque hang."
    )
    return links


# ---------------------------------------------------------------------------
# CPU references
# ---------------------------------------------------------------------------


def _reference_reduce_scatter(source: torch.Tensor, cluster_axis: int) -> torch.Tensor:
    """Sum over the axis, then give index `k` of the axis chunk `k` of the width."""

    extent = GALAXY_MESH_SHAPE[cluster_axis]
    summed = source.float().sum(dim=cluster_axis, keepdim=True)
    return torch.cat(torch.chunk(summed, extent, dim=3), dim=cluster_axis)


def _reference_all_gather(source: torch.Tensor, cluster_axis: int) -> torch.Tensor:
    """Concatenate the axis's blocks on the width, and replicate along the axis."""

    extent = GALAXY_MESH_SHAPE[cluster_axis]
    line = torch.cat([source.float().narrow(cluster_axis, index, 1) for index in range(extent)], dim=3)
    return torch.cat([line] * extent, dim=cluster_axis)


def _reference_all_reduce(source: torch.Tensor, cluster_axis: int) -> torch.Tensor:
    """Sum over the axis, replicated to every index of it."""

    extent = GALAXY_MESH_SHAPE[cluster_axis]
    return torch.cat([source.float().sum(dim=cluster_axis, keepdim=True)] * extent, dim=cluster_axis)


# ---------------------------------------------------------------------------
# The no-op detector and the correlation report
# ---------------------------------------------------------------------------


def _assert_no_poison_survived(actual: torch.Tensor, poison: torch.Tensor, case: str) -> None:
    """The no-op detector. This is the whole point of the file.

    Asserted on a *count* and not on `torch.equal`, which would pass as soon as
    one device moved data. Reported per mesh row and per mesh column, because a
    1D-multicast writer that reaches the 2D fabric would fail on the devices its
    multicast cannot reach rather than uniformly, and that is the signature that
    localizes it to a call site.

    Runs before any PCC, so that a collective which moved nothing reads as
    having moved nothing rather than as a numerics problem.
    """

    survived = actual == poison
    total = int(survived.sum())
    empty = int((actual == 0).sum())
    _report(f"{case} poison elements surviving", f"{total} of {survived.numel()}")
    _report(f"{case} exactly-zero elements", f"{empty} of {survived.numel()}")
    if total:
        per_device = survived.flatten(2).sum(dim=2).tolist()
        _report(f"{case} poison survivors per (mesh row, mesh column)", per_device)
    assert total == 0, (
        f"{case}: {total} of {survived.numel()} output elements still hold the poison pattern. "
        "The collective wrote nothing there: a 1D-multicast writer reached the 2D-torus fabric. "
        "Find the op and the call site -- every downstream module would otherwise produce "
        "plausible, wrong numbers with nothing raised."
    )
    assert empty != survived.numel(), f"{case}: the output is entirely zero, which no sum of the inputs can be"


def _assert_the_axis_arrived(actual: torch.Tensor, source: torch.Tensor, cluster_axis: int, case: str) -> None:
    """Positive control for `ttnn.all_reduce`, which takes no output tensor.

    `ttnn.all_reduce`'s signature is `input_tensor, cluster_axis, subdevice_id,
    memory_config, num_links, topology` -- there is nowhere to put a poison
    pattern, because the op allocates the buffer its result lands in. So the
    control is the arithmetic instead, and it is **per device** rather than
    aggregate, so that a single device whose writer no-ops cannot hide in
    thirty-one that worked:

    * the norm ratio. Summing `N` independent contributions gives a norm about
      `sqrt(N)` times any one of them -- 2.83 on axis 0, 2.00 on axis 1 -- while
      a device that got only its own shard sits at 1.00 and one that got nothing
      at 0.00. This also catches a device that received the *wrong* peer's data,
      whose ratio is likewise about 1.
    * the fraction of the device's output that is bit-equal to its own
      contribution. A no-op is 1.00 there. A real bf16 sum is a few parts in ten
      thousand, from elements where one contribution dominates the other `N - 1`
      by more than the mantissa can hold, so the threshold is set two orders of
      magnitude above that and still three below a no-op.

    Both are reported whether they pass or not, and both run before PCC.
    """

    extent = GALAXY_MESH_SHAPE[cluster_axis]
    result = actual.float()
    local = source.float()
    ratios = result.flatten(2).norm(dim=2) / local.flatten(2).norm(dim=2).clamp_min(1e-12)
    unchanged = (result == local).flatten(2).float().mean(dim=2)
    worst_ratio = float(ratios.min())
    worst_unchanged = float(unchanged.max())
    _report(f"{case} norm ratio vs own contribution", f"min={worst_ratio:.3f} max={float(ratios.max()):.3f}")
    _report(f"{case} norm ratio expected", f"~{math.sqrt(extent):.3f} for a {extent}-way sum")
    _report(f"{case} per-device norm ratio grid", [[round(value, 3) for value in row] for row in ratios.tolist()])
    _report(f"{case} largest per-device share bit-equal to own contribution", f"{worst_unchanged:.5f}")
    assert worst_ratio > 1.5, (
        f"{case}: the smallest per-device norm ratio is {worst_ratio:.3f}, so at least one device's result "
        f"is no larger than a single contribution. A {extent}-way sum of independent contributions should "
        f"be about {math.sqrt(extent):.3f}x. That device's reduction did not receive its peers."
    )
    assert worst_unchanged < 0.1, (
        f"{case}: on some device {worst_unchanged:.1%} of the reduced output is bit-equal to that device's "
        "own contribution. The collective returned the local shard, not a sum across the axis."
    )


def _assert_the_peers_overwrote_the_scratch(scratch: torch.Tensor, case: str) -> None:
    """The no-op detector for `all_reduce_create_qkv_heads`.

    The op allocates all four of its results, so there is no output to poison --
    but `buffer_tensor` is caller-allocated, and it is *exactly* the L1 target
    the column peers fabric-write into. Poison there sits where a 1D-multicast
    writer would have written and did not, which is a better place for it than
    the output.

    Two things make the check its own function rather than a reuse of
    `_assert_no_poison_survived`:

    * **It is by magnitude, not by bit-equality.** That buffer is `bfloat8_b`, a
      block-float format with one shared exponent per face row, so a host
      round-trip through `from_torch`/`to_torch` is not bit-preserving and an
      equality test against the host pattern would be unsound in both
      directions. Poison is `2 ** 100`; a real reduced value is order 1; the
      floor is `2 ** 40`. Nothing lives between them.
    * **It is per width slot.** The buffer is `GALAXY_COLUMNS` slots of
      `local_qkv_size`, one per column contribution. A working op writes the
      three peers' slots over the fabric; whether it also rewrites its *own*
      slot or reads its input directly is an implementation detail this test has
      no business pinning. So the assertion is that at least `GALAXY_COLUMNS - 1`
      slots lost their poison on every device -- which is precisely the
      no-op discriminator, since a no-op leaves three slots poisoned -- and the
      full count is reported so 3-versus-4 is on the record either way.
    """

    slots = scratch.float().reshape(
        GALAXY_ROWS, GALAXY_COLUMNS, GALAXY_PHYSICAL_BATCH, GALAXY_COLUMNS, scratch.shape[-1] // GALAXY_COLUMNS
    )
    survivors = (slots.abs() > _POISON_DETECTION_FLOOR).sum(dim=(2, 4))
    clean = (survivors == 0).sum(dim=2)
    _report(f"{case} poison survivors per (mesh row, mesh column, width slot)", survivors.tolist())
    _report(f"{case} slots fully overwritten per (mesh row, mesh column)", clean.tolist())
    _report(f"{case} slots needed", f"{GALAXY_COLUMNS - 1} of {GALAXY_COLUMNS} (the peers'; the local one is optional)")
    assert int(clean.min()) >= GALAXY_COLUMNS - 1, (
        f"{case}: some device has only {int(clean.min())} of {GALAXY_COLUMNS} scratch slots overwritten, so its "
        "column peers never fabric-wrote into the buffer. `all_reduce_create_qkv_heads` uses a 1D-multicast "
        "writer on the 2D-torus fabric and moves no data. Attention2D decode cannot use the fused path."
    )


def _assert_no_poison_leaked(actual: torch.Tensor, name: str, case: str) -> None:
    """Assert an op-allocated result carries neither poison nor nothing at all.

    The cheap companion to the scratch check: if the reduction read poison out of
    a buffer the writers failed to overwrite, it comes out the other side at
    `2 ** 100` scale. Layout-independent, so it runs before any PCC.
    """

    values = actual.float()
    largest = float(values.abs().max())
    nonzero = int((values != 0).sum())
    _report(f"{case} {name} max |value| / non-zero elements", f"{largest:.6g} / {nonzero} of {values.numel()}")
    assert largest < _POISON_DETECTION_FLOOR, (
        f"{case}: {name} carries values up to {largest:.6g}, which is poison read out of the scratch buffer "
        "rather than a reduction of the inputs."
    )
    assert nonzero > 0, f"{case}: {name} is entirely zero, which no reduction of the inputs can be"


def _report_sorted_pcc(expected: torch.Tensor, actual: torch.Tensor, case: str) -> None:
    """Correlate the sorted per-device values, which is layout-independent.

    Printed beside the positional PCC so that "the right data arrived in the
    wrong order" is one glance away from "the wrong data arrived". A high sorted
    PCC with a low positional one means this suite's assumption about the
    `q | k | v` width split or the head-major reshape is wrong, not that the
    collective failed -- and that distinction is the difference between a source
    fix and a one-line test fix.
    """

    golden = expected.flatten(2).sort(dim=2).values
    measured = actual.float().flatten(2).sort(dim=2).values
    _, value = comp_pcc(golden, measured, 0.0)
    _report(f"{case} sorted-value PCC (layout-independent)", round(float(value), 6))


def _assert_pcc_per_device(expected: torch.Tensor, actual: torch.Tensor, case: str) -> None:
    """Correlate per device, report the grid, and name a permutation if it is one.

    Aggregate PCC is close to useless for a column-local placement bug -- those
    give 0 to 0.01 overall -- so the grid is the primary number and the aggregate
    is reported beside it. `comp_pcc` logs nothing on success, so every value is
    printed whether it passes or not.

    On failure the best-matching *result* device is reported for each reference
    device. A channel-order or chunk-order mismatch shows up there as a clean
    permutation -- every reference device well-correlated with *some* result
    device, just not its own -- and that signature distinguishes it immediately
    from bad numerics, where correlation is bad everywhere. The one convention
    this file has to assume is that a reduce-scatter gives index `k` of the axis
    chunk `k` of the scattered dimension; if that is off, this report says so in
    one line instead of costing a re-run.
    """

    _, aggregate = comp_pcc(expected, actual.float(), 0.0)
    grid = []
    worst = (1.0, (0, 0))
    for row in range(GALAXY_ROWS):
        values = []
        for column in range(GALAXY_COLUMNS):
            _, value = comp_pcc(expected[row, column], actual[row, column].float(), 0.0)
            values.append(round(float(value), 4))
            if float(value) < worst[0]:
                worst = (float(value), (row, column))
        grid.append(values)
    _report(f"{case} aggregate PCC", round(float(aggregate), 6))
    _report(f"{case} per-device PCC (8 rows x 4 columns)", grid)
    _report(f"{case} worst device", f"{worst[1]} at PCC {worst[0]:.6f}")

    if worst[0] < _PCC:
        flat_expected = expected.flatten(0, 1).flatten(1)
        flat_actual = actual.float().flatten(0, 1).flatten(1)
        centred_expected = flat_expected - flat_expected.mean(dim=1, keepdim=True)
        centred_actual = flat_actual - flat_actual.mean(dim=1, keepdim=True)
        norms = centred_expected.norm(dim=1, keepdim=True) * centred_actual.norm(dim=1).unsqueeze(0)
        correlations = (centred_expected @ centred_actual.T) / norms.clamp_min(1e-12)
        best_values, best_devices = correlations.max(dim=1)
        # Index k is reference device k (row-major over the 8x4 mesh); the value is
        # the *result* device it correlates best with. `[0, 1, 2, ...]` is bad
        # numerics; any other permutation is a placement or chunk-order mismatch.
        _report(f"{case} best-matching result device per reference device", best_devices.tolist())
        _report(f"{case} best-match PCC per reference device", [round(value, 4) for value in best_values.tolist()])

    assert worst[0] >= _PCC, f"{case} failed per-device PCC>={_PCC}: worst {worst[1]} at {worst[0]:.6f}"


# ---------------------------------------------------------------------------
# 1. The cheapest check: which operations does the Blackhole path even issue?
# ---------------------------------------------------------------------------


@pytest.mark.device
@pytest.mark.blackhole
@pytest.mark.galaxy_bh
@pytest.mark.model
@galaxy_params
@galaxy
@pytest.mark.parametrize("model", _MODELS, ids=_MODEL_IDS)
@torch.no_grad()
def test_the_blackhole_plan_issues_none_of_the_five_silently_noop_fused_ccls(mesh_device, model):
    """Enumerate the collectives the path issues, against the real dispatch registry.

    Not a string search over source. Every Galaxy collective call site resolves
    its resource through `select_galaxy_resource(context, operation, axis,
    tensor)`, which is `context.resources(...)`, which can only return a key
    that `build_galaxy_{decode,prefill}_collectives` registered -- an operation
    absent from those plans raises `KeyError` at the call site and cannot be
    issued at all. So the plans *are* the issued-op list, and this asserts over
    them.

    The assertion is not a tautology. Of the five, `all_gather_concat` is a
    member of `CollectiveName` and therefore a key the registry can express: the
    test constructs one to prove it, and then asserts the Blackhole plans do not
    contain it. The other four are outside the operation domain entirely, which
    is asserted as such rather than passed over in silence.

    The decode placements are **not** resolved here, and cannot be:
    `resolve_galaxy_decode_placements` now resolves its topology from the mesh
    and therefore *raises* on Blackhole, at the first ring access, naming
    `capabilities.has_ring_matmul`. That is the correct behaviour -- a Blackhole
    decode placement set is a different recipe, not a re-parametrization of the
    ring one -- so this test supplies the seven placement fields
    `build_galaxy_decode_collectives` reads and reads back **only the keys**.

    That substitution is safe precisely because it is not the thing under test.
    The keys -- operation, cluster axis, tensor geometry, sequence key -- are
    derived from `geometry` alone; the placements only become the `memory_config`
    of persistent buffer specs that this test never allocates. `residual_memcfg`
    is the one field read structurally (`distributed_norm_stats_memory_config`
    takes its origin off the shard grid), so that one is the real Blackhole
    placement, resolved from the descriptor.
    """

    topology = bh_galaxy_topology(mesh_device)
    links = _pin_the_fabric_pairing(mesh_device, topology)

    geometry = GalaxyDenseGeometry(**model, max_seq_len=_MAX_SEQ_LEN, prefill_sequence_lengths=(128,))
    decode = build_galaxy_decode_collectives(mesh_device, geometry, _key_only_decode_placements(geometry, topology))
    prefill = build_galaxy_prefill_collectives(mesh_device, geometry, 128)

    decode_operations = tuple(plan.key.operation for plan in decode)
    prefill_operations = tuple(plan.key.operation for plan in prefill)
    _report("decode collectives issued", [(plan.key.operation, plan.key.cluster_axis) for plan in decode])
    _report("prefill collectives issued", [(plan.key.operation, plan.key.cluster_axis) for plan in prefill])
    _report("distinct decode operations", sorted(set(decode_operations)))
    _report("num_links per decode collective", [plan.num_links for plan in decode])
    _report("topology per decode collective", sorted({str(plan.topology) for plan in decode}))

    expressible = set(get_args(CollectiveName))
    fused = set(_FUSED_CCLS_THAT_SILENTLY_NOOP)
    # `all_gather_concat` is the one of the five the registry can name, so it is
    # the one whose absence is a decision rather than a type-level impossibility.
    assert fused & expressible == {"all_gather_concat"}, (
        "the set of silently-no-op fused CCLs that `CollectiveName` can express has changed; "
        f"expected {{'all_gather_concat'}}, got {sorted(fused & expressible)}"
    )
    assert GalaxyResourceKey("all_gather_concat", 0, (1, 1, 32, 32), 32).operation == "all_gather_concat", (
        "the registry must be able to key `all_gather_concat` for the assertion below to mean anything"
    )

    issued = set(decode_operations) | set(prefill_operations)
    assert not issued & fused, (
        f"the Blackhole Galaxy path issues {sorted(issued & fused)}, which use 1D-multicast writers that "
        "no-op on the 2D-torus fabric: they move no data and raise nothing. This is a capability-wiring "
        "defect rather than a fabric one -- fix it as a capability check, not an architecture branch."
    )
    assert topology.capabilities.has_fused_ccl is False, (
        "the Blackhole descriptor reports has_fused_ccl=True; the fused route is the no-op route here"
    )

    # Every plan's link count must be inside the architecture's budget. This is
    # the clamp in `plans.py::_links`, measured on the live descriptor: the
    # qualified Wormhole recipes ask for 4 and 3, and an unclamped 4 on a 2-link
    # fabric deadlocks with no traceback.
    for plan in (*decode, *prefill):
        assert 1 <= plan.num_links <= links, (
            f"{plan.key.operation} axis {plan.key.cluster_axis} requests {plan.num_links} links on a "
            f"{links}-link fabric. Over-requesting deadlocks rather than failing."
        )
    # Decode is all-Ring on Blackhole, which is correct -- what changes from
    # Wormhole is the fabric it pairs with, asserted in `_pin_the_fabric_pairing`.
    assert {plan.topology for plan in decode} == {ttnn.Topology.Ring}

    payload = ",".join(sorted(set(decode_operations))).encode()
    record_module_output(
        f"collectives_bh_issued_ops_dim{model['dim']}",
        torch.tensor(list(payload), dtype=torch.uint8),
    )


# ---------------------------------------------------------------------------
# 2. reduce_scatter -- the cheapest collective, and the narrowest output
# ---------------------------------------------------------------------------


@pytest.mark.device
@pytest.mark.blackhole
@pytest.mark.galaxy_bh
@pytest.mark.model
@galaxy_params
@galaxy
@pytest.mark.parametrize("cluster_axis", _AXES, ids=_AXIS_IDS)
@pytest.mark.parametrize("model", _MODELS, ids=_MODEL_IDS)
@torch.no_grad()
def test_reduce_scatter_moves_data_on_both_cluster_axes(mesh_device, model, cluster_axis):
    """Poison the scatter output, scatter into it, and prove no poison survived.

    `ttnn.reduce_scatter` accepts a pre-allocated `output_tensor`, which is what
    makes the poison detector possible here. The Wormhole objection to this op --
    that its program factory lays workers out from
    `worker_cores(TENSIX, sub_device_id).bounding_box()`, which on WH Galaxy
    spans the `x=4` prefetch sender column -- does not apply on the
    prefetcher-free Blackhole envelope: there is no sender column, the envelope
    is one contiguous rectangle, and its bounding box is itself.

    `use_l1_small_for_semaphores` matches the production prefill call site: the
    generic collectives create program-cache-lifetime semaphores, and in main L1
    they fragment the bank mid-region. It also decides which implementation runs:
    the small-shape direct path declines any request it cannot express, and an L1
    small semaphore region is one of them, so this call takes the ring
    minimal-async path -- as production does, for the same reason.

    **`num_workers_per_link` is the correction this test needed.** That ring path
    is the one collective that shifts its worker selection by the envelope
    origin, so the count it picks for itself -- 12 cores on two links -- walks off
    the far edge of a 10-wide envelope that starts at column 1 and aborts before
    an element moves. Asking for the largest count that survives the shift is a
    geometry question the descriptor answers; see
    `reduce_scatter_workers_per_link`.
    """

    topology = bh_galaxy_topology(mesh_device)
    links = _pin_the_fabric_pairing(mesh_device, topology)
    workers_per_link = reduce_scatter_workers_per_link(topology.worker_core_ranges, links)
    case = f"reduce_scatter axis {cluster_axis} dim {model['dim']}"
    # `None` would mean the envelope starts at the grid origin and nothing
    # shifts, which is what arm B of the column-0 question would produce. Report
    # the number either way: it is the difference between this op selecting 4
    # cores and selecting 12, and 12 is what aborted here.
    _report(
        f"{case} workers/link",
        "unconstrained (envelope origin is the grid origin)"
        if workers_per_link is None
        else f"{workers_per_link} = {reduce_scatter_worker_cores(workers_per_link, links)} cores, of "
        f"{ccl_offset_placeable_worker_cores(topology.worker_core_ranges)} placeable after the origin shift",
    )

    extent = GALAXY_MESH_SHAPE[cluster_axis]
    width = model["dim"] // GALAXY_COLUMNS
    scattered_width = width // extent
    input_shape = (1, 1, GALAXY_PHYSICAL_BATCH, width)
    output_shape = (1, 1, GALAXY_PHYSICAL_BATCH, scattered_width)
    _report(f"{case} shapes", f"in={input_shape} out={output_shape} extent={extent}")

    resources = _resources(
        mesh_device, topology, (_plan("reduce_scatter", cluster_axis, input_shape, output_shape, links),)
    )
    device_input = device_output = None
    try:
        resources.activate("decode")
        context = resources.context("decode")
        resource = exact_tensor_resource(context, "reduce_scatter", cluster_axis, input_shape)
        assert resource.num_links == links
        assert resource.topology == ttnn.Topology.Ring

        source = _mesh_source(width, seed=1000 + cluster_axis)
        poison = _poison_pattern(scattered_width)
        device_input = _to_mesh(source, mesh_device)
        device_output = _to_mesh(poison, mesh_device)
        # The poison has to be resident on *every* device before *any* device
        # enqueues the collective. Supplying `output_tensor` compiles the op's own
        # start barrier out (`do_init_barrier = !persistent_output_tensor
        # .has_value()`), so without this host barrier a fast peer could fabric-write
        # into this buffer before a slow peer had finished poisoning it -- and the
        # slow peer's poison write would then land on top of real data and read as a
        # no-op that never happened.
        resources.synchronize("decode")

        result = ttnn.reduce_scatter(
            device_input,
            3,
            cluster_axis=cluster_axis,
            num_links=resource.num_links,
            topology=resource.topology,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            subdevice_id=context.worker_sub_device_id,
            output_tensor=device_output,
            use_l1_small_for_semaphores=has_l1_small_region(mesh_device),
            num_workers_per_link=workers_per_link,
        )
        resources.synchronize("decode")
        actual = _compose(result, mesh_device)

        _assert_no_poison_survived(actual, poison, case)
        expected = _reference_reduce_scatter(source, cluster_axis)
        _assert_pcc_per_device(expected, actual, case)
        record_module_output(f"collectives_bh_reduce_scatter_axis{cluster_axis}_dim{model['dim']}", actual, expected)

        # Reported and asserted last, so a surprise here cannot cost the results
        # above. If the op allocated its own output instead of writing into the
        # poisoned one, the detector was void and the run has to know.
        result_address, poisoned_address = _buffer_address(result), _buffer_address(device_output)
        _report(f"{case} result/poisoned buffer address", f"{result_address} / {poisoned_address}")
        if result_address is not None and poisoned_address is not None:
            assert result_address == poisoned_address, (
                f"{case}: the result is not the buffer this test poisoned, so the poison check above "
                "proved nothing about this op. Re-check that `output_tensor` is honoured."
            )
    finally:
        deallocate_tensor(device_output)
        deallocate_tensor(device_input)
        resources.cleanup()


# ---------------------------------------------------------------------------
# 3. all_gather -- the same detector, on the op that widens rather than narrows
# ---------------------------------------------------------------------------


@pytest.mark.device
@pytest.mark.blackhole
@pytest.mark.galaxy_bh
@pytest.mark.model
@galaxy_params
@galaxy
@pytest.mark.parametrize("cluster_axis", _AXES, ids=_AXIS_IDS)
@pytest.mark.parametrize("model", _MODELS, ids=_MODEL_IDS)
@torch.no_grad()
def test_all_gather_moves_data_on_both_cluster_axes(mesh_device, model, cluster_axis):
    """Poison the gather output, gather into it, and prove no poison survived.

    The call is the production one: `collectives.py::GalaxyAttentionCollectives
    .gather_users` passes `output_tensor=resource.persistent_output_buffers[0]`
    to `ttnn.all_gather` with the resource's own links, topology and worker
    sub-device. A gather is the sharpest of the three for this detector, because
    every device's output is a *concatenation* of the axis -- so a writer that
    reaches only its own slot leaves `extent - 1` slots' worth of poison, and the
    per-device survivor counts say exactly which slots.
    """

    topology = bh_galaxy_topology(mesh_device)
    links = _pin_the_fabric_pairing(mesh_device, topology)
    case = f"all_gather axis {cluster_axis} dim {model['dim']}"

    extent = GALAXY_MESH_SHAPE[cluster_axis]
    gathered_width = model["dim"] // GALAXY_COLUMNS
    shard_width = gathered_width // extent
    input_shape = (1, 1, GALAXY_PHYSICAL_BATCH, shard_width)
    output_shape = (1, 1, GALAXY_PHYSICAL_BATCH, gathered_width)
    _report(f"{case} shapes", f"in={input_shape} out={output_shape} extent={extent}")

    resources = _resources(
        mesh_device, topology, (_plan("all_gather", cluster_axis, input_shape, output_shape, links),)
    )
    device_input = device_output = None
    try:
        resources.activate("decode")
        context = resources.context("decode")
        resource = exact_tensor_resource(context, "all_gather", cluster_axis, input_shape)
        assert resource.num_links == links
        assert resource.topology == ttnn.Topology.Ring

        source = _mesh_source(shard_width, seed=2000 + cluster_axis)
        poison = _poison_pattern(gathered_width)
        device_input = _to_mesh(source, mesh_device)
        device_output = _to_mesh(poison, mesh_device)
        # See the reduce_scatter test: a supplied output tensor compiles the op's
        # start barrier out, so the poison must be everywhere before anyone gathers.
        resources.synchronize("decode")

        result = ttnn.all_gather(
            device_input,
            3,
            cluster_axis=cluster_axis,
            num_links=resource.num_links,
            topology=resource.topology,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            subdevice_id=context.worker_sub_device_id,
            output_tensor=device_output,
        )
        resources.synchronize("decode")
        actual = _compose(result, mesh_device)

        _assert_no_poison_survived(actual, poison, case)
        expected = _reference_all_gather(source, cluster_axis)
        _assert_pcc_per_device(expected, actual, case)
        record_module_output(f"collectives_bh_all_gather_axis{cluster_axis}_dim{model['dim']}", actual, expected)

        result_address, poisoned_address = _buffer_address(result), _buffer_address(device_output)
        _report(f"{case} result/poisoned buffer address", f"{result_address} / {poisoned_address}")
        if result_address is not None and poisoned_address is not None:
            assert result_address == poisoned_address, (
                f"{case}: the result is not the buffer this test poisoned, so the poison check above "
                "proved nothing about this op. Re-check that `output_tensor` is honoured."
            )
    finally:
        deallocate_tensor(device_output)
        deallocate_tensor(device_input)
        resources.cleanup()


# ---------------------------------------------------------------------------
# 4. all_reduce -- no output tensor to poison, so the control is arithmetic
# ---------------------------------------------------------------------------


@pytest.mark.device
@pytest.mark.blackhole
@pytest.mark.galaxy_bh
@pytest.mark.model
@galaxy_params
@galaxy
@pytest.mark.parametrize("cluster_axis", _AXES, ids=_AXIS_IDS)
@pytest.mark.parametrize("model", _MODELS, ids=_MODEL_IDS)
@torch.no_grad()
def test_all_reduce_sums_across_both_cluster_axes(mesh_device, model, cluster_axis):
    """Prove the reduction received its peers, then correlate it.

    This is the call `collectives.py::GalaxyColumnAllReduce.__call__` issues on
    its non-persistent branch, with the same arguments. It is the one collective
    in this file with **no poison pattern**, and that is a limitation of the op
    rather than of the suite: `ttnn.all_reduce` takes no caller-allocated output
    tensor, so there is nowhere to put one. `_assert_the_axis_arrived` is the
    substitute and it runs before PCC for the same reason the poison check does.

    The input is interleaved DRAM at a tile-aligned width that divides by the
    axis extent, which keeps the op off the composite gather path -- the one
    whose `ttnn::concat` is handed no `sub_core_grids`, builds over the full
    compute grid, and is refused by a loaded sub-device manager with "Kernel
    group cores do not match sub device cores".
    """

    topology = bh_galaxy_topology(mesh_device)
    links = _pin_the_fabric_pairing(mesh_device, topology)
    case = f"all_reduce axis {cluster_axis} dim {model['dim']}"

    extent = GALAXY_MESH_SHAPE[cluster_axis]
    width = model["dim"] // GALAXY_COLUMNS
    shape = (1, 1, GALAXY_PHYSICAL_BATCH, width)
    assert width % (ttnn.TILE_SIZE * extent) == 0, (
        f"{case}: width {width} must be tile aligned and divisible by the axis extent {extent}, "
        "or the op falls back to the composite gather the decode sub-device manager refuses"
    )
    _report(f"{case} shapes", f"in={shape} out={shape} extent={extent}")

    resources = _resources(mesh_device, topology, (_plan("all_reduce", cluster_axis, shape, shape, links),))
    device_input = None
    try:
        resources.activate("decode")
        context = resources.context("decode")
        resource = exact_tensor_resource(context, "all_reduce", cluster_axis, shape)
        assert resource.num_links == links
        assert resource.topology == ttnn.Topology.Ring

        source = _mesh_source(width, seed=3000 + cluster_axis)
        device_input = _to_mesh(source, mesh_device)
        resources.synchronize("decode")

        result = ttnn.all_reduce(
            device_input,
            cluster_axis=cluster_axis,
            num_links=resource.num_links,
            topology=resource.topology,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            subdevice_id=context.worker_sub_device_id,
        )
        resources.synchronize("decode")
        actual = _compose(result, mesh_device)

        _assert_the_axis_arrived(actual, source, cluster_axis, case)
        expected = _reference_all_reduce(source, cluster_axis)
        _assert_pcc_per_device(expected, actual, case)
        record_module_output(f"collectives_bh_all_reduce_axis{cluster_axis}_dim{model['dim']}", actual, expected)
    finally:
        deallocate_tensor(device_input)
        resources.cleanup()


# ---------------------------------------------------------------------------
# 5. all_reduce_create_qkv_heads -- LAST, because this is the one that may hang
# ---------------------------------------------------------------------------


def _reference_create_qkv_heads(source: torch.Tensor) -> dict[str, torch.Tensor]:
    """The CPU reference for the fused column reduction plus head creation.

    Three steps, in the order the op performs them:

    1. reduce the fused QKV projection over the four mesh columns;
    2. take this column's `slice_size` users, starting at its `batch_offset` --
       `(0, 8, 16, 24)`, i.e. `Attention2DConfig.batch_offsets` for batch 32;
    3. split the row-local width into `local_heads` Q heads, then
       `local_kv_heads` K heads, then the same number of V heads.

    Returns per-device blocks indexed `[mesh row, mesh column]`, so every
    assertion against it is per device and none of it depends on how the mesh
    composes.
    """

    local_qkv = source.shape[-1]
    query_width = _LOCAL_HEADS * _HEAD_DIM
    key_width = _LOCAL_KV_HEADS * _HEAD_DIM
    reduced = _reference_all_reduce(source, 1)
    users = torch.stack(
        [reduced[:, column, offset : offset + _USERS_PER_COLUMN, :] for column, offset in enumerate(_BATCH_OFFSETS)],
        dim=1,
    )
    heads = (GALAXY_ROWS, GALAXY_COLUMNS, _USERS_PER_COLUMN)
    return {
        "reduced": reduced,
        "q": users[..., :query_width].reshape(*heads, _LOCAL_HEADS, _HEAD_DIM),
        "k": users[..., query_width : query_width + key_width].reshape(*heads, _LOCAL_KV_HEADS, _HEAD_DIM),
        "v": users[..., query_width + key_width : local_qkv].reshape(*heads, _LOCAL_KV_HEADS, _HEAD_DIM),
    }


def _compose_heads(tensor: ttnn.Tensor, mesh_device: ttnn.MeshDevice, heads: int) -> torch.Tensor:
    """Read a per-device `(1, users, heads, head_dim)` result back per device.

    `dims=(0, 1)` concatenates the mesh rows onto the tensor's size-1 leading
    axis and the mesh columns onto the user axis, giving
    `(8, 32, heads, head_dim)`; the four columns hold disjoint user groups in
    column order, so splitting the user axis back into `(4, 8)` recovers the
    `[mesh row, mesh column]` indexing every assertion here uses.
    """

    composed = ttnn.to_torch(
        tensor,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(0, 1), mesh_shape=GALAXY_MESH_SHAPE),
    )
    return composed.reshape(GALAXY_ROWS, GALAXY_COLUMNS, _USERS_PER_COLUMN, heads, _HEAD_DIM)


@pytest.mark.device
@pytest.mark.blackhole
@pytest.mark.galaxy_bh
@pytest.mark.model
@galaxy_params
@galaxy
@torch.no_grad()
def test_all_reduce_create_qkv_heads_moves_data_on_the_column_axis(mesh_device):
    """Settle whether the fused create-QKV-heads collective works on the 2D fabric.

    **This node runs last in the file and nothing should be scheduled behind
    it.** It is the one collective here that is neither known to no-op nor known
    to be safe, so it is the one that might hang rather than fail -- and a
    `TT_FATAL` out of `enqueue_mesh_workload` leaves the mesh un-drainable.

    It is *not* one of the five silently-no-op fused CCLs; see the module
    docstring. `llama_rs_create_heads` is the fused head-creating op that no-ops,
    and it is a different operation. This one is classed with the generic async
    CCLs, and `Attention2D` decode has no substitute for it: the 32-to-8
    column-user slice exists only inside it, because
    `nlp_create_qkv_heads_decode` takes no `batch_offset`/`slice_size`.

    The call is `collectives.py::GalaxyAttentionCollectives
    .reduce_create_qkv_heads`, argument for argument, and the resource is
    provisioned the way `plans.py::build_galaxy_decode_collectives` provisions
    `fused_qkv` -- `Topology.Ring`, `_links(mesh_device, 3)` clamped to the
    architecture budget, one semaphore per slot, and a width-sharded L1 scratch
    `GALAXY_COLUMNS` times the local QKV width.

    **One geometry, and that is not a shortcut.** `local_qkv_size` is
    `head_dim * (n_heads + 2 * n_kv_heads) / 8`, which is 1280 for Llama-3.3-70B
    and 1280 for Qwen3-32B, so a second parametrization would enqueue a
    byte-identical program. The test asserts the equality rather than trusting
    it.

    **Placements.** Four of the five come straight from the qualified decode
    recipe's ring-free formulas, resolved against the *Blackhole* descriptor.
    The fifth does not exist here: production width-shards the collective's
    input across the 24-core `gather_in0` ring, and Blackhole has no ring
    (`has_ring_matmul=False`). The substitute is the ring-free placement the
    *reduced output* already uses -- width-sharded across one core per head
    column -- which satisfies every shape rule the op validates, including that
    the scratch grid contain the output grid and that the scratch shard hold
    `ring_size` output shards. What this node measures is the op, not that
    placement.
    """

    topology = bh_galaxy_topology(mesh_device)
    links = _pin_the_fabric_pairing(mesh_device, topology)
    case = "all_reduce_create_qkv_heads axis 1"

    local_qkv = _local_qkv_size(LLAMA)
    assert local_qkv == _local_qkv_size(QWEN), (
        "the two Galaxy geometries no longer share a local QKV width, so this node has to be parametrized "
        f"over both: llama={local_qkv}, qwen={_local_qkv_size(QWEN)}"
    )
    # Axis 0 is not merely untried, it is illegal: `_CANONICAL_AXES` admits only
    # axis 1 for this operation, and the registry refuses to key anything else.
    with pytest.raises(ValueError, match="canonical cluster axis"):
        GalaxyResourceKey("all_reduce_create_qkv_heads", 0, (1, 1, GALAXY_PHYSICAL_BATCH, local_qkv), 32)

    input_shape = (1, 1, GALAXY_PHYSICAL_BATCH, local_qkv)
    scratch_width = local_qkv * GALAXY_COLUMNS
    qkv_cores = _worker_subgrid(topology, local_qkv // _HEAD_DIM, row_wise=False)
    head_cores = _worker_subgrid(topology, GALAXY_PHYSICAL_BATCH, row_wise=False)
    # `attention_qkv_reduced_memcfg` and `attention_qkv_scratch_memcfg` from the
    # qualified decode recipe, with the Blackhole envelope's cores. The input
    # borrows the reduced placement; see the docstring.
    input_memcfg = width_sharded_memory_config(local_qkv, qkv_cores)
    scratch_memcfg = width_sharded_memory_config(scratch_width, qkv_cores)
    heads_memcfg = height_sharded_memory_config(head_cores, _HEAD_DIM)
    _report(f"{case} qkv cores", f"{qkv_cores} ({qkv_cores.num_cores()} cores)")
    _report(f"{case} head cores", f"{head_cores} ({head_cores.num_cores()} cores)")
    _report(f"{case} shapes", f"in={input_shape} scratch_width={scratch_width} heads={_LOCAL_HEADS}/{_LOCAL_KV_HEADS}")

    resources = _resources(
        mesh_device, topology, (_plan("all_reduce_create_qkv_heads", 1, input_shape, input_shape, links),)
    )
    device_input = device_scratch = batch_offset = None
    try:
        resources.activate("decode")
        context = resources.context("decode")
        resource = exact_tensor_resource(context, "all_reduce_create_qkv_heads", 1, input_shape)
        key = resource.key
        assert resource.num_links == links
        assert resource.topology == ttnn.Topology.Ring

        source = _mesh_source(local_qkv, seed=4000)
        poison = _poison_pattern(scratch_width)
        # bfloat8_b on both, matching the production collective dtype and the
        # plan's own `_spec` default for this buffer.
        device_input = _to_mesh(source, mesh_device, dtype=ttnn.bfloat8_b, memory_config=input_memcfg)
        device_scratch = _to_mesh(poison, mesh_device, dtype=ttnn.bfloat8_b, memory_config=scratch_memcfg)
        # `_fused_batch_offsets` in `collectives.py`, spelling for spelling: a
        # `(GALAXY_COLUMNS, 1)` int32 column sharded over the mesh columns, so
        # device `(row, column)` reads its own user offset.
        batch_offset = ttnn.as_tensor(
            torch.tensor(_BATCH_OFFSETS, dtype=torch.int32).reshape(GALAXY_COLUMNS, 1),
            dtype=ttnn.int32,
            device=mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device=mesh_device, dims=(None, 0), mesh_shape=GALAXY_MESH_SHAPE),
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        # The poison must be resident on every device before any device starts
        # writing into a peer's copy of it -- the same ordering hazard the two
        # `output_tensor` tests above document, and here the op's own start
        # barrier is not in question because the scratch is always caller-owned.
        resources.synchronize("decode")

        reduced, q, k, v = ttnn.experimental.all_reduce_create_qkv_heads(
            device_input,
            device_scratch,
            cluster_axis=1,
            mesh_device=mesh_device,
            multi_device_global_semaphore=context.next_semaphore_handles(
                key.operation, key.cluster_axis, key.geometry, key.sequence_key
            ),
            num_heads=_LOCAL_HEADS,
            memory_config=input_memcfg,
            topology=resource.topology,
            num_links=resource.num_links,
            subdevice_id=context.worker_sub_device_id,
            num_kv_heads=_LOCAL_KV_HEADS,
            final_memory_config=heads_memcfg,
            batch_offset=batch_offset,
            slice_size=_USERS_PER_COLUMN,
            dtype=ttnn.bfloat16,
        )
        resources.synchronize("decode")

        # 1. The no-op detector, first and before anything numeric.
        _assert_the_peers_overwrote_the_scratch(_compose(device_scratch, mesh_device), case)

        actual = {
            "reduced": _compose(reduced, mesh_device),
            "q": _compose_heads(q, mesh_device, _LOCAL_HEADS),
            "k": _compose_heads(k, mesh_device, _LOCAL_KV_HEADS),
            "v": _compose_heads(v, mesh_device, _LOCAL_KV_HEADS),
        }
        # 2. No poison leaked out of the scratch into an op-allocated result.
        for name, values in actual.items():
            _assert_no_poison_leaked(values, name, case)

        # 3. Numerics. `reduced` first: its layout carries no assumption at all,
        # so it isolates the reduction from the head split.
        expected = _reference_create_qkv_heads(source)
        _assert_pcc_per_device(expected["reduced"], actual["reduced"], f"{case} reduced")
        for name in ("q", "k", "v"):
            _report_sorted_pcc(expected[name], actual[name], f"{case} {name}")
            _assert_pcc_per_device(expected[name], actual[name], f"{case} {name}")

        record_module_output(
            "collectives_bh_all_reduce_create_qkv_heads_axis1",
            actual["reduced"],
            actual["q"],
            actual["k"],
            actual["v"],
        )
    finally:
        deallocate_tensor(batch_offset)
        deallocate_tensor(device_scratch)
        deallocate_tensor(device_input)
        resources.cleanup()
