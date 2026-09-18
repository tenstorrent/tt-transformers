# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Real-hardware correctness tests for the common RMSNorm2D on a BH Galaxy.

Modelled on `test_rmsnorm_2d_wh_galaxy.py`. One structural difference is the
reason this file exists: **every test that touches the residual stream runs two
layers**, because the defect it is looking for is invisible at one.

Why two layers
--------------
`GalaxyCapabilities.has_fused_residual_norm` is `False` on the Blackhole
descriptor because the non-fused distributed RMSNorm **does not write the
residual sum back in place**. A model that relies on the fused path there
silently drops each layer's `ff_out` from the residual stream, and the
reference's own words for that defect's visibility are *"only visible across >1
layer"*: a single-layer module test passes and an 80-layer model is quietly
wrong.

What `RMSNorm2DResidualPolicy.NONE` does with a residual
--------------------------------------------------------
Read off `rmsnorm_2d.py` rather than assumed. `decode_forward` does

    x = ttnn.add(x, residual, memory_config=cfg.decode_residual_memcfg)
    return self._decode_distributed(x, release_input=True)

so the sum is a *local* tensor: it is normalized, then deallocated, and only the
norm output comes back -- a single tensor, not the fused path's
`(normed, residual)` pair. The caller's `residual` is never written. Prefill is
the same shape and does not even deallocate its sum. So a Blackhole decoder has
to own the residual stream itself, exactly the way this repo's already-qualified
*prefill* decoder does (`models/llama33_70b_galaxy/model.py`: `residual =
ttnn.add(h, attention_output, memory_config=..., dtype=...)` and then
`ff_norm.prefill_forward(residual)`). That is what the reference's
`unfuse_res_add` means.

Both call shapes are therefore measured at every layer, on the same data:

  A. `module(ff_out, residual=stream)` -- the module's own unfused add.
  B. `stream = ttnn.add(stream, ff_out); module(stream)` -- caller-owned, the
     shape a Blackhole model must use, and the only one that carries the stream
     into the next layer.

Both must equal `RMSNorm(stream + ff_out)`. Keeping them side by side is what
makes the predicted failure localizable: if only A is wrong the module's
internal add is not happening, if only B is wrong the caller-owned add is, and
if the *stream* assertion fails at layer 1 then layer 0's `ff_out` was dropped
-- the exact predicted failure, "layer 1 correct, layer 2 wrong by roughly one
`ff_out`". Every correctness claim also prints its correlation against the
dropped-`ff_out` hypothesis, because two independent normal tensors correlate at
about 0.71: bad numerics is low against *both* hypotheses, a dropped residual is
high against the dropped one.

The dtype recipe, and the half of it that is not expressible
------------------------------------------------------------
The reference's no-prefetcher path is **bfloat16 residual, bfloat8_b norm
output**; that is a recipe, not a tuning choice, and this repo's Galaxy
precision records the same reason ("the residual stream stays bfloat16 so an
80-layer running sum is never re-quantized").

The residual half is expressible and is used in *both* parametrizations here:
the caller-owned add always runs `dtype=ttnn.bfloat16`, so the multi-layer
residual claim is measured at the production residual dtype either way.

The norm-output half is **not** expressible through `RMSNorm2D`. No config field
reaches `ttnn.rms_norm_post_all_gather`'s dtype -- `decode_output_memcfg` and
`prefill_output_memcfg` carry placement only -- and the module never passes one,
so the norm output inherits its input's dtype. The `norm-bf8` parametrization
therefore hands the norm a bfloat8_b copy of the same residual sum (a second
`ttnn.add` at that dtype, the one dtype knob with a production call site), which
makes the norm output bfloat8_b as the recipe wants while leaving the carried
stream bfloat16. It is a strictly harder test than the recipe, since the
reference's norm *reads* the bfloat16 residual. Both arms are asserted and
recorded separately, at their own PCC floor, so "passes at bf16 and not at the
production dtype" is a recorded result rather than a choice made here.

That parametrization is on **prefill only**, and the reason is a dtype the test
cannot reach: `prefill_forward` asks `rms_norm_pre_all_gather` for
`dtype=ttnn.bfloat16` statistics explicitly, so the statistics stay bfloat16
whatever the norm input is, while `decode_forward` passes no dtype at all and
the statistics therefore follow the input. A bfloat8_b decode norm input would
hand `all_gather_async` bfloat8_b statistics and a bfloat16 persistent buffer,
and which of those two the op believes is not something this file can establish
without running it. Decode therefore runs at the production residual dtype
only, which is the arm the residual claim needs; the norm-precision question is
answered on prefill, where it costs no guess.

Blackhole deltas against the Wormhole suite
-------------------------------------------
* `num_links` is 2 -- the descriptor's `fabric_links`, gated below. Wormhole's 4
  deadlocks with no traceback and 1 has stalled a real axis-1 CCL.
* **No fused collective.** `fused_rms_minimal` is one of the five fused ops
  whose 1D-multicast writers no-op on the 2D-torus fabric, moving no data and
  raising nothing, so `RMSNorm2DResidualPolicy.FUSED_DECODE` is never selected
  here and `Topology.Ring` is paired with `FABRIC_2D_TORUS_XY` instead of
  `FABRIC_1D_RING`.
* **Nothing narrows `semaphore_cores`.** The Wormhole suite narrows them to the
  norm grid, which is safe *only* for the fused RMS all-gather that binds its
  semaphore to a grid it owns. `all_gather_async` picks sender cores from the
  worker sub-device, and a narrower semaphore leaves a sender polling L1 its own
  core never zeroed -- a hang, not a failure.
* **No hardcoded core coordinate.** The decode norm grid and the statistics
  shard come from the descriptor's `norm_origin` through the production recipe
  helpers; the Wormhole suite's literal `CoreCoord(2, 0)` stats placement, which
  exists only to satisfy the fused path's `_require_fused_stats_placement`, is
  not carried.
* The decode norm is still *sharded*, unlike the reference's prefetcher-free
  recipe, because the module cannot express anything else: `is_resolved()`
  requires `decode_progcfg` for a distributed geometry, and a sharded layernorm
  program config over a DRAM-interleaved input aborts. Only the *fused* decode
  norm is avoided. The grid it lands on -- two columns from `norm_origin`,
  `local_dim / 256` rows tall -- is inside the Blackhole worker envelope.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
import ttnn
from tests.models.galaxy.galaxy_hardware import BHGLX_DEVICE_PARAMS
from tests.modules._bh_galaxy_hardware import (
    bh_galaxy_mode_plan,
    bh_galaxy_resources_config,
    bh_galaxy_topology,
    compose_2d_sharded_tensor,
    deallocate_module_weights,
    deallocate_tensor,
    exact_tensor_resource,
    record_module_output,
    require_bh_galaxy_ccl_resources,
)
from tests.support.comparison import comp_pcc

from tt_transformers.models.galaxy import (
    GalaxyCollectivePlan,
    GalaxyResourceKey,
    GalaxyTensorSpec,
)
from tt_transformers.models.galaxy.recipes import (
    GALAXY_COLUMNS,
    GALAXY_MESH_SHAPE,
    GALAXY_PHYSICAL_BATCH,
    GALAXY_ROWS,
    TILE,
    core_ranges,
    distributed_norm_stats_memory_config,
    width_sharded_memory_config,
)
from tt_transformers.modules.lazy_weight import LazyWeight
from tt_transformers.modules.rmsnorm.rmsnorm_2d import (
    RMSNorm2D,
    RMSNorm2DConfig,
    RMSNorm2DGeometry,
    RMSNorm2DResidualPolicy,
)

EPS = 1e-6

#: Layers of residual stream every distributed test runs. Two is the minimum at
#: which a dropped `ff_out` is visible at all, and this file exists for that.
LAYERS = 2

#: Head-local Q/K norm width. `head_dim` on both Galaxy geometries, and the one
#: path in this module that needs no collective at all.
QK_NORM_WIDTH = 128

#: Fabric links per direction on Blackhole. Read off the descriptor rather than
#: passed to a collective from here; the constant exists so that a descriptor
#: reporting anything else fails loudly instead of silently re-tuning every
#: collective in the file.
BLACKHOLE_FABRIC_LINKS = 2

#: The origin `RMSNorm2D` hardcodes for the distributed decode norm grid, and
#: with it for the statistics shard it binds to that grid's first core. The
#: Blackhole descriptor's `norm_origin` is the same pair, which is what lets
#: this suite hand the module a descriptor-derived input placement and still
#: have the module's own resolved `decode_stats_memcfg` land on the core the
#: persistent all-gather buffer was allocated on. Nothing else would catch a
#: divergence: the collective would write the buffer allocated here while the op
#: was told a different placement.
RMSNORM_2D_NORM_ORIGIN = (2, 0)

#: Per-arm PCC floors. bfloat8_b stores a shared exponent per block, which costs
#: of the order of 1e-2 of correlation on a normalized output, so a single floor
#: would either excuse the bfloat16 arm or fail the production one for
#: quantization. Both arms print their measured value regardless: `comp_pcc`
#: logs nothing on success, so a passing stage would otherwise tell us nothing.
#: A dropped `ff_out` lands near 0.71 and is caught by either floor.
NORM_PRECISIONS = {
    "bf16": (ttnn.bfloat16, 0.99),
    "bf8": (ttnn.bfloat8_b, 0.98),
}


def _reference_norm(weight: torch.Tensor) -> torch.nn.RMSNorm:
    """Reference normalization: `torch.nn.RMSNorm`, as the Wormhole and 1D
    suites compare against, rather than a hand-written variance/rsqrt
    re-implementation and never another TT path."""
    reference = torch.nn.RMSNorm(weight.numel(), eps=EPS).to(torch.bfloat16)
    with torch.no_grad():
        reference.weight.copy_(weight)
    return reference


def _lazy(source: torch.Tensor, mesh_device: ttnn.MeshDevice) -> LazyWeight:
    """Wrap a norm weight, leaving every placement field for the module.

    `_resolve_2d_config` reshapes the source to `(1, 1, dim // 32, 32)` and
    resolves the layout, memory config and mesh mapper itself, and
    `resolve_lazy_weight` only fills fields that are `None`. Passing them from
    here would be a second copy of the module's own weight recipe.
    """

    return LazyWeight(source=source, device=mesh_device)


def _place_activation(
    source: torch.Tensor,
    mesh_device: ttnn.MeshDevice,
    *,
    memory_config: ttnn.MemoryConfig,
    dtype: ttnn.DataType,
    geometry: RMSNorm2DGeometry,
) -> ttnn.Tensor:
    """Place an activation exactly as `_load_input_device_tensor_2d` would.

    Built here, and handed to the module as a `ttnn.Tensor` rather than as a
    `LazyWeight`, so that the test owns the handle. A `LazyWeight` passed to the
    module records its device tensor on the module's *copy*: `resolve_lazy_weight`
    returns a `replace()`d object, so the caller's `lazy._value` stays `None` and
    the tensor it created can never be deallocated. The Wormhole suite's input
    cleanup is a no-op for that reason, which is harmless for two DRAM
    invocations and not harmless for a multi-layer L1-sharded decode stream.

    The placement itself is the module's: sharded on the last dimension across
    the four mesh columns and replicated across the eight rows for a distributed
    norm, fully replicated for a head-local one.
    """

    placements = (
        [ttnn.PlacementReplicate(), ttnn.PlacementShard(-1)]
        if geometry is RMSNorm2DGeometry.DISTRIBUTED
        else [ttnn.PlacementReplicate(), ttnn.PlacementReplicate()]
    )
    return LazyWeight(
        source=source,
        device=mesh_device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=memory_config,
        mesh_mapper_config=ttnn.MeshMapperConfig(
            placements=placements,
            mesh_shape_override=ttnn.MeshShape(*GALAXY_MESH_SHAPE),
        ),
    ).get_device_weight()


def _release(live: list[Any], *tensors: Any) -> None:
    """Deallocate `tensors` and drop them from the live set, once."""

    for tensor in tensors:
        if tensor is None:
            continue
        deallocate_tensor(tensor)
        live[:] = [existing for existing in live if existing is not tensor]


def _pcc(expected: torch.Tensor, actual: torch.Tensor, threshold: float) -> tuple[bool, float]:
    passing, value = comp_pcc(expected.float(), actual.float(), threshold)
    return bool(passing), float(value)


def _report_columns(expected: torch.Tensor, actual: torch.Tensor, threshold: float, label: str) -> None:
    """Correlate per mesh column, and name a permutation when there is one.

    Aggregate PCC is close to useless for a column-local sharding bug: those
    land in 0 to 0.01 overall, and what distinguishes a channel-order mismatch
    from bad numerics is that every column correlates well with *some* reference
    column, just not its own. Bad numerics is bad everywhere. The 4x4 search
    only runs when a column has already failed, so the passing path stays cheap.

    Only columns are available: `compose_2d_sharded_tensor` returns the first
    row replica, so a per-mesh-row correlation would need a second composer and
    this suite does not build one.
    """

    width = int(expected.shape[-1]) // GALAXY_COLUMNS

    def column(tensor: torch.Tensor, index: int) -> torch.Tensor:
        return tensor[..., index * width : (index + 1) * width]

    measured = [_pcc(column(expected, index), column(actual, index), threshold) for index in range(GALAXY_COLUMNS)]
    print(
        f"[{label}] per-mesh-column PCC: "
        + " ".join(f"c{index}={value:.6f}" for index, (_passing, value) in enumerate(measured))
    )
    for index, (passing, _value) in enumerate(measured):
        if passing:
            continue
        candidates = [
            _pcc(column(expected, reference), column(actual, index), threshold)[1]
            for reference in range(GALAXY_COLUMNS)
        ]
        best = max(range(GALAXY_COLUMNS), key=candidates.__getitem__)
        print(
            f"[{label}] column {index} correlates best with reference column {best} "
            f"({candidates[best]:.6f}); a permutation across columns is a channel-order "
            "mismatch rather than bad numerics"
        )


def _assert_claim(
    *,
    expected: torch.Tensor,
    dropped: torch.Tensor,
    actual: torch.Tensor,
    threshold: float,
    label: str,
    columns: bool,
) -> None:
    """Assert one correctness claim, against both competing hypotheses.

    `dropped` is what the tensor would be if this layer's `ff_out` never reached
    the residual stream. Printing both correlations is what makes the two
    failures distinguishable on the first run: a dropped residual scores high
    against `dropped`, bad numerics scores low against both.
    """

    passing, value = _pcc(expected, actual, threshold)
    _dropped_passing, dropped_value = _pcc(dropped, actual, threshold)
    print(f"[{label}] PCC(expected)={value:.6f} PCC(ff_out-dropped)={dropped_value:.6f} threshold={threshold}")
    if columns:
        _report_columns(expected, actual, threshold, label)
    assert passing, (
        f"{label} failed PCC>={threshold}: got {value:.6f}, while the "
        f"'this layer's ff_out never reached the residual' hypothesis scores {dropped_value:.6f}. "
        "Higher there means the residual add did not happen; similar low values in both mean bad numerics."
    )
    assert value > dropped_value, (
        f"{label} correlates better with the dropped-ff_out hypothesis ({dropped_value:.6f}) "
        f"than with the expected value ({value:.6f}): the residual add is not reaching this output."
    )


def _norm_grid_memory_config(topology: Any, local_dim: int) -> ttnn.MemoryConfig:
    """Return the distributed decode placement, derived from the descriptor.

    `distributed_norm_decode_memory_config` is the production form of this and
    needs a full `GalaxyDenseGeometry`; this builds the identical placement from
    the one number it actually depends on. Two columns from the descriptor's
    `norm_origin`, `local_dim / 256` rows tall, four width tiles per core -- the
    same grid `RMSNorm2D` resolves for itself, through the same canonical
    `core_ranges` and `width_sharded_memory_config` helpers, so the two cannot
    disagree about shard shape or about `CoreRangeSet` construction order.
    """

    origin_x, origin_y = topology.norm_origin
    if (origin_x, origin_y) != RMSNORM_2D_NORM_ORIGIN:
        pytest.fail(
            f"descriptor norm_origin {(origin_x, origin_y)} differs from the origin RMSNorm2D hardcodes "
            f"{RMSNORM_2D_NORM_ORIGIN}; the module would resolve its statistics shard onto a different core "
            "than the persistent all-gather buffer this suite allocates, which no assertion here would catch"
        )
    grid_height = local_dim // (GALAXY_ROWS * TILE)
    cores = core_ranges((origin_x, origin_y, origin_x + 1, origin_y + grid_height - 1))
    return width_sharded_memory_config(local_dim, cores)


def _require_unfused_residual(topology: Any) -> None:
    """Fail if the descriptor claims the fused residual norm on Blackhole.

    Every recipe in this file is the unfused one. If the capability ever reports
    `True` here, the right response is to re-derive the recipe, not to keep
    measuring the unfused path and report it as the Blackhole result.
    """

    if topology.capabilities.has_fused_residual_norm:
        pytest.fail(
            "the Blackhole descriptor reports has_fused_residual_norm=True; this suite measures the "
            "unfused caller-owned residual recipe that capability being False is the reason for"
        )


def _resources_config(
    mesh_device: ttnn.MeshDevice,
    topology: Any,
    stream_memcfg: ttnn.MemoryConfig,
    *,
    sequence_lengths: tuple[int, ...] = (128, 2048),
) -> Any:
    """Build the Blackhole statistics-gather resources for both modes.

    Decode mirrors the production `norm_stats` plan: `Topology.Ring`, one
    semaphore per slot, and a persistent buffer L1-sharded on the first core of
    the norm input grid, derived from `stream_memcfg` by
    `distributed_norm_stats_memory_config` rather than named. Deriving it is the
    point -- the module resolves `decode_stats_memcfg` from its own norm origin
    and hands that placement to `all_gather_async` alongside this buffer, so a
    second, independently written placement is exactly the disagreement that
    would make the collective write one address while the op reads another.

    Prefill mirrors production too: `Topology.Linear` and a DRAM persistent
    buffer. `prefill_forward` calls `ttnn.all_gather`, which takes no persistent
    output tensor at all, but `_select_all_gather_resources` still requires one
    resource per exact geometry, so the spec is an allocation the op never uses.
    """

    links = topology.fabric_links
    if links != BLACKHOLE_FABRIC_LINKS:
        pytest.fail(
            f"descriptor reports {links} fabric links, expected {BLACKHOLE_FABRIC_LINKS} on Blackhole. "
            "Over-requesting links deadlocks with no traceback and under-requesting has stalled a real "
            "axis-1 CCL, so this suite will not guess."
        )

    decode_stats = GalaxyCollectivePlan(
        key=GalaxyResourceKey("all_gather", 1, (1, 1, GALAXY_PHYSICAL_BATCH, TILE), GALAXY_PHYSICAL_BATCH),
        topology=ttnn.Topology.Ring,
        num_links=links,
        semaphores_per_slot=1,
        persistent_output_specs=(
            GalaxyTensorSpec(
                (1, 1, GALAXY_PHYSICAL_BATCH, TILE * GALAXY_COLUMNS),
                ttnn.bfloat16,
                ttnn.TILE_LAYOUT,
                distributed_norm_stats_memory_config(stream_memcfg),
            ),
        ),
    )
    prefill_stats = tuple(
        GalaxyCollectivePlan(
            key=GalaxyResourceKey("all_gather", 1, (1, 1, sequence, TILE), sequence),
            topology=ttnn.Topology.Linear,
            num_links=links,
            semaphores_per_slot=1,
            persistent_output_specs=(
                GalaxyTensorSpec(
                    (1, 1, sequence, TILE * GALAXY_COLUMNS),
                    ttnn.bfloat16,
                    ttnn.TILE_LAYOUT,
                    ttnn.DRAM_MEMORY_CONFIG,
                ),
            ),
        )
        for sequence in sequence_lengths
    )
    return bh_galaxy_resources_config(
        mesh_device,
        # `semaphore_cores` stays at the default worker envelope in both modes:
        # nothing here is the fused all-gather that owns its semaphore grid.
        prefill=bh_galaxy_mode_plan("prefill", prefill_stats, mesh_device, topology=topology),
        decode=bh_galaxy_mode_plan("decode", (decode_stats,), mesh_device, topology=topology),
    )


def _module(
    mesh_device: ttnn.MeshDevice,
    resources: Any,
    weight: torch.Tensor,
    *,
    geometry: RMSNorm2DGeometry,
    stream_memcfg: ttnn.MemoryConfig | None = None,
) -> RMSNorm2D:
    """Configure RMSNorm2D for the Blackhole prefetcher-free recipe.

    `residual_policy` is `NONE` in every case: `FUSED_DECODE` runs
    `fused_rms_minimal`, whose 1D-multicast writers no-op on the 2D-torus
    fabric, and the Blackhole descriptor already says so with
    `has_fused_residual_norm=False`.

    No prefetch context is passed. Milestone 1 is prefetcher-free, and RMSNorm2D
    reads a prefetch context only to assert its mesh and mode -- it never asks
    for a `global_cb` or a sub-device id -- so `require_bh_galaxy_ccl_resources`
    is the right helper and there is no producer to drain.
    """

    config = RMSNorm2DConfig(
        weight=_lazy(weight, mesh_device),
        mesh_device=mesh_device,
        tt_ccl=resources.ccl if resources is not None else None,
        decode_ccl_context=resources.context("decode") if resources is not None else None,
        prefill_ccl_context=resources.context("prefill") if resources is not None else None,
        collective_resource_selector=exact_tensor_resource if resources is not None else None,
        geometry=geometry,
        residual_policy=RMSNorm2DResidualPolicy.NONE,
        eps=EPS,
        # Decode: the descriptor-derived norm grid, used for the input, the
        # residual and the output so that nothing is relocated between layers.
        # None for a head-local norm, which keeps the module's own default of
        # interleaved DRAM and returns each tensor where it found it.
        decode_input_memcfg=stream_memcfg,
        decode_residual_memcfg=stream_memcfg,
        decode_output_memcfg=stream_memcfg,
        # Prefill runs interleaved in DRAM, as both Galaxy models do.
        prefill_input_memcfg=ttnn.DRAM_MEMORY_CONFIG,
        prefill_residual_memcfg=ttnn.DRAM_MEMORY_CONFIG,
        prefill_output_memcfg=ttnn.DRAM_MEMORY_CONFIG,
    )
    return RMSNorm2D.from_config(config)


def _two_layer_residual_stream(
    mesh_device: ttnn.MeshDevice,
    resources: Any,
    module: RMSNorm2D,
    *,
    mode: str,
    rows: int,
    dim: int,
    stream_memcfg: ttnn.MemoryConfig,
    norm_dtype: ttnn.DataType,
    threshold: float,
    reference: torch.nn.RMSNorm,
    record_name: str,
) -> None:
    """Run `LAYERS` layers of a residual stream and assert every layer.

    The stream lives on device across layers and is advanced only by the
    caller-owned `ttnn.add`, which is the whole claim: layer *n*'s `ff_out` is
    present in layer *n+1*'s input because the caller put it there, never
    because the norm wrote it back.

    One `resources.synchronize(mode)` per layer, scoped to the worker
    sub-device, keeps at most two collectives in flight -- the count the
    qualified Wormhole repeat test runs against the same two-slot semaphore
    ring -- and is what the composed reads below need anyway.
    """

    stream_torch = torch.randn(1, 1, rows, dim, dtype=torch.bfloat16)
    live: list[Any] = []
    recorded: list[torch.Tensor] = []
    resources.activate(mode)
    try:
        stream = _place_activation(
            stream_torch,
            mesh_device,
            memory_config=stream_memcfg,
            dtype=ttnn.bfloat16,
            geometry=RMSNorm2DGeometry.DISTRIBUTED,
        )
        live.append(stream)
        for layer in range(LAYERS):
            label = f"{mode} rows={rows} dim={dim} layer={layer}"
            ff_out_torch = torch.randn(1, 1, rows, dim, dtype=torch.bfloat16)
            stream_before = stream_torch
            stream_torch = stream_before + ff_out_torch
            expected_norm = reference(stream_torch)
            dropped_norm = reference(stream_before)

            ff_out = _place_activation(
                ff_out_torch,
                mesh_device,
                memory_config=stream_memcfg,
                dtype=ttnn.bfloat16,
                geometry=RMSNorm2DGeometry.DISTRIBUTED,
            )
            live.append(ff_out)

            # Call shape A: the module's own unfused add. It returns the norm
            # alone; the sum it built is internal and, in decode, already
            # deallocated. That single return value *is* the evidence for "does
            # not write the residual sum back in place".
            module_normed = module(ff_out, mode=mode, residual=stream)
            if isinstance(module_normed, tuple):
                pytest.fail(
                    f"{label}: RMSNorm2DResidualPolicy.NONE returned {len(module_normed)} tensors; this suite's "
                    "caller-owned residual recipe assumes the unfused path returns the norm only"
                )
            live.append(module_normed)

            # Call shape B: caller-owned add, always at the production residual
            # dtype, and the tensor that becomes the next layer's stream.
            summed = ttnn.add(stream, ff_out, memory_config=stream_memcfg, dtype=ttnn.bfloat16)
            live.append(summed)
            norm_input = summed
            if norm_dtype != ttnn.bfloat16:
                # The recipe's bfloat8_b norm output, produced the only way a
                # caller can: the norm inherits its input's dtype.
                norm_input = ttnn.add(stream, ff_out, memory_config=stream_memcfg, dtype=norm_dtype)
                live.append(norm_input)
            caller_normed = module(norm_input, mode=mode)
            live.append(caller_normed)

            resources.synchronize(mode)

            actual_module = compose_2d_sharded_tensor(module_normed, mesh_device)
            actual_caller = compose_2d_sharded_tensor(caller_normed, mesh_device)
            actual_stream = compose_2d_sharded_tensor(summed, mesh_device)

            # The stream first: if this fails at layer 1 while layer 0 passed,
            # that is the predicted failure and the norms below are downstream
            # of it.
            _assert_claim(
                expected=stream_torch,
                dropped=stream_before,
                actual=actual_stream,
                threshold=threshold,
                label=f"{label} caller-owned residual stream",
                columns=True,
            )
            _assert_claim(
                expected=expected_norm,
                dropped=dropped_norm,
                actual=actual_caller,
                threshold=threshold,
                label=f"{label} norm of caller-owned sum",
                columns=True,
            )
            _assert_claim(
                expected=expected_norm,
                dropped=dropped_norm,
                actual=actual_module,
                threshold=threshold,
                label=f"{label} norm with module's unfused add",
                columns=True,
            )
            recorded.extend((actual_stream, actual_caller, actual_module))

            _release(live, module_normed, caller_normed, ff_out, stream)
            if norm_input is not summed:
                _release(live, norm_input)
            stream = summed
        record_module_output(record_name, *recorded)
    finally:
        _release(live, *live)


@pytest.mark.device
@pytest.mark.blackhole
@pytest.mark.galaxy_bh
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
@pytest.mark.parametrize("device_params", [BHGLX_DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("projection", ["q_norm", "k_norm"])
@torch.no_grad()
def test_rmsnorm_2d_bh_galaxy_head_local_128_qk_decode_and_prefill_repeat(mesh_device, projection):
    """Head-local 128-wide Q/K norm: no collective, and the cheapest signal here.

    Deliberately first in the file. It needs no CCL resources and therefore no
    sub-device manager, so it exercises the module's own placement on Blackhole
    silicon with nothing else in the way -- if this fails, no distributed result
    below is worth reading. Qwen3-32B is the model that carries QK-norm; the
    width is `head_dim`, which is 128 on both Galaxy geometries.
    """

    torch.manual_seed(4 if projection == "q_norm" else 5)
    topology = bh_galaxy_topology(mesh_device)
    _require_unfused_residual(topology)
    weight = torch.randn(QK_NORM_WIDTH, dtype=torch.bfloat16)
    reference = _reference_norm(weight)
    module = None
    recorded: list[torch.Tensor] = []
    try:
        module = _module(mesh_device, None, weight, geometry=RMSNorm2DGeometry.HEAD_LOCAL)
        for invocation in range(2):
            for mode, rows in (("decode", GALAXY_PHYSICAL_BATCH), ("prefill", 128), ("prefill", 2048)):
                x_torch = torch.randn(1, 1, rows, QK_NORM_WIDTH, dtype=torch.bfloat16)
                x = _place_activation(
                    x_torch,
                    mesh_device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    dtype=ttnn.bfloat16,
                    geometry=RMSNorm2DGeometry.HEAD_LOCAL,
                )
                live: list[Any] = [x]
                try:
                    normed = module(x, mode=mode)
                    if normed is not x:
                        live.append(normed)
                    # No sub-device manager is loaded for a head-local norm, so
                    # the whole-device synchronize is the correct scope here.
                    ttnn.synchronize_device(mesh_device)
                    # The head-local weight and input are replicated, so the
                    # composer's column concatenation repeats the same 128
                    # columns four times.
                    actual = compose_2d_sharded_tensor(normed, mesh_device)[..., :QK_NORM_WIDTH]
                finally:
                    _release(live, *live)
                label = f"{projection} {mode} rows={rows} invocation={invocation}"
                passing, value = _pcc(reference(x_torch), actual, 0.99)
                print(f"[{label}] PCC={value:.6f} threshold=0.99")
                assert passing, f"{label} failed PCC>=0.99: got {value:.6f}"
                recorded.append(actual)
        record_module_output(f"rmsnorm_2d_bh_head_local_128_{projection}", *recorded)
    finally:
        deallocate_module_weights(module, "weight")


@pytest.mark.device
@pytest.mark.blackhole
@pytest.mark.galaxy_bh
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
@pytest.mark.parametrize("device_params", [BHGLX_DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("dim", [8192, 5120], ids=["llama-8192", "qwen-5120"])
@torch.no_grad()
def test_rmsnorm_2d_bh_galaxy_two_layer_residual_decode_batch_32(mesh_device, dim):
    """Two layers of decode residual stream, batch 32, sharded on the norm grid.

    The flagship node of this file: the residual defect is a decode defect (it
    is decode that the fused path was written for), and the decode norm here
    runs entirely on the descriptor's norm grid, inside the Blackhole worker
    envelope, with no interleaved full-grid program anywhere in it.

    Decode runs at the production residual dtype only -- bfloat16 stream,
    bfloat16 norm -- because `decode_forward` passes no dtype to
    `rms_norm_pre_all_gather` and the statistics therefore follow the norm
    input, which would put a bfloat8_b statistics tensor against this suite's
    bfloat16 persistent all-gather buffer. The norm-precision arms live on the
    prefill node, where the module pins the statistics dtype itself.
    """

    torch.manual_seed(2)
    topology = bh_galaxy_topology(mesh_device)
    _require_unfused_residual(topology)
    norm_dtype, threshold = NORM_PRECISIONS["bf16"]
    weight = torch.randn(dim, dtype=torch.bfloat16)
    reference = _reference_norm(weight)
    stream_memcfg = _norm_grid_memory_config(topology, dim // GALAXY_COLUMNS)
    resources = require_bh_galaxy_ccl_resources(
        mesh_device,
        config=_resources_config(mesh_device, topology, stream_memcfg),
    )
    module = None
    try:
        module = _module(
            mesh_device,
            resources,
            weight,
            geometry=RMSNorm2DGeometry.DISTRIBUTED,
            stream_memcfg=stream_memcfg,
        )
        _two_layer_residual_stream(
            mesh_device,
            resources,
            module,
            mode="decode",
            rows=GALAXY_PHYSICAL_BATCH,
            dim=dim,
            stream_memcfg=stream_memcfg,
            norm_dtype=norm_dtype,
            threshold=threshold,
            reference=reference,
            record_name=f"rmsnorm_2d_bh_two_layer_decode_batch32_dim{dim}_norm_bf16",
        )
    finally:
        try:
            resources.cleanup()
        finally:
            deallocate_module_weights(module, "weight")


@pytest.mark.device
@pytest.mark.blackhole
@pytest.mark.galaxy_bh
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
@pytest.mark.parametrize("device_params", [BHGLX_DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("dim", [8192, 5120], ids=["llama-8192", "qwen-5120"])
@pytest.mark.parametrize("seq_len", [128, 2048], ids=["seq128", "seq2048"])
@pytest.mark.parametrize("norm_precision", list(NORM_PRECISIONS), ids=[f"norm-{tag}" for tag in NORM_PRECISIONS])
@torch.no_grad()
def test_rmsnorm_2d_bh_galaxy_two_layer_residual_prefill(mesh_device, dim, seq_len, norm_precision):
    """Two layers of prefill residual stream, interleaved in DRAM.

    The same claim as the decode node, on the mode where this repo's Wormhole
    models *already* own the residual add, so a failure here separates the
    Blackhole recipe from the unfused pattern itself.

    Every op on this path -- the caller's `ttnn.add`, the module's internal add,
    and both halves of the interleaved norm -- auto-selects a compute grid from
    `compute_with_storage_grid_size()` anchored at `(0, 0)`, and none of them
    takes a core grid from its caller. That is placeable only under a partition
    covering the whole grid, which is what `bh_galaxy_mode_plan` loads for
    prefill and why it loads it; read its docstring before narrowing this. A
    `Kernel group cores do not match sub device cores` here means the partition
    moved, not that the recipe is wrong.
    """

    torch.manual_seed(3)
    topology = bh_galaxy_topology(mesh_device)
    _require_unfused_residual(topology)
    norm_dtype, threshold = NORM_PRECISIONS[norm_precision]
    weight = torch.randn(dim, dtype=torch.bfloat16)
    reference = _reference_norm(weight)
    # The decode plan is built for the same mesh even though this node never
    # activates it: `GalaxyResourcesConfig` requires both modes, and the
    # statistics buffer it allocates is one 32x128 bfloat16 shard.
    stream_memcfg = _norm_grid_memory_config(topology, dim // GALAXY_COLUMNS)
    resources = require_bh_galaxy_ccl_resources(
        mesh_device,
        config=_resources_config(mesh_device, topology, stream_memcfg, sequence_lengths=(seq_len,)),
    )
    module = None
    try:
        module = _module(
            mesh_device,
            resources,
            weight,
            geometry=RMSNorm2DGeometry.DISTRIBUTED,
            stream_memcfg=stream_memcfg,
        )
        _two_layer_residual_stream(
            mesh_device,
            resources,
            module,
            mode="prefill",
            rows=seq_len,
            dim=dim,
            stream_memcfg=ttnn.DRAM_MEMORY_CONFIG,
            norm_dtype=norm_dtype,
            threshold=threshold,
            reference=reference,
            record_name=f"rmsnorm_2d_bh_two_layer_prefill_seq{seq_len}_dim{dim}_norm_{norm_precision}",
        )
    finally:
        try:
            resources.cleanup()
        finally:
            deallocate_module_weights(module, "weight")
