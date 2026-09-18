# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Is the DRAM-interleaved MLP2D path numerically right on a Blackhole Galaxy?

The Wormhole MLP2D suite (`test_mlp_2d_wh_galaxy.py`) drives the module through a
*ring* recipe: DRAM-width-sharded weights, `MatmulMultiCoreReuseMultiCast1DProgramConfig`
with `gather_in0=True`, and L1-width-sharded activations pinned to 24 named ring
cores. Two separate failures make that recipe unusable here, and only one of them
is loud:

* **Loud.** The ring memory config's shard grid uses core column `x=6`, which
  falls outside the auto-selected 1D matmul compute grid: *"Tensor shard spec
  grid ... must lie within compute grid"*. It fails to compile, which is safe.
* **Silent.** Feeding a ring-sharded all-gather output into the interleaved W2
  matmul mismatches per-device channel order and yields MLP decode PCC ~ 0 with
  nothing raised.

The answer to both is DRAM interleaved with `program_config=None`, which is what
every matmul below uses. `GalaxyCapabilities.has_ring_matmul` is already `False`
on the Blackhole descriptor and `ring_core_coords` is `None` rather than
Wormhole's coordinates, so a consumer that still reaches for a ring fails loudly;
that much is settled. What was untested, and what this file measures, is whether
the DRAM-interleaved spelling of the same math is *correct*.

## Decode, and why `decode_fused_ccl=False` is in every config here

`MLP2D.decode_forward` stages 1-2 have two spellings. The default,
`decode_fused_ccl=True`, takes `ttnn.experimental.llama_rs_matmul` plus
`ttnn.experimental.llama_reduce_scatter` -- two of the five fused galaxy CCLs
that use 1D-multicast writers and therefore **no-op** on `FABRIC_2D_TORUS_XY`,
moving no data and raising nothing, so the collective *appears* to run. Every
decode test here passes `decode_fused_ccl=False`, which takes
`MLP2D._unfused_decode_w1_w3`: two `ttnn.linear` calls and two
`reduce_scatter_minimal_async` calls, the spelling `prefill_forward` already
uses for the same math. Stages 3-7 were already non-fused, so that flag is the
whole of the difference between the two decode graphs.

The unfused path also requires a `collective_resource_selector` --
`_resolve_mlp2d_config` refuses the combination otherwise -- because it issues
**two** axis-1 reduce-scatters whose outputs are live at once in stage 3's gated
multiply. With no selector both would alias one persistent output buffer and the
multiply would read W3 twice, silently. `exact_tensor_resource` honours the
`"w1"` / `"w3"` sequence keys, so the decode plan below carries two distinct
axis-1 reduce-scatter resources.

## The instrumentation, which is the point of the file

Aggregate PCC is close to useless for a column-local sharding bug: those give
PCC in 0-0.01 and need per-mesh-row and per-mesh-column correlation to localize.
`_report_permutation` correlates every actual mesh slice against **every**
reference slice and reports the argmax, so a channel-order mismatch appears as a
*permutation* -- each slice well-correlated with some slice, just not its own --
and is immediately distinguishable from bad numerics, where correlation is bad
everywhere. The full matrix is logged on every run, not only on failure, because
`comp_pcc` logs nothing on success and a passing stage tells you nothing unless
you print it. Decode additionally reports **per-user** correlation, one value per
activation row, which is what localizes a failure that hits some users and not
others.

Tests are ordered cheapest-first, and decode before prefill: decode runs 32 rows
against prefill's 128 and 2048, so it is both the cheaper node and the one
carrying the predicted failure.
"""

from __future__ import annotations

import math

import pytest
import torch
import ttnn
from loguru import logger
from tests.models.galaxy.galaxy_hardware import BHGLX_DEVICE_PARAMS, GALAXY_MESH_SHAPE
from tests.modules._bh_galaxy_hardware import (
    bh_galaxy_mode_plan,
    bh_galaxy_resources_config,
    bh_galaxy_topology,
    bh_galaxy_worker_sub_device,
    compose_2d_sharded_tensor,
    deallocate_module_weights,
    deallocate_tensor,
    exact_tensor_resource,
    record_module_output,
    require_bh_galaxy_prefetch_free_resources,
)
from tests.modules._hf_reference import get_mlp_weights_from_ref_model
from tests.modules._mlp_2d_galaxy import MLP_PCC_THRESHOLD, mlp_pcc
from tests.modules._mlp_2d_galaxy import prefill_weight_lazies as _dram_interleaved_weight_lazies
from tests.modules._mlp_2d_galaxy import reference_mlp as _reference_mlp

from tt_transformers.models.galaxy import GalaxyCollectivePlan, GalaxyResourceKey, GalaxyTensorSpec
from tt_transformers.models.galaxy.recipes import (
    GALAXY_COLUMNS,
    GALAXY_PHYSICAL_BATCH,
    GALAXY_ROWS,
    TILE,
    dense_matmul_program_config,
    galaxy_fabric_links,
    width_sharded_memory_config,
    worker_cores,
)
from tt_transformers.modules.lazy_weight import LazyWeight
from tt_transformers.modules.mlp.mlp_2d import (
    MLP2D,
    MLP2DConfig,
    _compute_kernel_config_hifi2_fp16,
    _default_prefill_program_config,
    _load_input_device_tensor,
    _prefetch_kwargs,
)

#: Per-core shard width of stage 6's all-reduce scratch, matching the qualified
#: Wormhole decode plan's `(TILE, 1024)` shard. The scratch's total width is this
#: times the worker-core count, because that is what width sharding means.
_ALL_REDUCE_SHARD_WIDTH = 1024

#: The mesh's fabric link budget, asserted rather than assumed.
#:
#: Blackhole Galaxy's mesh graph descriptor declares `channels { count: 2 }`
#: against Wormhole's 4. The ring and line CCLs index ethernet channels by link,
#: so an over-requested `num_links` overruns the channel array and **deadlocks
#: with no traceback**; the resulting `TT_FATAL` out of `enqueue_mesh_workload`
#: leaves the mesh un-drainable, where SIGTERM cannot service it and SIGKILL plus
#: a reset is the only recovery. That is the single most expensive mistake
#: available in this file, so the number is read from the descriptor via
#: `galaxy_fabric_links` and then checked against this literal before any
#: collective is planned. A descriptor that drifts fails on the host.
BLACKHOLE_FABRIC_LINKS = 2

#: bfloat8_b activations and collectives, matching the Wormhole suite. The wider
#: model also carries bfloat8_b weights; the narrower one keeps bfloat16, which is
#: the Wormhole suite's per-model choice and not a tuning decision made here.
ACTIVATION_DTYPE = ttnn.bfloat8_b

#: `MLP2DConfig.prefill_len_cutoff`'s resolved default. Above it,
#: `prefill_forward` reshapes the activation to `[1, seq // cutoff, cutoff, -1]`,
#: which changes the geometry the collectives are keyed on, so the resource plan
#: has to agree with it.
PREFILL_LEN_CUTOFF = 1024

#: The sequence length the cheap prefill node uses. 128 is the prefill chunk
#: alignment and stays below `PREFILL_LEN_CUTOFF`, so no reshape is involved.
PROBE_SEQUENCE = 128

#: Logical users for the padded-batch decode node.
#:
#: Deliberately **not** a multiple of `TILE`, so `tile_padded_batch_rows` is
#: strictly larger than the user count and the tile padding is real rather than
#: notional. The activation is 32 rows tall carrying 24 users, which is what
#: separates code that derives the decode height from code that happens to write
#: 32. At the production batch of 32 the two numbers coincide and the
#: distinction is untestable, which is why one node runs short.
PADDED_BATCH_USERS = 24


def tile_padded_batch_rows(users: int) -> int:
    """Return the decode activation height for `users` users.

    The decode height is the *tile-padded* batch, not the user count and not a
    fixed 32: a Galaxy decode step carries `GALAXY_PHYSICAL_BATCH` users at the
    production shape, where the two happen to be equal, and any shorter batch
    still occupies whole tiles. Every decode shape below is derived from this
    rather than written, so the resource plan, the activation and the reference
    move together.
    """

    if users <= 0:
        raise ValueError(f"users must be positive, got {users}")
    return TILE * math.ceil(users / TILE)


# =============================================================================
# Correlation instrumentation
# =============================================================================


def _correlation(expected: torch.Tensor, actual: torch.Tensor) -> float:
    """Pearson correlation of two tensors, flattened, in float32."""

    left = expected.detach().to(torch.float32).flatten()
    right = actual.detach().to(torch.float32).flatten()
    left = left - left.mean()
    right = right - right.mean()
    scale = float(left.norm()) * float(right.norm())
    if scale == 0.0:
        return float("nan")
    return float(torch.dot(left, right)) / scale


def _correlation_matrix(expected: list[torch.Tensor], actual: list[torch.Tensor]) -> list[list[float]]:
    """Correlate every actual slice against every expected slice.

    Row `i` is actual slice `i`; column `j` is expected slice `j`. The diagonal is
    the correlation each slice is supposed to have.
    """

    return [[_correlation(reference, measured) for reference in expected] for measured in actual]


def _format_matrix(matrix: list[list[float]]) -> str:
    header = "        " + " ".join(f"ref{index:<5d}" for index in range(len(matrix[0])))
    rows = [f"  got{index:<3d} " + " ".join(f"{value:+8.4f}" for value in row) for index, row in enumerate(matrix)]
    return "\n".join([header, *rows])


def _argmax(row: list[float]) -> int:
    """Index of the largest correlation in one matrix row, NaN ranking lowest."""

    best, best_value = 0, -2.0
    for index, value in enumerate(row):
        # NaN compares false against everything, so test it explicitly rather
        # than letting `max` keep whichever index it happened to see first.
        if value == value and value > best_value:
            best, best_value = index, value
    return best


def _report_permutation(label: str, expected: list[torch.Tensor], actual: list[torch.Tensor]) -> None:
    """Log the full correlation matrix, then assert it is the identity.

    A channel-order mismatch shows up here as a permutation: every actual slice
    correlates well with *some* reference slice, just not its own. Bad numerics
    show up as a matrix that is bad everywhere. The two are indistinguishable
    from an aggregate PCC and obvious from this matrix, which is why it is logged
    unconditionally and repeated in the assertion message.

    **Every reference slice must be distinguishable from the others.** Correlating
    against slices that are equal to one another makes the argmax a coin toss and
    the identity assertion meaningless; use `_report_correlations` for the
    replicated case instead.
    """

    assert len(expected) == len(actual), f"{label}: {len(actual)} measured slices for {len(expected)} reference slices"
    matrix = _correlation_matrix(expected, actual)
    rendered = _format_matrix(matrix)
    diagonal = [matrix[index][index] for index in range(len(matrix))]
    argmax = [_argmax(row) for row in matrix]
    logger.info(f"{label}: argmax={argmax} diagonal={[round(value, 5) for value in diagonal]}\n{rendered}")
    identity = list(range(len(matrix)))
    assert argmax == identity, (
        f"{label}: best-correlated reference slice per measured slice is {argmax}, expected {identity}. "
        f"A permutation here is the per-device channel-order failure, not bad numerics.\n{rendered}"
    )
    worst = min(diagonal)
    assert worst >= MLP_PCC_THRESHOLD, (
        f"{label}: slice correlation {worst} below {MLP_PCC_THRESHOLD} with the ordering intact, "
        f"so this is a numerics failure rather than a channel-order one.\n{rendered}"
    )


def _report_correlations(label: str, expected: torch.Tensor, actual: list[torch.Tensor]) -> None:
    """Log one correlation per measured slice against a single reference.

    For the replicated case: after the axis-0 all-reduce every mesh row holds the
    same result, so there is no permutation to detect and the useful question is
    which *row* disagrees. A cross-correlation matrix cannot answer that -- all
    eight reference slices would be identical -- so this reports the vector and
    names the offending rows.
    """

    values = [_correlation(expected, measured) for measured in actual]
    logger.info(f"{label}: per-slice correlation={[round(value, 5) for value in values]}")
    failing = [index for index, value in enumerate(values) if not value >= MLP_PCC_THRESHOLD]
    assert not failing, (
        f"{label}: slices {failing} fall below {MLP_PCC_THRESHOLD}; "
        f"per-slice correlation was {[round(value, 5) for value in values]}"
    )


def _report_per_user(label: str, expected: torch.Tensor, actual: torch.Tensor, *, users: int) -> None:
    """Correlate each decode activation row on its own, then bound the padding.

    One value per user, because a decode failure that hits some users and not
    others is invisible in an aggregate, and column-local sharding bugs need
    per-user correlation to localize. Rows at and
    above `users` are tile padding, whose reference is exactly zero -- silu(0)
    times 0 is 0, and 0 against W2 is 0 -- so correlation is undefined there and
    the check is a magnitude bound instead. Garbage from a write outside a
    collective's intended region shows up in that bound.
    """

    rows = int(expected.shape[-2])
    values = [_correlation(expected[..., row, :], actual[..., row, :]) for row in range(users)]
    logger.info(f"{label}: per-user correlation over {users} users={[round(value, 5) for value in values]}")
    failing = [row for row, value in enumerate(values) if not value >= MLP_PCC_THRESHOLD]
    assert not failing, (
        f"{label}: users {failing} fall below {MLP_PCC_THRESHOLD}; "
        f"per-user correlation was {[round(value, 5) for value in values]}"
    )
    if rows == users:
        return
    padding = actual[..., users:rows, :]
    real_scale = float(actual[..., :users, :].abs().max())
    padding_scale = float(padding.abs().max())
    logger.info(
        f"{label}: tile padding rows {users}..{rows - 1} max|x|={padding_scale} against real max|x|={real_scale}"
    )
    assert torch.isfinite(padding).all(), f"{label}: tile padding rows {users}..{rows - 1} are not finite"
    assert padding_scale <= 0.1 * real_scale, (
        f"{label}: tile padding rows {users}..{rows - 1} carry {padding_scale} against a real-row maximum of "
        f"{real_scale}; their reference is exactly zero, so something wrote outside its intended rows"
    )


def _log_stage_pcc(label: str, expected: torch.Tensor, actual: torch.Tensor) -> float:
    """Print one stage's PCC and assert it at the suite threshold.

    `comp_pcc` logs nothing on success, so the value is printed before it is
    asserted; a passing stage otherwise tells you nothing.
    """

    passing, value = mlp_pcc(expected, actual)
    logger.info(f"stage PCC {label}: {value}")
    assert passing, f"{label} failed PCC>={MLP_PCC_THRESHOLD}: {value}"
    return value


# =============================================================================
# Mesh composition and the per-device reference slices
# =============================================================================


def _compose_mesh_grid(tensor: ttnn.Tensor, mesh_device: ttnn.MeshDevice) -> torch.Tensor:
    """Return the mesh-resolved tensor with the mesh grid still visible.

    Mesh axis 0 (the 8 rows) lands on tensor dim 1 and mesh axis 1 (the 4
    columns) is concatenated onto the last dim, so `grid[:, r]` is mesh row `r`
    and the last dim splits into the four mesh columns in order. This is the
    uncollapsed view of what `compose_2d_sharded_tensor` returns; that helper
    takes `[:, :1]` because its callers only ever want the row-replicated result,
    and the discarded rows are exactly what the instrumentation here needs.
    """

    return ttnn.to_torch(
        tensor,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(1, 3), mesh_shape=GALAXY_MESH_SHAPE),
    ).to(torch.float32)


def _column_slices(row: torch.Tensor, count: int) -> list[torch.Tensor]:
    """Split a composed mesh row into its `count` equal width blocks."""

    width = row.shape[-1] // count
    assert row.shape[-1] == width * count, f"width {row.shape[-1]} does not split into {count} blocks"
    return [row[..., index * width : (index + 1) * width] for index in range(count)]


class _MeshSlices:
    """The per-device slice of each MLP stage, on a `(8, 4)` mesh.

    MLP2D shards `K` over the four mesh columns and the hidden dimension over the
    eight mesh rows, so device `(r, c)` sees one specific slice of every stage.
    Both staged tests need the same arithmetic and a second copy of it would be a
    second chance to get it wrong.
    """

    def __init__(self, dim: int, hidden_dim: int):
        self.local_k = dim // GALAXY_COLUMNS
        self.local_dim = dim // GALAXY_COLUMNS
        self.local_hidden = hidden_dim // GALAXY_ROWS
        self.gated_width = self.local_hidden // GALAXY_COLUMNS

    def projection_partial(self, activation: torch.Tensor, weight: torch.Tensor, row: int, column: int) -> torch.Tensor:
        """W1 or W3 on device `(row, column)`: a partial sum over `K`."""

        shard = activation[..., column * self.local_k : (column + 1) * self.local_k]
        return (
            shard
            @ weight[
                column * self.local_k : (column + 1) * self.local_k,
                row * self.local_hidden : (row + 1) * self.local_hidden,
            ]
        )

    def row_hidden(self, source: torch.Tensor, row: int) -> torch.Tensor:
        """Mesh row `row`'s slice of a full hidden-width tensor."""

        return source[..., row * self.local_hidden : (row + 1) * self.local_hidden]

    def scattered_chunk(self, source: torch.Tensor, row: int, chunk: int) -> torch.Tensor:
        """The axis-1 reduce-scatter's output chunk `chunk` in mesh row `row`."""

        start = row * self.local_hidden + chunk * self.gated_width
        return source[..., start : start + self.gated_width]

    def w2_partial(self, gated: torch.Tensor, w2: torch.Tensor, row: int, column: int) -> torch.Tensor:
        """W2 on device `(row, column)`: a partial sum over the hidden dimension."""

        return (
            self.row_hidden(gated, row)
            @ w2[
                row * self.local_hidden : (row + 1) * self.local_hidden,
                column * self.local_dim : (column + 1) * self.local_dim,
            ]
        )

    def output_column(self, source: torch.Tensor, column: int) -> torch.Tensor:
        """Mesh column `column`'s slice of the reduced output."""

        return source[..., column * self.local_dim : (column + 1) * self.local_dim]


def _reference_stages(x: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor, w3: torch.Tensor):
    """CPU float32 reference for each MLP stage. Never another TT path."""

    activation = x.to(torch.float32)
    w1_full = activation @ w1.to(torch.float32)
    w3_full = activation @ w3.to(torch.float32)
    gated_full = (w1_full * torch.sigmoid(w1_full)) * w3_full
    return w1_full, w3_full, gated_full, gated_full @ w2.to(torch.float32)


# =============================================================================
# Resource plan
# =============================================================================


def _num_links(mesh_device: ttnn.MeshDevice) -> int:
    links = galaxy_fabric_links(mesh_device)
    if links != BLACKHOLE_FABRIC_LINKS:
        pytest.fail(
            f"Blackhole Galaxy fabric link budget resolved to {links}, expected {BLACKHOLE_FABRIC_LINKS}. "
            "Refusing to plan a collective: an over-requested num_links deadlocks the mesh un-drainably."
        )
    return links


def _prefill_leading_shape(sequence: int) -> tuple[int, int, int]:
    """Mirror the leading dims `prefill_forward` presents to its collectives."""

    if sequence >= PREFILL_LEN_CUTOFF:
        assert sequence % PREFILL_LEN_CUTOFF == 0, f"sequence {sequence} must divide by {PREFILL_LEN_CUTOFF}"
        return (1, sequence // PREFILL_LEN_CUTOFF, PREFILL_LEN_CUTOFF)
    return (1, 1, sequence)


def _dram(shape: tuple[int, ...], mesh_device: ttnn.MeshDevice, dtype=ACTIVATION_DTYPE) -> GalaxyTensorSpec:
    return GalaxyTensorSpec(
        shape, dtype, ttnn.TILE_LAYOUT, ttnn.DRAM_MEMORY_CONFIG, ttnn.ReplicateTensorToMesh(mesh_device)
    )


def _all_reduce_output_cores(topology, local_dim: int) -> ttnn.CoreRangeSet:
    """The width-sharded L1 placement stage 6's all-reduce reads and writes.

    Production's `recipes.distributed_norm_decode_memory_config`, reproduced
    rather than imported because that helper takes a `GalaxyDenseGeometry` and
    this file builds no geometry: a two-wide grid anchored at the descriptor's
    `norm_origin` -- `(2, 0)` on both architectures, inside the Blackhole worker
    envelope -- and `local_dim / (GALAXY_ROWS * TILE)` rows tall, which keeps
    four width tiles and so a 128-column shard on every core.

    128 columns is not incidental. `all_reduce_async` validates
    `buffer_shard_volume >= output_shard_volume * ring_size`, the ring on axis 0
    is the eight mesh rows, and the buffer's shard is `(TILE, 1024)`; so
    `TILE * shard_width * 8 <= TILE * 1024` requires `shard_width <= 128`. This
    grid sits exactly at that bound, which is why it is production's grid and not
    a smaller one.
    """

    origin_x, origin_y = topology.norm_origin
    height = local_dim // (GALAXY_ROWS * TILE)
    return ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(origin_x, origin_y), ttnn.CoreCoord(origin_x + 1, origin_y + height - 1))}
    )


def _all_reduce_buffer(mesh_device: ttnn.MeshDevice, topology, rows: int) -> GalaxyTensorSpec:
    """Stage 6's caller-owned scratch: width-sharded L1 across the worker envelope.

    The shape's last dimension is `worker_cores * 1024` because a width-sharded
    tensor's width *is* cores times shard width -- 100 cores on Blackhole, so
    102400. Wormhole's 50 cores are where the `50 * 1024` literal in the
    qualified plan came from; it was that architecture's core count written as a
    constant, which is why the production path now derives it.

    `ShardTensor2dMesh(dims=(0, 1))` matches the qualified plan's `row_shard`
    mapper, so each device owns one `(rows, 102400)` slice rather than a replica.
    """

    cores = worker_cores(topology)
    width = cores.num_cores() * _ALL_REDUCE_SHARD_WIDTH
    return GalaxyTensorSpec(
        (GALAXY_ROWS, GALAXY_COLUMNS, rows, width),
        ACTIVATION_DTYPE,
        ttnn.TILE_LAYOUT,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(cores, (TILE, _ALL_REDUCE_SHARD_WIDTH), ttnn.ShardOrientation.ROW_MAJOR),
        ),
        ttnn.ShardTensor2dMesh(mesh_device, dims=(0, 1), mesh_shape=(GALAXY_ROWS, GALAXY_COLUMNS)),
    )


def _decode_collectives(
    mesh_device: ttnn.MeshDevice, dim: int, hidden_dim: int, rows: int, topology
) -> tuple[GalaxyCollectivePlan, ...]:
    """Plan the collectives `decode_forward` issues with `decode_fused_ccl=False`.

    Four, not the Wormhole plan's four-with-different-shapes: two axis-1
    reduce-scatters keyed `"w1"` and `"w3"`, the axis-1 all-gather that feeds W2,
    and the axis-0 all-reduce. The Wormhole decode plan gives the reduce-scatter
    a packet-sharded L1 intermediate and the all-reduce an L1 scratch across
    named worker cores, because both serve the 24-core ring; neither has a
    counterpart in an interleaved recipe, so every buffer here is DRAM.

    **Two distinct reduce-scatter resources is a correctness requirement, not
    tidiness.** Both outputs are live at once in stage 3's gated multiply, so one
    shared persistent buffer would make the multiply read W3 twice.
    `_resolve_mlp2d_config` refuses `decode_fused_ccl=False` without a selector
    for exactly this reason, and `exact_tensor_resource` derives the keys below
    from the tensor plus the module's stage string.

    `rows` is the tile-padded decode height, which is where
    `tile_padded_batch_rows` reaches the plan: the reduce-scatter's resources are
    keyed on `math.prod(shape[:-1])`, so a plan built for a 32-row activation
    cannot be picked up by a 64-row one.
    """

    links = _num_links(mesh_device)
    local_hidden = hidden_dim // GALAXY_ROWS
    gated_width = local_hidden // GALAXY_COLUMNS
    local_dim = dim // GALAXY_COLUMNS
    hidden_shape = (1, 1, rows, local_hidden)
    gated_shape = (1, 1, rows, gated_width)
    output_shape = (1, 1, rows, local_dim)
    plans = [
        GalaxyCollectivePlan(
            key=GalaxyResourceKey("reduce_scatter", 1, hidden_shape, (rows, stage)),
            topology=ttnn.Topology.Ring,
            num_links=links,
            semaphores_per_slot=3,
            persistent_output_specs=(_dram(gated_shape, mesh_device),),
            intermediate_output_specs=(_dram(hidden_shape, mesh_device),),
        )
        for stage in ("w1", "w3")
    ]
    plans.append(
        GalaxyCollectivePlan(
            # `_all_gather_axis1(gated, ..., "decode")` passes no stage key, so the
            # sequence key is the bare token count rather than a pair.
            key=GalaxyResourceKey("all_gather", 1, gated_shape, rows),
            topology=ttnn.Topology.Ring,
            num_links=links,
            persistent_output_specs=(_dram(hidden_shape, mesh_device),),
        )
    )
    plans.append(
        GalaxyCollectivePlan(
            key=GalaxyResourceKey("all_reduce", 0, output_shape, rows),
            topology=ttnn.Topology.Ring,
            num_links=links,
            # **The one buffer in this file that cannot be DRAM.** The comment
            # here used to reason that "in an interleaved recipe the buffer
            # follows the output it reduces, which is in DRAM", and that was
            # wrong against the op's own contract:
            # `all_reduce_async_device_operation.cpp:46-57` requires
            # `WIDTH_SHARDED` of the input, of this buffer **and** of the output
            # config, and `:28` additionally refuses a DRAM input on Blackhole
            # outright. Measured 2026-09-17: both `decode_batch_32_repeat` node
            # ids aborted here while the matmul nodes passed. There is no
            # interleaved spelling of this op on any architecture.
            #
            # So this follows the qualified Wormhole decode plan
            # (`plans.build_galaxy_decode_collectives`) rather than inventing a
            # size: a `(TILE, 1024)` shard width-sharded across the worker
            # envelope. The width is `worker_cores * 1024` because that is what
            # width sharding means -- 100 cores on Blackhole, so 102400, where
            # Wormhole's 50 cores gave the `50 * 1024` that plan used to spell as
            # a literal.
            #
            # The validation it has to satisfy is
            # `buffer_shard_volume >= output_shard_volume * ring_size`:
            # the output is `local_dim` over `_ALL_REDUCE_OUTPUT_CORES` cores, so
            # `TILE * local_dim / cores * GALAXY_ROWS <= TILE * 1024`, i.e.
            # `local_dim / cores <= 128`. Llama's 2048 needs >= 16 cores and
            # Qwen's 1280 >= 10, which is exactly what the residual grid below
            # provides -- the same grid production uses, for the same reason.
            persistent_output_specs=(_all_reduce_buffer(mesh_device, topology, rows),),
        )
    )
    return tuple(plans)


def _prefill_collectives(
    mesh_device: ttnn.MeshDevice, dim: int, hidden_dim: int, sequences: tuple[int, ...]
) -> tuple[GalaxyCollectivePlan, ...]:
    """Plan exactly the collectives `MLP2D.prefill_forward` issues.

    Every buffer is DRAM interleaved and replicated, which is the prefill shape
    the qualified Wormhole plan already used -- the ring-specific L1 placements in
    `_mlp_2d_galaxy.decode_ring_config` belong to the ring path and have no
    counterpart here. The only substantive difference from Wormhole is
    `num_links`: the Wormhole plan asks for 4 and Blackhole's fabric has 2.

    Keys are derived the same way `exact_tensor_resource` derives them from the
    tensor the TTNN op sees -- `math.prod(shape[:-1])`, optionally paired with the
    module's stage string -- so a resource allocated for one shape can never be
    picked up for another.
    """

    links = _num_links(mesh_device)
    local_hidden = hidden_dim // GALAXY_ROWS
    gated_width = local_hidden // GALAXY_COLUMNS
    local_dim = dim // GALAXY_COLUMNS
    row_scattered_dim = local_dim // GALAXY_ROWS

    plans: list[GalaxyCollectivePlan] = []
    for sequence in sequences:
        leading = _prefill_leading_shape(sequence)
        tokens = math.prod(leading)
        hidden_shape = (*leading, local_hidden)
        gated_shape = (*leading, gated_width)
        output_shape = (1, 1, sequence, local_dim)
        scattered_output_shape = (1, 1, sequence, row_scattered_dim)
        # Stages 2 and 4: the axis-1 reduce-scatter of each of W1 and W3, then the
        # axis-1 all-gather of the gated product.
        plans.extend(
            GalaxyCollectivePlan(
                key=GalaxyResourceKey("reduce_scatter", 1, hidden_shape, (tokens, stage)),
                topology=ttnn.Topology.Ring,
                num_links=links,
                semaphores_per_slot=3,
                persistent_output_specs=(_dram(gated_shape, mesh_device),),
                intermediate_output_specs=(_dram(hidden_shape, mesh_device),),
            )
            for stage in ("w1", "w3")
        )
        plans.append(
            GalaxyCollectivePlan(
                key=GalaxyResourceKey("all_gather", 1, gated_shape, (tokens, "gated")),
                topology=ttnn.Topology.Ring,
                num_links=links,
                persistent_output_specs=(_dram(hidden_shape, mesh_device),),
            )
        )
        # Stage 6: prefill's axis-0 all-reduce is a reduce-scatter followed by an
        # all-gather, both issued through the *non-persistent* `ttnn.reduce_scatter`
        # and `ttnn.all_gather`. Their persistent buffers go unused, but
        # `_validate_collective_resources` still requires them to exist, and the
        # topology and num_links it reads off them do reach the op.
        plans.append(
            GalaxyCollectivePlan(
                key=GalaxyResourceKey("reduce_scatter", 0, output_shape, (sequence, "final")),
                topology=ttnn.Topology.Ring,
                num_links=links,
                semaphores_per_slot=3,
                persistent_output_specs=(_dram(scattered_output_shape, mesh_device),),
                intermediate_output_specs=(_dram(output_shape, mesh_device),),
            )
        )
        plans.append(
            GalaxyCollectivePlan(
                key=GalaxyResourceKey("all_gather", 0, scattered_output_shape, (sequence, "final")),
                topology=ttnn.Topology.Ring,
                num_links=links,
                persistent_output_specs=(_dram(output_shape, mesh_device),),
            )
        )
    return tuple(plans)


def _require_dram_interleaved_capabilities(mesh_device: ttnn.MeshDevice):
    """Validate the descriptor, then assert the capabilities this file assumes."""

    topology = bh_galaxy_topology(mesh_device)
    capabilities = topology.capabilities
    if capabilities.has_ring_matmul or topology.ring_core_coords is not None:
        pytest.fail(
            "the Blackhole descriptor offers a ring matmul path; this suite is the DRAM-interleaved "
            "recipe and its resolved memory configs would no longer be the ones under test"
        )
    if capabilities.has_fused_ccl:
        pytest.fail(
            "the Blackhole descriptor reports has_fused_ccl=True; the fused 1D-multicast collectives "
            "no-op on FABRIC_2D_TORUS_XY, so every decode config here passes decode_fused_ccl=False"
        )
    return topology


def _resources(
    mesh_device: ttnn.MeshDevice,
    topology,
    dim: int,
    hidden_dim: int,
    *,
    decode_rows: int,
    prefill_sequences: tuple[int, ...],
):
    """Create the prefetch-free resources for both modes.

    `GalaxyResourcesConfig` requires both plans and `GalaxyModePlan` requires at
    least one collective each, so both are always planned even when one node only
    activates one of them. Both sets are DRAM buffers in the tens to hundreds of
    kilobytes, which is cheaper than a placeholder that describes something no
    test does.
    """

    config = bh_galaxy_resources_config(
        mesh_device,
        prefill=bh_galaxy_mode_plan(
            "prefill",
            _prefill_collectives(mesh_device, dim, hidden_dim, prefill_sequences),
            mesh_device,
            topology=topology,
        ),
        decode=bh_galaxy_mode_plan(
            "decode",
            _decode_collectives(mesh_device, dim, hidden_dim, decode_rows, topology),
            mesh_device,
            topology=topology,
        ),
    )
    return require_bh_galaxy_prefetch_free_resources(mesh_device, config=config)


# =============================================================================
# Module and activation construction
# =============================================================================


def _weight_dtype(dim: int):
    """The Wormhole suite's per-model weight dtype, carried unchanged."""

    return ttnn.bfloat16 if dim == 5120 else ttnn.bfloat8_b


#: The decode activation height every node id in this file uses. Derived, not
#: named: `tile_padded_batch_rows` pads to a tile, so both the 24-user
#: padded-batch node and the 32-user production node resolve to 32. Naming
#: 32 here would hide that the padding is what makes them agree.
_DECODE_ROWS = tile_padded_batch_rows(GALAXY_PHYSICAL_BATCH)


def _dram_interleaved_mlp(
    mesh_device: ttnn.MeshDevice, resources, weights, *, dim: int, slices: _MeshSlices, topology
) -> MLP2D:
    """Build MLP2D with every weight DRAM interleaved and every matmul unconfigured.

    `_dram_interleaved_weight_lazies` is `_mlp_2d_galaxy.prefill_weight_lazies`
    under a name that says what it is here: it already produces exactly the
    layout this recipe wants -- the 2D `(-1, -2)` / `(-2, -1)` mesh mappers with
    `ttnn.DRAM_MEMORY_CONFIG` -- so no new geometry is derived in this file.

    `w1_w3_memcfg` and `w2_memcfg` are the two fields that select a ring layout on
    Wormhole; pinning both to `ttnn.DRAM_MEMORY_CONFIG` is what makes this the
    interleaved recipe. The four decode placements have to be named too: their
    resolved defaults are `L1_MEMORY_CONFIG` and `L1_WIDTH_SHARDED_MEMORY_CONFIG`,
    which is not what this recipe is.

    **The decode program configs are NOT `None`, and that is a correction.** They
    were, on the reasoning that `None` *is* the value the unfused decode matmuls
    are given -- and on 2026-09-17 every node id in this file failed because of
    it:

        TT_FATAL ... MatmulMultiCoreReuseMultiCast1DProgramConfig: matmul
        grid_size 12-10 anchored at sub-device start 1-0 extends past the
        sub-device's worker bounding box

    With no program config ttnn auto-sizes the grid from the DEVICE's compute
    grid, 12 x 10 here, and anchors it at the loaded SUB-DEVICE's start (1, 0),
    so `1 + 12 - 1 = 12` overruns the envelope's last worker column, 10. The
    Wormhole envelope's bounding box is one column narrower than its compute
    grid; Blackhole's is two, which is why `None` never bit there.

    `dense_matmul_program_config` is the repo's answer to exactly this and says
    so -- its docstring records the Wormhole instance of the same overrun.
    It takes a topology, so it returns Blackhole's ten-column rectangle and
    sets `allowed_worker_cores`.

    The prefill path keeps `_default_prefill_program_config`, the module's own
    policy object returning `None` for any sequence length, passed explicitly
    rather than defaulted so the choice is stated. Prefill takes the same
    unconfigured route and is expected to hit the same wall; it is measured
    separately rather than fixed blind.

    `decode_fused_ccl=False` is the point of the whole file; see the module
    docstring. It also makes `collective_resource_selector` mandatory, which
    `_resolve_mlp2d_config` enforces.
    """

    w1, w2, w3 = weights
    weight_dtype = _weight_dtype(dim)
    lazy_w1, lazy_w2, lazy_w3 = _dram_interleaved_weight_lazies(w1, w2, w3, mesh_device, weight_dtype)
    return MLP2D.from_config(
        MLP2DConfig(
            w1=lazy_w1,
            w2=lazy_w2,
            w3=lazy_w3,
            prefill_w1=lazy_w1,
            prefill_w2=lazy_w2,
            prefill_w3=lazy_w3,
            mesh_device=mesh_device,
            tt_ccl=resources.ccl,
            collective_resource_selector=exact_tensor_resource,
            decode_fused_ccl=False,
            w1_w3_memcfg=ttnn.DRAM_MEMORY_CONFIG,
            w2_memcfg=ttnn.DRAM_MEMORY_CONFIG,
            decode_input_memcfg=ttnn.DRAM_MEMORY_CONFIG,
            decode_w2_input_memcfg=ttnn.DRAM_MEMORY_CONFIG,
            decode_w1_w3_output_memcfg=ttnn.DRAM_MEMORY_CONFIG,
            decode_w2_output_memcfg=ttnn.DRAM_MEMORY_CONFIG,
            prefill_input_memcfg=ttnn.DRAM_MEMORY_CONFIG,
            prefill_w1_w3_output_memcfg=ttnn.DRAM_MEMORY_CONFIG,
            prefill_w2_output_memcfg=ttnn.DRAM_MEMORY_CONFIG,
            decode_w1_w3_prg_config=dense_matmul_program_config(
                _DECODE_ROWS, slices.local_k, slices.local_hidden, topology
            ),
            decode_w2_prg_config=dense_matmul_program_config(
                _DECODE_ROWS, slices.local_hidden, slices.local_dim, topology
            ),
            prefill_w1_w3_prg_config=_default_prefill_program_config,
            prefill_w2_prg_config=_default_prefill_program_config,
            ff1_out_reduce_scatter_memcfg=ttnn.DRAM_MEMORY_CONFIG,
            # **The one placement in this config that is not DRAM**, and the
            # reason is the op rather than a preference. Stage 6's
            # `all_reduce_async` validates `WIDTH_SHARDED` on its input, its
            # caller-owned buffer and this output config, and refuses a DRAM
            # input outright on Blackhole -- so the DRAM-interleaved premise that
            # holds for every matmul here cannot hold for the final collective.
            # Measured 2026-09-17: the matmul nodes passed at PCC 0.9999 and both
            # end-to-end decode nodes aborted here.
            #
            # `decode_w2_output_memcfg` deliberately stays DRAM, so the w2 matmul
            # is byte-identical to the one already measured green; `_all_reduce_tg`
            # now stages an unsharded input into this placement, the way
            # `collectives.py::_all_reduce` always has.
            ff2_out_reduce_scatter_memcfg=width_sharded_memory_config(
                slices.local_dim, _all_reduce_output_cores(topology, slices.local_dim)
            ),
            sharded_attn_input_memcfg=ttnn.DRAM_MEMORY_CONFIG,
            decode_prefetch_context=resources.prefetch_context("decode"),
            prefill_prefetch_context=resources.prefetch_context("prefill"),
            activation_dtype=ACTIVATION_DTYPE,
            ccl_dtype=ACTIVATION_DTYPE,
            mul_dtype=ACTIVATION_DTYPE,
            w1_w3_dtype=weight_dtype,
            w2_dtype=weight_dtype,
        )
    )


def _activation_mesh_mapper() -> ttnn.MeshMapperConfig:
    """The activation placement `_load_input_device_tensor` applies, spelled out.

    Replicated over the 8 mesh rows and sharded on the last dim over the 4 mesh
    columns, because W1/W3 shard `K` on mesh axis 1. Copied from
    `mlp_2d._load_input_device_tensor` rather than re-derived; the single-matmul
    probe below needs it without constructing a module.
    """

    return ttnn.MeshMapperConfig(
        placements=[ttnn.PlacementReplicate(), ttnn.PlacementShard(-1)],
        mesh_shape_override=ttnn.MeshShape(*GALAXY_MESH_SHAPE),
    )


def _lazy_activation(x: torch.Tensor, mesh_device: ttnn.MeshDevice, dtype) -> LazyWeight:
    return LazyWeight(
        source=x,
        device=mesh_device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper_config=_activation_mesh_mapper(),
    )


def _decode_activation(users: int, dim: int) -> torch.Tensor:
    """A decode activation of `tile_padded_batch_rows(users)` rows.

    Rows at and above `users` are the tile padding and are left at zero, so the
    reference for them is exactly zero and `_report_per_user` can bound them.
    """

    rows = tile_padded_batch_rows(users)
    x = torch.zeros(1, 1, rows, dim, dtype=torch.bfloat16)
    x[..., :users, :] = torch.randn(1, 1, users, dim, dtype=torch.bfloat16)
    return x


# =============================================================================
# Node 1: the single-matmul signal, no collective at all
# =============================================================================


@pytest.mark.device
@pytest.mark.blackhole
@pytest.mark.galaxy_bh
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
@pytest.mark.parametrize(
    "dim,hidden_dim",
    [(8192, 28672), (5120, 25600)],
    ids=["llama-8192x28672", "qwen-5120x25600"],
)
@pytest.mark.parametrize("device_params", [BHGLX_DEVICE_PARAMS], indirect=True)
@torch.no_grad()
def test_mlp_2d_bh_galaxy_w1_matmul_dram_interleaved(mesh_device, dim, hidden_dim):
    """Does one DRAM-interleaved W1 matmul land the right data on all 32 devices?

    The cheapest node here by a wide margin, and the only one that issues no
    collective, so it cannot hang: if `num_links` or the fabric pairing is wrong
    this still completes and a later node is the one that stalls. It materializes
    one weight rather than three, plans no collective at all -- that is what
    `bh_galaxy_worker_sub_device` is for -- and answers the question that
    matters, whether the interleaved matmul is numerically right, at the earliest
    point where a per-device channel-order mismatch could be introduced.

    Device `(r, c)` should hold `x[..., c] @ w1[c, r]`: the activation shard for
    mesh column `c` against the W1 block whose `K` is column `c`'s slice and whose
    `N` is mesh row `r`'s slice. That is a partial sum over `K`, uncorrelated with
    any other column's partial, so a column permutation is unambiguous.
    """

    torch.manual_seed(0)
    topology = _require_dram_interleaved_capabilities(mesh_device)
    reference = _reference_mlp(dim, hidden_dim)
    w1, w2, w3 = get_mlp_weights_from_ref_model(reference)
    x = torch.randn(1, 1, PROBE_SEQUENCE, dim, dtype=torch.bfloat16)
    slices = _MeshSlices(dim, hidden_dim)
    activation = x.to(torch.float32)
    w1_float = w1.to(torch.float32)

    # Materialized once for all 32 devices: each entry is a CPU matmul, and the
    # three loops below would otherwise recompute every one of them three times.
    expected_partials = [
        [slices.projection_partial(activation, w1_float, row, column) for column in range(GALAXY_COLUMNS)]
        for row in range(GALAXY_ROWS)
    ]

    lazy_w1, _, _ = _dram_interleaved_weight_lazies(w1, w2, w3, mesh_device, _weight_dtype(dim))
    lazy_x = _lazy_activation(x, mesh_device, ACTIVATION_DTYPE)
    device_weight = None
    device_input = None
    projection = None
    # No resource owner and no collective plan: the sub-device is loaded only so
    # the matmul can be told which one it runs on, rather than auto-gridding over
    # a column the descriptor reserves for a reason whose failures are silent.
    with bh_galaxy_worker_sub_device(mesh_device, topology=topology) as (_, worker_sub_device_id):
        try:
            device_weight = lazy_w1.get_device_weight()
            device_input = lazy_x.get_device_weight()
            projection = ttnn.linear(
                device_input,
                device_weight,
                dtype=ACTIVATION_DTYPE,
                core_grid=None,
                compute_kernel_config=_compute_kernel_config_hifi2_fp16(),
                # NOT `program_config=None`. Measured 2026-09-17: with no
                # program config ttnn auto-sizes the matmul grid from the
                # DEVICE's compute grid -- 12 x 10 on Blackhole -- and anchors
                # it at the loaded SUB-DEVICE's start, (1, 0). `1 + 12 - 1 = 12`
                # exceeds the envelope's last worker column, 10:
                #   TT_FATAL ... MatmulMultiCoreReuseMultiCast1DProgramConfig:
                #   matmul grid_size 12-10 anchored at sub-device start 1-0
                #   extends past the sub-device's worker bounding box
                # `dense_matmul_program_config` exists for exactly this and
                # documents the Wormhole instance of it; it
                # takes a topology, so it returns Blackhole's ten-column
                # rectangle and sets `allowed_worker_cores`. The Wormhole
                # envelope's bbox is one column narrower than its compute grid,
                # which is why `None` never bit there; here it is two.
                program_config=dense_matmul_program_config(
                    PROBE_SEQUENCE, slices.local_k, slices.local_hidden, topology
                ),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                global_cb=None,
                sub_device_id=worker_sub_device_id,
            )
            ttnn.synchronize_device(mesh_device, sub_device_ids=[worker_sub_device_id])
            grid = _compose_mesh_grid(projection, mesh_device)

            rows = [grid[:, row : row + 1] for row in range(GALAXY_ROWS)]
            measured = [_column_slices(rows[row], GALAXY_COLUMNS) for row in range(GALAXY_ROWS)]
            # Per mesh column, inside one mesh row: the four blocks are four
            # different partial sums, so this is where an activation-shard mix-up
            # appears.
            for row in range(GALAXY_ROWS):
                _report_permutation(
                    f"w1 matmul mesh row {row} across mesh columns", expected_partials[row], measured[row]
                )
            # Per mesh row, at one mesh column: the eight blocks are eight
            # different W1 output slices, so this is where a weight-shard mix-up
            # appears.
            for column in range(GALAXY_COLUMNS):
                _report_permutation(
                    f"w1 matmul mesh column {column} across mesh rows",
                    [expected_partials[row][column] for row in range(GALAXY_ROWS)],
                    [measured[row][column] for row in range(GALAXY_ROWS)],
                )
            expected_grid = torch.cat([torch.cat(row, dim=-1) for row in expected_partials], dim=1)
            _log_stage_pcc("w1 matmul, all 32 devices", expected_grid, grid)
            record_module_output(f"mlp_2d_bh_w1_matmul_{dim}x{hidden_dim}", grid)
        finally:
            deallocate_tensor(projection)
            deallocate_tensor(device_input)
            deallocate_tensor(device_weight)


# =============================================================================
# Node 2: the whole unfused decode pipeline, one stage at a time
# =============================================================================


@pytest.mark.device
@pytest.mark.blackhole
@pytest.mark.galaxy_bh
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
@pytest.mark.parametrize(
    "dim,hidden_dim",
    [(8192, 28672), (5120, 25600)],
    ids=["llama-8192x28672", "qwen-5120x25600"],
)
@pytest.mark.parametrize("device_params", [BHGLX_DEVICE_PARAMS], indirect=True)
@torch.no_grad()
def test_mlp_2d_bh_galaxy_staged_decode_padded_batch(mesh_device, dim, hidden_dim):
    """Run the unfused decode pipeline stage by stage, printing each stage's PCC.

    **This is the predicted failure, with the instrumentation that can localize
    it.** Every call below is one of `decode_forward`'s own calls
    on the `decode_fused_ccl=False` branch, in its own order, with the module's
    own resolved config and its own collective helpers -- `_unfused_decode_w1_w3`
    for stages 1-2 and `_reduce_scatter_axis1` / `_all_gather_axis1` /
    `_all_reduce_tg` for the rest. Nothing is reimplemented and no TTNN spelling
    is invented; the only difference from calling `forward` is that each
    intermediate is composed and correlated before the next stage consumes it.

    The all-gather stage is the one predicted to break: its output is the
    tensor fed to the interleaved W2 matmul, and the order of its four gathered
    chunks is exactly the per-device channel order the ring recipe got wrong.

    The activation is `tile_padded_batch_rows(24) == 32` rows carrying 24 users,
    so the tile padding is real rather than notional and a decode height derived
    from the user count would be caught here. At the production batch of 32 the
    two numbers coincide and the distinction is untestable, which is what node 3
    runs instead.
    """

    torch.manual_seed(1)
    topology = _require_dram_interleaved_capabilities(mesh_device)
    reference = _reference_mlp(dim, hidden_dim)
    w1, w2, w3 = get_mlp_weights_from_ref_model(reference)
    users = PADDED_BATCH_USERS
    rows = tile_padded_batch_rows(users)
    assert rows > users, f"the padded-batch node needs tile padding; {users} users gave {rows} rows"
    x = _decode_activation(users, dim)
    expected_output = reference(x)
    w1_full, w3_full, gated_full, _ = _reference_stages(x, w1, w2, w3)
    slices = _MeshSlices(dim, hidden_dim)
    activation = x.to(torch.float32)
    w1_float, w2_float, w3_float = w1.to(torch.float32), w2.to(torch.float32), w3.to(torch.float32)

    resources = _resources(
        mesh_device, topology, dim, hidden_dim, decode_rows=rows, prefill_sequences=(PROBE_SEQUENCE,)
    )
    module = None
    stage_tensors: list[ttnn.Tensor] = []
    try:
        module = _dram_interleaved_mlp(mesh_device, resources, (w1, w2, w3), dim=dim, slices=slices, topology=topology)
        config = module.config
        module.load_device_weights("decode")
        prefetch_kwargs = _prefetch_kwargs(config.decode_prefetch_context)
        resources.activate("decode")

        device_input = _load_input_device_tensor(
            _lazy_activation(x, mesh_device, config.decode_activation_dtype), config, "decode"
        )
        stage_tensors.append(device_input)

        # --- STAGE 1: W1 and W3, DRAM interleaved, program_config=None ---
        projections = []
        for weight in (module.w1, module.w3):
            projections.append(
                ttnn.linear(
                    device_input,
                    weight,
                    dtype=config.decode_activation_dtype,
                    core_grid=None,
                    compute_kernel_config=config.decode_ff1_3_compute_kernel_cfg,
                    program_config=config.decode_w1_w3_prg_config,
                    memory_config=config.decode_w1_w3_output_memcfg,
                    **prefetch_kwargs,
                )
            )
        stage_tensors.extend(projections)
        resources.synchronize("decode")
        for name, projection, weight in (("w1", projections[0], w1_float), ("w3", projections[1], w3_float)):
            grid = _compose_mesh_grid(projection, mesh_device)
            for row in range(GALAXY_ROWS):
                _report_permutation(
                    f"decode {name} projection mesh row {row} across mesh columns",
                    [slices.projection_partial(activation, weight, row, column) for column in range(GALAXY_COLUMNS)],
                    _column_slices(grid[:, row : row + 1], GALAXY_COLUMNS),
                )

        # --- STAGE 2: the axis-1 reduce-scatter of each projection ---
        # Two distinct resources, keyed "w1" and "w3": both outputs are live in
        # stage 3, so a shared persistent buffer would silently read W3 twice.
        reduced = []
        for projection, key in zip(projections, ("w1", "w3")):
            reduced.append(
                module._reduce_scatter_axis1(projection, config.ff1_out_reduce_scatter_memcfg, "decode", key)
            )
        stage_tensors.extend(reduced)
        resources.synchronize("decode")
        for name, tensor, expected_source in (("w1", reduced[0], w1_full), ("w3", reduced[1], w3_full)):
            grid = _compose_mesh_grid(tensor, mesh_device)
            for row in range(GALAXY_ROWS):
                _report_permutation(
                    f"decode {name} reduce-scatter mesh row {row} across mesh columns",
                    [slices.scattered_chunk(expected_source, row, chunk) for chunk in range(GALAXY_COLUMNS)],
                    _column_slices(grid[:, row : row + 1], GALAXY_COLUMNS),
                )
            _log_stage_pcc(
                f"decode {name} reduce-scatter, all 32 devices",
                torch.cat([slices.row_hidden(expected_source, row) for row in range(GALAXY_ROWS)], dim=1),
                grid,
            )

        # --- STAGE 3: Activation + Multiply ---
        gated = ttnn.mul(
            reduced[0],
            reduced[1],
            input_tensor_a_activations=[config.mlp_activation_type],
            dtype=config.decode_mul_dtype,
            memory_config=config.ff1_out_reduce_scatter_memcfg,
        )
        stage_tensors.append(gated)
        resources.synchronize("decode")
        _log_stage_pcc(
            "decode silu(w1) * w3, all 32 devices",
            torch.cat([slices.row_hidden(gated_full, row) for row in range(GALAXY_ROWS)], dim=1),
            _compose_mesh_grid(gated, mesh_device),
        )

        # --- STAGE 4: All-gather before W2 ---
        # The stage predicted to break. Each device in mesh row `r` should
        # hold the whole of that row's hidden slice with the four gathered chunks
        # in mesh-column order, so correlating chunk `j` against every reference
        # chunk is what separates a reordering from a numerics fault.
        w2_input = module._all_gather_axis1(gated, config.decode_w2_input_memcfg, "decode")
        stage_tensors.append(w2_input)
        resources.synchronize("decode")
        gathered_grid = _compose_mesh_grid(w2_input, mesh_device)
        for row in range(GALAXY_ROWS):
            for column, replica in enumerate(_column_slices(gathered_grid[:, row : row + 1], GALAXY_COLUMNS)):
                _report_permutation(
                    f"decode all-gather mesh row {row} device column {column}, gathered chunk order",
                    [slices.scattered_chunk(gated_full, row, chunk) for chunk in range(GALAXY_COLUMNS)],
                    _column_slices(replica, GALAXY_COLUMNS),
                )

        # --- STAGE 5: W2, DRAM interleaved, program_config=None ---
        w2_projection = ttnn.linear(
            w2_input,
            module.w2,
            compute_kernel_config=config.decode_ff2_compute_kernel_cfg,
            dtype=config.decode_ccl_dtype,
            program_config=config.decode_w2_prg_config,
            memory_config=config.decode_w2_output_memcfg,
            core_grid=None,
            **prefetch_kwargs,
        )
        stage_tensors.append(w2_projection)
        resources.synchronize("decode")
        w2_grid = _compose_mesh_grid(w2_projection, mesh_device)
        for row in range(GALAXY_ROWS):
            _report_permutation(
                f"decode w2 projection mesh row {row} across mesh columns",
                [slices.w2_partial(gated_full, w2_float, row, column) for column in range(GALAXY_COLUMNS)],
                _column_slices(w2_grid[:, row : row + 1], GALAXY_COLUMNS),
            )

        # --- STAGE 6: the axis-0 all-reduce ---
        reduced_output = module._all_reduce_tg(
            w2_projection,
            cluster_axis=0,
            dim=3,
            sharded=True,
            memory_config=config.ff2_out_reduce_scatter_memcfg,
            ccl_dtype=config.decode_ccl_dtype,
            mode="decode",
        )
        stage_tensors.append(reduced_output)
        resources.synchronize("decode")
        output_grid = _compose_mesh_grid(reduced_output, mesh_device)
        expected_float = expected_output.to(torch.float32)
        # Every mesh row now holds the same reduced result, so there is no
        # ordering to detect across rows -- only whether the reduction reached
        # each of them.
        _report_correlations(
            "decode output per mesh row", expected_float, [output_grid[:, row : row + 1] for row in range(GALAXY_ROWS)]
        )
        _report_permutation(
            "decode output mesh row 0 across mesh columns",
            [slices.output_column(expected_float, column) for column in range(GALAXY_COLUMNS)],
            _column_slices(output_grid[:, 0:1], GALAXY_COLUMNS),
        )
        actual = compose_2d_sharded_tensor(reduced_output, mesh_device)
        _report_per_user("decode output", expected_float, actual.to(torch.float32), users=users)
        _log_stage_pcc(
            f"staged decode {users} users in {rows} rows, real users only",
            expected_output[..., :users, :],
            actual[..., :users, :],
        )
        record_module_output(f"mlp_2d_bh_staged_decode_u{users}_{dim}x{hidden_dim}", actual)
    finally:
        try:
            resources.cleanup()
        finally:
            for tensor in reversed(stage_tensors):
                # Per-tensor, because a module helper may already have released a
                # tensor it rewrapped, and one failed free in teardown must not
                # mask the assertion that brought us here.
                try:
                    deallocate_tensor(tensor)
                except Exception as error:  # noqa: BLE001 - teardown must not mask the real failure
                    logger.warning(f"ignoring deallocation failure during teardown: {error}")
            deallocate_module_weights(module, "w1", "w2", "w3", "prefill_w1", "prefill_w2", "prefill_w3")


# =============================================================================
# Node 3: the module's own unfused decode forward, production batch, twice
# =============================================================================


@pytest.mark.device
@pytest.mark.blackhole
@pytest.mark.galaxy_bh
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
@pytest.mark.parametrize(
    "dim,hidden_dim",
    [(8192, 28672), (5120, 25600)],
    ids=["llama-8192x28672", "qwen-5120x25600"],
)
@pytest.mark.parametrize("device_params", [BHGLX_DEVICE_PARAMS], indirect=True)
@torch.no_grad()
def test_mlp_2d_bh_galaxy_decode_batch_32_repeat(mesh_device, dim, hidden_dim):
    """`MLP2D.forward(mode="decode")` unfused, end to end, at the production batch.

    The module-level claim that node 2 decomposes, and the direct answer to the
    hypothesis: *MLP2D decode on Blackhole, DRAM-interleaved with
    `program_config=None`, holds PCC >= 0.99 against a CPU reference.* Run twice,
    because the second invocation is the one that hits the TTNN program cache and
    a program-cache-dependent fault is invisible on a single pass -- which is also
    what the Wormhole decode node does.

    Per-user, per-mesh-row and per-mesh-column correlation is reported for every
    invocation, not only the aggregate PCC, because a column-local fault gives an
    aggregate in 0-0.01 and nothing about which column.
    """

    torch.manual_seed(2)
    topology = _require_dram_interleaved_capabilities(mesh_device)
    reference = _reference_mlp(dim, hidden_dim)
    w1, w2, w3 = get_mlp_weights_from_ref_model(reference)
    users = GALAXY_PHYSICAL_BATCH
    rows = tile_padded_batch_rows(users)
    x = _decode_activation(users, dim)
    expected = reference(x)
    expected_float = expected.to(torch.float32)
    slices = _MeshSlices(dim, hidden_dim)

    resources = _resources(
        mesh_device, topology, dim, hidden_dim, decode_rows=rows, prefill_sequences=(PROBE_SEQUENCE,)
    )
    module = None
    recorded: list[torch.Tensor] = []
    try:
        module = _dram_interleaved_mlp(mesh_device, resources, (w1, w2, w3), dim=dim, slices=slices, topology=topology)
        config = module.config
        for invocation in range(2):
            device_input = _load_input_device_tensor(
                _lazy_activation(x, mesh_device, config.decode_activation_dtype), config, "decode"
            )
            resources.activate("decode")
            output = module(device_input, mode="decode")
            try:
                resources.synchronize("decode")
                case = f"decode batch {users} in {rows} rows, invocation {invocation}"
                grid = _compose_mesh_grid(output, mesh_device)
                _report_correlations(
                    f"{case}: per mesh row", expected_float, [grid[:, row : row + 1] for row in range(GALAXY_ROWS)]
                )
                _report_permutation(
                    f"{case}: mesh row 0 across mesh columns",
                    [slices.output_column(expected_float, column) for column in range(GALAXY_COLUMNS)],
                    _column_slices(grid[:, 0:1], GALAXY_COLUMNS),
                )
                actual = compose_2d_sharded_tensor(output, mesh_device)
                _report_per_user(case, expected_float, actual.to(torch.float32), users=users)
                _log_stage_pcc(case, expected, actual)
                if invocation == 1:
                    recorded.append(actual)
            finally:
                deallocate_tensor(output)
                deallocate_tensor(device_input)
        record_module_output(f"mlp_2d_bh_decode_batch{users}_{dim}x{hidden_dim}", *recorded)
    finally:
        try:
            resources.cleanup()
        finally:
            deallocate_module_weights(module, "w1", "w2", "w3", "prefill_w1", "prefill_w2", "prefill_w3")


# =============================================================================
# Node 4: the whole prefill pipeline, one stage at a time
# =============================================================================


@pytest.mark.device
@pytest.mark.blackhole
@pytest.mark.galaxy_bh
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
@pytest.mark.parametrize(
    "dim,hidden_dim",
    [(8192, 28672), (5120, 25600)],
    ids=["llama-8192x28672", "qwen-5120x25600"],
)
@pytest.mark.parametrize("device_params", [BHGLX_DEVICE_PARAMS], indirect=True)
@torch.no_grad()
def test_mlp_2d_bh_galaxy_staged_prefill_128(mesh_device, dim, hidden_dim):
    """The same staged treatment for prefill, which needed no source change.

    `prefill_forward` already used plain `ttnn.linear` plus the standard
    collectives, and `_resolve_mlp2d_config` already materialized
    `prefill_w1/w2/w3` at `ttnn.DRAM_MEMORY_CONFIG` unconditionally, so this node
    measures the interleaved recipe on a path that was always expressible. It
    runs after both decode nodes because 128 rows cost four times 32.
    """

    torch.manual_seed(3)
    topology = _require_dram_interleaved_capabilities(mesh_device)
    reference = _reference_mlp(dim, hidden_dim)
    w1, w2, w3 = get_mlp_weights_from_ref_model(reference)
    x = torch.randn(1, 1, PROBE_SEQUENCE, dim, dtype=torch.bfloat16)
    expected_output = reference(x)
    w1_full, w3_full, gated_full, _ = _reference_stages(x, w1, w2, w3)
    slices = _MeshSlices(dim, hidden_dim)
    activation = x.to(torch.float32)
    w1_float, w2_float, w3_float = w1.to(torch.float32), w2.to(torch.float32), w3.to(torch.float32)

    resources = _resources(
        mesh_device,
        topology,
        dim,
        hidden_dim,
        decode_rows=tile_padded_batch_rows(GALAXY_PHYSICAL_BATCH),
        prefill_sequences=(PROBE_SEQUENCE,),
    )
    module = None
    stage_tensors: list[ttnn.Tensor] = []
    try:
        module = _dram_interleaved_mlp(mesh_device, resources, (w1, w2, w3), dim=dim, slices=slices, topology=topology)
        config = module.config
        module.load_device_weights("prefill")
        prefetch_kwargs = _prefetch_kwargs(config.prefill_prefetch_context)
        resources.activate("prefill")

        device_input = _load_input_device_tensor(
            _lazy_activation(x, mesh_device, config.prefill_activation_dtype), config, "prefill"
        )
        stage_tensors.append(device_input)

        # --- STAGE 1: W1 and W3, DRAM interleaved, program_config=None ---
        projections = []
        for weight in (module.prefill_w1, module.prefill_w3):
            projections.append(
                ttnn.linear(
                    device_input,
                    weight,
                    dtype=config.prefill_activation_dtype,
                    core_grid=None,
                    compute_kernel_config=config.prefill_ff1_3_compute_kernel_cfg,
                    program_config=config.prefill_w1_w3_prg_config(PROBE_SEQUENCE),
                    memory_config=config.prefill_w1_w3_output_memcfg,
                    **prefetch_kwargs,
                )
            )
        stage_tensors.extend(projections)
        input_memory_config = projections[0].memory_config()
        resources.synchronize("prefill")
        for name, projection, weight in (("w1", projections[0], w1_float), ("w3", projections[1], w3_float)):
            grid = _compose_mesh_grid(projection, mesh_device)
            for row in range(GALAXY_ROWS):
                _report_permutation(
                    f"prefill {name} projection mesh row {row} across mesh columns",
                    [slices.projection_partial(activation, weight, row, column) for column in range(GALAXY_COLUMNS)],
                    _column_slices(grid[:, row : row + 1], GALAXY_COLUMNS),
                )

        # --- STAGE 2: the axis-1 reduce-scatter of each projection ---
        reduced = [
            module._reduce_scatter_axis1(projection, None, "prefill", key)
            for projection, key in zip(projections, ("w1", "w3"))
        ]
        stage_tensors.extend(reduced)
        resources.synchronize("prefill")
        for name, tensor, expected_source in (("w1", reduced[0], w1_full), ("w3", reduced[1], w3_full)):
            grid = _compose_mesh_grid(tensor, mesh_device)
            for row in range(GALAXY_ROWS):
                _report_permutation(
                    f"prefill {name} reduce-scatter mesh row {row} across mesh columns",
                    [slices.scattered_chunk(expected_source, row, chunk) for chunk in range(GALAXY_COLUMNS)],
                    _column_slices(grid[:, row : row + 1], GALAXY_COLUMNS),
                )
            _log_stage_pcc(
                f"prefill {name} reduce-scatter, all 32 devices",
                torch.cat([slices.row_hidden(expected_source, row) for row in range(GALAXY_ROWS)], dim=1),
                grid,
            )

        # --- STAGE 3: Activation + Multiply ---
        gated = ttnn.mul(
            reduced[0],
            reduced[1],
            input_tensor_a_activations=[config.mlp_activation_type],
            dtype=config.prefill_mul_dtype,
            memory_config=reduced[0].memory_config(),
        )
        stage_tensors.append(gated)
        resources.synchronize("prefill")
        _log_stage_pcc(
            "prefill silu(w1) * w3, all 32 devices",
            torch.cat([slices.row_hidden(gated_full, row) for row in range(GALAXY_ROWS)], dim=1),
            _compose_mesh_grid(gated, mesh_device),
        )

        # --- STAGE 4: All-gather before W2 ---
        w2_input = module._all_gather_axis1(gated, input_memory_config, "prefill", "gated")
        stage_tensors.append(w2_input)
        resources.synchronize("prefill")
        gathered_grid = _compose_mesh_grid(w2_input, mesh_device)
        for row in range(GALAXY_ROWS):
            for column, replica in enumerate(_column_slices(gathered_grid[:, row : row + 1], GALAXY_COLUMNS)):
                _report_permutation(
                    f"prefill all-gather mesh row {row} device column {column}, gathered chunk order",
                    [slices.scattered_chunk(gated_full, row, chunk) for chunk in range(GALAXY_COLUMNS)],
                    _column_slices(replica, GALAXY_COLUMNS),
                )

        # --- STAGE 5: W2, DRAM interleaved, program_config=None ---
        w2_projection = ttnn.linear(
            w2_input,
            module.prefill_w2,
            compute_kernel_config=config.prefill_ff2_compute_kernel_cfg,
            dtype=config.prefill_ccl_dtype,
            program_config=config.prefill_w2_prg_config(PROBE_SEQUENCE),
            memory_config=config.prefill_w2_output_memcfg,
            core_grid=None,
            **prefetch_kwargs,
        )
        stage_tensors.append(w2_projection)
        resources.synchronize("prefill")
        w2_grid = _compose_mesh_grid(w2_projection, mesh_device)
        for row in range(GALAXY_ROWS):
            _report_permutation(
                f"prefill w2 projection mesh row {row} across mesh columns",
                [slices.w2_partial(gated_full, w2_float, row, column) for column in range(GALAXY_COLUMNS)],
                _column_slices(w2_grid[:, row : row + 1], GALAXY_COLUMNS),
            )

        # --- STAGE 6: the axis-0 all-reduce, a reduce-scatter then an all-gather ---
        reduced_output = module._all_reduce_tg(
            w2_projection,
            cluster_axis=0,
            dim=3,
            sharded=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            ccl_dtype=config.prefill_ccl_dtype,
            mode="prefill",
        )
        stage_tensors.append(reduced_output)
        resources.synchronize("prefill")
        output_grid = _compose_mesh_grid(reduced_output, mesh_device)
        expected_float = expected_output.to(torch.float32)
        _report_correlations(
            "prefill output per mesh row",
            expected_float,
            [output_grid[:, row : row + 1] for row in range(GALAXY_ROWS)],
        )
        _report_permutation(
            "prefill output mesh row 0 across mesh columns",
            [slices.output_column(expected_float, column) for column in range(GALAXY_COLUMNS)],
            _column_slices(output_grid[:, 0:1], GALAXY_COLUMNS),
        )
        actual = compose_2d_sharded_tensor(reduced_output, mesh_device)
        _log_stage_pcc("staged prefill 128 final output", expected_output, actual)
        record_module_output(f"mlp_2d_bh_staged_prefill128_{dim}x{hidden_dim}", actual)
    finally:
        try:
            resources.cleanup()
        finally:
            for tensor in reversed(stage_tensors):
                try:
                    deallocate_tensor(tensor)
                except Exception as error:  # noqa: BLE001 - teardown must not mask the real failure
                    logger.warning(f"ignoring deallocation failure during teardown: {error}")
            deallocate_module_weights(module, "prefill_w1", "prefill_w2", "prefill_w3", "w1", "w2", "w3")


# =============================================================================
# Node 5: the module's own prefill forward, both sequence lengths, twice
# =============================================================================


@pytest.mark.device
@pytest.mark.blackhole
@pytest.mark.galaxy_bh
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
@pytest.mark.parametrize(
    "dim,hidden_dim",
    [(8192, 28672), (5120, 25600)],
    ids=["llama-8192x28672", "qwen-5120x25600"],
)
@pytest.mark.parametrize("device_params", [BHGLX_DEVICE_PARAMS], indirect=True)
@torch.no_grad()
def test_mlp_2d_bh_galaxy_prefill_128_then_2048_repeat(mesh_device, dim, hidden_dim):
    """`MLP2D.forward(mode="prefill")` end to end, against the HF CPU reference.

    The heaviest node in the file, and last for that reason. 2048 is above
    `prefill_len_cutoff`, so it exercises the `[1, seq // 1024, 1024, -1]` reshape
    and a second set of keyed collective resources; the pair is run twice for the
    program cache.
    """

    torch.manual_seed(4)
    topology = _require_dram_interleaved_capabilities(mesh_device)
    reference = _reference_mlp(dim, hidden_dim)
    w1, w2, w3 = get_mlp_weights_from_ref_model(reference)
    sequences = (128, 2048)
    slices = _MeshSlices(dim, hidden_dim)

    resources = _resources(
        mesh_device,
        topology,
        dim,
        hidden_dim,
        decode_rows=tile_padded_batch_rows(GALAXY_PHYSICAL_BATCH),
        prefill_sequences=sequences,
    )
    module = None
    recorded: list[torch.Tensor] = []
    try:
        module = _dram_interleaved_mlp(mesh_device, resources, (w1, w2, w3), dim=dim, slices=slices, topology=topology)
        config = module.config
        for invocation in range(2):
            for sequence in sequences:
                x = torch.randn(1, 1, sequence, dim, dtype=torch.bfloat16)
                expected = reference(x)
                expected_float = expected.to(torch.float32)
                device_input = _load_input_device_tensor(
                    _lazy_activation(x, mesh_device, config.prefill_activation_dtype), config, "prefill"
                )
                resources.activate("prefill")
                output = module(device_input, mode="prefill")
                try:
                    resources.synchronize("prefill")
                    case = f"prefill {sequence} invocation {invocation}"
                    grid = _compose_mesh_grid(output, mesh_device)
                    _report_correlations(
                        f"{case}: per mesh row",
                        expected_float,
                        [grid[:, row : row + 1] for row in range(GALAXY_ROWS)],
                    )
                    _report_permutation(
                        f"{case}: mesh row 0 across mesh columns",
                        [slices.output_column(expected_float, column) for column in range(GALAXY_COLUMNS)],
                        _column_slices(grid[:, 0:1], GALAXY_COLUMNS),
                    )
                    actual = compose_2d_sharded_tensor(output, mesh_device)
                    _log_stage_pcc(case, expected, actual)
                    if invocation == 1:
                        recorded.append(actual)
                finally:
                    deallocate_tensor(output)
                    deallocate_tensor(device_input)
        record_module_output(f"mlp_2d_bh_prefill_128_2048_{dim}x{hidden_dim}", *recorded)
    finally:
        try:
            resources.cleanup()
        finally:
            deallocate_module_weights(module, "prefill_w1", "prefill_w2", "prefill_w3", "w1", "w2", "w3")
