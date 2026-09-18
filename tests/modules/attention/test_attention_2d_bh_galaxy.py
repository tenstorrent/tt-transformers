# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Real-hardware bring-up for the common Attention2D on a Blackhole Galaxy.

The Blackhole analogue of `test_attention_2d_wh_galaxy.py`, and deliberately a
different *recipe* rather than the same recipe with different coordinates. Six
Blackhole facts shape it, and each one is named where it is used:

1. **The worker envelope is not the compute grid, and the two modes load
   different partitions.** Blackhole Galaxy's compute grid is `12 x 10` and
   dispatch is column 11, *inside* it, so the descriptor's worker envelope is
   `x=1..10, y=0..9`. `bh_galaxy_mode_plan` loads that for **decode**, where
   every tensor is sharded and every op therefore takes its cores from a shard
   spec. **Prefill loads the full compute grid**, because it is interleaved
   throughout and an interleaved op takes its cores from
   `split_work_to_cores(compute_with_storage_grid_size(), ...)` anchored at
   `(0, 0)` with no way to confine it - read that helper's docstring before
   narrowing anything here. Every core coordinate, matmul work grid and SDPA
   sub-core grid is read off the topology descriptor; none is written down. A
   Wormhole coordinate carried over unexamined is the most likely way this file
   would waste a reservation, so it contains no literal core coordinate at all.
2. **No ring matmul** (`has_ring_matmul` is `False`, `ring_core_coords` is
   `None`). The QKV and WO projections are interleaved-DRAM 2D-multicast
   matmuls, confined with `recipes.dense_matmul_program_config(..., topology)`.
   Passing the topology is load-bearing: without it that helper resolves
   Wormhole's three-column rectangle instead of Blackhole's ten. The
   confinement is kept for both modes even though prefill's partition would now
   admit an auto-selected grid: these are the file's most expensive programs,
   and the reference's one recorded measurement here is that folding column 11
   into a worker partition regresses prefill warmup.
3. **No fused galaxy CCLs** (`has_fused_ccl` is `False`). Five of them
   silently no-op on the 2D-torus fabric, so every collective below is a
   standard/stable op pinned to the worker sub-device. See
   `_AttentionCollectives` for the one place this costs something real: the
   decode 32-user -> 8-user column slice.
4. **No fused QK rotary** (`has_fused_qk_rotary` is `False`). This suite uses
   an identity rotation on both sides, exactly as the Wormhole suite does -
   RoPE has its own qualification and mixing the two would make a rotary defect
   read as an attention defect - so no rotary op is issued at all and the
   Q/K-disjointness the fused op demands is not a constraint here.
5. **Two fabric links, not four**, and `num_links` on one collective is a
   *different quantity* from that budget. The axis-1 collectives use one link
   (the reference's `num_all_gather_links = 1`); the axis-0 collectives use the
   whole budget, which `_links` clamps from Wormhole's 4 to Blackhole's 2.
   Over-requesting deadlocks with no traceback.
6. **SDPA `k_chunk_size` is 256, not 512**, for sequences at or above 2048.
   It now comes off the descriptor as `prefill_sdpa_k_chunk`, so
   `recipes.sdpa_program_config` resolves it per architecture and this suite
   uses the shared recipe for both stages rather than hand-building prefill.
   `q_chunk_size` stays 256 on both architectures -- the reference splits only
   the `k` half -- and `sub_core_grids` is passed on the prefill path purely
   for parity: `sdpa_program_factory.cpp` builds
   `CoreRange((0, 0), (grid.x - 1, grid.y - 1))` from
   `compute_with_storage_grid_size` and never reads the argument, so what places
   that program is the grid, and the partition it needs is one containing the
   whole grid anchored at the origin -- which is what `bh_galaxy_mode_plan`
   loads for prefill, and the reason it has to. Only `sdpa_decode`'s factory
   honours `sub_core_grids`, and there it must equal the grid's core count.

**Ordered cheapest-first, one correctness claim per node id.** Prefill needs no
column-user slicing and no fused collective at all, so it is both the cheapest
and the least-blocked signal; decode is last. Nothing here depends on state
another test left behind: two models in one process share compiled programs and
buffer addresses and hang at the second model's first decode, so every test is
independently runnable as a single node id and the run protocol is one node id
per process.

**Staged PCC is printed for every stage**, because `comp_pcc` logs nothing on
success and a final-output PCC on a module this size localizes nothing. The
stages are observed through `Attention2DLowLevelCallables`, which is the only
seam that does not require touching `src/`:

| stage | callable that sees it |
| --- | --- |
| QKV projection, column-reduced | `reduce_qkv`'s return value |
| created heads + Q/K norm (the rotary input) | `rotary`'s arguments |
| SDPA output | `gather_users`' argument (decode), `gather_heads`' (prefill) |
| WO projection, before the row reduction | `reduce_output`'s argument |

**Per-mesh-column and per-mesh-row correlation, not only aggregate PCC.** A
column-local sharding bug gives PCC in 0-0.01 and needs per-column correlation
to localize; a channel-order mismatch shows up as a *permutation* - every
column well correlated with *some* reference column, just not its own - and
that signature is what distinguishes it from bad numerics. Two places make this
free rather than expensive:

* after the axis-1 QKV reduction the four mesh columns must hold *identical*
  values, so disagreement between columns is simultaneously the no-op detector
  for that collective and the channel-order detector;
* after the axis-0 output reduction the eight mesh rows must hold identical
  values, likewise.

So no poison pattern is needed: a no-op axis-1 all-reduce leaves each column
holding its own partial sum, which the column report reads off directly.
"""

from __future__ import annotations

import gc
import math
import traceback
from dataclasses import dataclass
from typing import Any

import pytest
import torch
from transformers import AutoModelForCausalLM, LlamaConfig, Qwen3Config

# transformers 5.x moved no_init_weights to transformers.initialization; fall back
# to the old location for transformers < 5.x.
try:
    from transformers.initialization import no_init_weights
except ImportError:
    from transformers.modeling_utils import no_init_weights

import ttnn
from tests.models.galaxy.galaxy_hardware import BHGLX_DEVICE_PARAMS
from tests.modules._bh_galaxy_hardware import (
    bh_galaxy_mode_plan,
    bh_galaxy_resources_config,
    bh_galaxy_topology,
    deallocate_module_weights,
    deallocate_tensor,
    exact_tensor_resource,
    record_module_output,
    require_bh_galaxy_prefetch_free_resources,
)
from tests.modules._hf_reference import (
    HfAttentionWrapper,
    IdentityRotaryEmbedding,
    get_attention_weights_from_ref_model,
    reverse_permute_1d,
)
from tests.support.comparison import comp_pcc

from tt_transformers.models.galaxy import (
    GalaxyCollectivePlan,
    GalaxyColumnUserSelector,
    GalaxyResourceKey,
    GalaxyTensorSpec,
)
from tt_transformers.models.galaxy.recipes import (
    compute_kernel_config,
    dense_matmul_program_config,
    exact_gather_compute_kernel_config,
    height_sharded_memory_config,
    sdpa_program_config,
    worker_cores,
)
from tt_transformers.models.galaxy.topology import GalaxyChipTopology
from tt_transformers.modules.attention.attention_2d import (
    Attention2D,
    Attention2DConfig,
    Attention2DLowLevelCallables,
    Attention2DSequenceConfig,
    DecodeMetadata,
    KVCacheBinding,
    PrefillAttentionMode,
    PrefillCollectiveMode,
    PrefillMetadata,
    PrefillRecipeIdentity,
    PrefillRowMode,
)
from tt_transformers.modules.lazy_weight import LazyWeight
from tt_transformers.modules.rmsnorm.rmsnorm_2d import RMSNorm2DConfig, RMSNorm2DGeometry

_MESH_SHAPE = (8, 4)
_MESH_ROWS, _MESH_COLUMNS = _MESH_SHAPE
_BATCH_SIZE = 32
_USERS_PER_COLUMN = _BATCH_SIZE // _MESH_COLUMNS
_HEAD_DIM = 128
_TILE = 32
_MAX_SEQ_LEN = 2048
_PCC = 0.99
_DECODE_POSITIONS = (127, 128)

#: Cores the Qwen head-local Q/K decode norm runs its kernel on. One core wide
#: and eight tall is not a free choice: `_head_local_compute_memory_config`
#: rejects a grid wider than one core, requires a rectangle, and needs the
#: created heads' `users_per_column * padded_heads` rows to divide over the
#: cores in whole tiles. The *column* comes from the descriptor's envelope.
_HEAD_LOCAL_NORM_CORES = _USERS_PER_COLUMN

#: Token rows the *printed* staged reports cover. See `_stage_window`.
_STAGE_TOKENS = 256


@dataclass(frozen=True)
class _ModelSpec:
    """One model geometry. Constants taken from the Wormhole suite unchanged.

    `(8, 4)` is architecture-invariant, so the weight splits, the mesh mappers,
    the row/column sharding and `n_kv_heads == 8` - one KV head per mesh row -
    all carry from Wormhole without redesign. Qwen3-32B additionally carries
    `n_heads * head_dim (8192) != dim (5120)` and QK-norm; Llama-3.3-70B has
    neither.
    """

    name: str
    architecture: str
    dim: int
    n_heads: int
    n_kv_heads: int
    qk_norm: bool
    norm_eps: float

    @property
    def qkv_size(self) -> int:
        return (self.n_heads + 2 * self.n_kv_heads) * _HEAD_DIM

    @property
    def attention_dim(self) -> int:
        """`n_heads * head_dim`, which is not `dim` on Qwen3-32B."""

        return self.n_heads * _HEAD_DIM

    @property
    def local_qkv_size(self) -> int:
        return self.qkv_size // _MESH_ROWS

    @property
    def local_heads(self) -> int:
        return self.n_heads // _MESH_ROWS

    @property
    def local_kv_heads(self) -> int:
        return self.n_kv_heads // _MESH_ROWS

    @property
    def local_attention_dim(self) -> int:
        return self.attention_dim // _MESH_ROWS

    @property
    def local_dim(self) -> int:
        return self.dim // _MESH_COLUMNS


_MODEL_SPECS = (
    _ModelSpec("llama-70b", "llama", dim=8192, n_heads=64, n_kv_heads=8, qk_norm=False, norm_eps=1e-5),
    _ModelSpec("qwen3-32b", "qwen3", dim=5120, n_heads=40, n_kv_heads=8, qk_norm=True, norm_eps=1e-6),
)


# =============================================================================
# Blackhole geometry, all derived from the validated topology descriptor
# =============================================================================


def _links(topology: GalaxyChipTopology, requested: int) -> int:
    """Clamp a per-collective `num_links` to this architecture's link budget.

    Two different quantities are named in this file and only one of them is the
    mesh's budget. `1` on the column axis is the reference's
    `num_all_gather_links` for Blackhole - a deliberate per-collective choice
    below the budget, and the value `plans.py` already uses for the prefill QKV
    reduction. `4` *is* the Wormhole budget, i.e. "use every link", and clamps
    to Blackhole's 2 here. Over-requesting overruns the ethernet channel array
    and deadlocks the host with no traceback.
    """

    return min(requested, topology.fabric_links)


def _subgrid(topology: GalaxyChipTopology, count: int, *, row_wise: bool) -> ttnn.CoreRangeSet:
    """Take `count` worker cores from the envelope, anchored where it starts.

    The same construction `recipes._subgrid_cores` uses for the qualified
    Wormhole KV, SDPA and reduce-scatter placements, with the anchor read off
    the descriptor rather than named: it is `(1, 0)` on both architectures
    today, but on Blackhole it comes from `BLACKHOLE_FIRST_WORKER_COLUMN`, and
    if that constant moves after a timed prefill measurement this moves with
    it.
    """

    return ttnn.num_cores_to_corerangeset_in_subcoregrids(
        ttnn.CoreCoord(*topology.sampling_start_core), count, worker_cores(topology), row_wise=row_wise
    )


def _head_local_norm_cores(topology: GalaxyChipTopology) -> ttnn.CoreRangeSet:
    """Return the one-core-wide rectangle a head-local decode norm may use.

    Three properties are each something `ttnn.rms_norm` refuses without, and
    all three are why this is not simply `worker_cores()`:

    * **worker cores**, because an interleaved head-local norm spreads over the
      whole compute grid and aborts under the loaded sub-device manager;
    * **a rectangle**, which the sharded layernorm requires outright;
    * **one core wide**, so the whole `head_dim`-wide row lands on one core and
      the reduction needs no multicast.

    Blackhole's envelope is a single contiguous rectangle - there is no
    prefetch sender column splitting it - so its bounding box contains no core
    the sub-device does not own and the column can be taken from the box. The
    membership check stays anyway, because that is a property of *this*
    descriptor and not of the function.
    """

    workers = worker_cores(topology)
    origin = workers.bounding_box().start
    for offset in range(_HEAD_LOCAL_NORM_CORES):
        core = ttnn.CoreCoord(origin.x, origin.y + offset)
        if not workers.contains(core):
            raise ValueError(f"head-local norm core ({core.x}, {core.y}) is not a Blackhole worker core")
    return ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(origin.x, origin.y),
                ttnn.CoreCoord(origin.x, origin.y + _HEAD_LOCAL_NORM_CORES - 1),
            )
        }
    )


@dataclass(frozen=True)
class _DecodePlacements:
    """Blackhole decode placements, all derived from the worker envelope.

    These mirror `recipes.resolve_galaxy_decode_placements` for the attention
    stages the prefetcher-free path keeps: height-sharded created heads, KV and
    SDPA output, DRAM-interleaved everything the ring used to own. Height
    sharding under a *narrowed* sub-device is the production Wormhole decode
    shape, so it is proven under exactly this kind of partition; the DRAM
    interleaving is where Blackhole diverges.
    """

    heads_memcfg: ttnn.MemoryConfig
    kv_memcfg: ttnn.MemoryConfig
    sdpa_output_memcfg: ttnn.MemoryConfig
    gather_users_memcfg: ttnn.MemoryConfig
    sdpa_cores: ttnn.CoreRangeSet
    norm_cores: ttnn.CoreRangeSet


def _decode_placements(topology: GalaxyChipTopology) -> _DecodePlacements:
    head_cores = _subgrid(topology, _BATCH_SIZE, row_wise=False)
    kv_cores = _subgrid(topology, _USERS_PER_COLUMN, row_wise=False)
    sdpa_output_cores = _subgrid(topology, _USERS_PER_COLUMN, row_wise=True)
    gather_user_cores = _subgrid(topology, _BATCH_SIZE, row_wise=True)
    return _DecodePlacements(
        # `nlp_create_qkv_heads_decode` cuts the grid it is handed row-wise into
        # consecutive `batch`-core slices - Q, then K, then V - so this holds
        # three `users_per_column`-core slices with room to spare.
        heads_memcfg=height_sharded_memory_config(head_cores, _HEAD_DIM),
        kv_memcfg=height_sharded_memory_config(kv_cores, _HEAD_DIM),
        sdpa_output_memcfg=height_sharded_memory_config(sdpa_output_cores, _HEAD_DIM),
        gather_users_memcfg=height_sharded_memory_config(gather_user_cores, _HEAD_DIM),
        sdpa_cores=_subgrid(topology, _BATCH_SIZE, row_wise=True),
        norm_cores=_head_local_norm_cores(topology),
    )


# =============================================================================
# Collectives: standard ops only, each pinned to the worker sub-device
# =============================================================================


class _AttentionCollectives:
    """Attention low-level adapters over standard collectives, plus staging.

    **Why `reduce_create_qkv_heads` is `None` here, and what that costs.** The
    Wormhole decode recipe reduces and creates heads in one fused collective,
    `all_reduce_create_qkv_heads`, and that collective does two things at once:
    it reduces the QKV projection over the four mesh columns *and* it slices the
    32-user batch down to the `users_per_column` users this mesh column owns,
    through its `batch_offset` and `slice_size` arguments. On the 2D-torus
    fabric the fused galaxy collectives no-op silently, so the reduction has to
    move to standard ops - and the *slice* has nowhere to go.
    `Attention2D`'s own non-fused fallback calls
    `ttnn.experimental.nlp_create_qkv_heads_decode` with neither argument, so it
    creates heads for all 32 users while the contiguous KV cache, the decode
    SDPA and `current_positions` are all `users_per_column`-shaped.

    So the slice is done here, by `GalaxyColumnUserSelector`: a matmul against a
    one-hot selector whose rows differ per mesh column, which is the same
    mechanism the decode LM head already uses to hand `Sampling2D` its column's
    users, is production code rather than a test double, and is qualified on a
    Galaxy mesh as a bit-exact gather. It costs one extra matmul per decode
    step, and it means the decode graph measured here is **not** the production
    graph: it is the production graph with the fused collective replaced by a
    standard reduction plus an explicit column gather. That substitution is the
    point of the bring-up, and it is the first thing to revisit when the fused
    collective gains a 2D-fabric writer.

    `gather_users` uses `ttnn.all_gather`, which is what the production Galaxy
    collectives already use for this stage - `all_gather_concat`, the fused op,
    is never on this path - so no substitution is needed there.
    """

    def __init__(
        self,
        resources: Any,
        mesh_device: ttnn.MeshDevice,
        topology: GalaxyChipTopology,
        spec: _ModelSpec,
        placements: _DecodePlacements,
    ):
        self.resources = resources
        self.mesh_device = mesh_device
        self.topology = topology
        self.spec = spec
        self.placements = placements
        #: Composed host copies of every observable stage, keyed by stage name.
        #: Composed eagerly inside the graph body, because the device tensors are
        #: intermediates `Attention2D` frees as soon as the next stage owns its
        #: result.
        self.stages: dict[str, torch.Tensor] = {}
        self._selector = GalaxyColumnUserSelector(
            mesh_device,
            max_batch_size=_BATCH_SIZE,
            users_per_column=_USERS_PER_COLUMN,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            # Exact by default: a matmul is not a copy, and the default math
            # fidelity truncates the bfloat16 mantissa of its inputs, which turns
            # the "exact row gather" into a lossy one.
            compute_kernel_config=exact_gather_compute_kernel_config(),
            # Injected for two reasons. First, the default
            # `column_user_selector_program_config` resolves
            # `worker_matmul_rectangle()` on the **Wormhole** envelope, which is
            # the wrong rectangle here and takes no topology argument.
            #
            # Second, it is deliberately the 2D-multicast form rather than the
            # 1D `mcast_in0` form that helper builds, and the objection recorded
            # against the 2D form does not apply at this width. That objection is
            # about the *vocabulary*: with `M` one tile the 2D grid is one block
            # tall, so all of `N` lands on the rectangle's first row, and 504-600
            # vocabulary tiles would give `per_core_N = 168` and ~1 MB of L1 in
            # circular buffers. `N` here is `local_qkv_size`, 40 tiles at most,
            # spread over Blackhole's ten-column rectangle - `per_core_N` of four
            # and a four-tile `in1` buffer. The 1D form would instead want one
            # block per core across a hundred-core rectangle for forty blocks of
            # work, which is the shape that has no valid `per_core_N`.
            #
            # `rows` is the selector's own `M`: `users_per_column` rows, one
            # tile, and `local_k` is the one-tile `K` of a 32-wide one-hot.
            program_config=dense_matmul_program_config(_USERS_PER_COLUMN, _TILE, spec.local_qkv_size, topology),
        )

    def cleanup(self) -> None:
        self._selector.release()

    def clear_stages(self) -> None:
        self.stages.clear()

    # -- staging ----------------------------------------------------------

    def _stage(self, name: str, tensor: Any, dims: tuple[int, int]) -> None:
        """Compose one stage to host and keep it, without disturbing the graph.

        `dims` is the `(row axis, column axis)` pair of tensor dimensions the
        mesh rows and mesh columns are laid out on, so every stage is stored
        with both mesh axes addressable. That is what makes the per-column and
        per-row reports below possible at all: a composer that folded either
        axis away would leave only an aggregate.
        """

        # Deliberately not upcast to float32 here: at the full 2048-token
        # prefill one staged QKV tensor is 84 million elements on Llama, and six
        # of them held as float32 is a gigabyte of host memory for a report.
        # `_pcc` upcasts the small slices it actually compares.
        self.stages[name] = ttnn.to_torch(
            tensor, mesh_composer=ttnn.ConcatMesh2dToTensor(self.mesh_device, dims=dims, mesh_shape=_MESH_SHAPE)
        )

    # -- collectives ------------------------------------------------------

    def _all_reduce(self, tensor: ttnn.Tensor, *, mode: str, cluster_axis: int) -> ttnn.Tensor:
        """Reduce over one mesh axis with standard reduce-scatter + all-gather.

        Both halves are the stable composite ops, both name the worker
        sub-device, and both stay in interleaved DRAM. The persistent buffers
        the plan provisions for this key are deliberately unused: the composite
        ops allocate their own output, and borrowing a plan buffer here would
        compile out the op's start barrier for no benefit on a suite that runs
        eagerly.

        **`num_workers_per_link` is not optional on the decode envelope.** Left
        out, `reduce_scatter` sizes its own selection from the data and then
        shifts it by the envelope origin, which walks off the far edge and
        aborts before a single element moves -- measured on Blackhole, on a run
        whose `all_gather` on the same axis and the same envelope passed.
        See `reduce_scatter_workers_per_link` for why only this collective
        offsets.

        The count comes from the **context**, not from the descriptor, because
        the offset belongs to the sub-device that is actually loaded and the two
        modes no longer load the same one: prefill takes the full compute grid,
        whose origin *is* the grid origin, so nothing shifts and the context
        returns `None` -- the op's own data-sized choice, unconstrained. Sizing
        both modes against the descriptor's decode rectangle instead would pin
        prefill to one worker per link for a shift that does not happen there.
        """

        context = self.resources.context(mode)
        resource = exact_tensor_resource(context, "all_reduce", cluster_axis, tensor)
        reduced = ttnn.reduce_scatter(
            tensor,
            3,
            cluster_axis=cluster_axis,
            num_links=resource.num_links,
            topology=resource.topology,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            subdevice_id=context.worker_sub_device_id,
            num_workers_per_link=context.reduce_scatter_workers_per_link(resource.num_links),
        )
        output = ttnn.all_gather(
            reduced,
            3,
            cluster_axis=cluster_axis,
            num_links=resource.num_links,
            topology=resource.topology,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            subdevice_id=context.worker_sub_device_id,
        )
        # Sub-device-scoped, never a whole-device sync: a whole-device
        # synchronize on a mesh whose loaded manager owns only the worker
        # envelope is how a CCL hang gets misread as a compute hang.
        self.resources.synchronize(mode)
        deallocate_tensor(reduced)
        return output

    def is_borrowed_output(self, tensor: Any) -> bool:
        return False

    def rotary(self, q: ttnn.Tensor, k: ttnn.Tensor, rot_mats: Any, *, mode: str, **_: Any):
        """Identity rotation, and the staging point for the created heads.

        RoPE has its own hardware qualification, and `has_fused_qk_rotary` is
        `False` on Blackhole so the fused pair is not the path here in any case.
        Leaving the rotation out keeps this file measuring Attention2D's
        projection, head creation, Q/K norm, SDPA, cache and CCL ownership; the
        reference is driven with `IdentityRotaryEmbedding` so both sides apply
        the same no-op.
        """

        assert rot_mats is None, "this suite qualifies Attention2D with an identity rotation"
        # Q and K are height-sharded over mesh rows (heads) and mesh columns
        # (users in decode, replicated in prefill).
        self._stage(f"{mode}_q_heads", q, (0, 1) if mode == "decode" else (1, 0))
        self._stage(f"{mode}_k_heads", k, (0, 1) if mode == "decode" else (1, 0))
        return q, k

    def reduce_qkv(self, tensor: ttnn.Tensor, *, mode: str, **_: Any) -> ttnn.Tensor:
        """Reduce the fused QKV projection over the four mesh columns.

        In decode this also performs the column-user slice the fused collective
        would have done; see the class docstring for why that has to happen
        here and what it costs.
        """

        reduced = self._all_reduce(tensor, mode=mode, cluster_axis=1)
        # Staged *before* the slice, so the four mesh columns are still
        # replicas of one another and their disagreement is readable.
        self._stage(f"{mode}_qkv", reduced, (1, 3))
        if mode != "decode":
            return reduced
        sliced = self._selector(reduced)
        # Staged *after* the slice as well, which is not symmetry for its own
        # sake: on 2026-09-17 this suite's decode had `qkv` at PCC 0.999662 with
        # the four column replicas bit-identical
        # (`max_abs_delta_vs_columns=[0, 0, 0, 0]`) and the very next stage,
        # `q heads`, at `pcc=nan norm_ratio=inf` on all eight mesh rows. `inf`
        # is uninitialised memory rather than a numerical fault, so the defect
        # is a placement or a shard mapping — and it sits in the gap between
        # those two stages, which contained *two* operations with nothing
        # between them: this column-user slice and
        # `nlp_create_qkv_heads_decode`. One reading could not say which.
        #
        # `dims=(0, 1)` and not `(1, 3)`: after the slice each mesh column holds
        # its own disjoint `users_per_column` users rather than a replica of all
        # 32, so the column axis belongs on the user axis. That is the same
        # composition `_compose_heads` uses for the created heads, which is what
        # makes this stage and the next directly comparable.
        self._stage("decode_qkv_sliced", sliced, (0, 1))
        deallocate_tensor(reduced)
        return sliced

    def gather_heads(self, tensor: ttnn.Tensor, *, mode: str, **_: Any) -> ttnn.Tensor:
        """Return the concatenated heads unchanged, and stage them.

        Decode gathers users before the head concat and prefill keeps the
        row-local K shard the WO matmul expects, so no collective is required
        at this seam on either architecture.
        """

        if mode == "prefill":
            self._stage("prefill_attention", tensor, (1, 3))
        return tensor

    def gather_users(self, tensor: ttnn.Tensor, *, mode: str, **_: Any) -> ttnn.Tensor:
        """Gather each column's user group into the full physical batch."""

        if mode != "decode":
            raise ValueError("user gather is decode-only")
        self._stage("decode_attention", tensor, (0, 1))
        context = self.resources.context(mode)
        resource = exact_tensor_resource(context, "all_gather", 1, tensor)
        output = ttnn.all_gather(
            tensor,
            1,
            cluster_axis=1,
            num_links=resource.num_links,
            topology=resource.topology,
            memory_config=self.placements.gather_users_memcfg,
            subdevice_id=context.worker_sub_device_id,
        )
        self.resources.synchronize(mode)
        return output

    def reduce_output(self, tensor: ttnn.Tensor, *, mode: str, **_: Any) -> ttnn.Tensor:
        """Reduce the WO projection over the eight mesh rows."""

        self._stage(f"{mode}_wo", tensor, (1, 3))
        return self._all_reduce(tensor, mode=mode, cluster_axis=0)

    def callables(self) -> Attention2DLowLevelCallables:
        return Attention2DLowLevelCallables(
            rotary=self.rotary,
            reduce_qkv=self.reduce_qkv,
            gather_heads=self.gather_heads,
            reduce_output=self.reduce_output,
            is_borrowed_output=self.is_borrowed_output,
            # `None` selects Attention2D's standard reduce + create-heads route
            # rather than the fused collective. See the class docstring.
            reduce_create_qkv_heads=None,
            gather_users=self.gather_users,
        )


# =============================================================================
# Collective resource plans
# =============================================================================


def _all_reduce_plan(topology: GalaxyChipTopology, cluster_axis: int, shape: tuple[int, ...]) -> GalaxyCollectivePlan:
    """Plan one axis reduction.

    `Topology.Ring` on both axes, which is what `plans.py` uses and what the
    Blackhole port records as correct: on Blackhole the pairing is
    `Topology.Ring` with `FABRIC_2D_TORUS_XY`, not Wormhole's
    `Topology.Ring` with `FABRIC_1D_RING`. A mismatched topology/fabric pair is
    an opaque hang rather than an error, which is why this deviates
    deliberately from the Wormhole suite's `Topology.Linear` on the column axis
    instead of carrying it.
    """

    return GalaxyCollectivePlan(
        key=GalaxyResourceKey("all_reduce", cluster_axis, shape, math.prod(shape[:-1])),
        topology=ttnn.Topology.Ring,
        num_links=_links(topology, 1 if cluster_axis == 1 else 4),
        # Provisioned because `GalaxyCollectivePlan` requires a persistent
        # output, and unused because the composite collectives allocate their
        # own. Kept at the true output shape in DRAM so it stays cheap and
        # honest rather than a placeholder of the wrong size.
        persistent_output_specs=(GalaxyTensorSpec(shape, ttnn.bfloat16, ttnn.TILE_LAYOUT, ttnn.DRAM_MEMORY_CONFIG),),
    )


def _prefill_collectives(
    topology: GalaxyChipTopology, spec: _ModelSpec, sequence_length: int
) -> tuple[GalaxyCollectivePlan, ...]:
    leading = (1, 1, sequence_length)
    return (
        _all_reduce_plan(topology, 1, (*leading, spec.local_qkv_size)),
        _all_reduce_plan(topology, 0, (*leading, spec.local_dim)),
    )


def _decode_collectives(
    topology: GalaxyChipTopology, spec: _ModelSpec, placements: _DecodePlacements
) -> tuple[GalaxyCollectivePlan, ...]:
    leading = (1, 1, _BATCH_SIZE)
    users_shape = (1, _USERS_PER_COLUMN, spec.local_heads, _HEAD_DIM)
    return (
        _all_reduce_plan(topology, 1, (*leading, spec.local_qkv_size)),
        _all_reduce_plan(topology, 0, (*leading, spec.local_dim)),
        GalaxyCollectivePlan(
            key=GalaxyResourceKey("all_gather", 1, users_shape, math.prod(users_shape[:-1])),
            topology=ttnn.Topology.Ring,
            # One link, which is the reference's Blackhole `num_all_gather_links`
            # and a per-collective choice below the mesh's budget of two.
            num_links=_links(topology, 1),
            persistent_output_specs=(
                GalaxyTensorSpec(
                    (1, _BATCH_SIZE, spec.local_heads, _HEAD_DIM),
                    ttnn.bfloat16,
                    ttnn.TILE_LAYOUT,
                    placements.gather_users_memcfg,
                ),
            ),
        ),
    )


def _resources_config(
    mesh_device: ttnn.MeshDevice,
    topology: GalaxyChipTopology,
    spec: _ModelSpec,
    placements: _DecodePlacements,
    prefill_lengths: tuple[int, ...],
) -> Any:
    """Build the resources config for exactly the collectives one node id uses.

    `semaphore_cores` is left at `bh_galaxy_mode_plan`'s default, which is the
    whole worker sub-device. The generic async CCLs pick their sender worker
    cores from the sub-device minus the reserved output cores, so a narrower
    semaphore set leaves a sender polling an L1 address its own core never had
    reserved or zeroed - which hangs rather than fails. Nothing here binds a
    semaphore to a grid it owns, so nothing here may narrow it.

    Both mode plans always exist because `GalaxyResourcesConfig` requires both,
    and each needs at least one collective. The unused mode's persistent buffers
    are DRAM and small.
    """

    prefill = tuple(plan for length in prefill_lengths for plan in _prefill_collectives(topology, spec, length))
    return bh_galaxy_resources_config(
        mesh_device,
        prefill=bh_galaxy_mode_plan("prefill", prefill, mesh_device, topology=topology),
        decode=bh_galaxy_mode_plan(
            "decode", _decode_collectives(topology, spec, placements), mesh_device, topology=topology
        ),
    )


# =============================================================================
# Host reference
# =============================================================================


def _hf_config(spec: _ModelSpec) -> Any:
    """HuggingFace config for the attention geometry this spec models.

    Built locally rather than downloaded: only the attention block matters, so
    the vocabulary and MLP are shrunk to keep the host reference small. No
    checkpoint is needed, which is deliberate - a wrong `HF_HOME` silently
    skipping every real-checkpoint test is how a run looks green and measures
    nothing.
    """

    settings = dict(
        vocab_size=128,
        hidden_size=spec.dim,
        intermediate_size=256,
        num_hidden_layers=1,
        num_attention_heads=spec.n_heads,
        num_key_value_heads=spec.n_kv_heads,
        head_dim=_HEAD_DIM,
        max_position_embeddings=_MAX_SEQ_LEN,
        rms_norm_eps=spec.norm_eps,
        attention_bias=False,
    )
    if spec.architecture == "qwen3":
        return Qwen3Config(**settings, use_sliding_window=False)
    return LlamaConfig(**settings)


def _reference_attention(spec: _ModelSpec) -> HfAttentionWrapper:
    """Build the HuggingFace attention Attention2D is qualified against.

    Weights are drawn in bfloat16 and held in float32, so the reference runs at
    full precision on exactly the values the device receives.
    """

    with no_init_weights():
        hf_model = AutoModelForCausalLM.from_config(_hf_config(spec), torch_dtype=torch.float32)
    reference_attn = hf_model.model.layers[0].self_attn
    weight_scale = 1.0 / math.sqrt(spec.dim)
    with torch.no_grad():
        for _name, parameter in reference_attn.named_parameters():
            if parameter.dim() >= 2:  # projections
                values = torch.randn(parameter.shape, dtype=torch.bfloat16) * weight_scale
            else:  # Q/K norm gains
                values = 1.0 + 0.05 * torch.randn(parameter.shape, dtype=torch.bfloat16)
            parameter.copy_(values.float())
    return HfAttentionWrapper(reference_attn, _HEAD_DIM, IdentityRotaryEmbedding(_HEAD_DIM))


def _reference_decode(reference: HfAttentionWrapper, x: torch.Tensor, position: int) -> torch.Tensor:
    """One decode step for every user at `position`, from `(1, 1, batch, dim)`."""

    output = reference(x[0, 0].float().unsqueeze(1), start_pos=position, mask=None)
    return output.reshape(1, 1, _BATCH_SIZE, -1)


def _reference_prefill(reference: HfAttentionWrapper, x: torch.Tensor) -> torch.Tensor:
    """Causal prefill of one user, from `(1, 1, seq_len, dim)`."""

    reference.reset_cache()
    output = reference(x[0].float(), start_pos=0, mask=None)
    return output.reshape(1, 1, x.shape[-2], -1)


def _reference_cache_kv(reference: HfAttentionWrapper, index: Any) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference K/V at `index` of `[batch, seq_len, n_kv_heads, head_dim]`.

    Attention2D is fed Q/K projections in Meta (interleaved) layout while the HF
    reference keeps its own, so the cached keys differ by that per-head
    permutation; with identity rotation, applying it makes them equal. Values
    are unpermuted.
    """

    return reverse_permute_1d(reference.cache_k[index]), reference.cache_v[index]


def _rms_normalize(tensor: torch.Tensor, weight: torch.Tensor | None, eps: float) -> torch.Tensor:
    if weight is None:
        return tensor
    scale = torch.rsqrt(tensor.pow(2).mean(dim=-1, keepdim=True) + eps)
    return tensor * scale * weight


def _split_qkv(spec: _ModelSpec, qkv: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split a full `(rows, qkv_size)` projection into per-head Q, K and V.

    The layout is the row-fused one `get_attention_weights_from_ref_model`
    builds for eight mesh rows: eight consecutive `local_qkv_size` blocks, each
    holding that row's Q heads, then its K heads, then its V heads. Mesh row `r`
    carries Q heads `r * local_heads ...` and KV head `r`, so concatenating the
    blocks in row order recovers global head order - which is the same thing
    `nlp_create_qkv_heads` does on device, and the reason a per-row
    disagreement in the report below points at a row and not at a permutation.
    """

    local_q = spec.local_heads * _HEAD_DIM
    local_kv = spec.local_kv_heads * _HEAD_DIM
    q_parts, k_parts, v_parts = [], [], []
    for row in range(_MESH_ROWS):
        block = qkv[:, row * spec.local_qkv_size : (row + 1) * spec.local_qkv_size]
        q_parts.append(block[:, :local_q])
        k_parts.append(block[:, local_q : local_q + local_kv])
        v_parts.append(block[:, local_q + local_kv :])
    rows = qkv.shape[0]
    return (
        torch.cat(q_parts, dim=-1).reshape(rows, spec.n_heads, _HEAD_DIM),
        torch.cat(k_parts, dim=-1).reshape(rows, spec.n_kv_heads, _HEAD_DIM),
        torch.cat(v_parts, dim=-1).reshape(rows, spec.n_kv_heads, _HEAD_DIM),
    )


def _torch_decode_attention(
    spec: _ModelSpec, q: torch.Tensor, keys: torch.Tensor, values: torch.Tensor
) -> torch.Tensor:
    """Single-position attention for every user. `keys`/`values`: `(rows, L, n_kv, hd)`."""

    groups = spec.n_heads // spec.n_kv_heads
    k_rep = keys.repeat_interleave(groups, dim=2)
    v_rep = values.repeat_interleave(groups, dim=2)
    scores = (q.unsqueeze(1) * k_rep).sum(dim=-1) * (_HEAD_DIM**-0.5)
    weights = torch.softmax(scores, dim=1)
    return (weights.unsqueeze(-1) * v_rep).sum(dim=1)


def _stage_window(rows: int) -> slice:
    """Return the token rows the staged reports compare.

    **A cap, and the reason for it is host cost, not device cost.** At the full
    2048-token prefill the staged QKV projection composes to `(1, 8, 2048, 4 *
    local_qkv)`, which is 84 million elements on Llama, and a host causal
    attention over 2048 queries needs a `(64, 2048, 2048)` score matrix - about
    a gigabyte, times two for the softmax, for a report that is printed rather
    than asserted. The final output assertion against HuggingFace still covers
    every token, because it never forms an attention matrix.

    The window is the **last** rows rather than the first. Those queries attend
    over the deepest key history, so they are where a chunked-SDPA or
    `k_chunk_size` defect shows up - the Blackhole delta this suite exists to
    exercise - and a defect confined to the first chunk would still move the
    full-sequence output assertion.
    """

    return slice(max(0, rows - _STAGE_TOKENS), rows)


def _torch_prefill_attention(
    spec: _ModelSpec, q: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, window: slice
) -> torch.Tensor:
    """Causal attention for the queries in `window`, over the whole sequence.

    Keys and values are never windowed: query `p` attends over positions
    `0..p`, so truncating them would change the answer rather than the cost.
    All arguments are `(seq, heads, head_dim)`; the result is
    `(len(window), n_heads, head_dim)`.
    """

    groups = spec.n_heads // spec.n_kv_heads
    k_rep = keys.repeat_interleave(groups, dim=1)
    v_rep = values.repeat_interleave(groups, dim=1)
    queries = q[window]
    scores = torch.einsum("qhd,khd->hqk", queries, k_rep) * (_HEAD_DIM**-0.5)
    start = window.start or 0
    query_positions = torch.arange(start, start + queries.shape[0]).unsqueeze(1)
    key_positions = torch.arange(q.shape[0]).unsqueeze(0)
    causal = torch.where(key_positions <= query_positions, 0.0, float("-inf"))
    weights = torch.softmax(scores + causal, dim=-1)
    return torch.einsum("hqk,khd->qhd", weights, v_rep)


def _wo_partials(spec: _ModelSpec, attention: torch.Tensor, wo: torch.Tensor) -> torch.Tensor:
    """Per-mesh-row WO partial products, `(rows, tokens, dim)`.

    Mesh row `r` owns attention-dimension rows `r * local_attention_dim ...` of
    `wo`, and holds exactly the heads that feed them, so its matmul is a
    partial sum of the full output. The axis-0 reduction sums these eight.
    """

    flat = attention.reshape(attention.shape[0], -1)
    stride = spec.local_attention_dim
    return torch.stack(
        [
            flat[:, row * stride : (row + 1) * stride] @ wo[row * stride : (row + 1) * stride]
            for row in range(_MESH_ROWS)
        ]
    )


# =============================================================================
# Reporting: staged PCC, per-mesh-column, per-mesh-row, per-user
# =============================================================================


def _pcc(expected: torch.Tensor, actual: torch.Tensor) -> float:
    """Pearson correlation of two tensors, flattened. Printed, never silent."""

    left = expected.float().reshape(-1)
    right = actual.float().reshape(-1)
    left = left - left.mean()
    right = right - right.mean()
    denominator = (left.norm() * right.norm()).clamp_min(1e-12)
    return float((left @ right) / denominator)


def _report_stage(case: str, stage: str, expected: torch.Tensor, actual: torch.Tensor) -> float:
    value = _pcc(expected, actual)
    ratio = float(actual.float().norm() / expected.float().norm().clamp_min(1e-12))
    print(f"[bh-attn {case}] stage {stage}: pcc={value:.6f} norm_ratio={ratio:.4f}", flush=True)
    return value


def _report_column_replicas(case: str, stage: str, reference: torch.Tensor, per_column: list[torch.Tensor]) -> None:
    """Report mesh columns that are supposed to be replicas of one another.

    After the axis-1 QKV reduction all four columns hold the same value, so the
    maximum absolute difference between them is the **no-op detector** for that
    collective: a 1D-multicast writer that no-ops on the 2D fabric moves no
    data and raises nothing, leaving each column holding its own partial sum,
    and that shows up here as a non-zero spread even before PCC is consulted.
    No poison pattern is needed because a legitimate result has a spread of
    exactly zero.
    """

    for index, candidate in enumerate(per_column):
        spread = [round(float((candidate - other).abs().max()), 6) for other in per_column]
        print(
            f"[bh-attn {case}] stage {stage} mesh column {index}: pcc_vs_reference={_pcc(reference, candidate):.6f} "
            f"max_abs_delta_vs_columns={spread}",
            flush=True,
        )


def _report_column_permutation(
    case: str, stage: str, references: list[torch.Tensor], candidates: list[torch.Tensor]
) -> None:
    """Cross-correlate every device mesh column against every reference column.

    This is the report a column-local sharding bug needs, and the reason
    aggregate PCC is close to useless for one. Such a bug lands in 0-0.01 and
    says nothing about where. A **permutation** - every device column well
    correlated with *some* reference column, just not its own - is a
    channel-order mismatch, which is the silent Blackhole failure that comes
    from feeding one collective's per-device channel order into an interleaved
    matmul. Bad correlation *everywhere* is instead bad numerics. The two are
    indistinguishable without printing the whole row, so the whole row is
    printed on a pass as well as on a failure.
    """

    for index, candidate in enumerate(candidates):
        row = [_pcc(reference, candidate) for reference in references]
        best = max(range(len(row)), key=row.__getitem__)
        print(
            f"[bh-attn {case}] stage {stage} mesh column {index}: pcc_vs_own_column={row[index]:.6f} "
            f"best_matching_reference_column={best} pcc_vs_reference_columns={[round(value, 4) for value in row]}",
            flush=True,
        )


def _report_mesh_rows(case: str, stage: str, expected: torch.Tensor, per_row: list[torch.Tensor]) -> None:
    for index, candidate in enumerate(per_row):
        print(
            f"[bh-attn {case}] stage {stage} mesh row {index}: pcc_vs_reference={_pcc(expected, candidate):.6f}",
            flush=True,
        )


def _report_per_user(case: str, expected: torch.Tensor, actual: torch.Tensor) -> None:
    """Report every user's own correlation and its best match among the users.

    Per-user rather than aggregate because the users are exactly what the mesh
    columns carry in decode: a user landing in the wrong column shows up here as
    `best_matching_user != user` while the aggregate stays plausible.
    """

    expected_rows = expected.float().reshape(-1, expected.shape[-1])
    actual_rows = actual.float().reshape(-1, actual.shape[-1])
    expected_centered = expected_rows - expected_rows.mean(dim=1, keepdim=True)
    actual_centered = actual_rows - actual_rows.mean(dim=1, keepdim=True)
    correlations = (expected_centered @ actual_centered.T) / (
        expected_centered.norm(dim=1, keepdim=True) * actual_centered.norm(dim=1).unsqueeze(0)
    ).clamp_min(1e-12)
    own = correlations.diagonal()
    best_values, best_users = correlations.max(dim=1)
    ratios = actual_rows.norm(dim=1) / expected_rows.norm(dim=1).clamp_min(1e-12)
    print(
        f"[bh-attn {case}] per-user pcc={[round(value, 4) for value in own.tolist()]}",
        flush=True,
    )
    print(
        f"[bh-attn {case}] per-user best_match={best_users.tolist()} "
        f"best_pcc={[round(value, 4) for value in best_values.tolist()]} "
        f"norm_ratios={[round(value, 3) for value in ratios.tolist()]}",
        flush=True,
    )


def _assert_pcc(case: str, expected: torch.Tensor, actual: torch.Tensor) -> None:
    passing, message = comp_pcc(expected.float(), actual.float(), _PCC)
    print(f"[bh-attn {case}] pcc>={_PCC}: {'pass' if passing else 'FAIL'} ({message})", flush=True)
    if not passing:
        _report_per_user(case, expected, actual)
    assert passing, f"{case} failed PCC>={_PCC}: {message}"


# =============================================================================
# Module construction
# =============================================================================


def _mesh_mapper(*placements: Any) -> ttnn.MeshMapperConfig:
    return ttnn.MeshMapperConfig(placements=list(placements), mesh_shape_override=ttnn.MeshShape(*_MESH_SHAPE))


def _lazy_weight(source: torch.Tensor, mesh_device: ttnn.MeshDevice, mapper: Any) -> LazyWeight:
    """Place one projection weight, interleaved in DRAM.

    Interleaved, not DRAM width-sharded: `has_ring_matmul` is `False`, the ring
    memory configs are not placeable on this architecture, and
    `MatmulMultiCoreReuseMultiCastProgramConfig` requires an interleaved `in1`
    regardless. Feeding a ring-sharded input into an interleaved matmul is also
    the silent half of the Blackhole MLP failure - a per-device channel-order
    mismatch that yields PCC near zero with nothing raised - so this is a
    correctness choice, not a convenience.
    """

    return LazyWeight(
        source=source,
        device=mesh_device,
        mesh_mapper_config=mapper,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat8_b,
    )


def _runtime_tensor_factory(offsets: tuple[int, ...], lower: tuple[int, ...], upper: tuple[int, ...], mesh: Any):
    mapper = ttnn.ReplicateTensorToMesh(mesh)

    def make(values: tuple[int, ...]) -> ttnn.Tensor:
        return ttnn.from_torch(
            torch.tensor(values, dtype=torch.int32),
            device=mesh,
            mesh_mapper=mapper,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    return make(offsets), make(lower), make(upper)


def _qk_norm_config(
    weight: torch.Tensor | None,
    spec: _ModelSpec,
    mesh_device: ttnn.MeshDevice,
    placements: _DecodePlacements,
) -> RMSNorm2DConfig | None:
    """Head-local Q/K norm placement for the Blackhole worker envelope.

    **Decode names no placement and names compute cores instead**, which is the
    production shape and not a workaround. Two things make it necessary and the
    Wormhole suite's all-DRAM configuration wrong here:

    * an interleaved `ttnn.rms_norm` resolves `LayerNormDefaultProgramConfig`,
      which splits its rows over the whole `12 x 10` compute grid - including
      the dispatch column the loaded sub-device manager does not own - and
      aborts with *"Kernel group cores do not match sub device cores"*. The
      Wormhole module suites never hit this because their mode plan *is* the
      full compute grid; Blackhole's decode plan is the narrower worker
      envelope, so the narrow-partition behaviour that only production saw on
      Wormhole is the default here;
    * naming any single placement would relocate the created heads onto it.
      `Attention2D._apply_qk_norm` only relocates when `decode_input_memcfg` is
      set, so leaving it `None` keeps Q and K exactly where
      `nlp_create_qkv_heads_decode` put them and `RMSNorm2D` puts each back
      where it found it.

    **Prefill keeps interleaved DRAM, and no longer has to be confined.**
    `RMSNorm2D.prefill_forward` has no compute-core parameter for the head-local
    geometry, so a Qwen prefill issues exactly the unconfined interleaved
    `ttnn.rms_norm` described above - and the prefill sub-device is the full
    compute grid, which is the partition that program is placeable under. The
    missing `RMSNorm2DConfig.prefill_compute_cores` field is still worth having
    for the decode-shaped partition, but nothing in this file needs it.
    """

    if weight is None:
        return None
    return RMSNorm2DConfig(
        weight=LazyWeight(source=weight, device=mesh_device, dtype=ttnn.bfloat16),
        mesh_device=mesh_device,
        cluster_shape=_MESH_SHAPE,
        geometry=RMSNorm2DGeometry.HEAD_LOCAL,
        eps=spec.norm_eps,
        decode_input_memcfg=None,
        decode_output_memcfg=None,
        decode_compute_cores=placements.norm_cores,
        prefill_input_memcfg=ttnn.DRAM_MEMORY_CONFIG,
        prefill_output_memcfg=ttnn.DRAM_MEMORY_CONFIG,
        compute_kernel_config_prefill=compute_kernel_config(),
    )


def _sequence_config(
    spec: _ModelSpec,
    topology: GalaxyChipTopology,
    sequence_length: int,
) -> Attention2DSequenceConfig:
    """One frozen prefill recipe. Interleaved DRAM throughout, grids confined."""

    kernel = compute_kernel_config()
    workers = worker_cores(topology)
    return Attention2DSequenceConfig(
        identity=PrefillRecipeIdentity(
            sequence_length,
            PrefillRowMode.SINGLE_ROW,
            PrefillCollectiveMode.REGULAR,
            PrefillAttentionMode.REGULAR,
        ),
        qkv_program_config=dense_matmul_program_config(sequence_length, spec.local_dim, spec.local_qkv_size, topology),
        sdpa_program_config=sdpa_program_config(
            sequence_length, decode=False, sub_core_grids=workers, topology=topology
        ),
        wo_program_config=dense_matmul_program_config(
            sequence_length, spec.local_attention_dim, spec.local_dim, topology
        ),
        qkv_output_memory_config=ttnn.DRAM_MEMORY_CONFIG,
        heads_memory_config=ttnn.DRAM_MEMORY_CONFIG,
        kv_memory_config=ttnn.DRAM_MEMORY_CONFIG,
        sdpa_output_memory_config=ttnn.DRAM_MEMORY_CONFIG,
        concat_memory_config=ttnn.DRAM_MEMORY_CONFIG,
        wo_output_memory_config=ttnn.DRAM_MEMORY_CONFIG,
        qkv_kernel_config=kernel,
        sdpa_kernel_config=kernel,
        wo_kernel_config=kernel,
        activation_dtype=ttnn.bfloat16,
    )


def _make_module(
    spec: _ModelSpec,
    mesh_device: ttnn.MeshDevice,
    topology: GalaxyChipTopology,
    collectives: _AttentionCollectives,
    placements: _DecodePlacements,
    weights: dict[str, Any],
    prefill_lengths: tuple[int, ...],
    resources: Any,
) -> Attention2D:
    """Build the Blackhole decode/prefill attention geometry.

    **Both** prefetch contexts are named, which is a Blackhole-specific
    departure from the Wormhole suite. The contexts carry two things the module
    reads: `global_cb`, which `NullPrefetcher2D` sets to `None`, and
    `worker_sub_device_id`. Under a sub-device that is not the whole compute
    grid, every program enqueued must name exactly one sub-device or
    `fd_mesh_command_queue.cpp` refuses the workload outright - and on
    Blackhole *both* modes run on a narrowed envelope, because the full grid
    would put kernels on the dispatch column. The Wormhole suites can pass
    `prefill_prefetch_context=None` only because their prefill plan *is* the
    full compute grid.
    """

    kernel = compute_kernel_config()
    module = Attention2D.from_config(
        Attention2DConfig(
            wqkv=weights["wqkv"],
            wo=weights["wo"],
            # Left unset so prefill aliases the decode weights. Both modes want
            # the same interleaved-DRAM bfloat8_b placement here - there is no
            # ring weight to differ from - so a second copy would be one more
            # allocation measuring nothing. `release_device_weights` dedupes.
            prefill_wqkv=None,
            prefill_wo=None,
            n_heads=spec.n_heads,
            n_kv_heads=spec.n_kv_heads,
            head_dim=_HEAD_DIM,
            max_batch_size=_BATCH_SIZE,
            max_seq_len=_MAX_SEQ_LEN,
            low_level=collectives.callables(),
            runtime_tensor_factory=_runtime_tensor_factory,
            runtime_tensor_releaser=deallocate_tensor,
            q_norm_config=_qk_norm_config(weights["q_norm"], spec, mesh_device, placements),
            k_norm_config=_qk_norm_config(weights["k_norm"], spec, mesh_device, placements),
            mesh_device=mesh_device,
            architecture=topology.architecture,
            wqkv_mesh_mapper_config=weights["wqkv"].mesh_mapper_config,
            wo_mesh_mapper_config=weights["wo"].mesh_mapper_config,
            weight_memory_config=ttnn.DRAM_MEMORY_CONFIG,
            wo_weight_memory_config=ttnn.DRAM_MEMORY_CONFIG,
            weight_layout=ttnn.TILE_LAYOUT,
            wqkv_dtype=ttnn.bfloat8_b,
            wo_dtype=ttnn.bfloat8_b,
            decode_input_placement=ttnn.DRAM_MEMORY_CONFIG,
            decode_output_placement=ttnn.DRAM_MEMORY_CONFIG,
            prefill_input_placement=ttnn.DRAM_MEMORY_CONFIG,
            prefill_output_placement=ttnn.DRAM_MEMORY_CONFIG,
            decode_qkv_output_memory_config=ttnn.DRAM_MEMORY_CONFIG,
            decode_heads_memory_config=placements.heads_memcfg,
            decode_kv_memory_config=placements.kv_memcfg,
            decode_sdpa_output_memory_config=placements.sdpa_output_memcfg,
            decode_concat_memory_config=ttnn.DRAM_MEMORY_CONFIG,
            decode_concat_sub_core_grids=placements.gather_users_memcfg.shard_spec.grid,
            decode_wo_output_memory_config=ttnn.DRAM_MEMORY_CONFIG,
            # The topology is the fourth positional argument and it is not
            # optional here: without it these resolve Wormhole's three-column
            # rectangle instead of Blackhole's ten, which both under-uses the
            # mesh and anchors at a column the Blackhole envelope may not own.
            decode_program_config=dense_matmul_program_config(
                _BATCH_SIZE, spec.local_dim, spec.local_qkv_size, topology
            ),
            decode_sdpa_program_config=sdpa_program_config(
                _MAX_SEQ_LEN, decode=True, sub_core_grids=placements.sdpa_cores, topology=topology
            ),
            decode_wo_program_config=dense_matmul_program_config(
                _BATCH_SIZE, spec.local_attention_dim, spec.local_dim, topology
            ),
            decode_qkv_kernel_config=kernel,
            decode_sdpa_kernel_config=kernel,
            decode_wo_kernel_config=kernel,
            decode_activation_dtype=ttnn.bfloat16,
            decode_prefetch_context=resources.prefetch_context("decode"),
            prefill_prefetch_context=resources.prefetch_context("prefill"),
            prefill_sequence_configs={
                recipe.identity: recipe
                for recipe in (_sequence_config(spec, topology, length) for length in prefill_lengths)
            },
        )
    )
    module._hardware_collectives = collectives
    return module


def _make_cache(module: Attention2D, mesh_device: ttnn.MeshDevice) -> KVCacheBinding:
    """Allocate the contiguous KV cache, one mesh column's users per device.

    `n_kv_heads == 8` means one KV head per mesh row, which is
    architecture-invariant on `(8, 4)` and carries from Wormhole untouched. The
    batch shards over the four columns, so each device holds
    `users_per_column` users of its row's single KV head.
    """

    shape = (_BATCH_SIZE, module.config.n_kv_heads // _MESH_ROWS, _MAX_SEQ_LEN, _HEAD_DIM)
    mapper = ttnn.ShardTensor2dMesh(mesh_device, dims=(None, 0), mesh_shape=_MESH_SHAPE)
    tensors = [
        ttnn.from_torch(
            torch.zeros(shape, dtype=torch.bfloat16),
            device=mesh_device,
            mesh_mapper=mapper,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        for _ in range(2)
    ]
    return KVCacheBinding(tensors[0], tensors[1], owner=object(), mesh_device=mesh_device)


def _to_device_input(x: torch.Tensor, mesh_device: ttnn.MeshDevice) -> ttnn.Tensor:
    """Place an activation: hidden state sharded over the four mesh columns."""

    return ttnn.from_torch(
        x,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, 3), mesh_shape=_MESH_SHAPE),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _compose_output(output: ttnn.Tensor, mesh_device: ttnn.MeshDevice, spec: _ModelSpec) -> torch.Tensor:
    composed = ttnn.to_torch(
        output, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(1, 3), mesh_shape=_MESH_SHAPE)
    )
    return composed[:, :1, :, : spec.dim]


def _report_output_mesh(
    case: str, spec: _ModelSpec, expected: torch.Tensor, output: ttnn.Tensor, mesh_device: ttnn.MeshDevice
) -> None:
    """Report the reduced output per mesh row and per mesh column.

    Two independent checks off one composition:

    * **per mesh row.** After the axis-0 reduction all eight rows hold the same
      value, so a disagreement is the no-op detector for that collective and
      the localizer for a row-local defect.
    * **per mesh column, cross-correlated.** The output is column-sharded on
      `dim`, so mesh column `c` owns hidden channels `c * local_dim ...`. This
      is the one place a per-device channel-order mismatch can be *named*: it
      shows up as every device column correlating with some reference column
      other than its own. That signature is what distinguishes it from bad
      numerics, which correlate badly everywhere.
    """

    composed = ttnn.to_torch(
        output, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(1, 3), mesh_shape=_MESH_SHAPE)
    ).float()
    per_row = [composed[:, row : row + 1, :, : spec.dim] for row in range(_MESH_ROWS)]
    _report_mesh_rows(case, "output", expected, per_row)
    width = spec.local_dim
    _report_column_permutation(
        case,
        "output",
        [expected[..., column * width : (column + 1) * width].float() for column in range(_MESH_COLUMNS)],
        [composed[:, :1, :, column * width : (column + 1) * width] for column in range(_MESH_COLUMNS)],
    )


def _compose_cache(cache: ttnn.Tensor, mesh_device: ttnn.MeshDevice, spec: _ModelSpec) -> torch.Tensor:
    composed = ttnn.to_torch(
        cache, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(1, 0), mesh_shape=_MESH_SHAPE)
    )
    return composed[:_BATCH_SIZE, : spec.n_kv_heads, :, :_HEAD_DIM]


def _assert_cache(
    binding: KVCacheBinding,
    mesh_device: ttnn.MeshDevice,
    spec: _ModelSpec,
    expected_k: torch.Tensor,
    expected_v: torch.Tensor,
    index: Any,
    case: str,
) -> None:
    _assert_pcc(f"{case} K cache", expected_k, _compose_cache(binding.keys, mesh_device, spec)[index])
    _assert_pcc(f"{case} V cache", expected_v, _compose_cache(binding.values, mesh_device, spec)[index])


# =============================================================================
# Staged comparison
# =============================================================================


def _report_qkv_stage(
    case: str,
    mode: str,
    spec: _ModelSpec,
    stages: dict[str, torch.Tensor],
    expected_qkv: torch.Tensor,
    window: slice,
) -> None:
    """Report the column-reduced QKV projection per mesh row and mesh column.

    The staged tensor is `(1, 8, tokens, local_qkv * 4)`: mesh rows on
    dimension 1 and mesh columns on dimension 3. After the axis-1 reduction the
    four column copies of a given mesh row must be identical, so this one
    tensor carries the column replica report, the no-op detector for that
    collective, and the per-mesh-row projection check.

    This is the one staged comparison that is **asserted** rather than only
    printed. It is a plain matmul followed by one reduction, so this file's own
    host reference for it cannot be subtly wrong in the way a re-derived SDPA
    can be, and a QKV failure explains every stage downstream of it. Everything
    after it is printed, so a mistake in this file's reference can only
    annotate a run, never fail an otherwise-good one.
    """

    staged = stages.get(f"{mode}_qkv")
    if staged is None:
        return
    width = spec.local_qkv_size
    for row in range(_MESH_ROWS):
        reference = expected_qkv[:, row * width : (row + 1) * width]
        per_column = [staged[0, row, window, column * width : (column + 1) * width] for column in range(_MESH_COLUMNS)]
        if row == 0:
            _report_column_replicas(case, "qkv(mesh row 0)", reference, per_column)
        _report_stage(case, f"qkv mesh row {row}", reference, per_column[0])
    reference_all = torch.cat([expected_qkv[:, row * width : (row + 1) * width] for row in range(_MESH_ROWS)], dim=-1)
    device_all = torch.cat([staged[0, row, window, :width] for row in range(_MESH_ROWS)], dim=-1)
    _assert_pcc(f"{case} qkv projection", reference_all, device_all)


def _report_wo_stage(
    case: str,
    mode: str,
    spec: _ModelSpec,
    stages: dict[str, torch.Tensor],
    expected_partials: torch.Tensor,
    window: slice,
) -> None:
    """Report the WO projection partials, per mesh row and per mesh column.

    Before the axis-0 reduction each mesh row holds a genuinely *different*
    partial sum, so - unlike the QKV stage - the eight rows are expected to
    disagree with one another and each is checked against its own reference
    partial. Getting this the other way round is how a row-order defect hides:
    eight rows agreeing here would mean the row split of `wo` never happened.
    """

    staged = stages.get(f"{mode}_wo")
    if staged is None:
        return
    width = spec.local_dim
    for row in range(_MESH_ROWS):
        values = [
            _pcc(
                expected_partials[row][:, column * width : (column + 1) * width],
                staged[0, row, window, column * width : (column + 1) * width],
            )
            for column in range(_MESH_COLUMNS)
        ]
        print(
            f"[bh-attn {case}] stage wo mesh row {row}: pcc_per_column={[round(value, 6) for value in values]}",
            flush=True,
        )


def _report_sliced_qkv_stage(
    case: str,
    spec: _ModelSpec,
    stages: dict[str, torch.Tensor],
    expected_qkv: torch.Tensor,
) -> None:
    """Report the column-user slice on its own, between the reduction and the heads.

    **This exists because one reading could not tell two operations apart.** On
    2026-09-17 decode had `qkv` at PCC 0.999662 with the four column replicas
    bit-identical, and the next stage, `q heads`, at `pcc=nan norm_ratio=inf` on
    all eight mesh rows. Between them sat exactly two operations and no
    instrumentation: `GalaxyColumnUserSelector` and
    `nlp_create_qkv_heads_decode`. This stage splits them, which is the
    playbook's own rule that staged instrumentation beats a final-output PCC.

    Read it as a decision procedure:

    * **sliced is finite and correct, heads are `inf`** -> the selector is fine
      and the defect is in `nlp_create_qkv_heads_decode` or the placement it is
      handed. `heads_memcfg` is built by `_subgrid(..., row_wise=False)` while
      the op's factory flattens the grid it is given with
      `corerange_to_cores(..., /*row_wise=*/true)`, so the shard mapping and the
      op's walk are the first thing to compare.
    * **sliced is already `inf`** -> the selector, or the placement its matmul
      writes into, and `nlp_create_qkv_heads_decode` is merely propagating it.
    * **sliced is finite but wrong per user** -> a genuine user-order defect
      rather than uninitialised memory, which is a different bug from the one
      observed.

    The composition is `(8, 4, users_per_column, local_qkv)`: mesh rows on
    dimension 0, mesh columns on dimension 1, and after the slice a column holds
    its **own** users rather than a replica, so mesh column `c` holds physical
    users `c * users_per_column ...` per `Attention2DConfig.batch_offsets`.
    """

    staged = stages.get("decode_qkv_sliced")
    if staged is None:
        return
    width = spec.local_qkv_size
    # Defensive, and for the reason this file states about its other staged
    # reports: a mistake in the reference must be able to *annotate* a run and
    # never to fail an otherwise-good one. This stage's composed shape is
    # predicted rather than observed -- no run has produced it yet -- so a
    # surprise prints the shape and returns instead of raising an IndexError
    # that would redden a node id for a reporting bug.
    expected_shape = (_MESH_ROWS, _MESH_COLUMNS, _USERS_PER_COLUMN, width)
    if staged.ndim != 4 or tuple(staged.shape)[:2] != expected_shape[:2]:
        print(
            f"[bh-attn {case}] stage qkv sliced: SHAPE UNEXPECTED {tuple(staged.shape)}, "
            f"expected {expected_shape} -- reporting skipped, the run is unaffected",
            flush=True,
        )
        return
    for row in range(_MESH_ROWS):
        for column in range(_MESH_COLUMNS):
            offset = column * _USERS_PER_COLUMN
            reference = expected_qkv[offset : offset + _USERS_PER_COLUMN, row * width : (row + 1) * width]
            device = staged[row, column, :_USERS_PER_COLUMN, :width]
            _report_stage(case, f"qkv sliced mesh row {row} column {column}", reference, device)


def _report_head_stages(
    case: str,
    mode: str,
    spec: _ModelSpec,
    stages: dict[str, torch.Tensor],
    expected_q: torch.Tensor,
    expected_k: torch.Tensor,
    window: slice,
) -> None:
    """Report the created heads after Q/K norm - the stage the rotary would see.

    Two different device layouts, because the two modes create heads
    differently and the composer has to follow:

    * **decode** stages `(8, 32, padded_heads, head_dim)` - mesh rows on
      dimension 0, and the four columns' `users_per_column` groups
      concatenated on dimension 1, so dimension 1 is the physical batch in
      order. The head axis is padded to a tile by the create-heads op, which is
      that op's business, so only the real heads are read.
    * **prefill** stages `(4, 8 * local_heads, tokens, head_dim)` - mesh
      columns on dimension 0 (replicas of one another after the axis-1
      reduction) and the row-local heads concatenated on dimension 1, because
      prefill's heads are `[1, local_heads, sequence, head_dim]`.
    """

    for name, expected, local in (("q", expected_q, spec.local_heads), ("k", expected_k, spec.local_kv_heads)):
        staged = stages.get(f"{mode}_{name}_heads")
        if staged is None:
            continue
        if mode == "decode":
            # Where the non-finite values are, not just that the PCC is nan.
            # `nlp_create_qkv_heads_decode` pads the head axis to a tile
            # (`num_q_heads_padded = ceil(num_q_heads / 32) * 32`), so a decode
            # stage is `(8, 32, 32, head_dim)` on Llama with only the first
            # `local` head rows real and the remainder **uninitialised L1**.
            # That makes `norm_ratio=inf` ambiguous between two very different
            # faults, and 2026-09-17 could not tell them apart:
            #
            #   * non-finite in the REAL rows -> the op did not write the heads,
            #     or wrote them somewhere this composition does not look;
            #   * non-finite only in the PADDED rows -> the op is fine and the
            #     composition or the `:local` slice is reading the wrong rows,
            #     which would be a defect in this file rather than in the port.
            #
            # Counted rather than asserted: this is a report, and a report here
            # must never fail an otherwise-good run.
            real = staged[:, :, :local, :]
            padded = staged[:, :, local:, :]
            print(
                f"[bh-attn {case}] stage {name} heads layout: composed={tuple(staged.shape)} "
                f"real_rows=:{local} nonfinite_real={int((~torch.isfinite(real)).sum())}/{real.numel()} "
                f"nonfinite_padded={int((~torch.isfinite(padded)).sum())}/{padded.numel()}",
                flush=True,
            )
        for row in range(_MESH_ROWS):
            reference = expected[:, row * local : (row + 1) * local, :]
            if mode == "decode":
                device = staged[row, :, :local, :]
            else:
                device = staged[0, row * local : (row + 1) * local, window, :].transpose(0, 1)
            _report_stage(case, f"{name} heads mesh row {row}", reference, device)


def _report_attention_stage(
    case: str, mode: str, spec: _ModelSpec, stages: dict[str, torch.Tensor], expected: torch.Tensor, window: slice
) -> None:
    """Report the SDPA output, per mesh row.

    Decode sees the pre-gather output, which composes to `(8, 32,
    padded_heads, head_dim)`: dimension 1 is the physical batch in column
    order, so a user that landed in the wrong mesh column is visible here one
    stage earlier than in the final output.

    Prefill sees the post-concat output, `(1, 8, tokens, local_attention_dim *
    4)`, whose four column copies are replicas.
    """

    staged = stages.get(f"{mode}_attention")
    if staged is None:
        return
    local = spec.local_heads
    for row in range(_MESH_ROWS):
        reference = expected[:, row * local : (row + 1) * local, :]
        if mode == "decode":
            device = staged[row, :, :local, :]
        else:
            device = staged[0, row, window, : spec.local_attention_dim].reshape(-1, local, _HEAD_DIM)
        _report_stage(case, f"sdpa mesh row {row}", reference, device)


# =============================================================================
# Fixtures shared by every node id
# =============================================================================


def _extract_weights(spec: _ModelSpec, reference: HfAttentionWrapper, mesh_device: ttnn.MeshDevice) -> dict[str, Any]:
    """Extract and place the projections. `(8, 4)` makes the split unchanged.

    The row-fused QKV layout Attention2D expects is exactly what the shared
    extractor builds for a mesh row count of eight, and eight mesh rows is an
    architecture-invariant property of a Galaxy mesh, so nothing here is
    Blackhole-specific.
    """

    wqkv, wo, q_norm, k_norm, wqkv_bias = get_attention_weights_from_ref_model(
        reference.attention, num_devices=_MESH_ROWS
    )
    assert wqkv_bias is None, "Attention2D has no QKV bias path"
    assert (q_norm is not None) == spec.qk_norm
    wqkv = wqkv[0, 0].to(torch.bfloat16).contiguous()
    wo = wo[0, 0].to(torch.bfloat16).contiguous()
    return {
        "wqkv": _lazy_weight(wqkv, mesh_device, _mesh_mapper(ttnn.PlacementShard(1), ttnn.PlacementShard(0))),
        "wo": _lazy_weight(wo, mesh_device, _mesh_mapper(ttnn.PlacementShard(0), ttnn.PlacementShard(1))),
        "q_norm": q_norm.to(torch.bfloat16) if q_norm is not None else None,
        "k_norm": k_norm.to(torch.bfloat16) if k_norm is not None else None,
        "wqkv_host": wqkv.float(),
        "wo_host": wo.float(),
        "q_norm_host": q_norm.float() if q_norm is not None else None,
        "k_norm_host": k_norm.float() if k_norm is not None else None,
    }


def _run_prefill(mesh_device: ttnn.MeshDevice, spec: _ModelSpec, sequence_length: int) -> None:
    """One prefill length, invoked twice, with staged reporting on both passes.

    Twice because a single invocation cannot see state a module leaves behind,
    and repeated invocation is the form of that check which applies to
    Attention2D: it does **not** own the residual stream - the caller does, and
    on Blackhole the residual add is unfused - so the multi-layer dropped-
    residual defect class belongs to the decoder suite rather than here. What
    does belong here is that the second call through the same program cache,
    the same weights and the same KV cache is still right.
    """

    torch.manual_seed(17)
    topology = bh_galaxy_topology(mesh_device)
    placements = _decode_placements(topology)
    reference = _reference_attention(spec)
    weights = _extract_weights(spec, reference, mesh_device)
    resources = require_bh_galaxy_prefetch_free_resources(
        mesh_device,
        config=_resources_config(mesh_device, topology, spec, placements, (sequence_length,)),
    )
    collectives = _AttentionCollectives(resources, mesh_device, topology, spec, placements)
    module = binding = None
    try:
        module = _make_module(
            spec,
            mesh_device,
            topology,
            collectives,
            placements,
            weights,
            (sequence_length,),
            resources,
        )
        binding = _make_cache(module, mesh_device)
        module.bind_kv_cache(binding)
        resources.activate("prefill")
        actual = expected = None
        for invocation in range(2):
            case = f"prefill{sequence_length}-{spec.name}-{invocation}"
            x = torch.randn(1, 1, sequence_length, spec.dim, dtype=torch.bfloat16) * 0.05
            expected = _reference_prefill(reference, x)
            expected_k, expected_v = _reference_cache_kv(reference, 0)
            host_x = x[0, 0].float()
            expected_qkv = host_x @ weights["wqkv_host"]
            q_ref, k_ref, v_ref = _split_qkv(spec, expected_qkv)
            q_ref = _rms_normalize(q_ref, weights["q_norm_host"], spec.norm_eps)
            k_ref = _rms_normalize(k_ref, weights["k_norm_host"], spec.norm_eps)
            window = _stage_window(sequence_length)
            attention_ref = _torch_prefill_attention(spec, q_ref, k_ref, v_ref, window)
            partials_ref = _wo_partials(spec, attention_ref, weights["wo_host"])
            collectives.clear_stages()
            tt_x = _to_device_input(x, mesh_device)
            output = None
            try:
                output = module.prefill_forward(tt_x, None, PrefillMetadata(sequence_length, user_ids=(0,)))
                report = f"prefill {case}"
                _report_qkv_stage(report, "prefill", spec, collectives.stages, expected_qkv[window], window)
                _report_head_stages(report, "prefill", spec, collectives.stages, q_ref[window], k_ref[window], window)
                _report_attention_stage(report, "prefill", spec, collectives.stages, attention_ref, window)
                _report_wo_stage(report, "prefill", spec, collectives.stages, partials_ref, window)
                _report_output_mesh(f"prefill {case}", spec, expected, output, mesh_device)
                actual = _compose_output(output, mesh_device, spec)
                _assert_pcc(f"prefill {case} output", expected, actual)
                _assert_cache(
                    binding,
                    mesh_device,
                    spec,
                    expected_k.transpose(0, 1),
                    expected_v.transpose(0, 1),
                    (0, slice(None), slice(0, sequence_length)),
                    f"prefill {case}",
                )
            except BaseException:
                traceback.print_exc()
                raise
            finally:
                if output is not None and not module.output_is_borrowed(output):
                    deallocate_tensor(output)
                deallocate_tensor(tt_x)
        # Recorded, not compared: three PCC passes prove much less than one
        # byte-identical triple, and the comparison across processes belongs to
        # the job script because one pytest node id per process is load-bearing.
        record_module_output(f"attention_2d_bh_prefill{sequence_length}_{spec.name}", expected, actual)
    finally:
        _teardown(resources, module, collectives, binding, reference, weights)


def _teardown(
    resources: Any,
    module: Attention2D | None,
    collectives: _AttentionCollectives,
    binding: KVCacheBinding | None,
    reference: Any,
    weights: dict[str, Any],
) -> None:
    try:
        collectives.cleanup()
        resources.cleanup()
        if module is not None:
            module.close()
            module.release()
            deallocate_module_weights(module, "wqkv", "wo", "prefill_wqkv", "prefill_wo")
        if binding is not None:
            deallocate_tensor(binding.keys)
            deallocate_tensor(binding.values)
    finally:
        del reference, weights
        gc.collect()


# =============================================================================
# Tests, cheapest first
# =============================================================================


@pytest.mark.device
@pytest.mark.blackhole
@pytest.mark.galaxy_bh
@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [BHGLX_DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("spec", _MODEL_SPECS, ids=lambda spec: spec.name)
@torch.no_grad()
def test_attention_2d_bh_galaxy_prefill_128(mesh_device: ttnn.MeshDevice, spec: _ModelSpec):
    """Cheapest real signal: a 128-token prefill, twice, against HuggingFace.

    First because prefill needs no fused collective and no column-user slice:
    the axis-1 QKV reduction and the axis-0 output reduction are both standard
    composite ops, `nlp_create_qkv_heads` takes the whole sequence, and
    `gather_heads` is the identity. If this passes and decode does not, the
    difference is the decode graph and not the mesh, the fabric or the
    projections.
    """

    _run_prefill(mesh_device, spec, 128)


@pytest.mark.device
@pytest.mark.blackhole
@pytest.mark.galaxy_bh
@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [BHGLX_DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("spec", _MODEL_SPECS, ids=lambda spec: spec.name)
@torch.no_grad()
def test_attention_2d_bh_galaxy_prefill_2048(mesh_device: ttnn.MeshDevice, spec: _ModelSpec):
    """A full-`max_seq_len` prefill, which is the only node id that exercises
    the Blackhole SDPA `k_chunk_size` of 256.

    Below 2048 the chunk sizes are 64/64 on both architectures, so the 512 ->
    256 delta is unobservable at 128. It also fills the entire KV cache, so the
    cache assertion covers every position rather than a prefix.
    """

    _run_prefill(mesh_device, spec, _MAX_SEQ_LEN)


@pytest.mark.device
@pytest.mark.blackhole
@pytest.mark.galaxy_bh
@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [BHGLX_DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("spec", _MODEL_SPECS, ids=lambda spec: spec.name)
@torch.no_grad()
def test_attention_2d_bh_galaxy_decode(mesh_device: ttnn.MeshDevice, spec: _ModelSpec):
    """Two decode steps across a tile boundary, against HuggingFace.

    Last, and the most expensive: it is the only node id that needs the
    column-user slice, the user gather and the height-sharded decode
    placements, and the only one whose collectives run on `users_per_column`
    rather than whole-sequence geometry.

    Positions 127 and 128 straddle a tile boundary on purpose - a cache write
    that is right inside a tile and wrong at its edge passes at one position
    and fails at the other.

    **No prefetch-producer variant of this test exists, deliberately.** On
    Wormhole the equivalent `attention_decode_with_active_prefetch` passes its
    setup, its call, its PCC and its own cleanup and still exits non-zero,
    because the test starts the DRAM producer while attention decode is
    precisely the module that must *not* consume the global circular buffer, so
    nothing drains it and `ttnn.close_mesh_device` hangs afterwards. Blackhole
    milestone 1 is prefetcher-free: `require_bh_galaxy_prefetch_free_resources`
    uses `NullPrefetcher2D`, whose `launch_sender` is a no-op, so there is no
    producer to drain and no second variant to write.
    """

    torch.manual_seed(17)
    topology = bh_galaxy_topology(mesh_device)
    placements = _decode_placements(topology)
    reference = _reference_attention(spec)
    weights = _extract_weights(spec, reference, mesh_device)
    resources = require_bh_galaxy_prefetch_free_resources(
        mesh_device,
        config=_resources_config(mesh_device, topology, spec, placements, (128,)),
    )
    collectives = _AttentionCollectives(resources, mesh_device, topology, spec, placements)
    module = binding = None
    try:
        module = _make_module(
            spec,
            mesh_device,
            topology,
            collectives,
            placements,
            weights,
            (128,),
            resources,
        )
        binding = _make_cache(module, mesh_device)
        module.bind_kv_cache(binding)
        # Decode starts mid-cache without a prefill, so the reference history is
        # the same zero-filled span the freshly allocated device cache holds.
        reference.reset_cache_to_zeros(_BATCH_SIZE, spec.n_kv_heads, _DECODE_POSITIONS[0])
        resources.activate("decode")
        actual = expected = None
        for invocation, position in enumerate(_DECODE_POSITIONS):
            case = f"decode-{spec.name}-{invocation}@{position}"
            x = torch.randn(1, 1, _BATCH_SIZE, spec.dim, dtype=torch.bfloat16) * 0.05
            positions = torch.full((_BATCH_SIZE,), position, dtype=torch.long)
            expected = _reference_decode(reference, x, position)
            expected_k, expected_v = _reference_cache_kv(reference, (slice(None), position))
            host_x = x[0, 0].float()
            expected_qkv = host_x @ weights["wqkv_host"]
            q_ref, k_ref, v_ref = _split_qkv(spec, expected_qkv)
            q_ref = _rms_normalize(q_ref, weights["q_norm_host"], spec.norm_eps)
            k_ref = _rms_normalize(k_ref, weights["k_norm_host"], spec.norm_eps)
            history_k, history_v = _reference_cache_kv(reference, slice(None))
            attention_ref = _torch_decode_attention(spec, q_ref, history_k.float(), history_v.float())
            partials_ref = _wo_partials(spec, attention_ref, weights["wo_host"])
            collectives.clear_stages()
            tt_x = _to_device_input(x, mesh_device)
            tt_positions = ttnn.from_torch(
                positions[: module.config.users_per_column].to(torch.int32),
                device=mesh_device,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            output = None
            try:
                output = module.decode_forward(tt_x, None, DecodeMetadata(tt_positions))
                # The whole physical batch, uncapped: 32 users is the entire
                # decode geometry, so there is nothing to window away.
                whole = slice(None)
                report = f"decode {case}"
                _report_qkv_stage(report, "decode", spec, collectives.stages, expected_qkv, whole)
                _report_sliced_qkv_stage(report, spec, collectives.stages, expected_qkv)
                _report_head_stages(report, "decode", spec, collectives.stages, q_ref, k_ref, whole)
                _report_attention_stage(report, "decode", spec, collectives.stages, attention_ref, whole)
                _report_wo_stage(report, "decode", spec, collectives.stages, partials_ref, whole)
                _report_output_mesh(f"decode {case}", spec, expected, output, mesh_device)
                _assert_cache(
                    binding,
                    mesh_device,
                    spec,
                    expected_k,
                    expected_v,
                    (slice(None), slice(None), position),
                    f"decode {case}",
                )
                actual = _compose_output(output, mesh_device, spec)
                # Per-user unconditionally in decode: the 32 users are exactly
                # what the four mesh columns carry, so a user in the wrong
                # column is the defect this localizes and it must be visible on
                # a pass as well as on a failure.
                _report_per_user(f"decode {case}", expected[0, 0], actual[0, 0])
                _assert_pcc(f"decode {case} output", expected, actual)
            except BaseException:
                traceback.print_exc()
                raise
            finally:
                if output is not None and not module.output_is_borrowed(output):
                    deallocate_tensor(output)
                deallocate_tensor(tt_positions)
                deallocate_tensor(tt_x)
        record_module_output(f"attention_2d_bh_decode_{spec.name}", expected, actual)
    finally:
        _teardown(resources, module, collectives, binding, reference, weights)
