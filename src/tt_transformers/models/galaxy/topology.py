# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Resolved intra-chip geometry and fabric policy for one Galaxy architecture.

The `(8, 4)` mesh shape is architecture-invariant -- Blackhole Galaxy declares
the same `device_topology { dims: [8, 4] }` as Wormhole -- so everything derived
from the mesh survives a second architecture untouched: the weight splits, the
mesh mappers, the row/column sharding, the 8-users-per-column batch layout. What
does *not* survive is everything inside one chip: a different tensix grid, a
different DRAM bank count, a different dispatch column, a different fabric.

This module is where that difference lives, so the 2D modules and the recipes
never name a core coordinate that only one architecture has.

Two rules shape the contents.

**Derive what follows from the grid; name what only measurement knows.** The
compute grid, the DRAM view count and the prefill envelope are read from the
live device. But some boundaries are empirical: the reference port excludes
Blackhole's dispatch column from the worker envelope because folding it in
*regresses prefill warmup*, even though it sits inside
`compute_with_storage_grid_size()`. A purely derived resolver would include it
and be silently slower. Such facts are named constants carrying their
justification, and `reserved_columns` is the field that holds them.

**Order is part of the value, not an implementation detail.** `CoreRangeSet`
preserves constructor sequence, so the same cores built from a `list` and from a
`set` are unequal objects that iterate differently, and the ring kernels depend
on that order. Building a 24-core ring from a set produced *bit-for-bit
identical wrong output across runs* -- stable rather than noisy, which is what
made it look like anything except an ordering bug for weeks. Every sequence here
is a `tuple`, and the host tests assert order rather than membership.

That is also why this descriptor holds plain coordinate tuples rather than
`ttnn.CoreRangeSet` objects. Tuples compare element-wise and in order, are
hashable, survive a frozen dataclass, and can be asserted against a golden table
without depending on pybind equality semantics. The recipes turn them into
`CoreRangeSet`s at the point of use, freshly, exactly as they did before.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import ttnn

Coord = tuple[int, int]
Rect = tuple[int, int, int, int]

#: Fabric links per direction, per Galaxy architecture -- the single source, read
#: by both resolvers below and re-exported through `recipes`.
#:
#: The mesh graph descriptors are the origin: `single_galaxy_mesh_graph_descriptor
#: .textproto` declares `channels { count: 4 }` and the Blackhole equivalent
#: declares `channels { count: 2 }`. `tt_ccl.get_num_links` carries the same
#: budget derived independently from the device name (`("BHGLX", (2, 2))` against
#: `("TG", (4, 4))`); that the two agree is a real invariant, and on Blackhole it
#: is now a **measured** one: on 2026-09-16 a live BHGLX mesh reported
#: `get_num_links` of 2 on both cluster axes against this table's 2, and
#: `ccl_reserved_worker_cores` of 2.
#:
#: The chassis has more ethernet channels than that on some edges -- the control
#: plane warns `4 eth channels, but only 2 routing planes are available` on four
#: of the 32 devices -- which is consistent with 2 rather than in conflict with
#: it: the budget counts routing planes. Do not read the spare channels as room
#: for 4. An over-requested Galaxy `num_links` deadlocks with no traceback.
#:
#: Keyed on `arch()` rather than delegating to `get_num_links`, which needs
#: `get_device_ids()` and a pybind arch probe that the host-mocked meshes every
#: Galaxy geometry test uses cannot answer.
GALAXY_FABRIC_LINKS = {
    ttnn.device.Arch.WORMHOLE_B0: 4,
    ttnn.device.Arch.BLACKHOLE: 2,
}


@dataclass(frozen=True)
class GalaxyCapabilities:
    """Mechanisms one Galaxy architecture actually has.

    Modules assert the capability they need rather than an architecture name, so
    a second architecture is a new record here instead of a new branch at every
    call site.

    **`has_prefetcher` and `has_fused_ccl` are deliberately separate axes.** The
    reference port runs Blackhole decode with the prefetcher *on* and the fused
    galaxy collectives *off* simultaneously -- the ring matmuls must consume the
    prefetched global-CB weights, while `fused_rms_minimal`,
    `llama_rs_create_heads`, `all_gather_concat`, `llama_rs_matmul` and
    `llama_reduce_scatter` use 1D-multicast writers that silently no-op on the
    2D-torus fabric. That is its *tested* configuration, not a fallback.
    Collapsing these two flags into one architecture check would make it
    unexpressible and would cost the deferred prefetcher work a refactor of every
    call site. It is the cheapest thing on this list to get right and the most
    expensive to undo.
    """

    has_prefetcher: bool
    has_ring_matmul: bool
    has_fused_ccl: bool
    has_fused_residual_norm: bool
    has_fused_qk_rotary: bool
    has_distributed_sampling: bool


@dataclass(frozen=True)
class GalaxyChipTopology:
    """One architecture's intra-chip geometry, validated against a live device."""

    architecture: Any
    capabilities: GalaxyCapabilities

    #: Physical tensix grid, read from `compute_with_storage_grid_size()`.
    compute_grid: Coord
    #: DRAM views, read from `dram_grid_size().x`. 12 on Wormhole, 8 on Blackhole.
    dram_views: int

    #: Decode worker sub-device envelope, as `(x0, y0, x1, y1)` rectangles.
    worker_core_ranges: tuple[Rect, ...]
    #: Top-k placement. Must lie inside the worker envelope.
    topk_core_ranges: tuple[Rect, ...]

    #: The gather-in0 ring, in **ring walk order**. `None` when the architecture
    #: has no ring matmul path.
    ring_core_coords: tuple[Coord, ...] | None
    #: Global-CB receivers, in sender-pair order.
    ring_receiver_coords: tuple[Coord, ...] | None
    #: Ring hop cores. Sits on a dummy-receiver row: outside the ring shard grid,
    #: inside the global CB's core set.
    ring_hop_coords: tuple[Coord, ...] | None
    #: Compute grid the ring matmul program config declares.
    ring_matmul_grid: Coord | None

    #: Active prefetch senders. **Empty means prefetcher-free** -- an explicit
    #: marker, not an absent field. A path expressed as "the prefetcher fields do
    #: not exist" cannot later grow one without touching every consumer.
    prefetch_sender_coords: tuple[Coord, ...]
    #: Senders that read nothing. They exist so the global CB's `all_cores()`
    #: covers the complete worker set: the mapping's trailing entries are pure
    #: coverage, and omitting them fails the hard superset check the moment an
    #: unrelated fused matmul's gather-in0 program adds a hop core.
    dummy_sender_coords: tuple[Coord, ...]
    #: Receiver column pairs, one per active sender, as `(start, end)` coords.
    receiver_column_pairs: tuple[tuple[Coord, Coord], ...]
    #: Receiver ranges for the dummy senders, one group per dummy sender.
    dummy_receiver_ranges: tuple[tuple[Rect, ...], ...]

    #: Origin of the distributed-norm input shard grid. The fused-statistics
    #: buffer binds to the first core of this grid, so it is not free.
    norm_origin: Coord
    #: Anchor for `Sampling2D`'s core grids.
    sampling_start_core: Coord
    #: Compute grid the decode SDPA program config declares.
    decode_sdpa_grid: Coord
    #: `k_chunk_size` the **prefill** SDPA program config declares for sequences
    #: at or above 2048. The reference splits exactly this one number by
    #: architecture -- `k_chunk_size=(256 if self.is_blackhole else 512)` in
    #: `qwen_model_config.py`, guarded by `seqlen >= 2048 and chunk_start_idx == 0`
    #: -- and leaves `q_chunk_size` at 256 on both, so the pair is not symmetric
    #: and only the `k` half belongs here. Sequences below 2048 use 64/64 on both
    #: architectures and are not parameterized.
    prefill_sdpa_k_chunk: int

    #: Fabric links per direction. Also the worker cores a decode collective
    #: reserves for itself, one per link.
    fabric_links: int
    #: Device-open fabric configuration. Blackhole Galaxy runs column-axis
    #: collectives that need a 2D torus; the 1D configs throw `IndexError:
    #: map::at` on the cross-column route.
    fabric_config: Any
    dispatch_core_axis: Any

    #: Columns inside the compute grid that are **not** workers, and are not
    #: derivable as such. Carries the dispatch column.
    reserved_columns: tuple[int, ...] = ()
    #: Worker sub-device row cap, distinct from `compute_grid[1]`.
    sub_device_max_y: int | None = None

    #: L1 small region reserved on every mesh open, so generic collectives can
    #: place the semaphores they create at program-compile time -- in main L1
    #: mid-bank, where nothing can be placed across them afterwards.
    #:
    #: 32 kB was sized against Wormhole's 1 393 472 B L1 bank, and main L1
    #: shrinks by this much for *every* Galaxy op, so it is a field rather than
    #: a global.
    #:
    #: **Measured on a Blackhole Galaxy, 2026-09-16**: the region is present,
    #: `L1_SMALL` reports exactly 32 768 B per bank, and main L1 reports
    #: 1 428 608 B -- *larger* than Wormhole's bank, so the carve-out costs
    #: marginally less here than where the number was tuned. 32 kB carries; no
    #: per-architecture value is needed. Re-confirm through
    #: `device_utils.has_l1_small_region()` on a new chassis rather than
    #: assuming this reading does.
    #:
    #: Deliberately a literal rather than an import of
    #: `device_utils.GALAXY_L1_SMALL_SIZE`: that module pulls in `lazy_weight`,
    #: `loguru` and `torch`, and this one's only dependency is `ttnn`'s enums --
    #: which is what lets its whole validation surface be exercised without a
    #: device. `test_topology.py` asserts the two agree, so they cannot drift.
    l1_small_size: int = 32768

    #: Populated by `resolve_galaxy_chip_topology`; see `validate_against_device`.
    validated: bool = field(default=False, compare=False)

    # -- derived views ----------------------------------------------------

    @property
    def worker_coords(self) -> frozenset[Coord]:
        return _rect_coords(self.worker_core_ranges)

    @property
    def grid_coords(self) -> frozenset[Coord]:
        width, height = self.compute_grid
        return frozenset((x, y) for x in range(width) for y in range(height))

    @property
    def ccl_reserved_worker_cores(self) -> int:
        """Worker cores a decode collective needs for itself, one per link."""

        return self.fabric_links

    # -- validation -------------------------------------------------------

    def validate_against_device(self, mesh_device: Any) -> GalaxyChipTopology:
        """Return this descriptor checked against the grid the device reports.

        This is the cheap host-side check that catches placement errors before
        they reach silicon. The reference port hit exactly this class of failure
        on Blackhole -- a ring memory config whose shard grid used core column
        `x=6`, outside the auto-selected compute grid, rejected at runtime with
        *"Tensor shard spec grid ... must lie within compute grid"*. Every
        constraint below is one that would otherwise be discovered on an
        allocated node.
        """

        grid = mesh_device.compute_with_storage_grid_size()
        reported = (int(grid.x), int(grid.y))
        if reported != self.compute_grid:
            raise ValueError(
                f"{self.architecture} topology expects compute grid {self.compute_grid}, device reports {reported}"
            )

        dram_views = int(mesh_device.dram_grid_size().x)
        if dram_views != self.dram_views:
            raise ValueError(
                f"{self.architecture} topology expects {self.dram_views} DRAM views, device reports {dram_views}"
            )

        inside = self.grid_coords
        for name, coords in self._named_core_groups():
            outside = sorted(set(coords) - inside)
            if outside:
                raise ValueError(f"{self.architecture} {name} leaves the {self.compute_grid} compute grid: {outside}")

        workers = self.worker_coords
        if not workers:
            raise ValueError(f"{self.architecture} topology resolved an empty worker envelope")

        overlap = sorted(workers & set(self.prefetch_sender_coords + self.dummy_sender_coords))
        if overlap:
            raise ValueError(f"{self.architecture} prefetch senders overlap the worker envelope: {overlap}")

        reserved = sorted(x for x, _ in workers if x in self.reserved_columns)
        if reserved:
            raise ValueError(
                f"{self.architecture} worker envelope includes reserved column(s) {sorted(set(reserved))}; "
                "these are excluded by measurement, not by derivation, and folding them in regresses "
                "prefill warmup with nothing failing"
            )

        for name, coords in (
            ("top-k grid", _rect_coords(self.topk_core_ranges)),
            ("ring cores", frozenset(self.ring_core_coords or ())),
            ("ring hop cores", frozenset(self.ring_hop_coords or ())),
            ("norm origin", frozenset({self.norm_origin})),
            ("sampling start core", frozenset({self.sampling_start_core})),
        ):
            escaped = sorted(coords - workers)
            if escaped:
                raise ValueError(f"{self.architecture} {name} lies outside the worker envelope: {escaped}")

        if self.sub_device_max_y is not None:
            over = sorted(y for _, y in workers if y > self.sub_device_max_y)
            if over:
                raise ValueError(
                    f"{self.architecture} worker envelope reaches row {max(over)} "
                    f"above the sub-device row cap {self.sub_device_max_y}"
                )

        self._validate_prefetch_mapping()
        return replace(self, validated=True)

    def _named_core_groups(self) -> tuple[tuple[str, tuple[Coord, ...]], ...]:
        groups: list[tuple[str, tuple[Coord, ...]]] = [
            ("worker envelope", tuple(sorted(self.worker_coords))),
            ("top-k grid", tuple(sorted(_rect_coords(self.topk_core_ranges)))),
            ("prefetch senders", self.prefetch_sender_coords),
            ("dummy senders", self.dummy_sender_coords),
            ("norm origin", (self.norm_origin,)),
            ("sampling start core", (self.sampling_start_core,)),
        ]
        for name, coords in (
            ("ring cores", self.ring_core_coords),
            ("ring receivers", self.ring_receiver_coords),
            ("ring hop cores", self.ring_hop_coords),
        ):
            if coords:
                groups.append((name, coords))
        pairs = tuple(coord for pair in self.receiver_column_pairs for coord in pair)
        if pairs:
            groups.append(("receiver column pairs", pairs))
        dummies = tuple(sorted(_rect_coords(tuple(r for group in self.dummy_receiver_ranges for r in group))))
        if dummies:
            groups.append(("dummy receivers", dummies))
        return tuple(groups)

    def _validate_prefetch_mapping(self) -> None:
        """Check the global CB's coverage property, which fails hard and late.

        The mapping's sender list and receiver list are zipped, so unequal
        lengths silently truncate. And the union of every receiver group must
        cover the complete worker set: a minimal mapping holding only the active
        senders passes every module test and then fails a hard superset check --
        *"Specified cores are not contained in associated GlobalCircularBuffer"*
        -- the first time an unrelated fused matmul's gather-in0 program adds a
        hop core. On Wormhole the failing core was `(3, 6)`, which appears only
        in dummy mapping entry 16.
        """

        if not self.prefetch_sender_coords:
            return

        senders = len(self.prefetch_sender_coords) + len(self.dummy_sender_coords)
        receivers = len(self.receiver_column_pairs) + len(self.dummy_receiver_ranges)
        if senders != receivers:
            raise ValueError(
                f"{self.architecture} prefetch mapping has {senders} senders and {receivers} receiver groups"
            )

        covered = set()
        for start, end in self.receiver_column_pairs:
            covered |= _rect_coords(((*start, *end),))
        for group in self.dummy_receiver_ranges:
            covered |= _rect_coords(group)
        for hop in self.ring_hop_coords or ():
            if hop not in covered:
                raise ValueError(
                    f"{self.architecture} ring hop core {hop} is not covered by the global CB's receiver mapping; "
                    "a gather-in0 program that reaches it fails the GlobalCircularBuffer superset check"
                )

        missing = sorted(self.worker_coords - covered)
        if missing:
            raise ValueError(
                f"{self.architecture} prefetch receiver mapping does not cover worker core(s) {missing}; "
                "trailing dummy entries exist precisely to complete this coverage"
            )

        # Senders and receivers partition the whole tensix grid. This is the
        # property that makes the 8 dummy entries look redundant and be load
        # bearing, so assert it rather than leave it as folklore: if a future
        # edit drops an entry, the coverage check above may still pass while the
        # global CB's `all_cores()` silently stops describing the full chip.
        senders_and_receivers = covered | set(self.prefetch_sender_coords) | set(self.dummy_sender_coords)
        uncovered = sorted(self.grid_coords - senders_and_receivers)
        if uncovered:
            raise ValueError(
                f"{self.architecture} prefetch senders and receivers do not cover the "
                f"{self.compute_grid} grid; missing {uncovered}"
            )


def _rect_coords(rects: tuple[Rect, ...]) -> frozenset[Coord]:
    return frozenset((x, y) for x0, y0, x1, y1 in rects for x in range(x0, x1 + 1) for y in range(y0, y1 + 1))


# ---------------------------------------------------------------------------
# Worker selection under `reduce_scatter`'s envelope-origin offset
# ---------------------------------------------------------------------------

#: Worker counts per direction `ttnn`'s reduce-scatter family will choose from.
#:
#: Read from `reduce_scatter_default_workers` in
#: `ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_common/
#: reduce_scatter_program_utils.cpp`, whose own comment attributes the ladder to
#: a perf sweep (`test_reduce_scatter_hyperparameter_sweep_perf_galaxy.py`).
#: Staying on it means this repository never asks for a worker count the op's own
#: tuning never measured; it only ever asks for a *smaller* one.
REDUCE_SCATTER_WORKER_LADDER = (8, 4, 2, 1)


def ccl_offset_placeable_worker_cores(worker_core_ranges: tuple[Rect, ...]) -> int:
    """Return how many workers a CCL that offsets by the envelope origin can place.

    **This exists because `reduce_scatter` is the one collective that offsets its
    worker selection, and it offsets by the worker sub-device's own origin.**
    `ttnn::ccl::choose_worker_cores` defaults `core_grid_offset` to `(0, 0)` and
    every `all_gather` factory passes that explicitly -- but
    `reduce_scatter_program_factory.cpp` passes
    `worker_cores(TENSIX, sub_device_id).bounding_box().start_coord`, commented
    *"first core in the subdevice is our offset"*. The selection walk already
    starts inside the sub-device, so adding the origin counts it twice, and
    `ccl_common.cpp` flags the consequence itself: *"the loops above shift each
    candidate by core_grid_offset without re-checking it, so an offset near the
    grid edge can push cores off the worker grid entirely"*. Past the edge it is
    a `TT_FATAL`, not a hang:

        Core grid offset 1-0 pushed 1 of the 12 selected worker cores
        (first: 11-0) off the worker grid; kernels cannot be placed there.

    **Measured on a Blackhole Galaxy:** refused exactly there, on a run whose
    `all_gather` on the same axis and the same envelope passed -- which is the
    asymmetry this function encodes.

    So it is architecture-neutral in form and consequential only where the
    envelope origin is not `(0, 0)`. Wormhole's prefill envelope starts at the
    grid origin and is unaffected; Blackhole's starts at the first worker column,
    one in from the grid origin, and loses its last column to the shift.

    The walk is reproduced rather than approximated -- per range, row-major, in
    descriptor order -- because the selection takes the first N candidates in
    that order and then checks all of them. A count is therefore placeable iff it
    does not exceed the longest all-placeable prefix, which is what this returns.
    Order is part of the value here for the same reason it is everywhere else in
    this module.
    """

    if not worker_core_ranges:
        return 0
    workers = _rect_coords(worker_core_ranges)
    offset_x = min(x0 for x0, _, _, _ in worker_core_ranges)
    offset_y = min(y0 for _, y0, _, _ in worker_core_ranges)
    placeable = 0
    for x0, y0, x1, y1 in worker_core_ranges:
        for y in range(y0, y1 + 1):
            for x in range(x0, x1 + 1):
                if (x + offset_x, y + offset_y) not in workers:
                    return placeable
                placeable += 1
    return placeable


def reduce_scatter_worker_cores(workers_per_link: int, num_links: int) -> int:
    """Return the cores `reduce_scatter` places for one worker count.

    From `reduce_scatter_core_count_per_link`:
    `num_directions * (num_mux_cores + num_workers_per_direction)`, times the
    links. Two facts make the arithmetic worth writing down rather than
    estimating:

    * the `num_workers_per_link` keyword is **per direction**, and there are
      always two directions, so it costs twice what its name suggests;
    * a mux core is added per direction *unless* the worker count is 1, where the
      factory drops it (`if (num_workers_per_direction == 1) {
      num_mux_cores_per_direction_per_link = 0; }`).

    So on a two-link mesh the ladder is 4, 12, 20 and 36 cores -- not 4, 8, 16,
    32. An estimate that missed the mux or the doubling would predict that 4
    workers fit in 9 cores, and it would be wrong by a factor of five.
    """

    if workers_per_link < 1:
        raise ValueError(f"workers_per_link must be positive, got {workers_per_link}")
    if num_links < 1:
        raise ValueError(f"num_links must be positive, got {num_links}")
    directions = 2
    mux_cores_per_direction = 0 if workers_per_link == 1 else 1
    return num_links * directions * (mux_cores_per_direction + workers_per_link)


def reduce_scatter_workers_per_link(worker_core_ranges: tuple[Rect, ...], num_links: int) -> int | None:
    """Return the `num_workers_per_link` `ttnn.reduce_scatter` can place here.

    Pass the result to the op. Leaving the keyword out instead lets it size its
    own selection from the data, which is what refused on Blackhole: 2 workers
    per direction on 2 links is 12 cores, and the shifted envelope holds 9.

    **`None` means "do not constrain it", and it is the answer for every
    envelope that starts at the grid origin** -- which is Wormhole's. With no
    shift, every candidate the walk reaches is inside the envelope whatever the
    count, so the op's own data-sized choice is placeable by construction and
    overriding it would substitute this repository's guess for a perf sweep's
    answer. `None` reaches the op as an absent argument, so a call site can ask
    unconditionally and the qualified Wormhole path does not move.

    **Takes the ranges rather than a descriptor, deliberately.** The offset is
    the *loaded sub-device's* origin, and a mode whose sub-device is not the
    descriptor's decode envelope must ask about the sub-device it actually
    loaded. That is both architectures: prefill loads the full compute grid on
    each, so it gets `None` here and only decode is constrained. Answering from
    a descriptor would pin prefill to decode's count for a shift that does not
    happen to it.

    Fails closed when the envelope shifts and nothing on the ladder fits. That
    is not hypothetical and not a Blackhole case: Wormhole's *decode* envelope is
    split by the prefetch sender column, so the shift walks straight into it and
    leaves room for two cores. `ttnn.reduce_scatter` is already documented as
    unusable under that partition for exactly that reason, and this function
    reaches the same conclusion from the geometry alone rather than from the
    recorded error.
    """

    if not worker_core_ranges:
        raise ValueError("cannot size a reduce_scatter worker selection for an empty worker envelope")
    if min(x0 for x0, _, _, _ in worker_core_ranges) == 0 and min(y0 for _, y0, _, _ in worker_core_ranges) == 0:
        return None
    placeable = ccl_offset_placeable_worker_cores(worker_core_ranges)
    for workers_per_link in REDUCE_SCATTER_WORKER_LADDER:
        if reduce_scatter_worker_cores(workers_per_link, num_links) <= placeable:
            return workers_per_link
    smallest = reduce_scatter_worker_cores(REDUCE_SCATTER_WORKER_LADDER[-1], num_links)
    raise ValueError(
        f"no reduce_scatter worker count fits the worker envelope {worker_core_ranges} on {num_links} links: "
        f"the smallest selection is {smallest} cores and the envelope places {placeable} once "
        "reduce_scatter's factory shifts the walk by the envelope origin. Use the persistent "
        "`reduce_scatter_minimal_async` form, which takes its own worker count, or a partition whose "
        "origin is the grid origin."
    )


# ---------------------------------------------------------------------------
# Wormhole Galaxy
# ---------------------------------------------------------------------------

#: The qualified Wormhole Galaxy geometry.
#:
#: Every sequence here is transcribed verbatim from the constants the module
#: tests qualified on real Galaxy hardware. They are **named, not derived**, and
#: that is deliberate: the ring order is a physical walk of the chip, not a
#: comprehension, and the reference port's own Wormhole tables are hand-written
#: 24- and 32-entry literals for the same reason. `test_topology.py` pins every
#: one of them element-wise and in order, so this file cannot drift from the
#: geometry that was qualified.
WORMHOLE_GALAXY_TOPOLOGY = GalaxyChipTopology(
    architecture=ttnn.device.Arch.WORMHOLE_B0,
    capabilities=GalaxyCapabilities(
        has_prefetcher=True,
        has_ring_matmul=True,
        has_fused_ccl=True,
        has_fused_residual_norm=True,
        has_fused_qk_rotary=True,
        has_distributed_sampling=True,
    ),
    compute_grid=(7, 10),
    dram_views=12,
    worker_core_ranges=((1, 0, 3, 9), (5, 0, 6, 9)),
    topk_core_ranges=((1, 0, 3, 9),),
    ring_core_coords=(
        (6, 6),
        (6, 7),
        (6, 9),
        (6, 0),
        (6, 1),
        (6, 2),
        (6, 4),
        (6, 5),
        (5, 5),
        (5, 6),
        (5, 7),
        (5, 9),
        (5, 0),
        (5, 1),
        (5, 2),
        (5, 4),
        (1, 4),
        (1, 5),
        (1, 9),
        (1, 0),
        (2, 0),
        (2, 4),
        (2, 5),
        (2, 9),
    ),
    ring_receiver_coords=(
        (1, 9),
        (2, 9),
        (1, 0),
        (2, 0),
        (1, 4),
        (2, 4),
        (1, 5),
        (2, 5),
        (5, 0),
        (6, 0),
        (5, 9),
        (6, 9),
        (5, 1),
        (6, 1),
        (5, 7),
        (6, 7),
        (5, 6),
        (6, 6),
        (5, 2),
        (6, 2),
        (5, 4),
        (6, 4),
        (5, 5),
        (6, 5),
    ),
    ring_hop_coords=((3, 6),),
    ring_matmul_grid=(8, 3),
    prefetch_sender_coords=(
        (0, 9),
        (0, 0),
        (0, 4),
        (0, 5),
        (4, 0),
        (4, 9),
        (4, 1),
        (4, 7),
        (4, 6),
        (4, 2),
        (4, 4),
        (4, 5),
    ),
    dummy_sender_coords=((0, 1), (0, 2), (0, 3), (0, 6), (0, 7), (0, 8), (4, 3), (4, 8)),
    receiver_column_pairs=tuple(((1, y), (2, y)) for y in (9, 0, 4, 5))
    + tuple(((5, y), (6, y)) for y in (0, 9, 1, 7, 6, 2, 4, 5)),
    dummy_receiver_ranges=(
        ((3, 0, 3, 0), (1, 1, 3, 1)),
        ((1, 2, 3, 2),),
        ((1, 3, 3, 3), (3, 4, 3, 4)),
        ((3, 5, 3, 5), (1, 6, 3, 6)),
        ((1, 7, 3, 7),),
        ((1, 8, 3, 8), (3, 9, 3, 9)),
        ((5, 3, 6, 3),),
        ((5, 8, 6, 8),),
    ),
    norm_origin=(2, 0),
    sampling_start_core=(1, 0),
    decode_sdpa_grid=(8, 4),
    prefill_sdpa_k_chunk=512,
    fabric_links=GALAXY_FABRIC_LINKS[ttnn.device.Arch.WORMHOLE_B0],
    fabric_config=ttnn.FabricConfig.FABRIC_1D_RING,
    dispatch_core_axis=ttnn.DispatchCoreAxis.COL,
    # Wormhole Galaxy dispatch sits at column 7 and above, outside the 7-wide
    # compute grid, so nothing inside the grid needs excluding.
    reserved_columns=(),
    sub_device_max_y=None,
)


def _resolve_wormhole(compute_grid: Coord, dram_views: int) -> GalaxyChipTopology:
    """Return the Wormhole descriptor.

    Wormhole Galaxy's `galaxy: col:` core-descriptor key is a fixed `7 x 10`, and
    every core table above was qualified against exactly that grid, so this
    refuses anything else rather than reinterpreting hand-measured coordinates
    against a shape they were never checked on.
    """

    if compute_grid != WORMHOLE_GALAXY_TOPOLOGY.compute_grid:
        raise ValueError(
            f"Wormhole Galaxy expects compute grid {WORMHOLE_GALAXY_TOPOLOGY.compute_grid}, "
            f"device reports {compute_grid}; the qualified core tables are specific to it"
        )
    return replace(WORMHOLE_GALAXY_TOPOLOGY, dram_views=dram_views)


# --------------------------------------------------------------------------
# Blackhole Galaxy -- milestone 1, prefetcher-free
# --------------------------------------------------------------------------

#: Lowest worker column on Blackhole. **This is the milestone-1 open question**,
#: and it is named here rather than inferred so that settling it is a one-line
#: change with a test attached.
#:
#: The reference port's worker envelope is `cols 1..10`, and column 0 is excluded
#: *because that is where its prefetcher senders live*. Milestone 1 has no
#: senders, so there is no such reason, and the prefetcher-free envelope is
#: plausibly `cols 0..10` -- 10 more cores. That is an inference, not a reading:
#: the reference has no prefetcher-free Blackhole worker range to copy.
#:
#: 1 is the conservative choice, and conservative is right here because the
#: failure is asymmetric. Too few worker cores costs throughput and nothing else.
#: Too many puts tensors on a column that is reserved for a reason nobody has
#: written down, and every failure in that class is silent.
BLACKHOLE_FIRST_WORKER_COLUMN = 1

#: Worker sub-device row cap, or `None` for the full grid height.
#:
#: The reference sets `sub_core_max_y = 7` on Blackhole unconditionally, but its
#: recorded rationale is *"to match the 24-core ring geometry"* -- 8 rows x 3
#: columns. Milestone 1 has no ring, so there is no row extent to match. Two
#: facts corroborate treating the cap as prefetcher baggage: the reference writes
#: it as `7 if is_blackhole else 9` rather than gating it on `use_prefetcher`,
#: and its own Llama path does not apply it at all. Worth 12 worker cores, and
#: still **[measure]**.
BLACKHOLE_SUB_DEVICE_MAX_Y: int | None = None


def _resolve_blackhole(compute_grid: Coord, dram_views: int) -> GalaxyChipTopology:
    """Return the Blackhole descriptor for whatever grid the device reports.

    Unlike Wormhole, the grid is **derived**, because Blackhole harvesting is
    per-part: the reference's chassis measures `12 x 10` (the 1x-harvested key),
    but `13 x 10` unharvested and `11 x 10` are both shapes a real part can
    present. Everything positional is expressed against the reported width so a
    differently-harvested board resolves rather than failing.

    The one thing that is **not** derived is the dispatch column. It sits inside
    `compute_with_storage_grid_size()`, so a purely derived envelope would fold
    it into the workers -- and the reference records that doing so *"regresses
    prefill warmup"* with nothing raising. It is excluded by measurement.
    """

    width, height = compute_grid
    dispatch_column = width - 1
    last_worker_column = dispatch_column - 1
    if last_worker_column < BLACKHOLE_FIRST_WORKER_COLUMN:
        raise ValueError(f"Blackhole Galaxy compute grid {compute_grid} is too narrow to host a worker envelope")

    max_y = height - 1 if BLACKHOLE_SUB_DEVICE_MAX_Y is None else min(BLACKHOLE_SUB_DEVICE_MAX_Y, height - 1)
    workers = ((BLACKHOLE_FIRST_WORKER_COLUMN, 0, last_worker_column, max_y),)

    return GalaxyChipTopology(
        architecture=ttnn.device.Arch.BLACKHOLE,
        # Milestone 1 is the prefetcher-free path: it matches the reference's own
        # Blackhole default, which is where external runners are pointed, and it
        # is the smaller port. Every capability below is False for a recorded
        # reason, not as a placeholder. Turning one on is a deliberate, testable
        # change.
        capabilities=GalaxyCapabilities(
            # The 12-DRAM-view, Wormhole-grid global-CB mechanism. Deferred.
            has_prefetcher=False,
            # The gather-in0 ring's memory configs are not placeable here: the
            # ring memcfg's shard grid falls outside the auto-selected 1D matmul
            # compute grid, and feeding a ring-sharded all-gather output into the
            # interleaved W2 matmul mismatches per-device channel order and
            # yields MLP decode PCC ~ 0 *silently*.
            has_ring_matmul=False,
            # `fused_rms_minimal`, `llama_rs_create_heads`, `all_gather_concat`,
            # `llama_rs_matmul` and `llama_reduce_scatter` use 1D-multicast
            # writers that **no-op** on the 2D-torus fabric. They move no data
            # and raise nothing, so the collective appears to run.
            has_fused_ccl=False,
            # The non-fused distributed RMSNorm does not write the residual sum
            # back in place, so relying on the fused path silently drops each
            # layer's `ff_out` from the residual stream -- visible only across
            # more than one layer.
            has_fused_residual_norm=False,
            # The Blackhole fallback is the non-fused rotary pair.
            has_fused_qk_rotary=False,
            # `ttnn.sampling`'s pipeline is unavailable; greedy sampling routes
            # through all-gather plus `ttnn.argmax`, which also changes the
            # return shape callers see.
            has_distributed_sampling=False,
        ),
        compute_grid=compute_grid,
        # 8 on Blackhole against Wormhole's 12. Sender count and shard widths
        # follow from it.
        dram_views=dram_views,
        worker_core_ranges=workers,
        # Mirrors Wormhole's three-column top-k. The reference moves top-k to
        # cols 4-10 on Blackhole, but only to clear the *resident global CB* on
        # receiver columns 1-3, which a prefetcher-free path does not have.
        topk_core_ranges=((BLACKHOLE_FIRST_WORKER_COLUMN, 0, min(3, last_worker_column), max_y),),
        # No ring matmul path: `has_ring_matmul` is False, and `None` makes any
        # consumer that asks for one fail loudly instead of receiving Wormhole's
        # coordinates. The deferred work restores these, it does not invent them.
        ring_core_coords=None,
        ring_receiver_coords=None,
        ring_hop_coords=None,
        ring_matmul_grid=None,
        # Empty is the explicit prefetcher-free marker, not an absent field.
        prefetch_sender_coords=(),
        dummy_sender_coords=(),
        receiver_column_pairs=(),
        dummy_receiver_ranges=(),
        norm_origin=(2, 0),
        sampling_start_core=(BLACKHOLE_FIRST_WORKER_COLUMN, 0),
        decode_sdpa_grid=(8, 4),
        # 256, not Wormhole's 512, for sequences at or above 2048. This is the
        # reference's own architecture split and the only chunk size it splits;
        # `q_chunk_size` stays 256 on both.
        prefill_sdpa_k_chunk=256,
        # Two, not four: the mesh graph descriptor declares `channels { count: 2 }`,
        # and the ring/line CCLs index ethernet channels by link, so 4 overruns
        # the available channels and deadlocks.
        fabric_links=GALAXY_FABRIC_LINKS[ttnn.device.Arch.BLACKHOLE],
        # Column-axis (cluster_axis=1) collectives run on device here and need a
        # 2D torus. `FABRIC_1D` and `FABRIC_1D_RING` throw `IndexError: map::at`
        # on the cross-column route. This is a device-open parameter, so it has
        # to be known before the mesh is opened.
        fabric_config=ttnn.FabricConfig.FABRIC_2D_TORUS_XY,
        dispatch_core_axis=ttnn.DispatchCoreAxis.COL,
        reserved_columns=(dispatch_column,),
        sub_device_max_y=BLACKHOLE_SUB_DEVICE_MAX_Y,
    )


_RESOLVERS = {
    ttnn.device.Arch.WORMHOLE_B0: _resolve_wormhole,
    ttnn.device.Arch.BLACKHOLE: _resolve_blackhole,
}

#: The Blackhole chassis shape the reference port measured, and the one this port
#: expects to deploy on. The others resolve too; this is the default when no
#: device is at hand.
#:
#: **Confirmed on silicon**: a deployed chassis reports
#: `compute_with_storage_grid_size()` of 12 x 10 and `dram_grid_size().x` of 8,
#: so this chassis *is* the one the reference characterized -- which is why the
#: other reference-sourced numbers in this file keep their weight instead of
#: becoming suspect. Grid *uniformity across the 32 devices* remains unchecked:
#: `MeshDevice` exposes no `get_devices()` on ttnn 0.77.0, so per-device
#: geometry is unreadable rather than merely unmeasured, and the descriptor
#: still resolves one grid for the whole mesh.
BLACKHOLE_GALAXY_COMPUTE_GRID = (12, 10)
BLACKHOLE_GALAXY_DRAM_VIEWS = 8


def supported_galaxy_architectures() -> tuple[Any, ...]:
    return tuple(_RESOLVERS)


def galaxy_chip_topology(architecture: Any, compute_grid: Coord | None = None, dram_views: int | None = None):
    """Return the descriptor for one architecture, without a device to check it.

    Fails closed. The mesh contract is an allowlist, and a second architecture
    must *extend* it rather than relax it: an architecture with no descriptor has
    no qualified geometry, and inventing one produces silent misplacement rather
    than an error.
    """

    if architecture not in _RESOLVERS:
        supported = ", ".join(str(arch) for arch in _RESOLVERS)
        raise ValueError(f"no Galaxy topology for architecture {architecture}; supported: {supported}")
    if architecture == ttnn.device.Arch.BLACKHOLE:
        compute_grid = compute_grid or BLACKHOLE_GALAXY_COMPUTE_GRID
        dram_views = BLACKHOLE_GALAXY_DRAM_VIEWS if dram_views is None else dram_views
    else:
        compute_grid = compute_grid or WORMHOLE_GALAXY_TOPOLOGY.compute_grid
        dram_views = WORMHOLE_GALAXY_TOPOLOGY.dram_views if dram_views is None else dram_views
    return _RESOLVERS[architecture](compute_grid, dram_views)


def resolve_galaxy_chip_topology(mesh_device: Any) -> GalaxyChipTopology:
    """Return the descriptor for this mesh, built and checked against what it reports."""

    architecture = mesh_device.arch()
    if architecture not in _RESOLVERS:
        supported = ", ".join(str(arch) for arch in _RESOLVERS)
        raise ValueError(f"no Galaxy topology for architecture {architecture}; supported: {supported}")
    grid = mesh_device.compute_with_storage_grid_size()
    topology = _RESOLVERS[architecture]((int(grid.x), int(grid.y)), int(mesh_device.dram_grid_size().x))
    return topology.validate_against_device(mesh_device)


def galaxy_device_params(architecture: Any) -> dict[str, Any]:
    """Return the `open_mesh_device` parameters one Galaxy architecture needs.

    The fabric configuration is chosen **before** the mesh opens, which is why it
    lives on the descriptor rather than being probed afterwards. Opening a
    Blackhole Galaxy with `FABRIC_1D_RING` -- the Wormhole value, and until now
    the only one this repository had -- throws `IndexError: map::at` on the first
    cross-column collective.
    """

    topology = galaxy_chip_topology(architecture)
    return {
        "dispatch_core_axis": topology.dispatch_core_axis,
        "fabric_config": topology.fabric_config,
        "l1_small_size": topology.l1_small_size,
    }
