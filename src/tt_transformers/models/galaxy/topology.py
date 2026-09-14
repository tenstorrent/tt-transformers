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
    #: place the semaphores they create at program-compile time.
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
# Wormhole Galaxy
# ---------------------------------------------------------------------------

#: The qualified Wormhole Galaxy geometry.
#:
#: Every sequence here is transcribed verbatim from the constants the Milestone A
#: module tests qualified on `wh-glx6u-05`. They are **named, not derived**, and
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
    fabric_links=4,
    fabric_config=ttnn.FabricConfig.FABRIC_1D_RING,
    dispatch_core_axis=ttnn.DispatchCoreAxis.COL,
    # Wormhole Galaxy dispatch sits at column 7 and above, outside the 7-wide
    # compute grid, so nothing inside the grid needs excluding.
    reserved_columns=(),
    sub_device_max_y=None,
)

_TOPOLOGIES = {WORMHOLE_GALAXY_TOPOLOGY.architecture: WORMHOLE_GALAXY_TOPOLOGY}


def supported_galaxy_architectures() -> tuple[Any, ...]:
    return tuple(_TOPOLOGIES)


def galaxy_chip_topology(architecture: Any) -> GalaxyChipTopology:
    """Return the descriptor for one architecture, without a device to check it.

    Fails closed. The mesh contract is an allowlist, and a second architecture
    must *extend* it rather than relax it: an architecture with no descriptor has
    no qualified geometry, and inventing one produces silent misplacement rather
    than an error.
    """

    if architecture not in _TOPOLOGIES:
        supported = ", ".join(str(arch) for arch in _TOPOLOGIES)
        raise ValueError(f"no Galaxy topology for architecture {architecture}; supported: {supported}")
    return _TOPOLOGIES[architecture]


def resolve_galaxy_chip_topology(mesh_device: Any) -> GalaxyChipTopology:
    """Return the descriptor for this mesh, validated against what it reports."""

    return galaxy_chip_topology(mesh_device.arch()).validate_against_device(mesh_device)
