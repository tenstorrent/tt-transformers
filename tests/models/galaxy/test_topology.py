# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Host-only contracts for `GalaxyChipTopology`.

The Wormhole descriptor replaced module-level literals in a hardware-qualified
path that this repository cannot currently re-run, so the golden tables below
are the regression gate. They are transcribed from the constants as they stood
at the commit that introduced the descriptor, and they are compared **in order**:
`CoreRangeSet` preserves constructor sequence, the ring kernels depend on that
sequence, and a ring built in the wrong order produced bit-for-bit identical
*wrong* output across runs -- stable rather than noisy, which is what disguised
it as anything but an ordering bug.

If a change here is deliberate, the ring geometry moved, and every constraint in
`recipes.py` that is stated in terms of ring size has to be re-walked.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import ttnn

from tt_transformers.models.galaxy import recipes
from tt_transformers.models.galaxy.prefetch import galaxy_sender_receiver_mapping
from tt_transformers.models.galaxy.topology import (
    WORMHOLE_GALAXY_TOPOLOGY,
    galaxy_chip_topology,
    galaxy_device_params,
    resolve_galaxy_chip_topology,
    supported_galaxy_architectures,
)

# --- golden tables, as they stood before the descriptor existed -------------

GOLDEN_RING_CORES = (
    (6, 6), (6, 7), (6, 9), (6, 0), (6, 1), (6, 2), (6, 4), (6, 5),
    (5, 5), (5, 6), (5, 7), (5, 9), (5, 0), (5, 1), (5, 2), (5, 4),
    (1, 4), (1, 5), (1, 9), (1, 0), (2, 0), (2, 4), (2, 5), (2, 9),
)  # fmt: skip
GOLDEN_RING_RECEIVERS = (
    (1, 9), (2, 9), (1, 0), (2, 0), (1, 4), (2, 4), (1, 5), (2, 5),
    (5, 0), (6, 0), (5, 9), (6, 9), (5, 1), (6, 1), (5, 7), (6, 7),
    (5, 6), (6, 6), (5, 2), (6, 2), (5, 4), (6, 4), (5, 5), (6, 5),
)  # fmt: skip
GOLDEN_HOP_CORES = ((3, 6),)
GOLDEN_WORKER_RANGES = ((1, 0, 3, 9), (5, 0, 6, 9))
GOLDEN_TOPK_RANGES = ((1, 0, 3, 9),)
GOLDEN_PREFETCH_SENDERS = (
    (0, 9), (0, 0), (0, 4), (0, 5), (4, 0), (4, 9),
    (4, 1), (4, 7), (4, 6), (4, 2), (4, 4), (4, 5),
)  # fmt: skip
GOLDEN_DUMMY_SENDERS = ((0, 1), (0, 2), (0, 3), (0, 6), (0, 7), (0, 8), (4, 3), (4, 8))
GOLDEN_RECEIVER_PAIRS = tuple(((1, y), (2, y)) for y in (9, 0, 4, 5)) + tuple(
    ((5, y), (6, y)) for y in (0, 9, 1, 7, 6, 2, 4, 5)
)
GOLDEN_DUMMY_RECEIVER_RANGES = (
    ((3, 0, 3, 0), (1, 1, 3, 1)),
    ((1, 2, 3, 2),),
    ((1, 3, 3, 3), (3, 4, 3, 4)),
    ((3, 5, 3, 5), (1, 6, 3, 6)),
    ((1, 7, 3, 7),),
    ((1, 8, 3, 8), (3, 9, 3, 9)),
    ((5, 3, 6, 3),),
    ((5, 8, 6, 8),),
)
GOLDEN_NORM_ORIGIN = (2, 0)
GOLDEN_SAMPLING_START = (1, 0)
GOLDEN_COMPUTE_GRID = (7, 10)
GOLDEN_DECODE_SDPA_GRID = (8, 4)
GOLDEN_RING_MATMUL_GRID = (8, 3)
GOLDEN_DRAM_VIEWS = 12


def _mesh(*, arch=ttnn.device.Arch.WORMHOLE_B0, grid=GOLDEN_COMPUTE_GRID, dram_width=GOLDEN_DRAM_VIEWS, devices=32):
    mesh = MagicMock(spec=ttnn.MeshDevice)
    mesh.shape = recipes.GALAXY_MESH_SHAPE
    mesh.get_num_devices.return_value = devices
    mesh.arch.return_value = arch
    mesh.dram_grid_size.return_value = SimpleNamespace(x=dram_width, y=1)
    mesh.compute_with_storage_grid_size.return_value = ttnn.CoreCoord(grid[0], grid[1])
    return mesh


@pytest.mark.host
@pytest.mark.model
def test_wormhole_descriptor_reproduces_the_qualified_tables_in_order():
    """Every core sequence matches the pre-descriptor literal, element for element."""

    topology = WORMHOLE_GALAXY_TOPOLOGY

    assert topology.ring_core_coords == GOLDEN_RING_CORES
    assert topology.ring_receiver_coords == GOLDEN_RING_RECEIVERS
    assert topology.ring_hop_coords == GOLDEN_HOP_CORES
    assert topology.worker_core_ranges == GOLDEN_WORKER_RANGES
    assert topology.topk_core_ranges == GOLDEN_TOPK_RANGES
    assert topology.prefetch_sender_coords == GOLDEN_PREFETCH_SENDERS
    assert topology.dummy_sender_coords == GOLDEN_DUMMY_SENDERS
    assert topology.receiver_column_pairs == GOLDEN_RECEIVER_PAIRS
    assert topology.dummy_receiver_ranges == GOLDEN_DUMMY_RECEIVER_RANGES
    assert topology.norm_origin == GOLDEN_NORM_ORIGIN
    assert topology.sampling_start_core == GOLDEN_SAMPLING_START
    assert topology.compute_grid == GOLDEN_COMPUTE_GRID
    assert topology.decode_sdpa_grid == GOLDEN_DECODE_SDPA_GRID
    assert topology.ring_matmul_grid == GOLDEN_RING_MATMUL_GRID
    assert topology.dram_views == GOLDEN_DRAM_VIEWS
    assert topology.fabric_links == 4
    assert len(topology.ring_core_coords) == recipes.RING_CORE_COUNT


@pytest.mark.host
@pytest.mark.model
def test_recipe_helpers_still_build_the_qualified_core_sets():
    """The no-argument helpers keep their Wormhole answers after the refactor.

    Callers were deliberately not moved, so this checks the seam itself: the
    helpers now read the descriptor, and must still produce the `CoreRangeSet`
    the qualified recipes were built against -- including iteration order, which
    the ring kernels consume.
    """

    # `CoreRangeSet` equality is *order-sensitive* -- that is the whole point.
    # The same cores built from a `list` and from a `set` compare unequal and
    # iterate differently, so comparing against a set rebuilt from the golden
    # table in golden order pins both membership and sequence in one assertion.
    assert recipes.ring_cores() == recipes.core_points(GOLDEN_RING_CORES)
    assert recipes.ring_receiver_cores() == recipes.core_points(GOLDEN_RING_RECEIVERS)
    assert recipes.ring_hop_cores() == recipes.core_points(GOLDEN_HOP_CORES)
    assert recipes.worker_cores() == recipes.core_ranges(*GOLDEN_WORKER_RANGES)
    assert recipes.topk_cores() == recipes.core_ranges(*GOLDEN_TOPK_RANGES)
    assert tuple((core.x, core.y) for core in recipes.prefetch_sender_cores()) == GOLDEN_PREFETCH_SENDERS

    assert recipes.worker_cores().num_cores() == 50
    assert recipes.ring_cores().num_cores() == recipes.RING_CORE_COUNT


@pytest.mark.host
@pytest.mark.model
def test_prefetch_mapping_is_unchanged_and_order_preserving():
    """The sender/receiver mapping keeps its length, order and coverage."""

    mapping = galaxy_sender_receiver_mapping()

    assert len(mapping) == len(GOLDEN_PREFETCH_SENDERS) + len(GOLDEN_DUMMY_SENDERS) == 20
    senders = tuple((core.x, core.y) for core, _ in mapping)
    assert senders == GOLDEN_PREFETCH_SENDERS + GOLDEN_DUMMY_SENDERS

    expected_receivers = tuple(recipes.core_ranges((*start, *end)) for start, end in GOLDEN_RECEIVER_PAIRS) + tuple(
        recipes.core_ranges(*group) for group in GOLDEN_DUMMY_RECEIVER_RANGES
    )
    assert tuple(receivers for _, receivers in mapping) == expected_receivers

    # Entry 16 (index 15) is the dummy entry carrying the hop core `(3, 6)`, and
    # it is the one the Wormhole project paid for: an "obviously correct" minimal
    # 12-entry mapping passes every module test and then fails the
    # GlobalCircularBuffer superset check the first time a fused matmul's
    # gather-in0 program reaches that core.
    hop_entry = mapping[15][1]
    assert hop_entry.subtract(recipes.core_points(GOLDEN_HOP_CORES)).num_cores() == hop_entry.num_cores() - 1


@pytest.mark.host
@pytest.mark.model
def test_descriptor_validates_against_a_live_grid():
    """Resolution checks the descriptor against what the device reports."""

    resolved = resolve_galaxy_chip_topology(_mesh())
    assert resolved.validated
    assert resolved.compute_grid == GOLDEN_COMPUTE_GRID

    # Validation returns a new descriptor rather than mutating the shared one.
    assert WORMHOLE_GALAXY_TOPOLOGY.validated is False

    with pytest.raises(ValueError, match="expects compute grid"):
        resolve_galaxy_chip_topology(_mesh(grid=(12, 10)))

    # Resolution *adopts* the reported DRAM view count instead of refusing it:
    # `dram_views` sizes the DRAM-sharded weight configs and is a property of the
    # chassis, not of the hand-measured core tables that the compute grid gates.
    # So the descriptor's own `DRAM views` guard cannot fire through the
    # resolver -- which never disagrees with itself -- and fires where it is
    # actually load-bearing: a descriptor built without a device, then checked
    # against one that disagrees.
    assert resolve_galaxy_chip_topology(_mesh(dram_width=8)).dram_views == 8
    with pytest.raises(ValueError, match="DRAM views"):
        WORMHOLE_GALAXY_TOPOLOGY.validate_against_device(_mesh(dram_width=8))

    # Blackhole has a descriptor, so it resolves; an architecture without one
    # still fails closed.
    with pytest.raises(ValueError, match="no Galaxy topology"):
        resolve_galaxy_chip_topology(_mesh(arch=ttnn.device.Arch.QUASAR))


@pytest.mark.host
@pytest.mark.model
def test_validation_rejects_placements_outside_the_worker_envelope():
    """The cheap host check that would have caught the reference's `x=6` failure.

    The reference port put a ring memory config's shard grid on a core column
    outside the auto-selected compute grid and learned about it from
    *"Tensor shard spec grid ... must lie within compute grid"* on hardware.
    Every constraint here is one an allocated node would otherwise discover.
    """

    mesh = _mesh()

    # A top-k grid that escapes the workers.
    escaped = replace(WORMHOLE_GALAXY_TOPOLOGY, topk_core_ranges=((0, 0, 0, 9),))
    with pytest.raises(ValueError, match="top-k grid lies outside the worker envelope"):
        escaped.validate_against_device(mesh)

    # A norm origin on a prefetch sender column.
    with pytest.raises(ValueError, match="norm origin lies outside the worker envelope"):
        replace(WORMHOLE_GALAXY_TOPOLOGY, norm_origin=(4, 0)).validate_against_device(mesh)

    # A core outside the reported compute grid entirely.
    with pytest.raises(ValueError, match="leaves the .* compute grid"):
        replace(WORMHOLE_GALAXY_TOPOLOGY, norm_origin=(9, 0)).validate_against_device(mesh)

    # A reserved column folded into the workers. Empty on Wormhole, so state the
    # rule against a synthetic one: this is the guard that keeps Blackhole's
    # dispatch column out of the envelope, where including it regresses prefill
    # warmup with nothing failing.
    with pytest.raises(ValueError, match="reserved column"):
        replace(WORMHOLE_GALAXY_TOPOLOGY, reserved_columns=(3,)).validate_against_device(mesh)


@pytest.mark.host
@pytest.mark.model
def test_validation_rejects_an_incomplete_prefetch_mapping():
    """Dropping a dummy entry must fail on the host, not on the ring."""

    short = replace(
        WORMHOLE_GALAXY_TOPOLOGY,
        dummy_sender_coords=GOLDEN_DUMMY_SENDERS[:-1],
        dummy_receiver_ranges=GOLDEN_DUMMY_RECEIVER_RANGES[:-1],
    )
    with pytest.raises(ValueError, match="does not cover worker core"):
        short.validate_against_device(_mesh())

    # Sender and receiver lists are zipped, so unequal lengths would silently
    # truncate the mapping rather than raise.
    lopsided = replace(WORMHOLE_GALAXY_TOPOLOGY, dummy_sender_coords=GOLDEN_DUMMY_SENDERS[:-1])
    with pytest.raises(ValueError, match="senders and .* receiver groups"):
        lopsided.validate_against_device(_mesh())

    # The hop core lives only in a dummy receiver group, so losing that group
    # must be reported as the superset failure it will become.
    without_hop_group = replace(
        WORMHOLE_GALAXY_TOPOLOGY,
        dummy_sender_coords=GOLDEN_DUMMY_SENDERS[:3] + GOLDEN_DUMMY_SENDERS[4:],
        dummy_receiver_ranges=GOLDEN_DUMMY_RECEIVER_RANGES[:3] + GOLDEN_DUMMY_RECEIVER_RANGES[4:],
    )
    with pytest.raises(ValueError, match="hop core|does not cover worker core"):
        without_hop_group.validate_against_device(_mesh())


@pytest.mark.host
@pytest.mark.model
def test_galaxy_mesh_gate_is_an_allowlist_keyed_on_available_geometry():
    """`validate_galaxy_mesh` admits exactly the architectures with a descriptor."""

    # Both architectures with a descriptor, in registration order. Blackhole
    # joined when its resolver landed: the allowlist extends by construction,
    # which is the property this asserts -- not that the list stays at one.
    assert supported_galaxy_architectures() == (ttnn.device.Arch.WORMHOLE_B0, ttnn.device.Arch.BLACKHOLE)
    recipes.validate_galaxy_mesh("probe", _mesh())
    recipes.validate_galaxy_mesh("probe", _mesh(arch=ttnn.device.Arch.BLACKHOLE))

    with pytest.raises(ValueError, match="has no Galaxy topology"):
        recipes.validate_galaxy_mesh("probe", _mesh(arch=ttnn.device.Arch.QUASAR))
    with pytest.raises(ValueError, match="no Galaxy topology for"):
        galaxy_chip_topology(ttnn.device.Arch.QUASAR)

    # The mesh shape stays a hard equality: `(8, 4)` is architecture-invariant,
    # so generalizing it would weaken the gate for no gain.
    wrong_shape = _mesh()
    wrong_shape.shape = (4, 8)
    with pytest.raises(ValueError, match="logical mesh shape"):
        recipes.validate_galaxy_mesh("probe", wrong_shape)


@pytest.mark.host
@pytest.mark.model
def test_capabilities_keep_prefetcher_and_fused_ccl_on_separate_axes():
    """The one capability decision milestone 1 must get right for later work.

    The reference port's *tested* Blackhole decode path is prefetcher-on with
    the fused galaxy collectives off: the ring matmuls must consume the
    prefetched global-CB weights, while `fused_rms_minimal` and its four peers
    use 1D-multicast writers that silently no-op on the 2D-torus fabric.
    Collapsing these into one architecture check makes that configuration
    unexpressible, and it is the cheapest thing here to get right.
    """

    capabilities = WORMHOLE_GALAXY_TOPOLOGY.capabilities
    assert capabilities.has_prefetcher is True
    assert capabilities.has_fused_ccl is True

    hybrid = replace(capabilities, has_fused_ccl=False)
    assert hybrid.has_prefetcher is True and hybrid.has_fused_ccl is False


# ---------------------------------------------------------------------------
# Blackhole Galaxy -- milestone 1, prefetcher-free
# ---------------------------------------------------------------------------


@pytest.mark.host
@pytest.mark.model
@pytest.mark.parametrize("grid", [(11, 10), (12, 10), (13, 10)], ids=["11x10", "12x10", "13x10"])
def test_blackhole_descriptor_resolves_for_every_harvesting_shape(grid):
    """Blackhole harvesting is per-part, so the envelope is derived from the grid.

    `12 x 10` is the shape the reference measured and the one this port expects
    to deploy on; `13 x 10` unharvested and `11 x 10` are both shapes a real part
    can present, and a differently-harvested board must resolve rather than fail.
    """

    topology = resolve_galaxy_chip_topology(_mesh(arch=ttnn.device.Arch.BLACKHOLE, grid=grid, dram_width=8))

    width = grid[0]
    assert topology.compute_grid == grid
    assert topology.worker_core_ranges == ((1, 0, width - 2, 9),)

    # The dispatch column is excluded by *measurement*, not derivation: it sits
    # inside `compute_with_storage_grid_size()`, and folding it into the workers
    # regresses prefill warmup with nothing raising.
    assert topology.reserved_columns == (width - 1,)
    assert all(x != width - 1 for x, _ in topology.worker_coords)


@pytest.mark.host
@pytest.mark.model
def test_blackhole_milestone_one_is_prefetcher_free_by_explicit_marker():
    """Absence is expressed, not implied.

    `prefetch_sender_coords = ()` plus `has_prefetcher = False` are the markers.
    A path expressed as "the prefetcher fields do not exist" could not later grow
    one without touching every consumer, which is the whole cost the deferred
    work is trying to avoid.
    """

    topology = resolve_galaxy_chip_topology(_mesh(arch=ttnn.device.Arch.BLACKHOLE, grid=(12, 10), dram_width=8))

    assert topology.prefetch_sender_coords == ()
    assert topology.dummy_sender_coords == ()
    assert topology.capabilities.has_prefetcher is False

    # No ring path: `None` rather than Wormhole's coordinates, so a consumer that
    # asks for one fails loudly instead of receiving geometry for another chip.
    assert topology.ring_core_coords is None
    assert topology.ring_matmul_grid is None
    with pytest.raises(ValueError, match="has no ring matmul path"):
        recipes.ring_cores(topology)

    assert topology.worker_coords  # 100 cores at 12x10
    assert len(topology.worker_coords) == 100


@pytest.mark.host
@pytest.mark.model
def test_blackhole_fabric_and_link_budget():
    """The fabric config is a device-open parameter, so it lives on the descriptor.

    Column-axis (`cluster_axis=1`) collectives run on device on Blackhole Galaxy
    and require a 2D torus; `FABRIC_1D` and `FABRIC_1D_RING` throw
    `IndexError: map::at` on the cross-column route. The value therefore has to
    be known before the mesh opens, which is why it is not probed afterwards.
    """

    topology = resolve_galaxy_chip_topology(_mesh(arch=ttnn.device.Arch.BLACKHOLE, grid=(12, 10), dram_width=8))

    assert topology.fabric_config is ttnn.FabricConfig.FABRIC_2D_TORUS_XY
    assert topology.dispatch_core_axis is ttnn.DispatchCoreAxis.COL
    assert topology.dram_views == 8

    # Two links, not four: the ring and line CCLs index ethernet channels by
    # link, so four overruns the available channels and deadlocks.
    assert topology.fabric_links == 2
    assert topology.ccl_reserved_worker_cores == 2

    params = galaxy_device_params(ttnn.device.Arch.BLACKHOLE)
    assert params["fabric_config"] is ttnn.FabricConfig.FABRIC_2D_TORUS_XY
    assert params["dispatch_core_axis"] is ttnn.DispatchCoreAxis.COL
    assert galaxy_device_params(ttnn.device.Arch.WORMHOLE_B0)["fabric_config"] is ttnn.FabricConfig.FABRIC_1D_RING


@pytest.mark.host
@pytest.mark.model
def test_blackhole_capabilities_are_all_false_for_a_recorded_reason():
    """Milestone 1 turns off exactly the four mechanisms the Wormhole recipes assume."""

    capabilities = resolve_galaxy_chip_topology(
        _mesh(arch=ttnn.device.Arch.BLACKHOLE, grid=(12, 10), dram_width=8)
    ).capabilities

    assert capabilities == replace(
        capabilities,
        has_prefetcher=False,
        has_ring_matmul=False,
        has_fused_ccl=False,
        has_fused_residual_norm=False,
        has_fused_qk_rotary=False,
        has_distributed_sampling=False,
    )


@pytest.mark.host
@pytest.mark.model
def test_wormhole_descriptor_is_untouched_by_the_blackhole_branch():
    """No Wormhole value moves when a second architecture is added."""

    wormhole = resolve_galaxy_chip_topology(_mesh())

    assert wormhole.compute_grid == GOLDEN_COMPUTE_GRID
    assert wormhole.worker_core_ranges == GOLDEN_WORKER_RANGES
    assert wormhole.ring_core_coords == GOLDEN_RING_CORES
    assert wormhole.prefetch_sender_coords == GOLDEN_PREFETCH_SENDERS
    assert wormhole.fabric_links == 4
    assert wormhole.fabric_config is ttnn.FabricConfig.FABRIC_1D_RING
    assert wormhole.reserved_columns == ()
    assert wormhole.capabilities.has_prefetcher is True

    # Wormhole Galaxy's core-descriptor key is a fixed 7x10 and every table above
    # was qualified against it, so a differently-shaped grid is refused rather
    # than reinterpreted.
    with pytest.raises(ValueError, match="specific to it"):
        resolve_galaxy_chip_topology(_mesh(grid=(12, 10)))


@pytest.mark.host
@pytest.mark.model
def test_galaxy_mesh_gate_now_admits_blackhole():
    """`validate_galaxy_mesh` admits an architecture once its geometry exists."""

    assert supported_galaxy_architectures() == (ttnn.device.Arch.WORMHOLE_B0, ttnn.device.Arch.BLACKHOLE)
    recipes.validate_galaxy_mesh("probe", _mesh(arch=ttnn.device.Arch.BLACKHOLE, grid=(12, 10), dram_width=8))

    # The mesh shape stays a hard equality on both architectures: Blackhole
    # Galaxy declares the same `device_topology { dims: [8, 4] }`, so `(8, 4)` is
    # architecture-invariant and generalizing it would weaken the gate for no gain.
    wrong_shape = _mesh(arch=ttnn.device.Arch.BLACKHOLE, grid=(12, 10), dram_width=8)
    wrong_shape.shape = (4, 8)
    with pytest.raises(ValueError, match="logical mesh shape"):
        recipes.validate_galaxy_mesh("probe", wrong_shape)


@pytest.mark.host
@pytest.mark.model
@pytest.mark.parametrize(
    "vocab_size,expected",
    [(128256, 129024), (151936, 153600)],
    ids=["llama-3.3-70b", "qwen3-32b"],
)
def test_vocabulary_padding_is_identical_on_both_architectures(vocab_size, expected):
    """The Blackhole padded vocabulary must equal the Wormhole one.

    `galaxy_padded_vocab_size` pads to `GALAXY_ROWS * RING_ALIGNMENT`, and
    `RING_ALIGNMENT` is a function of the ring size. Ring-exactness is
    load-bearing rather than cosmetic: `all_reduce_async`'s reduction kernel
    opens with `cb_in.wait_front(ring_size * block_num_tiles)` on *every* output
    core, so a width with no divisor in the chosen core count leaves the last
    core waiting for tiles the fabric never sends, the program never signals
    completion, and the host blocks in `wait_for_outstanding_reads` with no
    traceback and no abort -- the mesh has to be reset.

    The values are expected to carry because the Galaxy reference holds
    `RING_SIZE = 24` on Blackhole *deliberately*, keeping all the weight-sharding
    math by widening receivers-per-reader from 2 to 3 (`8 x 3 = 24 = 12 x 2`).
    **If this assertion ever fails, the ring size moved**, and every constraint
    stated in terms of `ring_size` has to be re-walked -- starting with
    `lm_head_reduce_core_count`'s divisor search, which this padding is what
    makes exact by construction.

    Beware the `24 -> 16` figure that circulates for Blackhole: that is the 1D
    LM head, a different path with different weight sharding, and carrying it
    into Galaxy vocabulary arithmetic would break exactly this invariant.
    """

    assert recipes.galaxy_padded_vocab_size(vocab_size) == expected

    # Divisor-exactness, which is the property the padding exists to guarantee.
    local = expected // recipes.GALAXY_ROWS
    assert local % recipes.TILE == 0
    assert recipes.pad_ring_width(local) == local

    # Worker counts differ between the architectures (50 on Wormhole, 100 on
    # Blackhole at 12x10), so check the reduction resolves on both.
    # Worker counts and the per-link reserve both differ between the
    # architectures (50 cores / 4 links on Wormhole, 100 / 2 on Blackhole at
    # 12x10), and starving `all_reduce_async` of worker cores warns and then
    # segmentation-faults rather than raising, so check both resolve.
    for available, reserved in ((50, 4), (100, 2)):
        count = recipes.lm_head_reduce_core_count(local, available, reserved_worker_cores=reserved)
        assert (local // recipes.TILE) % count == 0
        assert count <= available - reserved


@pytest.mark.host
@pytest.mark.model
def test_l1_small_region_matches_the_shared_device_constant():
    """The descriptor's literal must equal `device_utils.GALAXY_L1_SMALL_SIZE`.

    `topology.py` keeps its own literal rather than importing that module, whose
    chain reaches `lazy_weight`, `loguru` and `torch`; the descriptor's only
    dependency is `ttnn`'s enums, which is what lets its validation surface be
    exercised without a device at all. This test is the price of that, and it is
    the right trade: the drift it guards against is a host-testable equality,
    where the testability it buys is not replaceable.
    """

    from tt_transformers.device_utils import GALAXY_L1_SMALL_SIZE

    assert WORMHOLE_GALAXY_TOPOLOGY.l1_small_size == GALAXY_L1_SMALL_SIZE
    for architecture in supported_galaxy_architectures():
        assert galaxy_device_params(architecture)["l1_small_size"] == GALAXY_L1_SMALL_SIZE
