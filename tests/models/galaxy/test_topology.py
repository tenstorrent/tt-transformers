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
    with pytest.raises(ValueError, match="DRAM views"):
        resolve_galaxy_chip_topology(_mesh(dram_width=8))
    with pytest.raises(ValueError, match="no Galaxy topology"):
        resolve_galaxy_chip_topology(_mesh(arch=ttnn.device.Arch.BLACKHOLE))


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

    assert supported_galaxy_architectures() == (ttnn.device.Arch.WORMHOLE_B0,)
    recipes.validate_galaxy_mesh("probe", _mesh())

    with pytest.raises(ValueError, match="has no Galaxy topology"):
        recipes.validate_galaxy_mesh("probe", _mesh(arch=ttnn.device.Arch.BLACKHOLE))
    with pytest.raises(ValueError, match="no Galaxy topology for"):
        galaxy_chip_topology(ttnn.device.Arch.BLACKHOLE)

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
