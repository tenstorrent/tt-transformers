# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""What a Blackhole Galaxy `(8, 4)` mesh actually reports on silicon.

**This is the first thing to run on a Blackhole Galaxy, and nothing here needs a
checkpoint, a model, or a weight.** It opens the mesh, reads what the device says
about itself, and checks it against `GalaxyChipTopology` -- so the answers that
the whole port's geometry is built on are measured rather than assumed, in about
as long as opening the mesh takes.

Five of the port plan's open questions are answered here, and each is recorded
with its consequence rather than merely asserted:

* **Grid uniformity across all 32 devices** (§8 Q1). The *shape* is known from
  the reference: `12 x 10`, the 1x-harvested key. What is not known is whether
  harvesting is uniform across our chassis. A mixed-harvesting mesh forces the
  descriptor to take a per-mesh minimum, which no current code does.
* **Cluster identity** (§3). `ClusterType.BLACKHOLE_GALAXY` is what
  `get_logical_sku` and the marker policy key on, so if this mesh does not report
  it, the whole detection route is wrong.
* **The link budget** (§8 Q5). `GALAXY_FABRIC_LINKS` and `tt_ccl.get_num_links`
  derive the same number from different sources -- an architecture table and the
  device name. Nothing on the host can prove they agree. This can.
* **`GALAXY_L1_SMALL_SIZE`** (§8 Q4). 32 kB was sized against Wormhole's
  1 393 472 B bank, and main L1 shrinks by exactly that much for *every* Galaxy
  op.
* **The fabric configuration** (§3). Opening this mesh at all, with
  `FABRIC_2D_TORUS_XY`, is the measurement: the Wormhole `FABRIC_1D_RING` is
  reported to throw `IndexError: map::at` on the cross-column route.

Deliberately read-only. It runs no collective and allocates no tensor, so it
cannot leave the mesh in a state the next run has to recover from -- which
matters when the recovery unit is a reset cycle.
"""

from __future__ import annotations

import pytest
import torch
import ttnn
from tests.models.galaxy.galaxy_hardware import BHGLX_DEVICE_PARAMS, GALAXY_MESH_SHAPE

from tt_transformers.device_utils import GALAXY_L1_SMALL_SIZE, get_device_name, has_l1_small_region
from tt_transformers.models.galaxy import recipes
from tt_transformers.models.galaxy.topology import (
    BLACKHOLE_FIRST_WORKER_COLUMN,
    BLACKHOLE_GALAXY_COMPUTE_GRID,
    resolve_galaxy_chip_topology,
)
from tt_transformers.modules.tt_ccl import get_num_links

galaxy = pytest.mark.parametrize("mesh_device", [pytest.param(GALAXY_MESH_SHAPE, id="8x4")], indirect=True)
galaxy_params = pytest.mark.parametrize("device_params", [BHGLX_DEVICE_PARAMS], indirect=True)


def _report(title: str, value: object) -> None:
    print(f"[bhglx] {title}: {value}", flush=True)


@pytest.mark.device
@pytest.mark.blackhole
@pytest.mark.galaxy_bh
@pytest.mark.model
@galaxy_params
@galaxy
@torch.no_grad()
def test_blackhole_galaxy_reports_the_geometry_the_descriptor_expects(mesh_device):
    """Resolve the descriptor against the live mesh and report every input."""

    grid = mesh_device.compute_with_storage_grid_size()
    dram = mesh_device.dram_grid_size()
    _report("compute_with_storage_grid_size", f"x={grid.x} y={grid.y}")
    _report("dram_grid_size", f"x={dram.x} y={dram.y}")
    _report("arch", mesh_device.arch())
    _report("device name", get_device_name(mesh_device))
    _report("cluster type", ttnn.cluster.get_cluster_type())

    assert mesh_device.arch() == ttnn.device.Arch.BLACKHOLE
    assert tuple(mesh_device.shape) == GALAXY_MESH_SHAPE
    assert mesh_device.get_num_devices() == 32

    # `resolve_galaxy_chip_topology` runs every containment check in
    # `validate_against_device`, so a placement that leaves the worker envelope
    # or the compute grid fails here rather than inside a kernel.
    topology = resolve_galaxy_chip_topology(mesh_device)
    _report("resolved worker envelope", topology.worker_core_ranges)
    _report("worker cores", len(topology.worker_coords))
    _report("reserved (dispatch) columns", topology.reserved_columns)
    assert topology.validated

    # Not an assertion about correctness -- a record of whether this chassis is
    # the one the reference measured. A different harvesting key resolves fine;
    # it just means the port is running on a shape nobody has characterized.
    if (int(grid.x), int(grid.y)) != BLACKHOLE_GALAXY_COMPUTE_GRID:
        _report(
            "NOTE: grid differs from the reference chassis",
            f"expected {BLACKHOLE_GALAXY_COMPUTE_GRID}, got {(int(grid.x), int(grid.y))}",
        )


@pytest.mark.device
@pytest.mark.blackhole
@pytest.mark.galaxy_bh
@pytest.mark.model
@galaxy_params
@galaxy
@torch.no_grad()
def test_blackhole_galaxy_harvesting_is_uniform_across_the_mesh(mesh_device):
    """§8 Q1: a mixed-harvesting mesh would need a per-mesh minimum nothing takes.

    The descriptor is resolved once, from the mesh's own
    `compute_with_storage_grid_size()`. If individual devices disagree, that
    single answer is wrong for some of them -- and the failure is a placement on
    a core that exists on most of the chassis and not on one board, which is the
    silent-on-31-devices class.
    """

    # Guarded the way `llm_runtime/execution.py` and the T3K prefill suite guard
    # it: `get_devices` is not present on every mesh binding, and a probe that
    # cannot enumerate the mesh should say so rather than assert something else.
    if not hasattr(mesh_device, "get_devices"):
        pytest.skip("this ttnn build's MeshDevice does not expose get_devices(); per-device harvesting unreadable")

    grids = {}
    for index, device in enumerate(mesh_device.get_devices()):
        grid = device.compute_with_storage_grid_size()
        dram = device.dram_grid_size()
        grids[index] = ((int(grid.x), int(grid.y)), int(dram.x))
    _report("devices enumerated", len(grids))

    distinct = sorted(set(grids.values()))
    _report("distinct (compute_grid, dram_views) across the mesh", distinct)
    if len(distinct) > 1:
        offenders = {index: value for index, value in grids.items() if value != distinct[0]}
        _report("devices differing from the first", offenders)

    assert len(distinct) == 1, (
        f"harvesting is not uniform across the 32 devices: {distinct}. "
        "The topology descriptor resolves one grid for the whole mesh, so a mixed "
        "chassis needs a per-mesh minimum that no current code takes."
    )


@pytest.mark.device
@pytest.mark.blackhole
@pytest.mark.galaxy_bh
@pytest.mark.model
@galaxy_params
@galaxy
@torch.no_grad()
def test_blackhole_galaxy_link_budget_agrees_across_both_sources(mesh_device):
    """§8 Q5: the two independent link tables must report the same number.

    `GALAXY_FABRIC_LINKS` keys on the architecture; `tt_ccl.get_num_links` keys
    on the device name that `device_utils` derives. They were written from
    different evidence and nothing on the host can compare them, because
    `get_num_links` needs a real device. Disagreement is not academic: too many
    links overruns the ethernet channel array and deadlocks, too few has caused a
    real CCL stall, and neither raises.
    """

    topology = resolve_galaxy_chip_topology(mesh_device)
    by_architecture = topology.fabric_links
    by_device_name = [get_num_links(mesh_device, cluster_axis=axis) for axis in (0, 1)]
    _report("links by architecture table", by_architecture)
    _report("links by device name, per axis", by_device_name)

    assert by_architecture == 2, f"expected 2 fabric links on Blackhole Galaxy, descriptor says {by_architecture}"
    assert by_device_name == [by_architecture, by_architecture], (
        f"link budget disagrees: architecture table says {by_architecture}, "
        f"get_num_links says {by_device_name}. One of the two is wrong, and an "
        "incorrect Galaxy num_links deadlocks rather than failing."
    )
    assert topology.ccl_reserved_worker_cores == by_architecture


@pytest.mark.device
@pytest.mark.blackhole
@pytest.mark.galaxy_bh
@pytest.mark.model
@galaxy_params
@galaxy
@torch.no_grad()
def test_blackhole_galaxy_opens_with_an_l1_small_region(mesh_device):
    """§8 Q4: confirm the L1 small region rather than assuming 32 kB carries.

    Generic collectives create their semaphores at *program compile* time, in
    main L1 mid-bank, where nothing can be placed across them afterwards. The
    `L1_SMALL` region is where they go instead. The size was chosen against
    Wormhole's 1 393 472 B bank, and main L1 shrinks by exactly this much for
    every Galaxy op, so the question is whether it is still the right size here
    -- not merely whether the region exists.
    """

    assert has_l1_small_region(mesh_device), (
        "the mesh opened without an L1_SMALL region; generic collectives will "
        "place their semaphores mid-bank in main L1"
    )
    view = ttnn.get_memory_view(mesh_device, ttnn.BufferType.L1_SMALL)
    main = ttnn.get_memory_view(mesh_device, ttnn.BufferType.L1)
    _report("L1_SMALL total_bytes_per_bank", view.total_bytes_per_bank)
    _report("L1 total_bytes_per_bank", main.total_bytes_per_bank)
    _report("GALAXY_L1_SMALL_SIZE", GALAXY_L1_SMALL_SIZE)
    assert view.total_bytes_per_bank >= GALAXY_L1_SMALL_SIZE


@pytest.mark.device
@pytest.mark.blackhole
@pytest.mark.galaxy_bh
@pytest.mark.model
@galaxy_params
@galaxy
@torch.no_grad()
def test_blackhole_galaxy_worker_envelope_excludes_the_dispatch_column(mesh_device):
    """The one placement rule milestone 1 must honour, and the open question beside it.

    Column 11 on a `12 x 10` grid is excluded from the workers *empirically*:
    it is inside `compute_with_storage_grid_size()`, and folding it in regresses
    prefill warmup with nothing failing. This pins that it stays out.

    It also reports how many cores the conservative first-worker-column choice
    is costing, which is the open half of §8 Q2: the reference excludes column 0
    because its prefetcher senders live there, and milestone 1 has none.
    """

    topology = resolve_galaxy_chip_topology(mesh_device)
    grid = mesh_device.compute_with_storage_grid_size()
    full = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
    workers = recipes.worker_cores(topology)

    _report("worker_cores", f"{workers} ({workers.num_cores()} cores)")
    _report("cores in the compute grid but in no partition", full.subtract(workers))

    assert workers.subtract(full).num_cores() == 0, "worker_cores() leaves the compute grid"
    dispatch_column = int(grid.x) - 1
    assert all(x != dispatch_column for x, _ in topology.worker_coords)
    assert topology.reserved_columns == (dispatch_column,)

    # The open half of §8 Q2, reported rather than asserted: settling it is a
    # one-line change to BLACKHOLE_FIRST_WORKER_COLUMN.
    _report("BLACKHOLE_FIRST_WORKER_COLUMN", BLACKHOLE_FIRST_WORKER_COLUMN)
    _report(
        "cores column 0 would add if it rejoined the workers",
        int(grid.y) * BLACKHOLE_FIRST_WORKER_COLUMN,
    )
