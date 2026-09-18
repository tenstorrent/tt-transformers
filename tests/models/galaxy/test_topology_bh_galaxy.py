# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""What a Blackhole Galaxy `(8, 4)` mesh actually reports on silicon.

**This is the first thing to run on a Blackhole Galaxy, and nothing here needs a
checkpoint, a model, or a weight.** It opens the mesh, reads what the device says
about itself, and checks it against `GalaxyChipTopology` -- so the answers that
the whole port's geometry is built on are measured rather than assumed, in about
as long as opening the mesh takes.

Five open questions about this chassis are answered here, and each is recorded
with its consequence rather than merely asserted:

* **Grid uniformity across all 32 devices.** The *shape* is known from
  the reference: `12 x 10`, the 1x-harvested key. What is not known is whether
  harvesting is uniform across our chassis. A mixed-harvesting mesh forces the
  descriptor to take a per-mesh minimum, which no current code does. This was
  unmeasurable for two windows because `MeshDevice.get_devices()` does not exist
  on ttnn 0.77.0; a `(1, 1)` submesh per coordinate reads the same numbers
  through a binding that does, and the DRAM half has a cheaper route still.
* **Cluster identity.** `ClusterType.BLACKHOLE_GALAXY` is what
  `get_logical_sku` and the marker policy key on, so if this mesh does not report
  it, the whole detection route is wrong.
* **The link budget.** `GALAXY_FABRIC_LINKS` and `tt_ccl.get_num_links`
  derive the same number from different sources -- an architecture table and the
  device name. Nothing on the host can prove they agree. This can.
* **`GALAXY_L1_SMALL_SIZE`.** 32 kB was sized against Wormhole's
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

import re
import time
from pathlib import Path

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


#: How much wall clock the 1x1-submesh route may spend reading the whole mesh
#: before this suite hands the question back to its own job. A submesh's cost is
#: unmeasured, and this suite is the cheap read-only probe that runs first.
_SUBMESH_ROUTE_BUDGET_SECONDS = 300


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


def _mesh_coordinates() -> tuple[tuple[int, int], ...]:
    rows, columns = GALAXY_MESH_SHAPE
    return tuple((row, column) for row in range(rows) for column in range(columns))


def _grids_via_get_devices(mesh_device) -> dict[object, tuple[tuple[int, int], int]] | None:
    """Route 1: enumerate the mesh directly. Absent on ttnn 0.77.0."""

    if not hasattr(mesh_device, "get_devices"):
        return None
    return {
        index: ((int(device.compute_with_storage_grid_size().x), int(device.compute_with_storage_grid_size().y)),
                int(device.dram_grid_size().x))
        for index, device in enumerate(mesh_device.get_devices())
    }  # fmt: skip


def _harvesting_via_cluster_descriptor(mesh_device) -> dict[object, tuple[object, ...]] | None:
    """Route 3: read every chip's harvesting out of the serialized cluster descriptor.

    **This is the route that answered grid uniformity**, on 2026-09-17, after the submesh
    route below segfaulted. It needs no submesh, no program, no tensor and no
    `quiesce_devices`: the descriptor is a YAML dump the driver already has.

    It reads the *cause* rather than a derived grid, which is why it is a better
    instrument and not merely a cheaper one. What determines whether one
    mesh-level `compute_with_storage_grid_size()` is valid for all 32 devices is:

    * `popcount(harvest_mask)` -- **how many** tensix rows each chip loses. The
      measured answer is **1 on every chip**, from 14 *distinct* mask values. So
      the harvested row sits in a different place per chip and every chip loses
      the same count;
    * `noc_translation` -- **`true`** on all 32, which is the mechanism that
      makes the position irrelevant. The surviving rows are mapped onto a
      contiguous logical grid, so every chip presents the same logical
      `12 x 10` however its physical row falls;
    * `dram_harvesting_mask` -- **0** on all 32, corroborating
      `get_dram_bank_table`'s uniform 8 banks from an independent source.

    So the tuple compared for uniformity is
    `(harvested_row_count, dram_harvesting_mask, noc_translation)` and **not**
    the mask value. Asserting the masks were equal would fail on a healthy
    chassis: 14 distinct positions is the expected, correct reading.
    """

    serialize = getattr(getattr(ttnn, "cluster", None), "serialize_cluster_descriptor", None)
    if not callable(serialize):
        return None
    try:
        path = serialize()
        text = Path(str(path)).read_text()
    except Exception as exception:  # noqa: BLE001 - a route that cannot answer is a result
        _report("cluster descriptor route failed", f"{type(exception).__name__}: {exception}")
        return None

    section = text.split("harvesting:")
    if len(section) < 2:
        _report("cluster descriptor route", "no 'harvesting:' section")
        return None

    readings: dict[object, tuple[object, ...]] = {}
    for chip, body in re.findall(r"^  (\d+):\n((?:    \w+: \S+\n)+)", section[1], re.M):
        fields = dict(re.findall(r"    (\w+): (\S+)", body))
        if "harvest_mask" not in fields:
            continue
        readings[int(chip)] = (
            bin(int(fields["harvest_mask"])).count("1"),
            int(fields.get("dram_harvesting_mask", 0)),
            fields.get("noc_translation", "unknown"),
        )
    if not readings:
        return None
    positions = {bin(int(m)).count("1") for m in re.findall(r"    harvest_mask: (\S+)", section[1])}
    _report("chips read from the cluster descriptor", len(readings))
    _report("distinct harvested-row counts", sorted(positions))
    return readings


def _grids_via_submeshes(mesh_device) -> dict[object, tuple[tuple[int, int], int]] | None:
    """Route 2: carve a 1x1 submesh per coordinate and ask *it* for the grid.

    **NOT IN THE DEFAULT ROUTE LIST: it segfaults on ttnn 0.77.0.** Measured
    twice, on a healthy chassis: it carves submesh
    `(0, 0)` in 0.08 s, reports `((12, 10), 8)` correctly, and then the process
    dies with **SIGSEGV** -- once inside this suite (taking the whole pytest
    process with it, so every node id behind it is lost) and once inside
    `q1_per_device_geometry.sh`'s standalone probe. Same crash, two callers.

    **`create_submesh` is not broken in general, so do not read it that way.**
    `tests/conftest.py` carves submeshes routinely with the one-argument
    `create_submesh(ttnn.MeshShape(shape))` form and every galaxy suite that
    requests a sub-shape depends on it. What crashes is the *two*-argument
    shape-plus-coordinate form used here, at shape `(1, 1)` -- which is also the
    only form that can address one specific device, and therefore the only one
    this route could have used.

    That makes it worse than an absent route. An absent route returns `None` and
    the next one is tried; this one kills the process, and because a crash
    part-way through carving cannot run the `quiesce_devices` below, it can leave
    a child submesh holding the shared command queue for whatever runs next.

    It is kept, unwired, because the reasoning is sound and the binding is
    present -- so a future ttnn may simply fix it, and `_harvesting_via_cluster_descriptor`
    already answers the question meanwhile. **Re-wire it only behind a fresh
    measurement that it no longer crashes**, and never in a suite that has node
    ids after it.

    `MeshDevice.compute_with_storage_grid_size`, `dram_grid_size` and `arch` are
    each documented as reporting *"the first device in the device mesh"*, so on a
    `(1, 1)` submesh at `(row, column)` they report that one device. That is the
    per-device route grid uniformity was missing: it needs no `get_devices()`, allocates no
    tensor and runs no program.

    **Quiesce before the fixture closes the mesh.** `quiesce_devices` exists for
    exactly this and says so: *"Call before closing a mesh that has carved
    submeshes so the shared command queue is idle, otherwise close throws 'cq is
    in use by child submesh'"*. A probe that skipped it would pass every
    assertion and then fail the node id in teardown -- the shape that has cost a
    whole investigation once before, arriving from a different direction.
    """

    if not hasattr(mesh_device, "create_submesh"):
        return None

    def read_one(row: int, column: int) -> tuple[tuple[int, int], int]:
        submesh = mesh_device.create_submesh(ttnn.MeshShape(1, 1), ttnn.MeshCoordinate(row, column))
        grid = submesh.compute_with_storage_grid_size()
        return (int(grid.x), int(grid.y)), int(submesh.dram_grid_size().x)

    grids: dict[object, tuple[tuple[int, int], int]] = {}
    try:
        # Cost the first carve before committing to all 32, the way
        # `q1_per_device_geometry.sh` does: a submesh's cost is unmeasured, and
        # thirty-two of an expensive one would quietly consume a suite's budget
        # on a borrowed machine. Over the cap this reports the measured cost and
        # stops, which the runner counts as a failure and a human reads -- rather
        # than a partial read, which cannot settle uniformity anyway.
        started = time.monotonic()
        grids[(0, 0)] = read_one(0, 0)
        elapsed = time.monotonic() - started
        _report("first 1x1 submesh seconds", f"{elapsed:.2f}")
        projected = elapsed * len(_mesh_coordinates())
        if projected > _SUBMESH_ROUTE_BUDGET_SECONDS:
            pytest.skip(
                f"the 1x1 submesh route costs {elapsed:.1f}s per device, {projected:.0f}s for the mesh, "
                f"over this suite's {_SUBMESH_ROUTE_BUDGET_SECONDS}s cap -- queue it as its own job "
                "(`q1_per_device_geometry.sh`) rather than inside a probe suite"
            )
        for row, column in _mesh_coordinates()[1:]:
            grids[(row, column)] = read_one(row, column)
    finally:
        # `quiesce_devices` exists for exactly this and says so. It runs even on
        # the skip path, because a skipped node id that leaves the parent mesh
        # unclosable fails the *next* one.
        quiesce = getattr(mesh_device, "quiesce_devices", None)
        if callable(quiesce):
            quiesce()
    return grids


def _name_the_device(mesh_device, coordinate: object) -> str:
    """Name a device so a human can act on it: logical id, and the ASIC id if readable.

    The logical id collides across the meshes on a host, which is what
    `get_chip_unique_id_from_fabric_node_id`'s own docstring warns about. The
    ASIC unique id does not -- it is the identity a board fault should be
    reported under, and window 2's faulted machine went unreported partly
    because nothing named the board.
    """

    if not isinstance(coordinate, tuple):
        return f"device index {coordinate}"
    coord = ttnn.MeshCoordinate(*coordinate)
    parts = [f"mesh coord {coordinate}"]
    get_device_id = getattr(mesh_device, "get_device_id", None)
    if callable(get_device_id):
        parts.append(f"logical id {int(get_device_id(coord))}")
    fabric_node_id = getattr(mesh_device, "get_fabric_node_id", None)
    unique_id = getattr(getattr(ttnn, "cluster", None), "get_chip_unique_id_from_fabric_node_id", None)
    if callable(fabric_node_id) and callable(unique_id):
        node = fabric_node_id(coord)
        parts.append(f"ASIC id {unique_id(int(node.mesh_id), int(node.chip_id)):#x}")
    return ", ".join(parts)


@pytest.mark.device
@pytest.mark.blackhole
@pytest.mark.galaxy_bh
@pytest.mark.model
@galaxy_params
@galaxy
@torch.no_grad()
def test_blackhole_galaxy_harvesting_is_uniform_across_the_mesh(mesh_device):
    """A mixed-harvesting mesh would need a per-mesh minimum nothing takes.

    The descriptor is resolved once, from the mesh's own
    `compute_with_storage_grid_size()`. If individual devices disagree, that
    single answer is wrong for some of them -- and the failure is a placement on
    a core that exists on most of the chassis and not on one board, which is the
    silent-on-31-devices class.

    **Answered 2026-09-17, and by the third route.** `MeshDevice.get_devices()`
    is absent on ttnn 0.77.0, which is what blocked this for two windows. The
    `(1, 1)` submesh route that replaced it **segfaults** on that build -- see
    `_grids_via_submeshes`, which is deliberately no longer in the list below,
    because a crashing route is worse than an absent one: it takes the whole
    pytest process with it. The cluster descriptor answers the same question
    without a submesh, and reads the physical cause rather than a derived grid.

    The measured answer is the benign one: **uniform in count, non-uniform in
    position.** Every chip loses exactly one tensix row, from 14 distinct
    positions, and `noc_translation` is `true` everywhere -- so each chip
    presents the same logical `12 x 10` and the descriptor's single mesh-level
    reading is valid for all 32. No per-mesh minimum is needed.

    Which route answered is reported, because "the answer" and "the route that
    produced it" are different facts and the next ttnn may move either.

    A skip here means every route was absent, which is itself the recording grid uniformity
    asks for -- and the job runner counts a skip as a failure, so it reaches a
    human rather than passing as a green.
    """

    routes = (
        ("get_devices", _grids_via_get_devices),
        ("cluster descriptor harvesting", _harvesting_via_cluster_descriptor),
    )
    grids: dict[object, tuple[object, ...]] | None = None
    for name, route in routes:
        grids = route(mesh_device)
        _report(f"per-device route '{name}'", "available" if grids is not None else "absent")
        if grids:
            _report("route used", name)
            break

    if not grids:
        pytest.skip(
            "no per-device geometry route on this ttnn build (tried: "
            f"{', '.join(name for name, _ in routes)}); grid uniformity stays unmeasurable here"
        )

    _report("devices read", len(grids))
    assert len(grids) == mesh_device.get_num_devices(), (
        f"read {len(grids)} devices out of {mesh_device.get_num_devices()}; a partial read cannot "
        "settle uniformity, and the devices it missed are the ones a mixed chassis would hide in"
    )

    distinct = sorted(set(grids.values()))
    # The tuple's meaning is route-dependent -- `(compute_grid, dram_views)` from
    # a per-device grid route, `(harvested_rows, dram_harvesting_mask,
    # noc_translation)` from the descriptor -- so the label does not name fields
    # the active route may not have produced.
    _report("distinct per-device readings across the mesh", distinct)
    if len(distinct) > 1:
        majority = max(distinct, key=lambda value: sum(1 for read in grids.values() if read == value))
        for coordinate, value in sorted(grids.items(), key=repr):
            if value != majority:
                _report(f"DIFFERS {value} vs majority {majority}", _name_the_device(mesh_device, coordinate))

    assert len(distinct) == 1, (
        f"harvesting is not uniform across the {len(grids)} devices: {distinct}. "
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
def test_blackhole_galaxy_dram_banks_agree_with_the_descriptor(mesh_device):
    """The DRAM half of grid uniformity, plus the unmeasured DRAM bank size.

    `ttnn.cluster.get_dram_bank_table(device_id)` returns one entry per DRAM bank
    for an opened device, each carrying `bank_size` -- so it reads per-device DRAM
    geometry through a logical chip id, with no mesh enumeration and no submesh.
    It is the cheapest per-device route of all, and it is **deliberately not** the
    uniformity test above: harvesting removes tensix rows, which this cannot see.

    Two plan rows are settled here. `dram_views` is resolved from
    `dram_grid_size().x` and consumed as a sender count and a shard width, so a
    bank count that disagrees with it is a descriptor error. And the DRAM bank
    size is still **[measure]** in the plan -- carried as Wormhole's 2 GiB against
    a Blackhole expectation of about 3.98 GiB, which nothing has read.
    """

    bank_table = getattr(getattr(ttnn, "cluster", None), "get_dram_bank_table", None)
    get_device_id = getattr(mesh_device, "get_device_id", None)
    if not callable(bank_table) or not callable(get_device_id):
        pytest.skip("this ttnn build exposes no ttnn.cluster.get_dram_bank_table / MeshDevice.get_device_id")

    topology = resolve_galaxy_chip_topology(mesh_device)
    reads: dict[tuple[int, int], tuple[int, tuple[int, ...]]] = {}
    for coordinate in _mesh_coordinates():
        table = bank_table(int(get_device_id(ttnn.MeshCoordinate(*coordinate))))
        reads[coordinate] = (len(table), tuple(sorted({int(entry["bank_size"]) for entry in table})))

    distinct = sorted(set(reads.values()))
    _report("distinct (bank count, bank sizes) across the mesh", distinct)
    _report("descriptor dram_views", topology.dram_views)
    if len(distinct) > 1:
        majority = max(distinct, key=lambda value: sum(1 for read in reads.values() if read == value))
        for coordinate, value in sorted(reads.items()):
            if value != majority:
                _report(f"DIFFERS {value} vs majority {majority}", _name_the_device(mesh_device, coordinate))

    assert len(distinct) == 1, f"DRAM geometry is not uniform across the mesh: {distinct}"
    bank_count, bank_sizes = distinct[0]
    assert bank_count == topology.dram_views, (
        f"the device reports {bank_count} DRAM banks and the descriptor resolves {topology.dram_views} views "
        "from dram_grid_size().x; the descriptor's number is a sender count and a shard width, so it has to "
        "be the one the allocator agrees with"
    )
    # Recorded, not asserted: the plan carries Wormhole's 2 GiB and marks the
    # Blackhole figure [measure]. One reading is what closes that row.
    _report("DRAM bank size(s), bytes", bank_sizes)
    _report("DRAM bank size(s), GiB", tuple(round(size / 1024**3, 3) for size in bank_sizes))


@pytest.mark.device
@pytest.mark.blackhole
@pytest.mark.galaxy_bh
@pytest.mark.model
@galaxy_params
@galaxy
@torch.no_grad()
def test_blackhole_galaxy_link_budget_agrees_across_both_sources(mesh_device):
    """The two independent link tables must report the same number.

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
    """Confirm the L1 small region rather than assuming 32 kB carries.

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
    is costing, which is still open: the reference excludes column 0
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

    # Still open, reported rather than asserted: settling it is a
    # one-line change to BLACKHOLE_FIRST_WORKER_COLUMN.
    _report("BLACKHOLE_FIRST_WORKER_COLUMN", BLACKHOLE_FIRST_WORKER_COLUMN)
    _report(
        "cores column 0 would add if it rejoined the workers",
        int(grid.y) * BLACKHOLE_FIRST_WORKER_COLUMN,
    )
