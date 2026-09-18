# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Host tests for the Blackhole Galaxy hardware helper.

`_bh_galaxy_hardware.py` is test plumbing, which normally would not get tests
of its own. It gets them because six independently-written module suites build
on it, so a defect there is six wrong results rather than one, and because two
real defects were already found in it by inspection:

1. **`activate` did not load the sub-device manager.** `GalaxyResources`
   calls exactly one thing when a mode becomes active --
   `self._prefetcher.activate(mode)` -- and `NullPrefetcher2D.activate` only
   records the mode, because it owns no manager. Handed to
   `create_galaxy_resources` alone it left every program running against
   whichever sub-device happened to be loaded.
2. **`cleanup` did not remove the managers.** `GalaxyResources.cleanup()`
   synchronizes and cleans the CCL and does not touch the injected prefetcher,
   so an owner that is not in one of `GalaxyHardwareResources`' three slots
   never resets the stall group. That is the D-2 shape: every assertion passes
   and then `ttnn.close_mesh_device` hangs, and the node id fails for a reason
   unrelated to the module under test.

Neither is visible in a passing module suite, which is exactly why they are
asserted here. The mesh is a stand-in that records the sub-device calls, so
these run on the host -- no device, and no `ttnn` beyond the types.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import ttnn
from tests.modules._bh_galaxy_hardware import (
    _PrefetchFreeOwner,
    bh_galaxy_mode_plan,
    bh_galaxy_resources_config,
    record_module_output,
)

from tt_transformers.models.galaxy import GalaxyCollectivePlan, GalaxyResourceKey, GalaxyTensorSpec
from tt_transformers.models.galaxy.recipes import worker_cores
from tt_transformers.models.galaxy.topology import BLACKHOLE_GALAXY_COMPUTE_GRID, galaxy_chip_topology

BLACKHOLE = ttnn.device.Arch.BLACKHOLE
BLACKHOLE_TOPOLOGY = galaxy_chip_topology(BLACKHOLE, compute_grid=BLACKHOLE_GALAXY_COMPUTE_GRID)


class RecordingMesh:
    """A Galaxy mesh that records every sub-device call made against it.

    Records rather than asserts, so a test can state the *sequence* it
    expects. Both defects above are sequence defects -- a missing
    `load-manager`, a missing `remove-manager` -- and neither is expressible as
    a check on a single call.
    """

    def __init__(self, arch=BLACKHOLE, grid=BLACKHOLE_GALAXY_COMPUTE_GRID):
        self.shape = (8, 4)
        self.events: list[tuple] = []
        self._grid = grid
        self._arch = arch
        self._index = 0

    def get_num_devices(self):
        return 32

    def arch(self):
        return self._arch

    def compute_with_storage_grid_size(self):
        return ttnn.CoreCoord(self._grid[0], self._grid[1])

    def dram_grid_size(self):
        return SimpleNamespace(x=8, y=1)

    def create_sub_device_manager(self, sub_devices, local_l1_size):
        manager = f"manager-{self._index}"
        self._index += 1
        self.events.append(("create-manager", manager, len(sub_devices), local_l1_size))
        return manager

    def load_sub_device_manager(self, manager):
        self.events.append(("load-manager", manager))

    def set_sub_device_stall_group(self, stall_group):
        self.events.append(("stall-group", tuple(stall_group)))

    def reset_sub_device_stall_group(self):
        self.events.append(("reset-stall-group",))

    def clear_loaded_sub_device_manager(self):
        self.events.append(("clear-manager",))

    def remove_sub_device_manager(self, manager):
        self.events.append(("remove-manager", manager))

    def kinds(self):
        return [event[0] for event in self.events]


def _collective(sequence=32):
    return GalaxyCollectivePlan(
        key=GalaxyResourceKey("all_gather", 1, (1, 1, sequence, 32), sequence),
        topology=ttnn.Topology.Ring,
        num_links=BLACKHOLE_TOPOLOGY.fabric_links,
        semaphores_per_slot=1,
        persistent_output_specs=(
            GalaxyTensorSpec((1, 1, sequence, 128), ttnn.bfloat16, ttnn.TILE_LAYOUT, ttnn.DRAM_MEMORY_CONFIG),
        ),
    )


def _config(mesh):
    return bh_galaxy_resources_config(
        mesh,
        prefill=bh_galaxy_mode_plan("prefill", (_collective(128),), mesh, topology=BLACKHOLE_TOPOLOGY),
        decode=bh_galaxy_mode_plan("decode", (_collective(32),), mesh, topology=BLACKHOLE_TOPOLOGY),
    )


# --------------------------------------------------------------------------- #
# The mode plan -- the one genuinely Blackhole-specific thing in the helper
# --------------------------------------------------------------------------- #


@pytest.mark.host
def test_the_mode_plan_uses_the_descriptor_envelope_not_the_compute_grid():
    """100 cores, not 120, and the 20 missing ones are named.

    The Wormhole helper's `galaxy_mode_plan` uses the full compute-grid
    rectangle, which is safe there because Wormhole Galaxy's dispatch sits at
    column 7 and above -- outside its 7-wide grid. On Blackhole the grid is
    12 x 10 and dispatch is column 11, *inside* it, so the same plan would put
    worker kernels on dispatch cores.
    """

    mesh = RecordingMesh()
    plan = bh_galaxy_mode_plan("decode", (_collective(),), mesh, topology=BLACKHOLE_TOPOLOGY)

    assert plan.worker_cores.num_cores() == 100
    grid = mesh.compute_with_storage_grid_size()
    assert grid.x * grid.y == 120
    # Column 11 (dispatch) and column 0 are the two excluded ones.
    for core_range in plan.worker_cores.ranges():
        assert core_range.start.x >= 1
        assert core_range.end.x <= 10


@pytest.mark.host
def test_the_mode_plan_is_one_subdevice_with_subdevice_zero_as_worker():
    """No prefetcher means no sender partition, so there is one sub-device.

    Wormhole's prefetch decode plan has two (senders at `SubDeviceId(0)`,
    workers at `SubDeviceId(1)`). Carrying that shape here would name a worker
    sub-device that does not exist.
    """

    mesh = RecordingMesh()
    plan = bh_galaxy_mode_plan("decode", (_collective(),), mesh, topology=BLACKHOLE_TOPOLOGY)

    assert len(plan.sub_devices) == 1
    assert plan.worker_sub_device_id == ttnn.SubDeviceId(0)
    assert plan.stall_group == (ttnn.SubDeviceId(0),)


@pytest.mark.host
def test_the_mode_plan_semaphores_cover_the_whole_worker_subdevice_by_default():
    """Defect D3: a narrower semaphore set hangs the collective, not fails it.

    The generic async CCLs pick sender worker cores from the worker sub-device
    minus the reserved output cores, so a global semaphore allocated on a
    narrower set leaves a sender polling an L1 address its own core never had
    reserved or zeroed. That cost four consecutive 2700 s timeouts on Wormhole
    *after* one process had already passed.
    """

    mesh = RecordingMesh()
    plan = bh_galaxy_mode_plan("decode", (_collective(),), mesh, topology=BLACKHOLE_TOPOLOGY)

    assert plan.allow_narrow_semaphore_cores is False
    assert plan.worker_cores.subtract(plan.semaphore_cores).num_cores() == 0


@pytest.mark.host
def test_narrowing_the_semaphores_requires_saying_so():
    """Narrowing is legitimate only for a collective that owns its grid.

    It cannot be inferred from the plan, because both the safe and unsafe forms
    key on the same `all_gather` operation name -- so the opt-in is explicit,
    and the helper sets it only when a caller passed `semaphore_cores`.
    """

    mesh = RecordingMesh()
    narrow = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(2, 0))})
    plan = bh_galaxy_mode_plan("decode", (_collective(),), mesh, topology=BLACKHOLE_TOPOLOGY, semaphore_cores=narrow)
    assert plan.allow_narrow_semaphore_cores is True
    assert plan.semaphore_cores is narrow


@pytest.mark.host
def test_the_mode_plan_uses_the_canonical_core_range_converter():
    """`CoreRangeSet` preserves *constructor* sequence, so order is identity.

    The same cores built in a different order are unequal objects with a
    different iteration order, and ring kernels depend on that order. Building
    rings from sets produced bit-for-bit identical *wrong* output across runs
    -- stable rather than noisy, which is what made it look unlike a race for
    weeks. So the plan's cores must be the descriptor's own conversion, not a
    second construction of the same geometry.
    """

    mesh = RecordingMesh()
    plan = bh_galaxy_mode_plan("decode", (_collective(),), mesh, topology=BLACKHOLE_TOPOLOGY)
    assert plan.worker_cores == worker_cores(BLACKHOLE_TOPOLOGY)


@pytest.mark.host
def test_the_resources_config_reads_the_architecture_off_the_mesh():
    """A Blackhole suite must not be able to declare itself Wormhole.

    Both architectures now pass `GalaxyResourcesConfig`'s gate, so a config
    that declared the wrong one would resolve the wrong geometry and pass for
    the wrong reason. Reading it off the mesh removes the opportunity.
    """

    assert _config(RecordingMesh()).architecture == BLACKHOLE


# --------------------------------------------------------------------------- #
# _PrefetchFreeOwner -- the two defects
# --------------------------------------------------------------------------- #


@pytest.mark.host
def test_activate_loads_the_subdevice_manager():
    """Defect 1. `NullPrefetcher2D.activate` alone loads nothing."""

    mesh = RecordingMesh()
    owner = _PrefetchFreeOwner(mesh, _config(mesh))
    try:
        # Two managers, one per mode, created up front.
        assert mesh.kinds().count("create-manager") == 2
        assert "load-manager" not in mesh.kinds()

        context = owner.activate("decode")

        assert "load-manager" in mesh.kinds(), "activate did not load the sub-device manager"
        assert "stall-group" in mesh.kinds(), "activate did not set the stall group"
        # And the context the module actually reads is the null prefetcher's,
        # so a module asking whether a global CB exists gets `None`.
        assert context.global_cb is None
        assert context.worker_sub_device_id == ttnn.SubDeviceId(0)
    finally:
        owner.cleanup()


@pytest.mark.host
def test_cleanup_resets_the_stall_group_and_removes_every_manager():
    """Defect 2. Skipping this is a green test that hangs in teardown."""

    mesh = RecordingMesh()
    owner = _PrefetchFreeOwner(mesh, _config(mesh))
    owner.activate("decode")
    owner.cleanup()

    kinds = mesh.kinds()
    assert "reset-stall-group" in kinds, "cleanup left the stall group set"
    assert "clear-manager" in kinds, "cleanup left a sub-device manager loaded"
    # Both managers, not just the active one.
    assert kinds.count("remove-manager") == 2, f"cleanup removed {kinds.count('remove-manager')} of 2 managers"


@pytest.mark.host
def test_cleanup_is_idempotent():
    """Teardown runs from a `finally` that may already have run."""

    mesh = RecordingMesh()
    owner = _PrefetchFreeOwner(mesh, _config(mesh))
    owner.activate("decode")
    owner.cleanup()
    removed = mesh.kinds().count("remove-manager")
    owner.cleanup()
    assert mesh.kinds().count("remove-manager") == removed


@pytest.mark.host
def test_the_prefetch_context_carries_no_global_cb_and_still_names_the_subdevice():
    """The whole point of `NullPrefetcher2D`: absence expressed, not implied.

    A module confined to a narrowed partition reads both `global_cb` and
    `worker_sub_device_id` off its prefetch context, so the partition can only
    reach it through one of them. `global_cb=None` says "no ring";
    `worker_sub_device_id` still says which sub-device to name, and every
    program enqueued must name exactly one or `fd_mesh_command_queue.cpp`
    refuses the workload.
    """

    mesh = RecordingMesh()
    owner = _PrefetchFreeOwner(mesh, _config(mesh))
    try:
        for mode in ("prefill", "decode"):
            context = owner.context(mode)
            assert context.mode == mode
            assert context.global_cb is None
            assert context.sub_device_manager_id is None
            assert context.worker_sub_device_id == ttnn.SubDeviceId(0)
    finally:
        owner.cleanup()


@pytest.mark.host
def test_borrow_context_refuses_a_borrower_that_disagrees_about_the_partition():
    """Both collaborators validate, and neither check is redundant.

    A module that disagrees about the partition places tensors on cores the
    loaded manager does not own and aborts with "Kernel group cores do not
    match sub device cores".
    """

    mesh = RecordingMesh()
    config = _config(mesh)
    owner = _PrefetchFreeOwner(mesh, config)
    try:
        plan = config.decode
        # The honest borrow succeeds.
        assert (
            owner.borrow_context(
                "decode",
                sub_devices=plan.sub_devices,
                worker_sub_device_id=plan.worker_sub_device_id,
                stall_group=plan.stall_group,
                local_l1_size=plan.local_l1_size,
            )
            is not None
        )
        # A borrower naming the wrong worker sub-device does not.
        with pytest.raises(ValueError):
            owner.borrow_context(
                "decode",
                sub_devices=plan.sub_devices,
                worker_sub_device_id=ttnn.SubDeviceId(1),
                stall_group=(ttnn.SubDeviceId(1),),
                local_l1_size=plan.local_l1_size,
            )
    finally:
        owner.cleanup()


@pytest.mark.host
def test_a_failure_during_construction_does_not_leak_the_managers():
    """`_CCLOnlySubdeviceOwner.__init__` creates managers, so it owns cleanup.

    From that line on every later failure has to reach `cleanup()`, or the
    caller gets an exception *and* a mesh with two managers it can no longer
    name. A duplicate weight name is the cheapest way to fail inside the
    constructor's `try`.
    """

    mesh = RecordingMesh()
    config = _config(mesh)
    assert mesh.kinds().count("create-manager") == 0

    with pytest.raises(ValueError, match="already registered"):
        _PrefetchFreeOwner(mesh, config, weights=(("dup", object()), ("dup", object())))

    assert mesh.kinds().count("create-manager") == 2
    assert mesh.kinds().count("remove-manager") == 2, "construction failure leaked the sub-device managers"


# --------------------------------------------------------------------------- #
# record_module_output
# --------------------------------------------------------------------------- #


@pytest.mark.host
def test_recording_is_off_unless_the_harness_asks(monkeypatch):
    """A bare `pytest` of the same node id must behave identically.

    The three-process protocol is the job script's; the test only records when
    told to, so nothing about a local run differs from a harness run except the
    files on disk.
    """

    monkeypatch.delenv("BH_GALAXY_OUTPUT_DIR", raising=False)
    import torch

    assert record_module_output("anything", torch.zeros(2)) is None


@pytest.mark.host
def test_recording_writes_one_file_per_run_tag(tmp_path, monkeypatch):
    """The filename carries the tag, because the comparison is across files.

    Two parametrizations of one test must not collide, which is why the caller
    supplies a name that includes its parameters and this only appends the tag.
    """

    import torch

    monkeypatch.setenv("BH_GALAXY_OUTPUT_DIR", str(tmp_path))
    first = torch.randn(3, 4)
    second = torch.randn(3, 4)

    monkeypatch.setenv("BH_GALAXY_RUN_TAG", "p1")
    path_one = record_module_output("rms_llama8192", first, second)
    monkeypatch.setenv("BH_GALAXY_RUN_TAG", "p2")
    path_two = record_module_output("rms_llama8192", first, second)

    assert path_one is not None and path_two is not None
    assert path_one != path_two
    assert path_one.name == "rms_llama8192.p1.pt"
    assert path_two.name == "rms_llama8192.p2.pt"

    reloaded = torch.load(path_one, map_location="cpu", weights_only=False)
    assert len(reloaded) == 2
    assert torch.equal(reloaded[0], first)
    assert torch.equal(reloaded[1], second)
    # Contiguous on disk, so the saved bytes depend on the values rather than
    # on how the mesh composer happened to stride the result.
    assert all(tensor.is_contiguous() for tensor in reloaded)
