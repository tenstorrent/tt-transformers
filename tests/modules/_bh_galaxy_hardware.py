# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Shared real-hardware plumbing for Blackhole Galaxy 2D module tests.

The Blackhole analogue of `_wh_galaxy_hardware.py`, and deliberately much
smaller than it. Two facts make it so.

**Milestone 1 on Blackhole is prefetcher-free**, so almost everything the
Wormhole helper exists for -- the 12 real prefetch senders, the 8 dummy senders
that exist only so the global circular buffer covers every remaining worker
core, the global-CB size, the sender/worker sub-device partition -- has no
counterpart here. What replaces it is `NullPrefetcher2D`, which honours the
same register -> seal -> activate -> cleanup protocol and returns contexts whose
`global_cb` is `None`.

**The rest of the plumbing is architecture-neutral**, and is imported from the
Wormhole helper rather than copied. `_CCLOnlySubdeviceOwner`,
`compose_2d_sharded_tensor`, `exact_tensor_resource` and the deallocation
helpers read nothing per-architecture: they work off the mode plan and the
`(8, 4)` mesh shape, which both Galaxy architectures share. A second, subtly
different copy would measure itself rather than the module -- the same reason
the module-resolution recon used the repo's own fakes. The file it imports from
keeps its Wormhole name because renaming a hardware-qualified helper is not
worth the churn in this milestone; the import below is the one place that
asymmetry shows.

**The one genuinely Blackhole-specific thing here is the mode plan**, and it is
the reason this file exists at all. See `bh_galaxy_mode_plan`.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
import torch
import ttnn

# Architecture-neutral plumbing. See the module docstring for why these are
# imported from the Wormhole-named helper rather than duplicated.
from tests.modules._wh_galaxy_hardware import (
    GalaxyHardwareResources,
    _CCLOnlySubdeviceOwner,
    compose_2d_sharded_tensor,
    deallocate_module_weights,
    deallocate_tensor,
    exact_tensor_resource,
    sub_device_only_prefetch_context,
)

from tt_transformers.models.galaxy import GalaxyModePlan, GalaxyResourcesConfig, create_galaxy_resources
from tt_transformers.models.galaxy.prefetch import NullPrefetcher2D
from tt_transformers.models.galaxy.recipes import galaxy_prefill_mode_plan_cores, galaxy_topology, worker_cores
from tt_transformers.models.galaxy.topology import GalaxyChipTopology

__all__ = [
    "bh_galaxy_mode_plan",
    "bh_galaxy_resources_config",
    "bh_galaxy_topology",
    "bh_galaxy_worker_sub_device",
    "compose_2d_sharded_tensor",
    "deallocate_module_weights",
    "deallocate_tensor",
    "exact_tensor_resource",
    "record_module_output",
    "require_bh_galaxy_ccl_resources",
    "require_bh_galaxy_prefetch_free_resources",
    "sub_device_only_prefetch_context",
]


def bh_galaxy_topology(mesh_device: ttnn.MeshDevice) -> GalaxyChipTopology:
    """Return the validated Blackhole Galaxy descriptor, or fail loudly.

    Fails rather than skips on the wrong architecture. A Blackhole suite that
    silently skipped on a Wormhole mesh would make a misconfigured run look
    green, which is the `HF_HOME` failure mode in a different costume.
    """

    architecture = mesh_device.arch()
    if architecture != ttnn.device.Arch.BLACKHOLE:
        pytest.fail(
            f"Blackhole Galaxy suite requires a Blackhole mesh, got {architecture}. "
            "Set MESH_DEVICE=BHGLX and open the mesh with BHGLX_DEVICE_PARAMS."
        )
    topology = galaxy_topology(mesh_device)
    if topology.capabilities.has_prefetcher:
        pytest.fail(
            "the Blackhole descriptor reports has_prefetcher=True; milestone 1 is "
            "prefetcher-free and every recipe in these suites assumes it"
        )
    return topology


def bh_galaxy_mode_plan(
    mode: str,
    collectives: tuple[Any, ...],
    mesh_device: ttnn.MeshDevice,
    *,
    topology: GalaxyChipTopology | None = None,
    semaphore_cores: Any | None = None,
) -> GalaxyModePlan:
    """Build the prefetcher-free Blackhole worker envelope for one mode.

    **The worker cores come from the topology descriptor, not from the compute
    grid**, and that is the whole point of this function. The Wormhole helper's
    `galaxy_mode_plan` uses the full `compute_with_storage_grid_size()`
    rectangle, which is safe there because Wormhole Galaxy's dispatch sits at
    column 7 and above -- outside the 7-wide `galaxy: col:` compute grid. On
    Blackhole the reported grid is 12 x 10 and the column the reference
    excludes, 11, is **inside** it.

    **Be precise about why 11 is excluded, because it is easy to overstate.**
    The descriptor's recorded reason is a *measurement*, and the measured
    symptom is a **prefill-warmup regression with nothing raising** -- the
    reference's words are that folding column 11 into the worker sub-device
    *"regresses prefill warmup"*, and that on a COL-dispatch galaxy col 11 is
    therefore "not a plain worker". Whether a program placed there would be
    *rejected* is **unmeasured on this chassis**, and there is a reason to
    doubt it: `compute_with_storage_grid_size()` reports 12, and a core that
    tt-metal considered a dispatch core would not normally be inside that
    answer. So do not expect `Kernels cannot be placed on dispatch cores!`;
    expect a slower warmup, or nothing.

    The envelope is still the right choice for a sub-device, for the reason the
    descriptor gives: too few worker cores costs throughput and nothing else,
    while too many puts tensors on a column reserved for a reason nobody has
    written down, and every failure in that class is silent. It is conservative
    by design, not defensive against a predicted abort.

    The descriptor carries both decisions -- `reserved_columns=(11,)` and the
    column-0 one (`BLACKHOLE_FIRST_WORKER_COLUMN = 1`, 100
    cores) -- so reading the envelope from it gets this right without naming
    either number here, and if a timed prefill moves a constant, every suite
    moves with it and none needs editing.

    The cheap experiment that settles the legality question is a suite which
    loads **no** sub-device manager and lets an op auto-grid over the whole
    reported 12 x 10 -- which is what the Embedding2D and RoPE prefill suites
    do. If those run, a full-grid program is legal here and the exclusion is
    purely about warmup.

    **One sub-device, not two.** With no prefetcher there is no sender
    partition, so the whole envelope is the worker sub-device and
    `SubDeviceId(0)` is the worker. That is the shape §6.3 designed for the
    prefetcher-free path.

    **Prefill gets the full compute grid, which is what production gives it.**
    `plans.galaxy_prefill_mode_plan` builds its single prefill sub-device from
    `recipes.galaxy_prefill_mode_plan_cores` -- literally
    `CoreRange((0, 0), (grid.x - 1, grid.y - 1))` -- on both architectures, so
    calling the same helper here is parity rather than a choice this file makes.
    It matters much more on Blackhole than it reads: Wormhole's compute grid is
    the prefill partition, while Blackhole's is two columns wider than the
    worker envelope.

    **Handing prefill the decode envelope instead makes the whole interleaved
    prefill path unplaceable**, which is why this is not the conservative
    option it looks like. A ttnn op given an interleaved input takes its cores
    from `split_work_to_cores(device->compute_with_storage_grid_size(), ...)`,
    whose walk starts at `(0, 0)` and, row-wise, crosses the full grid width --
    so it lands on column 0 immediately and on column 11 as soon as it needs
    twelve cores. None of these ops has a parameter to confine it:
    `nlp_create_qkv_heads`, `nlp_concat_heads`, `fill_cache`,
    `paged_fill_cache`, `update_cache`'s interleaved branch and `typecast` take
    no `sub_core_grids` at all, and prefill SDPA's program factory accepts the
    argument but never reads it -- only `sdpa_decode`'s does. Under a
    sub-device anchored at `(1, 0)` every one of them aborts on *"Kernel group
    cores do not match sub device cores"* before an element moves. Decode is
    unaffected because its tensors are sharded, and a sharded input makes each
    of those ops read the shard spec's own grid instead.

    Placing kernels across the whole grid is measured rather than assumed: the
    embedding and RoPE prefill node ids load no sub-device manager, auto-grid
    over the reported 12 x 10, and pass (log §10.1). The default sub-device they
    ran under covers exactly these cores; `SubDeviceManager::validate_sub_devices`
    asks only that a sub-device's tensix cores lie inside the compute grid and
    rejects nothing for being dispatch. What the reference does record as a cost
    is a **prefill warmup regression** from folding column 11 into a worker
    partition -- a performance reading, and the price of this being placeable at
    all. The expensive ops keep their confinement anyway: every matmul here
    still goes through `dense_matmul_program_config(..., topology)`, which
    rectangles inside `worker_cores()` and so touches neither column 0 nor 11.

    Decode keeps the descriptor's envelope, so the column-0 question and
    every reading already taken under it are untouched.

    `semaphore_cores` narrows where the mode's global semaphores are allocated.
    Leave it at the default for the generic async CCLs: they choose sender
    worker cores from the worker sub-device minus the reserved output cores, so
    a semaphore narrower than the sub-device leaves a sender polling an L1
    address its own core never had reserved or zeroed, which **hangs** rather
    than fails. Narrow it only for a collective that binds its semaphore to a
    grid it owns.
    """

    resolved = topology if topology is not None else bh_galaxy_topology(mesh_device)
    # Both core sets come from `recipes`, and it matters that neither is built
    # here: `CoreRangeSet` preserves *constructor* sequence, so the same cores
    # built in a different order are unequal objects with a different iteration
    # order, and ring kernels depend on that order. A second construction path
    # for the same geometry is the hazard, whichever geometry it is.
    #
    # `galaxy_prefill_mode_plan_cores` reads the grid off the mesh rather than
    # the descriptor, which is the same source `_resolve_blackhole` derives the
    # descriptor from, so the two cannot disagree about the width.
    cores = galaxy_prefill_mode_plan_cores(mesh_device) if mode == "prefill" else worker_cores(resolved)
    worker_id = ttnn.SubDeviceId(0)
    return GalaxyModePlan(
        mode=mode,
        sub_devices=(ttnn.SubDevice([cores]),),
        worker_sub_device_id=worker_id,
        stall_group=(worker_id,),
        semaphore_cores=semaphore_cores or cores,
        worker_cores=cores,
        allow_narrow_semaphore_cores=semaphore_cores is not None,
        collectives=collectives,
    )


@contextmanager
def bh_galaxy_worker_sub_device(
    mesh_device: ttnn.MeshDevice,
    *,
    topology: GalaxyChipTopology | None = None,
) -> Any:
    """Load the Blackhole worker envelope as a sub-device, for a module with no collective.

    `Embedding2D` and `RotarySetup2D` issue no collective, and
    `GalaxyModePlan.__post_init__` refuses a plan with `collectives=()` -- *"at
    least one collective is required"* -- so they cannot reach the envelope
    through `bh_galaxy_mode_plan`. Inventing a placeholder collective to get
    past that check would put an unused persistent buffer on the mesh and make
    the plan describe something the test does not do.

    So this is the narrow route: the same geometry, from the same converter, with
    no collective and no resource owner. It exists so all six suites get their
    envelope from one place. `CoreRangeSet` preserves constructor sequence and
    ring kernels depend on that order, so a second construction path for the
    same cores is exactly the hazard worth avoiding even where no ring is
    involved.

    Yields the `CoreRangeSet` and the `SubDeviceId`, and unwinds in the reverse
    order on the way out -- stall group, loaded manager, then the manager
    itself. Getting that order wrong leaves the mesh with a manager it cannot
    name, and `ttnn.close_mesh_device` hangs after a nominally green test.
    """

    resolved = topology if topology is not None else bh_galaxy_topology(mesh_device)
    cores = worker_cores(resolved)
    worker_id = ttnn.SubDeviceId(0)
    manager = mesh_device.create_sub_device_manager([ttnn.SubDevice([cores])], 0)
    loaded = False
    try:
        mesh_device.load_sub_device_manager(manager)
        loaded = True
        mesh_device.set_sub_device_stall_group([worker_id])
        yield cores, worker_id
    finally:
        try:
            if loaded:
                mesh_device.reset_sub_device_stall_group()
                mesh_device.clear_loaded_sub_device_manager()
        finally:
            mesh_device.remove_sub_device_manager(manager)


def bh_galaxy_resources_config(
    mesh_device: ttnn.MeshDevice,
    *,
    prefill: GalaxyModePlan,
    decode: GalaxyModePlan,
) -> GalaxyResourcesConfig:
    """Build the resources config with the architecture read off the mesh.

    Read off the mesh rather than passed in, so a Blackhole suite cannot
    accidentally construct a Wormhole-declared config that then resolves
    Wormhole geometry and passes for the wrong reason. The architecture gate in
    `GalaxyResourcesConfig.__post_init__` accepts both Galaxy architectures as
    of phase 5 item 0, so it is the caller's job not to lie to it.
    """

    return GalaxyResourcesConfig(architecture=mesh_device.arch(), prefill=prefill, decode=decode)


def require_bh_galaxy_ccl_resources(
    mesh_device: ttnn.MeshDevice,
    *,
    config: GalaxyResourcesConfig,
) -> GalaxyHardwareResources:
    """Create Galaxy CCL resources over the Blackhole envelope, no producer.

    The sub-device owner is the Wormhole helper's `_CCLOnlySubdeviceOwner`,
    which is architecture-neutral: it creates one sub-device manager per mode
    from the plans it is given and loads the one it is told to. All the
    Blackhole-specific content is already in the plans.
    """

    subdevices = _CCLOnlySubdeviceOwner(mesh_device, config)
    try:
        owner = create_galaxy_resources(mesh_device=mesh_device, config=config, prefetcher=subdevices)
    except Exception:
        subdevices.cleanup()
        raise
    return GalaxyHardwareResources(owner=owner, ccl=owner.ccl, prefetcher=subdevices)


class _PrefetchFreeOwner:
    """Own the sub-device managers, serve `NullPrefetcher2D`'s contexts.

    Two collaborators have to be one object here, and finding out why is worth
    recording, because getting it wrong produces a test that passes every
    assertion it makes and then hangs in teardown.

    `GalaxyResources` calls exactly one thing when a mode becomes active:
    `self._prefetcher.activate(mode)`. On Wormhole that is the real
    `Prefetcher2D`, which owns its sub-device manager and loads it there; in
    the Wormhole CCL-only test route it is `_CCLOnlySubdeviceOwner`, which does
    the same. `NullPrefetcher2D` deliberately does **neither** -- it owns no
    manager because it owns no global circular buffer, and says so. So handing
    `NullPrefetcher2D` to `create_galaxy_resources` on its own leaves the
    manager never loaded, and every program then runs against whatever
    sub-device is loaded rather than the worker envelope.

    The second half is teardown. `GalaxyResources.cleanup()` synchronizes and
    cleans the CCL and does **not** touch the injected prefetcher, so the only
    thing that reaches a sub-device owner is whatever
    `GalaxyHardwareResources.cleanup()` finds in its three slots. An object
    that is not in one of them never resets the stall group and never removes
    its managers -- which is the D-2 shape exactly: assertions pass, then
    `ttnn.close_mesh_device` hangs after the test is nominally green.

    So: this owns both, loads the manager on `activate`, and serves the null
    prefetcher's `global_cb=None` contexts to the module.
    """

    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        config: GalaxyResourcesConfig,
        *,
        weights: tuple[tuple[str, Any], ...] = (),
    ):
        # `_CCLOnlySubdeviceOwner.__init__` already creates the sub-device
        # managers, so from this line on there is something to clean up and
        # every later failure has to reach `cleanup()`. `_null` is bound before
        # the `try` for the same reason -- `cleanup()` must not raise
        # `AttributeError` and mask the real exception.
        self._subdevices = _CCLOnlySubdeviceOwner(mesh_device, config)
        self._null: Any = None
        try:
            self._null = NullPrefetcher2D(mesh_device, config, expected_weight_count=len(weights))
            self._null.initialize()
            for name, tensor in weights:
                self._null.register_weight(name, tensor)
            self._null.seal()
        except Exception:
            self.cleanup()
            raise

    @property
    def mesh_device(self) -> Any:
        return self._null.mesh_device

    def context(self, mode: str) -> Any:
        return self._null.context(mode)

    def borrow_context(self, mode: str, **policy: Any) -> Any:
        # Validated by both: the sub-device owner checks the partition matches
        # the resolved plan, and the null prefetcher checks the worker
        # sub-device id and stall group. A module that disagrees about the
        # partition places tensors on cores the loaded manager does not own and
        # aborts with "Kernel group cores do not match sub device cores", so
        # neither check is redundant.
        self._subdevices.borrow_context(mode, **policy)
        return self._null.borrow_context(mode, **policy)

    def activate(self, mode: str) -> Any:
        # Order matters: load the manager, then hand back the context the
        # module reads. The reverse would return a context naming a
        # sub-device that is not yet loaded.
        self._subdevices.activate(mode)
        return self._null.activate(mode)

    def cleanup(self) -> None:
        try:
            if self._null is not None:
                self._null.cleanup()
        finally:
            # Always, even if the null prefetcher's cleanup raised: this is
            # what resets the stall group and removes the managers, and
            # skipping it is what leaves the mesh undrainable.
            self._subdevices.cleanup()


def require_bh_galaxy_prefetch_free_resources(
    mesh_device: ttnn.MeshDevice,
    *,
    config: GalaxyResourcesConfig,
    weights: tuple[tuple[str, Any], ...] = (),
) -> GalaxyHardwareResources:
    """Create resources whose prefetch contexts exist but carry no global CB.

    Use this where a module reads a *prefetch* context -- for its
    `sub_device_id`, or to decide whether a global circular buffer is available
    -- rather than only a CCL context. `NullPrefetcher2D` is the production
    answer, not a test double: weights stay DRAM-interleaved, `global_cb` is
    `None`, and `launch_sender` is a no-op, so nothing has to be drained.

    That last property is why this is the safe default on Blackhole. On
    Wormhole, a suite that starts the DRAM producer must also *drain* it, or
    `ttnn.close_mesh_device` hangs in the fixture after the test's own
    assertions have already passed -- defect D-2's residue, and the reason
    `attention_decode_with_active_prefetch` is a known-red node id. There is no
    producer here, so that failure mode does not exist.

    The sub-device managers still have to be created and loaded by someone, and
    `NullPrefetcher2D` deliberately does not do it. `_PrefetchFreeOwner` is the
    object that does both; read its docstring before changing anything here,
    because the failure mode of getting it wrong is a green test that hangs in
    teardown.
    """

    owner_adapter = _PrefetchFreeOwner(mesh_device, config, weights=weights)
    try:
        owner = create_galaxy_resources(mesh_device=mesh_device, config=config, prefetcher=owner_adapter)
    except Exception:
        owner_adapter.cleanup()
        raise
    # `prefetcher=owner_adapter` so `GalaxyHardwareResources.cleanup()` reaches
    # the sub-device managers; `prefetch_context(mode)` then resolves through
    # the adapter to the null prefetcher's `global_cb=None` context.
    return GalaxyHardwareResources(owner=owner, ccl=owner.ccl, prefetcher=owner_adapter)


#: Where `record_module_output` writes, when the harness asks it to.
#:
#: Unset means "do not record", so the same node id runs identically under a
#: bare `pytest` and under the three-process protocol below.
_OUTPUT_DIR_ENV = "BH_GALAXY_OUTPUT_DIR"
_RUN_TAG_ENV = "BH_GALAXY_RUN_TAG"


def record_module_output(name: str, *tensors: torch.Tensor) -> Path | None:
    """Save this process's outputs so a later process can compare them.

    **Three PCC passes prove much less than one byte-identical triple**, and
    this is the mechanism for the latter. The fused-RMSNorm statistics circular
    buffer binds to the first core of the norm input shard grid and aliases
    whatever the allocator left there; it has produced PCC of 0.0977 / 0.1394 /
    0.1701 / 0.9999 across processes **on an unchanged test**. A suite that
    asserts PCC >= 0.99 three times would have passed on three of those four.

    The comparison deliberately does **not** live here. One pytest node id per
    process is not a style rule -- the ttnn program cache belongs to the mesh
    device and the process, and the weight cache is keyed on
    `MeshDevice.id()` -- so "the same test, three fresh processes" is a
    property of the *run*, not of a test. The job script runs the node id three
    times with a different `BH_GALAXY_RUN_TAG` and then compares the files with
    `torch.equal`. Recording is all that can honestly happen in-process.

    Returns the path written, or `None` when recording is switched off.
    """

    directory = os.environ.get(_OUTPUT_DIR_ENV, "").strip()
    if not directory:
        return None
    tag = os.environ.get(_RUN_TAG_ENV, "").strip() or "untagged"
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    path = target / f"{name}.{tag}.pt"
    # `.cpu()` and `.contiguous()` so the saved bytes depend on the values and
    # not on how the composer happened to stride the result.
    torch.save([tensor.detach().cpu().contiguous() for tensor in tensors], path)
    return path
