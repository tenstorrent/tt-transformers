# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Device topology naming helpers shared by TTTv2 modules."""

from __future__ import annotations

import gc

import ttnn

from tt_transformers.modules.lazy_weight import LazyWeight

#: The ``L1_SMALL`` region every Galaxy mesh is opened with.
#:
#: The generic collectives `ttnn.reduce_scatter` and `ttnn.all_gather` create their
#: internal global synchronisation semaphores when their program is **compiled**,
#: and those 32-byte allocations live for the life of the program cache entry. With
#: no ``L1_SMALL`` bank they land in main L1, wherever the transients live at that
#: moment leave room - mid-bank - and nothing can be placed across them afterwards.
#: On Galaxy that breaks the weight prefetcher outright: its global circular buffer
#: is released for prefill and can then neither be restored over the stranded
#: semaphores nor placed below them, so `prefill -> decode` - the serving loop -
#: fails with either the restore guard or an allocator OOM.
#:
#: 32 kB is far more than the semaphores need (about five 32-byte semaphores per
#: distinct collective program, so a few hundred bytes for a whole model); it is
#: the smallest round size that leaves the region's sizing a non-question, and it
#: is the size the fix was measured with. Main L1 shrinks by exactly this much for
#: every Galaxy op, which is why the number lives here once rather than per suite.
GALAXY_L1_SMALL_SIZE = 32768


def is_blackhole() -> bool:
    return "blackhole" in ttnn.get_arch_name()


def has_l1_small_region(mesh_device: ttnn.MeshDevice) -> bool:
    """Return whether this mesh was opened with a non-empty ``L1_SMALL`` bank.

    The caller's question is always "can a collective put its semaphores out of
    main L1", so the mesh is asked rather than assumed: `ttnn.reduce_scatter` takes
    `use_l1_small_for_semaphores` and allocates from ``L1_SMALL``
    *unconditionally* when told to, so passing a blind `True` to a mesh opened
    without the region turns a fragmentation warning into an allocation failure.
    `ttnn.all_gather` makes exactly this check for itself
    (`all_gather_multicast_factory.cpp` ~35: bank size ``> 0`` or fall back to L1
    with a warning); this is the same rule, one level up, for the ops that do not.
    """

    try:
        return ttnn.get_memory_view(mesh_device, ttnn.BufferType.L1_SMALL).total_bytes_per_bank > 0
    except Exception:  # noqa: BLE001 - an allocator that cannot describe the region does not have one
        return False


def get_device_name(mesh_device: ttnn.MeshDevice, num_devices: int | None = None) -> str:
    """Return the product/topology name for a TT mesh device.

    By default, the full mesh device count is used. CCL callers can pass a
    host-local device count when they need link-count tuning for the current
    process rather than for the full mesh.
    """
    num_devices = mesh_device.get_num_devices() if num_devices is None else num_devices
    dram_grid_size = mesh_device.dram_grid_size()

    if ttnn.device.is_blackhole(mesh_device):
        device_names = {
            1: "P100" if dram_grid_size and dram_grid_size.x == 7 else "P150",
            2: "P300",
            4: "P150x4",
            8: "P150x8",
            32: "BHGLX",
        }
    elif ttnn.device.is_wormhole_b0(mesh_device):
        device_names = {
            1: "N150",
            2: "N300",
            4: "N150x4",
            8: "T3K",
            32: "TG",
        }
    else:
        raise ValueError(f"Unsupported architecture: {ttnn.get_arch_name()}")

    if num_devices in device_names:
        return device_names[num_devices]
    raise ValueError(f"Unsupported number of devices: {num_devices} for {ttnn.get_arch_name()}")


def cleanup_ttnn_value(value):
    if value is None:
        return

    if isinstance(value, ttnn.Tensor):
        ttnn.deallocate(value)
        return

    if isinstance(value, dict):
        for nested_value in value.values():
            cleanup_ttnn_value(nested_value)
        return

    if isinstance(value, (list, tuple, set)):
        for nested_value in value:
            cleanup_ttnn_value(nested_value)


def cleanup_object_graph(obj, seen=None):
    if obj is None:
        return
    if seen is None:
        seen = set()

    obj_id = id(obj)
    if obj_id in seen:
        return
    seen.add(obj_id)

    if isinstance(obj, ttnn.Tensor):
        cleanup_ttnn_value(obj)
        return

    if isinstance(obj, LazyWeight):
        if obj._value is not None:
            cleanup_ttnn_value(obj._value)
            obj._value = None
        return

    if isinstance(obj, dict):
        for value in obj.values():
            cleanup_object_graph(value, seen)
        return

    if isinstance(obj, (list, tuple, set)):
        for value in obj:
            cleanup_object_graph(value, seen)
        return

    state = getattr(obj, "__dict__", None)
    if state is None:
        return

    for name, value in list(state.items()):
        cleanup_object_graph(value, seen)
        if isinstance(value, ttnn.Tensor):
            setattr(obj, name, None)

    if hasattr(obj, "_device_weights_loaded"):
        obj._device_weights_loaded = False


def cleanup_model_case(model, mesh_device):
    ttnn.synchronize_device(mesh_device)
    if model is not None:
        cleanup_object_graph(model)
    ttnn.synchronize_device(mesh_device)
    gc.collect()


def cleanup_dp_model_case(group, lanes, models, parent_mesh, submeshes):
    """Release one carved DP case and restore command-queue ownership to its parent."""

    failures = []

    def run(action):
        try:
            action()
        except Exception as error:
            failures.append(error)

    if group is not None:
        run(group.cleanup)
    else:
        for lane in lanes:
            run(lane.cleanup)

    for model, submesh in models:
        run(lambda model=model, submesh=submesh: cleanup_model_case(model, submesh))

    # A carved child owns a distinct MeshCommandQueue over its parent's physical
    # devices. Drain every live child through the parent before closing the child
    # handles and returning command-queue ownership to the fixture-owned parent.
    run(parent_mesh.quiesce_devices)
    for submesh in submeshes:
        run(lambda submesh=submesh: ttnn.close_mesh_device(submesh))

    if failures:
        primary = failures[0]
        add_note = getattr(primary, "add_note", None)
        if add_note is not None:
            for failure in failures[1:]:
                add_note(f"additional DP cleanup failure: {type(failure).__name__}: {failure}")
        raise primary
