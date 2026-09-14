# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Canonical WH Galaxy `Prefetcher2D` construction policy.

The sender/receiver mapping, dummy-core padding, and global circular-buffer
size are properties of the Wormhole Galaxy decode topology, not of any model.
A model creates exactly one prefetcher for its mesh, registers its prefetched
decode weights in issue order, seals registration, and hands the resolved
contexts to its module configs.
"""

from __future__ import annotations

import functools
from typing import Any

import ttnn

from tt_transformers.models.galaxy.recipes import prefetch_sender_cores
from tt_transformers.models.galaxy.topology import WORMHOLE_GALAXY_TOPOLOGY, GalaxyChipTopology
from tt_transformers.models.galaxy.resources import GalaxyModePlan, GalaxyResourcesConfig
from tt_transformers.modules.prefetcher import (
    GlobalCBPlacement,
    Prefetcher2D,
    Prefetcher2DConfig,
    Prefetcher2DModeConfig,
)

GALAXY_GLOBAL_CB_SIZE = 728 * 1088

#: L1 per bank held above the global circular buffer while it is created.
#:
#: L1 is allocated top-down and a buffer's address never moves, so anything
#: long-lived allocated while the ~774 kB global CB is resident is stranded below
#: it for the life of the process - and the prefill embedding's static circular
#: buffers reach 630080, so one stranded byte down there makes every subsequent
#: prefill unplaceable. Measured on `(8, 4)` with Llama-3.3-70B: the first decode
#: strands 32 B at 545760 (`logs/a3_clash_steps_l1.log`), and with this headroom
#: reserved the only L1 left below 630080 is the global CB itself
#: (`logs/b1_headroom_only_l1.log`).
#:
#: 64 kiB is chosen against a measured need of 32 B plus the 2 kB decode rotary
#: transformation matrix, and it costs nothing but where the global CB sits: the
#: buffer simply lands 64 kiB lower in a bank with 1 265 696 B free at that
#: moment.
GALAXY_GLOBAL_CB_HEADROOM = 64 * 1024

#: One global-CB placement record per mesh device, for the life of the process.
#:
#: The ttnn program cache belongs to the mesh device and outlives any one model,
#: and two structurally identical Galaxy models hash to the same program-cache
#: keys - so the second model in a process reuses decode programs compiled for the
#: first, and those programs carry the first model's global circular buffer
#: addresses. The buffer therefore has to land on the same L1 blocks for *every*
#: model in the process, not merely for every recreation within one model. Measured
#: on `(8, 4)`: with the record held per owner, the second Qwen model's first
#: decode hung in `FDMeshCommandQueue::wait_for_outstanding_reads` with both
#: models' weights already resident
#: (`tttv2_milestone_c_runs/c-defects3/logs/m1b_hang_bt.txt`).
#:
#: Keyed by `id(mesh_device)` because a `ttnn.MeshDevice` is a nanobind object and
#: takes neither weak references nor attributes. A process opens one mesh, and
#: `forget_galaxy_global_cb_placement` exists for a test that wants a clean one.
_GLOBAL_CB_PLACEMENTS: dict[int, GlobalCBPlacement] = {}


def galaxy_global_cb_placement(mesh_device: Any) -> GlobalCBPlacement:
    """Return the process-wide global-CB placement record for one mesh device."""

    return _GLOBAL_CB_PLACEMENTS.setdefault(id(mesh_device), GlobalCBPlacement())


def forget_galaxy_global_cb_placement(mesh_device: Any) -> None:
    """Drop a mesh device's placement record, so the next creation sets it afresh.

    Only correct when nothing in the ttnn program cache still refers to the old
    buffer - in practice, a test that clears the cache or opens a new mesh.
    """

    _GLOBAL_CB_PLACEMENTS.pop(id(mesh_device), None)


def release_galaxy_global_cb_placement(mesh_device: Any) -> None:
    """Retire the mesh's cached programs, then forget its placement record.

    Called when a Galaxy owner's global circular buffer has been released and its
    L1 is back - `Prefetcher2DConfig.on_global_cb_released`.

    Two facts make this the model layer's job rather than the module's, and both
    are about the *mesh* rather than about any one owner:

    * the ttnn program cache belongs to the mesh device and outlives a model's
      `close()`. Its keys are op-and-config hashes, so two structurally identical
      Galaxy models hash to the same keys and the second model's decode is a cache
      **hit** on programs compiled for the first. Those programs carry the first
      buffer's captured `buffer_address()` and `config_address()`
      (`circular_buffer.cpp:179`, re-sent by `dispatch.cpp:3035` on every launch),
      so once that buffer is gone every one of them is a wrong-address read
      waiting to happen. Clearing the cache is what makes them stop being that;
    * a cached program also holds its **semaphores**, and a semaphore is a
      32-byte L1 allocation. Measured on `(8, 4)`, one-layer subsets
      (`tttv2_milestone_c_runs/c-defects4/logs/s3_qwen_two_pools_sub_table.log`):
      a second model in one process found **77 extra 32-byte blocks** resident -
      2 464 B - while every block larger than 32 bytes matched the first model's
      in size and count. Those 77 holes are not a capacity problem; they are a
      *fragmentation* problem. `FreeListOpt::allocate` takes the smallest free
      block that fits, so they are taken in preference to the low region, and the
      64 kB block that sat at 1381856 for the first model was displaced to
      1272480 for the second - 109 376 B lower, below the free top the first
      creation recorded, which is precisely what made the buffer unplaceable.

    Clearing the cache addresses both at once, and it is safe in the order used
    here because the owner announces the release only after the buffer's last
    reference is gone.
    """

    try:
        mesh_device.clear_program_cache()
    finally:
        forget_galaxy_global_cb_placement(mesh_device)


def _ranges(coordinates: tuple[tuple[int, int, int, int], ...]) -> ttnn.CoreRangeSet:
    return ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(x0, y0), ttnn.CoreCoord(x1, y1)) for x0, y0, x1, y1 in coordinates]
    )


def galaxy_sender_receiver_mapping(topology: GalaxyChipTopology | None = None) -> tuple[tuple[Any, Any], ...]:
    """Return the canonical `(sender core, receiver core set)` prefetch mapping.

    The trailing entries carry dummy senders that read nothing. They are not
    padding: the leading entries are the active DRAM readers, and the rest exist
    so the global circular buffer's `all_cores()` covers the complete worker set.
    A minimal mapping of only the active senders passes every module test and
    then fails a hard superset check -- *"Specified cores are not contained in
    associated GlobalCircularBuffer"* -- the first time an unrelated fused matmul
    reaches a hop core. On Wormhole that core is `(3, 6)`, present only in dummy
    entry 16. `GalaxyChipTopology` asserts that coverage on the host.
    """

    resolved = topology if topology is not None else WORMHOLE_GALAXY_TOPOLOGY
    senders = prefetch_sender_cores(resolved) + tuple(ttnn.CoreCoord(x, y) for x, y in resolved.dummy_sender_coords)
    receivers = tuple(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(*start), ttnn.CoreCoord(*end))})
        for start, end in resolved.receiver_column_pairs
    ) + tuple(_ranges(coordinates) for coordinates in resolved.dummy_receiver_ranges)
    if len(senders) != len(receivers):
        raise ValueError("Galaxy prefetch sender and receiver counts must match")
    return tuple(zip(senders, receivers))


def galaxy_address_memory_config(weight_count: int, topology: GalaxyChipTopology | None = None) -> ttnn.MemoryConfig:
    """Return the packed weight-address placement on the real sender cores."""

    if weight_count <= 0:
        raise ValueError("prefetched weight count must be positive")
    senders = prefetch_sender_cores(topology)
    sender_cores = ttnn.CoreRangeSet([ttnn.CoreRange(core, core) for core in senders])
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(sender_cores, [1, weight_count], ttnn.ShardOrientation.ROW_MAJOR),
    )


def galaxy_dram_prefetch_start(*, tensors_per_layer: int, num_layers: int) -> Any:
    """Return the Galaxy decode prefetch producer.

    The DRAM prefetcher streams one layer's tensor set at a time and walks the
    remaining layers through the packed address table, so it receives the first
    layer's tensors plus the address metadata rather than every weight.
    """

    if tensors_per_layer <= 0 or num_layers <= 0:
        raise ValueError("tensors_per_layer and num_layers must be positive")

    def start(context: Any) -> Any:
        weights = list(context.weights[:tensors_per_layer])
        if len(weights) != tensors_per_layer:
            raise RuntimeError(
                f"decode prefetch requires {tensors_per_layer} registered weights per layer, got {len(weights)}"
            )
        return ttnn.dram_prefetcher(
            weights + [context.weight_address_metadata],
            num_layers=num_layers,
            global_cb=context.global_cb,
        )

    return start


def _mode_config(plan: GalaxyModePlan) -> Prefetcher2DModeConfig:
    return Prefetcher2DModeConfig(
        mode=plan.mode,
        sub_devices=plan.sub_devices,
        worker_sub_device_id=plan.worker_sub_device_id,
        stall_group=plan.stall_group,
        local_l1_size=plan.local_l1_size,
    )


def build_galaxy_prefetcher_config(
    mesh_device: Any,
    resources_config: GalaxyResourcesConfig,
    *,
    expected_weight_count: int,
    global_cb_size: int | None = GALAXY_GLOBAL_CB_SIZE,
    prefetch_num_layers: int = 1,
    defer_global_cb: bool = True,
    release_global_cb_on_prefill: bool = True,
    global_cb_headroom: int = GALAXY_GLOBAL_CB_HEADROOM,
) -> Prefetcher2DConfig:
    """Resolve the prefetcher policy that matches a Galaxy resource config.

    ``defer_global_cb`` defaults to ``True`` here, and only here: on this mesh
    the global CB's ~774 kB of L1 per sender/receiver core makes every prefill
    program that needs static circular buffers on those cores unplaceable, and
    the Galaxy models all run prefill before decode. See the field's own
    docstring in ``Prefetcher2DConfig``; the production Galaxy prefetcher makes
    the same choice for the same reason.
    """

    return Prefetcher2DConfig(
        mesh_device=mesh_device,
        architecture=resources_config.architecture,
        prefill=_mode_config(resources_config.prefill),
        decode=_mode_config(resources_config.decode),
        sender_receiver_mapping=galaxy_sender_receiver_mapping(),
        global_cb_size=global_cb_size,
        expected_weight_count=expected_weight_count,
        address_repeat_count=len(prefetch_sender_cores()),
        address_memory_config=galaxy_address_memory_config(expected_weight_count),
        address_mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        prefetch_num_layers=prefetch_num_layers,
        mesh_shape=resources_config.mesh_shape,
        defer_global_cb=defer_global_cb,
        # Defaults to **True** since 2026-08-30, and the reason is arithmetic
        # rather than preference. 869 056 B of L1 lie above the 630080 that the
        # prefill embedding's static circular buffers reach; the model's resident
        # L1 after a decode is 127 776 B and the global circular buffer is
        # 792 256 B including its config buffer. 127 776 + 792 256 = 920 032, so
        # the buffer and a prefill program of this shape cannot coexist under any
        # allocation order. `defer_global_cb` covers only the first prefill;
        # without this flag every prefill *after* a decode aborts at
        # `program.cpp:1763`, which is what blocked Llama's repeated-request gate
        # and, by construction, blocks serving. Paired with `global_cb_headroom`,
        # which is what stops the release from being defeated by a buffer
        # stranded underneath the CB.
        release_global_cb_on_prefill=release_global_cb_on_prefill,
        global_cb_headroom=global_cb_headroom,
        # When this owner's buffer is gone, the mesh's cached programs still hold
        # its addresses and its semaphores. Retiring them is a mesh-level action
        # and so it is resolved here, not inside the module.
        on_global_cb_released=functools.partial(release_galaxy_global_cb_placement, mesh_device),
    )


def build_galaxy_prefetcher(
    mesh_device: Any,
    resources_config: GalaxyResourcesConfig,
    *,
    expected_weight_count: int,
    global_cb_size: int | None = GALAXY_GLOBAL_CB_SIZE,
    prefetch_num_layers: int = 1,
    release_global_cb_on_prefill: bool = True,
    global_cb_headroom: int = GALAXY_GLOBAL_CB_HEADROOM,
    **injections: Any,
) -> Prefetcher2D:
    """Create an initialized, unsealed `Prefetcher2D` for one Galaxy mesh."""

    # Shared across every model this process builds on this mesh: the ttnn program
    # cache is the mesh device's and it outlives any one model. A caller that
    # injects its own record keeps it.
    injections.setdefault("global_cb_placement", galaxy_global_cb_placement(mesh_device))

    prefetcher = Prefetcher2D(
        build_galaxy_prefetcher_config(
            mesh_device,
            resources_config,
            expected_weight_count=expected_weight_count,
            global_cb_size=global_cb_size,
            prefetch_num_layers=prefetch_num_layers,
            release_global_cb_on_prefill=release_global_cb_on_prefill,
            global_cb_headroom=global_cb_headroom,
        ),
        **injections,
    )
    try:
        prefetcher.initialize()
    except BaseException:
        prefetcher.cleanup()
        raise
    return prefetcher
