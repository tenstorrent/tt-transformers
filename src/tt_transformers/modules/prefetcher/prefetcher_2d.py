# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Lifecycle owner for Wormhole Galaxy 2D weight prefetch resources."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal, Protocol, runtime_checkable

import torch
import ttnn
from loguru import logger

PrefetcherMode = Literal["prefill", "decode"]
WeightCompatibilityValidator = Callable[[str, Any, tuple[Any, ...]], None]
GlobalCBSizeDeriver = Callable[[tuple[Any, ...]], int]
DramPrefetchStart = Callable[["Prefetcher2DContext"], Any]
DramPrefetchStop = Callable[[Any, Any], Any]


def _validate_wh_galaxy(mesh_device: Any, mesh_shape: tuple[int, int], architecture: Any) -> None:
    if architecture != ttnn.device.Arch.WORMHOLE_B0:
        raise ValueError(f"Prefetcher2D requires Wormhole B0, got {architecture}")
    if mesh_shape != (8, 4):
        raise ValueError(f"Prefetcher2D requires logical mesh shape (8, 4), got {mesh_shape}")
    if tuple(mesh_device.shape) != mesh_shape:
        raise ValueError(f"mesh device shape {tuple(mesh_device.shape)} does not match resolved shape {mesh_shape}")
    if mesh_device.get_num_devices() != 32:
        raise ValueError(f"Prefetcher2D requires exactly 32 devices, got {mesh_device.get_num_devices()}")
    if mesh_device.arch() != architecture:
        raise ValueError("mesh device architecture does not match the resolved architecture")


#: Free gaps above the low region are finite, so the reservation loop terminates;
#: this only bounds a pathological free list rather than expressing a policy.
_MAX_GLOBAL_CB_RESERVATION_STEPS = 64


def _default_l1_block_table(mesh_device: Any) -> tuple[tuple[int, int, bool], ...]:
    """Return every L1 block as ``(address, size, allocated)``, allocator coordinates.

    `ttnn.get_memory_view` reports each block's address **without** the allocator's
    ``offset_bytes_``, while every allocator message adds it, so this adds
    `ttnn.get_allocator_base_address` and returns the numbers the allocator itself
    would print. Mixing the two coordinate systems is what made three earlier
    investigations of the Llama L1 address clash unreadable.

    The **whole** table, free blocks included, and for two reasons that the lowest
    occupied address alone cannot serve:

    * a global circular buffer is **two** allocations, a `size`-byte data buffer
      and a 192-byte config page, and a cached program holds the address of both
      (`CircularBufferImpl::set_global_circular_buffer` captures
      ``buffer_address()`` *and* ``config_address()`` once, and
      `dispatch.cpp` re-sends the captured pair on every launch). So both have to
      come back to the same address, and only the whole table can say whether they
      did;
    * `FreeListOpt::allocate` takes the **smallest free block that fits**, so a
      192-byte free gap anywhere in L1 will capture the config page in preference
      to the region below the data buffer. Measured on `(8, 4)`, two fresh
      processes:
      the data buffer came back at 510816 both times and the config page moved
      from 510624 to **1367872**.
    """

    view = ttnn.get_memory_view(mesh_device, ttnn.BufferType.L1)
    offset = ttnn.get_allocator_base_address(mesh_device, ttnn.BufferType.L1)
    blocks = tuple(
        (int(block["address"]) + offset, int(block["size"]), block.get("allocated") == "yes")
        for block in view.block_table
    )
    if not any(allocated for _, _, allocated in blocks):
        raise RuntimeError("no allocated L1 blocks: the mesh allocator view is empty")
    return blocks


def _allocated(table: tuple[tuple[int, int, bool], ...]) -> frozenset[tuple[int, int]]:
    """Return the allocated ``(address, size)`` blocks of an L1 block table."""

    return frozenset((address, size) for address, size, allocated in table if allocated)


def _lowest_occupied(table: tuple[tuple[int, int, bool], ...]) -> int:
    """Return the lowest allocated address in an L1 block table."""

    return min(address for address, _, allocated in table if allocated)


def _free_gaps_above_the_low_region(table: tuple[tuple[int, int, bool], ...]) -> tuple[tuple[int, int], ...]:
    """Return the free blocks that are not the contiguous low region, smallest first.

    Every free block other than the lowest-addressed one is a *gap*: a hole in the
    resident region, left by something freed. `FreeListOpt::allocate` prefers the
    smallest block that fits, so any such gap will capture an allocation in
    preference to the low region - which is exactly how the global circular
    buffer's 192-byte config page ends up 850 kB away from its data buffer.
    Filling them first is what makes the creation's placement deterministic.

    Sorted ascending by size, because a reservation of exactly a gap's size is how
    that gap is taken: the allocator's own preference is the mechanism.
    """

    free = [(address, size) for address, size, allocated in table if not allocated]
    if not free:
        return ()
    low_region = min(free)
    return tuple(sorted((block for block in free if block != low_region), key=lambda block: block[1]))


def _default_reserve_l1(mesh_device: Any, size: int) -> Any:
    """Hold exactly `size` bytes per L1 bank, as one sharded page on one worker core.

    One buffer and no companion allocation, so the address the global circular
    buffer lands at afterwards is an exact function of `size`. L1 allocation is
    lock-step across banks on this part, so a single-core shard reserves the same
    address range on every core of the mesh.
    """

    if size % 32:
        raise ValueError(f"an L1 reservation must be a multiple of 32 bytes, got {size}")
    elements = size // 2
    core = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(1, 0))})
    memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(core, [1, elements], ttnn.ShardOrientation.ROW_MAJOR),
    )
    return ttnn.allocate_tensor_on_device(
        ttnn.Shape((1, elements)), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, mesh_device, memory_config
    )


@dataclass
class GlobalCBPlacement:
    """Where a mesh's global circular buffer has to be created, once one has been.

    Mutable and shareable **on purpose**, and the scope is the reason. The ttnn
    program cache belongs to the *mesh device*, not to a `Prefetcher2D`, and it
    outlives any one model: two structurally identical models in one process hash
    to the same program-cache keys, so the second model's decode reuses programs
    compiled for the first - and those programs carry the first model's global
    circular buffer addresses, captured once by
    `CircularBufferImpl::set_global_circular_buffer` and re-sent on every launch.
    So "the buffer must come back to the same L1 blocks" is a per-process,
    per-mesh invariant, and a record held privately by one owner cannot express
    it.

    An owner given no record keeps its own, which is exactly the previous
    per-owner behaviour. `src/tt_transformers/models/galaxy/prefetch.py` holds one per
    mesh device for the Galaxy models, because "one process, one mesh, several
    models" is a model-level fact.
    """

    #: the lowest occupied L1 address the first creation reserved down to
    free_top: int | None = None
    #: the ``(address, size)`` blocks that creation added
    blocks: frozenset[tuple[int, int]] | None = None


@dataclass(frozen=True)
class Prefetcher2DModeConfig:
    mode: PrefetcherMode
    sub_devices: tuple[Any, ...]
    worker_sub_device_id: Any
    stall_group: tuple[Any, ...]
    local_l1_size: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "sub_devices", tuple(self.sub_devices))
        object.__setattr__(self, "stall_group", tuple(self.stall_group))
        if self.mode not in ("prefill", "decode"):
            raise ValueError(f"unsupported prefetcher mode: {self.mode}")
        if not self.sub_devices:
            raise ValueError("at least one subdevice must be configured")
        if self.worker_sub_device_id is None:
            raise ValueError("worker_sub_device_id must be resolved")
        if not self.stall_group:
            raise ValueError("stall_group must be resolved")
        if self.local_l1_size < 0:
            raise ValueError("local_l1_size cannot be negative")


@dataclass(frozen=True)
class Prefetcher2DConfig:
    """Frozen, fully resolved prefetcher construction policy."""

    mesh_device: Any
    architecture: Any
    prefill: Prefetcher2DModeConfig
    decode: Prefetcher2DModeConfig
    sender_receiver_mapping: tuple[tuple[Any, Any], ...]
    global_cb_size: int | None
    expected_weight_count: int
    address_repeat_count: int
    address_memory_config: Any
    address_mesh_mapper: Any
    prefetch_num_layers: int = 1
    mesh_shape: tuple[int, int] = (8, 4)
    #: Allocate the global circular buffer on the first ``activate("decode")``
    #: instead of in ``seal()``.
    #:
    #: The global CB is ~774 kB of L1 on every sender/receiver core and nothing
    #: can free it, so with the default ``False`` a *prefill* program that needs
    #: static circular buffers on those cores cannot be placed at all:
    #:
    #:     TT_THROW ... Statically allocated circular buffers in program 100
    #:     clash with L1 buffers on core range [0-0 - 0-3]. L1 buffer allocated
    #:     at 579104 and static circular buffer region ends at 630080
    #:                                              (from ttnn.embedding, prefill)
    #:
    #: Prefill never reads the buffer - ``seal()`` already hands the prefill
    #: context ``global_cb=None`` - so holding it through prefill buys nothing.
    #: The production Galaxy prefetcher makes the same choice and says so:
    #: ``self.global_circular_buffer = None  # Global CB will only be allocated
    #: before decode runs`` in ``models/demos/llama3_70b_galaxy/tt/
    #: prefetcher_common.py``, with allocation in its own ``create_global_cb()``.
    #:
    #: Defaults to ``False`` so the existing qualification of this module is
    #: bit-for-bit unchanged unless a caller asks for the deferral.
    defer_global_cb: bool = False
    #: Release the global circular buffer again on ``activate("prefill")``.
    #:
    #: ``defer_global_cb`` only helps the *first* prefill. Once decode has been
    #: activated the buffer is resident again, so a **prefill after a decode** is
    #: back to being unplaceable — which is what a second `GalaxyDirectRunner` in
    #: one process does, and it aborts with:
    #:
    #:     TT_THROW ... Statically allocated circular buffers in program 100
    #:     clash with L1 buffers on core range [0-0 - 0-3]
    #:
    #: This releases it on the way into prefill and lets ``_ensure_global_cb``
    #: recreate it on the way back into decode. There is no `deallocate` on a
    #: `global_circular_buffer`, so releasing means dropping **every** reference
    #: and letting the C++ destructor free the L1 — ``cleanup()`` alone does
    #: not, because the mode contexts still hold handles.
    #:
    #: **Defaults to False, and the default is the qualified path.** Two risks the
    #: flag exists to keep away from it: the recreated buffer must land at the same
    #: L1 address, or decode programs already in the ttnn program cache hold stale
    #: addresses (a silent corruption, not an error); and it changes the
    #: mode-switching of the one module every qualified decode path depends on.
    #: Requires ``defer_global_cb``, since both describe the same lifetime.
    release_global_cb_on_prefill: bool = False
    #: Reserve this many bytes of L1 per bank **above** the global circular
    #: buffer while it is being created, then release them again.
    #:
    #: L1 is allocated top-down, and the global CB is ~774 kB per bank. So with
    #: the buffer resident, **every** long-lived allocation made afterwards lands
    #: *below* it and stays there for the life of the process, because a buffer's
    #: address never moves. Measured on `(8, 4)` with Llama-3.3-70B: the
    #: first decode strands a 32-byte L1 buffer at **545760**, which is below the
    #: 630080 that the prefill embedding's static circular buffers reach, so the
    #: next prefill aborts with
    #:
    #:     TT_THROW ... Statically allocated circular buffers in program 100
    #:     clash with L1 buffers on core range [0-0 - 0-3]. L1 buffer allocated
    #:     at 545760 and static circular buffer region ends at 630080
    #:
    #: and it aborts *even with* ``release_global_cb_on_prefill``, because
    #: releasing the buffer does not move what was stranded underneath it. Read
    #: the message carefully: ``validate_circular_buffer_region`` computes one
    #: device-wide lowest-occupied L1 address before it loops over circular-buffer
    #: core ranges, so ``[0-0 - 0-3]`` names the *circular buffers*, not the
    #: clashing buffer.
    #:
    #: Reserving headroom first makes the global CB land that much lower and
    #: leaves a free gap above it. ``FreeListOpt::allocate`` scans free blocks by
    #: **ascending size class**, so a later small allocation takes the small gap
    #: in preference to the large low block, and nothing is stranded below the
    #: buffer. Measured with 65 536 B of headroom on the same reproduction
    #: the only L1 below 630080 afterwards is the global CB itself, and the
    #: 32-byte block at 545760 is gone.
    #:
    #: This value is the headroom for the **first** creation only. Every later
    #: creation reserves whatever it takes to put the buffer back on the same
    #: floor - see `Prefetcher2D._allocate_global_cb`, and
    #: ``release_global_cb_on_prefill`` for why that matters.
    #:
    #: Defaults to 0, which is exactly the previous behaviour.
    global_cb_headroom: int = 0

    #: Called once from `cleanup`, after the global circular buffer's last
    #: reference is dropped and its L1 is back.
    #:
    #: The buffer's addresses are captured **once** per program
    #: (`CircularBufferImpl::set_global_circular_buffer` takes
    #: ``buffer_address()`` and ``config_address()``; `dispatch.cpp` re-sends the
    #: captured pair on every launch), so a program that outlives the buffer holds
    #: a dead address. Whether anything *can* outlive it is not a fact this module
    #: knows: the ttnn program cache belongs to the mesh device, and how many
    #: models share one mesh is a model-level question. So the owner announces
    #: that the buffer is gone and the caller decides what that means.
    #:
    #: Defaults to `None`, which is exactly the previous behaviour.
    on_global_cb_released: Callable[[], None] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "sender_receiver_mapping",
            tuple(tuple(pair) for pair in self.sender_receiver_mapping),
        )
        _validate_wh_galaxy(self.mesh_device, self.mesh_shape, self.architecture)
        if self.prefill.mode != "prefill" or self.decode.mode != "decode":
            raise ValueError("prefill and decode subdevice configs must match their declared modes")
        if not self.sender_receiver_mapping or any(len(pair) != 2 for pair in self.sender_receiver_mapping):
            raise ValueError("sender_receiver_mapping must contain resolved sender/receiver pairs")
        if self.address_repeat_count > len(self.sender_receiver_mapping):
            raise ValueError("address_repeat_count cannot exceed the sender/receiver mapping count")
        if self.global_cb_size is not None and self.global_cb_size <= 0:
            raise ValueError("global_cb_size must be positive when specified")
        if self.expected_weight_count <= 0:
            raise ValueError("expected_weight_count must be positive")
        if self.address_repeat_count <= 0:
            raise ValueError("address_repeat_count must be positive")
        if self.address_memory_config is None or self.address_mesh_mapper is None:
            raise ValueError("address tensor placement must be fully resolved")
        if self.prefetch_num_layers <= 0:
            raise ValueError("prefetch_num_layers must be positive")
        if self.release_global_cb_on_prefill and not self.defer_global_cb:
            raise ValueError("release_global_cb_on_prefill requires defer_global_cb")
        if self.global_cb_headroom < 0:
            raise ValueError("global_cb_headroom cannot be negative")
        if self.expected_weight_count % self.prefetch_num_layers != 0:
            raise ValueError("expected_weight_count must be divisible by prefetch_num_layers")


@dataclass(frozen=True)
class Prefetcher2DContext:
    """Immutable resources borrowed by modules and model-owned executors."""

    mode: PrefetcherMode
    mesh_device: Any
    sub_device_manager_id: Any
    worker_sub_device_id: Any
    stall_group: tuple[Any, ...]
    global_cb: Any
    weights: tuple[Any, ...]
    weight_addresses: Any
    weight_address_metadata: Any

    @property
    def sub_device_id(self) -> Any:
        return self.worker_sub_device_id


@runtime_checkable
class Prefetcher2DResourceOwner(Protocol):
    """Structural owner API consumed by model-level resource collaborators."""

    @property
    def mesh_device(self) -> Any: ...

    def borrow_context(
        self,
        mode: PrefetcherMode,
        *,
        sub_devices: tuple[Any, ...],
        worker_sub_device_id: Any,
        stall_group: tuple[Any, ...],
        local_l1_size: int,
    ) -> Prefetcher2DContext: ...

    def activate(self, mode: PrefetcherMode) -> Prefetcher2DContext: ...


class Prefetcher2D:
    """Own managers, packed addresses, global CB, and running prefetch results."""

    def __init__(
        self,
        config: Prefetcher2DConfig,
        *,
        create_global_cb: Callable[[Any, list[tuple[Any, Any]], int], Any] | None = None,
        create_address_metadata: Callable[..., Any] | None = None,
        deallocate: Callable[[Any], None] | None = None,
        validate_weight_compatibility: WeightCompatibilityValidator | None = None,
        derive_global_cb_size: GlobalCBSizeDeriver | None = None,
        dram_prefetch_start: DramPrefetchStart | None = None,
        dram_prefetch_stop: DramPrefetchStop | None = None,
        l1_block_table: Callable[[Any], tuple[tuple[int, int, bool], ...]] | None = None,
        reserve_l1: Callable[[Any, int], Any] | None = None,
        global_cb_placement: GlobalCBPlacement | None = None,
    ):
        self.config = config
        self._create_global_cb = create_global_cb or ttnn.create_global_circular_buffer
        self._create_address_metadata = create_address_metadata or ttnn.as_tensor
        self._deallocate = deallocate or ttnn.deallocate
        self._validate_weight_compatibility = validate_weight_compatibility or self._default_validate_weight
        self._derive_global_cb_size = derive_global_cb_size or self._default_derive_global_cb_size
        self._dram_prefetch_start = dram_prefetch_start or self._default_dram_prefetch_start
        self._dram_prefetch_stop = dram_prefetch_stop or self._default_dram_prefetch_stop
        self._l1_block_table = l1_block_table or _default_l1_block_table
        self._reserve_l1 = reserve_l1 or _default_reserve_l1
        #: The owner's answer to "is a trace capture in progress?". See
        #: `set_capture_probe`. `None` is the unconditional behaviour every
        #: caller had before tracing existed.
        self._capture_probe: Any = None
        #: Where the persistent sender is launched: at the operation boundary,
        #: which is what every caller had before tracing, or inside the decode
        #: graph body. See `set_sender_launch_site`.
        self._sender_launch_site: str = "boundary"
        #: The free-region top the first global-CB creation reserved down to, and
        #: the L1 blocks that creation added. Remembered so every later creation
        #: reproduces both exactly: a decode program already in the ttnn program
        #: cache holds the buffer's addresses, and a buffer recreated elsewhere is
        #: read at the wrong place. The *blocks* rather than the lowest occupied
        #: address, because a global circular buffer is two allocations and a
        #: cached program holds both - see `_default_l1_block_table`. Shared when
        #: the caller shares it, because the program cache is per mesh device and
        #: outlives any one model - see `GlobalCBPlacement`.
        self._global_cb_placement = global_cb_placement or GlobalCBPlacement()
        self._managers: dict[PrefetcherMode, Any] = {}
        self._registered_weights: OrderedDict[str, Any] = OrderedDict()
        self._global_cb: Any = None
        self._resolved_global_cb_size: int | None = None
        self._weight_address_metadata: Any = None
        self._contexts: dict[PrefetcherMode, Prefetcher2DContext] = {}
        self._active_mode: PrefetcherMode | None = None
        self._loaded_mode: PrefetcherMode | None = None
        self._prefetch_result: Any = None
        self._retained_prefetch_resources: list[Any] = []
        self._initialized = False
        self._sealed = False
        self._cleaned = False

    @property
    def initialized(self) -> bool:
        return self._initialized

    @property
    def sealed(self) -> bool:
        return self._sealed

    @property
    def active_mode(self) -> PrefetcherMode | None:
        return self._active_mode

    @property
    def mesh_device(self) -> Any:
        return self.config.mesh_device

    @property
    def prefetch_result(self) -> Any:
        return self._prefetch_result

    @property
    def resolved_global_cb_size(self) -> int | None:
        return self._resolved_global_cb_size

    @property
    def borrowed_weights(self) -> tuple[Any, ...]:
        return tuple(self._registered_weights.values())

    @property
    def owned_resources(self) -> tuple[Any, ...]:
        resources = list(self._managers.values())
        resources.extend(
            resource for resource in (self._global_cb, self._weight_address_metadata) if resource is not None
        )
        resources.extend(self._retained_prefetch_resources)
        if self._prefetch_result is not None:
            resources.append(self._prefetch_result)
        return tuple(resources)

    def initialize(self) -> None:
        self._ensure_open()
        if self._initialized:
            return

        mesh = self.config.mesh_device
        created: list[Any] = []
        try:
            for mode_config in (self.config.prefill, self.config.decode):
                manager = mesh.create_sub_device_manager(list(mode_config.sub_devices), mode_config.local_l1_size)
                self._managers[mode_config.mode] = manager
                created.append(manager)
        except Exception:
            for manager in reversed(created):
                mesh.remove_sub_device_manager(manager)
            self._managers.clear()
            raise

        self._initialized = True

    def register_weight(self, name: str, tensor: Any) -> None:
        self._ensure_open()
        if not self._initialized:
            raise RuntimeError("Prefetcher2D must be initialized before weight registration")
        if self._sealed:
            raise RuntimeError("weight registration is sealed")
        if not name:
            raise ValueError("registered weight name cannot be empty")
        if name in self._registered_weights:
            raise ValueError(f"weight is already registered: {name}")
        if len(self._registered_weights) >= self.config.expected_weight_count:
            raise ValueError("registered weight count exceeds the resolved configuration")
        if not callable(getattr(tensor, "buffer_address", None)):
            raise TypeError("registered weights must be materialized device tensors")

        tensor_device = getattr(tensor, "device", None)
        tensor_device = tensor_device() if callable(tensor_device) else tensor_device
        if tensor_device is not None and tensor_device != self.config.mesh_device:
            raise ValueError("registered weight belongs to a different mesh")
        self._validate_weight_compatibility(name, tensor, tuple(self._registered_weights.values()))
        self._registered_weights[name] = tensor

    def seal(self) -> tuple[Prefetcher2DContext, Prefetcher2DContext]:
        self._ensure_open()
        if not self._initialized:
            raise RuntimeError("Prefetcher2D must be initialized before sealing")
        if self._sealed:
            return self.context("prefill"), self.context("decode")
        if len(self._registered_weights) != self.config.expected_weight_count:
            raise RuntimeError(
                f"expected {self.config.expected_weight_count} registered weights, got {len(self._registered_weights)}"
            )

        weights = tuple(self._registered_weights.values())
        configured_cb_size = self.config.global_cb_size
        resolved_cb_size = configured_cb_size or self._derive_global_cb_size(weights)
        if resolved_cb_size <= 0:
            raise ValueError("resolved global CB size must be positive")
        weight_addresses = MappingProxyType(
            {name: tensor.buffer_address() for name, tensor in self._registered_weights.items()}
        )
        addresses = torch.tensor(tuple(weight_addresses.values()), dtype=torch.int64)
        addresses = addresses.repeat(self.config.address_repeat_count, 1)

        global_cb = None
        metadata = None
        try:
            self._configure_mode_resources(self.config.decode)
            if not self.config.defer_global_cb:
                global_cb = self._allocate_global_cb(resolved_cb_size)
            metadata = self._create_address_metadata(
                addresses,
                device=self.config.mesh_device,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=self.config.address_memory_config,
                mesh_mapper=self.config.address_mesh_mapper,
            )
        except Exception:
            rollback_error = None
            for resource in (metadata,):
                if resource is not None:
                    try:
                        self._deallocate(resource)
                    except Exception as exc:
                        if rollback_error is None:
                            rollback_error = exc
            if rollback_error is not None:
                raise RuntimeError("Prefetcher2D sealing and rollback both failed") from rollback_error
            raise

        self._global_cb = global_cb
        self._weight_address_metadata = metadata
        self._resolved_global_cb_size = resolved_cb_size
        self._contexts = {
            "prefill": self._make_context(
                self.config.prefill,
                global_cb=None,
                weights=(),
                weight_addresses=weight_addresses,
            ),
            "decode": self._make_context(
                self.config.decode,
                global_cb=global_cb,
                weights=weights,
                weight_addresses=weight_addresses,
            ),
        }
        self._sealed = True
        return self._contexts["prefill"], self._contexts["decode"]

    def context(self, mode: PrefetcherMode) -> Prefetcher2DContext:
        self._ensure_open()
        if not self._sealed:
            raise RuntimeError("Prefetcher2D contexts are unavailable until registration is sealed")
        try:
            return self._contexts[mode]
        except KeyError as exc:
            raise ValueError(f"unsupported prefetcher mode: {mode}") from exc

    def borrow_context(
        self,
        mode: PrefetcherMode,
        *,
        sub_devices: tuple[Any, ...],
        worker_sub_device_id: Any,
        stall_group: tuple[Any, ...],
        local_l1_size: int,
    ) -> Prefetcher2DContext:
        """Return a sealed context after exact subdevice-policy validation."""

        context = self.context(mode)
        mode_config = self.config.prefill if mode == "prefill" else self.config.decode
        expected = (
            mode_config.sub_devices,
            mode_config.worker_sub_device_id,
            mode_config.stall_group,
            mode_config.local_l1_size,
        )
        requested = (tuple(sub_devices), worker_sub_device_id, tuple(stall_group), local_l1_size)
        if requested != expected:
            raise ValueError(f"{mode} Galaxy resources do not match the Prefetcher2D subdevice policy")
        return context

    def activate(self, mode: PrefetcherMode) -> Prefetcher2DContext:
        self._ensure_open()
        context = self.context(mode)
        previous_mode = self._active_mode
        previous_was_prefetching = self._prefetch_result is not None
        if previous_was_prefetching:
            self._stop_prefetch()

        try:
            if mode == "prefill":
                self._release_global_cb()
            self._configure_mode(context)
            if mode == "decode":
                self._ensure_global_cb(context)
                if self._sender_launch_site == "graph":
                    # The owner launches the sender from inside the decode graph
                    # body, so the boundary must not launch it as well; it still
                    # leaves the stall group exactly where `_start_prefetch`
                    # would. See `set_sender_launch_site`.
                    self.config.mesh_device.set_sub_device_stall_group(list(context.stall_group))
                elif self._capturing():
                    # A trace capture records the decode graph instead of
                    # running it, so the consumer of this persistent sender
                    # never executes: `ttnn.dram_prefetcher` would be dispatched
                    # for real, block on global-CB credits that never arrive, and
                    # strand itself on the prefetch sender cores for the rest of
                    # the process. That is fatal one capture region later,
                    # because the prefill subdevice is the *whole* compute grid
                    # and includes those cores - the first blocking `finish` in
                    # prefill mode then waits on a worker-completion count that
                    # can never be reached, which is the hang observed inside
                    # `ttnn.end_trace_capture`.
                    #
                    # So the launch is skipped, and only the launch: the global
                    # CB is created above (the captured programs read its
                    # addresses) and the stall group is left exactly where
                    # `_start_prefetch` would leave it, so the captured graph and
                    # the device state around it are otherwise identical. Replay
                    # is unaffected - every `decode_forward` activates this
                    # context first, so the sender is launched immediately before
                    # the replay that consumes it, which is what the eager path
                    # has always done.
                    self.config.mesh_device.set_sub_device_stall_group(list(context.stall_group))
                else:
                    self._start_prefetch(context)
        except Exception as activation_error:
            if self._prefetch_result is not None:
                self._stop_prefetch(suppress_errors=True)
            try:
                if previous_mode is None:
                    self.config.mesh_device.reset_sub_device_stall_group()
                    self.config.mesh_device.clear_loaded_sub_device_manager()
                    self._loaded_mode = None
                else:
                    previous = self.context(previous_mode)
                    self._configure_mode(previous)
                    if previous_was_prefetching:
                        self._start_prefetch(previous)
                self._active_mode = previous_mode
            except Exception as rollback_error:
                self._active_mode = None
                raise RuntimeError("Prefetcher2D activation and rollback both failed") from rollback_error
            raise activation_error

        self._active_mode = mode
        return context

    def set_capture_probe(self, probe: Any) -> None:
        """Install the owner's answer to "is a trace capture in progress?".

        Model-owned wiring, in the same shape as `GalaxyResources`: the executor
        knows, because the common runtime's `ProgramCompiler` publishes it around
        every capture region, and this owner is what dispatches the persistent
        prefetch sender. `None` restores the unconditional launch, which is what
        every caller before tracing had.
        """

        if probe is not None and not callable(probe):
            raise TypeError("capture probe must be callable or None")
        self._capture_probe = probe

    def set_sender_launch_site(self, site: str) -> None:
        """Choose where `ttnn.dram_prefetcher` is launched for decode.

        ``"boundary"`` is the behaviour every caller had before tracing: the
        persistent sender is dispatched by `activate("decode")`, immediately
        before the graph that consumes it. That is correct for eager execution
        and it is **not** correct under tracing, because the graph the sender
        feeds is *recorded* rather than run, so capture and replay see different
        sender state and a replay can read a global-CB slot the sender has not
        filled.

        ``"graph"`` moves the launch into the decode graph body, where the
        capture records it and every replay performs it in the recorded order
        against its own consumers. It is what TTTv1 does
        (`models/demos/llama3_70b_galaxy/tt/llama_model.py:790-808`: the
        `create_global_cb()` / `ttnn.dram_prefetcher` / `set_sub_device_stall_group`
        trio at the top of `forward` in decode mode) and it is the one
        structural difference between its traced decode and this one.

        The default is unchanged, so `GalaxyDirectRunner` and every existing
        caller behaves byte for byte as before.
        """

        if site not in ("boundary", "graph"):
            raise ValueError(f"sender launch site must be 'boundary' or 'graph', got {site!r}")
        self._sender_launch_site = site

    @property
    def sender_launch_site(self) -> str:
        return self._sender_launch_site

    def launch_sender(self) -> None:
        """Dispatch the persistent decode sender from inside the graph body.

        Only valid under `set_sender_launch_site("graph")`; the boundary site
        launches it itself and a second launch would strand a sender.
        """

        self._ensure_open()
        if self._sender_launch_site != "graph":
            raise RuntimeError("launch_sender requires set_sender_launch_site('graph')")
        if self._active_mode != "decode":
            raise RuntimeError("the persistent sender is decode-only; activate decode first")
        if self._prefetch_result is not None:
            self._stop_prefetch()
        self._start_prefetch(self.context("decode"))

    def _capturing(self) -> bool:
        probe = self._capture_probe
        return bool(probe()) if probe is not None else False

    def _allocate_global_cb(self, size: int) -> Any:
        """Create the global circular buffer at a **stable** L1 address.

        Two things have to be true at once, and both are measured on `(8, 4)`:

        * nothing long-lived may be allocated *below* the buffer. L1 is allocated
          top-down and an address never moves, so anything allocated while the
          ~774 kB buffer is resident is stranded underneath it for the life of the
          process - and the Galaxy prefill embedding's static circular buffers
          reach 630080, so one stranded byte down there makes every later prefill
          unplaceable: the first decode strands 32 B at 545760, and releasing
          the buffer does not move it.
          `global_cb_headroom` fixes that by making the buffer land that much
          lower and leaving a free gap above it, which `FreeListOpt` - which scans
          free blocks by ascending size class - hands to the small allocations in
          preference to the large low block.
        * the buffer must come back to the **same address** every time it is
          recreated. `release_global_cb_on_prefill` frees it on the way into
          prefill, and decode programs already in the ttnn program cache hold its
          address; recreating it somewhere else is a silent wrong-address read,
          not an error. Measured: with a *fixed* headroom the second creation
          lands 32 416 B lower and the decode after it hangs.

        Two things make the placement reproducible, and both follow from the one
        allocator rule that governs all of this: `FreeListOpt::allocate` takes the
        **smallest free block that fits**.

        * **hold every free gap above the low region.** Any hole in the resident
          region will capture an allocation in preference to the low region, and
          the creation's 192-byte config page is small enough to fit almost any
          hole. Measured on `(8, 4)`, two fresh processes: the
          data buffer came back at 510816 both times while the config page moved
          from 510624 to **1367872**, 850 kB away. That is a stale-address hazard
          and not a cosmetic one -
          `CircularBufferImpl::set_global_circular_buffer` captures
          ``buffer_address()`` *and* ``config_address()`` once, and a cached
          program re-sends the captured pair on every launch. With the gaps held,
          the low region is the only free block and the two allocations are forced
          to be adjacent.
        * **reproduce the free-region top** the first creation saw, so the low
          region's top is where it was and the pair lands where it landed. One
          reservation was not enough before the gaps were being filled, and the
          number said why: the leftover of the previous headroom is a free gap of
          exactly the missing size, so a single reservation of that size lands in
          it and moves the top by nothing. On the production path the buffer came
          back 32 736 B high for exactly that reason, on both models. Both
          loops are bounded and raise rather than spin.
        """

        mesh_device = self.config.mesh_device
        mapping = list(self.config.sender_receiver_mapping)
        if self.config.global_cb_headroom <= 0:
            return self._create_global_cb(mesh_device, mapping, size)

        placement = self._global_cb_placement
        table = self._l1_block_table(mesh_device)
        before = _allocated(table)
        lowest = _lowest_occupied(table)
        if placement.free_top is None:
            target_top = lowest - self.config.global_cb_headroom
        else:
            target_top = placement.free_top
            if lowest < target_top:
                # Diagnostic, not a behaviour change: name the blocks that are
                # sitting below the free top the first creation saw. Whether they
                # are residue the previous owner failed to return or allocations
                # this owner made for a larger configuration is the whole question,
                # and the addresses answer it.
                intruders = sorted((address, size) for address, size in _allocated(table) if address < target_top)
                logger.info(f"[prefetcher] blocks below the first creation's free top {target_top}: {intruders}")
                raise RuntimeError(
                    "cannot restore the global circular buffer to its original L1 address: the lowest "
                    f"occupied L1 is {lowest}, already below the free top {target_top} the first "
                    "creation had"
                )

        reservations: list[Any] = []
        try:
            # Fill every free gap above the low region first. The allocator takes
            # the smallest free block that fits, so an unfilled gap captures the
            # global CB's config page and moves it away from its data buffer; with
            # the gaps held, the low region is the only free block and the
            # creation's two allocations are forced to be adjacent and reproducible.
            for _ in range(_MAX_GLOBAL_CB_RESERVATION_STEPS):
                gaps = _free_gaps_above_the_low_region(table)
                # An L1 reservation is a whole number of 32-byte units, so a gap
                # that is not a multiple of 32 is taken as far as it can be and
                # leaves a sliver behind. That is safe rather than approximate:
                # the leftover is under 32 bytes and the smallest thing the
                # creation allocates is the 192-byte config page, so no sliver can
                # capture any of it.
                takeable = [gap - gap % 32 for _, gap in gaps]
                takeable = [take for take in takeable if take >= 32]
                if not takeable:
                    break
                reservations.append(self._reserve_l1(mesh_device, min(takeable)))
                table = self._l1_block_table(mesh_device)
            else:
                raise RuntimeError(
                    f"could not fill the free L1 gaps above the low region in "
                    f"{_MAX_GLOBAL_CB_RESERVATION_STEPS} reservations"
                )
            lowest = _lowest_occupied(table)
            # Then reproduce the free-region top the first creation saw, so every
            # allocation the creation makes reproduces with it.
            for _ in range(_MAX_GLOBAL_CB_RESERVATION_STEPS):
                if lowest <= target_top:
                    break
                reservations.append(self._reserve_l1(mesh_device, lowest - target_top))
                lowest = _lowest_occupied(self._l1_block_table(mesh_device))
            else:
                raise RuntimeError(
                    f"could not lower the L1 free top to {target_top} in "
                    f"{_MAX_GLOBAL_CB_RESERVATION_STEPS} reservations; it is still {lowest}"
                )
            global_cb = self._create_global_cb(mesh_device, mapping, size)
        finally:
            # There is no `deallocate` on a global circular buffer, but every
            # reservation is an ordinary tensor and this holds its only reference.
            for reservation in reversed(reservations):
                self._deallocate(reservation)
            reservations.clear()

        # The creation's own allocations, by address and size. Comparing the
        # whole set rather than the lowest occupied address is what makes the
        # check exact: the creation makes *two* allocations and a cached program
        # holds the address of both, so "the lowest occupied address is where it
        # was" is neither necessary nor sufficient.
        added = _allocated(self._l1_block_table(mesh_device)) - before
        # Logged, not just checked. The placement is the thing every cached decode
        # program depends on, and when two models share a process it is the only
        # way to see that the second one's buffer landed where the first one's did.
        logger.info(f"[prefetcher] global circular buffer at L1 blocks {sorted(added)}")
        if placement.blocks is None:
            placement.free_top = target_top
            placement.blocks = added
        elif added != placement.blocks:
            raise RuntimeError(
                "the recreated global circular buffer did not land on its original L1 blocks: "
                f"expected {sorted(placement.blocks)}, got {sorted(added)}"
            )
        return global_cb

    def _ensure_global_cb(self, context: Prefetcher2DContext) -> None:
        """Allocate the deferred global circular buffer and bind it to `context`.

        Called from ``activate("decode")`` before the prefetch program is
        enqueued, which is the first moment anything reads the buffer.

        The binding is an ``object.__setattr__`` on a frozen dataclass, and that
        is deliberate rather than lazy typing. Module configs capture the
        *context object* at construction (`MLP2DConfig.decode_prefetch_context`,
        read as ``getattr(context, "global_cb", None)`` at call time), so
        replacing the entry in ``self._contexts`` would leave every already-built
        module holding a context whose ``global_cb`` is still ``None``. The field
        is bound exactly once, from ``None`` to the buffer, and never rebound.
        """

        if not self.config.defer_global_cb or self._global_cb is not None:
            return
        # Reached either after sealing (never created) or after
        # `_release_global_cb` freed it on the way into prefill.
        if self._resolved_global_cb_size is None:
            raise RuntimeError("global CB size was not resolved during sealing")
        global_cb = self._allocate_global_cb(self._resolved_global_cb_size)
        self._global_cb = global_cb
        object.__setattr__(context, "global_cb", global_cb)

    def _release_global_cb(self) -> None:
        """Drop every reference to the global CB so its L1 is freed.

        Only when `release_global_cb_on_prefill` is set. Called before the prefill
        sub-device manager is loaded, and after any prefetch program has been
        stopped by `activate`, so nothing is reading the buffer.

        There is no `deallocate` on a `global_circular_buffer`; the L1 is held by
        the C++ object and freed by its destructor, so **every** reference has to
        go. `gc.collect()` is not called: CPython frees the object as soon as the
        last reference is cleared, and a collect here would be a much bigger
        hammer than this needs.

        This used to clear two references - this owner's and the sealed decode
        context's - and that was the same incomplete-reference bug.
        A `Prefetcher2DContext` is captured **by value** at module construction,
        so every module built against a context holds its own reference to the
        buffer, and clearing the owner's map leaves all of them live. The
        destructor therefore never ran, the L1 was never returned, and the
        following prefill aborted at an L1 address that had not moved - which was
        once read as "the buffer's L1 is not returned when the last reference
        goes".

        That reading was wrong, and it is measured to be wrong: a second
        production-size buffer is created the instant the first is dropped, and
        792 256 B per bank come back once `cleanup()` clears every context -
        against a `GALAXY_GLOBAL_CB_SIZE` of 792 064 B. The references were the
        problem, not the destructor.
        """

        if not self.config.release_global_cb_on_prefill or self._global_cb is None:
            return
        for context in self._contexts.values():
            object.__setattr__(context, "global_cb", None)
        self._global_cb = None
        # Printed, not logged at debug, and unconditional once the flag is on: the
        # only way to tell "the release did not run" from "the release did not
        # help" in a device log is to see this line.
        print("[prefetcher] released the global circular buffer on entering prefill", flush=True)

    def cleanup(self) -> None:
        if self._cleaned:
            return

        first_error: Exception | None = None

        def attempt(action: Callable[[], None]) -> None:
            nonlocal first_error
            try:
                action()
            except Exception as exc:
                if first_error is None:
                    first_error = exc

        if self._prefetch_result is not None:
            attempt(self._stop_prefetch)
            if self._prefetch_result is not None:
                self._retained_prefetch_resources.append(self._prefetch_result)
                self._prefetch_result = None
        mesh = self.config.mesh_device
        if self._loaded_mode is not None:
            attempt(mesh.reset_sub_device_stall_group)
            attempt(mesh.clear_loaded_sub_device_manager)
            self._active_mode = None
            self._loaded_mode = None

        resources = list(reversed(self._retained_prefetch_resources))
        self._retained_prefetch_resources.clear()
        if self._weight_address_metadata is not None:
            resources.append(self._weight_address_metadata)
        seen: set[int] = set()
        for resource in resources:
            if id(resource) not in seen:
                attempt(lambda resource=resource: self._deallocate(resource))
                seen.add(id(resource))
        self._weight_address_metadata = None
        # Every context this owner handed out has to give the buffer back too.
        # Module configs capture the *context object* at construction
        # (`MLP2DConfig.decode_prefetch_context`), so clearing `self._contexts`
        # below drops this owner's map and leaves the modules holding a context
        # whose `global_cb` is still the live buffer. There is no `deallocate` on
        # a `global_circular_buffer`: its ~774 kB of L1 per sender/receiver core
        # is held by the C++ object and freed by its destructor, so *every*
        # Python reference has to go before the L1 comes back. Measured on
        # `(8, 4)`: with the last reference dropped the allocator does return it
        # and a second buffer of the same size can be created, which is what made
        # a surviving reference the explanation for the second model in a process
        # finding 923 776 of 1 393 472 B per bank still allocated after the first
        # was closed, collected, and cleaned up.
        for context in self._contexts.values():
            object.__setattr__(context, "global_cb", None)
        self._global_cb = None

        for mode in ("decode", "prefill"):
            manager = self._managers.pop(mode, None)
            if manager is not None:
                attempt(lambda manager=manager: mesh.remove_sub_device_manager(manager))

        self._contexts.clear()
        # Announce the release only now: every reference to the buffer is gone
        # above, so by this point its L1 is genuinely back and a listener that
        # retires programs holding its address is retiring them against a fact.
        if self.config.on_global_cb_released is not None:
            attempt(self.config.on_global_cb_released)
        self._cleaned = True
        if first_error is not None:
            raise first_error

    def __enter__(self) -> Prefetcher2D:
        self.initialize()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.cleanup()

    def _make_context(
        self,
        mode_config: Prefetcher2DModeConfig,
        *,
        global_cb: Any,
        weights: tuple[Any, ...],
        weight_addresses: Any,
    ) -> Prefetcher2DContext:
        return Prefetcher2DContext(
            mode=mode_config.mode,
            mesh_device=self.config.mesh_device,
            sub_device_manager_id=self._managers[mode_config.mode],
            worker_sub_device_id=mode_config.worker_sub_device_id,
            stall_group=mode_config.stall_group,
            global_cb=global_cb,
            weights=weights,
            weight_addresses=weight_addresses,
            weight_address_metadata=self._weight_address_metadata,
        )

    def _configure_mode(self, context: Prefetcher2DContext) -> None:
        mode_config = self.config.prefill if context.mode == "prefill" else self.config.decode
        self._configure_mode_resources(mode_config)

    def _configure_mode_resources(self, mode_config: Prefetcher2DModeConfig) -> None:
        mesh = self.config.mesh_device
        mesh.load_sub_device_manager(self._managers[mode_config.mode])
        if mode_config.mode == "prefill":
            mesh.set_sub_device_stall_group(list(mode_config.stall_group))
        else:
            mesh.set_sub_device_stall_group(
                [ttnn.SubDeviceId(index) for index in range(len(self.config.decode.sub_devices))]
            )
        self._loaded_mode = mode_config.mode

    def _start_prefetch(self, context: Prefetcher2DContext) -> None:
        result = self._dram_prefetch_start(context)
        if result is None:
            raise RuntimeError("dram prefetch start must return an owned result")
        self._prefetch_result = result
        self.config.mesh_device.set_sub_device_stall_group(list(context.stall_group))

    def _stop_prefetch(self, *, suppress_errors: bool = False) -> None:
        result = self._prefetch_result
        if result is None:
            return
        try:
            sync_result = self._dram_prefetch_stop(self.config.mesh_device, result)
        except Exception:
            if suppress_errors:
                self._prefetch_result = None
                self._retained_prefetch_resources.append(result)
                return
            raise
        self._prefetch_result = None
        if sync_result is None:
            return
        self._retained_prefetch_resources.append(sync_result)

    def _default_validate_weight(self, name: str, tensor: Any, existing: tuple[Any, ...]) -> None:
        del name
        address = tensor.buffer_address()
        if not isinstance(address, int) or address < 0:
            raise ValueError("registered weight buffer address must be a non-negative integer")
        if any(other is tensor or other.buffer_address() == address for other in existing):
            raise ValueError("registered weights must refer to distinct device buffers")

    def _default_derive_global_cb_size(self, weights: tuple[Any, ...]) -> int:
        sizes: list[int] = []
        for weight in weights:
            buffer_size = getattr(weight, "buffer_size", None)
            if callable(buffer_size):
                sizes.append(int(buffer_size()))
        if sizes:
            return 2 * max(sizes)
        if self.config.global_cb_size is None:
            raise ValueError("global_cb_size requires an injected deriver when weights do not expose buffer_size()")
        return self.config.global_cb_size

    def _default_dram_prefetch_start(self, context: Prefetcher2DContext) -> Any:
        return ttnn.dram_prefetcher(
            list(context.weights) + [context.weight_address_metadata],
            num_layers=self.config.prefetch_num_layers,
            global_cb=context.global_cb,
        )

    @staticmethod
    def _default_dram_prefetch_stop(mesh_device: Any, result: Any) -> Any:
        del mesh_device
        ttnn.deallocate(result)
        return None

    def _ensure_open(self) -> None:
        if self._cleaned:
            raise RuntimeError("Prefetcher2D has been cleaned up")
