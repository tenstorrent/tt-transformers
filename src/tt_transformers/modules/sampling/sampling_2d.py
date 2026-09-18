# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Sampling for the canonical Galaxy ``(8, 4)`` mesh.

The vocabulary is sharded over mesh rows and user slots are sharded over mesh
columns. Sampling controls are invocation data: the config owns only the
capacity and lazy device buffers used to transfer those values.
"""

from __future__ import annotations

import hashlib
import secrets
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Any

import torch
import ttnn

from tt_transformers.device_ownership import compatibility_default_device
from tt_transformers.modules.galaxy_mesh import (
    GALAXY_MESH_SHAPE,
    is_galaxy_architecture_name,
    require_galaxy_mesh,
)
from tt_transformers.modules.lazy_buffer import LazyBuffer, resolve_lazy_buffer
from tt_transformers.modules.lightweightmodule import LightweightModule
from tt_transformers.sampling.vocab_padding import build_invalid_vocab_mask


@dataclass(frozen=True)
class _HostMeshMapper:
    """Placement metadata used when config tests provide a mock mesh."""

    dims: tuple[int | None, int | None]
    mesh_shape: tuple[int, int]


def _subgrid(grid: Any, start_core: Any, count: int) -> Any:
    """Return the first `count` cores of `grid`, or `None` if that is not possible.

    `ttnn.num_cores_to_corerangeset_in_subcoregrids` is a positional-only
    binding over real `CoreCoord`/`CoreRangeSet` objects, so it cannot be handed
    the placeholder grids the config tests use. Returning `None` there keeps
    those tests exercising the surrounding logic, and on hardware a real grid is
    always present.
    """

    if grid is None:
        return None
    try:
        start = start_core or grid.bounding_box().start
        return ttnn.num_cores_to_corerangeset_in_subcoregrids(start, count, grid, True)
    except (TypeError, AttributeError):
        return None


def _no_sub_device() -> None:
    """Name no sub-device, which is what `ttnn.all_gather` defaults to anyway."""

    return None


@dataclass(frozen=True)
class Sampling2DCall:
    """Normalized, immutable values for one sampling invocation."""

    slot_ids: tuple[int, ...]
    top_k: tuple[int, ...]
    top_p: tuple[float, ...]
    temperature: tuple[float, ...]
    seed: tuple[int | None, ...]
    forced_argmax: tuple[bool, ...]


@dataclass(frozen=True)
class Sampling2DConfig:
    """Declarative Sampling2D configuration.

    ``vocab_size`` is the logical token vocabulary. ``padded_vocab_size`` is
    the LM-head output width and may be larger for tile/device alignment.
    """

    vocab_size: int
    padded_vocab_size: int | None = None
    mesh_device: ttnn.MeshDevice | None = None
    cluster_shape: tuple[int, int] = GALAXY_MESH_SHAPE
    architecture: Any = "wormhole_b0"
    max_batch_size: int = 32
    max_top_k: int = 32
    sampling_all_gather_axis: int = 0
    user_shard_axis: int = 1
    num_gather_links: int = 1
    sampling_memory_config: ttnn.MemoryConfig | None = None
    sub_core_grids: Any = None
    sub_core_grid_topk: Any = None
    start_core: ttnn.CoreCoord | None = None
    #: Whether the physical batch is **replicated** across the mesh's
    #: `user_shard_axis` instead of sharded over it.
    #:
    #: `False`, the default, is the placement this module was written for and
    #: the one both of its older device tests build: `max_batch_size // 4`
    #: users per column, every per-user buffer sharded over the four columns.
    #:
    #: `True` is the placement the **Galaxy decode graph actually produces**.
    #: `Llama33_70BGalaxyTransformer2D.decode_forward` keeps the physical batch
    #: replicated across columns and shards the *vocabulary* over the eight
    #: rows, so `lm_head_2d` hands the sampler `[1, 1, 32, padded_vocab // 8]`
    #: on every device - and `model.py` says so in terms, at the site where it
    #: builds this config: "Sampling2D consumes logits whose users are sharded
    #: over the four mesh columns, while the decode graph keeps the physical
    #: batch replicated across columns."
    #:
    #: The common runtime requires the same thing of the *output*: the
    #: device-feedback path reshapes the sampled tokens to the shape of a
    #: `ReplicateTensorToMesh` token tensor (`decode.py:693`), which is 32 wide
    #: on every device. So this is not one of two equivalent placements - it is
    #: the one this executor path needs, and it previously had no expression
    #: here. Under `False` the mask add aborts on the mismatch it cannot
    #: broadcast:
    #:
    #:     ttnn.add([1,1,32,16128] bf16, [1,1,8,16128] bf16)
    #:     TT_THROW @ binary_ng_device_operation.cpp:224 Invalid subtile
    #:     broadcast type
    replicate_users: bool = False
    #: Resolve the decode worker sub-device at call time. `ttnn.all_gather`
    #: takes no `sub_core_grids`; given no `subdevice_id` it places its workers
    #: on `get_sub_device_ids().at(0)`, which under the Galaxy decode manager is
    #: the *prefetch sender* partition rather than the worker one. That is legal
    #: and silently unordered: fast dispatch orders programs per sub-device
    #: only, so a gather on sub-device 0 never waits for the `topk` that feeds
    #: it. On a program-cache miss the binary load, the gather's own semaphore
    #: `Synchronize` and the JIT gap serialize it by accident; on a hit all
    #: three vanish and the gather consumes the *previous* call's `topk` output.
    #: Callable, not a value: the id belongs to the live operation-boundary
    #: context, which is created after this config.
    ccl_sub_device_id: Callable[[], object] = _no_sub_device

    # Persistent mutable state. Values are refreshed for every invocation.
    top_k_buffer: LazyBuffer | ttnn.Tensor | None = None
    top_p_buffer: LazyBuffer | ttnn.Tensor | None = None
    temperature_buffer: LazyBuffer | ttnn.Tensor | None = None
    seed_buffer: LazyBuffer | ttnn.Tensor | None = None
    user_ids: LazyBuffer | ttnn.Tensor | None = None
    index_offsets: LazyBuffer | ttnn.Tensor | None = None
    local_indices: LazyBuffer | ttnn.Tensor | None = None
    invalid_vocab_mask: LazyBuffer | ttnn.Tensor | None = None

    @property
    def vocab_shards(self) -> int:
        return self.cluster_shape[self.sampling_all_gather_axis]

    @property
    def user_shards(self) -> int:
        return self.cluster_shape[self.user_shard_axis]

    @property
    def users_per_shard(self) -> int:
        if self.replicate_users:
            return self.max_batch_size
        return self.max_batch_size // self.user_shards

    @staticmethod
    def _buffer_resolved(buffer: LazyBuffer | ttnn.Tensor | None) -> bool:
        if isinstance(buffer, ttnn.Tensor):
            return True
        return isinstance(buffer, LazyBuffer) and buffer.is_resolved()

    def is_resolved(self) -> bool:
        required = (
            self.top_k_buffer,
            self.top_p_buffer,
            self.temperature_buffer,
            self.seed_buffer,
            self.user_ids,
            self.index_offsets,
            self.local_indices,
        )
        if self.mesh_device is None or self.padded_vocab_size is None:
            return False
        if not all(self._buffer_resolved(buffer) for buffer in required):
            return False
        if self.padded_vocab_size > self.vocab_size:
            return self._buffer_resolved(self.invalid_vocab_mask)
        return True


class Sampling2D(LightweightModule):
    """Top-k/top-p sampler for 2D-sharded Galaxy logits."""

    def __init__(
        self,
        vocab_size: int,
        padded_vocab_size: int | None = None,
        mesh_device: ttnn.MeshDevice | None = None,
        **kwargs,
    ):
        super().__init__()
        self.config = _resolve_sampling2d_config(
            Sampling2DConfig(
                vocab_size=vocab_size,
                padded_vocab_size=padded_vocab_size,
                mesh_device=mesh_device,
                **kwargs,
            )
        )
        self._device_buffers_loaded = False

    @classmethod
    def from_config(cls, config: Sampling2DConfig) -> Sampling2D:
        instance = object.__new__(cls)
        super(Sampling2D, instance).__init__()
        instance.config = _resolve_sampling2d_config(config)
        instance._device_buffers_loaded = False
        return instance

    def slot_placement(self, slot_id: int) -> tuple[int, int]:
        """Return ``(mesh column, local user index)`` for a global slot."""
        if slot_id < 0 or slot_id >= self.config.max_batch_size:
            raise ValueError(f"slot_id must be in [0, {self.config.max_batch_size}), got {slot_id}")
        return divmod(slot_id, self.config.users_per_shard)

    def prepare_call(
        self,
        *,
        top_k: int | Sequence[int] | torch.Tensor,
        top_p: float | Sequence[float] | torch.Tensor,
        temperature: float | Sequence[float] | torch.Tensor,
        seed: int | None | Sequence[int | None] | torch.Tensor = None,
        forced_argmax: bool | Sequence[bool] | torch.Tensor = False,
        slot_ids: Sequence[int] | torch.Tensor | None = None,
        update_buffers: bool = True,
    ) -> Sampling2DCall:
        """Normalize one call without retaining request values on the module."""
        slots = _normalize_slots(slot_ids, self.config.max_batch_size)
        size = len(slots)
        call = Sampling2DCall(
            slot_ids=slots,
            top_k=tuple(int(value) for value in _broadcast(top_k, size, "top_k")),
            top_p=tuple(float(value) for value in _broadcast(top_p, size, "top_p")),
            temperature=tuple(float(value) for value in _broadcast(temperature, size, "temperature")),
            seed=tuple(_optional_int(value, "seed") for value in _broadcast(seed, size, "seed")),
            forced_argmax=tuple(bool(value) for value in _broadcast(forced_argmax, size, "forced_argmax")),
        )
        self._validate_call(call)
        if update_buffers:
            self._update_call_buffers(call)
        return call

    def _validate_call(self, call: Sampling2DCall) -> None:
        if len(set(call.slot_ids)) != len(call.slot_ids):
            raise ValueError("slot_ids must be unique")
        for slot in call.slot_ids:
            self.slot_placement(slot)
        for value in call.top_k:
            if value < 1 or value > self.config.max_top_k:
                raise ValueError(f"top_k must be in [1, {self.config.max_top_k}], got {value}")
        for value in call.top_p:
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"top_p must be in [0, 1], got {value}")
        for value in call.temperature:
            if value < 0.0:
                raise ValueError(f"temperature must be nonnegative, got {value}")
        for value in call.seed:
            if value is not None and value < 0:
                raise ValueError(f"seed must be nonnegative, got {value}")

    def _update_call_buffers(self, call: Sampling2DCall) -> None:
        cfg = self.config
        k_values = torch.ones(cfg.max_batch_size, dtype=torch.int32)
        p_values = torch.zeros(cfg.max_batch_size, dtype=torch.bfloat16)
        # ttnn.sampling's ``temp`` argument is the *reciprocal* temperature: the kernel
        # multiplies the candidate logits by it before the softmax. This buffer therefore
        # carries 1/T, never T. 1.0 is its own reciprocal, so greedy slots are unaffected -
        # which is why passing T straight through was invisible on the greedy path.
        temperature_values = torch.ones(cfg.max_batch_size, dtype=torch.bfloat16)
        seed_values = torch.tensor([secrets.randbits(31) for _ in range(cfg.max_batch_size)], dtype=torch.int32)

        for index, slot in enumerate(call.slot_ids):
            force_greedy = call.forced_argmax[index] or call.temperature[index] == 0.0
            k_values[slot] = 1 if force_greedy else call.top_k[index]
            p_values[slot] = 0.0 if force_greedy else call.top_p[index]
            temperature_values[slot] = 1.0 if force_greedy else 1.0 / call.temperature[index]
            if call.seed[index] is not None:
                seed_values[slot] = _device_seed(call.seed[index], slot)

        _update_lazy(cfg.top_k_buffer, k_values)
        _update_lazy(cfg.top_p_buffer, p_values)
        _update_lazy(cfg.temperature_buffer, temperature_values)
        _update_lazy(cfg.seed_buffer, seed_values)

    def sample_host(
        self,
        logits: torch.Tensor,
        *,
        top_k: int | Sequence[int] | torch.Tensor,
        top_p: float | Sequence[float] | torch.Tensor,
        temperature: float | Sequence[float] | torch.Tensor,
        seed: int | None | Sequence[int | None] | torch.Tensor = None,
        forced_argmax: bool | Sequence[bool] | torch.Tensor = False,
        slot_ids: Sequence[int] | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Host reference path used by focused tests and integration validation."""
        if logits.ndim < 2:
            raise ValueError("logits must have a batch and vocabulary dimension")
        if logits.shape[-1] != self.config.padded_vocab_size:
            raise ValueError(f"expected padded logits width {self.config.padded_vocab_size}, got {logits.shape[-1]}")
        call = self.prepare_call(
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            seed=seed,
            forced_argmax=forced_argmax,
            slot_ids=slot_ids,
            update_buffers=False,
        )
        rows = logits.reshape(-1, logits.shape[-1])
        if rows.shape[0] != len(call.slot_ids):
            raise ValueError(f"logits has {rows.shape[0]} rows but the call has {len(call.slot_ids)} slots")

        # Invalid LM-head padding is removed before either argmax or top-k.
        rows = rows[:, : self.config.vocab_size].float()
        output = []
        for row_index, slot in enumerate(call.slot_ids):
            row = rows[row_index]
            if call.forced_argmax[row_index] or call.temperature[row_index] == 0.0:
                token = torch.argmax(row)
            else:
                k = min(call.top_k[row_index], self.config.vocab_size)
                values, indices = torch.topk(row / call.temperature[row_index], k=k)
                probabilities = torch.softmax(values, dim=-1)
                p = call.top_p[row_index]
                if p < 1.0:
                    cumulative = probabilities.cumsum(dim=-1)
                    remove = cumulative - probabilities > p
                    probabilities = probabilities.masked_fill(remove, 0.0)
                    probabilities /= probabilities.sum()
                generator = torch.Generator(device="cpu")
                request_seed = call.seed[row_index]
                generator.manual_seed(secrets.randbits(63) if request_seed is None else _host_seed(request_seed, slot))
                selected = torch.multinomial(probabilities.cpu(), 1, generator=generator)
                token = indices.cpu()[selected].squeeze(0)
            token_id = int(token.item())
            if token_id >= self.config.vocab_size:
                raise RuntimeError(f"sampling selected padded vocabulary id {token_id}")
            output.append(token_id)
        return torch.tensor(output, dtype=torch.int64, device=logits.device)

    def load_device_buffers(self) -> None:
        if self._device_buffers_loaded:
            return
        cfg = self.config
        self._top_k = _materialize(cfg.top_k_buffer)
        self._top_p = _materialize(cfg.top_p_buffer)
        self._temperature = _materialize(cfg.temperature_buffer)
        self._seeds = _materialize(cfg.seed_buffer)
        self._user_ids = _materialize(cfg.user_ids)
        self._index_offsets = _materialize(cfg.index_offsets)
        self._local_indices = _materialize(cfg.local_indices)
        self._invalid_vocab_mask = _materialize(cfg.invalid_vocab_mask) if cfg.invalid_vocab_mask is not None else None
        self._device_buffers_loaded = True

    def _sampling_core_grid(self):
        """Return the `ttnn.sampling` core grid, or `None` for the whole device.

        `ttnn.sampling` documents that a supplied grid "must supply at least
        ``num_users`` cores", so this takes exactly `users_per_shard` of
        `sub_core_grids` from `start_core` - the construction the production 1D
        sampler makes for the same op. Given none, the op takes the whole
        compute grid, which under a loaded sub-device manager is
        `TT_FATAL @ program.cpp:2205 Kernel group cores do not match sub device
        cores`.
        """

        return _subgrid(self.config.sub_core_grids, self.config.start_core, self.config.users_per_shard)

    def _place_for_topk(self, logits):
        """Return `logits` in a placement `ttnn.topk` can reduce in a sub-device.

        Identity unless the input is interleaved *and* carries implicit tile
        padding on the user axis, which is exactly the case whose padding fill
        `ttnn.topk` would place on the whole compute grid. The shard is a width
        shard over as many cores of `sub_core_grid_topk` as divide the width in
        whole tiles - a width-sharded L1 shard must be a whole number of tiles
        wide, and the two Galaxy vocabularies do not share a divisor:

            TT_FATAL ... Physical shard shape (32, 537) must be tile {32, 32} sized!

        so the count is searched for rather than named, the way
        `recipes.lm_head_reduce_core_count` searches for the LM head reduction's.

        One known limit, stated rather than guarded: the shard height is one
        tile, which is right for every Galaxy decode shape (`users_per_shard` is
        8, so the padded height is 32) and wrong for a hypothetical input padded
        to two tiles or more. That case aborts in `interleaved_to_sharded` on the
        shard shape rather than sampling anything, so it is loud, not silent.
        """

        cfg = self.config
        if cfg.sub_core_grid_topk is None or logits.memory_config().is_sharded():
            return logits
        logical = tuple(int(value) for value in logits.shape)
        tile = ttnn.TILE_SIZE
        height_padded = bool(logical[-2] % tile)
        width = logical[-1]
        if not height_padded or width % tile:
            return logits
        tiles = width // tile
        available = cfg.sub_core_grid_topk.num_cores()
        count = next((value for value in range(min(available, tiles), 0, -1) if not tiles % value), 0)
        if count == 0:
            return logits
        cores = _subgrid(cfg.sub_core_grid_topk, cfg.start_core, count)
        if cores is None:
            return logits
        return ttnn.interleaved_to_sharded(
            logits,
            ttnn.create_sharded_memory_config(
                shape=(tile, width // count),
                core_grid=cores,
                strategy=ttnn.ShardStrategy.WIDTH,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            ),
        )

    def decode_forward(
        self,
        logits,
        *,
        top_k,
        top_p,
        temperature,
        seed=None,
        forced_argmax=False,
        slot_ids=None,
        tt_out_tok=None,
        update_buffers=True,
    ):
        if isinstance(logits, torch.Tensor):
            return self.sample_host(
                logits,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                seed=seed,
                forced_argmax=forced_argmax,
                slot_ids=slot_ids,
            )

        # `update_buffers=False` runs the op chain against controls a caller
        # already wrote. `_update_call_buffers` performs four host-to-device
        # writes and draws a fresh random seed, and a caller capturing this chain
        # into a trace must do both *outside* the capture region: a trace records
        # program dispatch, so a write inside it is neither legal nor replayed.
        # The default is unchanged, and `prepare_call` still validates the call.
        self.prepare_call(
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            seed=seed,
            forced_argmax=forced_argmax,
            slot_ids=slot_ids,
            update_buffers=update_buffers,
        )
        self.load_device_buffers()
        cfg = self.config
        owned = []

        def own(tensor):
            if tensor is not logits and not any(value is tensor for value in owned):
                owned.append(tensor)
            return tensor

        try:
            sampled_logits = logits
            # The Galaxy decode LM head hands the sampler its logits
            # **width-sharded in L1 over the 24-core `gather_in0` ring**
            # (`lm_head_2d.py:330` resolves `decode_output_memcfg`, and the
            # comment at `:215` says why the ring is the only placement whose
            # circular buffers fit beside the resident decode activations).
            # Every eltwise op below takes `sub_core_grids`, and **that argument
            # is honoured only for an interleaved input**: given a sharded one
            # the op resolves its cores from the shard spec, and for a
            # `CoreRangeSet` of 24 scattered single cores that is the *bounding
            # box*. Under the loaded decode partition the ring's bounding box
            # `(1,0)-(6,9)` contains exactly two cores that belong to no
            # sub-device - `(4,3)` and `(4,8)`, the two rows of the prefetch
            # sender column that carry no sender - so 58 of its 60 cores
            # intersect and the program aborts before it runs:
            #
            #     TT_FATAL @ program.cpp:2205: num_intersections == num_cores
            #     Kernel group cores do not match sub device cores for TENSIX
            #
            # Flattening the input once, here, puts every
            # placement decision downstream back on the interleaved path this
            # module's qualified device tests measure. Seven placements of the
            # real tensor were tried in situ and only the two that go through an
            # interleaved tensor first survive: all five that keep it sharded
            # abort, including the one that asks for the contiguous `topk` grid,
            # because the grid argument is not consulted at all.
            #
            # `sharded_to_interleaved` itself places: unlike the eltwise ops it
            # uses the shard's own 24 cores rather than their bounding box.
            if sampled_logits.memory_config().is_sharded():
                sampled_logits = own(ttnn.sharded_to_interleaved(sampled_logits, ttnn.DRAM_MEMORY_CONFIG))
            if sampled_logits.dtype != ttnn.bfloat16:
                sampled_logits = own(
                    ttnn.typecast(sampled_logits, dtype=ttnn.bfloat16, sub_core_grids=cfg.sub_core_grids)
                )
            if self._invalid_vocab_mask is not None:
                sampled_logits = own(
                    ttnn.add(
                        sampled_logits,
                        self._invalid_vocab_mask,
                        memory_config=sampled_logits.memory_config(),
                        sub_core_grids=cfg.sub_core_grids,
                    )
                )
            # `ttnn.topk` fills its input's implicit tile padding before it
            # reduces, and `fill_implicit_tile_padding` takes no sub-core grid:
            # the *interleaved* fill_pad factory resolves its cores from
            # `device->compute_with_storage_grid_size()`, while the *sharded*
            # one uses `shard_spec.grid`. Decode logits arrive with
            # `users_per_shard` logical rows in a 32-row tile, so the fill
            # always fires, and under a loaded sub-device manager the
            # interleaved placement aborts before the reduction runs:
            #
            #     TT_FATAL @ program.cpp:2205: num_intersections == num_cores
            #     Kernel group cores do not match sub device cores
            #
            # so a padded input is width-sharded onto `sub_core_grid_topk`
            # first. The explicit `memory_config` is part of the same fix: with
            # none, `ttnn.topk` inherits the input's, and it rejects a sharded
            # output (`topk_device_operation.cpp:149`).
            topk_input = own(self._place_for_topk(sampled_logits))
            local_values, local_indices = ttnn.topk(
                topk_input,
                k=cfg.max_top_k,
                dim=-1,
                indices_tensor=self._local_indices,
                sub_core_grids=cfg.sub_core_grid_topk,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            own(local_values)
            own(local_indices)
            # `subdevice_id` on both gathers, and `sub_core_grids` on the int32
            # `add` below, keep every op in this chain inside the one sub-device
            # dispatch orders programs within. See `ccl_sub_device_id`: the two
            # gathers and that `add` are the only sites here that resolve their
            # placement from a *default*, and each default names sub-device 0.
            ccl_sub_device_id = cfg.ccl_sub_device_id()
            gathered_values = own(
                ttnn.all_gather(
                    local_values,
                    dim=3,
                    num_links=cfg.num_gather_links,
                    cluster_axis=cfg.sampling_all_gather_axis,
                    topology=ttnn.Topology.Linear,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    subdevice_id=ccl_sub_device_id,
                )
            )
            gathered_indices = own(
                ttnn.all_gather(
                    local_indices,
                    dim=3,
                    num_links=cfg.num_gather_links,
                    cluster_axis=cfg.sampling_all_gather_axis,
                    topology=ttnn.Topology.Linear,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    subdevice_id=ccl_sub_device_id,
                )
            )
            gathered_indices = own(ttnn.typecast(gathered_indices, dtype=ttnn.int32, sub_core_grids=cfg.sub_core_grids))
            global_indices = own(
                ttnn.add(
                    self._index_offsets,
                    gathered_indices,
                    dtype=ttnn.int32,
                    sub_core_grids=cfg.sub_core_grids,
                )
            )
            global_indices = own(ttnn.untilize(global_indices, use_multicore=True, sub_core_grids=cfg.sub_core_grids))
            # `sub_core_grids` for the same reason `topk` needs a placement: with
            # none the seed program takes the whole compute grid, which under a
            # loaded sub-device manager is `TT_FATAL @ program.cpp:2205 Kernel
            # group cores do not match sub device cores`.
            ttnn.manual_seed(seeds=self._seeds, user_ids=self._user_ids, sub_core_grids=cfg.sub_core_grids)
            result = ttnn.sampling(
                gathered_values,
                global_indices,
                k=self._top_k,
                p=self._top_p,
                temp=self._temperature,
                sub_core_grids=self._sampling_core_grid(),
                output_tensor=tt_out_tok,
            )
            owned[:] = [tensor for tensor in owned if tensor is not result]
            return result
        finally:
            for tensor in reversed(owned):
                ttnn.deallocate(tensor)

    def forward(self, logits, **kwargs):
        return self.decode_forward(logits, **kwargs)

    def release(self) -> None:
        failures = []
        for name in (
            "top_k_buffer",
            "top_p_buffer",
            "temperature_buffer",
            "seed_buffer",
            "user_ids",
            "index_offsets",
            "local_indices",
            "invalid_vocab_mask",
        ):
            buffer = getattr(self.config, name)
            if isinstance(buffer, LazyBuffer):
                try:
                    buffer.release()
                except BaseException as error:
                    failures.append(error)
        self._device_buffers_loaded = False
        for name in (
            "_top_k",
            "_top_p",
            "_temperature",
            "_seeds",
            "_user_ids",
            "_index_offsets",
            "_local_indices",
            "_invalid_vocab_mask",
        ):
            if hasattr(self, name):
                delattr(self, name)
        if failures:
            raise failures[0]


def _resolve_sampling2d_config(config: Sampling2DConfig) -> Sampling2DConfig:
    """Validate Galaxy geometry and fill all lazy buffer specifications."""
    _validate_static_config(config)
    mesh_device = config.mesh_device or compatibility_default_device(
        ttnn, owner="modules.sampling._resolve_sampling2d_config"
    )
    if mesh_device is None:
        raise ValueError("Sampling2D mesh_device must be provided")
    require_galaxy_mesh("Sampling2D", mesh_device)

    padded_vocab_size = config.padded_vocab_size or config.vocab_size
    vocab_shards = config.cluster_shape[config.sampling_all_gather_axis]
    if padded_vocab_size < config.vocab_size:
        raise ValueError("padded_vocab_size must be at least vocab_size")
    if padded_vocab_size % vocab_shards != 0:
        raise ValueError(f"padded_vocab_size must be divisible by {vocab_shards} vocabulary shards")
    local_vocab = padded_vocab_size // vocab_shards
    if local_vocab % ttnn.TILE_SIZE != 0:
        raise ValueError(f"local padded vocabulary width must be tile aligned, got {local_vocab}")
    required_multiple = vocab_shards * ttnn.TILE_SIZE
    minimum_padded_vocab_size = ((config.vocab_size + required_multiple - 1) // required_multiple) * required_multiple
    # A *multiple* of the vocabulary-shard tile, not the *minimal* one. This used
    # to demand exactly the minimum, and that forbade the only width the Galaxy
    # decode chain can run: `all_reduce_async`'s reduction kernel waits for a full
    # shard on every output core, so the LM head's reduced logits must be an exact
    # multiple of `cores * shard_width`, and Llama's minimal padding leaves 501
    # tiles per device - a width no usable core count divides. See
    # `galaxy_padded_vocab_size`, which pads to a ring-exact width.
    #
    # The check still fails closed on a nonsense width: padding may not add a whole
    # extra vocabulary shard per mesh row. The masking that makes the padding
    # harmless is `LMHead2D`'s -inf invalid-logits mask, upstream of here.
    if padded_vocab_size < minimum_padded_vocab_size:
        raise ValueError(
            f"padded_vocab_size must be at least the Galaxy-aligned width {minimum_padded_vocab_size}, "
            f"got {padded_vocab_size}"
        )
    if padded_vocab_size >= minimum_padded_vocab_size + required_multiple * vocab_shards:
        raise ValueError(
            f"padded_vocab_size {padded_vocab_size} pads vocab_size {config.vocab_size} by more than one "
            f"vocabulary shard per mesh row; that is a geometry mistake, not a padding choice"
        )

    memory_config = config.sampling_memory_config or ttnn.DRAM_MEMORY_CONFIG
    start_core = config.start_core or ttnn.CoreCoord(0, 0)
    # Every per-user buffer is placed the same way the logits are. With the
    # batch sharded over the columns that is a shard on the user axis; with it
    # replicated across them, the user axis is replicated too and only the
    # vocabulary axis is still shared out. See `replicate_users`.
    user_axis = None if config.replicate_users else 0
    user_row_axis = None if config.replicate_users else 2
    user_mapper = _make_mesh_mapper(mesh_device, (None, user_axis), config.cluster_shape)
    user_row_mapper = _make_mesh_mapper(mesh_device, (None, user_row_axis), config.cluster_shape)
    logits_mapper = _make_mesh_mapper(mesh_device, (3, user_row_axis), config.cluster_shape)
    replicated_mapper = _make_mesh_mapper(mesh_device, (None, None), config.cluster_shape)

    def resolve_buffer(value, source, *, dtype, layout, mapper, buffer_memory_config=ttnn.DRAM_MEMORY_CONFIG):
        defaults = dict(
            dtype=dtype,
            layout=layout,
            device=mesh_device,
            mesh_mapper=mapper,
            memory_config=buffer_memory_config,
        )
        if value is None:
            return LazyBuffer(source=source, **defaults)
        if isinstance(value, ttnn.Tensor):
            return value
        return resolve_lazy_buffer(value, **defaults)

    batch = config.max_batch_size
    max_top_k = config.max_top_k
    offsets = torch.arange(vocab_shards, dtype=torch.int32) * local_vocab
    offsets = offsets.repeat_interleave(max_top_k).view(1, 1, 1, -1).expand(1, 1, batch, -1).contiguous()
    local_indices = torch.arange(local_vocab, dtype=torch.int32).view(1, 1, 1, -1)
    local_indices = local_indices.expand(1, 1, batch, -1).contiguous()
    invalid_mask = build_invalid_vocab_mask(config.vocab_size, padded_vocab_size, batch)

    values = dict(
        mesh_device=mesh_device,
        padded_vocab_size=padded_vocab_size,
        sampling_memory_config=memory_config,
        start_core=start_core,
        top_k_buffer=resolve_buffer(
            config.top_k_buffer,
            torch.ones(batch, dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mapper=user_mapper,
        ),
        top_p_buffer=resolve_buffer(
            config.top_p_buffer,
            torch.zeros(batch, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mapper=user_mapper,
        ),
        temperature_buffer=resolve_buffer(
            config.temperature_buffer,
            torch.ones(batch, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mapper=user_mapper,
        ),
        seed_buffer=resolve_buffer(
            config.seed_buffer,
            torch.arange(batch, dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mapper=user_mapper,
        ),
        user_ids=resolve_buffer(
            config.user_ids,
            torch.arange(config.users_per_shard, dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mapper=replicated_mapper,
        ),
        index_offsets=resolve_buffer(
            config.index_offsets,
            offsets,
            dtype=ttnn.int32,
            layout=ttnn.TILE_LAYOUT,
            mapper=user_row_mapper,
            buffer_memory_config=memory_config,
        ),
        local_indices=resolve_buffer(
            config.local_indices,
            local_indices,
            dtype=ttnn.uint16,
            layout=ttnn.TILE_LAYOUT,
            mapper=user_row_mapper,
        ),
        invalid_vocab_mask=(
            resolve_buffer(
                config.invalid_vocab_mask,
                invalid_mask,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                mapper=logits_mapper,
            )
            if invalid_mask is not None or config.invalid_vocab_mask is not None
            else None
        ),
    )
    resolved = replace(config, **values)
    assert resolved.is_resolved()
    return resolved


def _validate_static_config(config: Sampling2DConfig) -> None:
    if tuple(config.cluster_shape) != GALAXY_MESH_SHAPE:
        raise ValueError(f"Sampling2D supports only cluster_shape {GALAXY_MESH_SHAPE}")
    if not is_galaxy_architecture_name(config.architecture):
        raise ValueError(f"Sampling2D supports only Galaxy architectures, got {config.architecture}")
    if config.sampling_all_gather_axis != 0 or config.user_shard_axis != 1:
        raise ValueError("Sampling2D requires vocabulary axis 0 and user axis 1")
    if config.vocab_size <= 0:
        raise ValueError("vocab_size must be positive")
    if config.max_batch_size != 32:
        raise ValueError("Sampling2D requires Galaxy physical batch 32")
    if config.max_top_k <= 0:
        raise ValueError("max_top_k must be positive")
    if config.num_gather_links <= 0:
        raise ValueError("num_gather_links must be positive")
    if config.sub_core_grids is None or config.sub_core_grid_topk is None:
        raise ValueError("Sampling2D requires explicit sub_core_grids and sub_core_grid_topk resources")


def _make_mesh_mapper(mesh_device, dims, mesh_shape):
    try:
        return ttnn.ShardTensor2dMesh(mesh_device, dims=dims, mesh_shape=mesh_shape)
    except TypeError:
        if type(mesh_device).__module__ == "unittest.mock":
            # Host/config tests intentionally use a mock mesh and never materialize.
            return _HostMeshMapper(dims=dims, mesh_shape=tuple(mesh_shape))
        raise


def _normalize_slots(slot_ids, capacity: int) -> tuple[int, ...]:
    if slot_ids is None:
        return tuple(range(capacity))
    if isinstance(slot_ids, torch.Tensor):
        slot_ids = slot_ids.reshape(-1).tolist()
    return tuple(int(slot) for slot in slot_ids)


def _broadcast(value, size: int, name: str) -> list:
    if isinstance(value, torch.Tensor):
        value = value.reshape(-1).tolist()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        values = list(value)
        if len(values) != size:
            raise ValueError(f"{name} has {len(values)} values for {size} active slots")
        return values
    return [value] * size


def _optional_int(value, name: str) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} values must be integers or None") from error


def _seed_digest(seed: int, slot: int, *, bits: int) -> int:
    payload = f"sampling2d:{seed}:{slot}".encode("ascii")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little") & ((1 << bits) - 1)


def _device_seed(seed: int, slot: int) -> int:
    return _seed_digest(seed, slot, bits=31)


def _host_seed(seed: int, slot: int) -> int:
    return _seed_digest(seed, slot, bits=63)


def _update_lazy(buffer, source: torch.Tensor) -> None:
    if not isinstance(buffer, LazyBuffer):
        raise TypeError("per-call sampling state must use LazyBuffer")
    buffer.update(source)


def _materialize(buffer):
    if isinstance(buffer, ttnn.Tensor):
        return buffer
    return buffer.get_device_buffer()
