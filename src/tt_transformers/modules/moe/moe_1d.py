# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Reusable mixture-of-experts module for 1D Blackhole meshes."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import torch
import ttnn

from tt_transformers.modules.lazy_weight import LazyWeight, resolve_lazy_weight
from tt_transformers.modules.lightweightmodule import LightweightModule
from tt_transformers.modules.mode import Mode, normalize_mode
from tt_transformers.modules.tt_ccl import (
    CCL_CHUNKS_PER_SYNC,
    CCL_NUM_BUFFERS_PER_CHANNEL,
    CCL_NUM_WORKERS_PER_LINK,
    TT_CCL,
    default_topology,
    get_tt_ccl,
)
from tt_transformers.ops.moe.fused_decode import FusedMoEDecode, SharedExpert, round_sig, router_weight
from tt_transformers.ops.moe.grouped_matmul import GroupedExpertMatmul
from tt_transformers.ops.moe.sorted_dispatch import GROUP, dispatch_rows, gather_add_rows, gather_rows, sorted_dispatch

MoEParallelism = Literal["tensor", "expert"]
TILE = 32

_fused_decode_cache: dict[tuple, FusedMoEDecode] = {}


def get_fused_decode(mesh_device: ttnn.MeshDevice, *args, **kwargs) -> FusedMoEDecode:
    """Get or create the FusedMoEDecode for mesh_device and these arguments, shared by every layer that matches."""
    key = (mesh_device.id(), args, tuple(sorted(kwargs.items())))
    if key not in _fused_decode_cache:
        _fused_decode_cache[key] = FusedMoEDecode(mesh_device, *args, **kwargs)
    return _fused_decode_cache[key]


def clear_fused_decode_cache():
    """Clear cache (for testing)."""
    _fused_decode_cache.clear()


@dataclass(frozen=True)
class MoE1DWeights:
    router: LazyWeight
    up_gate: LazyWeight
    down: LazyWeight
    shared_up_gate: LazyWeight | None = None
    shared_down: LazyWeight | None = None


def _cache(cache_dir: Path | None, prefix: str, name: str) -> tuple[Path, str] | None:
    return (cache_dir, f"{prefix}_{name}") if cache_dir is not None else None


def _pack_experts(
    gate_proj: torch.Tensor,
    up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    *,
    parallelism: MoEParallelism,
    num_devices: int,
    gate_first: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack HF expert tensors as [1, E, H, 2I] and [1, E, I, H]."""
    if gate_proj.shape != up_proj.shape or gate_proj.ndim != 3:
        raise ValueError("gate_proj and up_proj must have matching [E, I, H] shapes")
    experts, intermediate, hidden = gate_proj.shape
    if down_proj.shape != (experts, hidden, intermediate):
        raise ValueError("down_proj must have shape [E, H, I]")
    if parallelism not in ("tensor", "expert"):
        raise ValueError(f"unsupported MoE parallelism: {parallelism}")
    if num_devices <= 0:
        raise ValueError("num_devices must be positive")

    gate = gate_proj.transpose(1, 2).float()
    up = up_proj.transpose(1, 2).float()
    first, second = (gate, up) if gate_first else (up, gate)
    if parallelism == "tensor":
        if intermediate % num_devices:
            raise ValueError("expert intermediate size must divide evenly across devices")
        local_intermediate = intermediate // num_devices
        shards = [slice(i * local_intermediate, (i + 1) * local_intermediate) for i in range(num_devices)]
        packed = torch.cat([torch.cat([first[..., shard], second[..., shard]], -1) for shard in shards], -1)
    else:
        if experts % num_devices:
            raise ValueError("number of experts must divide evenly across devices")
        packed = torch.cat([first, second], -1)

    return round_sig(packed).unsqueeze(0), round_sig(down_proj.transpose(1, 2)).unsqueeze(0)


def _largest_divisor_at_most(value: int, limit: int) -> int:
    divisor = max(1, min(limit, value))
    while divisor > 1 and value % divisor:
        divisor -= 1
    return divisor


def _sparse_matmul_program_config(
    m: int,
    n: int,
    k: int,
    *,
    target_cores: int,
    in0_block_w_cap: int,
) -> ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig:
    """Create the 1D multicast config used by sparse expert matmuls."""
    n_tiles = max(1, math.ceil(n / TILE))
    k_tiles = max(1, math.ceil(k / TILE))
    m_tiles = max(1, math.ceil(m / TILE))
    cores = _largest_divisor_at_most(n_tiles, max(1, target_cores))
    grid_y = _largest_divisor_at_most(cores, 8)
    grid_x = cores // grid_y
    per_core_n = n_tiles // cores
    subblock_w = _largest_divisor_at_most(per_core_n, 8)
    subblock_h = _largest_divisor_at_most(m_tiles, max(1, 8 // subblock_w))
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(grid_x, grid_y),
        in0_block_w=_largest_divisor_at_most(k_tiles, max(1, in0_block_w_cap)),
        out_subblock_h=subblock_h,
        out_subblock_w=subblock_w,
        out_block_h=m_tiles,
        out_block_w=per_core_n,
        per_core_M=m_tiles,
        per_core_N=per_core_n,
        fuse_batch=False,
        fused_activation=None,
        mcast_in0=True,
    )


def moe_weights_from_hf(
    router: torch.Tensor,
    gate_proj: torch.Tensor,
    up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    *,
    num_devices: int,
    parallelism: MoEParallelism,
    weight_dtype: ttnn.DataType,
    down_dtype: ttnn.DataType | None = None,
    shared_expert: SharedExpert | None = None,
    gate_first: bool = False,
    cache_dir: Path | None = None,
    cache_prefix: str = "moe",
) -> MoE1DWeights:
    """Create cached LazyWeights from HF router and expert tensors."""
    if router.ndim != 2 or router.shape != (gate_proj.shape[0], gate_proj.shape[2]):
        raise ValueError("router must have shape [E, H]")
    if shared_expert is not None and parallelism != "tensor":
        raise ValueError("shared experts require tensor parallelism")

    up_gate, down = _pack_experts(
        gate_proj,
        up_proj,
        down_proj,
        parallelism=parallelism,
        num_devices=num_devices,
        gate_first=gate_first,
    )
    router_source = router_weight(router, shared_expert.expert_gate if shared_expert is not None else None)

    def lazy(source, dtype, name):
        return LazyWeight(
            source=source,
            dtype=dtype,
            cache_dir_weight_name=_cache(cache_dir, cache_prefix, name),
        )

    shared_up_gate = shared_down = None
    if shared_expert is not None:
        shared_gate = shared_expert.gate_proj.unsqueeze(0)
        shared_up = shared_expert.up_proj.unsqueeze(0)
        shared_dn = shared_expert.down_proj.unsqueeze(0)
        shared_up_source, shared_down_source = _pack_experts(
            shared_gate,
            shared_up,
            shared_dn,
            parallelism="tensor",
            num_devices=num_devices,
            gate_first=gate_first,
        )
        shared_up_gate = lazy(shared_up_source, weight_dtype, "shared_up_gate")
        shared_down = lazy(shared_down_source, down_dtype or weight_dtype, "shared_down")

    return MoE1DWeights(
        router=lazy(router_source, ttnn.bfloat16, "router"),
        up_gate=lazy(up_gate, weight_dtype, "up_gate"),
        down=lazy(down, down_dtype or weight_dtype, "down"),
        shared_up_gate=shared_up_gate,
        shared_down=shared_down,
    )


@dataclass
class MoE1DConfig:
    router: LazyWeight
    up_gate: LazyWeight
    down: LazyWeight
    k: int
    parallelism: MoEParallelism
    shared_up_gate: LazyWeight | None = None
    shared_down: LazyWeight | None = None
    expert_ids: LazyWeight | None = None
    mesh_device: ttnn.MeshDevice | None = None
    tt_ccl: TT_CCL | None = None
    topology: ttnn.Topology | None = None
    fused_decode: FusedMoEDecode | None = None
    hidden: int | None = None
    intermediate: int | None = None
    num_experts: int | None = None
    experts_per_device: int | None = None
    gate_first: bool = False
    num_reduce_scatter_links: int = 1
    prefill_num_reduce_scatter_links: int | None = None
    decode_input_memcfg: ttnn.MemoryConfig = ttnn.L1_MEMORY_CONFIG
    decode_output_memcfg: ttnn.MemoryConfig = ttnn.L1_MEMORY_CONFIG
    prefill_input_memcfg: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG
    prefill_output_memcfg: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG
    decode_reduce_intermediate_memcfg: ttnn.MemoryConfig = ttnn.L1_MEMORY_CONFIG
    prefill_reduce_intermediate_memcfg: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG
    prefill_chunk_size: int = TILE
    prefill_gate_up_program_config: ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig | None = None
    prefill_down_program_config: ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig | None = None
    prefill_compute_kernel_config: ttnn.DeviceComputeKernelConfig | None = None
    prefill_activation_dtype: ttnn.DataType = ttnn.bfloat16
    prefill_output_tile: ttnn.Tile | None = None
    decode_reduce_chunks_per_sync: int = 1
    decode_reduce_num_workers_per_link: int = 1
    # expert-sorted routed prefill from this many tokens (None: always the chunked sparse prefill), in slices of at
    # most prefill_sorted_max_tokens, with these K-blocks for the grouped gate|up and down matmuls
    prefill_sorted_min_tokens: int | None = None
    prefill_sorted_max_tokens: int = 4096
    prefill_sorted_gate_up_in0_block_w: int = 16
    prefill_sorted_down_in0_block_w: int = 4


class MoE1D(LightweightModule):
    """Mixture-of-experts layer with fused decode, and chunked sparse or expert-sorted prefill."""

    def __init__(
        self,
        router: LazyWeight,
        up_gate: LazyWeight,
        down: LazyWeight,
        k: int,
        parallelism: MoEParallelism,
        *,
        shared_up_gate: LazyWeight | None = None,
        shared_down: LazyWeight | None = None,
        gate_first: bool = False,
    ):
        super().__init__()
        self._initialize(
            MoE1DConfig(
                router=router,
                up_gate=up_gate,
                down=down,
                k=k,
                parallelism=parallelism,
                shared_up_gate=shared_up_gate,
                shared_down=shared_down,
                gate_first=gate_first,
            )
        )

    @classmethod
    def from_config(cls, config: MoE1DConfig) -> MoE1D:
        if not isinstance(config, MoE1DConfig):
            raise TypeError("MoE1D.from_config expects MoE1DConfig")
        instance = object.__new__(cls)
        super(MoE1D, instance).__init__()
        instance._initialize(config)
        return instance

    def _initialize(self, config: MoE1DConfig) -> None:
        self.config = resolve_moe1d_config(config)
        cfg = self.config
        self._decode_op = cfg.fused_decode or get_fused_decode(
            cfg.mesh_device,
            cfg.hidden,
            cfg.intermediate,
            cfg.num_experts,
            cfg.k,
            cfg.up_gate.dtype,
            down_dtype=cfg.down.dtype,
            shared=cfg.shared_up_gate is not None,
            ep_experts=cfg.experts_per_device if cfg.parallelism == "expert" else None,
            gate_first=cfg.gate_first,
        )
        self._device_weights_loaded = False
        self._decode_impl = self._decode_shared if cfg.shared_up_gate is not None else self._decode_routed
        self._prefill_impl = self._prefill_shared if cfg.shared_up_gate is not None else self._prefill_routed
        self._activate_gate_up = self._activate_gate_first if cfg.gate_first else self._activate_up_first
        self._routing_matrix = (
            self._routing_matrix_tensor_parallel
            if cfg.parallelism == "tensor"
            else self._routing_matrix_expert_parallel
        )
        self._local_experts = cfg.num_experts if cfg.parallelism == "tensor" else cfg.experts_per_device
        if cfg.prefill_sorted_min_tokens is not None:
            self._sorted_gate_up, self._sorted_down = _grouped_matmuls(cfg, self._local_experts)
        if cfg.mesh_device.get_num_devices() == 1:
            self._reduce_decode = self._clone_decode
            self._reduce_prefill = self._clone_prefill
        else:
            self._reduce_decode = self._reduce_decode_multi_device
            self._reduce_prefill = self._reduce_prefill_multi_device

    def load_device_weights(self) -> None:
        if self._device_weights_loaded:
            return
        self.router = self.config.router.get_device_weight()
        self.up_gate = self.config.up_gate.get_device_weight()
        self.down = self.config.down.get_device_weight()
        self.expert_ids = self.config.expert_ids.get_device_weight()
        if self.config.shared_up_gate is not None:
            self.shared_up_gate = self.config.shared_up_gate.get_device_weight()
            self.shared_down = self.config.shared_down.get_device_weight()
        if self.config.prefill_sorted_min_tokens is not None and self.config.parallelism == "expert":
            self._expert_offset = _expert_offsets(self.config.mesh_device, self._local_experts)
        self._device_weights_loaded = True

    def decode_forward(self, x: ttnn.Tensor | LazyWeight) -> ttnn.Tensor:
        """x [1, 1, users, H]. One user runs the fused decode; the fused op computes a single token, so more users run
        the routed sparse path over every row."""
        self.load_device_weights()
        x = _load_input(x, self.config, self.config.decode_input_memcfg)
        if x.shape[-2] == 1:
            return self._reduce_decode(self._decode_impl(x))
        rows = ttnn.to_memory_config(x, self.config.prefill_input_memcfg)
        partial = self._prefill_impl(rows)
        _deallocate_if_distinct(rows, x)
        output = self._reduce_decode(partial)
        ttnn.deallocate(partial)
        return output

    def prefill_forward(self, x: ttnn.Tensor | LazyWeight) -> ttnn.Tensor:
        self.load_device_weights()
        x = _load_input(x, self.config, self.config.prefill_input_memcfg)
        return self._reduce_prefill(self._prefill_impl(x))

    def forward(self, x: ttnn.Tensor | LazyWeight, mode: str | Mode) -> ttnn.Tensor:
        return self.decode_forward(x) if normalize_mode(mode) == "decode" else self.prefill_forward(x)

    def _decode_routed(self, x: ttnn.Tensor) -> ttnn.Tensor:
        return self._decode_op(x, self.router, self.up_gate, self.down)

    def _decode_shared(self, x: ttnn.Tensor) -> ttnn.Tensor:
        return self._decode_op(
            x,
            self.router,
            self.up_gate,
            self.down,
            shared_up_gate=self.shared_up_gate,
            shared_down=self.shared_down,
        )

    def _use_sorted(self, x: ttnn.Tensor) -> bool:
        minimum = self.config.prefill_sorted_min_tokens
        return minimum is not None and x.shape[-2] >= minimum

    def _prefill_routed(self, x: ttnn.Tensor) -> ttnn.Tensor:
        if self._use_sorted(x):
            return self._sorted_prefill(x)
        logits = ttnn.linear(x, self.router, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        routed = self._routed_prefill(x, logits)
        ttnn.deallocate(logits)
        return routed

    def _prefill_shared(self, x: ttnn.Tensor) -> ttnn.Tensor:
        cfg = self.config
        logits = ttnn.linear(x, self.router, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        routed = self._sorted_prefill(x) if self._use_sorted(x) else self._routed_prefill(x, logits)
        gate_logit = ttnn.slice(logits, (0, 0, 0, cfg.num_experts), (1, 1, x.shape[-2], cfg.num_experts + 1))
        gate = ttnn.sigmoid(gate_logit)
        expert = self._dense_expert_prefill(x, self.shared_up_gate, self.shared_down)
        shared = ttnn.multiply(expert, gate, memory_config=cfg.prefill_output_memcfg)
        output = ttnn.add(routed, shared, memory_config=cfg.prefill_output_memcfg)
        for tensor in (logits, routed, gate_logit, gate, expert, shared):
            ttnn.deallocate(tensor)
        return output

    def _routed_prefill(self, x: ttnn.Tensor, logits: ttnn.Tensor) -> ttnn.Tensor:
        cfg = self.config
        expert_logits = ttnn.slice(logits, (0, 0, 0, 0), (1, 1, x.shape[-2], cfg.num_experts))
        values, indices = ttnn.topk(expert_logits, k=cfg.k, dim=-1)
        weights = ttnn.softmax(values, dim=-1)
        routing = self._routing_matrix(weights, indices)
        output = self._sparse_expert_prefill(x, routing)
        for tensor in (expert_logits, values, indices, weights, routing):
            ttnn.deallocate(tensor)
        return output

    def _sorted_prefill(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """Routed experts over the tokens in slices of at most prefill_sorted_max_tokens."""
        cfg = self.config
        tokens, step = x.shape[-2], cfg.prefill_sorted_max_tokens
        if tokens <= step:
            return self._sorted_slice(x)
        pieces = []
        for start in range(0, tokens, step):
            end = min(tokens, start + step)
            piece = ttnn.slice(x, (0, 0, start, 0), (1, 1, end, cfg.hidden), memory_config=ttnn.DRAM_MEMORY_CONFIG)
            pieces.append(self._sorted_slice(piece))
            _deallocate_if_distinct(piece, x)
        output = ttnn.concat(pieces, dim=2, memory_config=cfg.prefill_output_memcfg)
        for piece in pieces:
            ttnn.deallocate(piece)
        return output

    def _sorted_slice(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """Dispatch the routed (token, slot) pairs expert by expert, run each 32-row group through its expert, and
        sum each token's weighted expert outputs back: this device's routed partial [1, 1, T, H]."""
        cfg, local = self.config, self._local_experts
        tokens, groups_rows = x.shape[-2], dispatch_rows(x.shape[-2], cfg.k, local)
        ids, scores = self._sorted_routing(x)
        src_tok, weights, group_ids, undispatch = sorted_dispatch(ids, scores, local, rows=groups_rows)
        groups = group_ids.shape[-1]
        for tensor in (ids, scores):
            ttnn.deallocate(tensor)

        rows = ttnn.to_layout(ttnn.reshape(x, (tokens, cfg.hidden)), ttnn.ROW_MAJOR_LAYOUT)
        gathered = gather_rows(rows, src_tok, group_ids, local)
        ttnn.deallocate(rows)
        ttnn.deallocate(src_tok)
        projected = self._sorted_gate_up(gathered, self.up_gate, group_ids)
        ttnn.deallocate(gathered)
        activated = self._activate_gate_up(projected)
        ttnn.deallocate(projected)
        row_weights = ttnn.to_layout(ttnn.reshape(weights, (1, groups, GROUP, 1)), ttnn.TILE_LAYOUT)
        ttnn.deallocate(weights)
        weighted = ttnn.multiply(activated, row_weights, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(activated)
        ttnn.deallocate(row_weights)
        expert_out = self._sorted_down(weighted, self.down, group_ids)
        ttnn.deallocate(weighted)
        ttnn.deallocate(group_ids)
        if not self._sorted_down.untilize_out:
            tiled, expert_out = expert_out, ttnn.to_layout(expert_out, ttnn.ROW_MAJOR_LAYOUT)
            ttnn.deallocate(tiled)
        output = gather_add_rows(
            ttnn.reshape(expert_out, (groups_rows, cfg.hidden)),
            undispatch,
            tokens,
            cfg.k,
            memory_config=cfg.prefill_output_memcfg,
        )
        ttnn.deallocate(expert_out)
        ttnn.deallocate(undispatch)
        return output

    def _sorted_routing(self, x: ttnn.Tensor) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """Top-k ids local to this device (others: local_experts, routed elsewhere) and their softmax weights."""
        cfg, tokens = self.config, x.shape[-2]
        logits = ttnn.linear(x, self.router, dtype=ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        expert_logits = ttnn.slice(logits, (0, 0, 0, 0), (1, 1, tokens, cfg.num_experts))
        ttnn.deallocate(logits)
        values, indices = ttnn.topk(expert_logits, k=cfg.k, dim=-1)
        ttnn.deallocate(expert_logits)
        probabilities = ttnn.softmax(values, dim=-1, numeric_stable=True)
        scores = ttnn.typecast(probabilities, ttnn.bfloat16)
        ttnn.deallocate(values)
        ttnn.deallocate(probabilities)
        if cfg.parallelism == "tensor":
            return indices, scores
        global_ids = ttnn.typecast(ttnn.typecast(indices, ttnn.uint32), ttnn.float32)
        ttnn.deallocate(indices)
        local = ttnn.sub(global_ids, self._expert_offset)
        ttnn.deallocate(global_ids)
        here = ttnn.logical_and(ttnn.ge(local, 0.0), ttnn.lt(local, float(self._local_experts)))
        kept = ttnn.where(here, local, float(self._local_experts))
        ids = ttnn.typecast(kept, ttnn.uint32)
        for tensor in (local, here, kept):
            ttnn.deallocate(tensor)
        return ids, scores

    def _routing_matrix_tensor_parallel(self, weights: ttnn.Tensor, indices: ttnn.Tensor) -> ttnn.Tensor:
        cfg = self.config
        # empty + fill stays on device; ttnn.zeros uploads from host, which a trace cannot capture
        empty = ttnn.empty(
            [1, 1, weights.shape[-2], cfg.num_experts],
            ttnn.bfloat16,
            ttnn.TILE_LAYOUT,
            cfg.mesh_device,
            ttnn.DRAM_MEMORY_CONFIG,
        )
        zeros = ttnn.fill(empty, 0.0, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(empty)
        scatter_indices = ttnn.typecast(indices, ttnn.uint32) if indices.dtype != ttnn.uint32 else indices
        routing = ttnn.scatter(zeros, dim=-1, index=scatter_indices, src=weights)
        ttnn.deallocate(zeros)
        if scatter_indices is not indices:
            ttnn.deallocate(scatter_indices)
        return routing

    def _routing_matrix_expert_parallel(self, weights: ttnn.Tensor, indices: ttnn.Tensor) -> ttnn.Tensor:
        selected = ttnn.eq(self.expert_ids, indices)
        selected_bf16 = ttnn.typecast(selected, ttnn.bfloat16)
        ttnn.deallocate(selected)
        weighted = ttnn.multiply(selected_bf16, weights)
        ttnn.deallocate(selected_bf16)
        weights_by_expert = ttnn.sum(weighted, dim=-1, keepdim=True)
        routing = ttnn.permute(weights_by_expert, (0, 3, 2, 1), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(weighted)
        ttnn.deallocate(weights_by_expert)
        return routing

    def _sparse_expert_prefill(self, x: ttnn.Tensor, routing: ttnn.Tensor) -> ttnn.Tensor:
        cfg = self.config
        local_experts = cfg.num_experts if cfg.parallelism == "tensor" else cfg.experts_per_device
        chunk = cfg.prefill_chunk_size
        tokens = x.shape[-2]
        padding = (-tokens) % chunk
        padded_x = x
        padded_routing = routing
        if padding:
            padded_x = ttnn.pad(x, [(0, 0), (0, 0), (0, padding), (0, 0)], value=0.0)
            padded_routing = ttnn.pad(routing, [(0, 0), (0, 0), (0, padding), (0, 0)], value=0.0)

        outputs = []
        padded_tokens = tokens + padding
        for start in range(0, padded_tokens, chunk):
            x_chunk = ttnn.slice(
                padded_x,
                (0, 0, start, 0),
                (1, 1, start + chunk, cfg.hidden),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            routing_chunk = ttnn.slice(
                padded_routing,
                (0, 0, start, 0),
                (1, 1, start + chunk, local_experts),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            mask_tiled = ttnn.sum(routing_chunk, dim=2, keepdim=True, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            mask = ttnn.to_layout(mask_tiled, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            _deallocate_if_distinct(mask_tiled, mask)
            projected = ttnn.sparse_matmul(
                x_chunk,
                self.up_gate,
                sparsity=mask,
                nnz=None,
                is_input_a_sparse=False,
                is_input_b_sparse=True,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                output_tile=cfg.prefill_output_tile,
                program_config=cfg.prefill_gate_up_program_config,
                compute_kernel_config=cfg.prefill_compute_kernel_config,
                dtype=cfg.prefill_activation_dtype,
            )
            _deallocate_if_distinct(x_chunk, padded_x)
            projected = ttnn.reshape(
                projected,
                (1, local_experts, chunk, 2 * cfg.intermediate),
            )
            activated = self._activate_gate_up(projected)
            ttnn.deallocate(projected)
            expert_output = ttnn.sparse_matmul(
                activated,
                self.down,
                sparsity=mask,
                nnz=None,
                is_input_a_sparse=True,
                is_input_b_sparse=True,
                memory_config=cfg.prefill_output_memcfg,
                output_tile=cfg.prefill_output_tile,
                program_config=cfg.prefill_down_program_config,
                compute_kernel_config=cfg.prefill_compute_kernel_config,
                dtype=cfg.prefill_activation_dtype,
            )
            ttnn.deallocate(activated)
            expert_output = ttnn.reshape(expert_output, (1, local_experts, chunk, cfg.hidden))
            routing_by_expert = ttnn.permute(
                routing_chunk,
                (0, 3, 2, 1),
                memory_config=cfg.prefill_output_memcfg,
            )
            weighted = ttnn.multiply(
                expert_output,
                routing_by_expert,
                memory_config=cfg.prefill_output_memcfg,
            )
            for tensor in (routing_by_expert, mask, expert_output):
                ttnn.deallocate(tensor)
            _deallocate_if_distinct(routing_chunk, padded_routing)
            outputs.append(ttnn.experimental.fast_reduce_nc(weighted, dims=[1]))
            ttnn.deallocate(weighted)

        _deallocate_if_distinct(padded_x, x)
        _deallocate_if_distinct(padded_routing, routing)
        if len(outputs) == 1:
            output = outputs[0]
        else:
            output = ttnn.concat(outputs, dim=2, memory_config=cfg.prefill_output_memcfg)
            for tensor in outputs:
                ttnn.deallocate(tensor)
        output = ttnn.reshape(output, (1, 1, padded_tokens, cfg.hidden))
        if padding:
            trimmed = ttnn.slice(
                output,
                (0, 0, 0, 0),
                (1, 1, tokens, cfg.hidden),
                memory_config=cfg.prefill_output_memcfg,
            )
            _deallocate_if_distinct(output, trimmed)
            output = trimmed
        return output

    def _dense_expert_prefill(self, x: ttnn.Tensor, up_gate: ttnn.Tensor, down: ttnn.Tensor) -> ttnn.Tensor:
        cfg = self.config
        projected = ttnn.matmul(x, up_gate, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        activated = self._activate_gate_up(projected)
        ttnn.deallocate(projected)
        output = ttnn.matmul(activated, down, memory_config=cfg.prefill_output_memcfg)
        ttnn.deallocate(activated)
        return output

    def _activate_gate_first(self, projected: ttnn.Tensor) -> ttnn.Tensor:
        gate, up = ttnn.split(projected, self.config.intermediate, dim=-1)
        gate = ttnn.silu(gate)
        activated = ttnn.multiply(gate, up, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(gate)
        ttnn.deallocate(up)
        return activated

    def _activate_up_first(self, projected: ttnn.Tensor) -> ttnn.Tensor:
        activated = ttnn.swiglu(projected, dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        shape = list(projected.shape)
        if activated.shape[-2] == shape[-2]:
            return activated
        # swiglu rounds the logical rows up to the tile
        trimmed = ttnn.slice(activated, [0] * len(shape), [*shape[:-1], self.config.intermediate])
        _deallocate_if_distinct(activated, trimmed)
        return trimmed

    def _clone_decode(self, output: ttnn.Tensor) -> ttnn.Tensor:
        return ttnn.clone(output, memory_config=self.config.decode_output_memcfg)

    def _clone_prefill(self, output: ttnn.Tensor) -> ttnn.Tensor:
        if output.memory_config() == self.config.prefill_output_memcfg:
            return output
        converted = ttnn.to_memory_config(output, self.config.prefill_output_memcfg)
        ttnn.deallocate(output)
        return converted

    def _reduce_decode_multi_device(self, output: ttnn.Tensor) -> ttnn.Tensor:
        cfg = self.config
        return ttnn.experimental.reduce_scatter_minimal_async(
            output,
            persistent_output_buffers=None,
            dim=3,
            multi_device_global_semaphore=cfg.tt_ccl.get_and_cycle_rs_semaphore_handles(),
            barrier_semaphore=cfg.tt_ccl.get_and_cycle_barrier_semaphore_handle(),
            num_links=cfg.num_reduce_scatter_links,
            memory_config=cfg.decode_output_memcfg,
            intermediate_memory_config=cfg.decode_reduce_intermediate_memcfg,
            topology=cfg.topology,
            chunks_per_sync=cfg.decode_reduce_chunks_per_sync,
            num_workers_per_link=cfg.decode_reduce_num_workers_per_link,
            num_buffers_per_channel=CCL_NUM_BUFFERS_PER_CHANNEL,
        )

    def _reduce_prefill_multi_device(self, output: ttnn.Tensor) -> ttnn.Tensor:
        cfg = self.config
        reduced = ttnn.experimental.reduce_scatter_minimal_async(
            output,
            persistent_output_buffers=None,
            dim=3,
            multi_device_global_semaphore=cfg.tt_ccl.get_and_cycle_rs_semaphore_handles(),
            barrier_semaphore=cfg.tt_ccl.get_and_cycle_barrier_semaphore_handle(),
            num_links=cfg.prefill_num_reduce_scatter_links,
            memory_config=cfg.prefill_output_memcfg,
            intermediate_memory_config=cfg.prefill_reduce_intermediate_memcfg,
            topology=cfg.topology,
            chunks_per_sync=CCL_CHUNKS_PER_SYNC,
            num_workers_per_link=CCL_NUM_WORKERS_PER_LINK,
            num_buffers_per_channel=CCL_NUM_BUFFERS_PER_CHANNEL,
        )
        ttnn.deallocate(output)
        return reduced


def _mesh_mapper(parallelism: MoEParallelism, tensor: str, num_devices: int) -> ttnn.MeshMapperConfig | None:
    if parallelism == "tensor":
        dim = {"up_gate": -1, "down": -2}.get(tensor)
    else:
        dim = {"up_gate": 1, "down": 1, "expert_ids": 1}.get(tensor)
    return (
        None
        if dim is None
        else ttnn.MeshMapperConfig(
            placements=[ttnn.PlacementShard(dim)],
            mesh_shape_override=ttnn.MeshShape([num_devices]),
        )
    )


def _resolve_mesh(config: MoE1DConfig) -> ttnn.MeshDevice:
    mesh_device = config.mesh_device or getattr(config.router, "device", None)
    if mesh_device is None:
        raise ValueError("MoE1D requires a mesh_device or a router associated with one")
    return mesh_device


def resolve_moe1d_config(config: MoE1DConfig) -> MoE1DConfig:
    """Resolve dimensions, distribution, collectives, and LazyWeights."""
    if not isinstance(config, MoE1DConfig):
        raise TypeError("resolve_moe1d_config expects MoE1DConfig")
    if config.parallelism not in ("tensor", "expert"):
        raise ValueError(f"unsupported MoE parallelism: {config.parallelism}")
    if (config.shared_up_gate is None) != (config.shared_down is None):
        raise ValueError("shared_up_gate and shared_down must be provided together")
    if config.parallelism == "expert" and config.shared_up_gate is not None:
        raise ValueError("shared experts require tensor parallelism")

    mesh_device = _resolve_mesh(config)
    if mesh_device.arch() != ttnn.device.Arch.BLACKHOLE:
        raise ValueError("MoE1D fused decode requires Blackhole")
    num_devices = mesh_device.get_num_devices()
    if config.parallelism == "tensor" and config.gate_first and num_devices > 1:
        raise ValueError("multi-device tensor parallelism requires [up | gate] packing")
    hidden = config.router.source.shape[-2] if config.hidden is None else config.hidden
    num_experts = config.router.source.shape[-1] - 32 if config.num_experts is None else config.num_experts
    if tuple(config.router.source.shape) != (hidden, num_experts + 32):
        raise ValueError("router must use the packed [H, E + 32] layout")
    if config.up_gate.source.shape[:3] != (1, num_experts, hidden):
        raise ValueError("up_gate must have shape [1, E, H, 2I]")
    if config.parallelism == "tensor":
        if config.up_gate.source.shape[-1] % (2 * num_devices):
            raise ValueError("tensor-parallel up_gate width must divide across devices")
        intermediate = (
            config.up_gate.source.shape[-1] // (2 * num_devices) if config.intermediate is None else config.intermediate
        )
        experts_per_device = None
    else:
        if num_experts % num_devices:
            raise ValueError("number of experts must divide across devices")
        intermediate = config.up_gate.source.shape[-1] // 2 if config.intermediate is None else config.intermediate
        experts_per_device = (
            num_experts // num_devices if config.experts_per_device is None else config.experts_per_device
        )
        if experts_per_device * num_devices != num_experts:
            raise ValueError("experts_per_device must cover all experts")
    if tuple(config.down.source.shape) != (
        1,
        num_experts,
        intermediate * num_devices if config.parallelism == "tensor" else intermediate,
        hidden,
    ):
        raise ValueError("down does not match the packed expert layout")
    if config.prefill_chunk_size <= 0 or config.prefill_chunk_size % TILE:
        raise ValueError("prefill_chunk_size must be a positive multiple of 32")
    if config.num_reduce_scatter_links <= 0:
        raise ValueError("num_reduce_scatter_links must be positive")
    prefill_num_reduce_scatter_links = (
        config.num_reduce_scatter_links
        if config.prefill_num_reduce_scatter_links is None
        else config.prefill_num_reduce_scatter_links
    )
    if prefill_num_reduce_scatter_links <= 0:
        raise ValueError("prefill_num_reduce_scatter_links must be positive")
    if config.prefill_sorted_min_tokens is not None and config.prefill_sorted_min_tokens <= 0:
        raise ValueError("prefill_sorted_min_tokens must be positive (or None to disable the sorted prefill)")
    if config.prefill_sorted_max_tokens <= 0 or config.prefill_sorted_max_tokens % TILE:
        raise ValueError("prefill_sorted_max_tokens must be a positive multiple of 32")

    local_experts = num_experts if config.parallelism == "tensor" else experts_per_device
    live_fraction = 1.0 - (1.0 - 1.0 / num_experts) ** (config.prefill_chunk_size * config.k)
    expected_live_experts = max(1, int(local_experts * live_fraction))
    prefill_gate_up_program_config = config.prefill_gate_up_program_config or _sparse_matmul_program_config(
        config.prefill_chunk_size,
        2 * intermediate,
        hidden,
        target_cores=max(8, min(32, expected_live_experts // 2)),
        in0_block_w_cap=64,
    )
    prefill_down_program_config = config.prefill_down_program_config or _sparse_matmul_program_config(
        config.prefill_chunk_size,
        hidden,
        intermediate,
        target_cores=max(8, min(32, expected_live_experts // 4)),
        in0_block_w_cap=32,
    )
    prefill_compute_kernel_config = config.prefill_compute_kernel_config or ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    topology = config.topology if config.topology is not None else default_topology(mesh_device)
    tt_ccl = config.tt_ccl or get_tt_ccl(mesh_device)
    common = dict(device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG, layout=ttnn.TILE_LAYOUT)
    router = resolve_lazy_weight(config.router, mesh_mapper_config=None, **common)
    up_gate = resolve_lazy_weight(
        config.up_gate,
        mesh_mapper_config=_mesh_mapper(config.parallelism, "up_gate", num_devices),
        **common,
    )
    down = resolve_lazy_weight(
        config.down,
        mesh_mapper_config=_mesh_mapper(config.parallelism, "down", num_devices),
        **common,
    )

    shared_up_gate = shared_down = None
    if config.shared_up_gate is not None:
        expected_up = (1, 1, hidden, 2 * intermediate * num_devices)
        expected_down = (1, 1, intermediate * num_devices, hidden)
        if tuple(config.shared_up_gate.source.shape) != expected_up:
            raise ValueError(f"shared_up_gate must have shape {expected_up}")
        if tuple(config.shared_down.source.shape) != expected_down:
            raise ValueError(f"shared_down must have shape {expected_down}")
        shared_up_gate = resolve_lazy_weight(
            config.shared_up_gate,
            mesh_mapper_config=_mesh_mapper("tensor", "up_gate", num_devices),
            **common,
        )
        shared_down = resolve_lazy_weight(
            config.shared_down,
            mesh_mapper_config=_mesh_mapper("tensor", "down", num_devices),
            **common,
        )

    ids = torch.arange(num_experts, dtype=torch.int32).reshape(1, num_experts, 1, 1)
    expert_ids = resolve_lazy_weight(
        LazyWeight(source=ids, dtype=ttnn.uint16),
        device=mesh_device,
        mesh_mapper_config=_mesh_mapper(config.parallelism, "expert_ids", num_devices),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )
    return replace(
        config,
        router=router,
        up_gate=up_gate,
        down=down,
        shared_up_gate=shared_up_gate,
        shared_down=shared_down,
        mesh_device=mesh_device,
        tt_ccl=tt_ccl,
        topology=topology,
        hidden=hidden,
        intermediate=intermediate,
        num_experts=num_experts,
        experts_per_device=experts_per_device,
        prefill_num_reduce_scatter_links=prefill_num_reduce_scatter_links,
        expert_ids=expert_ids,
        prefill_gate_up_program_config=prefill_gate_up_program_config,
        prefill_down_program_config=prefill_down_program_config,
        prefill_compute_kernel_config=prefill_compute_kernel_config,
        prefill_output_tile=config.prefill_output_tile or ttnn.Tile([TILE, TILE]),
    )


def _grouped_matmuls(cfg: MoE1DConfig, local_experts: int) -> tuple[GroupedExpertMatmul, GroupedExpertMatmul]:
    """The sorted prefill's gate|up and down matmuls; down writes ROW_MAJOR rows when its weights stay resident."""
    compute = cfg.prefill_compute_kernel_config
    gate_up = GroupedExpertMatmul(
        cfg.hidden,
        2 * cfg.intermediate,
        local_experts,
        cfg.up_gate.dtype,
        compute,
        in0_block_w=cfg.prefill_sorted_gate_up_in0_block_w,
    )
    down_args = (cfg.intermediate, cfg.hidden, local_experts, cfg.down.dtype, compute)
    down = GroupedExpertMatmul(*down_args, in0_block_w=cfg.prefill_sorted_down_in0_block_w)
    if down.resident:
        down = GroupedExpertMatmul(*down_args, in0_block_w=cfg.prefill_sorted_down_in0_block_w, untilize_out=True)
    return gate_up, down


def _expert_offsets(mesh_device: ttnn.MeshDevice, local_experts: int) -> ttnn.Tensor:
    """[1, 1, 1, 1] FP32 per device: the global id of its first expert."""
    num_devices = mesh_device.get_num_devices()
    offsets = torch.arange(num_devices, dtype=torch.float32).mul(local_experts).reshape(num_devices, 1, 1, 1)
    return ttnn.from_torch(
        offsets,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0) if num_devices > 1 else None,
    )


def _deallocate_if_distinct(tensor: ttnn.Tensor, keep: ttnn.Tensor) -> None:
    """Free tensor unless it shares keep's buffer (pad within the tile padding and full-range slices alias)."""
    if tensor.buffer_address() != keep.buffer_address():
        ttnn.deallocate(tensor)


def _load_input(x: ttnn.Tensor | LazyWeight, config: MoE1DConfig, memory_config: ttnn.MemoryConfig) -> ttnn.Tensor:
    if isinstance(x, LazyWeight):
        return resolve_lazy_weight(
            x,
            device=config.mesh_device,
            mesh_mapper_config=None,
            memory_config=memory_config,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
        ).get_device_weight()
    if not isinstance(x, ttnn.Tensor):
        raise TypeError("x must be a ttnn.Tensor or LazyWeight")
    if x.memory_config() != memory_config:
        raise ValueError("input tensor memory config does not match MoE1D config")
    return x
