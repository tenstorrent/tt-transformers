# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""One-token fused MoE decode: router matmul -> ExpertGateUp -> ExpertDown, three programs per layer.

The routed expert tensors are the ones a model already holds for prefill ([1, E, H, 2 I] packed gate|up and [1, E, I, H]
down, DRAM interleaved, bfp8 or bfp4), stored rounded by `round_sig`. Built here from the HF weights: the router weight
with the shared-expert gate as column E ([H, E + 32] BF16) and the shared expert in the routed layout. The result is
this device's partial sum; the layer reduces it across devices.

Numerics: custom_mm multiplies the weight operand at 5 significant bits and the activation at 7 (LoFi), so weights are
stored rounded to 5 bits and the gate|up activations are rounded to 7 before the down projection; FP32 accumulation.

In the source bring-up (Ornith-1.5-35B-A3B, Blackhole, traced) these three programs replaced the 29-op ttnn MoE decode:
177 -> 38 us per layer.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import NamedTuple

import torch
import ttnn

from ._program import TILE
from .expert_down import ExpertDown
from .expert_gate_up import ExpertGateUp, ep_offsets

WEIGHT_SIG_BITS = 5  # significant bits custom_mm's LoFi multiply reads from the weight operand
CACHE_SUFFIX = f".r{WEIGHT_SIG_BITS}"  # for weight-cache names of rounded tensors


def round_sig(t: torch.Tensor, bits: int = WEIGHT_SIG_BITS) -> torch.Tensor:
    """Round to nearest (half away from zero) with `bits` significant bits, on the FP32 bit pattern."""
    drop = 24 - bits  # FP32 has 24 significant bits
    i = t.float().contiguous().view(torch.int32)
    return ((i + (1 << (drop - 1))) & ~((1 << drop) - 1)).view(torch.float32)


class SharedExpert(NamedTuple):
    """HF shared-expert weights: expert_gate [1, H] (the sigmoid of its logit scales the expert), gate_proj / up_proj
    [S, H], down_proj [H, S], with S = inter * devices (split across the devices like the routed experts)."""

    expert_gate: torch.Tensor
    gate_proj: torch.Tensor
    up_proj: torch.Tensor
    down_proj: torch.Tensor


def _router_program_config(mesh_device, hidden: int, width: int):
    """1D (mcast_in0) decode matmul, one output tile column per core: a small grid beats the DRAM-sharded one on this
    bandwidth-bound skinny matmul (Ornith-1.5-35B-A3B: 24.7 -> 7.5 us)."""
    k_tiles, n_tiles = hidden // TILE, width // TILE
    cols = min(mesh_device.compute_with_storage_grid_size().x, n_tiles)
    rows = math.ceil(n_tiles / cols)
    per_core_n = math.ceil(n_tiles / (cols * rows))
    in0_block_w = 16 if k_tiles % 16 == 0 else max(d for d in range(1, 9) if k_tiles % d == 0)
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(cols, rows),
        in0_block_w=in0_block_w,
        out_subblock_h=1,
        out_subblock_w=max(i for i in range(1, 5) if per_core_n % i == 0),
        per_core_M=1,
        per_core_N=per_core_n,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    )


class FusedMoEDecode:
    def __init__(
        self,
        mesh_device,
        *,
        hidden: int,
        inter: int,
        num_experts: int,
        k: int,
        weight_dtype,
        router_weight: torch.Tensor,
        down_dtype=None,
        shared_expert: SharedExpert | None = None,
        ep_experts: int | None = None,
        gate_first: bool = False,
        router_compute_config=None,
        cache: dict | None = None,
        cache_name: Callable[[str], str | None] | None = None,
    ):
        """One instance per layer. router_weight: HF [E, H]. inter: the expert width each device holds (the full width
        under expert parallelism, where each device holds ep_experts whole experts). down_dtype: when the down weights
        differ from gate|up. router_compute_config: the router matmul's (default HiFi4, FP32 accumulation).
        cache: a model-owned dict, so the layers share one set of ops and L1 buffers. cache_name(name): the weight
        cache path for a built tensor, or None."""
        down_dtype = down_dtype or weight_dtype
        devices = mesh_device.get_num_devices()
        H, E = hidden, num_experts
        self.mesh_device = mesh_device
        self.shared = shared_expert is not None
        name = cache_name or (lambda _: None)

        def upload(t, dtype, label, mapper):
            return ttnn.as_tensor(
                t.contiguous(),
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                cache_file_name=name(label),
                **({"mesh_mapper": mapper} if mapper is not None else {}),
            )

        def shard(dim):
            return ttnn.ShardTensorToMesh(mesh_device, dim=dim) if devices > 1 else None

        rw = torch.zeros(H, E + TILE)  # column E: the shared-expert gate logit (zero without one)
        rw[:, :E] = router_weight.T.float()
        if self.shared:
            rw[:, E] = shared_expert.expert_gate.reshape(-1).float()
        self.router_weight = upload(
            rw, ttnn.bfloat16, "fused_decode.router", ttnn.ReplicateTensorToMesh(mesh_device) if devices > 1 else None
        )
        self.router_program_config = _router_program_config(mesh_device, H, E + TILE)
        self.router_compute_config = router_compute_config or ttnn.init_device_compute_kernel_config(
            mesh_device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=False
        )
        if self.shared:
            # the routed layout, per device shard: [1, 1, H, 2 I] packed like the routed experts, [1, 1, I, H] down
            g, u = shared_expert.gate_proj.T.float(), shared_expert.up_proj.T.float()  # [H, S]
            first, second = (g, u) if gate_first else (u, g)
            packed = torch.cat(
                [
                    torch.cat([first[:, d * inter : (d + 1) * inter], second[:, d * inter : (d + 1) * inter]], -1)
                    for d in range(devices)
                ],
                -1,
            )
            self.shared_up_gate = upload(
                round_sig(packed).reshape(1, 1, H, -1),
                weight_dtype,
                "fused_decode.shared_up_gate" + CACHE_SUFFIX,
                shard(-1),
            )
            self.shared_down = upload(
                round_sig(shared_expert.down_proj.T.float()).reshape(1, 1, -1, H),
                down_dtype,
                "fused_decode.shared_down" + CACHE_SUFFIX,
                shard(-2),
            )

        cache = {} if cache is None else cache
        key = ("fused_moe_decode", H, inter, E, k, weight_dtype, down_dtype, self.shared, ep_experts, gate_first)
        if key not in cache:
            ep = dict(shared=self.shared, ep_experts=ep_experts)
            gate_up = ExpertGateUp(mesh_device, H, inter, E, k, weight_dtype, gate_first=gate_first, **ep)
            down = ExpertDown(mesh_device, H, inter, E, k, down_dtype, **ep)
            cache[key] = (
                gate_up,
                down,
                *gate_up.allocate_outputs(),
                down.allocate_output(),
                ep_offsets(mesh_device, ep_experts) if ep_experts else None,
            )
        self.gate_up, self.down, self.act, self.route, self.out, self.ep_offset = cache[key]

    def logits(self, x):
        """Router logits [1, 1, 1, num_experts + 32] BF16 in L1 for the normed hidden x."""
        return ttnn.linear(
            x,
            self.router_weight,
            compute_kernel_config=self.router_compute_config,
            program_config=self.router_program_config,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

    def __call__(self, x, up_gate, down):
        """x: the normed hidden [1, 1, 1, H] BF16 TILE in L1. up_gate, down: this layer's routed expert tensors. Returns
        this device's partial sum (routed + shared) in a model-wide [1, 1, 1, H] buffer: reduce it before the next
        layer's MoE runs."""
        logits = self.logits(x)
        self.gate_up(
            x,
            logits,
            up_gate,
            self.act,
            self.route,
            shared_up_gate=self.shared_up_gate if self.shared else None,
            ep_offset=self.ep_offset,
        )
        ttnn.deallocate(logits)
        return self.down(self.act, self.route, down, self.out, shared_down=self.shared_down if self.shared else None)
