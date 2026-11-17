# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""One-token fused MoE decode: router matmul -> ExpertGateUp -> ExpertDown, three programs per layer.

The routed expert tensors are the ones a model already holds for prefill ([1, E, H, 2 I] packed gate|up and [1, E, I, H]
down, DRAM interleaved, bfp8 or bfp4), stored rounded by `round_sig`. The router weight carries the shared-expert gate
as column E ([H, E + 32] BF16, `router_weight`) and the shared expert uses the routed layout (`shared_expert_weights`).
The result is this device's partial sum; the layer reduces it across devices.

Numerics: custom_mm multiplies the weight operand at 5 significant bits and the activation at 7 (LoFi), so weights are
stored rounded to 5 bits and the gate|up activations are rounded to 7 before the down projection; FP32 accumulation.

In the source bring-up (Ornith-1.5-35B-A3B, Blackhole, traced) these three programs replaced the 29-op ttnn MoE decode:
177 -> 38 us per layer.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import torch
import ttnn

from ._program import TILE
from .expert_down import ExpertDown
from .expert_gate_up import ExpertGateUp, ep_offsets

WEIGHT_SIG_BITS = 5  # significant bits custom_mm's LoFi multiply reads from the weight operand


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


def router_weight(router: torch.Tensor, expert_gate: torch.Tensor | None = None) -> torch.Tensor:
    """HF router [E, H] (and the shared-expert gate [1, H]) -> [H, E + 32], column E the shared gate (zero without)."""
    E, H = router.shape
    rw = torch.zeros(H, E + TILE)
    rw[:, :E] = router.T.float()
    if expert_gate is not None:
        rw[:, E] = expert_gate.reshape(-1).float()
    return rw


def shared_expert_weights(shared: SharedExpert, inter: int, devices: int, gate_first: bool = False):
    """-> (up_gate [1, 1, H, 2 S], down [1, 1, S, H]), rounded: each device's inter-wide shard (up_gate split on the
    last dim, down on dim -2) is packed like the routed experts."""
    g, u = shared.gate_proj.T.float(), shared.up_proj.T.float()  # [H, S]
    first, second = (g, u) if gate_first else (u, g)
    shard = [slice(d * inter, (d + 1) * inter) for d in range(devices)]
    up_gate = torch.cat([torch.cat([first[:, s], second[:, s]], -1) for s in shard], -1)
    return round_sig(up_gate)[None, None], round_sig(shared.down_proj.T)[None, None]


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
        hidden: int,
        inter: int,
        num_experts: int,
        k: int,
        weight_dtype,
        *,
        down_dtype=None,
        shared: bool = True,
        ep_experts: int | None = None,
        gate_first: bool = False,
        router_compute_config=None,
    ):
        """One instance per model: the layers share its ops and L1 buffers and pass their weights per call. inter: the
        expert width each device holds (the full width under expert parallelism, where each device holds ep_experts
        whole experts). down_dtype: when the down weights differ from gate|up. router_compute_config: the router
        matmul's (default HiFi4, FP32 accumulation)."""
        self.mesh_device = mesh_device
        self.hidden, self.num_experts = hidden, num_experts
        ep = dict(shared=shared, ep_experts=ep_experts)
        self.gate_up = ExpertGateUp(
            mesh_device, hidden, inter, num_experts, k, weight_dtype, gate_first=gate_first, **ep
        )
        self.down = ExpertDown(mesh_device, hidden, inter, num_experts, k, down_dtype or weight_dtype, **ep)
        self.router_program_config = _router_program_config(mesh_device, hidden, num_experts + TILE)
        self.router_compute_config = router_compute_config or ttnn.init_device_compute_kernel_config(
            mesh_device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=False
        )
        self.act, self.route = self.gate_up.allocate_outputs()
        self.out = self.down.allocate_output()
        self.ep_offset = ep_offsets(mesh_device, ep_experts) if ep_experts else None

    def logits(self, x, router_weight):
        """Router logits [1, 1, 1, num_experts + 32] BF16 in L1 for the normed hidden x."""
        shape = [self.hidden, self.num_experts + TILE]
        if router_weight.dtype != ttnn.bfloat16 or list(router_weight.padded_shape)[-2:] != shape:
            raise ValueError(f"router_weight must be BF16 of shape {shape} (router_weight())")
        return ttnn.linear(
            x,
            router_weight,
            compute_kernel_config=self.router_compute_config,
            program_config=self.router_program_config,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

    def __call__(self, x, router_weight, up_gate, down, *, shared_up_gate=None, shared_down=None):
        """x: the normed hidden [1, 1, 1, H] BF16 TILE in L1. router_weight, up_gate, down, shared_up_gate, shared_down:
        this layer's weights on device (`router_weight`, `shared_expert_weights`). Returns this device's partial sum
        (routed + shared) in the model-wide [1, 1, 1, H] buffer: reduce it before the next layer's MoE runs."""
        logits = self.logits(x, router_weight)
        self.gate_up(x, logits, up_gate, self.act, self.route, shared_up_gate=shared_up_gate, ep_offset=self.ep_offset)
        ttnn.deallocate(logits)
        return self.down(self.act, self.route, down, self.out, shared_down=shared_down)
