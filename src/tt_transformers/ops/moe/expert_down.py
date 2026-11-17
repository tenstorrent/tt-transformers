# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Fused MoE decode, down stage: the down projection and weighted sum of the slots ExpertGateUp wrote, in one program.

Each core owns a few output columns of one DRAM bank and streams those columns of every live slot's down weights (read
in place from the ttnn [1, E, I, H] TILE expert tensors, DRAM interleaved), stacked along K, so one LoFi custom_mm over
[1, slots * I] x [slots * I, cols] gives the expert sum (the routing weights are already in the activations). The result
is this device's partial sum in row 0 of a [1, 1, 32, H] tile row; the layer's reduce across devices consumes it.
"""

from __future__ import annotations

import torch
import ttnn

from ._program import (
    TILE,
    TILE_BYTES,
    WEIGHT_CB_MAX_BYTES,
    bank_workers,
    cb,
    core_range_set,
    place_along_rows,
    reader_writer_compute,
    replicate,
)
from .expert_gate_up import MAX_K


class ExpertDown:
    def __init__(
        self,
        mesh_device,
        hidden: int,
        inter: int,
        num_experts: int,
        k: int,
        weight_dtype,
        *,
        shared: bool = True,
        ep_experts: int | None = None,
    ):
        """Matches the ExpertGateUp it follows: inter is the expert width this device holds, shared adds slot 0, and
        ep_experts (expert parallelism) reads the route record's live count of local experts."""
        if weight_dtype not in TILE_BYTES:
            raise ValueError(f"expert weights must be bfloat8_b or bfloat4_b, got {weight_dtype}")
        if not 1 <= k <= MAX_K:
            raise ValueError(f"k must be in [1, {MAX_K}], got {k}")
        if hidden % TILE or inter % TILE:
            raise ValueError(f"hidden ({hidden}) and inter ({inter}) must be multiples of {TILE}")
        slots, kt = (k + 1 if shared else k), inter // TILE
        if kt % 2 or slots * kt > 256:  # custom_mm: an even K of at most 256 tiles, for any live slot count
            raise ValueError(
                f"custom_mm needs an even inter / {TILE} and at most 256 stacked tiles, got {slots} x {kt}"
            )
        if ep_experts is not None and (shared or ep_experts <= 0 or num_experts % ep_experts):
            raise ValueError("expert parallelism needs shared=False and ep_experts dividing num_experts")
        self.banks = len(bank_workers(mesh_device))
        h_tiles = hidden // TILE
        if h_tiles % self.banks:
            raise ValueError(f"{h_tiles} hidden tiles do not split over {self.banks} DRAM banks")
        self.m = h_tiles // self.banks
        readers = max(2, -(-self.m // 4))  # at most 4 output columns per core: the half-sync FP32 DST
        if self.m % readers:
            raise ValueError(f"{self.m} columns per bank do not split over {readers} cores")
        self.cols, self.tile_bytes = self.m // readers, TILE_BYTES[weight_dtype]
        if slots * kt * self.cols * self.tile_bytes > WEIGHT_CB_MAX_BYTES:
            raise ValueError("the stacked down weights of a core do not fit in L1")

        self.mesh_device = mesh_device
        self.hidden, self.inter, self.num_experts, self.k = hidden, inter, num_experts, k
        self.shared, self.ep_experts = shared, ep_experts
        self.weight_dtype = weight_dtype
        self.kt, self.slots = kt, slots
        placed = place_along_rows(mesh_device, {b: readers for b in range(self.banks)})
        self.assign = [(c, b, r) for b in range(self.banks) for r, c in enumerate(placed[b])]
        self.cores = core_range_set([c for c, _, _ in self.assign])

    def allocate_output(self):
        """The partial-sum buffer [1, 1, 1, hidden] BF16 TILE, zeroed in L1 (only row 0 is ever written). Build it before
        any trace capture (host upload) and reuse it across layers and steps."""
        return ttnn.from_torch(
            torch.zeros(1, 1, 1, self.hidden, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            **replicate(self.mesh_device),
        )

    def __call__(self, act, route, down, out, *, shared_down=None):
        """act, route: as ExpertGateUp wrote them. down: the routed weights [1, E, inter, hidden] (E = ep_experts or
        num_experts), shared_down [1, 1, inter, hidden], both DRAM interleaved in the weight dtype. Writes this device's
        partial sum into out (`allocate_output`) and returns it."""
        self._check(act, route, down, out, shared_down)
        ws = shared_down if self.shared else down  # not read without a shared expert
        cb_in0, cb_w, cb_scr, cb_out = 0, 1, 3, 16
        kin = self.slots * self.kt
        cbs = [
            cb(self.cores, cb_in0, ttnn.bfloat16, 64, kin, tiny=True),
            cb(self.cores, cb_w, self.weight_dtype, self.tile_bytes, kin * self.cols),
            cb(self.cores, cb_scr, ttnn.uint16, 64, 1),
            cb(self.cores, cb_out, ttnn.bfloat16, 64, self.cols, tiny=True),
        ]
        ep_arg = [route.buffer_address()] if self.ep_experts else []
        rd_rt, wr_rt = {}, {}
        for c, b, r in self.assign:
            c0 = r * self.cols
            vc = (b + r) & 0x3  # spread the cores' reads over the NOC virtual channels
            rd_rt[(c.x, c.y)] = [down.buffer_address(), ws.buffer_address(), route.buffer_address(), b, c0, vc]
            wr_rt[(c.x, c.y)] = [act.buffer_address(), out.buffer_address(), b, c0] + ep_arg
        rd_ct = [cb_w, cb_scr, self.kt, self.m, self.cols, self.k, self.tile_bytes]
        rd_ct += ttnn.TensorAccessorArgs(route).get_compile_time_args()
        wr_ct = [cb_in0, cb_out, self.slots, 2 * self.inter, self.cols, self.banks]
        for t in (act, out) + ((route,) if self.ep_experts else ()):
            wr_ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        cp_ct = [cb_in0, cb_w, cb_out, kin, self.cols] + ([self.slots] if self.ep_experts else [])
        defines = [] if self.shared else [("NO_SHARED", "1")]
        if self.ep_experts:
            defines.append(("EP_EXPERTS", str(self.ep_experts)))
        kernels = reader_writer_compute(self.cores, "dn", (rd_ct, rd_rt), (wr_ct, wr_rt), (cp_ct, None), defines)
        ttnn.generic_op([down, ws, act, route, out], ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
        return out

    def _check(self, act, route, down, out, shared_down):
        def dram_weights(t, experts):
            return (
                t.dtype == self.weight_dtype
                and t.layout == ttnn.TILE_LAYOUT
                and list(t.padded_shape) == [1, experts, self.inter, self.hidden]
                and t.memory_config() == ttnn.DRAM_MEMORY_CONFIG
            )

        # the kernels address the weights by DRAM bank and write row 0 of each output tile, so anything else reads or
        # writes the wrong data without an error
        if not dram_weights(down, self.ep_experts or self.num_experts):
            raise ValueError("down must be the routed expert weights, TILE and DRAM interleaved in the weight dtype")
        if self.shared != (shared_down is not None) or (self.shared and not dram_weights(shared_down, 1)):
            raise ValueError("shared_down must be given exactly when the op has a shared expert, laid out as down")
        if list(act.shape) != [1, 1, self.k + 1, self.inter] or list(route.shape) != [1, 1, 1, 32]:
            raise ValueError("act and route must come from ExpertGateUp.allocate_outputs()")
        if (
            out.dtype != ttnn.bfloat16
            or out.layout != ttnn.TILE_LAYOUT
            or list(out.padded_shape) != [1, 1, TILE, self.hidden]
        ):
            raise ValueError(f"out must be BF16 TILE of padded shape [1, 1, 32, {self.hidden}] (allocate_output)")
