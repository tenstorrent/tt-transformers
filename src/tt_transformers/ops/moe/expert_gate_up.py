# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Fused MoE decode, gate|up stage: routing and the routed (+ shared) expert SwiGLU for one token in one program.

Every core takes the softmax top-k and the shared-gate sigmoid from the router logits itself (kernels/router_topk.hpp),
then streams its expert columns and writes ``w_s * silu(x Wg_s) * (x Wu_s)`` into the activation buffer, one row per
slot (slot 0 = the shared expert, 1..k = the routed experts in top-k order), plus the route record the down stage
reads. Expert weights are read in place from the ttnn expert tensors ([1, E, H, 2 I] TILE, DRAM interleaved), with no
decode copy: when the 2 I / 32 column tiles are a multiple of the bank count, each expert's columns in a bank are one
contiguous [H / 32, m] run (gu_layout). Matmuls: custom_mm over 1x32 activation tiles, LoFi, FP32 accumulation; the
weights should be stored rounded to 5 significant bits, the width LoFi reads.

In the source bring-up (Ornith-1.5-35B-A3B, Blackhole, traced) the router matmul, this stage and the down stage replaced
the 29-op ttnn MoE decode (router chain, sparse matmuls, SwiGLU, weighting): 177 -> 38 us per layer.
"""

from __future__ import annotations

import torch
import ttnn

from ._program import (
    TILE,
    TILE_BYTES,
    bank_workers,
    cb,
    core_range_set,
    packet_bytes,
    place_along_rows,
    reader_writer_compute,
)

READERS = 4  # cores per column group
RING_BLOCKS = 5  # bank blocks in each core's weight ring (3 read ahead of compute)
RING_BYTES = 768 * 1024
RING_MAX_BYTES = 1024 * 1024  # the 3-block minimum may exceed RING_BYTES up to here, leaving the rest of L1 to the op
MAX_K = 15  # route record: ids at [0, k), the live count at 15, slot weights at [16, 17 + k)


def gu_layout(n_tiles: int, banks: int):
    """Gate|up column groups (kernels/gu_layout.hpp): m = n_tiles / banks columns per bank and expert; a group reads
    `nb` whole [kt, m] bank blocks per slot (bank q; banks q and q + banks / 2 when m is odd, where the two halves'
    columns sit in opposite banks) and owns the column pairs among them. -> (m, nb, groups, [[pair index] per group])."""
    m = n_tiles // banks
    nb = 2 if m % 2 else 1
    groups = banks // nb
    out = []
    for q in range(groups):
        cols = [q + h * (banks // 2) + banks * c for h in range(nb) for c in range(m)]
        out.append([c for c in cols if c < n_tiles // 2])
    assert sorted(p for g in out for p in g) == list(range(n_tiles // 2))
    return m, nb, groups, out


def ring_blocks(kt: int, m: int, tile_bytes: int) -> int:
    """Blocks in the weight ring: RING_BLOCKS, fewer (>= 3) when a [kt, m] block would push it past RING_BYTES."""
    block = kt * m * tile_bytes
    if 3 * block > RING_MAX_BYTES:
        raise ValueError(f"a 3-block weight ring of {3 * block} bytes does not fit in L1")
    return max(3, min(RING_BLOCKS, RING_BYTES // block))


def ep_offsets(mesh_device, experts_per_die: int):
    """Expert parallelism: [1, 1, 1, 32] UINT32 per die holding its first expert id (die d: d * experts_per_die), one
    DRAM page at the same address on every die, so the runtime args stay identical across the mesh."""
    n = mesh_device.get_num_devices()
    t = torch.zeros(n, 1, 1, 32, dtype=torch.int32)
    t[:, 0, 0, 0] = torch.arange(n, dtype=torch.int32) * experts_per_die
    return ttnn.from_torch(
        t,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )


def _replicate(mesh_device) -> dict:
    return {"mesh_mapper": ttnn.ReplicateTensorToMesh(mesh_device)} if mesh_device.get_num_devices() > 1 else {}


class ExpertGateUp:
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
        gate_first: bool = False,
    ):
        """inter: the expert width this device holds. shared: a shared expert takes slot 0. ep_experts: expert
        parallelism, the whole experts each die holds (num_experts stays the global count the router ranks).
        gate_first: weights packed [gate | up] instead of [up | gate]."""
        if weight_dtype not in TILE_BYTES:
            raise ValueError(f"expert weights must be bfloat8_b or bfloat4_b, got {weight_dtype}")
        if num_experts <= 0 or num_experts % 64 or num_experts >= 1 << 16:
            raise ValueError(f"num_experts must be a positive multiple of 64 below 65536, got {num_experts}")
        if not 1 <= k <= min(MAX_K, num_experts):
            raise ValueError(f"k must be in [1, {MAX_K}], got {k}")
        if hidden % TILE or inter % TILE:
            raise ValueError(f"hidden ({hidden}) and inter ({inter}) must be multiples of {TILE}")
        if not 2 <= hidden // TILE <= 256 or hidden // TILE % 2:
            raise ValueError(f"custom_mm needs an even hidden / {TILE} in [2, 256], got {hidden // TILE}")
        if ep_experts is not None and (shared or ep_experts <= 0 or num_experts % ep_experts):
            raise ValueError("expert parallelism needs shared=False and ep_experts dividing num_experts")
        self.banks = len(bank_workers(mesh_device))
        n_tiles = 2 * inter // TILE
        if n_tiles % self.banks:
            raise ValueError(f"{n_tiles} gate|up column tiles do not split over {self.banks} DRAM banks")
        self.m, self.nb, groups, self.group_pairs = gu_layout(n_tiles, self.banks)
        if self.m * self.nb > 8:
            raise ValueError(f"{self.m} gate|up columns per bank need more than the 8 FP32 DST tiles")

        self.mesh_device = mesh_device
        self.hidden, self.inter, self.num_experts, self.k = hidden, inter, num_experts, k
        self.shared, self.ep_experts, self.gate_first = shared, ep_experts, gate_first
        self.weight_dtype = weight_dtype
        self.kt, self.n_tiles = hidden // TILE, n_tiles
        self.tile_bytes = TILE_BYTES[weight_dtype]
        self.ring = ring_blocks(self.kt, self.m, self.tile_bytes)
        self.page_bytes = packet_bytes(self.kt * self.m, self.tile_bytes)
        self.dst_full = self.m * self.nb > 4  # FP32 DST: 4 tiles half-sync, 8 full-sync
        readers = min(READERS, k)
        # core r of each group streams slots r + 1, r + 1 + readers, ...; core 0 also takes the shared expert
        self.slots = [([0] if r == 0 and shared else []) + list(range(1 + r, k + 1, readers)) for r in range(readers)]
        placed = place_along_rows(mesh_device, {q: readers for q in range(groups)})
        self.assign = [(c, q, r) for q in range(groups) for r, c in enumerate(placed[q])]
        self.cores = core_range_set([c for c, _, _ in self.assign])

    def allocate_outputs(self):
        """(act [1, 1, k + 1, inter] BF16 row-major, route [1, 1, 1, 32] UINT16 row-major), zeroed in L1. Build them
        before any trace capture (host upload) and reuse them across layers and steps."""
        kw = dict(device=self.mesh_device, memory_config=ttnn.L1_MEMORY_CONFIG, **_replicate(self.mesh_device))
        act = ttnn.from_torch(
            torch.zeros(1, 1, self.k + 1, self.inter, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            **kw,
        )
        route = ttnn.from_torch(
            torch.zeros(1, 1, 1, 32, dtype=torch.int32), dtype=ttnn.uint16, layout=ttnn.ROW_MAJOR_LAYOUT, **kw
        )
        return act, route

    def __call__(self, x, logits, up_gate, act, route, *, shared_up_gate=None, ep_offset=None):
        """x: [1, 1, 32, hidden] BF16 TILE in L1 (row 0 the token). logits: [1, 1, 32, num_experts + 32] BF16 TILE in
        L1, column num_experts the shared-expert gate logit (zero without a shared expert). up_gate: the routed
        weights [1, E, hidden, 2 inter] (E = ep_experts or num_experts), shared_up_gate [1, 1, hidden, 2 inter], both
        DRAM interleaved in the weight dtype. ep_offset: from `ep_offsets`. Writes act and route (`allocate_outputs`).

        route: [k ids (local under expert parallelism) | 0... | count at 15 | shared weight | k routed weights | 0...]
        (BF16 weights). act row s: slot s's activations; under expert parallelism only rows 1..count are written."""
        self._check(x, logits, up_gate, act, route, shared_up_gate, ep_offset)
        ws = shared_up_gate if self.shared else up_gate  # not read without a shared expert
        cb_x, cb_w, cb_route, cb_scr, cb_part, cb_lg, cb_act = 0, 1, 2, 3, 4, 5, 16
        lg_bytes = 64 * (self.num_experts // TILE + 1)  # the gathered logits row (router_topk.hpp)
        cbs = [
            cb(self.cores, cb_x, ttnn.bfloat16, 64, self.kt, tiny=True),
            cb(self.cores, cb_w, self.weight_dtype, self.tile_bytes, self.ring * self.kt * self.m),
            cb(self.cores, cb_route, ttnn.uint16, 128, 1),
            cb(self.cores, cb_scr, ttnn.bfloat16, lg_bytes + (128 if self.ep_experts else 0), 1),
            cb(self.cores, cb_part, ttnn.uint32, 8 * self.k, 1),
            cb(self.cores, cb_lg, ttnn.bfloat16, lg_bytes, 1),
            cb(self.cores, cb_act, ttnn.bfloat16, 64, 2 * len(self.group_pairs[0]), tiny=True),
        ]
        ep_arg = [ep_offset.buffer_address()] if self.ep_experts else []
        rd_rt, wr_rt, cp_rt = {}, {}, {}
        for i, (c, q, r) in enumerate(self.assign):
            sl = self.slots[r]
            vc = (q + r) & 0x3  # spread the cores' reads over the NOC virtual channels
            rd_args = [logits.buffer_address(), up_gate.buffer_address(), ws.buffer_address(), q, vc, len(sl)]
            rd_rt[(c.x, c.y)] = rd_args + sl + ep_arg
            buffers = [x.buffer_address(), act.buffer_address(), route.buffer_address(), logits.buffer_address()]
            wr_rt[(c.x, c.y)] = buffers + [q, int(i == 0), len(sl)] + sl  # core 0 writes the route record
            cp_rt[(c.x, c.y)] = [len(sl)]
        rd_ct = [cb_w, cb_route, cb_scr, self.kt, self.n_tiles, self.num_experts, self.k, cb_part, self.tile_bytes]
        rd_ct += [self.page_bytes, self.banks, self.ring] + ttnn.TensorAccessorArgs(logits).get_compile_time_args()
        wr_ct = [cb_x, cb_act, cb_route, self.kt, self.k, self.num_experts, 2 * self.inter, cb_part, cb_lg, self.m]
        wr_ct += [self.banks]
        for t in (x, act, route, logits):
            wr_ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        cp_ct = [cb_x, cb_w, cb_act, self.kt, self.m, self.banks]
        defines = [("EP_EXPERTS", str(self.ep_experts))] if self.ep_experts else []
        if self.shared:  # keys the program cache: the shared expert only adds runtime args
            defines.append(("SHARED", "1"))
        if self.gate_first:
            defines.append(("GATE_FIRST", "1"))
        kernels = reader_writer_compute(
            self.cores, "gu", (rd_ct, rd_rt), (wr_ct, wr_rt), (cp_ct, cp_rt), defines, dst_full=self.dst_full
        )
        io = [up_gate, ws, x, logits, route, act] + ([ep_offset] if self.ep_experts else [])
        ttnn.generic_op(io, ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
        return act, route

    def _check(self, x, logits, up_gate, act, route, shared_up_gate, ep_offset):
        def l1_bf16_tile(t, width):
            return (
                t.dtype == ttnn.bfloat16
                and t.layout == ttnn.TILE_LAYOUT
                and list(t.padded_shape) == [1, 1, TILE, width]
                and t.memory_config().buffer_type == ttnn.BufferType.L1
            )

        def dram_weights(t, experts):
            return (
                t.dtype == self.weight_dtype
                and t.layout == ttnn.TILE_LAYOUT
                and list(t.padded_shape) == [1, experts, self.hidden, 2 * self.inter]
                and t.memory_config() == ttnn.DRAM_MEMORY_CONFIG
            )

        # the kernels read row 0 of x and logits in 32-byte pieces into 32-byte L1 offsets, and address the weights
        # by DRAM bank, so anything else reads the wrong data without an error
        if not l1_bf16_tile(x, self.hidden):
            raise ValueError(f"x must be BF16 TILE in L1 of padded shape [1, 1, 32, {self.hidden}]")
        if not l1_bf16_tile(logits, self.num_experts + TILE):
            raise ValueError(f"logits must be BF16 TILE in L1 of padded shape [1, 1, 32, {self.num_experts + TILE}]")
        if not dram_weights(up_gate, self.ep_experts or self.num_experts):
            raise ValueError("up_gate must be the routed expert weights, TILE and DRAM interleaved in the weight dtype")
        if self.shared != (shared_up_gate is not None) or (self.shared and not dram_weights(shared_up_gate, 1)):
            raise ValueError(
                "shared_up_gate must be given exactly when the op has a shared expert, laid out as up_gate"
            )
        if bool(self.ep_experts) != (ep_offset is not None):
            raise ValueError("ep_offset must be given exactly when the op is expert parallel")
        if list(act.shape) != [1, 1, self.k + 1, self.inter] or list(route.shape) != [1, 1, 1, 32]:
            raise ValueError("act and route must come from allocate_outputs()")
