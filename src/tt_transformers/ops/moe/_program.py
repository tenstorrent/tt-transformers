# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Core placement and program-descriptor helpers shared by the fused MoE decode ops."""

from __future__ import annotations

from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"
TILE = 32
TILE_BYTES = {ttnn.bfloat4_b: 576, ttnn.bfloat8_b: 1088}  # the expert weight dtypes the LoFi matmuls read
MAX_PACKET_BYTES = 16384
WEIGHT_CB_MAX_BYTES = 1024 * 1024  # a core's weight buffer, leaving the rest of the 1.5 MiB L1 to the op


def packet_bytes(tiles: int, tile_bytes: int) -> int:
    """Largest NOC packet holding whole tiles that divides a `tiles`-tile block."""
    return max(d for d in range(1, tiles + 1) if tiles % d == 0 and d * tile_bytes <= MAX_PACKET_BYTES) * tile_bytes


def bank_workers(mesh_device):
    """The optimal reader core of each DRAM bank."""
    return list(ttnn.device.get_optimal_dram_bank_to_logical_worker_assignment(mesh_device, ttnn.NOC.NOC_0))


def place_along_rows(mesh_device, per_bank: dict[int, int]) -> dict[int, list]:
    """{bank: count} -> {bank: [CoreCoord]}: each bank's cores on its optimal reader's row, nearest first (+dx before
    -dx), then the rows above and below (off-row placement measured slower)."""
    workers = bank_workers(mesh_device)
    grid = mesh_device.compute_with_storage_grid_size()
    used = set()
    out = {}
    for b, n in per_bank.items():
        w = workers[b]
        candidates = (
            (w.x + dx, w.y + dy) for dy in (0, -1, 1, -2, 2) for d in range(grid.x) for dx in ((d, -d) if d else (0,))
        )
        cores = []
        for x, y in candidates:
            if len(cores) == n:
                break
            if 0 <= x < grid.x and 0 <= y < grid.y and (x, y) not in used:
                used.add((x, y))
                cores.append(ttnn.CoreCoord(x, y))
        if len(cores) != n:
            raise ValueError(f"no room for {n} cores near DRAM bank {b} on a {grid.x} x {grid.y} grid")
        out[b] = cores
    return out


def core_range_set(cores) -> ttnn.CoreRangeSet:
    return ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])


def cb(cores, index: int, dtype, page: int, pages: int, tiny: bool = False) -> ttnn.CBDescriptor:
    """tiny: 1x32 activation tiles."""
    fmt = dict(buffer_index=index, data_format=dtype, page_size=page)
    if tiny:
        fmt["tile"] = ttnn.TileDescriptor(ttnn.Tile([1, TILE]))
    return ttnn.CBDescriptor(
        total_size=page * pages, core_ranges=cores, format_descriptors=[ttnn.CBFormatDescriptor(**fmt)]
    )


def runtime_args(args: dict[tuple[int, int], list[int]]) -> ttnn.RuntimeArgs:
    rt = ttnn.RuntimeArgs()
    for (x, y), a in args.items():
        rt[x][y] = a
    return rt


def reader_writer_compute(cores, name: str, reader, writer, compute, defines=(), dst_full: bool = False):
    """Kernel descriptors for kernels/{name}_reader.cpp (NCRISC, NOC0), {name}_writer.cpp (BRISC, NOC1) and
    {name}_compute.cpp (LoFi, FP32 accumulation). reader / writer / compute: (compile args, {(x, y): runtime args})."""
    (rd_ct, rd_rt), (wr_ct, wr_rt), (cp_ct, cp_rt) = reader, writer, compute
    dm = ttnn.DataMovementConfigDescriptor
    defines = list(defines)
    return [
        ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / f"{name}_reader.cpp"),
            core_ranges=cores,
            compile_time_args=rd_ct,
            runtime_args=runtime_args(rd_rt),
            config=dm(processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_0),
            defines=defines,
        ),
        ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / f"{name}_writer.cpp"),
            core_ranges=cores,
            compile_time_args=wr_ct,
            runtime_args=runtime_args(wr_rt),
            config=dm(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_1),
            defines=defines,
        ),
        ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / f"{name}_compute.cpp"),
            core_ranges=cores,
            compile_time_args=cp_ct,
            runtime_args=runtime_args(cp_rt),
            defines=defines,
            config=ttnn.ComputeConfigDescriptor(
                math_fidelity=ttnn.MathFidelity.LoFi,
                fp32_dest_acc_en=True,
                math_approx_mode=False,
                dst_full_sync_en=dst_full,
            ),
        ),
    ]
