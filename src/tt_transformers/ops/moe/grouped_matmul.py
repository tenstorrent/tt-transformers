# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Grouped expert matmul for the expert-sorted MoE prefill: group g of 32 activation rows times expert group_ids[g].

x [1, G, 32, K] holds the dispatched rows group by group (sorted_dispatch.gather_rows) and group_ids [1, G] the expert
of each group, read on device at run time, so one program serves any routing and stays trace-safe. Groups whose id is
past the experts (padding) are skipped: no reads and no output. Each core takes consecutive whole groups.

When an expert's whole [K, N] weight block fits L1 next to the other buffers, it stays resident across a run of
consecutive groups of the same expert (the dispatch puts them side by side), so each core reads it once per run;
otherwise the K-blocks stream per group through the stock ttnn matmul compute kernel. With resident weights the output
can be written ROW_MAJOR (pack-untilize), as the un-dispatch reads it.
"""

from __future__ import annotations

import ttnn

from ._program import KERNEL_DIR, TILE, runtime_args

L1_CB_BUDGET = 1_300_000  # bytes of circular buffers per core; kernels and stacks take the rest of the 1.5 MB L1
DRAM_ALIGN = 64
UNTILIZE_MAX_W = 16  # pack-untilize is exact up to 16 tiles of row width
STREAM_COMPUTE = "ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_large_block_zm_fused_bias_activation.cpp"
TILE_BYTES = {ttnn.bfloat16: 2048, ttnn.bfloat8_b: 1088, ttnn.bfloat4_b: 576}
CB_IN0, CB_IN1, CB_BIAS, CB_OUT, CB_INTERM, CB_IDX, CB_IDX_IN0, CB_IN0_TRANSPOSED = 0, 1, 3, 4, 5, 7, 8, 10


def _align(n: int, alignment: int = DRAM_ALIGN) -> int:
    return -(-n // alignment) * alignment


def _largest_divisor(n: int, at_most: int) -> int:
    return max(d for d in range(1, min(n, at_most) + 1) if n % d == 0)


class GroupedExpertMatmul:
    def __init__(
        self,
        k: int,
        n: int,
        num_experts: int,
        weight_dtype,
        compute_kernel_config,
        *,
        in0_block_w: int = 16,
        untilize_out: bool = False,
    ):
        """weights [1, num_experts, k, n] TILE in weight_dtype. in0_block_w: the largest K-block (tiles) to stream;
        smaller when L1 is short. untilize_out: ROW_MAJOR output, which needs the resident weights."""
        if weight_dtype not in TILE_BYTES:
            raise ValueError(f"weights must be bfloat16, bfloat8_b or bfloat4_b, got {weight_dtype}")
        if k % TILE or n % TILE:
            raise ValueError(f"k ({k}) and n ({n}) must be multiples of {TILE}")
        self.k, self.n, self.num_experts, self.weight_dtype = k, n, num_experts, weight_dtype
        self.config = compute_kernel_config
        fp32 = compute_kernel_config.fp32_dest_acc_en
        self.kt, self.nt = k // TILE, n // TILE
        self.out_subblock_w = _largest_divisor(self.nt, 4 if fp32 else 8)

        weight_tile = _align(TILE_BYTES[weight_dtype])
        out_bytes = self.nt * TILE_BYTES[ttnn.bfloat16]
        fixed = out_bytes + self.nt * (4096 if fp32 else 2048)  # output block and partials

        def fits(block: int, resident: bool) -> bool:
            weights = self.kt * self.nt * weight_tile if resident else 2 * block * self.nt * weight_tile
            return 2 * block * TILE_BYTES[ttnn.bfloat16] + weights + fixed <= L1_CB_BUDGET

        self.resident = fits(1, resident=True)
        block = _largest_divisor(self.kt, in0_block_w)
        while block > 1 and not fits(block, self.resident):
            block = _largest_divisor(self.kt, block - 1)
        if not fits(block, self.resident):
            raise ValueError(f"a [{k}, {n}] {weight_dtype} expert block does not fit L1")
        self.in0_block_w, self.num_blocks = block, self.kt // block
        self.packer_l1_acc = compute_kernel_config.packer_l1_acc and self.num_blocks > 2

        self.untilize_chunk_w = 0
        if untilize_out:
            if not self.resident:
                raise ValueError("untilize_out needs the expert weights resident in L1")
            w = self.out_subblock_w
            self.untilize_chunk_w = max(c for c in range(w, UNTILIZE_MAX_W + 1, w) if self.nt % c == 0)

    @property
    def untilize_out(self) -> bool:
        return self.untilize_chunk_w > 0

    def allocate_output(self, mesh_device, groups: int, memory_config=ttnn.DRAM_MEMORY_CONFIG):
        layout = ttnn.ROW_MAJOR_LAYOUT if self.untilize_out else ttnn.TILE_LAYOUT
        return ttnn.empty([1, groups, TILE, self.n], ttnn.bfloat16, layout, mesh_device, memory_config)

    def __call__(self, x, weights, group_ids, out=None):
        """x [1, G, 32, k] BF16 TILE, weights [1, num_experts, k, n] TILE, group_ids [1, G] uint16 ROW_MAJOR ->
        [1, G, 32, n] BF16 (ROW_MAJOR with untilize_out). Padding groups of the output are not written."""
        groups = group_ids.shape[-1]
        self._check(x, weights, group_ids, groups)
        device = x.device()
        out = out if out is not None else self.allocate_output(device, groups)
        if list(out.shape) != [1, groups, TILE, self.n] or out.layout != (
            ttnn.ROW_MAJOR_LAYOUT if self.untilize_out else ttnn.TILE_LAYOUT
        ):
            raise ValueError("out must come from allocate_output for this many groups")

        grid = device.compute_with_storage_grid_size()
        _, all_cores, group_1, group_2, per_core_1, per_core_2 = ttnn.split_work_to_cores(grid, groups, True)
        idx_page = _align(groups * 2)
        defines = [("SKIP_PADDING_GROUPS", "1")]
        if self.resident:
            defines.append(("RESIDENT_IN1", "1"))
        if self.untilize_out:
            defines += [("UNTILIZE_OUT", "1"), ("UNTILIZE_CHUNK_W", str(self.untilize_chunk_w))]

        reader_rt, writer_rt, first = {}, {}, 0
        for core in ttnn.corerange_to_cores(all_cores, None, True):
            count = per_core_1 if group_1.contains(core) else per_core_2
            xy = (core.x, core.y)
            reader_rt[xy] = [x.buffer_address(), first * self.kt, count, group_ids.buffer_address(), first]
            writer_rt[xy] = [
                weights.buffer_address(),
                count,
                out.buffer_address(),
                first * self.nt,
                group_ids.buffer_address(),
                first,
            ]
            first += count

        dm = ttnn.DataMovementConfigDescriptor
        kernels = [
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "gm_reader.cpp"),
                core_ranges=all_cores,
                compile_time_args=self._reader_args(x, group_ids, idx_page),
                named_compile_time_args=[("cb_in0", CB_IN0), ("cb_idx_in0", CB_IDX_IN0)],
                runtime_args=runtime_args(reader_rt),
                config=dm(processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_1),
                defines=defines,
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "gm_writer.cpp"),
                core_ranges=all_cores,
                compile_time_args=self._writer_args(weights, out, group_ids, idx_page, groups),
                named_compile_time_args=[("cb_in1", CB_IN1), ("cb_out", CB_OUT), ("cb_idx", CB_IDX)],
                runtime_args=runtime_args(writer_rt),
                config=dm(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_0),
                defines=defines,
            ),
        ]
        for cores, count in ((group_1, per_core_1), (group_2, per_core_2)):
            if cores.num_cores():
                kernels.append(self._compute(cores, count))
        cbs = self._cbs(all_cores, idx_page)
        ttnn.generic_op([x, weights, group_ids, out], ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
        return out

    def _reader_args(self, x, group_ids, idx_page):
        block_tiles = self.in0_block_w  # per_core_M = 1 tile row
        args = [1, self.kt, self.in0_block_w, self.in0_block_w, 1, block_tiles, 0, 0, self.num_blocks, 0, self.kt]
        args += ttnn.TensorAccessorArgs(x).get_compile_time_args() + [idx_page, self.num_experts]
        return args + ttnn.TensorAccessorArgs(group_ids).get_compile_time_args()

    def _writer_args(self, weights, out, group_ids, idx_page, groups):
        w, nt = self.out_subblock_w, self.nt
        args = [1, nt, self.in0_block_w * nt, nt, self.in0_block_w, nt * self.in0_block_w, self.num_blocks, 0]
        args += [self.kt * nt, 1, nt, w, nt, w, 1, w, nt // w, 1, nt, idx_page, groups, self.num_experts]
        for t in (weights, out, group_ids):
            args += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        return args

    def _compute(self, cores, groups_per_core):
        w, nt, block = self.out_subblock_w, self.nt, self.in0_block_w
        args = [block, 1, block, block, nt // w, nt * block, nt, self.num_blocks, 1, 1, 1, w, w]
        args += [groups_per_core, nt, int(self.untilize_out), 1, 0]
        defines = [("UNTILIZE_CHUNK_W", str(self.untilize_chunk_w))] if self.untilize_out else []
        if self.packer_l1_acc:
            defines.append(("PACKER_L1_ACC", "1"))
        if self.config.fp32_dest_acc_en:
            defines.append(("FP32_DEST_ACC_EN", "1"))
        names = ("cb_in0", "cb_in1", "cb_bias", "cb_out", "cb_intermed0", "cb_in0_transposed")
        indices = (CB_IN0, CB_IN1, CB_BIAS, CB_OUT, CB_INTERM, CB_IN0_TRANSPOSED)
        return ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "gm_compute.cpp") if self.resident else STREAM_COMPUTE,
            core_ranges=cores,
            compile_time_args=args,
            named_compile_time_args=list(zip(names, indices)),
            defines=defines,
            runtime_args=[],
            config=ttnn.ComputeConfigDescriptor(
                math_fidelity=self.config.math_fidelity,
                fp32_dest_acc_en=self.config.fp32_dest_acc_en,
                math_approx_mode=self.config.math_approx_mode,
                dst_full_sync_en=self.config.dst_full_sync_en,
            ),
        )

    def _cbs(self, cores, idx_page):
        def cb(index_formats, pages, page):
            descriptors = [
                ttnn.CBFormatDescriptor(buffer_index=i, data_format=f, page_size=page) for i, f in index_formats
            ]
            return ttnn.CBDescriptor(total_size=pages * page, core_ranges=cores, format_descriptors=descriptors)

        weight_tile = _align(TILE_BYTES[self.weight_dtype])
        in1_pages = self.kt * self.nt if self.resident else 2 * self.in0_block_w * self.nt
        partials = ttnn.float32 if self.config.fp32_dest_acc_en else ttnn.bfloat16
        cbs = [
            cb([(CB_IN0, ttnn.bfloat16)], 2 * self.in0_block_w, TILE_BYTES[ttnn.bfloat16]),
            cb([(CB_IN1, self.weight_dtype)], in1_pages, weight_tile),
            cb([(CB_IDX, ttnn.uint16)], 1, idx_page),
            cb([(CB_IDX_IN0, ttnn.uint16)], 1, idx_page),
        ]
        if partials == ttnn.bfloat16:
            cbs.append(cb([(CB_OUT, ttnn.bfloat16), (CB_INTERM, ttnn.bfloat16)], self.nt, TILE_BYTES[ttnn.bfloat16]))
        else:
            cbs.append(cb([(CB_OUT, ttnn.bfloat16)], self.nt, TILE_BYTES[ttnn.bfloat16]))
            cbs.append(cb([(CB_INTERM, ttnn.float32)], self.nt, 4096))
        return cbs

    def _check(self, x, weights, group_ids, groups):
        if (
            x.dtype != ttnn.bfloat16
            or x.layout != ttnn.TILE_LAYOUT
            or list(x.padded_shape) != [1, groups, TILE, self.k]
        ):
            raise ValueError(f"x must be BF16 TILE [1, {groups}, 32, {self.k}]")
        if (
            weights.dtype != self.weight_dtype
            or weights.layout != ttnn.TILE_LAYOUT
            or list(weights.padded_shape) != [1, self.num_experts, self.k, self.n]
            or weights.is_sharded()
        ):
            raise ValueError(f"weights must be {self.weight_dtype} TILE [1, {self.num_experts}, {self.k}, {self.n}]")
        if group_ids.dtype != ttnn.uint16 or group_ids.layout != ttnn.ROW_MAJOR_LAYOUT or len(group_ids.shape) != 2:
            raise ValueError("group_ids must be uint16 ROW_MAJOR [1, G]")
