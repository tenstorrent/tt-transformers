# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Expert-sorted MoE prefill bookkeeping: dispatch, gather and un-dispatch of token rows.

For T tokens routed to k experts each, `sorted_dispatch` lays the T * k (token, slot) pairs out expert by expert, each
expert's rows padded to a multiple of `group` rows, so a grouped matmul can run every group of rows through one expert.
`gather_rows` builds those groups from the token rows and `gather_add_rows` sums each token's k expert outputs back.
"""

from __future__ import annotations

import ttnn

from ._program import KERNEL_DIR, TILE, core_range_set, reader_writer_compute, runtime_args

GROUP = 32
PAD_GROUP_ID = 0xFFFF  # the expert of a padding group: past any expert, so consumers skip it
MAX_K = 32  # the top-k outputs are read as one tile column
L1_ALIGN = 32
BLOCK_L1_BYTES = 1_200_000  # the row-major and tilized blocks of a gather core
BF16_TILE_BYTES = TILE * TILE * 2


def dispatch_rows(tokens: int, k: int, num_experts: int, group: int = GROUP) -> int:
    """Rows that hold any routing: the pairs, every expert's last group padded, and the trailing padding group that
    takes the pairs routed off this device."""
    pairs = -(-(tokens * k) // TILE) * TILE
    expert_padding = -(-((group - 1) * num_experts) // group) * group
    return pairs + expert_padding + group


def _align(n: int) -> int:
    return -(-n // L1_ALIGN) * L1_ALIGN


def _scratch(core, index: int, dtype, size: int) -> ttnn.CBDescriptor:
    size = _align(size)
    return ttnn.CBDescriptor(
        total_size=size,
        core_ranges=core,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=size)],
    )


def _blocks(cores, index: int, pages: int, page: int = BF16_TILE_BYTES, dtype=ttnn.bfloat16) -> ttnn.CBDescriptor:
    return ttnn.CBDescriptor(
        total_size=pages * page,
        core_ranges=cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page)],
    )


def _accessor_args(*tensors) -> list[int]:
    return [a for t in tensors for a in ttnn.TensorAccessorArgs(t).get_compile_time_args()]


def _col_splits(width_tiles: int, rows: int, blocks: int, cores: int) -> int:
    """Split the width into column blocks until there is a work item per core, then until `blocks` blocks fit L1."""
    splits = 1
    for s in (1, 2, 3, 4, 6, 8, 12, 16):
        if width_tiles % s:
            continue
        splits = s
        if rows * s >= cores:
            break
    while (width_tiles // splits) * BF16_TILE_BYTES * blocks > BLOCK_L1_BYTES and width_tiles % (2 * splits) == 0:
        splits *= 2
    return splits


def _work(grid, items: int) -> tuple[list[tuple[ttnn.CoreCoord, int, int]], ttnn.CoreRangeSet]:
    """Items spread over the grid row-wise: ([(core, first item, count)], cores)."""
    _, all_cores, group_1, _, per_core_1, per_core_2 = ttnn.split_work_to_cores(grid, items, True)
    work, first = [], 0
    for core in ttnn.corerange_to_cores(all_cores, None, True):
        count = per_core_1 if group_1.contains(core) else per_core_2
        work.append((core, first, count))
        first += count
    return work, all_cores


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sorted_dispatch(
    ids,
    scores,
    num_experts: int,
    *,
    rows: int,
    group: int = GROUP,
    pad_group_id: int = PAD_GROUP_ID,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
):
    """ids [.., T, k] uint16/uint32 and scores [.., T, k] BF16, TILE (the top-k outputs; ids >= num_experts are routed
    elsewhere and dropped) -> ROW_MAJOR sticks:

    - src_tok [1, rows] uint32: the token of each dispatched row (0 on padding rows)
    - weights [1, rows] BF16: its routing weight (0 on padding rows)
    - group_ids [1, rows / group] uint16: the expert of each group (pad_group_id on padding groups)
    - undispatch [1, T * k] uint32: the dispatched row of each (token, slot) pair, token-major
    """
    tokens, k = ids.shape[-2], ids.shape[-1]
    _check(ids.dtype in (ttnn.uint16, ttnn.uint32) and scores.dtype == ttnn.bfloat16, "ids uint16/uint32, scores BF16")
    _check(ids.layout == scores.layout == ttnn.TILE_LAYOUT, "ids and scores must be TILE (the top-k outputs)")
    _check(list(scores.shape)[-2:] == [tokens, k] and 1 <= k <= MAX_K, f"ids and scores must be [.., T, k <= {MAX_K}]")
    _check(group > 0 and group % TILE == 0 and rows % group == 0, "group and rows must be multiples of 32 and group")
    _check(rows >= dispatch_rows(tokens, k, num_experts, group), "rows is below the worst-case padded dispatch")

    device = ids.device()
    pairs, groups = tokens * k, rows // group
    outputs = [
        ttnn.empty([1, rows], ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, device, memory_config),
        ttnn.empty([1, rows], ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, device, memory_config),
        ttnn.empty([1, groups], ttnn.uint16, ttnn.ROW_MAJOR_LAYOUT, device, memory_config),
        ttnn.empty([1, pairs], ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, device, memory_config),
    ]
    src_tok, weights, group_ids, undispatch = outputs

    tiles = -(-tokens // TILE)
    ids_elem = 2 if ids.dtype == ttnn.uint16 else 4
    ids_page, scores_page = TILE * TILE * ids_elem, BF16_TILE_BYTES
    out_pages = [rows * 4, rows * 2, groups * 2, pairs * 4]
    core = core_range_set([ttnn.CoreCoord(0, 0)])
    names = ["cb_ids", "cb_scores", "cb_src", "cb_w", "cb_gid", "cb_und", "cb_scratch"]
    sizes = [tiles * ids_page, tiles * scores_page, *out_pages, num_experts * 4 * 3]
    formats = [ttnn.uint32, ttnn.bfloat16, ttnn.uint32, ttnn.bfloat16, ttnn.uint16, ttnn.uint32, ttnn.uint32]
    cbs = [_scratch(core, i, f, s) for i, (f, s) in enumerate(zip(formats, sizes))]

    ct = [pairs, tokens, k, num_experts, group, rows, groups, ids_page, scores_page, *out_pages, tiles, ids_elem]
    ct += [pad_group_id] + _accessor_args(ids, scores, *outputs)
    kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "sd_dispatch.cpp"),
        core_ranges=core,
        compile_time_args=ct,
        named_compile_time_args=list(zip(names, range(len(names)))),
        runtime_args=runtime_args({(0, 0): [t.buffer_address() for t in (ids, scores, *outputs)]}),
        config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_0),
    )
    program = ttnn.ProgramDescriptor(kernels=[kernel], semaphores=[], cbs=cbs)
    ttnn.generic_op([ids, scores, src_tok, weights, group_ids, undispatch], program)
    return src_tok, weights, group_ids, undispatch


def gather_rows(rows, src_tok, group_ids, num_experts: int, *, memory_config=ttnn.DRAM_MEMORY_CONFIG):
    """rows [T, H] BF16 ROW_MAJOR, src_tok and group_ids from `sorted_dispatch` -> [1, G, 32, H] BF16 TILE, group g
    holding its 32 dispatched token rows. Padding groups are not written."""
    _check(rows.dtype == ttnn.bfloat16 and rows.layout == ttnn.ROW_MAJOR_LAYOUT, "rows must be BF16 ROW_MAJOR")
    _check(len(rows.shape) == 2 and rows.shape[-1] % TILE == 0, "rows must be [T, H] with H a multiple of 32")
    groups = group_ids.shape[-1]
    _check(group_ids.dtype == ttnn.uint16 and src_tok.dtype == ttnn.uint32, "group_ids uint16, src_tok uint32")
    _check(list(src_tok.shape) == [1, groups * GROUP], "src_tok must be [1, 32 * groups]")

    device, hidden = rows.device(), rows.shape[-1]
    width = hidden // TILE
    grid = device.compute_with_storage_grid_size()
    col_splits = _col_splits(width, groups, 4, grid.x * grid.y)
    work, cores = _work(grid, groups * col_splits)
    block = width // col_splits
    out = ttnn.empty([1, groups, GROUP, hidden], ttnn.bfloat16, ttnn.TILE_LAYOUT, device, memory_config)

    cb_in, cb_out, cb_src, cb_gid, cb_flag = range(5)
    cbs = [
        _blocks(cores, cb_in, 2 * block),
        _blocks(cores, cb_out, 2 * block),
        _blocks(cores, cb_src, 1, _align(GROUP * 4), ttnn.uint32),
        _blocks(cores, cb_gid, 1, _align(groups * 2), ttnn.uint16),
        _blocks(cores, cb_flag, 4, L1_ALIGN, ttnn.uint32),
    ]
    rd_ct = [num_experts, block, col_splits, block * TILE * 2, groups, cb_in, cb_src, cb_gid, cb_flag]
    rd_ct += _accessor_args(rows, src_tok, group_ids)
    wr_ct = [width, block, col_splits, cb_out, cb_flag] + _accessor_args(out)
    rd_rt, wr_rt, cp_rt = {}, {}, {}
    for core, first, count in work:
        xy = (core.x, core.y)
        rd_rt[xy] = [rows.buffer_address(), src_tok.buffer_address(), group_ids.buffer_address(), first, count]
        wr_rt[xy] = [out.buffer_address(), first, count]
        cp_rt[xy] = [count]
    kernels = reader_writer_compute(
        cores,
        "gr",
        (rd_ct, rd_rt),
        (wr_ct, wr_rt),
        ([block, cb_in, cb_out], cp_rt),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=False,
    )
    ttnn.generic_op([rows, src_tok, group_ids, out], ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
    return out


def gather_add_rows(rows, undispatch, tokens: int, k: int, *, memory_config=ttnn.DRAM_MEMORY_CONFIG):
    """rows [R, H] BF16 ROW_MAJOR (the weighted expert output of each dispatched row) and undispatch from
    `sorted_dispatch` -> [1, 1, T, H] BF16 TILE: each token's k rows summed in slot order. The trailing group of rows
    (pairs routed to another device) is never read, so it need not hold zeros."""
    _check(rows.dtype == ttnn.bfloat16 and rows.layout == ttnn.ROW_MAJOR_LAYOUT, "rows must be BF16 ROW_MAJOR")
    _check(len(rows.shape) == 2 and rows.shape[-1] % TILE == 0, "rows must be [R, H] with H a multiple of 32")
    _check(1 <= k <= MAX_K, f"k must be in [1, {MAX_K}]")
    _check(undispatch.dtype == ttnn.uint32 and list(undispatch.shape) == [1, tokens * k], "undispatch must be [1, T*k]")

    device, hidden = rows.device(), rows.shape[-1]
    width, token_tiles = hidden // TILE, -(-tokens // TILE)
    grid = device.compute_with_storage_grid_size()
    col_splits = _col_splits(width, token_tiles, 6, grid.x * grid.y)
    work, cores = _work(grid, token_tiles * col_splits)
    block = width // col_splits
    out = ttnn.empty([1, 1, tokens, hidden], ttnn.bfloat16, ttnn.TILE_LAYOUT, device, memory_config)

    cb_in, cb_til, cb_acc_a, cb_acc_b, cb_und, cb_out = range(6)
    cbs = [
        _blocks(cores, cb_in, 2 * block),
        _blocks(cores, cb_til, block),
        _blocks(cores, cb_acc_a, block),
        _blocks(cores, cb_acc_b, block),
        _blocks(cores, cb_und, 1, _align(TILE * k * 4), ttnn.uint32),
        _blocks(cores, cb_out, block),
    ]
    pad_row = rows.shape[0] - GROUP
    rd_ct = [tokens, k, block, col_splits, block * TILE * 2, cb_in, cb_und, pad_row] + _accessor_args(rows, undispatch)
    wr_ct = [width, block, col_splits, cb_out] + _accessor_args(out)
    cp_ct = [block, k, cb_in, cb_til, cb_acc_a, cb_acc_b, cb_out]
    rd_rt, wr_rt, cp_rt = {}, {}, {}
    for core, first, count in work:
        xy = (core.x, core.y)
        rd_rt[xy] = [rows.buffer_address(), undispatch.buffer_address(), first, count]
        wr_rt[xy] = [out.buffer_address(), first, count]
        cp_rt[xy] = [count]
    kernels = reader_writer_compute(
        cores,
        "ga",
        (rd_ct, rd_rt),
        (wr_ct, wr_rt),
        (cp_ct, cp_rt),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=False,
    )
    ttnn.generic_op([rows, undispatch, out], ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
    return out
