// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Grouped expert matmul (ops/moe/grouped_matmul.py), weight reader and output writer.
//
// This core owns `batch` consecutive groups from `start_batch`. For every group it reads the [K, N] weights of expert
// ids[start_batch + b] (read from the device-resident id list at run time, so the program serves any routing and is
// trace-safe) and writes the group's [32, N] output block. gm_reader.cpp reads the same groups' activations.
//
// Derived from ttnn's reader_writer_bmm_tile_layout_in1.cpp (interleaved in1 and output only): compile-time args 0..18
// are that kernel's; the id-list accessor args follow the in1 and output accessor args.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#if defined(SKIP_PADDING_GROUPS) || defined(RESIDENT_IN1)
#include "ckernel.h"
#include "ckernel_defs.h"
#endif
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    uint32_t rt_args_idx = 0;
    const uint32_t in1_tensor_addr = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t batch = get_arg_val<uint32_t>(rt_args_idx++);  // groups owned by this core
    const uint32_t out_tensor_addr = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t out_tensor_start_tile_id = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t idx_tensor_addr = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t start_batch = get_arg_val<uint32_t>(rt_args_idx++);  // first compact group of this core

    constexpr uint32_t in1_tensor_stride_w = get_compile_time_arg_val(0);
    constexpr uint32_t in1_tensor_stride_h = get_compile_time_arg_val(1);
    constexpr uint32_t in1_tensor_next_block_stride = get_compile_time_arg_val(2);
    constexpr uint32_t in1_block_w = get_compile_time_arg_val(3);
    constexpr uint32_t in1_block_h = get_compile_time_arg_val(4);
    constexpr uint32_t in1_block_num_tiles = get_compile_time_arg_val(5);
    constexpr uint32_t num_blocks = get_compile_time_arg_val(6);
    // arg 7 (bcast_B) is unused: in1 is always indexed per group
    constexpr uint32_t KtNt = get_compile_time_arg_val(8);
    constexpr uint32_t out_tensor_stride_w = get_compile_time_arg_val(9);
    constexpr uint32_t out_tensor_stride_h = get_compile_time_arg_val(10);
    constexpr uint32_t out_tensor_next_subblock_stride_w = get_compile_time_arg_val(11);
    constexpr uint32_t out_tensor_next_subblock_stride_h = get_compile_time_arg_val(12);
    constexpr uint32_t out_subblock_w = get_compile_time_arg_val(13);
    constexpr uint32_t out_subblock_h = get_compile_time_arg_val(14);
    constexpr uint32_t out_subblock_tile_count = get_compile_time_arg_val(15);
    constexpr uint32_t out_num_subblocks_w = get_compile_time_arg_val(16);
    constexpr uint32_t out_num_subblocks_h = get_compile_time_arg_val(17);
    constexpr uint32_t MtNt = get_compile_time_arg_val(18);
    constexpr uint32_t idx_page_bytes = get_compile_time_arg_val(19);
    constexpr uint32_t num_groups_total = get_compile_time_arg_val(20);  // length of the id list (bounds ASSERT)
    constexpr uint32_t num_experts = get_compile_time_arg_val(21);       // sparse groups in in1 (bounds ASSERT)

    constexpr uint32_t dfb_id_in1 = get_named_compile_time_arg_val("cb_in1");
    constexpr uint32_t dfb_id_out0 = get_named_compile_time_arg_val("cb_out");
    constexpr uint32_t dfb_id_idx = get_named_compile_time_arg_val("cb_idx");

    constexpr auto in1_args = TensorAccessorArgs<22>();
    constexpr auto out_args = TensorAccessorArgs<in1_args.next_compile_time_args_offset()>();
    constexpr auto idx_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();

    Noc noc;
    DataflowBuffer dfb_in1(dfb_id_in1);
    DataflowBuffer dfb_out(dfb_id_out0);
    DataflowBuffer dfb_idx(dfb_id_idx);

    const uint32_t in1_single_tile_size_bytes = dfb_in1.get_tile_size();
    const uint32_t in1_aligned_tile_size_bytes =
        (in1_single_tile_size_bytes + (DRAM_ALIGNMENT - 1)) & ~(DRAM_ALIGNMENT - 1);
    const auto s1 = TensorAccessor(in1_args, in1_tensor_addr);
    const uint32_t output_single_tile_size_bytes = dfb_out.get_tile_size();
    const auto s_out = TensorAccessor(out_args, out_tensor_addr);
    const auto s_idx = TensorAccessor(idx_args, idx_tensor_addr);

    // The whole id list is one ROW_MAJOR stick (validated on the host): one page-0 read.
    dfb_idx.reserve_back(1);
    const uint32_t idx_l1_addr = dfb_idx.get_write_ptr();
    noc.async_read(s_idx, dfb_idx, idx_page_bytes, {.page_id = 0}, {.offset_bytes = 0});
    noc.async_read_barrier();
    const volatile tt_l1_ptr uint16_t* ids = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(idx_l1_addr);

    for (uint32_t b = 0; b < batch; ++b) {
        const uint32_t gb = start_batch + b;
        ASSERT(gb < num_groups_total);
        const uint32_t group_id = ids[gb];
#ifdef RESIDENT_IN1
        // weights once per expert: consecutive groups of this core that share the expert form a RUN; the whole [K, N]
        // block is read into cb_in1 (sized for it) at the run's first group and stays resident until the compute
        // kernel pops it after the run's last group (run code through the BRISC mailbox: 0 skip, 1 single, 2 first,
        // 3 middle, 4 last). The id list is this forward's routing (device-resident), so runs are found at run time.
        const bool group_valid = group_id < num_experts;
        const bool prev_same = group_valid && b > 0 && ids[gb - 1] == group_id;
        const bool next_same = group_valid && (b + 1 < batch) && ids[gb + 1] == group_id;
        const uint32_t run_code = !group_valid ? 0u : (prev_same ? (next_same ? 3u : 4u) : (next_same ? 2u : 1u));
        ckernel::mailbox_write(ckernel::ThreadId::UnpackThreadId, run_code);
        ckernel::mailbox_write(ckernel::ThreadId::MathThreadId, run_code);
        ckernel::mailbox_write(ckernel::ThreadId::PackThreadId, run_code);
        if (!group_valid) {
            out_tensor_start_tile_id += MtNt;
            continue;
        }
        if (!prev_same) {
            // the run's first group: read every K-block of the expert into the resident CB region in one reservation
            constexpr uint32_t resident_tiles = num_blocks * in1_block_num_tiles;
            dfb_in1.reserve_back(resident_tiles);
            uint32_t in1_tensor_current_block_start_tile_id = group_id * KtNt;
            uint32_t in1_write_offset = 0;
            for (uint32_t block = 0; block < num_blocks; ++block) {
                uint32_t in1_tensor_row_start_tile_id = in1_tensor_current_block_start_tile_id;
                for (uint32_t h = 0; h < in1_block_h; ++h) {
                    uint32_t in1_tensor_tile_id = in1_tensor_row_start_tile_id;
                    for (uint32_t w = 0; w < in1_block_w; ++w) {
                        noc.async_read(
                            s1,
                            dfb_in1,
                            in1_single_tile_size_bytes,
                            {.page_id = in1_tensor_tile_id},
                            {.offset_bytes = in1_write_offset});
                        in1_write_offset += in1_aligned_tile_size_bytes;
                        in1_tensor_tile_id += in1_tensor_stride_w;
                    }
                    in1_tensor_row_start_tile_id += in1_tensor_stride_h;
                }
                in1_tensor_current_block_start_tile_id += in1_tensor_next_block_stride;
            }
            noc.async_read_barrier();
            dfb_in1.push_back(resident_tiles);
        }
#elif defined(SKIP_PADDING_GROUPS)
        // padding groups (id >= num_experts) are skipped by all three kernels: the compute kernel learns it through
        // the BRISC mailbox (get_batch_from_reader), the in0 reader reads the same id list
        const bool group_valid = group_id < num_experts;
        ckernel::mailbox_write(ckernel::ThreadId::UnpackThreadId, group_valid);
        ckernel::mailbox_write(ckernel::ThreadId::MathThreadId, group_valid);
        ckernel::mailbox_write(ckernel::ThreadId::PackThreadId, group_valid);
        if (!group_valid) {
            out_tensor_start_tile_id += MtNt;
            continue;
        }
#else
        ASSERT(group_id < num_experts);
#endif
#ifndef RESIDENT_IN1
        uint32_t in1_tensor_current_block_start_tile_id = group_id * KtNt;
        for (uint32_t block = 0; block < num_blocks; ++block) {
            dfb_in1.reserve_back(in1_block_num_tiles);
            uint32_t in1_write_offset = 0;
            uint32_t in1_tensor_row_start_tile_id = in1_tensor_current_block_start_tile_id;
            for (uint32_t h = 0; h < in1_block_h; ++h) {
                uint32_t in1_tensor_tile_id = in1_tensor_row_start_tile_id;
                for (uint32_t w = 0; w < in1_block_w; ++w) {
                    noc.async_read(
                        s1,
                        dfb_in1,
                        in1_single_tile_size_bytes,
                        {.page_id = in1_tensor_tile_id},
                        {.offset_bytes = in1_write_offset});
                    in1_write_offset += in1_aligned_tile_size_bytes;
                    in1_tensor_tile_id += in1_tensor_stride_w;
                }
                in1_tensor_row_start_tile_id += in1_tensor_stride_h;
            }
            in1_tensor_current_block_start_tile_id += in1_tensor_next_block_stride;
            noc.async_read_barrier();
            dfb_in1.push_back(in1_block_num_tiles);
        }
#endif  // !RESIDENT_IN1

#ifdef UNTILIZE_OUT
        // ROW_MAJOR output: the compute kernel pack-untilized this group's whole [32, N] block (per_core_M = 1 tile
        // row, per_core_N = all N tiles) into cb_out as 32 contiguous rows of N elements; store them as the group's 32
        // sticks (stick id = compact group * 32 + r). The tile-geometry runtime args are unused in this mode.
        {
            // the compute kernel pushes the [32, N] block in chunks of UNTILIZE_CHUNK_W tiles (row width limit of the
            // pack-untilize path); every chunk is 32 partial sticks of chunk_w * 32 elements at column offset c * chunk
            constexpr uint32_t chunk_w = UNTILIZE_CHUNK_W;
            constexpr uint32_t chunk_bytes = chunk_w * 32 * 2;
            constexpr uint32_t n_chunks = (out_subblock_w * out_num_subblocks_w) / chunk_w;
            const uint32_t stick0 = gb * 32;
            for (uint32_t c = 0; c < n_chunks; ++c) {
                dfb_out.wait_front(chunk_w);
                for (uint32_t r = 0; r < 32; ++r) {
                    noc.async_write(
                        dfb_out,
                        s_out,
                        chunk_bytes,
                        {.offset_bytes = r * chunk_bytes},
                        {.page_id = stick0 + r, .offset_bytes = c * chunk_bytes});
                }
                noc.async_write_barrier();
                dfb_out.pop_front(chunk_w);
            }
            out_tensor_start_tile_id += MtNt;
            continue;
        }
#endif
        uint32_t out_tensor_sbh_start_tile_id = out_tensor_start_tile_id;
        for (uint32_t sbh = 0; sbh < out_num_subblocks_h; ++sbh) {
            uint32_t out_tensor_sbw_start_tile_id = out_tensor_sbh_start_tile_id;
            for (uint32_t sbw = 0; sbw < out_num_subblocks_w; ++sbw) {
                uint32_t out_tensor_sb_row_start_tile_id = out_tensor_sbw_start_tile_id;
                dfb_out.wait_front(out_subblock_tile_count);
                uint32_t out_read_offset = 0;
                for (uint32_t h = 0; h < out_subblock_h; ++h) {
                    uint32_t out_tensor_tile_id = out_tensor_sb_row_start_tile_id;
                    for (uint32_t w = 0; w < out_subblock_w; ++w) {
                        noc.async_write(
                            dfb_out,
                            s_out,
                            output_single_tile_size_bytes,
                            {.offset_bytes = out_read_offset},
                            {.page_id = out_tensor_tile_id});
                        out_read_offset += output_single_tile_size_bytes;
                        out_tensor_tile_id += out_tensor_stride_w;
                    }
                    out_tensor_sb_row_start_tile_id += out_tensor_stride_h;
                }
                noc.async_write_barrier();
                dfb_out.pop_front(out_subblock_tile_count);
                out_tensor_sbw_start_tile_id += out_tensor_next_subblock_stride_w;
            }
            out_tensor_sbh_start_tile_id += out_tensor_next_subblock_stride_h;
        }
        out_tensor_start_tile_id += MtNt;
    }
    dfb_idx.push_back(1);
}
