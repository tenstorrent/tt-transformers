// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Grouped expert matmul (ops/moe/grouped_matmul.py), compute with the expert weights resident in L1. Derived from
// ttnn's bmm_large_block_zm_fused_bias_activation.cpp (same compile-time args and CB names). gm_writer.cpp pushes an
// expert's whole [K, N] weight block once per run of consecutive groups of that expert and sends a run code per group
// through the BRISC mailbox: 0 padding group (skip), 1 single-group run, 2 first of a run, 3 middle, 4 last. The
// K-blocks are indexed inside the resident block, which is popped after the run's last group. With UNTILIZE_CHUNK_W
// the output is pack-untilized to row-major in chunks of that many tiles.
#include <cstdint>

#include "api/compute/matmul.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/pack_untilize.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/compute/eltwise_binary.h"

FORCE_INLINE void reload_from_cb_to_dst(
    uint32_t in0_dfb_id,
    uint32_t in1_dfb_id,
    uint32_t mm_partials_dfb_id,
    bool in1_transpose_tile,
    uint32_t out_subblock_num_tiles,
    uint32_t out_subblock_w,
    uint32_t out_subblock_h,
    uint32_t in0_block_w) {
    DataflowBuffer mm_partials_dfb(mm_partials_dfb_id);
    // Reconfigure input
    reconfig_data_format_srca(in1_dfb_id, mm_partials_dfb_id);
    copy_init(mm_partials_dfb_id);
    mm_partials_dfb.wait_front(out_subblock_num_tiles);

    uint32_t start_dst_index = 0;
    uint32_t start_tile_index = 0;
    copy_block(mm_partials_dfb_id, start_tile_index, start_dst_index, out_subblock_num_tiles);

    mm_partials_dfb.pop_front(out_subblock_num_tiles);
    // Reconfigure srcA back
    reconfig_data_format_srca(mm_partials_dfb_id, in1_dfb_id);
    matmul_block_init(in0_dfb_id, in1_dfb_id, in1_transpose_tile, out_subblock_w, out_subblock_h, in0_block_w);
}

void kernel_main() {
    constexpr uint32_t in0_block_w = get_compile_time_arg_val(0);        // inner block size in tiles
    constexpr uint32_t in0_num_subblocks = get_compile_time_arg_val(1);  // outer row block size (in inner row blocks)
    constexpr uint32_t in0_block_num_tiles =
        get_compile_time_arg_val(2);  // out_subblock_h*in0_block_w*in0_num_subblocks;
    constexpr uint32_t in0_subblock_num_tiles = get_compile_time_arg_val(3);  // out_subblock_h*in0_block_w
    constexpr uint32_t in1_num_subblocks =
        get_compile_time_arg_val(4);  // outer column block size (in inner column blocks)
    constexpr uint32_t in1_block_num_tiles =
        get_compile_time_arg_val(5);                               // out_subblock_w*in0_block_w* in1_num_subblocks;
    constexpr uint32_t in1_block_w = get_compile_time_arg_val(6);  // out_subblock_w*in1_num_subblocks
    constexpr uint32_t num_blocks_inner_dim = get_compile_time_arg_val(7);     // outer inner dim (in inner dim blocks)
    constexpr uint32_t num_blocks_w_dim = get_compile_time_arg_val(8);         // outer inner dim (in inner dim blocks)
    constexpr uint32_t num_blocks_h_dim = get_compile_time_arg_val(9);         // outer inner dim (in inner dim blocks)
    constexpr uint32_t out_subblock_h = get_compile_time_arg_val(10);          // inner row block size in tiles
    constexpr uint32_t out_subblock_w = get_compile_time_arg_val(11);          // inner column block size in tiles
    constexpr uint32_t out_subblock_num_tiles = get_compile_time_arg_val(12);  // out_subblock_h * out_subblock_w;
    constexpr uint32_t batch = get_compile_time_arg_val(13);                   // batch dim
    constexpr uint32_t out_block_num_tiles = get_compile_time_arg_val(14);     // number of tiles in out_block
    constexpr bool untilize_out = get_compile_time_arg_val(15);                // untilize output

    constexpr uint32_t out_block_w = out_subblock_w * in1_num_subblocks;

    constexpr uint32_t in0_dfb_id = get_named_compile_time_arg_val("cb_in0");
    constexpr uint32_t in1_dfb_id = get_named_compile_time_arg_val("cb_in1");
    constexpr uint32_t out_dfb_id = get_named_compile_time_arg_val("cb_out");
    constexpr uint32_t mm_partials_dfb_id = get_named_compile_time_arg_val("cb_intermed0");
    constexpr uint32_t untilize_mode_out_dfb_id = untilize_out ? mm_partials_dfb_id : out_dfb_id;

    DataflowBuffer in0_dfb(in0_dfb_id);
    DataflowBuffer in1_dfb(in1_dfb_id);
    DataflowBuffer mm_partials_dfb(mm_partials_dfb_id);
    DataflowBuffer untilize_mode_out_dfb(untilize_mode_out_dfb_id);

    constexpr uint32_t mm_out_dfb_id = untilize_mode_out_dfb_id;
    DataflowBuffer mm_out_dfb(mm_out_dfb_id);

    constexpr uint32_t in1_transpose_tile = false;

    constexpr bool spill = num_blocks_inner_dim > 1;

    compute_kernel_hw_startup<SrcOrder::Reverse>(in0_dfb_id, in1_dfb_id, mm_partials_dfb_id);
    matmul_block_init(in0_dfb_id, in1_dfb_id, in1_transpose_tile, out_subblock_w, out_subblock_h, in0_block_w);
    for (uint32_t b = 0; b < batch; b++) {
        // run code from the in1 reader: 0 skip, 1 single, 2 first of a run, 3 middle, 4 last
        uint32_t run_code = 1;
        UNPACK(run_code = mailbox_read(ckernel::ThreadId::BriscThreadId);)
        MATH(run_code = mailbox_read(ckernel::ThreadId::BriscThreadId);)
        PACK(run_code = mailbox_read(ckernel::ThreadId::BriscThreadId);)
        if (run_code == 0) {
            continue;
        }
        const bool run_first = (run_code == 1) || (run_code == 2);
        const bool run_last = (run_code == 1) || (run_code == 4);
        if (run_first) {
            // the whole expert block (all K-blocks) becomes resident in cb_in1 for this run
            in1_dfb.wait_front(num_blocks_inner_dim * in1_block_num_tiles);
        }

        for (uint32_t bh = 0; bh < num_blocks_h_dim; ++bh) {
            for (uint32_t bw = 0; bw < num_blocks_w_dim; ++bw) {
                bool enable_reload = false;

                if constexpr (batch > 1 || num_blocks_h_dim > 1 || num_blocks_w_dim > 1) {
                    PACK((pack_reconfig_data_format(mm_partials_dfb_id)));
                }

                for (uint32_t block = 0; block < num_blocks_inner_dim; block++) {
                    bool last_out = block == (num_blocks_inner_dim - 1);
                    in0_dfb.wait_front(in0_block_num_tiles);
                    const uint32_t in1_resident_block_offset = block * in1_block_num_tiles;

                    int in0_index_subblock_offset = 0;
                    for (uint32_t in0_subblock = 0; in0_subblock < in0_num_subblocks; in0_subblock++) {
                        int in1_index_subblock_offset = 0;
                        for (uint32_t in1_subblock = 0; in1_subblock < in1_num_subblocks; in1_subblock++) {
                            tile_regs_acquire();
                            if (enable_reload) {
                                reload_from_cb_to_dst(
                                    in0_dfb_id,
                                    in1_dfb_id,
                                    mm_partials_dfb_id,
                                    in1_transpose_tile,
                                    out_subblock_num_tiles,
                                    out_subblock_w,
                                    out_subblock_h,
                                    in0_block_w);
                            }

                            // Compute output sub-block
                            uint32_t dst_index =
                                0;  // start at 0, each call to matmul_block internally increments dst_index
                            uint32_t in0_index = in0_index_subblock_offset;  // offset into in0 block
                            uint32_t in1_index = in1_resident_block_offset + in1_index_subblock_offset;  // resident set
                            // inner dim that we accumulate is the inner dim of in0/in1, which is in0_block_w
                            for (uint32_t inner_dim_idx = 0; inner_dim_idx < in0_block_w; ++inner_dim_idx) {
                                // matmul outer product of (out_subblock_h x out_subblock_w) tiles that fill dst
                                // accumulation is done by iterating matmul_block across inner dim
                                // in0_block_w is passed as innder dim (kt) to matmul_block, internally used to stride
                                // in0
                                matmul_block(
                                    in0_dfb_id,
                                    in1_dfb_id,
                                    in0_index,
                                    in1_index,
                                    dst_index,
                                    in1_transpose_tile,
                                    out_subblock_w,
                                    out_subblock_h,
                                    in0_block_w);
                                in0_index++;               // stride right by 1
                                in1_index += in1_block_w;  // to stride down by 1 need to stride by in_per_core_w
                                                           // (should be called in1_block_w)
                            }

                            if (last_out) {
                                tile_regs_commit();
                                mm_out_dfb.reserve_back(out_subblock_num_tiles);

                                tile_regs_wait();

#if defined FP32_DEST_ACC_EN or defined PACKER_L1_ACC
                                PACK((pack_reconfig_data_format(mm_out_dfb_id)));
#endif

#ifdef PACKER_L1_ACC
                                PACK((llk_pack_reconfig_l1_acc(0)));
#endif
                                uint32_t start_dst_index = 0;
                                pack_block(start_dst_index, mm_out_dfb_id, out_subblock_num_tiles);

                                tile_regs_release();
                                mm_out_dfb.push_back(out_subblock_num_tiles);

                            } else {
                                tile_regs_commit();
                                mm_partials_dfb.reserve_back(out_subblock_num_tiles);
                                tile_regs_wait();

#ifdef PACKER_L1_ACC
                                if (block == 0) {  // no accumulation for first iteration
                                    PACK((llk_pack_reconfig_l1_acc(0)));
                                } else if (block == 1) {
                                    PACK((llk_pack_reconfig_l1_acc(1)));
                                }
#endif

                                uint32_t start_dst_index = 0;
                                pack_block(start_dst_index, mm_partials_dfb_id, out_subblock_num_tiles);

                                tile_regs_release();
                                mm_partials_dfb.push_back(out_subblock_num_tiles);
                            }

                            in1_index_subblock_offset += out_subblock_w;
                        }
                        in0_index_subblock_offset += in0_subblock_num_tiles;
                    }

#ifdef PACKER_L1_ACC
                    // Last iteration does spill and reload to output buffer
                    if (block < num_blocks_inner_dim - 2) {
                        for (uint32_t s = 0; s < out_block_num_tiles; s += out_subblock_num_tiles) {
                            mm_partials_dfb.wait_front(out_subblock_num_tiles);
                            mm_partials_dfb.pop_front(out_subblock_num_tiles);
                        }
                    }
                    if (block == num_blocks_inner_dim - 2) {
                        enable_reload = true;
                    }  // reload when last iteration
#else
                    if constexpr (spill) {
                        enable_reload = true;
                    }
#endif

                    in0_dfb.pop_front(in0_block_num_tiles);
                }
                if (run_last) {
                    in1_dfb.pop_front(num_blocks_inner_dim * in1_block_num_tiles);
                }

#ifdef UNTILIZE_CHUNK_W
                {
                    reconfig_data_format_srca(in1_dfb_id, mm_partials_dfb_id);
#if defined FP32_DEST_ACC_EN or defined PACKER_L1_ACC
                    PACK((pack_reconfig_data_format(out_dfb_id)));
#endif
#ifdef PACKER_L1_ACC
                    PACK((llk_pack_reconfig_l1_acc(0)));
#endif
                    // The pack-untilize path is exact up to 16 tiles of row width (2026-10-01: 24+ tiles came out wrong
                    // on Blackhole), so a wide output block is untilized in chunks of UNTILIZE_CHUNK_W tiles, each
                    // pushed as its own row-major [32, chunk] piece; the writer stores them as partial sticks.
                    constexpr uint32_t untilize_chunk_w = UNTILIZE_CHUNK_W;
                    constexpr uint32_t subblocks_per_chunk = untilize_chunk_w / out_subblock_w;
                    constexpr uint32_t n_chunks = out_block_w / untilize_chunk_w;
                    static_assert(untilize_chunk_w % out_subblock_w == 0 && out_block_w % untilize_chunk_w == 0);
                    pack_untilize_dest_init<out_subblock_w, untilize_chunk_w>(out_dfb_id);
                    copy_init(mm_partials_dfb_id);
                    for (uint32_t in0_subblock_i = 0; in0_subblock_i < in0_num_subblocks; ++in0_subblock_i) {
                        DataflowBuffer interm_dfb(mm_partials_dfb_id);
                        DataflowBuffer out_dfb_u(out_dfb_id);
                        const uint32_t row_tiles = out_subblock_num_tiles * in1_num_subblocks;
                        interm_dfb.wait_front(row_tiles);
                        for (uint32_t c = 0; c < n_chunks; ++c) {
                            for (uint32_t h = 0; h < out_subblock_h; ++h) {
                                out_dfb_u.reserve_back(untilize_chunk_w);
                                for (uint32_t n = 0; n < subblocks_per_chunk; ++n) {
                                    const uint32_t sb = c * subblocks_per_chunk + n;
                                    tile_regs_acquire();
                                    for (uint32_t w = 0; w < out_subblock_w; ++w) {
                                        copy_tile(
                                            mm_partials_dfb_id,
                                            sb * out_subblock_num_tiles + h * out_subblock_w + w,
                                            w);
                                    }
                                    tile_regs_commit();
                                    tile_regs_wait();
                                    pack_untilize_dest<out_subblock_w, untilize_chunk_w>(out_dfb_id, 1, n);
                                    tile_regs_release();
                                }
                                out_dfb_u.push_back(untilize_chunk_w);
                            }
                        }
                        interm_dfb.pop_front(row_tiles);
                    }
                    pack_untilize_uninit(mm_partials_dfb_id);
                }
#endif
                if constexpr (batch > 1 || num_blocks_w_dim > 1 || num_blocks_h_dim > 1) {
                    // reconfigure unpacker df for src A
                    reconfig_data_format_srca(mm_partials_dfb_id, in1_dfb_id);
                    // reconfigure init for matmul
                    matmul_block_init(
                        in0_dfb_id, in1_dfb_id, in1_transpose_tile, out_subblock_w, out_subblock_h, in0_block_w);
                }
            }
        }
    }
}
