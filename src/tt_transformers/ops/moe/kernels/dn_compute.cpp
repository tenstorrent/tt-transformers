// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Down projection + weighted expert sum (TRISC) of the fused MoE decode (ops/moe/expert_down.py ExpertDown):
// out[:, cols] = act (1 x (k + 1) K, slot-major) . Wstack ((k + 1) K x cols, row-major), one custom_mm call
// (1x32 activation tiles, LoFi, FP32 accumulation), packed as `cols` 1x32 tiles.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/experimental/custom_mm.h"
#include "api/compute/pack.h"

void kernel_main() {
    constexpr uint32_t cb_in0 = get_compile_time_arg_val(0);
    constexpr uint32_t cb_w = get_compile_time_arg_val(1);
    constexpr uint32_t cb_out = get_compile_time_arg_val(2);
    constexpr uint32_t kt = get_compile_time_arg_val(3);  // (k + 1) * K tiles
    constexpr uint32_t cols = get_compile_time_arg_val(4);

    constexpr bool transpose = false;
    constexpr bool split_acc = true;
    constexpr bool dense_packing = false;
    custom_mm_block_init<transpose, split_acc, dense_packing>(cb_in0, cb_w, cb_out, cols);

#ifdef EP_EXPERTS
    // this die's window of the top-k: live slots from BRISC (the down kernels' K is slots x K per slot); 0 -> no output
    uint32_t live = 0;
    UNPACK(live = ckernel::mailbox_read(ckernel::ThreadId::BriscThreadId));
    MATH(live = ckernel::mailbox_read(ckernel::ThreadId::BriscThreadId));
    PACK(live = ckernel::mailbox_read(ckernel::ThreadId::BriscThreadId));
    if (live == 0) {
        custom_mm_block_uninit<dense_packing>();
        return;
    }
    const uint32_t k_tiles = live * (kt / get_compile_time_arg_val(5));  // kt = slots x K per slot
#else
    constexpr uint32_t k_tiles = kt;
#endif
    cb_wait_front(cb_in0, k_tiles);
    cb_wait_front(cb_w, k_tiles * cols);
    tile_regs_acquire();
    custom_mm_block<true>(cb_in0, cb_w, 0, 0, 0, k_tiles, cols);
    tile_regs_commit();
    cb_pop_front(cb_w, k_tiles * cols);
    cb_pop_front(cb_in0, k_tiles);

    cb_reserve_back(cb_out, cols);
    tile_regs_wait();
    for (uint32_t c = 0; c < cols; ++c) {
        pack_tile(c, cb_out);
    }
    tile_regs_release();
    cb_push_back(cb_out, cols);
    custom_mm_block_uninit<dense_packing>();
}
