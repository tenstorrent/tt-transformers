// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// MoE un-dispatch, compute: per work item tilize the K gathered row-major blocks and add them up (bf16, slot order
// 0..K-1, the same association as the k successive ttnn.add calls it replaces). Slot 0 is tilized straight into
// accumulator A; every further slot is tilized into `til` and added into the other accumulator (ping-pong); the LAST
// add packs into the dedicated output buffer `out` (the writer must never see an intermediate sum: a shared buffer with
// two consumers desynchronised the pointers and hung the two-item cores). The tilize goes through the kernel library
// helper (the production tilize op's path: register reconfiguration + fast tilize + init/uninit per block); the add
// re-configures the unpack/pack registers for the tiled operands.

#include <cstdint>

#include "api/compute/cb_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/reg_api.h"
#include "api/compute/tilize.h"
#include "api/compute/tile_move_copy.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"

void kernel_main() {
    constexpr uint32_t Wt_c = get_compile_time_arg_val(0);
    constexpr uint32_t K = get_compile_time_arg_val(1);
    constexpr uint32_t cb_in = get_compile_time_arg_val(2);
    constexpr uint32_t cb_til = get_compile_time_arg_val(3);
    constexpr uint32_t cb_acc_a = get_compile_time_arg_val(4);
    constexpr uint32_t cb_acc_b = get_compile_time_arg_val(5);
    constexpr uint32_t cb_out = get_compile_time_arg_val(6);
    const uint32_t item_count = get_arg_val<uint32_t>(0);
    constexpr uint32_t dst_tiles = 8;

    compute_kernel_hw_startup(cb_in, cb_til, cb_acc_a);

    for (uint32_t it = 0; it < item_count; ++it) {
        if constexpr (K == 1) {
            compute_kernel_lib::tilize<Wt_c, cb_in, cb_out>(1);
            continue;
        }
        compute_kernel_hw_startup(cb_in, cb_til, cb_acc_a);  // back from the binary-add state to the tilize state
        compute_kernel_lib::tilize<Wt_c, cb_in, cb_acc_a>(1);
        uint32_t cur = cb_acc_a;
        uint32_t nxt = cb_acc_b;
        for (uint32_t j = 1; j < K; ++j) {
            if (j == K - 1) {
                nxt = cb_out;
            }
            compute_kernel_lib::tilize<Wt_c, cb_in, cb_til>(1);
            // tilize leaves the unpack/math/pack state configured for the tilize path; the binary add needs the full
            // start-up sequence again (same mid-kernel re-init the layernorm kernel does after its tilize, tt-metal
            // #52395)
            compute_kernel_hw_startup(cur, cb_til, nxt);
            add_init(cur, cb_til);
            cb_wait_front(cur, Wt_c);
            cb_wait_front(cb_til, Wt_c);
            cb_reserve_back(nxt, Wt_c);
            for (uint32_t i0 = 0; i0 < Wt_c; i0 += dst_tiles) {
                const uint32_t n = (Wt_c - i0) < dst_tiles ? (Wt_c - i0) : dst_tiles;
                tile_regs_acquire();
                for (uint32_t d = 0; d < n; ++d) {
                    add_tiles(cur, cb_til, i0 + d, i0 + d, d);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t d = 0; d < n; ++d) {
                    pack_tile(d, nxt);
                }
                tile_regs_release();
            }
            cb_pop_front(cur, Wt_c);
            cb_pop_front(cb_til, Wt_c);
            cb_push_back(nxt, Wt_c);
            const uint32_t tmp = cur;
            cur = nxt;
            nxt = tmp;  // (unused after the last add: cur is cb_out, which the writer drains)
        }
    }
}
