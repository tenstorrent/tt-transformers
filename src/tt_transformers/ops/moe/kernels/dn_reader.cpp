// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Expert down-projection weight streamer (NCRISC, NOC0) of the fused MoE decode
// (ops/moe/expert_down.py ExpertDown).
//
// Streams straight from the ttnn down tensors [1, E, K, H] (TILE, DRAM interleaved; page p in bank p % banks). With
// H tiles a multiple of the bank count, bank b holds output columns b, b + banks, ... of every expert, and expert e's
// share is one contiguous run of K x m tiles (m = H tiles / banks), row-major over (k, local column). This core owns
// local columns [c0, c0 + cols) of bank b: per slot (0 = shared expert, s = routed expert ids[s - 1]) and K row, one
// read of `cols` tiles, all in flight at once. The slots are stacked along K in cb_w, so the compute does the weighted
// expert sum as one [1, (k + 1) K] x [(k + 1) K, cols] matmul (the routing weights are already in the activations).
//
// The routed ids come from the route record the gate|up op wrote.
//
// NO_SHARED: routed experts only (slots 1 .. k). EP_EXPERTS (with NO_SHARED): the route record's local ids, its count
// (route[15]) of them.
//
// runtime args: [w_addr, ws_addr, route_addr, bank, c0, vc]

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "noc_state.hpp"

void kernel_main() {
    const uint32_t w_addr = get_arg_val<uint32_t>(0);
    const uint32_t ws_addr = get_arg_val<uint32_t>(1);
    const uint32_t route_addr = get_arg_val<uint32_t>(2);
    const uint32_t bank = get_arg_val<uint32_t>(3);
    const uint32_t c0 = get_arg_val<uint32_t>(4);
    const uint32_t vc = get_arg_val<uint32_t>(5);

    constexpr uint32_t cb_w = get_compile_time_arg_val(0);
    constexpr uint32_t cb_scr = get_compile_time_arg_val(1);
    constexpr uint32_t kt = get_compile_time_arg_val(2);  // K tiles per expert (I / 32)
    constexpr uint32_t m = get_compile_time_arg_val(3);   // local columns per bank and expert
    constexpr uint32_t cols = get_compile_time_arg_val(4);
    constexpr uint32_t topk = get_compile_time_arg_val(5);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(6);
    constexpr auto route_args = TensorAccessorArgs<7>();

#ifdef NO_SHARED
    constexpr uint32_t S0 = 1;  // no shared expert: slots 1 .. (routed only)
#else
    constexpr uint32_t S0 = 0;
#endif
    constexpr uint32_t num_slots = topk + 1 - S0;
    constexpr uint32_t row_bytes = cols * tile_bytes;
    constexpr uint32_t slot_tiles = kt * cols;

    cb_reserve_back(cb_scr, 1);
    const uint32_t scr = get_write_ptr(cb_scr);
    uint32_t ids[topk];
    const auto s_route = TensorAccessor(route_args, route_addr, 64);
    noc_async_read(s_route.get_noc_addr(0), scr, 64);
    noc_async_read_barrier();
    invalidate_l1_cache();
    volatile tt_l1_ptr uint16_t* route = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scr);
    for (uint32_t j = 0; j < topk; ++j) {
        ids[j] = route[j];
    }
#ifdef EP_EXPERTS
    static_assert(S0 == 1, "expert parallelism: routed experts only");
    const uint32_t n_live = route[15];
#else
    constexpr uint32_t n_live = topk;
#endif

    const uint64_t routed_noc = get_noc_addr_from_bank_id<true>(bank, w_addr);
    const uint64_t shared_noc = get_noc_addr_from_bank_id<true>(bank, ws_addr);
    // every slot's rows in flight at once (the compute needs the whole stack), one barrier
    reset_noc_trid_barrier_counter(NOC_CLEAR_OUTSTANDING_REQ_MASK, noc_index);
    const uint32_t live_slots = num_slots - (topk - n_live);
    if (live_slots == 0) {  // EP: none of the top-k on this die (the writer writes zeros)
        restore_read_cmd_buf();
        return;
    }
    cb_reserve_back(cb_w, live_slots * slot_tiles);
    uint32_t l1 = get_write_ptr(cb_w);
    for (uint32_t s = S0; s < n_live + 1; ++s) {
        const uint64_t bank_noc = s == 0 ? shared_noc : routed_noc;
        const uint32_t e = s == 0 ? 0 : ids[s - 1];
        noc_async_read_one_packet_set_state<true>(bank_noc, row_bytes, vc);
        noc_async_read_set_trid(1);
        uint32_t src = (e * kt * m + c0) * tile_bytes;
        for (uint32_t k = 0; k < kt; ++k) {
            noc_async_read_one_packet_with_state_with_trid((uint32_t)bank_noc, src, l1, 1);
            src += m * tile_bytes;
            l1 += row_bytes;
        }
    }
    noc_async_read_barrier_with_trid(1);
    cb_push_back(cb_w, live_slots * slot_tiles);
    restore_read_cmd_buf();
}
