// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Down-projection activation gather + output scatter (BRISC, NOC1) of the fused MoE decode
// (ops/moe/expert_down.py ExpertDown).
//
// 1. in0 = the [k + 1, I] BF16 activation buffer (row s = w_s * silu(gate) * up of slot s, written by the gate|up op)
//    as (k + 1) * I / 32 1x32 tiles, slot-major: the K order of the stacked down weights in cb_w.
// 2. Writes each 1x32 output tile (output column n = bank + banks * (c0 + c)) into row 0 of tile n of the [32, H]
//    BF16 TILE output (16 values at face 0 row 0, 16 at face 1 row 0); rows 1..31 are never written (zero buffer).
//
// NO_SHARED: routed experts only (activation rows 1 .. k; num_slots = k). EP_EXPERTS: the live count is the route
// record's (route[15], read here; runtime arg 4 = the route address), sent to the compute threads; 0 -> zeros out.
//
// runtime args: [act_addr, out_addr, bank, c0, (EP_EXPERTS) route_addr]

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "ckernel.h"
#include "ckernel_defs.h"

void kernel_main() {
    const uint32_t act_addr = get_arg_val<uint32_t>(0);
    const uint32_t out_addr = get_arg_val<uint32_t>(1);
    const uint32_t bank = get_arg_val<uint32_t>(2);
    const uint32_t c0 = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_in0 = get_compile_time_arg_val(0);
    constexpr uint32_t cb_out = get_compile_time_arg_val(1);
    constexpr uint32_t num_slots = get_compile_time_arg_val(2);
    constexpr uint32_t act_row_bytes = get_compile_time_arg_val(3);
    constexpr uint32_t cols = get_compile_time_arg_val(4);
    constexpr uint32_t banks = get_compile_time_arg_val(5);
    constexpr auto act_args = TensorAccessorArgs<6>();
    constexpr auto out_args = TensorAccessorArgs<act_args.next_compile_time_args_offset()>();

#ifdef NO_SHARED
    constexpr uint32_t S0 = 1;  // no shared expert: slots 1 .. (routed only)
#else
    constexpr uint32_t S0 = 0;
#endif
    constexpr uint32_t kin = num_slots * act_row_bytes / 64;

    const auto s_act = TensorAccessor(act_args, act_addr, act_row_bytes);
    const auto s_out = TensorAccessor(out_args, out_addr, 2048);

#ifdef EP_EXPERTS
    constexpr auto route_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();
    const auto s_route = TensorAccessor(route_args, get_arg_val<uint32_t>(4), 64);
    uint32_t live = 0;
    {
        cb_reserve_back(cb_in0, kin);  // the route record lands in the (still unused) activation buffer
        const uint32_t rec = get_write_ptr(cb_in0);
        noc_async_read(s_route.get_noc_addr(0), rec, 64);
        noc_async_read_barrier();
        invalidate_l1_cache();
        live = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(rec)[15];
    }
    ckernel::mailbox_write(ckernel::ThreadId::UnpackThreadId, live);
    ckernel::mailbox_write(ckernel::ThreadId::MathThreadId, live);
    ckernel::mailbox_write(ckernel::ThreadId::PackThreadId, live);
    if (live == 0) {  // none of the top-k on this die: its partial sum is zero
        const uint32_t zeros = MEM_ZEROS_BASE;
        for (uint32_t c = 0; c < cols; ++c) {
            const uint32_t n = bank + banks * (c0 + c);
            noc_async_write(zeros, s_out.get_noc_addr(n, 0), 32);
            noc_async_write(zeros, s_out.get_noc_addr(n, 512), 32);
        }
        noc_async_write_barrier();
        return;
    }
    const uint32_t n_slots = live;
#else
    constexpr uint32_t n_slots = num_slots;
#endif
    cb_reserve_back(cb_in0, kin);
    const uint32_t in0 = get_write_ptr(cb_in0);
    for (uint32_t s = 0; s < n_slots; ++s) {
        noc_async_read(s_act.get_noc_addr(S0 + s), in0 + s * act_row_bytes, act_row_bytes);
    }
    noc_async_read_barrier();
    cb_push_back(cb_in0, n_slots * act_row_bytes / 64);

    cb_wait_front(cb_out, cols);
    const uint32_t src = get_read_ptr(cb_out);
    for (uint32_t c = 0; c < cols; ++c) {
        const uint32_t n = bank + banks * (c0 + c);
        noc_async_write(src + c * 64, s_out.get_noc_addr(n, 0), 32);
        noc_async_write(src + c * 64 + 32, s_out.get_noc_addr(n, 512), 32);
    }
    noc_async_write_barrier();
    cb_pop_front(cb_out, cols);
}
