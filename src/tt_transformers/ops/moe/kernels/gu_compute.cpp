// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Expert gate|up matmul + SwiGLU (TRISC) of the fused MoE decode (ops/moe/expert_gate_up.py ExpertGateUp).
//
// Per slot this core owns (shared expert or a routed expert) and each column pair i of its group (gu_layout.hpp):
//   u = x . W[:, up col],  g = x . W[:, gate col]   (custom_mm over the group's [kt, m] bank blocks, ct = m; 1x32
//                                                    activation tiles, LoFi, FP32 accumulation)
//   act = w_s * silu(g) * u                         (SFPU on the result rows; w_s from BRISC via the mailbox)
// rounded to the 7 significant bits the down projection's LoFi custom_mm reads exactly, packed as one 1x32 tile per
// pair (pair order).

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/experimental/custom_mm.h"
#include "api/compute/pack.h"
#include "gu_layout.hpp"

// SFPU ops over the result rows only: a 1x32 matmul result occupies the first rows of DST faces 0 and 1
// (VectorMode::R, 2 iterations), as in the gpt-oss fused decode (models/demos/gpt_oss/tt/fused_decode/kernels).
constexpr uint32_t kRowIters = 2;
template <int OP>
ALWI void scalar_rows(uint32_t idst, uint32_t s) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_binop_with_scalar,
        (APPROX, OP, kRowIters, DST_ACCUM_MODE),
        idst,
        VectorMode::R,
        s));
}
ALWI void silu_rows(uint32_t idst) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE, DST_ACCUM_MODE, calculate_silu, (DST_ACCUM_MODE, kRowIters), idst, VectorMode::R));
}
ALWI void mul_rows(uint32_t a, uint32_t b, uint32_t out) {
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_sfpu_binary_mul,
        (APPROX, ckernel::BinaryOp::MUL, kRowIters, DST_ACCUM_MODE),
        a,
        b,
        out,
        VectorMode::R)));
}

#ifdef TRISC_MATH
namespace ckernel::sfpu {
// Round to 7 significant bits (6 explicit mantissa bits), half away from zero: the srcB operand width custom_mm
// multiplies at LoFi, so the down projection reads its activations exactly instead of truncated.
template <int ITERATIONS>
inline void round_sig7() {
    for (int d = 0; d < ITERATIONS; d++) {
        const sfpi::vFloat x = sfpi::dst_reg[0];
        sfpi::vInt v = sfpi::as<sfpi::vInt>(x);
        v = (v + 0x10000) & sfpi::vInt(0xFFFE0000);
        sfpi::dst_reg[0] = sfpi::as<sfpi::vFloat>(v);
        sfpi::dst_reg++;
    }
}
}  // namespace ckernel::sfpu
#endif

ALWI void round_rows(uint32_t idst) {
    MATH(SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, round_sig7, (kRowIters), idst, VectorMode::R));
}

void kernel_main() {
    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_w = get_compile_time_arg_val(1);
    constexpr uint32_t cb_act = get_compile_time_arg_val(2);
    constexpr uint32_t kt = get_compile_time_arg_val(3);
    constexpr uint32_t m = get_compile_time_arg_val(4);      // up|gate columns per bank and expert
    constexpr uint32_t banks = get_compile_time_arg_val(5);
    constexpr uint32_t nb = gu_layout::blocks<m>();
    constexpr uint32_t np = gu_layout::pairs<m>();

    constexpr bool transpose = false;
    constexpr bool split_acc = true;
    constexpr bool dense_packing = false;
    constexpr uint32_t ct = m;  // one bank block's columns
    custom_mm_block_init<transpose, split_acc, dense_packing>(cb_x, cb_w, cb_act, ct);

#ifdef EP_EXPERTS
    uint32_t num_slots = 0;  // this die's window of the top-k: the live slot count from BRISC
    UNPACK(num_slots = ckernel::mailbox_read(ckernel::ThreadId::BriscThreadId));
    MATH(num_slots = ckernel::mailbox_read(ckernel::ThreadId::BriscThreadId));
    PACK(num_slots = ckernel::mailbox_read(ckernel::ThreadId::BriscThreadId));
#else
    const uint32_t num_slots = get_arg_val<uint32_t>(0);  // this core's slot count
#endif

    cb_wait_front(cb_x, kt);
    for (uint32_t i = 0; i < num_slots; ++i) {
        tile_regs_acquire();
        gu_layout::static_for<nb>([&](auto h) {
            cb_wait_front(cb_w, kt * m);
            custom_mm_block<true>(cb_x, cb_w, 0, 0, decltype(h)::value * m, kt, m);
            cb_pop_front(cb_w, kt * m);
        });

        silu_tile_init();
        gu_layout::static_for<np>([&](auto p) {
            constexpr uint32_t u = gu_layout::up_dst<m, banks>(decltype(p)::value);
#ifdef GATE_FIRST
            silu_rows(u);  // [gate | up] packing: the first-half column is the gate
#else
            silu_rows(gu_layout::gate_dst<m, banks>(u));
#endif
        });
        mul_binary_tile_init();
        gu_layout::static_for<np>([&](auto p) {
            constexpr uint32_t u = gu_layout::up_dst<m, banks>(decltype(p)::value);
            mul_rows(u, gu_layout::gate_dst<m, banks>(u), u);
        });
        binop_with_scalar_tile_init();
        // the slot weight arrives once the router tail is done; the shared expert's matmuls run before that
        uint32_t score = 0;
        MATH(score = ckernel::mailbox_read(ckernel::ThreadId::BriscThreadId));
        gu_layout::static_for<np>([&](auto p) {
            scalar_rows<MUL_UNARY>(gu_layout::up_dst<m, banks>(decltype(p)::value), score);
        });
        gu_layout::static_for<np>([&](auto p) { round_rows(gu_layout::up_dst<m, banks>(decltype(p)::value)); });
        tile_regs_commit();

        cb_reserve_back(cb_act, np);
        tile_regs_wait();
        gu_layout::static_for<np>([&](auto p) { pack_tile(gu_layout::up_dst<m, banks>(decltype(p)::value), cb_act); });
        tile_regs_release();
        cb_push_back(cb_act, np);

        custom_mm_block_init_short<transpose, split_acc, dense_packing>(cb_x, cb_w, cb_act, ct);
    }
    cb_pop_front(cb_x, kt);
    custom_mm_block_uninit<dense_packing>();
}
