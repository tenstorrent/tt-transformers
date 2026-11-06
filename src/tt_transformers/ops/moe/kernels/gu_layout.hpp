// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Column layout of the fused MoE decode's gate|up weights (ops/moe/expert_gate_up.py gu_layout; keep the two in sync).
//
// A [1, E, K, N] TILE tensor interleaved over `banks` DRAM banks (page p in bank p % banks) with N = banks * m tiles
// keeps, per expert and bank b, the [kt, m] block of columns b, b + banks, ..., b + (m - 1) banks, row-major over
// (k, local column): one contiguous run of kt * m tiles. The packed [up | gate] weight has up column u and gate column
// u + N / 2, which is in the same bank when m is even and in bank b + banks / 2 when m is odd. A core group therefore
// reads, per slot, `nb` whole bank blocks (nb = 1: bank q; nb = 2: banks q and q + banks / 2, q < banks / 2) into DST
// tiles 0 .. nb * m - 1 (block h's local column c -> DST h * m + c) and owns `pairs` (up, gate) column pairs: the
// up columns among its blocks. The pattern is the same for every group q (q < banks / 2 when nb = 2).

#pragma once

#include <stdint.h>

#include <type_traits>

namespace gu_layout {

template <uint32_t m>
constexpr uint32_t blocks() {
    return m % 2 ? 2 : 1;
}
template <uint32_t m>
constexpr uint32_t tiles() {  // DST tiles of the matmul result
    return m * blocks<m>();
}
template <uint32_t m>
constexpr uint32_t pairs() {
    return tiles<m>() / 2;
}
// column of DST tile d relative to the group's first bank q
template <uint32_t m, uint32_t banks>
constexpr uint32_t col_off(uint32_t d) {
    return d < m ? banks * d : banks / 2 + banks * (d - m);
}
// DST tile of the i-th up column (ascending DST order)
template <uint32_t m, uint32_t banks>
constexpr uint32_t up_dst(uint32_t i) {
    for (uint32_t d = 0; d < tiles<m>(); ++d) {
        if (col_off<m, banks>(d) < banks * m / 2) {
            if (i == 0) {
                return d;
            }
            --i;
        }
    }
    return 0;
}
// DST tile of the gate column paired with the up column in DST tile d
template <uint32_t m, uint32_t banks>
constexpr uint32_t gate_dst(uint32_t d) {
    for (uint32_t g = 0; g < tiles<m>(); ++g) {
        if (col_off<m, banks>(g) == col_off<m, banks>(d) + banks * m / 2) {
            return g;
        }
    }
    return 0;
}
// global pair index (= up column) of the group's i-th pair
template <uint32_t m, uint32_t banks>
constexpr uint32_t pair_index(uint32_t q, uint32_t i) {
    return q + col_off<m, banks>(up_dst<m, banks>(i));
}

// f(std::integral_constant<uint32_t, i>) for i = 0 .. n - 1: compile-time DST indices for the SFPU calls (runtime
// indices computed by these helpers crashed the SFPI compiler: internal compiler error in gimple-rvtt-synth)
template <uint32_t n, uint32_t i = 0, typename F>
inline __attribute__((always_inline)) void static_for(F&& f) {
    if constexpr (i < n) {
        f(std::integral_constant<uint32_t, i>{});
        static_for<n, i + 1>(f);
    }
}

}  // namespace gu_layout
