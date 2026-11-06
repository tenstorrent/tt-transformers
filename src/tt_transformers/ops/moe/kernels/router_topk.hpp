// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Softmax top-k router for data-movement cores (no FPU). Top-k by order-preserving integer keys of the BF16 logits (ties
// to the lower id), softmax over the selected logits in 64-bit fixed point (exp2 = integer shift x degree-5 polynomial,
// ~1e-5 relative), rounded to BF16. Softmax over all experts -> top-k -> renormalize equals top-k of the logits ->
// softmax over the k selected (the global partition function cancels).

#pragma once

#include <stdint.h>

namespace moe_topk {

// BF16 bits -> Q16.16 fixed point (|x| < 2^14).
inline int32_t bf16_to_q16(uint16_t b) {
    const int32_t e = (b >> 7) & 0xFF;
    if (e == 0) {
        return 0;
    }
    const int32_t m = (b & 0x7F) | 0x80;
    const int32_t shift = e - 118;  // value = m * 2^(e - 134); in Q16: m << (e - 118)
    int32_t v = shift >= 0 ? (shift > 22 ? (m << 22) : (m << shift)) : (shift < -31 ? 0 : (m >> -shift));
    return (b & 0x8000) ? -v : v;
}

// 2^frac(t) in Q16 ([65536, 131072)) and floor(t) for t = d * log2(e), d <= 0 in Q16: exp(d) = p * 2^(n - 16).
inline uint32_t exp2_parts(int32_t d, int32_t& n) {
    const int32_t t = static_cast<int32_t>((static_cast<int64_t>(d) * 94548) >> 16);  // d * log2(e), Q16
    n = t >> 16;                                                                      // floor
    const uint32_t f = static_cast<uint32_t>(t) & 0xFFFF;                             // [0, 1) in Q16
    uint32_t p = 87;                                                                  // 2^f, Horner, Q16
    p = 630 + ((p * f) >> 16);
    p = 3638 + ((p * f) >> 16);
    p = 15743 + ((p * f) >> 16);
    p = 45426 + ((p * f) >> 16);
    p = 65536 + ((p * f) >> 16);
    return p;
}

// Unsigned fixed point with `frac` fractional bits, value in [0, 1] -> BF16 bits, round to nearest.
inline uint16_t fixed_to_bf16(uint64_t w, int32_t frac) {
    if (w == 0) {
        return 0;
    }
    const uint32_t hi = static_cast<uint32_t>(w >> 32);
    int32_t msb = hi ? 63 - __builtin_clz(hi) : 31 - __builtin_clz(static_cast<uint32_t>(w));
    const int32_t shift = msb - 7;
    uint64_t m = shift > 0 ? (w + (1ull << (shift - 1))) >> shift : w << -shift;
    if (m >= 0x100) {  // rounding carried into a new bit
        m >>= 1;
        msb += 1;
    }
    return static_cast<uint16_t>(((127 + msb - frac) << 7) | (static_cast<uint32_t>(m) & 0x7F));
}

// sigmoid(x) of a BF16 logit -> BF16 bits: e = exp(-|x|) in Q32, 1 / (1 + e) or e / (1 + e) in Q27.
inline uint16_t sigmoid_bf16(uint16_t b) {
    const int32_t v = bf16_to_q16(b);
    int32_t n;
    const uint32_t p = exp2_parts(v < 0 ? v : -v, n);
    const uint64_t e = n < -32 ? 0 : (n < -16 ? static_cast<uint64_t>(p >> (-16 - n)) : static_cast<uint64_t>(p) << (16 + n));
    const uint64_t den = (1ull << 32) + e;
    return fixed_to_bf16(((v < 0 ? e : (1ull << 32)) << 27) / den, 27);
}

// Top-k (sorted descending, ties to the lower id) of the BF16 values in 32-bit words [w0, w1) (value ids 2 w0 ..):
// keys[k] (order keys, 0 = empty) and ids[k].
template <uint32_t topk>
inline void select_range(const uint32_t* words, uint32_t w0, uint32_t w1, uint32_t* keys_out, uint32_t* ids_out) {
    // keys / ids live in registers (fully unrolled constant indexing; the kernel stack is in L1); a new value
    // bubbles up from the last slot with compare-swaps. The running k-th largest key is keys[topk - 1].
    uint32_t keys[topk], ids[topk];
#pragma GCC unroll 16
    for (uint32_t j = 0; j < topk; ++j) {
        keys[j] = 0;
        ids[j] = 0;
    }
    auto push = [&](uint32_t key, uint32_t id) __attribute__((always_inline)) {
        keys[topk - 1] = key;
        ids[topk - 1] = id;
#pragma GCC unroll 16
        for (uint32_t j = topk - 1; j > 0; --j) {
            if (keys[j] > keys[j - 1]) {
                const uint32_t tk = keys[j], ti = ids[j];
                keys[j] = keys[j - 1];
                ids[j] = ids[j - 1];
                keys[j - 1] = tk;
                ids[j - 1] = ti;
            }
        }
    };
#pragma GCC unroll 1
    for (uint32_t i = w0; i < w1; ++i) {
        const uint32_t v = words[i];
        // both halves' order keys at once: flip the sign bit, and the magnitude bits of negative values
        const uint32_t sgn = (v >> 15) & 0x00010001u;
        const uint32_t kk = v ^ 0x80008000u ^ ((sgn << 15) - sgn);
        const uint32_t k0 = kk & 0xFFFF, k1 = kk >> 16;
        if (k0 > keys[topk - 1]) {
            push(k0, 2 * i);
        }
        if (k1 > keys[topk - 1]) {
            push(k1, 2 * i + 1);
        }
    }
#pragma GCC unroll 16
    for (uint32_t j = 0; j < topk; ++j) {
        keys_out[j] = keys[j];
        ids_out[j] = ids[j];
    }
}

// keys / ids <- the top-k of (keys, ids) and (keys2, ids2), both sorted; on equal keys the first list (lower ids) wins.
template <uint32_t topk>
inline void merge(uint32_t* keys, uint32_t* ids, const uint32_t* keys2, const uint32_t* ids2) {
    uint32_t mk[topk], mi[topk];
    uint32_t a = 0, b = 0;
    for (uint32_t j = 0; j < topk; ++j) {
        if (keys[a] >= keys2[b]) {
            mk[j] = keys[a];
            mi[j] = ids[a++];
        } else {
            mk[j] = keys2[b];
            mi[j] = ids2[b++];
        }
    }
    for (uint32_t j = 0; j < topk; ++j) {
        keys[j] = mk[j];
        ids[j] = mi[j];
    }
}

template <uint32_t topk>
inline void softmax_selected(const uint32_t* words, const uint32_t* ids, uint16_t* weights) {
    // softmax over the selected: e_j = exp(v_j - v_0) = p_j 2^(n_j - 16) in Q32 (p_j << (16 + n_j), n_j <= 0;
    // terms below 2^-16 of the largest dropped); weights in Q27 = e_j * (2^59 / sum) >> 32 (one 64-bit division)
    const uint16_t* vals = reinterpret_cast<const uint16_t*>(words);
    const int32_t v0 = bf16_to_q16(vals[ids[0]]);
    uint64_t e[topk];
    uint64_t sum = 0;
    for (uint32_t j = 0; j < topk; ++j) {
        int32_t nj;
        const uint32_t p = exp2_parts(bf16_to_q16(vals[ids[j]]) - v0, nj);
        e[j] = nj < -16 ? 0 : static_cast<uint64_t>(p) << (16 + nj);
        sum += e[j];
    }
    const uint64_t inv = (1ull << 59) / sum;  // sum in [2^32, 2^36)
    for (uint32_t j = 0; j < topk; ++j) {
        weights[j] = fixed_to_bf16((e[j] * inv) >> 32, 27);
    }
}

// The logits row as the routing kernels read it: row 0 of each [32, 32] BF16 TILE page, faces 0 and 1 back to back.
// Column E holds the shared-expert gate logit, so the row spans E / 32 + 1 tiles.
constexpr uint32_t LG_PAGE = 2048;
constexpr uint32_t LG_HALF = 32;
constexpr uint32_t LG_FACE1 = 512;
constexpr uint32_t LG_TILE_L1 = 2 * LG_HALF;  // bytes per tile of the gathered row
constexpr uint32_t lg_tiles(uint32_t e) { return e / 32 + 1; }

// Router of one row: top-k of experts [e0, e1) of the gathered row `lg`.
template <uint32_t topk>
inline void route_select(uint32_t lg, uint32_t e0, uint32_t e1, uint32_t* keys, uint32_t* ids) {
    select_range<topk>(reinterpret_cast<const uint32_t*>(lg), e0 / 2, e1 / 2, keys, ids);
}
// Routing weights of the selected (BF16 bits); returns the shared-expert weight, sigmoid of column E.
template <uint32_t topk, uint32_t num_experts>
inline uint16_t route_weights(uint32_t lg, const uint32_t* ids, uint16_t* w) {
    softmax_selected<topk>(reinterpret_cast<const uint32_t*>(lg), ids, w);
    return sigmoid_bf16(reinterpret_cast<const uint16_t*>(lg)[num_experts]);
}

// ---- Expert parallelism (EP_EXPERTS = experts per die; Qwen3-Coder-30B): each die holds the whole experts
// [off, off + EP_EXPERTS) and keeps its window of the global top-k. off comes from a per-device tensor (one DRAM page in
// bank 0, the same address on every die), so the programs' runtime args stay identical across the mesh.
// DRAM -> L1 reads keep the source's 64-byte alignment: the offset lands at the next 64-byte boundary of its scratch
// (callers reserve 128 bytes from the scratch pointer: the alignment padding and the 64-byte read).
constexpr uint32_t align64(uint32_t a) { return (a + 63) & ~63u; }
#ifdef KERNEL_BUILD
inline uint32_t read_ep_offset(uint32_t addr, uint32_t l1_128) {
    const uint32_t dst = align64(l1_128);
    noc_async_read(get_noc_addr_from_bank_id<true>(0, addr), dst, 64);
    noc_async_read_barrier();
    invalidate_l1_cache();
    return *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst);
}
#endif
// ids / w (top-k order) -> this die's window, compacted in place as local ids; returns the count
template <uint32_t topk>
inline uint32_t local_window(uint32_t* ids, uint16_t* w, uint32_t off, uint32_t n_local) {
    uint32_t n = 0;
    for (uint32_t j = 0; j < topk; ++j) {
        if (ids[j] - off < n_local) {  // unsigned: ids below the window wrap past it
            ids[n] = ids[j] - off;
            w[n] = w[j];
            ++n;
        }
    }
    return n;
}

}  // namespace moe_topk
