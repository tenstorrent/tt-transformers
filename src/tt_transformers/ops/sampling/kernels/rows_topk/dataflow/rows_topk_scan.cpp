// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Rows top-k, scan core (rows_topk.py RowsTopK): reads rows 0 .. B-1 of its `nt` logits tiles (BF16 32x32,
// any TensorAccessor layout; B <= 16: faces 0 / 1 rows 0 .. B-1 only, 64-byte aligned), keeps each row's K best as a
// sorted list of packed keys (orderable BF16 << 16 | (0xFFFF - local column): one u32 compare orders by value, then lower
// column first; the local column counts from this core's first tile, so any vocab width fits), writes the lists into its slot of the merge buffer on the merge core and increments its semaphore.
//
// runtime args: [logits_addr, t0, nt, merge_buf_addr, slot]

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

static inline __attribute__((always_inline)) uint32_t order_key(uint32_t v) {
    return (v & 0x8000) ? (~v & 0xFFFF) : (v | 0x8000);
}

void kernel_main() {
    const uint32_t logits_addr = get_arg_val<uint32_t>(0);
    const uint32_t t0 = get_arg_val<uint32_t>(1);
    const uint32_t nt = get_arg_val<uint32_t>(2);
    const uint32_t merge_addr = get_arg_val<uint32_t>(3);
    const uint32_t slot = get_arg_val<uint32_t>(4);

    constexpr uint32_t B = get_compile_time_arg_val(0);
    constexpr uint32_t K = get_compile_time_arg_val(1);
    constexpr uint32_t cb_buf = get_compile_time_arg_val(2);
    constexpr uint32_t cb_list = get_compile_time_arg_val(3);
    constexpr uint32_t merge_x = get_compile_time_arg_val(4);
    constexpr uint32_t merge_y = get_compile_time_arg_val(5);
    constexpr auto lg_args = TensorAccessorArgs<6>();
    static_assert(B <= 16, "faces 0 / 1 only");
    constexpr uint32_t tile_bytes = 2048;
    constexpr uint32_t rows_bytes = ((B * 32 + 63) / 64) * 64;  // rows 0 .. B-1 of one face, 64-byte aligned
    constexpr uint32_t stride = 2 * rows_bytes;                  // per tile: face 0 rows, face 1 rows

    const uint32_t buf = get_write_ptr(cb_buf);
    {
        DeviceZoneScopedN("TK_READ");
        const auto s_lg = TensorAccessor(lg_args, logits_addr, tile_bytes);
        for (uint32_t t = 0; t < nt; ++t) {
            const uint64_t src = s_lg.get_noc_addr(t0 + t);
            noc_async_read(src, buf + t * stride, rows_bytes);
            noc_async_read(src + 512, buf + t * stride + rows_bytes, rows_bytes);
        }
        noc_async_read_barrier();
        invalidate_l1_cache();
    }

    uint32_t* lists = reinterpret_cast<uint32_t*>(get_write_ptr(cb_list));
    {
        DeviceZoneScopedN("TK_SCAN");
        for (uint32_t b = 0; b < B; ++b) {
            uint32_t top[K];
            for (uint32_t i = 0; i < K; ++i) {
                top[i] = 0;
            }
            uint32_t thresh = 0;
            auto consider = [&](uint32_t key) __attribute__((always_inline)) {
                uint32_t i = K - 1;
                while (i > 0 && top[i - 1] < key) {
                    top[i] = top[i - 1];
                    --i;
                }
                top[i] = key;
                thresh = top[K - 1];
            };
            for (uint32_t t = 0; t < nt; ++t) {
                const uint32_t col0 = 0xFFFF - t * 32;  // inverted local column of value 0 of the tile
                for (uint32_t f = 0; f < 2; ++f) {
                    const uint32_t* row = reinterpret_cast<const uint32_t*>(buf + t * stride + f * rows_bytes + b * 32);
                    const uint32_t c0 = col0 - f * 16;
                    for (uint32_t j = 0; j < 8; ++j) {
                        const uint32_t w = row[j];
                        const uint32_t k0 = (order_key(w & 0xFFFF) << 16) | (c0 - 2 * j);
                        const uint32_t k1 = (order_key(w >> 16) << 16) | (c0 - 2 * j - 1);
                        if (k0 > thresh) {
                            consider(k0);
                        }
                        if (k1 > thresh) {
                            consider(k1);
                        }
                    }
                }
            }
            for (uint32_t i = 0; i < K; ++i) {
                lists[b * K + i] = top[i];
            }
        }
    }
    noc_async_write(get_write_ptr(cb_list), get_noc_addr(merge_x, merge_y, merge_addr + slot * B * K * 4), B * K * 4);
    noc_async_write_barrier();
    noc_semaphore_inc(get_noc_addr(merge_x, merge_y, get_semaphore(0)), 1);
    noc_async_atomic_barrier();
}
