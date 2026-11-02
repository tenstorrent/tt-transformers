// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Rows top-k, merge core (rows_topk.py RowsTopK): once the `ncores` scan cores have written their sorted
// per-row lists into the merge buffer (semaphore 0), merges them per row with a max-heap of the list heads into the
// row's K best, and writes the [32, K] outputs as ttnn.topk does: values BF16 TILE, indices UINT16 TILE (UINT32 when
// WIDE; the column on this device), rows >= B copies of row 0. List keys carry the scan core's local column: the merge
// orders by value, then by device column (core first column + local), so ties go to the lower column as before.
//
// runtime args: [merge_buf_addr, values_addr, indices_addr, first column of each scan core...]

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

static inline uint16_t from_key(uint32_t k) {
    return (k & 0x8000) ? (k & 0x7FFF) : (~k & 0xFFFF);
}

void kernel_main() {
    const uint32_t merge_addr = get_arg_val<uint32_t>(0);
    const uint32_t values_addr = get_arg_val<uint32_t>(1);
    const uint32_t indices_addr = get_arg_val<uint32_t>(2);

    constexpr uint32_t B = get_compile_time_arg_val(0);
    constexpr uint32_t K = get_compile_time_arg_val(1);
    constexpr uint32_t ncores = get_compile_time_arg_val(2);
    constexpr uint32_t cb_out = get_compile_time_arg_val(3);
    constexpr bool WIDE = get_compile_time_arg_val(4) != 0;
    constexpr auto v_args = TensorAccessorArgs<5>();
    constexpr auto i_args = TensorAccessorArgs<v_args.next_compile_time_args_offset()>();
    static_assert(K == 32, "one [32, 32] output tile");

    noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(0)), ncores);
    invalidate_l1_cache();

    const uint32_t* lists = reinterpret_cast<const uint32_t*>(merge_addr);
    uint32_t first[ncores];
    for (uint32_t c = 0; c < ncores; ++c) {
        first[c] = get_arg_val<uint32_t>(3 + c);
    }
    uint16_t* vt = reinterpret_cast<uint16_t*>(get_write_ptr(cb_out));
    uint16_t* it16 = vt + 1024;
    uint32_t* it32 = reinterpret_cast<uint32_t*>(vt + 1024);
    // (key, list) a ranks before (key, list) b: higher value, then lower device column (an exhausted list: key 0, last)
    auto col = [&](uint32_t k, uint32_t c) -> uint32_t { return k ? first[c] + (0xFFFF - (k & 0xFFFF)) : 0xFFFFFFFF; };
    auto before = [&](uint32_t ka, uint32_t ca, uint32_t kb, uint32_t cb) -> bool {
        return (ka >> 16) != (kb >> 16) ? (ka >> 16) > (kb >> 16) : col(ka, ca) < col(kb, cb);
    };
    auto pos = [](uint32_t r, uint32_t j) -> uint32_t {
        return ((r >= 16) * 2 + (j >= 16)) * 256 + (r % 16) * 16 + (j % 16);
    };
    {
        DeviceZoneScopedN("TK_MERGE");
        for (uint32_t b = 0; b < B; ++b) {
            // max-heap over the lists' heads: (key, list) packed as key in heap_k, list in heap_l
            uint32_t heap_k[ncores];
            uint16_t heap_l[ncores];
            uint16_t ptr[ncores];
            uint32_t n = 0;
            for (uint32_t c = 0; c < ncores; ++c) {
                ptr[c] = 0;
                uint32_t k = lists[(c * B + b) * K];
                uint32_t i = n++;
                while (i > 0 && before(k, c, heap_k[(i - 1) / 2], heap_l[(i - 1) / 2])) {
                    heap_k[i] = heap_k[(i - 1) / 2];
                    heap_l[i] = heap_l[(i - 1) / 2];
                    i = (i - 1) / 2;
                }
                heap_k[i] = k;
                heap_l[i] = c;
            }
            for (uint32_t j = 0; j < K; ++j) {
                const uint32_t k = heap_k[0];
                const uint32_t c = heap_l[0];
                vt[pos(b, j)] = from_key(k >> 16);
                if constexpr (WIDE) {
                    it32[pos(b, j)] = col(k, c);
                } else {
                    it16[pos(b, j)] = col(k, c);
                }
                // replace the root with the list's next head and sift down
                const uint32_t nk = ++ptr[c] < K ? lists[(c * B + b) * K + ptr[c]] : 0;
                uint32_t i = 0;
                while (true) {
                    uint32_t l = 2 * i + 1;
                    if (l >= n) {
                        break;
                    }
                    if (l + 1 < n && before(heap_k[l + 1], heap_l[l + 1], heap_k[l], heap_l[l])) {
                        ++l;
                    }
                    if (!before(heap_k[l], heap_l[l], nk, c)) {
                        break;
                    }
                    heap_k[i] = heap_k[l];
                    heap_l[i] = heap_l[l];
                    i = l;
                }
                heap_k[i] = nk;
                heap_l[i] = c;
            }
        }
        for (uint32_t r = B; r < 32; ++r) {
            for (uint32_t j = 0; j < K; ++j) {
                vt[pos(r, j)] = vt[pos(0, j)];
                if constexpr (WIDE) {
                    it32[pos(r, j)] = it32[pos(0, j)];
                } else {
                    it16[pos(r, j)] = it16[pos(0, j)];
                }
            }
        }
    }
    const auto s_v = TensorAccessor(v_args, values_addr, 2048);
    constexpr uint32_t i_bytes = WIDE ? 4096 : 2048;
    const auto s_i = TensorAccessor(i_args, indices_addr, i_bytes);
    noc_async_write(get_write_ptr(cb_out), s_v.get_noc_addr(0), 2048);
    noc_async_write(get_write_ptr(cb_out) + 2048, s_i.get_noc_addr(0), i_bytes);
    noc_async_write_barrier();
}
