// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// MoE un-dispatch, reader: for every work item (32-token tile rt, column block cs) and every slot j, gather the 32
// rows undispatch[t*K + j] (columns [cs*Hc, (cs+1)*Hc)) into one row-major block (32 x Hc bf16) for the compute
// kernel to tilize and accumulate. Tokens past T (last partial tile) and pairs routed to another device (rows from
// pad_row on: the trailing padding group, which the grouped matmul never writes) read as zeros.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t rows_addr = get_arg_val<uint32_t>(0);
    const uint32_t und_addr = get_arg_val<uint32_t>(1);
    const uint32_t item_start = get_arg_val<uint32_t>(2);
    const uint32_t item_count = get_arg_val<uint32_t>(3);

    constexpr uint32_t T = get_compile_time_arg_val(0);
    constexpr uint32_t K = get_compile_time_arg_val(1);
    constexpr uint32_t Wt_c = get_compile_time_arg_val(2);
    constexpr uint32_t col_splits = get_compile_time_arg_val(3);
    constexpr uint32_t row_bytes = get_compile_time_arg_val(4);  // Hc * 2
    constexpr uint32_t cb_in = get_compile_time_arg_val(5);
    constexpr uint32_t cb_und = get_compile_time_arg_val(6);
    constexpr uint32_t pad_row = get_compile_time_arg_val(7);
    constexpr auto rows_args = TensorAccessorArgs<8>();
    constexpr auto und_args = TensorAccessorArgs<rows_args.next_compile_time_args_offset()>();

    Noc noc;
    const auto s_rows = TensorAccessor(rows_args, rows_addr);
    const auto s_und = TensorAccessor(und_args, und_addr);
    DataflowBuffer dfb_in(cb_in);
    const uint32_t und_l1 = DataflowBuffer(cb_und).get_write_ptr();
    volatile tt_l1_ptr uint32_t* und = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(und_l1);

    for (uint32_t it = 0; it < item_count; ++it) {
        const uint32_t item = item_start + it;
        const uint32_t rt = item / col_splits;
        const uint32_t cs = item % col_splits;
        const uint32_t t0 = rt * 32;
        const uint32_t n_rows = (T - t0) < 32 ? (T - t0) : 32;
        // the destination rows of this token tile: K*n_rows uint32 out of the single [1, T*K] stick
        noc.async_read(
            s_und, CoreLocalMem<uint32_t>(und_l1), n_rows * K * 4, {.page_id = 0, .offset_bytes = t0 * K * 4}, {});
        noc.async_read_barrier();
        for (uint32_t j = 0; j < K; ++j) {
            dfb_in.reserve_back(Wt_c);
            const uint32_t base = dfb_in.get_write_ptr();
            for (uint32_t r = 0; r < n_rows; ++r) {
                const uint32_t src_row = und[r * K + j];
                if (src_row >= pad_row) {
                    volatile tt_l1_ptr uint32_t* z = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base + r * row_bytes);
                    for (uint32_t w = 0; w < row_bytes / 4; ++w) {
                        z[w] = 0;
                    }
                    continue;
                }
                noc.async_read(
                    s_rows,
                    CoreLocalMem<uint32_t>(base + r * row_bytes),
                    row_bytes,
                    {.page_id = src_row, .offset_bytes = cs * row_bytes},
                    {});
            }
            if (n_rows < 32) {
                volatile tt_l1_ptr uint32_t* z =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base + n_rows * row_bytes);
                for (uint32_t w = 0; w < (32 - n_rows) * row_bytes / 4; ++w) {
                    z[w] = 0;
                }
            }
            noc.async_read_barrier();
            dfb_in.push_back(Wt_c);
        }
    }
}
