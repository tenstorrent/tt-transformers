// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// MoE dispatch gather, reader: for every work item (group g, column block cs) gather the group's 32 source rows
// (columns [cs*Hc, (cs+1)*Hc)) into one row-major block (32 x Hc bf16) for the compute kernel to tilize. A padding
// group (group_ids[g] >= num_experts) produces a zero block without any row read, and a 0 flag for the writer.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t rows_addr = get_arg_val<uint32_t>(0);
    const uint32_t src_addr = get_arg_val<uint32_t>(1);
    const uint32_t gid_addr = get_arg_val<uint32_t>(2);
    const uint32_t item_start = get_arg_val<uint32_t>(3);
    const uint32_t item_count = get_arg_val<uint32_t>(4);

    constexpr uint32_t num_experts = get_compile_time_arg_val(0);
    constexpr uint32_t Wt_c = get_compile_time_arg_val(1);
    constexpr uint32_t col_splits = get_compile_time_arg_val(2);
    constexpr uint32_t row_bytes = get_compile_time_arg_val(3);  // Hc * 2
    constexpr uint32_t G = get_compile_time_arg_val(4);
    constexpr uint32_t cb_in = get_compile_time_arg_val(5);
    constexpr uint32_t cb_src = get_compile_time_arg_val(6);
    constexpr uint32_t cb_gid = get_compile_time_arg_val(7);
    constexpr uint32_t cb_flag = get_compile_time_arg_val(8);
    constexpr auto rows_args = TensorAccessorArgs<9>();
    constexpr auto src_args = TensorAccessorArgs<rows_args.next_compile_time_args_offset()>();
    constexpr auto gid_args = TensorAccessorArgs<src_args.next_compile_time_args_offset()>();

    Noc noc;
    const auto s_rows = TensorAccessor(rows_args, rows_addr);
    const auto s_src = TensorAccessor(src_args, src_addr);
    const auto s_gid = TensorAccessor(gid_args, gid_addr);
    DataflowBuffer dfb_in(cb_in);
    DataflowBuffer dfb_flag(cb_flag);
    const uint32_t src_l1 = DataflowBuffer(cb_src).get_write_ptr();
    const uint32_t gid_l1 = DataflowBuffer(cb_gid).get_write_ptr();
    volatile tt_l1_ptr uint32_t* src = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src_l1);
    volatile tt_l1_ptr uint16_t* gid = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(gid_l1);

    // the whole group-id stick once (G uint16, a few KB)
    noc.async_read(s_gid, CoreLocalMem<uint32_t>(gid_l1), G * 2, {.page_id = 0, .offset_bytes = 0}, {});
    noc.async_read_barrier();

    for (uint32_t it = 0; it < item_count; ++it) {
        const uint32_t item = item_start + it;
        const uint32_t g = item / col_splits;
        const uint32_t cs = item % col_splits;
        const bool real = gid[g] < num_experts;

        dfb_flag.reserve_back(1);
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dfb_flag.get_write_ptr()) = real ? 1u : 0u;
        dfb_flag.push_back(1);

        dfb_in.reserve_back(Wt_c);
        const uint32_t base = dfb_in.get_write_ptr();
        if (real) {
            // this group's 32 source token ids: 128 bytes out of the single [1, G*32] stick
            noc.async_read(
                s_src, CoreLocalMem<uint32_t>(src_l1), 32 * 4, {.page_id = 0, .offset_bytes = g * 32 * 4}, {});
            noc.async_read_barrier();
            for (uint32_t r = 0; r < 32; ++r) {
                noc.async_read(
                    s_rows,
                    CoreLocalMem<uint32_t>(base + r * row_bytes),
                    row_bytes,
                    {.page_id = src[r], .offset_bytes = cs * row_bytes},
                    {});
            }
            noc.async_read_barrier();
        } else {
            volatile tt_l1_ptr uint32_t* z = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base);
            for (uint32_t w = 0; w < 32 * row_bytes / 4; ++w) {
                z[w] = 0;
            }
        }
        dfb_in.push_back(Wt_c);
    }
}
