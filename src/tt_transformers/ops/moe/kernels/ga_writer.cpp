// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// MoE un-dispatch, writer: the accumulated Wt_c tiles of every work item go to output tiles (rt, cs*Wt_c + i).

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t item_start = get_arg_val<uint32_t>(1);
    const uint32_t item_count = get_arg_val<uint32_t>(2);

    constexpr uint32_t Wt = get_compile_time_arg_val(0);
    constexpr uint32_t Wt_c = get_compile_time_arg_val(1);
    constexpr uint32_t col_splits = get_compile_time_arg_val(2);
    constexpr uint32_t cb_final = get_compile_time_arg_val(3);  // dedicated output buffer (never an add intermediate)
    constexpr auto out_args = TensorAccessorArgs<4>();

    Noc noc;
    const auto s_out = TensorAccessor(out_args, out_addr);
    DataflowBuffer dfb_out(cb_final);
    const uint32_t tile_bytes = dfb_out.get_tile_size();

    for (uint32_t it = 0; it < item_count; ++it) {
        const uint32_t item = item_start + it;
        const uint32_t rt = item / col_splits;
        const uint32_t cs = item % col_splits;
        dfb_out.wait_front(Wt_c);
        const uint32_t tile0 = rt * Wt + cs * Wt_c;
        for (uint32_t i = 0; i < Wt_c; ++i) {
            noc.async_write(dfb_out, s_out, tile_bytes, {.offset_bytes = i * tile_bytes}, {.page_id = tile0 + i});
        }
        noc.async_write_barrier();
        dfb_out.pop_front(Wt_c);
    }
}
