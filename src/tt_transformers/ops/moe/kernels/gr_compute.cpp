// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// MoE dispatch gather, compute: tilize every gathered row-major block (32 x Hc bf16 -> Wt_c tiles).
#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/tilize.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"

void kernel_main() {
    constexpr uint32_t Wt_c = get_compile_time_arg_val(0);
    constexpr uint32_t cb_in = get_compile_time_arg_val(1);
    constexpr uint32_t cb_out = get_compile_time_arg_val(2);
    const uint32_t item_count = get_arg_val<uint32_t>(0);

    compute_kernel_hw_startup(cb_in, cb_out);
    compute_kernel_lib::tilize<Wt_c, cb_in, cb_out>(item_count);
}
