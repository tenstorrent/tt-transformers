// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/dataflow/dataflow_api.h"

// Puts the read command buffer back the way the firmware's noc_init leaves it (static VC 1, transaction id 0). The
// readers here set a per-bank static VC and transaction ids; the firmware re-initializes the command buffers only when
// a kernel's NoC mode differs from the previous one, so the state otherwise outlives the kernel and every later kernel's
// reads on this core run on that VC (measured: 2-4 % slower prefill after the 9..32-row decode ops had run).
FORCE_INLINE void restore_read_cmd_buf() {
    noc_async_read_one_packet_set_state<true>(get_noc_addr(MEM_ZEROS_BASE), 0, 1);
    noc_async_read_set_trid(0);
}
