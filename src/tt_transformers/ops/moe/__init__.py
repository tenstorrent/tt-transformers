# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Mixture-of-experts decode operations (Blackhole): `fused_decode.FusedMoEDecode`, the one-token fused MoE decode
(router matmul, `expert_gate_up.ExpertGateUp`, `expert_down.ExpertDown`)."""
