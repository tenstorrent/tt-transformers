# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Qwen3.8 Blackhole generic operations for stock TTNN 0.79.0."""

from .gdn_decode_step import gdn_decode_step
from .gdn_spec_step import gdn_spec_step
from .qkv_causal_conv1d_silu_tile import qkv_causal_conv1d_silu_tile

__all__ = ["gdn_decode_step", "gdn_spec_step", "qkv_causal_conv1d_silu_tile"]
