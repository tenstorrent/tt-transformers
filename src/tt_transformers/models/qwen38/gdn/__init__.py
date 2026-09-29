# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Gated DeltaNet (linear attention) for Qwen3.5-9B, split into config/weights/state/prefill/decode.

The orchestrating layer lives in ``gated_deltanet.py``; this package re-exports it
(and ``GDNConfig``) as the public API.
"""

from tt_transformers.models.qwen38.gdn.config import GDNConfig
from tt_transformers.models.qwen38.gdn.gated_deltanet import Qwen36GatedDeltaNet

__all__ = ["Qwen36GatedDeltaNet", "GDNConfig"]
