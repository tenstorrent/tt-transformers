# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Qwen3.8 test-only constants formerly supplied by the tt-metal demo."""

from __future__ import annotations

import os
from pathlib import Path

import torch
import ttnn

_MESH_SHAPE = {"P150": (1, 1), "P150x4": (1, 4), "P150x8": (1, 8)}.get(os.environ.get("MESH_DEVICE"), (1, 4))
_MULTI = _MESH_SHAPE != (1, 1)
BLOCK_SIZE = 64
_TP_TRACE_REGION_SIZE = 1024 * 1024 * 1024
DEVICE_PARAMS = [
    {
        "l1_small_size": 24576,
        "num_command_queues": 2,
        **(
            {"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": _TP_TRACE_REGION_SIZE} if _MULTI else {}
        ),
    }
]

SHARED_PROMPTS_DIR = str(Path(__file__).with_name("assets"))


def _get_prompt(seqlen, tokenizer, max_prompt_len=None):
    """Return a deterministic prompt without depending on a tt-metal checkout."""
    cap = seqlen if max_prompt_len is None else min(seqlen, max_prompt_len)
    text = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": "Explain why careful measurements and reproducible tests matter in systems engineering.",
            },
        ],
        add_generation_prompt=True,
        tokenize=False,
    )
    ids = tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"]
    while ids.shape[1] < cap:
        ids = torch.cat((ids, ids), dim=1)
    return ids[:, :cap]


def _should_use_chunked_trace(model):
    return any(
        (not layer.is_full_attention)
        and getattr(getattr(layer.attention, "weights", None), "use_chunk_seq_prefill", False)
        for layer in model.layers
    )
