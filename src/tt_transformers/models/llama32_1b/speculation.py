# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Sequential target logits for synchronous CPU n-gram proposals."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

SUPPORTED_BATCHES = (1, 2, 4)
SUPPORTED_DRAFT_LENGTHS = (1, 2, 3)
SUPPORTED_CHECKPOINT = "meta-llama/Llama-3.2-1B"
_INTEGER_DTYPES = (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)


def verify_logits(
    *,
    tokens: torch.Tensor,
    start_pos: torch.Tensor,
    page_table: torch.Tensor,
    num_valid_drafts: torch.Tensor,
    accepted_counts: torch.Tensor,
    batch_size: int,
    vocab_size: int,
    max_seq_len: int,
    block_size: int,
    num_blocks: int,
    decode: Callable[..., Any],
    slot_remap: torch.Tensor | None = None,
) -> torch.Tensor:
    """Validate the complete block before issuing any ordinary target forward."""

    for name, value in (
        ("tokens", tokens),
        ("start_pos", start_pos),
        ("page_table", page_table),
        ("num_valid_drafts", num_valid_drafts),
        ("accepted_counts", accepted_counts),
    ):
        _validate_integer_tensor(name, value)
    if tokens.ndim != 2 or start_pos.shape != tokens.shape:
        raise ValueError("tokens and start_pos must have matching [B, 1+K] shapes")
    batch, width = tokens.shape
    if batch_size not in SUPPORTED_BATCHES or batch != batch_size:
        raise ValueError(f"verification batch {batch} must equal supported lane capacity {batch_size}")
    if width - 1 not in SUPPORTED_DRAFT_LENGTHS:
        raise ValueError(f"verification K={width - 1} must be one of {SUPPORTED_DRAFT_LENGTHS}")
    if page_table.ndim != 2 or page_table.shape[0] != batch:
        raise ValueError("page_table must have shape [B, pages]")
    for name, value in (("num_valid_drafts", num_valid_drafts), ("accepted_counts", accepted_counts)):
        if value.shape != (batch,):
            raise ValueError(f"{name} must have shape [{batch}]")
    if bool(((num_valid_drafts < 0) | (num_valid_drafts >= width)).any()):
        raise ValueError(f"num_valid_drafts must be in [0, {width - 1}]")
    if bool(((accepted_counts < 1) | (accepted_counts > width)).any()):
        raise ValueError(f"accepted_counts must be in [1, {width}]")
    if slot_remap is not None:
        _validate_integer_tensor("slot_remap", slot_remap)
        if slot_remap.shape != (batch,) or not torch.equal(
            slot_remap.to(torch.long).sort().values, torch.arange(batch)
        ):
            raise ValueError(f"slot_remap must be a permutation of [0, {batch - 1}]")

    live = start_pos[:, 0] >= 0
    if bool((~live & (num_valid_drafts != 0)).any()):
        raise ValueError("padding rows must have zero num_valid_drafts")
    columns = torch.arange(width)
    valid = live[:, None] & (columns[None, :] <= num_valid_drafts[:, None])
    expected_positions = start_pos[:, :1] + columns[None, :]
    if bool((valid & (start_pos != expected_positions)).any()):
        raise ValueError("valid start_pos columns must be consecutive from each row's anchor")
    if bool((~valid & (start_pos != -1)).any()):
        raise ValueError("inactive start_pos columns must equal -1")
    if bool((valid & ((tokens < 0) | (tokens >= vocab_size))).any()):
        raise ValueError(f"valid tokens must be in [0, {vocab_size - 1}]")
    if bool((valid & (start_pos >= max_seq_len)).any()):
        raise ValueError(f"valid start_pos must be below max_seq_len={max_seq_len}")
    for row in range(batch):
        if not bool(live[row]):
            continue
        last_position = int(start_pos[row, int(num_valid_drafts[row])])
        used_blocks = last_position // block_size + 1
        if used_blocks > page_table.shape[1]:
            raise ValueError(f"page_table is too narrow for verification row {row}")
        allocated = page_table[row, :used_blocks]
        if bool(((allocated < 0) | (allocated >= num_blocks)).any()):
            raise ValueError(f"page_table row {row} references unallocated KV blocks")

    logits = torch.zeros((batch, width, vocab_size), dtype=torch.float32)
    for column in range(width):
        active = valid[:, column]
        if not bool(active.any()):
            break
        column_tokens = torch.where(active, tokens[:, column], 0).contiguous()
        column_positions = torch.where(active, start_pos[:, column], -1).contiguous()
        output = decode(tokens=column_tokens, start_pos=column_positions, page_table=page_table)
        output = output[0] if isinstance(output, tuple) and len(output) == 2 else output
        if (
            not isinstance(output, torch.Tensor)
            or output.device.type != "cpu"
            or not output.is_floating_point()
            or output.shape != (batch, 1, vocab_size)
        ):
            raise TypeError(f"ordinary target decode must return CPU floating logits [{batch}, 1, {vocab_size}]")
        # Copy before another forward can reuse the runtime's output storage.
        logits[active, column] = output[active, 0].float()
    return logits


def _validate_integer_tensor(name: str, value: torch.Tensor) -> None:
    if not isinstance(value, torch.Tensor) or value.device.type != "cpu" or value.dtype not in _INTEGER_DTYPES:
        raise TypeError(f"{name} must be a CPU integer torch.Tensor")
