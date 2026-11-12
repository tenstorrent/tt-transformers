# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-side helpers for the packed grammar masks vLLM builds for structured outputs.

vLLM packs one request's allowed tokens into ``ceil(vocab_size / 32)`` int32
words: bit ``j`` of word ``w`` allows token ``32 * w + j``. The device sampler
holds a fixed ``[max_batch_size, padded_vocab_size / 32]`` copy, sharded on the
vocab axis like the logits, and unpacks it on device. These helpers validate a
vLLM mask and lay it out for that buffer. They run on the host only and import
no ttnn.
"""

from __future__ import annotations

import torch

GRAMMAR_WORD_BITS = 32


def packed_grammar_width(vocab_size: int) -> int:
    """Number of int32 words vLLM uses for one row of a ``vocab_size`` mask."""
    if vocab_size <= 0:
        raise ValueError(f"vocab_size must be positive, got {vocab_size}")
    return (vocab_size + GRAMMAR_WORD_BITS - 1) // GRAMMAR_WORD_BITS


def grammar_layout_error(padded_vocab_size: int, num_vocab_shards: int) -> str | None:
    """Why a sampler with this logits layout cannot apply a grammar mask, or ``None``.

    Each vocab shard unpacks its own slice of the mask, so every shard must hold
    whole 32-token words. A word that straddles two shards cannot be split.
    """
    if num_vocab_shards <= 0:
        raise ValueError(f"num_vocab_shards must be positive, got {num_vocab_shards}")
    if padded_vocab_size % (GRAMMAR_WORD_BITS * num_vocab_shards) != 0:
        return (
            "device grammar sampling requires every vocab shard to hold whole 32-token mask words: "
            f"padded_vocab_size={padded_vocab_size} is not a multiple of "
            f"{GRAMMAR_WORD_BITS} * {num_vocab_shards} shards"
        )
    return None


def format_grammar_bitmask(
    grammar_bitmask: torch.Tensor,
    *,
    vocab_size: int,
    padded_vocab_size: int,
    max_batch_size: int,
) -> torch.Tensor:
    """Validate a vLLM grammar mask and pad it to the device sampler's shape.

    Returns an int32 ``[max_batch_size, padded_vocab_size / 32]`` tensor. Rows past
    the mask's batch are all-allowed (``-1``). In the mask's own rows, every token
    id at or above ``vocab_size`` is blocked: the tail bits of the last real word
    and every padding word.

    Raises ``TypeError`` or ``ValueError`` before anything is uploaded when the
    mask has the wrong dtype or shape, or when a row allows no real token.
    """
    if not isinstance(grammar_bitmask, torch.Tensor):
        raise TypeError(f"grammar_bitmask must be a torch.Tensor, got {type(grammar_bitmask).__name__}")
    if grammar_bitmask.dtype != torch.int32:
        raise TypeError(f"grammar_bitmask must have dtype torch.int32, got {grammar_bitmask.dtype}")
    if grammar_bitmask.ndim != 2:
        raise ValueError(
            f"grammar_bitmask must have shape [batch, ceil(vocab_size / 32)], got {tuple(grammar_bitmask.shape)}"
        )
    if padded_vocab_size < vocab_size or padded_vocab_size % GRAMMAR_WORD_BITS != 0:
        raise ValueError(
            "padded_vocab_size must be a multiple of 32 and at least vocab_size, "
            f"got vocab_size={vocab_size}, padded_vocab_size={padded_vocab_size}"
        )

    packed_width = packed_grammar_width(vocab_size)
    if grammar_bitmask.shape[1] != packed_width:
        raise ValueError(
            "grammar_bitmask packed width does not match vocab_size: "
            f"got {grammar_bitmask.shape[1]}, expected {packed_width} for vocab_size={vocab_size}"
        )
    batch = int(grammar_bitmask.shape[0])
    if batch > max_batch_size:
        raise ValueError(f"grammar_bitmask batch {batch} exceeds sampler batch {max_batch_size}")

    formatted = torch.full((max_batch_size, padded_vocab_size // GRAMMAR_WORD_BITS), -1, dtype=torch.int32)
    active = grammar_bitmask.detach().to(device="cpu").clone()
    tail_bits = vocab_size % GRAMMAR_WORD_BITS
    if tail_bits:
        active[:, -1] &= (1 << tail_bits) - 1

    empty_rows = torch.nonzero(~(active != 0).any(dim=1)).flatten()
    if empty_rows.numel():
        raise ValueError(f"grammar_bitmask contains rows with no allowed token: rows={empty_rows.tolist()!r}")

    formatted[:batch] = 0
    formatted[:batch, :packed_width] = active
    return formatted


def all_allowed_grammar_bitmask(max_batch_size: int, padded_vocab_size: int) -> torch.Tensor:
    """The formatted mask that allows every token in every row."""
    if padded_vocab_size % GRAMMAR_WORD_BITS != 0:
        raise ValueError(f"padded_vocab_size must be a multiple of 32, got {padded_vocab_size}")
    return torch.full((max_batch_size, padded_vocab_size // GRAMMAR_WORD_BITS), -1, dtype=torch.int32)


def unpack_grammar_bitmask(packed: torch.Tensor, vocab_size: int) -> torch.Tensor:
    """Host reference: a bool ``[rows, vocab_size]`` allow table from packed words."""
    shifts = torch.arange(GRAMMAR_WORD_BITS, dtype=torch.int32)
    bits = (packed.to(torch.int32).unsqueeze(-1) >> shifts) & 1
    return bits.reshape(packed.shape[0], -1)[:, :vocab_size].bool()
