# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for standalone TT Transformers tests."""

from __future__ import annotations

import zlib


def stable_model_seed(model_name: str) -> int:
    """Stable 32-bit seed derived from model name.

    Python's built-in hash is randomized per process, which breaks reproducibility
    across runs and can mismatch on-disk cached weights. Use CRC32 instead.

    NOTE: Avoid a single hardcoded global seed (e.g., 1234) for all models; a
    per-model stable seed keeps caches distinct and reduces correlated RNG paths.
    """
    return zlib.crc32(model_name.encode("utf-8")) & 0xFFFFFFFF


def hf_config_or_skip(model_name: str, **kwargs):
    """``AutoConfig.from_pretrained``, skipping the test when the config isn't available.

    Skips when the hub can't be reached and the config isn't in the local cache, or when the repo
    is gated and this machine has no access. Any other error, such as a misspelled model id,
    still fails the test.
    """
    import pytest
    from huggingface_hub.errors import GatedRepoError, LocalEntryNotFoundError
    from transformers import AutoConfig

    try:
        return AutoConfig.from_pretrained(model_name, **kwargs)
    except OSError as error:
        cause: BaseException | None = error
        while cause is not None and not isinstance(cause, (LocalEntryNotFoundError, GatedRepoError)):
            cause = cause.__cause__
        if cause is None:
            raise
        pytest.skip(f"Hugging Face config for {model_name} is not available: {str(error).splitlines()[0]}")
