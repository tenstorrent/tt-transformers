# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for WH Galaxy `(8, 4)` hardware tests of the 2D models.

These are test-side utilities only: checkpoint skips, the reference-token files
the accuracy gates measure against, and the teacher-forcing accuracy convention
the 1D demos already use. Keeping the convention identical is what makes the
Galaxy numbers comparable to the existing product gates.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
import torch
import ttnn

from tt_transformers.models.galaxy.topology import galaxy_device_params

GALAXY_MESH_SHAPE = (8, 4)
GALAXY_PHYSICAL_BATCH = 32
GALAXY_USERS_PER_COLUMN = 8

#: The dispatch/fabric parameters every qualified Galaxy recipe was tuned with.
#:
#: ``l1_small_size`` is not a tuning parameter but a correctness one: it is where
#: the generic collectives put their program-cache-lifetime semaphores, and
#: without it they fragment main L1 and break the weight prefetcher's
#: `prefill -> decode` restore. See `GALAXY_L1_SMALL_SIZE`.
#:
#: **The fabric configuration is architecture-specific and chosen before the mesh
#: opens**, which is why it comes from the topology descriptor rather than being
#: probed afterwards. Blackhole Galaxy runs its column-axis (``cluster_axis=1``)
#: collectives on device and needs ``FABRIC_2D_TORUS_XY``; opening it with the
#: Wormhole ``FABRIC_1D_RING`` below throws ``IndexError: map::at`` on the first
#: cross-column route.
GALAXY_DEVICE_PARAMS = galaxy_device_params(ttnn.device.Arch.WORMHOLE_B0)

#: The Blackhole Galaxy equivalent. Separate rather than resolved at fixture time
#: because ``device_params`` is a collection-time parametrization: the value has
#: to exist before any device is open to be read from.
BHGLX_DEVICE_PARAMS = galaxy_device_params(ttnn.device.Arch.BLACKHOLE)

#: Repository root, derived from this file rather than the working directory, so the
#: asset lookups below do not depend on where pytest was invoked.
_REPO_ROOT = Path(__file__).resolve().parents[3]

#: Reference *token* assets -- the small `.refpt` files that ship in the repository.
#: tt-metal kept them flat under `models/tt_transformers/tests/reference_outputs`;
#: this package stores them per model directory under `tests/assets`. The tt-metal
#: path resolved to nothing here, so every gate reading tokens skipped silently.
_TOKEN_ASSET_ROOT = _REPO_ROOT / "tests/assets/reference_outputs"


def local_files_only() -> bool:
    return any(
        os.getenv(name, "").lower() in {"1", "true", "yes"} for name in ("CI", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    )


def hf_config_or_skip(hf_model: str, *, revision: str | None = None) -> Any:
    """Return the checkpoint config, or skip when it cannot be resolved."""

    from transformers import AutoConfig

    try:
        return AutoConfig.from_pretrained(hf_model, revision=revision, local_files_only=local_files_only())
    except BaseException as error:  # noqa: BLE001 - any resolution failure is a skip, not a defect
        pytest.skip(f"checkpoint {hf_model!r} is unavailable: {error}")


def deallocate(tensor: Any) -> None:
    if tensor is None:
        return
    is_allocated = getattr(tensor, "is_allocated", None)
    if callable(is_allocated) and not is_allocated():
        return
    release = getattr(tensor, "deallocate", None)
    if callable(release):
        release(True)


def load_reference_tokens(model_name: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(reference_tokens, top5_tokens)`` for a checkpoint name.

    ``reference_tokens`` is a real-text token sequence, returned **one
    dimensional**; ``top5_tokens[i]`` holds the reference implementation's five
    most likely next tokens at position ``i``. Neither the Galaxy model nor a
    Hugging Face model needs to be loaded to score against them.

    The stored files hold ``reference_tokens`` with a leading batch axis --
    ``generate_reference_outputs.py`` writes ``encoded_tokens_tensor[:, :n]``,
    so ``Llama-3.3-70B-Instruct.refpt`` is ``(1, 1024)``. Every consumer here
    and in the 1D demos treats the sequence as flat: ``len(reference_tokens)``
    is the sequence length, and ``reference_tokens[i]`` is a token. Returning
    the tensor raw made ``len()`` report **1**, so a caller asking for a
    512-token prompt saw "reference sequence has 1 tokens" and *skipped* -- the
    accuracy gate could never run, and it failed open rather than loud.
    ``tests/demos/llama33_70b/demo.py`` already squeezes at its own call site;
    this does it once, here, for every caller.
    """

    matches = sorted(_TOKEN_ASSET_ROOT.glob(f"*/{model_name}.refpt"))
    if not matches:
        pytest.skip(f"reference token file not found: {_TOKEN_ASSET_ROOT}/*/{model_name}.refpt")
    if len(matches) > 1:
        raise AssertionError(f"{model_name}.refpt is ambiguous across {[str(p.parent.name) for p in matches]}")
    data = torch.load(matches[0], map_location="cpu", weights_only=False)
    reference_tokens = data["reference_tokens"]
    if reference_tokens.dim() > 1:
        reference_tokens = reference_tokens.reshape(-1)
    return reference_tokens, data["top5_tokens"]


def align_top5(top5_tokens: torch.Tensor, reference_tokens: torch.Tensor, prompt_len: int) -> torch.Tensor:
    """Align ``top5_tokens`` with the teacher-forcing targets.

    The stored files predate a single offset convention, so the slice whose
    top-1 column best matches the known target tokens is the correct one. This
    mirrors ``select_teacher_forcing_top5_slice`` in the 1D demos so the two
    accuracy numbers mean the same thing.
    """

    num_target = len(reference_tokens) - prompt_len
    if num_target <= 0:
        raise ValueError("prompt_len must be smaller than the reference length")
    targets = reference_tokens[prompt_len : prompt_len + num_target]
    candidates = []
    for start in (prompt_len - 1, prompt_len, 0):
        end = start + num_target
        if start < 0 or end > top5_tokens.shape[0]:
            continue
        aligned = top5_tokens[start:end]
        probe = min(16, num_target)
        score = sum(int(aligned[index, 0].item() == targets[index].item()) for index in range(probe))
        candidates.append((score, start, aligned))
    if not candidates:
        raise ValueError(
            f"cannot align top5: prompt_len={prompt_len}, num_target={num_target}, top5_len={top5_tokens.shape[0]}"
        )
    return max(candidates, key=lambda candidate: candidate[0])[2]


def teacher_forcing_accuracy(predictions: Sequence[int], reference_top5: torch.Tensor) -> tuple[float, float]:
    """Return ``(top1, top5)`` accuracy as fractions in ``[0, 1]``.

    ``top1`` is the share of positions where the model's argmax equals the
    reference's argmax; ``top5`` is the share where it appears anywhere in the
    reference's five most likely tokens.
    """

    if len(predictions) != reference_top5.shape[0]:
        raise ValueError(f"{len(predictions)} predictions for {reference_top5.shape[0]} reference rows")
    top1 = sum(1 for index, token in enumerate(predictions) if int(reference_top5[index, 0]) == token)
    top5 = sum(1 for index, token in enumerate(predictions) if token in reference_top5[index, :].tolist())
    return top1 / len(predictions), top5 / len(predictions)


__all__ = [
    "BHGLX_DEVICE_PARAMS",
    "GALAXY_DEVICE_PARAMS",
    "GALAXY_MESH_SHAPE",
    "GALAXY_PHYSICAL_BATCH",
    "GALAXY_USERS_PER_COLUMN",
    "align_top5",
    "deallocate",
    "hf_config_or_skip",
    "load_reference_tokens",
    "local_files_only",
    "teacher_forcing_accuracy",
]
