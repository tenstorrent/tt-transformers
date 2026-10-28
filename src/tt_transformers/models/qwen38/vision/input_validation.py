# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-side admission and bucketing for Qwen3.8 packed image inputs."""

import os

import torch

DEFAULT_QWEN36_VISION_MAX_PATCHES = 4095
HIGH_DETAIL_QWEN36_VISION_MAX_PATCHES = 8191
QWEN36_VISION_BUCKET_SIZE = 2048
QWEN36_HIGH_DETAIL_BUCKETS = (2048, 4096, 6144, 8192)


def qwen36_vision_high_detail_enabled() -> bool:
    """Return whether the finite, prewarmed high-detail image path is enabled."""

    raw = os.environ.get("QWEN36_VISION_HIGH_DETAIL", "0")
    if raw not in ("0", "1"):
        raise ValueError(f"QWEN36_VISION_HIGH_DETAIL must be exactly '0' or '1', got {raw!r}")
    return raw == "1"


def get_qwen36_vision_max_patches() -> int:
    """Return the per-image patch limit covered by the selected warmup policy."""

    supported_limit = (
        HIGH_DETAIL_QWEN36_VISION_MAX_PATCHES
        if qwen36_vision_high_detail_enabled()
        else DEFAULT_QWEN36_VISION_MAX_PATCHES
    )
    raw = os.environ.get("QWEN36_VISION_MAX_PATCHES", str(supported_limit))
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"QWEN36_VISION_MAX_PATCHES must be a positive integer, got {raw!r}") from exc
    if value <= 0:
        raise ValueError(f"QWEN36_VISION_MAX_PATCHES must be a positive integer, got {raw!r}")
    if value > supported_limit:
        mode = "high-detail" if qwen36_vision_high_detail_enabled() else "standard"
        raise ValueError(f"QWEN36_VISION_MAX_PATCHES={value} exceeds the pre-warmed {mode} limit of {supported_limit}")
    return value


def get_qwen36_vision_bucket(image_grid_thw: torch.Tensor) -> int:
    """Return the fixed physical bucket for one already-validated image.

    A strictly larger bucket is intentional. It gives the padding rows their
    own attention window in high-detail mode and keeps descriptor shapes fixed.
    """

    patch_count = int(image_grid_thw.prod().item())
    bucket = ((patch_count // QWEN36_VISION_BUCKET_SIZE) + 1) * QWEN36_VISION_BUCKET_SIZE
    supported = QWEN36_HIGH_DETAIL_BUCKETS if qwen36_vision_high_detail_enabled() else (2048, 4096)
    if bucket not in supported:
        raise ValueError(
            f"processed image requires unsupported vision bucket {bucket}: "
            f"patches={patch_count}, pre-warmed buckets={supported}"
        )
    return bucket


def build_qwen36_window_boundaries(cu_seqlens: torch.Tensor, *, seq_len: int) -> list[int]:
    """Validate HF cumulative lengths and append the padding-only window."""

    boundaries = [int(value) for value in cu_seqlens.tolist()]
    if not boundaries or boundaries[0] != 0:
        raise ValueError(f"vision cumulative lengths must start at zero, got {boundaries}")
    if any(left >= right for left, right in zip(boundaries, boundaries[1:])):
        raise ValueError(f"vision cumulative lengths must be strictly increasing, got {boundaries}")
    if boundaries[-1] >= int(seq_len):
        raise ValueError(
            f"vision cumulative lengths must leave a non-empty padding window: last={boundaries[-1]}, bucket={seq_len}"
        )
    boundaries.append(int(seq_len))
    return boundaries


def validate_qwen36_packed_images(
    pixel_values,
    image_grid_thw,
    *,
    spatial_merge_size: int,
    max_patches: int | None = None,
) -> torch.Tensor:
    """Validate packed image tensors and return a normalized ``[N, 3]`` grid.

    The tower processes images independently, so the limit applies to each
    image rather than to the sum of packed rows.
    """

    if not isinstance(pixel_values, torch.Tensor) or pixel_values.dim() != 2:
        raise ValueError(
            "pixel_values must be a packed rank-2 torch tensor "
            f"[num_patches, patch_dim], got {type(pixel_values).__name__} "
            f"with shape {getattr(pixel_values, 'shape', None)}"
        )

    if not isinstance(image_grid_thw, torch.Tensor):
        image_grid_thw = torch.as_tensor(image_grid_thw)
    if image_grid_thw.dim() == 1:
        image_grid_thw = image_grid_thw.unsqueeze(0)
    if image_grid_thw.dim() != 2 or image_grid_thw.shape[1] != 3:
        raise ValueError(f"image_grid_thw must have shape [N, 3], got {tuple(image_grid_thw.shape)}")
    if image_grid_thw.numel() == 0 or bool((image_grid_thw <= 0).any()):
        raise ValueError("image_grid_thw must contain one or more strictly positive (t,h,w) rows")
    if bool((image_grid_thw[:, 0] != 1).any()):
        raise ValueError("Qwen high-detail image serving requires t=1 per image; video is not supported")

    merge = int(spatial_merge_size)
    if merge <= 0:
        raise ValueError(f"spatial_merge_size must be positive, got {spatial_merge_size!r}")
    if bool((image_grid_thw[:, 1:] % merge != 0).any()):
        raise ValueError(
            f"grid height and width must be divisible by spatial_merge_size={merge}: {image_grid_thw.tolist()}"
        )

    patch_counts = [int(row.prod().item()) for row in image_grid_thw]
    expected_patches = sum(patch_counts)
    if int(pixel_values.shape[0]) != expected_patches:
        raise ValueError(
            f"pixel_values has {int(pixel_values.shape[0])} packed rows, but image_grid_thw requires {expected_patches}"
        )

    limit = get_qwen36_vision_max_patches() if max_patches is None else int(max_patches)
    if limit <= 0:
        raise ValueError(f"max_patches must be positive, got {max_patches!r}")
    oversized = [(index, count) for index, count in enumerate(patch_counts) if count > limit]
    if oversized:
        details = ", ".join(f"image[{index}]={count}" for index, count in oversized)
        raise ValueError(
            f"processed image exceeds the pre-warmed vision limit: {details} patches, maximum={limit} per image"
        )

    for row in image_grid_thw:
        get_qwen36_vision_bucket(row.unsqueeze(0))
    return image_grid_thw
