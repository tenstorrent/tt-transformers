# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from tt_transformers.models.qwen38.vision.input_validation import (
    build_qwen36_window_boundaries,
    get_qwen36_vision_bucket,
    get_qwen36_vision_max_patches,
    qwen36_vision_high_detail_enabled,
    validate_qwen36_packed_images,
)


def _pixels(count: int) -> torch.Tensor:
    return torch.zeros(count, 8, dtype=torch.bfloat16)


@pytest.mark.host
@pytest.mark.model
@pytest.mark.parametrize(
    "grid,count",
    [
        ([[1, 32, 32]], 1024),
        ([[1, 62, 66]], 4092),
        ([[1, 32, 64], [1, 32, 64]], 4096),
    ],
)
def test_standard_validator_accepts_each_supported_image(grid, count):
    normalized = validate_qwen36_packed_images(_pixels(count), torch.tensor(grid), spatial_merge_size=2)
    assert normalized.shape == (len(grid), 3)


@pytest.mark.host
@pytest.mark.model
@pytest.mark.parametrize("count,grid", [(4096, [[1, 64, 64]]), (7296, [[1, 76, 96]])])
def test_standard_validator_rejects_larger_buckets(count, grid):
    with pytest.raises(ValueError, match=rf"image\[0\]={count}.*maximum=4095"):
        validate_qwen36_packed_images(_pixels(count), torch.tensor(grid), spatial_merge_size=2)


@pytest.mark.host
@pytest.mark.model
@pytest.mark.parametrize(
    "pixels,grid,error",
    [
        (_pixels(4), torch.tensor([1, 2]), "shape \\[N, 3\\]"),
        (_pixels(4), torch.tensor([[1, 0, 4]]), "strictly positive"),
        (_pixels(8), torch.tensor([[2, 2, 2]]), "t=1"),
        (_pixels(6), torch.tensor([[1, 3, 2]]), "divisible"),
        (_pixels(3), torch.tensor([[1, 2, 2]]), "requires 4"),
    ],
)
def test_validator_rejects_malformed_input(pixels, grid, error):
    with pytest.raises(ValueError, match=error):
        validate_qwen36_packed_images(pixels, grid, spatial_merge_size=2)


@pytest.mark.host
@pytest.mark.model
@pytest.mark.parametrize("value", ["true", "yes", "2", "-1", ""])
def test_high_detail_environment_is_strict_boolean(monkeypatch, value):
    monkeypatch.setenv("QWEN36_VISION_HIGH_DETAIL", value)
    with pytest.raises(ValueError, match="must be exactly '0' or '1'"):
        qwen36_vision_high_detail_enabled()


@pytest.mark.host
@pytest.mark.model
@pytest.mark.parametrize("value", ["0", "-1", "bad"])
def test_patch_limit_environment_must_be_positive_integer(monkeypatch, value):
    monkeypatch.setenv("QWEN36_VISION_MAX_PATCHES", value)
    with pytest.raises(ValueError, match="must be a positive integer"):
        get_qwen36_vision_max_patches()


@pytest.mark.host
@pytest.mark.model
def test_patch_limit_cannot_exceed_prewarmed_mode(monkeypatch):
    monkeypatch.setenv("QWEN36_VISION_MAX_PATCHES", "4096")
    with pytest.raises(ValueError, match="standard limit of 4095"):
        get_qwen36_vision_max_patches()

    monkeypatch.setenv("QWEN36_VISION_HIGH_DETAIL", "1")
    monkeypatch.setenv("QWEN36_VISION_MAX_PATCHES", "8192")
    with pytest.raises(ValueError, match="high-detail limit of 8191"):
        get_qwen36_vision_max_patches()


@pytest.mark.host
@pytest.mark.model
def test_high_detail_covers_all_four_finite_buckets_and_multiple_images(monkeypatch):
    monkeypatch.setenv("QWEN36_VISION_HIGH_DETAIL", "1")
    cases = (
        (torch.tensor([[1, 32, 32]]), 2048),
        (torch.tensor([[1, 58, 62]]), 4096),
        (torch.tensor([[1, 64, 64]]), 6144),
        (torch.tensor([[1, 76, 96]]), 8192),
    )
    for grid, bucket in cases:
        count = int(grid.prod().item())
        validate_qwen36_packed_images(_pixels(count), grid, spatial_merge_size=2)
        assert get_qwen36_vision_bucket(grid) == bucket

    four_native = torch.tensor([[1, 76, 96]] * 4)
    normalized = validate_qwen36_packed_images(_pixels(4 * 7296), four_native, spatial_merge_size=2)
    assert torch.equal(normalized, four_native)


@pytest.mark.host
@pytest.mark.model
def test_high_detail_rejects_the_first_unwarmed_bucket(monkeypatch):
    monkeypatch.setenv("QWEN36_VISION_HIGH_DETAIL", "1")
    grid = torch.tensor([[1, 64, 128]])
    with pytest.raises(ValueError, match=r"image\[0\]=8192.*maximum=8191"):
        validate_qwen36_packed_images(_pixels(8192), grid, spatial_merge_size=2)


@pytest.mark.host
@pytest.mark.model
def test_window_boundaries_append_one_padding_only_segment():
    assert build_qwen36_window_boundaries(torch.tensor([0, 7296]), seq_len=8192) == [0, 7296, 8192]
    with pytest.raises(ValueError, match="start at zero"):
        build_qwen36_window_boundaries(torch.tensor([1, 7296]), seq_len=8192)
    with pytest.raises(ValueError, match="strictly increasing"):
        build_qwen36_window_boundaries(torch.tensor([0, 128, 128]), seq_len=2048)
    with pytest.raises(ValueError, match="non-empty padding window"):
        build_qwen36_window_boundaries(torch.tensor([0, 2048]), seq_len=2048)


@pytest.mark.host
@pytest.mark.model
def test_tt_processor_rejects_unsupported_geometry_as_request_validation(monkeypatch):
    from vllm.exceptions import VLLMValidationError
    from vllm.model_executor.models.qwen3_5 import Qwen3VLMultiModalProcessor

    from tt_transformers.models.qwen38.qwen36_vllm import TTQwen3VLMultiModalProcessor

    monkeypatch.setattr(Qwen3VLMultiModalProcessor, "_get_mm_fields_config", lambda *args: {"delegated": True})
    processor = object.__new__(TTQwen3VLMultiModalProcessor)
    processor.info = SimpleNamespace(
        get_hf_config=lambda: SimpleNamespace(vision_config=SimpleNamespace(spatial_merge_size=2))
    )
    with pytest.raises(VLLMValidationError) as exc_info:
        processor._get_mm_fields_config(
            {"pixel_values": _pixels(7296), "image_grid_thw": torch.tensor([[1, 76, 96]])},
            {},
        )
    assert exc_info.value.parameter == "image"
    assert "maximum=4095" in str(exc_info.value)


@pytest.mark.host
@pytest.mark.model
def test_tt_processor_delegates_high_detail_and_text_inputs(monkeypatch):
    from vllm.model_executor.models.qwen3_5 import Qwen3VLMultiModalProcessor

    from tt_transformers.models.qwen38.qwen36_vllm import TTQwen3VLMultiModalProcessor

    sentinel = {"delegated": True}
    monkeypatch.setenv("QWEN36_VISION_HIGH_DETAIL", "1")
    monkeypatch.setattr(Qwen3VLMultiModalProcessor, "_get_mm_fields_config", lambda *args: sentinel)
    processor = object.__new__(TTQwen3VLMultiModalProcessor)
    processor.info = SimpleNamespace(
        get_hf_config=lambda: SimpleNamespace(vision_config=SimpleNamespace(spatial_merge_size=2))
    )
    assert processor._get_mm_fields_config({}, {}) is sentinel
    assert (
        processor._get_mm_fields_config(
            {"pixel_values": _pixels(7296), "image_grid_thw": torch.tensor([[1, 76, 96]])},
            {},
        )
        is sentinel
    )


@pytest.mark.host
@pytest.mark.model
def test_tt_processor_rejects_precomputed_image_embeds():
    from vllm.exceptions import VLLMValidationError

    from tt_transformers.models.qwen38.qwen36_vllm import TTQwen3VLMultiModalProcessor

    processor = object.__new__(TTQwen3VLMultiModalProcessor)
    with pytest.raises(VLLMValidationError, match="image_embeds") as exc_info:
        processor._get_mm_fields_config({"image_embeds": torch.zeros(1, 8)}, {})
    assert exc_info.value.parameter == "image"


@pytest.mark.host
@pytest.mark.model
def test_dflash_adapter_registers_tt_geometry_validator():
    from tt_transformers.models.qwen38.qwen36_vllm import TTQwen3VLMultiModalProcessor
    from tt_transformers.models.qwen38.qwen36_vllm_dflash import (
        TTQwen3VLMultiModalProcessor as DFlashMultiModalProcessor,
    )

    assert DFlashMultiModalProcessor is TTQwen3VLMultiModalProcessor
