# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch


@pytest.mark.host
@pytest.mark.model
def test_interpolated_positions_match_the_hf_vision_model():
    from transformers.models.qwen3_5 import Qwen3_5VisionConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel

    from tt_transformers.models.qwen38.vision.model import interpolate_pos_embeddings

    torch.manual_seed(0)
    config = Qwen3_5VisionConfig(
        depth=1, hidden_size=64, intermediate_size=64, num_heads=2, out_hidden_size=64, num_position_embeddings=2304
    )
    reference = Qwen3_5VisionModel(config).eval()
    grid_thw = torch.tensor([[1, 16, 24], [2, 32, 8]])
    with torch.no_grad():
        actual = interpolate_pos_embeddings(reference, grid_thw)
    assert actual.shape == (int(grid_thw.prod(-1).sum()), 64)

    # The HF model's own interpolation (deprecated, so absent from newer transformers).
    if not hasattr(reference, "fast_pos_embed_interpolate"):
        pytest.skip("this transformers version has no fast_pos_embed_interpolate")
    with torch.no_grad(), pytest.warns(FutureWarning):
        expected = reference.fast_pos_embed_interpolate(grid_thw)
    torch.testing.assert_close(actual, expected)
