# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Component PCC: single-device gated attention (layer 3) vs the HF reference.

``device`` and ``setup`` come from tests/unit/conftest.py. (The TP loader's
q_norm/k_norm +1 regression lives with the TP attention tests in
``tests/test_attention_tp.py::test_attention_tp_qknorm_offset``.)
"""

import pytest
import torch
import ttnn
from loguru import logger

from tests.models.qwen38.test_factory import compute_pcc, get_pcc_threshold, run_for_blackhole

from .conftest import DEVICE_PARAMS, hf_layer_state, hf_text_config

pytestmark = [run_for_blackhole(), pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)]


@pytest.mark.device
@pytest.mark.model
def test_gated_attention_pcc(device, setup, request):
    """Compare TTNN gated attention against the torch reference for layer 3."""
    args, sd, raw = setup
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Attention, Qwen3_5TextRotaryEmbedding

    from tt_transformers.models.qwen38.rope import Qwen38RoPESetup

    layer_num = 3
    B, T = 1, 4

    x = torch.randn(B, T, args.dim, dtype=torch.bfloat16)

    # HF reference (transformers' modeling_qwen3_5), loaded from the raw checkpoint weights.
    config = hf_text_config(args)
    reference = Qwen3_5Attention(config, layer_num).to(torch.bfloat16).eval()
    reference.load_state_dict(hf_layer_state(raw, layer_num, "self_attn"))
    position_ids = torch.arange(T).unsqueeze(0)
    with torch.no_grad():
        position_embeddings = Qwen3_5TextRotaryEmbedding(config)(x, position_ids)
        ref_out, _ = reference(x, position_embeddings=position_embeddings, attention_mask=None)

    # TTNN
    from tt_transformers.models.qwen38.attention import AttentionConfig, Qwen38GatedAttention
    from tt_transformers.models.qwen38.substate import substate

    attn_state = substate(sd, f"layers.{layer_num}.self_attn")
    attn = Qwen38GatedAttention(device, AttentionConfig.from_args(args), attn_state)
    rope = Qwen38RoPESetup(device, args)
    pos = torch.arange(T).unsqueeze(0)
    cos_ttnn, sin_ttnn = rope.get_rot_mats(pos)

    x_t = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out = ttnn.to_torch(attn.forward(x_t, cos_ttnn, sin_ttnn))

    pcc = compute_pcc(ref_out, out)
    logger.info(f"Gated Attention PCC: {pcc:.6f}")
    logger.info(
        f"Ref range: [{ref_out.min():.4f}, {ref_out.max():.4f}]  TTNN range: [{out.min():.4f}, {out.max():.4f}]"
    )
    assert pcc > get_pcc_threshold(request), f"Gated Attention PCC too low: {pcc}"
