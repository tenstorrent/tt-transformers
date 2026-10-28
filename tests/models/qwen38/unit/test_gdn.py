# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Component PCC: single-device Gated DeltaNet (layer 0) vs the HF reference.

``device`` and ``setup`` come from tests/unit/conftest.py.
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
def test_deltanet_pcc(device, setup, request):
    """Compare TTNN deltanet against the torch reference for layer 0."""
    args, sd, raw = setup
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet

    from tt_transformers.models.qwen38.gdn import GDNConfig, Qwen38GatedDeltaNet
    from tt_transformers.models.qwen38.substate import substate

    layer_num = 0
    B, T = 1, 4

    x = torch.randn(B, T, args.dim, dtype=torch.bfloat16)

    # HF reference (transformers' modeling_qwen3_5), loaded from the raw checkpoint weights.
    reference = Qwen3_5GatedDeltaNet(hf_text_config(args), layer_num).float().eval()
    reference.load_state_dict({k: v.float() for k, v in hf_layer_state(raw, layer_num, "linear_attn").items()})
    with torch.no_grad():
        ref_out = reference(x.float())

    # TTNN
    deltanet = Qwen38GatedDeltaNet(device, GDNConfig.from_args(args), substate(sd, f"layers.{layer_num}.linear_attn"))
    deltanet.reset_state(B)
    x_t = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out = ttnn.to_torch(deltanet.forward(x_t, mode="recurrent"))

    pcc = compute_pcc(ref_out, out)
    logger.info(f"DeltaNet PCC: {pcc:.6f}")
    logger.info(
        f"Ref range: [{ref_out.min():.4f}, {ref_out.max():.4f}]  TTNN range: [{out.min():.4f}, {out.max():.4f}]"
    )
    assert pcc > get_pcc_threshold(request), f"DeltaNet PCC too low: {pcc}"
