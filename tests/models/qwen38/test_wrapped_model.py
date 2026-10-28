# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.

# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
import ttnn
from loguru import logger

from tests.models.qwen38.test_factory import _resolve_mesh_shape
from tt_transformers.models.qwen38.v1.ccl import TT_CCL
from tt_transformers.models.qwen38.v1.utility import comp_allclose, comp_pcc
from tt_transformers.models.qwen38.vision.model import DropInVisionTransformer
from tt_transformers.models.qwen38.vision.vision_model_config import VisionModelArgs


@torch.no_grad()
@pytest.mark.parametrize(
    "mesh_device",
    [_resolve_mesh_shape()],
    indirect=True,
)
@pytest.mark.parametrize(
    "num_layers",
    [None, 1, 2],  # None means all layers, specific numbers will run fewer layers
    ids=["all_layers", "single_layer", "two_layers"],
)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.device
@pytest.mark.model
def test_wrapped_vision_model_inference(
    mesh_device,
    reset_seeds,
    ensure_gc,
    num_layers,
    is_ci_env,
    request,
):
    test_id = request.node.callspec.id
    if is_ci_env and "two_layers" not in test_id:
        pytest.skip("CI only runs the two_layers test")

    dtype = ttnn.bfloat8_b
    # The per-block suite uses 0.85 for layers 24-27 under the production BF8
    # policy. Preserve 0.99 for focused shallow checks and apply that existing
    # accumulated-error contract to the full 27-layer tower.
    pcc = 0.99 if num_layers and num_layers <= 3 else 0.85
    batch_size = 1  # For prefill we only support batch_size = 1

    # Example inputs for http://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg
    # pixel_values are produced by Qwen2_5_VLImageProcessor, these come from the above img
    # image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
    #     The temporal, height and width of feature shape of each image in LLM.
    # Keep the component test inside the serving contract's pre-warmed
    # QWEN36_VISION_MAX_PATCHES=4095 admission ceiling. Oversized media has
    # separate rejection coverage in test_vision_context.py.
    image_grid_thw = torch.tensor([[1, 32, 32]])
    ref_seq_len = image_grid_thw[0, 1] * image_grid_thw[0, 2]
    # pad seq_len to be divisible by 128 (MAX_QKV_MM_SEQ_LEN from tt_transformers model)
    seq_len = ((ref_seq_len // 128) + 1) * 128
    pt_pixel_values = torch.randn([ref_seq_len, 1536])

    model_args = VisionModelArgs(mesh_device, dummy_weights=True, max_batch_size=batch_size, max_seq_len=seq_len)
    if num_layers:
        model_args.hf_config.vision_config.depth = num_layers
        from transformers import logging as transformers_logging

        # Set logging level to ERROR to suppress warnings about unexpected keys
        transformers_logging.set_verbosity_error()
    else:
        num_layers = model_args.hf_config.vision_config.depth

    # Create reference model
    reference_model = model_args.reference_vision_model(depth=model_args.hf_config.vision_config.depth)
    torch_model = DropInVisionTransformer(reference_model, model_args, tt_ccl=TT_CCL(mesh_device))

    # Run reference model
    reference_output = reference_model(pt_pixel_values, image_grid_thw).pooler_output
    tt_output = torch_model(pt_pixel_values, image_grid_thw)

    # Serving gathers the final embeddings to host before replaying a text
    # trace, so the current wrapper returns a CPU tensor. Keep compatibility
    # with the older device-tensor contract for lower-level experiments.
    if isinstance(tt_output, torch.Tensor):
        tt_output_torch = tt_output
    else:
        tt_output_torch = ttnn.to_torch(
            tt_output,
            mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=3),
        )
    logger.info("output shape", tt_output_torch.shape)
    tt_output_torch = tt_output_torch.squeeze(0).squeeze(0)
    assert tt_output_torch.shape == reference_output.shape

    # Compare outputs
    passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc)
    logger.info(comp_allclose(reference_output, tt_output_torch))
    logger.info(f"PCC: {pcc_message}")

    # Generate test summary message
    test_desc = f"Vision Transformer Model ({num_layers} layers)"

    if passing:
        logger.info(f"{test_desc} Passed!")
    else:
        logger.warning(f"{test_desc} Failed!")
    assert passing, f"PCC value is lower than {pcc} for some of the outputs. Check Warnings!"
