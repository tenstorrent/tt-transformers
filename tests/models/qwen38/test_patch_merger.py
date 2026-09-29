# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from loguru import logger

from tests.models.qwen38.test_factory import _resolve_mesh_shape
from tt_transformers.models.qwen38.v1.ccl import TT_CCL
from tt_transformers.models.qwen38.v1.load_checkpoints import convert_hf_to_meta
from tt_transformers.models.qwen38.v1.utility import comp_allclose, comp_pcc
from tt_transformers.models.qwen38.vision.patch_merger import PatchMerger
from tt_transformers.models.qwen38.vision.vision_model_config import VisionModelArgs


@torch.no_grad()
@pytest.mark.parametrize(
    "mesh_device",
    [_resolve_mesh_shape()],
    indirect=True,
)
@pytest.mark.parametrize(
    "rows",
    (14308,),  # from 3B test image
)
@pytest.mark.parametrize(
    "batch_size",
    (1,),
)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.device
@pytest.mark.model
def test_patch_merger_inference(rows, batch_size, mesh_device, reset_seeds, ensure_gc):
    dtype = ttnn.bfloat8_b

    model_args = VisionModelArgs(mesh_device, dummy_weights=True, max_batch_size=batch_size, max_seq_len=rows)

    # Create reference model with correct dimensions
    reference_model = model_args.reference_patch_merger()

    state_dict = convert_hf_to_meta(reference_model.state_dict(), model_args.head_dim)
    state_dict_prefix = model_args.get_state_dict_prefix("PatchMerger")
    state_dict = {f"{state_dict_prefix}.{k}": v for k, v in state_dict.items()}

    tt_ccl = TT_CCL(mesh_device)
    tt_model = PatchMerger(
        mesh_device=mesh_device,
        args=model_args,
        state_dict=state_dict,
        state_dict_prefix=state_dict_prefix,
        weight_cache_path=None,  # Don't cache random weights
        dtype=dtype,
        tt_ccl=tt_ccl,
    )

    # Input shape should match context_dim. The TP PatchMerger consumes a tensor
    # fractured along dim=3 (the hidden dim), matching the vision block output.
    torch_input = torch.randn(batch_size, 1, rows, model_args.hf_config.vision_config.hidden_size, dtype=torch.bfloat16)
    reference_output = reference_model(torch_input)

    tt_input = ttnn.from_torch(
        torch_input,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(
            mesh_device,
            dims=(None, -1),
            mesh_shape=model_args.cluster_shape,
        ),
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )

    logger.info("Run PatchMerger")
    tt_output = tt_model(tt_input)

    # The TP PatchMerger output is fractured along dim=3 (out_hidden_size); concat
    # along that axis to reassemble the full output.
    tt_output_torch = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=3),
    )
    tt_output_torch = (
        tt_output_torch[:, 0:1, :, : model_args.hf_config.vision_config.out_hidden_size].squeeze(0).squeeze(0)
    )

    pcc_required = 0.99
    passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc_required)

    logger.info(comp_allclose(reference_output, tt_output_torch))
    logger.info(f"PCC: {pcc_message}")
    if passing:
        logger.info("PatchMerger Passed!")
    else:
        logger.warning("PatchMerger Failed!")

    assert passing, f"PatchMerger output does not meet PCC requirement {pcc_required}: {pcc_message}."
