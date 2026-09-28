# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os

import torch
import ttnn
from loguru import logger
from transformers.vision_utils import get_vision_interpolation_indices_and_weights

from tt_transformers.models.qwen38.v1.ccl import TT_CCL
from tt_transformers.models.qwen38.v1.common import get_rot_transformation_mat
from tt_transformers.models.qwen38.v1.load_checkpoints import (
    convert_hf_to_meta,
    convert_rope_style_hf_to_meta,
    standardize_hf_keys_multimodal,
)
from tt_transformers.models.qwen38.v1.utility import comp_pcc
from tt_transformers.models.qwen38.vision.functional import qwen3_5_vision_transformer_preprocess
from tt_transformers.modules.lightweightmodule import LightweightModule

from .patch_merger import PatchMerger
from .vision_block import VisionBlock
from .vision_model_config import VisionModelArgs


def interpolate_pos_embeddings(reference_model, grid_thw, dtype=None):
    """Build Qwen3.5 learned vision positions with the Transformers 5.17 public API."""
    interp_indices, interp_weights = get_vision_interpolation_indices_and_weights(
        grid_thw,
        num_grid_per_side=reference_model.num_grid_per_side,
        mode=reference_model.interpolation_mode,
        align_corners=reference_model.interpolation_align_corners,
        spatial_merge_size=reference_model.config.spatial_merge_size,
    )
    pos_embeds = (reference_model.pos_embed(interp_indices) * interp_weights[:, :, None]).sum(1)
    return pos_embeds if dtype is None else pos_embeds.to(dtype)


class VisionTransformer(LightweightModule):
    """
    Vision Transformer model for Qwen 3 VL.
    This implements only the transformer blocks part of the vision transformer.
    Patch embedding and merging should be done outside this class.
    """

    def __init__(
        self,
        args,
        dtype,
        state_dict,
        weight_cache_path,
        tt_ccl=None,
    ):
        """
        Initialize the Vision Transformer model.

        Args:
            args (VisionModelArgs): Model arguments
            dtype (ttnn.dtype): Data type for computations
            state_dict (dict): State dictionary containing model weights
            weight_cache_path (str): Path to weight cache
            tt_ccl (TT_CCL, optional): CCL helper. The vision tower is always
                tensor-parallel, so one is constructed internally if not given.
        """
        super().__init__()
        self.args = args
        self.dtype = dtype
        self.weight_cache_path = weight_cache_path

        # The vision tower is always tensor-parallel and needs a TT_CCL on hand.
        self.tt_ccl = tt_ccl if tt_ccl is not None else TT_CCL(args.mesh_device)

        # Create transformation matrix for RoPE QK prefill
        transformation_mat_torch = get_rot_transformation_mat(
            args.head_dim
        )  # todo)) args.head_dim is ignored inside the function
        self.transformation_mats = {
            "prefill": ttnn.as_tensor(
                transformation_mat_torch,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=args.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(args.mesh_device),
            )
        }

        # Create vision blocks
        self.blocks = []
        for i in range(args.hf_config.vision_config.depth):
            block = VisionBlock(
                mesh_device=args.mesh_device,
                state_dict=state_dict,
                weight_cache_path=weight_cache_path,
                layer_num=i,
                dtype=dtype,
                transformation_mats=self.transformation_mats,
                args=args,
                tt_ccl=self.tt_ccl,
            )
            self.blocks.append(block)

        # The Megatron-style PatchMerger consumes a fractured-along-dim=3 tensor
        # directly (no pre-merger all-gather) and produces a fractured-along-dim=3
        # output, mirroring the LLM's DistributedNorm + LMHead final stretch in
        # `tt_transformers.tt.model`.
        self.patch_merger = PatchMerger(
            mesh_device=args.mesh_device,
            args=args,
            state_dict=state_dict,
            weight_cache_path=weight_cache_path,
            state_dict_prefix=args.get_state_dict_prefix("PatchMerger"),
            dtype=dtype,
            tt_ccl=self.tt_ccl,
        )

    def prepare_input(self, patch_input, seq_len=None):
        """Convert a patchified torch input to a ttnn tensor
        Args:
            patch_input (torch.Tensor): Patchified input tensor
            seq_len (int): Sequence length

        Returns:
            ttnn.Tensor: Prepared input tensor
        """
        patch_seq_len, _ = patch_input.shape
        x = patch_input
        seq_len = ((patch_seq_len // 128) + 1) * 128 if seq_len is None else seq_len
        x = torch.nn.functional.pad(x, (0, 0, 0, seq_len - patch_seq_len)).unsqueeze(0)
        x = self.args.prepare_residual_tensor_prefill(x)
        return x

    def forward(
        self,
        x,
        unpadded_seq_len,
        rot_mats,
    ):
        """
        Forward pass through the Vision Transformer blocks.

        Args:
            x (ttnn.Tensor): Input tensor [batch_size, 1, seq_len, hidden_dim].
                This arrives fractured along dim=3 (the hidden dim) because
                `prepare_residual_tensor_prefill` shards at load time.
            unpadded_seq_len (int): Real sequence length before padding.
            rot_mats (list): Rotation matrices for positional embeddings.

        Returns:
            ttnn.Tensor: Output tensor, fractured along dim=3 (each device owns
                out_hidden_size/TP), matching the LLM lm_head contract.
        """
        for i, block in enumerate(self.blocks):
            x = block(
                x,
                rot_mats=rot_mats,
            )

        # The PatchMerger consumes the block output fractured along dim=3
        # directly (its first op is a DistributedLayerNorm that all-gathers
        # internally) and produces a fractured-along-dim=3 output. No
        # pre-merger all-gather is needed.
        # Keep the padded bucket through the merger. Slicing here made the
        # merger program depend on exact processor output U even though every
        # transformer block already runs at fixed S. U is divisible by the
        # spatial merge area, so trimming merged host rows later preserves all
        # real outputs.
        x = self.patch_merger(x)
        return x


class DropInVisionTransformer(torch.nn.Module):
    """Wraps VisionTransformer to be a drop-in replacement for
    Qwen2_5_VisionTransformerPretrainedModel. It uses the reference model
    for certain preprocessing steps like patch embedding and index calculation.
    """

    def __init__(
        self,
        reference_model,
        model_args: VisionModelArgs,
        dtype=ttnn.bfloat8_b,
        debug=False,
        tt_ccl=None,
    ):
        """
        Initialize the TorchVisionTransformer wrapper.

        Args:
            reference_model (Qwen2_5_VisionTransformerPretrainedModel): Initialized reference HF model instance.
            model_args (VisionModelArgs): Model configuration arguments.
            dtype (ttnn.dtype): Compute dtype for weights.
            debug (bool): If True, run the reference path alongside and report PCC.
            tt_ccl (TT_CCL, optional): Reuse a CCL helper across multiple models.
        """
        super().__init__()
        self.reference_model = reference_model
        self.model_args = model_args
        self.debug = debug

        state_dict = standardize_hf_keys_multimodal(reference_model.state_dict())
        state_dict = convert_hf_to_meta(state_dict, model_args.head_dim)
        state_dict_prefix = model_args.get_state_dict_prefix("VisionTransformer")
        state_dict = {f"{state_dict_prefix}.{k}": v for k, v in state_dict.items()}

        # Initialize TT model
        self.tt_model = VisionTransformer(
            args=model_args,
            state_dict=state_dict,
            weight_cache_path=model_args.weight_cache_path(dtype),
            dtype=dtype,
            tt_ccl=tt_ccl,
        )

    @property
    def dtype(self):
        return self.reference_model.dtype

    @property
    def spatial_merge_size(self):
        return self.model_args.hf_config.vision_config.spatial_merge_size

    def forward(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        """
        Forward pass mimicking the Qwen3_5_VisionTransformerPretrainedModel interface.

        Args:
            pixel_values (torch.Tensor): Input pixel values tensor (equivalent to hidden_states for the ref model start).
                                         Shape typically [num_patches, hidden_size] or similar before patch_embed.
            grid_thw (torch.Tensor): Tensor describing the grid dimensions (time, height, width) for each image/video.
                                     Shape [num_images_or_videos, 3].

        Returns:
            torch.Tensor: Output tensor with shape [1, B, seq_len, out_hidden_size].
        """
        if not isinstance(pixel_values, torch.Tensor) or pixel_values.dim() != 2:
            raise ValueError(
                "pixel_values must be a packed rank-2 torch tensor "
                f"[num_patches, patch_dim], got {type(pixel_values).__name__} "
                f"with shape {getattr(pixel_values, 'shape', None)}"
            )
        if not isinstance(grid_thw, torch.Tensor):
            grid_thw = torch.as_tensor(grid_thw)
        if grid_thw.dim() == 1:
            grid_thw = grid_thw.unsqueeze(0)
        if grid_thw.dim() != 2 or grid_thw.shape[1] != 3:
            raise ValueError(f"grid_thw must have shape [N, 3], got {tuple(grid_thw.shape)}")
        if grid_thw.numel() == 0 or bool((grid_thw <= 0).any()):
            raise ValueError("grid_thw must contain one or more strictly positive (t,h,w) rows")
        merge = int(self.spatial_merge_size)
        if bool((grid_thw[:, 1:] % merge != 0).any()):
            raise ValueError(
                f"grid height and width must be divisible by spatial_merge_size={merge}: {grid_thw.tolist()}"
            )
        expected_patches = sum(int(row.prod().item()) for row in grid_thw)
        if int(pixel_values.shape[0]) != expected_patches:
            raise ValueError(
                f"pixel_values has {int(pixel_values.shape[0])} packed rows, but grid_thw requires {expected_patches}"
            )
        max_patches = int(os.environ.get("QWEN36_VISION_MAX_PATCHES", "4095"))
        oversized = [int(row.prod().item()) for row in grid_thw if int(row.prod().item()) > max_patches]
        if oversized:
            raise ValueError(
                "processed image exceeds the pre-warmed vision limit: "
                f"patches={oversized}, maximum={max_patches}; configure the Qwen image processor "
                "with max_pixels <= maximum*patch_size**2"
            )

        # Process each image separately, as V1 does. Returned rows are host
        # owned, so no request-local vision allocation survives into parked
        # text or DFlash traces.
        all_pixel_values = pixel_values
        all_grid_thw = grid_thw
        final_outputs = []
        # todo)) refactor this code to leverage tt-mesh's ttnn.ShardTensorToMesh(mesh_device, dim=batch_size_dim) for data parallelism
        for grid_thw in all_grid_thw:
            # --- pick out the pixel_values for this users' images (grid_thw.prod() pixels) ---
            image_patch_count = int(grid_thw.prod().item())
            pixel_values = all_pixel_values[:image_patch_count, :]
            all_pixel_values = all_pixel_values[image_patch_count:, :]
            # --- Preprocessing ---
            # 1. Calculate total unpadded sequence length
            grid_thw = grid_thw.unsqueeze(0)
            unpadded_seq_len = grid_thw.prod(dim=1).sum().item()
            # Calculate padded sequence length (divisible by 2048) required by models/tt_transformers/tt/attention.py::forward_prefill
            seq_len = ((unpadded_seq_len // 2048) + 1) * 2048

            # 2. Use preprocessing function from reference/functional to get indices and embeddings
            cu_seqlens, position_embeddings = qwen3_5_vision_transformer_preprocess(
                seq_len=unpadded_seq_len,
                grid_thw=grid_thw,
                head_dim=self.model_args.head_dim,
                spatial_merge_size=self.model_args.hf_config.vision_config.spatial_merge_size,
            )

            # 3. Use reference model's patch embedding
            patch_input = self.reference_model.patch_embed(pixel_values)
            pos_embeds = interpolate_pos_embeddings(self.reference_model, grid_thw, dtype=patch_input.dtype)
            patch_input = patch_input + pos_embeds

            # 4. Prepare rotational embeddings (cos, sin) -> pad -> convert to TT tensors
            cos_orig, sin_orig = position_embeddings
            cos_orig, sin_orig = convert_rope_style_hf_to_meta(cos_orig, sin_orig)
            # pad sequence length with cos = 1, sin = 0 (identity rotation)
            cos_padded = (
                torch.nn.functional.pad(cos_orig, (0, 0, 0, seq_len - unpadded_seq_len), value=1)
                .unsqueeze(0)
                .unsqueeze(0)
            )
            sin_padded = (
                torch.nn.functional.pad(sin_orig, (0, 0, 0, seq_len - unpadded_seq_len), value=0)
                .unsqueeze(0)
                .unsqueeze(0)
            )
            # Convert to TT tensors on the mesh device
            cos = ttnn.from_torch(
                cos_padded,
                dtype=ttnn.bfloat16,  # Use bfloat16 for RoPE
                layout=ttnn.TILE_LAYOUT,
                device=self.model_args.mesh_device,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.model_args.mesh_device),
            )
            sin = ttnn.from_torch(
                sin_padded,
                dtype=ttnn.bfloat16,  # Use bfloat16 for RoPE
                layout=ttnn.TILE_LAYOUT,
                device=self.model_args.mesh_device,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.model_args.mesh_device),
            )
            rot_mats = [cos, sin]

            # 5. Prepare input tensor for the TT model using window_index
            tt_input = self.tt_model.prepare_input(patch_input, seq_len)

            # --- TT Model Execution ---
            tt_out = self.tt_model(
                tt_input,
                unpadded_seq_len=unpadded_seq_len,
                rot_mats=rot_mats,  # Use rot_mats generated in this forward pass
            )

            # deallocate device tensors that are not needed by decode
            ttnn.deallocate(tt_input)
            ttnn.deallocate(cos)
            ttnn.deallocate(sin)

            # --- Postprocessing ---
            # 1. Extract the relevant output part and adjust shape (matching test logic).
            # The merger output is fractured along dim=3, so each device owns
            # out_hidden_size/TP channels.
            out_hidden_size = (
                self.model_args.hf_config.vision_config.out_hidden_size // self.model_args.cluster_shape[1]
            )
            # Output shape from TT is [1, B=1, S, H_out_padded], slice H and squeeze B, batch dims.
            # The slice/reshape args are GLOBAL shapes; the tensor is already
            # fractured along dim=3 and ttnn handles the per-device extents internally.
            final_output = ttnn.reshape(tt_out[:, 0:1, :, :out_hidden_size], (-1, out_hidden_size))
            final_output_host = ttnn.to_torch(
                final_output,
                mesh_composer=ttnn.ConcatMeshToTensor(self.model_args.mesh_device, dim=1),
            ).to(torch.bfloat16)
            real_merged_rows = unpadded_seq_len // (merge**2)
            final_output_host = final_output_host[:real_merged_rows].contiguous()
            ttnn.deallocate(tt_out)

            if self.debug:
                logger.info("DropInVisionTransformer: Debug enabled, running reference model...")
                reference_output = self.reference_model.forward(pixel_values, grid_thw)
                _, pcc = comp_pcc(reference_output, final_output_host)
                logger.info(f"DropInVisionTransformer: PCC to reference model: {pcc}")

            # 2. The merger already produces a tensor fractured along the hidden
            # dim (dim=3 in 4D / dim=1 in the 2D-reshaped view), which is the
            # desired output sharding.
            # final_output_sharded = final_output

            # 3. Aggregate in batched users list
            final_outputs.append(final_output_host)

        # Host packing is cheap after the merger's 4x sequence reduction and
        # keeps the request lifetime independent of TT allocation.
        return final_outputs[0] if len(final_outputs) == 1 else torch.cat(final_outputs, dim=0)
