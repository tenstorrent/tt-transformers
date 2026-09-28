# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Request-local multimodal state for Qwen3.8 serving.

The vision tower produces one packed embedding matrix for all media in a request.  Keeping
the grids, placeholder-to-row mapping, and M-RoPE result beside that matrix makes the state
portable across prefill implementations and, critically, prevents a later request from
overwriting process-global modality fields while an earlier request is still decoding.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import torch

Modality = Literal["image", "video"]


@dataclass(frozen=True)
class PlaceholderMapping:
    """Rows in the packed embeddings belonging to one media item."""

    modality: Modality
    grid_index: int
    token_positions: torch.Tensor
    embedding_start: int
    embedding_end: int


@dataclass
class VisionContext:
    """All multimodal data owned by one admitted request.

    ``embeddings`` is a packed host BF16 tensor in serving.  The vision tower is
    eager, while text prefill/decode use parked traces; materialising the final
    vision rows on the host lets the tower release every request-owned device
    allocation before a text trace is replayed.  The other fields are small host
    tensors prepared once before prefill and reused by every chunk/trace of that
    request.
    """

    embeddings: object
    image_grid_thw: torch.Tensor | None = None
    video_grid_thw: torch.Tensor | None = None
    image_token_id: int | None = None
    video_token_id: int | None = None
    placeholder_mapping: tuple[PlaceholderMapping, ...] = field(default_factory=tuple)
    mrope_cos: torch.Tensor | None = None
    mrope_sin: torch.Tensor | None = None
    rope_delta: int = 0

    @classmethod
    def for_images(cls, embeddings, image_grid_thw, *, image_token_id: int) -> VisionContext:
        return cls(
            embeddings=embeddings,
            image_grid_thw=_normalise_grid(image_grid_thw),
            image_token_id=int(image_token_id),
        )

    @classmethod
    def for_videos(cls, embeddings, video_grid_thw, *, video_token_id: int) -> VisionContext:
        return cls(
            embeddings=embeddings,
            video_grid_thw=_normalise_grid(video_grid_thw),
            video_token_id=int(video_token_id),
        )

    @property
    def has_images(self) -> bool:
        return self.image_grid_thw is not None

    @property
    def has_videos(self) -> bool:
        return self.video_grid_thw is not None

    def token_id_for(self, modality: Modality) -> int:
        token_id = self.image_token_id if modality == "image" else self.video_token_id
        if token_id is None:
            raise ValueError(f"{modality} token id is missing from VisionContext")
        return token_id

    def prepare(self, input_ids: torch.Tensor, rope_setup) -> VisionContext:
        """Validate placeholders and compute this request's immutable M-RoPE state."""
        from tt_transformers.models.qwen38.attention.rope_tp import get_rope_index, get_rot_mats

        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        if input_ids.shape[0] != 1:
            raise ValueError(f"VisionContext preparation currently requires B=1, got {tuple(input_ids.shape)}")
        input_ids = input_ids.to(torch.long)

        mappings = []
        embedding_offset = 0
        for modality, grids in (("image", self.image_grid_thw), ("video", self.video_grid_thw)):
            if grids is None:
                continue
            token_id = self.token_id_for(modality)
            positions = torch.nonzero(input_ids[0] == token_id, as_tuple=False).reshape(-1)
            item_sizes = [
                int(grid[0])
                * (int(grid[1]) // int(rope_setup.spatial_merge_size))
                * (int(grid[2]) // int(rope_setup.spatial_merge_size))
                for grid in grids
            ]
            expected = sum(item_sizes)
            if int(positions.numel()) != expected:
                raise ValueError(
                    f"{modality} placeholder count {int(positions.numel())} does not match "
                    f"the grids' {expected} packed tokens"
                )
            position_offset = 0
            for grid_index, size in enumerate(item_sizes):
                mappings.append(
                    PlaceholderMapping(
                        modality=modality,
                        grid_index=grid_index,
                        token_positions=positions[position_offset : position_offset + size].clone(),
                        embedding_start=embedding_offset,
                        embedding_end=embedding_offset + size,
                    )
                )
                position_offset += size
                embedding_offset += size

        if int(self.embeddings.shape[0]) != embedding_offset:
            raise ValueError(
                f"packed vision embeddings have {int(self.embeddings.shape[0])} rows, "
                f"but placeholders require {embedding_offset}"
            )

        mm_token_type_ids = torch.zeros_like(input_ids)
        if self.image_grid_thw is not None:
            mm_token_type_ids[input_ids == self.token_id_for("image")] = 1
        if self.video_grid_thw is not None:
            mm_token_type_ids[input_ids == self.token_id_for("video")] = 2
        position_ids, deltas = get_rope_index(
            input_ids,
            mm_token_type_ids,
            image_grid_thw=self.image_grid_thw,
            video_grid_thw=self.video_grid_thw,
            spatial_merge_size=rope_setup.spatial_merge_size,
        )
        cos, sin = get_rot_mats(
            rope_setup.inv_freq,
            position_ids,
            rope_setup.mrope_section,
            rope_setup.attention_scaling,
        )
        self.placeholder_mapping = tuple(mappings)
        self.mrope_cos = cos[0].to(torch.bfloat16)
        self.mrope_sin = sin[0].to(torch.bfloat16)
        self.rope_delta = int(deltas[0, 0].item())
        return self

    def placeholder_positions(self, modality: Modality | None = None) -> torch.Tensor:
        positions = [
            item.token_positions for item in self.placeholder_mapping if modality is None or item.modality == modality
        ]
        return torch.cat(positions) if positions else torch.empty(0, dtype=torch.long)

    def placeholder_mask(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Boolean mask for every image/video placeholder owned by this request."""
        mask = torch.zeros_like(token_ids, dtype=torch.bool)
        if self.image_grid_thw is not None:
            mask |= token_ids == self.token_id_for("image")
        if self.video_grid_thw is not None:
            mask |= token_ids == self.token_id_for("video")
        return mask


def _normalise_grid(grid_thw) -> torch.Tensor:
    grid = grid_thw if isinstance(grid_thw, torch.Tensor) else torch.as_tensor(grid_thw)
    grid = grid.to(dtype=torch.long)
    if grid.dim() == 1:
        grid = grid.unsqueeze(0)
    if grid.dim() != 2 or grid.shape[1] != 3:
        raise ValueError(f"grid_thw must have shape [N, 3], got {tuple(grid.shape)}")
    return grid
