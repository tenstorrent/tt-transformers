# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""RoPE setup for Qwen3.5-9B Gated Attention layers.

Qwen3.5 uses partial rotary embeddings: only 25% of the head dimensions
(64 out of 256) receive rotary position encoding. The remaining 192 dimensions
pass through unchanged. The gated attention TTNN op handles the partial
application internally — we just need to generate cos/sin for the rotary
portion (head_dim=64).
"""

import torch
import ttnn


def compute_rope_freqs(head_dim: int, max_seq_len: int, theta: float = 10_000_000.0):
    """Compute RoPE frequency tensors (cos, sin) for given head_dim.

    Args:
        head_dim: Dimension of the rotary portion (64 for Qwen3.5).
        max_seq_len: Maximum sequence length to precompute.
        theta: RoPE base frequency.

    Returns:
        cos: torch.Tensor [max_seq_len, head_dim]
        sin: torch.Tensor [max_seq_len, head_dim]
    """
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    positions = torch.arange(max_seq_len, dtype=torch.float32)
    angles = torch.outer(positions, freqs)  # [max_seq_len, head_dim // 2]
    cos = torch.cat([torch.cos(angles), torch.cos(angles)], dim=-1)  # [max_seq_len, head_dim]
    sin = torch.cat([torch.sin(angles), torch.sin(angles)], dim=-1)  # [max_seq_len, head_dim]
    return cos, sin


class Qwen36RoPESetup:
    """Precomputes and stores RoPE cos/sin tensors for Qwen3.5.

    Usage:
        rope = Qwen36RoPESetup(device, args)
        cos, sin = rope.get_rot_mats(position_ids)
    """

    def __init__(self, device, args):
        self.device = device
        self.head_dim = args.rope_head_dim  # 64
        self.max_seq_len = args.max_seq_len
        self.theta = args.rope_theta

        self.cos_cpu, self.sin_cpu = compute_rope_freqs(
            head_dim=self.head_dim,
            max_seq_len=self.max_seq_len,
            theta=args.rope_theta,
        )

        # --- M-RoPE (multimodal rotary) request-independent configuration ----------------------
        # Per-request cos/sin and rope_delta live in VisionContext.  Keeping them here used to let
        # the next prefill overwrite the request still being decoded.
        self.mrope_section = list(args.mrope_section)
        self.attention_scaling = args.rope_attention_scaling
        self.spatial_merge_size = args.spatial_merge_size
        self.image_token_id = args.image_token_id
        self.video_token_id = args.video_token_id
        self.inv_freq = 1.0 / (self.theta ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim))

        # Pre-compute full RoPE table on device for fast decode lookups
        # Shape: [1, max_seq_len, head_dim] on device
        # mesh_mapper replicates to all devices; on a 1-device mesh this is a no-op.
        self.cos_device = ttnn.from_torch(
            self.cos_cpu.unsqueeze(0),  # [1, max_seq_len, head_dim]
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device),
        )
        self.sin_device = ttnn.from_torch(
            self.sin_cpu.unsqueeze(0),  # [1, max_seq_len, head_dim]
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device),
        )

    def get_rot_mats(self, position_ids: torch.Tensor):
        """Get cos/sin matrices for given positions.

        Args:
            position_ids: torch.Tensor [B, T] or [T] — position indices.

        Returns:
            cos_ttnn: ttnn.Tensor [B, T, head_dim] on device
            sin_ttnn: ttnn.Tensor [B, T, head_dim] on device
        """
        if position_ids.dim() == 1:
            position_ids = position_ids.unsqueeze(0)

        B, T = position_ids.shape

        # Fast path for single-position decode: slice from pre-computed device table
        if T == 1 and B == 1:
            pos = position_ids.item()
            cos = self.cos_device[:, pos : pos + 1, :]
            sin = self.sin_device[:, pos : pos + 1, :]
            return cos, sin

        # General path for prefill (variable positions)
        flat_pos = position_ids.reshape(-1)
        cos = self.cos_cpu[flat_pos].reshape(B, T, self.head_dim)
        sin = self.sin_cpu[flat_pos].reshape(B, T, self.head_dim)

        cos_ttnn = ttnn.from_torch(
            cos,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
        )
        sin_ttnn = ttnn.from_torch(
            sin,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
        )
        return cos_ttnn, sin_ttnn

    def get_cos_sin_host(self, pos):
        """Return cos/sin at position as host ttnn tensors for copy_host_to_device_tensor.

        Returns tensors on HOST (no device= arg) for fast DMA to pre-allocated device buffers.
        Shape: [1, 1, rope_head_dim] — must match _trace_cos/_trace_sin device buffer shapes.
        Layout: TILE_LAYOUT — must match device buffer layout for copy compatibility.

        `pos` is the ROPE position (= KV position + rope_delta for a multimodal request); the
        caller is responsible for the offset so decode reads the absolute 1D table correctly.
        """
        cos = self.cos_cpu[pos : pos + 1].unsqueeze(0).contiguous()  # [1, 1, 64]
        sin = self.sin_cpu[pos : pos + 1].unsqueeze(0).contiguous()  # [1, 1, 64]
        cos_host = ttnn.from_torch(cos, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        sin_host = ttnn.from_torch(sin, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        return cos_host, sin_host

    # -------------------------------------------------------------------------
    # M-RoPE (multimodal) — per-request 3D position handling
    # -------------------------------------------------------------------------
    def prepare_context(self, input_ids, vision_context):
        """Populate M-RoPE fields on ``vision_context``; text has no mutable rope state."""
        if vision_context is None:
            return 0
        vision_context.prepare(input_ids, self)
        return vision_context.rope_delta

    def _extend_context_table(self, vision_context, length):
        """Grow the per-request M-RoPE table to >= `length` positions with text-continuation rows
        (post-prompt positions have t==h==w, advancing as rope_pos = seq_idx + rope_delta). Used so
        the masked-bucket padding past the real prompt still has cos/sin."""
        cur = vision_context.mrope_cos.shape[0]
        if length <= cur:
            return
        pos = torch.arange(cur, length, dtype=torch.float32) + vision_context.rope_delta
        emb = torch.cat([torch.outer(pos, self.inv_freq)] * 2, dim=-1)
        vision_context.mrope_cos = torch.cat([vision_context.mrope_cos, emb.cos().to(torch.bfloat16)], dim=0)
        vision_context.mrope_sin = torch.cat([vision_context.mrope_sin, emb.sin().to(torch.bfloat16)], dim=0)

    def prefill_cos_sin_torch(self, start, length, vision_context=None):
        """Torch bf16 cos/sin [length, head_dim] for SEQUENCE positions [start, start+length).

        Uses the per-request M-RoPE table when staged (build_request_rope); otherwise ordinary 1D
        RoPE at absolute positions [start, start+length) — byte-identical to the pre-M-RoPE path."""
        if vision_context is not None:
            end = start + length
            if end > vision_context.mrope_cos.shape[0]:
                self._extend_context_table(vision_context, end)
            return vision_context.mrope_cos[start:end], vision_context.mrope_sin[start:end]
        t = torch.arange(start, start + length, dtype=torch.float32)
        emb = torch.cat([torch.outer(t, self.inv_freq)] * 2, dim=-1)
        return emb.cos().to(torch.bfloat16), emb.sin().to(torch.bfloat16)

    def get_prefill_rot_mats(self, start, length, vision_context=None):
        """ttnn cos/sin [1, length, head_dim] (replicated) for SEQUENCE positions [start, start+length),
        M-RoPE-aware. Drop-in for the prefill sites that previously called get_rot_mats(arange(...))."""
        cos_t, sin_t = self.prefill_cos_sin_torch(start, length, vision_context=vision_context)
        cos = ttnn.from_torch(
            cos_t.unsqueeze(0),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
        )
        sin = ttnn.from_torch(
            sin_t.unsqueeze(0),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
        )
        return cos, sin
