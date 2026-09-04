# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
TTTv2-style Embedding module for 1D-topology devices: N150 (1x1), N300 (1x2), T3K (1x8).

Single unified Embedding1D class:
  - forward(x): Token embedding lookup, optionally scaled by embed_scale.

Execution path:
  embedding(x, weights) → [multiply(embed_scale) if embed_scale != 1.0]

This module replaces both TTTv1 Embedding and ScaledEmbedding classes.
"""

from dataclasses import dataclass, replace

import ttnn

from tt_transformers.device_ownership import compatibility_default_device
from tt_transformers.modules.lazy_weight import LazyWeight, resolve_lazy_weight
from tt_transformers.modules.lightweightmodule import LightweightModule

# =============================================================================
# Top-level config dataclass
# =============================================================================


@dataclass
class Embedding1DConfig:
    """
    Central configuration for Embedding1D - the single source of truth for all settings.

    Simple usage (all defaults):
        config = Embedding1DConfig(weights=lazy_weights)

    With scale:
        config = Embedding1DConfig(weights=lazy_weights, embed_scale=3072**0.5)

    Full customization:
        config = Embedding1DConfig(
            weights=lazy_weights,
            mesh_device=custom_device,
            weights_memcfg=custom_memcfg,
        )
    """

    # Required: embedding weight table (vocab_size, dim), sharded on dim=-1
    weights: LazyWeight

    # Optional: device (derived from weights if None)
    mesh_device: ttnn.MeshDevice | None = None

    # Optional: scaling factor applied after lookup (1.0 = no scaling)
    embed_scale: float = 1.0

    # Optional: power-user overrides (None = compute defaults)
    weights_dtype: ttnn.DataType | None = None
    weights_memcfg: ttnn.MemoryConfig | None = None
    output_memcfg: ttnn.MemoryConfig | None = None

    def is_resolved(self) -> bool:
        """Check if all fields are resolved."""
        return all(getattr(self, f) is not None for f in self.__dataclass_fields__)


# =============================================================================
# Embedding1D - Unified Embedding for 1D-topology devices
# =============================================================================


class Embedding1D(LightweightModule):
    """
    Embedding for non-TG devices supporting token lookup with optional scaling.

    Replaces both TTTv1 Embedding and ScaledEmbedding.

    Simple API (90% of users):
        emb = Embedding1D(weights)

    With scaling:
        emb = Embedding1D(weights, embed_scale=3072**0.5)

    Power API (10% of users):
        config = Embedding1DConfig(weights=lazy_w, weights_memcfg=custom_cfg)
        emb = Embedding1D.from_config(config)

    Execution path:
      embedding(x, weights) → [multiply(embed_scale) if embed_scale != 1.0]
    """

    def __init__(self, weights: LazyWeight, embed_scale: float = 1.0):
        """
        Simple API - derives all config from weights.

        Args:
            weights: Embedding weight table (vocab_size, dim), sharded on last dim.
            embed_scale: Scale factor applied after lookup (default 1.0 = no scaling).
        """
        super().__init__()
        self.config = _resolve_embedding1d_config(Embedding1DConfig(weights=weights, embed_scale=embed_scale))
        self._device_weights_loaded = False

    @classmethod
    def from_config(cls, config: Embedding1DConfig):
        """
        Power API - any level of customization via config.

        Override any subset of fields in Embedding1DConfig:
            config = Embedding1DConfig(weights=w, embed_scale=2.0)
            emb = Embedding1D.from_config(config)
        """
        instance = object.__new__(cls)
        super(Embedding1D, instance).__init__()
        instance.config = _resolve_embedding1d_config(config)
        instance._device_weights_loaded = False
        return instance

    def load_device_weights(self):
        if self._device_weights_loaded:
            return

        assert self.config.is_resolved(), "config must be resolved before loading device weights!"
        self.weights = self.config.weights.get_device_weight()
        self._device_weights_loaded = True

    def forward(self, x: ttnn.Tensor | LazyWeight) -> ttnn.Tensor:
        """
        Token embedding lookup.

        Args:
            x: Token IDs tensor (uint32), shape [1, 1, 1, seq_len].

        Returns:
            Embedded tokens, shape [1, 1, seq_len, dim].
        """
        self.load_device_weights()
        x = _load_input_device_tensor(x, self.config)

        out = ttnn.embedding(
            x,
            self.weights,
            layout=ttnn.TILE_LAYOUT,
            memory_config=self.config.output_memcfg,
        )

        if self.config.embed_scale != 1.0:
            out = ttnn.multiply(out, self.config.embed_scale, memory_config=self.config.output_memcfg)

        return out

    # [INFO] this is the entry point for TTTv1 model_config.py and will retire with TTTv1


# =============================================================================
# Config resolution
# =============================================================================


def _resolve_embedding1d_config(config: Embedding1DConfig) -> Embedding1DConfig:
    """Materialize the config to known good defaults using replace pattern."""

    to_set = {}

    # --- Phase 1: Foundational fields ---

    # Derive mesh_device from weights
    mesh_device = config.mesh_device
    if mesh_device is None:
        mesh_device = config.weights.device
    if mesh_device is None:
        mesh_device = compatibility_default_device(ttnn, owner="modules.embedding._resolve_embedding1d_config")
    if config.mesh_device is None:
        to_set["mesh_device"] = mesh_device

    assert mesh_device is not None, "mesh_device must be available at this point!"

    # --- Phase 2: Default configs ---

    if config.weights_dtype is None:
        to_set["weights_dtype"] = ttnn.bfloat16

    if config.weights_memcfg is None:
        to_set["weights_memcfg"] = ttnn.DRAM_MEMORY_CONFIG

    if config.output_memcfg is None:
        to_set["output_memcfg"] = ttnn.DRAM_MEMORY_CONFIG

    # --- Phase 3: Resolve LazyWeight ---

    num_devices = mesh_device.get_num_devices()
    weights_memcfg = (
        config.weights_memcfg
        if config.weights_memcfg is not None
        else to_set.get("weights_memcfg", ttnn.DRAM_MEMORY_CONFIG)
    )
    weights_dtype = (
        config.weights_dtype if config.weights_dtype is not None else to_set.get("weights_dtype", ttnn.bfloat16)
    )

    to_set["weights"] = resolve_lazy_weight(
        config.weights,
        device=mesh_device,
        memory_config=weights_memcfg,
        mesh_mapper_config=ttnn.MeshMapperConfig(
            placements=[ttnn.PlacementShard(-1)],
            mesh_shape_override=ttnn.MeshShape([num_devices]),
        ),
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=weights_dtype,
    )

    resolved_config = replace(config, **to_set)
    assert resolved_config.is_resolved(), "Config must be resolved!"
    assert resolved_config.weights.is_resolved(), "Weights must be resolved!"

    return resolved_config


def _load_input_device_tensor(x: ttnn.Tensor | LazyWeight, config: Embedding1DConfig) -> ttnn.Tensor:
    """Resolve the input tensor to ttnn tensor if x is a LazyWeight, otherwise return as-is."""
    if isinstance(x, LazyWeight):
        resolved_x = resolve_lazy_weight(
            x,
            device=config.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper_config=None,  # replicated
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        return resolved_x.get_device_weight()

    assert isinstance(x, ttnn.Tensor), "x must be a ttnn tensor at this point!"
    return x
