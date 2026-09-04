# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Neutral page-table and mesh/submesh construction helpers."""

from __future__ import annotations

import torch
import ttnn
from loguru import logger


def make_contiguous_page_table(batch_size: int, max_seq_len: int, block_size: int = 32) -> torch.Tensor:
    """Create a contiguous demo page table with one disjoint block range per user."""
    if min(batch_size, max_seq_len, block_size) <= 0:
        raise ValueError("page-table dimensions must be positive")
    blocks_per_user = (max_seq_len + block_size - 1) // block_size
    return torch.arange(batch_size * blocks_per_user, dtype=torch.int32).reshape(batch_size, blocks_per_user)


def _mesh_shape_tuple(mesh_shape):
    return tuple(int(dim) for dim in mesh_shape)


def _galaxy_data_parallel_submesh_shape(devices_per_group):
    # Galaxy DP groups should follow the 4x8 row-oriented view recommended by
    # the runtime, so DP=4 maps to four routeable 1x8 T3K-like submeshes.
    if devices_per_group >= 8 and devices_per_group % 8 == 0:
        return ttnn.MeshShape(devices_per_group // 8, 8)
    # Smaller DP groups still use contiguous 1D row submeshes; callers select
    # linear CCL when these groups are too small for ring topology.
    return ttnn.MeshShape(1, devices_per_group)


def create_submeshes(mesh_device, data_parallel):
    mesh_device_type = getattr(ttnn, "MeshDevice", None)
    if mesh_device_type is None:
        mesh_device_type = getattr(getattr(ttnn, "device", None), "Device", None)
    if mesh_device_type is None or not isinstance(mesh_device, mesh_device_type) or data_parallel == 1:
        return [mesh_device]

    num_rows, num_cols = _mesh_shape_tuple(mesh_device.shape)
    num_devices = num_rows * num_cols
    assert num_devices % data_parallel == 0, f"Unsupported device split: {num_devices} devices, {data_parallel} groups"

    if num_devices == 32:
        if (num_rows, num_cols) != (4, 8):
            logger.info(f"Reshaping 32-device mesh from {(num_rows, num_cols)} to (4, 8) for DP submeshes")
            mesh_device.reshape(ttnn.MeshShape(4, 8))
        return mesh_device.create_submeshes(_galaxy_data_parallel_submesh_shape(num_devices // data_parallel))

    return mesh_device.create_submeshes(ttnn.MeshShape(1, num_devices // data_parallel))
