# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace

import pytest
import torch
import ttnn

from tt_transformers import tensor_utils
from tt_transformers.tensor_utils import (
    get_rot_transformation_mat,
    load_cached_tensor,
    pad_dim_to_size,
    pad_to_shape,
    parse_shard_dims_from_mesh_mapper_config,
    program_config_to_dict,
    program_config_to_str,
    zeros_like_kv_cache,
    zeros_like_paged_cache,
)


@pytest.mark.host
def test_pad_dim_to_size(expect_error):
    """Test the pad_dim_to_size utility function."""

    # Test padding on last dimension
    x = torch.randn(1, 1, 32, 100)
    padded = pad_dim_to_size(x, dim=-1, size=128)
    assert padded.shape == (1, 1, 32, 128)

    # Original data should be preserved
    assert torch.allclose(padded[:, :, :, :100], x)

    # Padding should be zeros
    assert torch.allclose(padded[:, :, :, 100:], torch.zeros(1, 1, 32, 28))

    # Test no padding needed
    x2 = torch.randn(1, 1, 32, 128)
    padded2 = pad_dim_to_size(x2, dim=-1, size=128)
    assert torch.equal(padded2, x2)

    # Test padding on different dimension
    x3 = torch.randn(1, 1, 24, 128)
    padded3 = pad_dim_to_size(x3, dim=-2, size=32)
    assert padded3.shape == (1, 1, 32, 128)

    # Test error when target size is smaller
    with expect_error(ValueError, "smaller than current size"):
        pad_dim_to_size(x, dim=-1, size=50)


@pytest.mark.host
def test_pad_dim_to_size_positive_dim():
    """Test pad_dim_to_size with positive dimension index."""

    x = torch.randn(2, 3, 4, 5)

    # Pad dim=0
    padded = pad_dim_to_size(x, dim=0, size=4)
    assert padded.shape == (4, 3, 4, 5)
    assert torch.equal(padded[:2], x)

    # Pad dim=1
    padded = pad_dim_to_size(x, dim=1, size=8)
    assert padded.shape == (2, 8, 4, 5)
    assert torch.equal(padded[:, :3], x)


@pytest.mark.host
def test_pad_to_shape():
    """Test the pad_to_shape utility function."""

    # Pad multiple dimensions at once
    x = torch.randn(1, 2, 24, 100)
    padded = pad_to_shape(x, (1, 4, 32, 128))
    assert padded.shape == (1, 4, 32, 128)

    # Original data preserved
    assert torch.allclose(padded[:, :2, :24, :100], x)

    # Padding is zeros
    assert torch.allclose(padded[:, 2:, :, :], torch.zeros(1, 2, 32, 128))
    assert torch.allclose(padded[:, :, 24:, :], torch.zeros(1, 4, 8, 128))
    assert torch.allclose(padded[:, :, :, 100:], torch.zeros(1, 4, 32, 28))


@pytest.mark.host
def test_pad_to_shape_no_op():
    """Test pad_to_shape returns same tensor when no padding needed."""

    x = torch.randn(1, 2, 32, 128)
    padded = pad_to_shape(x, (1, 2, 32, 128))
    assert padded is x  # Should be the exact same object


@pytest.mark.host
def test_pad_to_shape_single_dim():
    """Test pad_to_shape with only one dimension needing padding."""

    x = torch.randn(2, 3, 4, 5)
    padded = pad_to_shape(x, (2, 3, 4, 8))
    assert padded.shape == (2, 3, 4, 8)
    assert torch.equal(padded[:, :, :, :5], x)


@pytest.mark.host
def test_pad_to_shape_error_on_smaller_target(expect_error):
    """Test pad_to_shape raises error when target is smaller than source."""

    x = torch.randn(2, 3, 4, 5)
    with expect_error(ValueError, "smaller than current size"):
        pad_to_shape(x, (2, 3, 4, 3))


@pytest.mark.host
def test_parse_shard_dims_from_mesh_mapper_config():
    """Test parsing shard dims from MeshMapperConfig repr.

    This test will fail if TTNN changes the repr format, alerting us to update the parser.
    """

    # Single shard dimension
    config1 = ttnn.MeshMapperConfig(
        placements=[ttnn.PlacementShard(-1)],
        mesh_shape_override=ttnn.MeshShape([8]),
    )
    assert parse_shard_dims_from_mesh_mapper_config(config1) == [-1]

    # Different shard dimension
    config2 = ttnn.MeshMapperConfig(
        placements=[ttnn.PlacementShard(-2)],
        mesh_shape_override=ttnn.MeshShape([4]),
    )
    assert parse_shard_dims_from_mesh_mapper_config(config2) == [-2]

    # Positive dimension
    config3 = ttnn.MeshMapperConfig(
        placements=[ttnn.PlacementShard(0)],
        mesh_shape_override=ttnn.MeshShape([2]),
    )
    assert parse_shard_dims_from_mesh_mapper_config(config3) == [0]

    # Two dimensions sharded (2D mesh)
    config4 = ttnn.MeshMapperConfig(
        placements=[ttnn.PlacementShard(-2), ttnn.PlacementShard(-1)],
        mesh_shape_override=ttnn.MeshShape([2, 4]),
    )
    assert parse_shard_dims_from_mesh_mapper_config(config4) == [-2, -1]

    # Mixed: one sharded, one replicated (only shard dims should be returned)
    config5 = ttnn.MeshMapperConfig(
        placements=[ttnn.PlacementReplicate(), ttnn.PlacementShard(-1)],
        mesh_shape_override=ttnn.MeshShape([2, 4]),
    )
    assert parse_shard_dims_from_mesh_mapper_config(config5) == [-1]

    # All replicated (no shard dims)
    config6 = ttnn.MeshMapperConfig(
        placements=[ttnn.PlacementReplicate()],
        mesh_shape_override=ttnn.MeshShape([8]),
    )
    assert parse_shard_dims_from_mesh_mapper_config(config6) == []


@pytest.mark.host
def test_get_rot_transformation_mat_tile_size():
    """Verify decode transformation matrix is TILE_SIZE x TILE_SIZE with correct pattern."""
    mat = get_rot_transformation_mat(dhead=32)
    assert mat.shape == (1, 1, 32, 32)
    # Permutation pattern: even→odd = +1, odd→even = -1
    assert mat[0, 0, 0, 1].item() == 1.0
    assert mat[0, 0, 1, 0].item() == -1.0
    assert mat[0, 0, 2, 3].item() == 1.0
    assert mat[0, 0, 3, 2].item() == -1.0
    # Diagonal is zero
    assert mat[0, 0, 0, 0].item() == 0.0
    assert mat[0, 0, 1, 1].item() == 0.0


@pytest.mark.host
def test_get_rot_transformation_mat_large():
    """Verify the matrix works for arbitrary dhead (e.g., head_dim=128 for prefill)."""
    mat = get_rot_transformation_mat(dhead=128)
    assert mat.shape == (1, 1, 128, 128)
    # Pattern extends to last pair
    assert mat[0, 0, 126, 127].item() == 1.0
    assert mat[0, 0, 127, 126].item() == -1.0
    # Off-pattern entries are zero
    assert mat[0, 0, 0, 2].item() == 0.0
    assert mat[0, 0, 0, 3].item() == 0.0


@pytest.mark.host
def test_get_rot_transformation_mat():
    """
    Test that get_rot_transformation_mat produces the correct rotation matrix for RoPE.

    The rotation transformation matrix is used by ttnn.experimental.rotary_embedding_llama.
    It has the pattern:
    - rot_emb_matrix[i, i+1] = 1 for even i
    - rot_emb_matrix[i+1, i] = -1 for even i
    """
    result = get_rot_transformation_mat()

    # Validate shape
    assert result.shape == (1, 1, 32, 32), f"Expected shape (1, 1, 32, 32), got {result.shape}"

    # Validate specific known values
    # Position (0, 1) should be 1
    assert result[0, 0, 0, 1].item() == pytest.approx(1.0)
    # Position (1, 0) should be -1
    assert result[0, 0, 1, 0].item() == pytest.approx(-1.0)
    # Position (0, 0) should be 0
    assert result[0, 0, 0, 0].item() == pytest.approx(0.0)
    # Position (2, 3) should be 1
    assert result[0, 0, 2, 3].item() == pytest.approx(1.0)
    # Position (3, 2) should be -1
    assert result[0, 0, 3, 2].item() == pytest.approx(-1.0)
    # Position (30, 31) should be 1
    assert result[0, 0, 30, 31].item() == pytest.approx(1.0)
    # Position (31, 30) should be -1
    assert result[0, 0, 31, 30].item() == pytest.approx(-1.0)

    # Validate that non-pattern positions are 0
    assert result[0, 0, 0, 2].item() == pytest.approx(0.0)
    assert result[0, 0, 1, 1].item() == pytest.approx(0.0)


@pytest.mark.host
def test_zeros_like_kv_cache():
    """Test zeros_like_kv_cache creates correct shape tensor."""
    batch_size, n_kv_heads, max_seq_len, head_dim = 32, 8, 2048, 128

    result = zeros_like_kv_cache(batch_size, n_kv_heads, max_seq_len, head_dim)

    assert result.shape == (batch_size, n_kv_heads, max_seq_len, head_dim)
    assert result.dtype == torch.float32
    assert torch.all(result == 0)


@pytest.mark.host
def test_zeros_like_paged_cache():
    """Test zeros_like_paged_cache creates correct shape tensor."""
    from dataclasses import dataclass

    @dataclass
    class MockPagedConfig:
        max_num_blocks: int = 64
        block_size: int = 64

    paged_config = MockPagedConfig()
    n_kv_heads = 8
    head_dim = 128

    result = zeros_like_paged_cache(paged_config, n_kv_heads, head_dim)

    assert result.shape == (paged_config.max_num_blocks, n_kv_heads, paged_config.block_size, head_dim)
    assert result.dtype == torch.float32
    assert torch.all(result == 0)


@pytest.mark.host
def test_program_config_to_dict_with_to_json():
    """Test program_config_to_dict for a config that has to_json (matmul configs)."""
    cfg = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
        in0_block_w=4,
        per_core_M=1,
        per_core_N=2,
    )
    d = program_config_to_dict(cfg)

    assert isinstance(d, dict)
    assert d["type"] == "MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig"
    assert d["in0_block_w"] == 4
    assert d["per_core_M"] == 1
    assert d["per_core_N"] == 2
    assert "fused_activation" in d

    json_str = json.dumps(d, sort_keys=True)
    roundtrip = json.loads(json_str)
    assert roundtrip == d


@pytest.mark.host
def test_program_config_to_dict_without_to_json():
    """Serialize public attributes when TTNN 0.79 program configs have no to_json."""
    cfg = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
        q_chunk_size=256,
        k_chunk_size=256,
    )
    d = program_config_to_dict(cfg)

    assert isinstance(d, dict)
    assert d["type"] == "SDPAProgramConfig"
    assert d["compute_with_storage_grid_size"] == {"x": 8, "y": 8}
    assert d["q_chunk_size"] == 256
    assert d["k_chunk_size"] == 256
    assert d["exp_approx_mode"] is None
    assert d["max_cores_per_head_batch"] == 16
    assert d["sub_core_grids"] is None


@pytest.mark.host
def test_program_config_to_str():
    """Test program_config_to_str returns valid sorted JSON."""
    cfg = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(in0_block_w=2, per_core_M=3, per_core_N=4)
    result = program_config_to_str(cfg)

    parsed = json.loads(result)
    assert parsed["in0_block_w"] == 2
    assert parsed["per_core_M"] == 3
    assert parsed["per_core_N"] == 4

    assert result == json.dumps(parsed, sort_keys=True)


class _FakeHostTensor:
    def __init__(self, name, *, topology=None):
        self.name = name
        self.dtype = ttnn.bfloat8_b
        self.layout = ttnn.TILE_LAYOUT
        self.tile = f"{name}-tile"
        self._topology = topology
        self.topology_updates = []
        self.uploads = []

    def memory_config(self):
        return f"{self.name}-memcfg"

    def tensor_topology(self):
        return self._topology

    def update_tensor_topology(self, topology):
        self.topology_updates.append(topology)

    def to(self, device, memory_config):
        self.uploads.append((device, memory_config))
        return f"{self.name}-on-device"


def _write_cache_file(tmp_path, size):
    path = tmp_path / "weight.tensorbin"
    path.write_bytes(b"\0" * size)
    return path


@pytest.mark.host
def test_load_cached_tensor_loads_small_files_directly(tmp_path, monkeypatch):
    monkeypatch.setattr(tensor_utils, "PINNED_UPLOAD_THRESHOLD_BYTES", 16)
    calls = []
    monkeypatch.setattr(ttnn, "load_tensor", lambda path, device=None: calls.append((path, device)) or "direct")
    path = _write_cache_file(tmp_path, 16)

    assert load_cached_tensor(path, "mesh") == "direct"
    assert calls == [(str(path), "mesh")]


def _patch_rebuild(monkeypatch, shard_count):
    """Fake the ttnn calls of the rebuild path; returns (mapped, shards, calls, owned)."""
    monkeypatch.setattr(tensor_utils, "PINNED_UPLOAD_THRESHOLD_BYTES", 16)
    mapped = _FakeHostTensor("mapped", topology="mapped-topology")
    shards = [_FakeHostTensor(f"shard{index}") for index in range(shard_count)]
    owned = _FakeHostTensor("owned")
    calls = {"load": [], "from_torch": [], "from_host_shards": []}

    def from_torch(tensor, **kwargs):
        calls["from_torch"].append((tensor, kwargs))
        return _FakeHostTensor(f"rebuilt-{tensor}")

    def from_host_shards(tensors, shape):
        calls["from_host_shards"].append(([tensor.name for tensor in tensors], shape))
        return owned

    monkeypatch.setattr(ttnn, "using_distributed_env", lambda: False)
    monkeypatch.setattr(ttnn, "load_tensor", lambda path, device=None: calls["load"].append((path, device)) or mapped)
    monkeypatch.setattr(ttnn, "get_device_tensors", lambda tensor: shards if tensor is mapped else None)
    monkeypatch.setattr(ttnn, "to_torch", lambda shard: f"torch-{shard.name}")
    monkeypatch.setattr(ttnn, "from_torch", from_torch)
    monkeypatch.setattr(ttnn, "from_host_shards", from_host_shards)
    return mapped, shards, calls, owned


_MESH_1X2 = SimpleNamespace(get_num_devices=lambda: 2, shape="mesh-1x2")


@pytest.mark.host
def test_load_cached_tensor_rebuilds_sharded_files_on_the_device_mesh_shape(tmp_path, monkeypatch):
    mapped, shards, calls, owned = _patch_rebuild(monkeypatch, shard_count=2)
    path = _write_cache_file(tmp_path, 17)

    assert load_cached_tensor(path, _MESH_1X2, pad_value=-1.0) == "owned-on-device"
    assert calls["load"] == [(str(path), None)]
    assert calls["from_torch"] == [
        (
            f"torch-{shard.name}",
            {
                "dtype": ttnn.bfloat8_b,
                "layout": ttnn.TILE_LAYOUT,
                "tile": f"{shard.name}-tile",
                "memory_config": f"{shard.name}-memcfg",
                "pad_value": -1.0,
            },
        )
        for shard in shards
    ]
    assert calls["from_host_shards"] == [(["rebuilt-torch-shard0", "rebuilt-torch-shard1"], "mesh-1x2")]
    assert owned.topology_updates == ["mapped-topology"]
    assert owned.uploads == [(_MESH_1X2, "mapped-memcfg")]
    assert mapped.uploads == []


@pytest.mark.host
def test_load_cached_tensor_uploads_a_replicated_file_as_its_rebuilt_shard(tmp_path, monkeypatch):
    mapped, _, calls, _ = _patch_rebuild(monkeypatch, shard_count=1)
    path = _write_cache_file(tmp_path, 17)

    assert load_cached_tensor(path, _MESH_1X2) == "rebuilt-torch-shard0-on-device"
    assert calls["from_host_shards"] == []
    assert calls["from_torch"][0][1]["pad_value"] is None
    assert mapped.uploads == []


@pytest.mark.host
def test_load_cached_tensor_keeps_the_direct_upload_for_other_shard_layouts(tmp_path, monkeypatch):
    mapped, _, calls, _ = _patch_rebuild(monkeypatch, shard_count=3)
    path = _write_cache_file(tmp_path, 17)

    assert load_cached_tensor(path, _MESH_1X2) == "mapped-on-device"
    assert calls["from_torch"] == []
    assert mapped.uploads == [(_MESH_1X2, "mapped-memcfg")]


@pytest.mark.host
def test_load_cached_tensor_keeps_the_direct_load_on_multi_host_meshes(tmp_path, monkeypatch):
    monkeypatch.setattr(tensor_utils, "PINNED_UPLOAD_THRESHOLD_BYTES", 16)
    calls = []
    monkeypatch.setattr(ttnn, "using_distributed_env", lambda: True)
    monkeypatch.setattr(ttnn, "load_tensor", lambda path, device=None: calls.append((path, device)) or "direct")
    path = _write_cache_file(tmp_path, 17)

    assert load_cached_tensor(path, "mesh") == "direct"
    assert calls == [(str(path), "mesh")]


if __name__ == "__main__":
    test_pad_dim_to_size()
    print("  ✓ test_pad_dim_to_size")

    test_pad_dim_to_size_positive_dim()
    print("  ✓ test_pad_dim_to_size_positive_dim")

    test_pad_to_shape()
    print("  ✓ test_pad_to_shape")

    test_pad_to_shape_no_op()
    print("  ✓ test_pad_to_shape_no_op")

    test_pad_to_shape_single_dim()
    print("  ✓ test_pad_to_shape_single_dim")

    test_pad_to_shape_error_on_smaller_target()
    print("  ✓ test_pad_to_shape_error_on_smaller_target")

    test_parse_shard_dims_from_mesh_mapper_config()
    print("  ✓ test_parse_shard_dims_from_mesh_mapper_config")

    test_get_rot_transformation_mat_tile_size()
    print("  ✓ test_get_rot_transformation_mat_tile_size")

    test_get_rot_transformation_mat_large()
    print("  ✓ test_get_rot_transformation_mat_large")
    test_get_rot_transformation_mat()
    print("  ✓ test_get_rot_transformation_mat")

    test_zeros_like_kv_cache()
    print("  ✓ test_zeros_like_kv_cache")

    test_zeros_like_paged_cache()
    print("  ✓ test_zeros_like_paged_cache")

    test_program_config_to_dict_with_to_json()
    print("  ✓ test_program_config_to_dict_with_to_json")

    test_program_config_to_dict_without_to_json()
    print("  ✓ test_program_config_to_dict_without_to_json")

    test_program_config_to_str()
    print("  ✓ test_program_config_to_str")

    print("\nAll tensor_utils tests passed! ✓")
