# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for the LMHead1D module (1D mesh topology: N150, N300, T3K).

This test suite verifies:
1. Unit tests for config dataclass (no device needed)
2. LMHead1D matches torch.nn.Linear reference for logit computation
3. from_model_args backward compatibility
"""

import math
import os
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
import ttnn
from examples.common.auto_compose import to_torch_auto_compose
from loguru import logger
from tests.support.comparison import comp_allclose, comp_pcc

from tt_transformers.modules.lazy_weight import LazyWeight
from tt_transformers.modules.lm_head.lm_head_1d import (
    LMHead1D,
    LMHead1DConfig,
    _validate_lm_head_program_configs,
    resolve_lm_head_1d_arch_config,
)
from tt_transformers.tensor_utils import TILE_SIZE

# 1D module suites target the T3K; skip when the host system is a Galaxy.
pytestmark = pytest.mark.usefixtures("skip_on_galaxy_system")

# ============================================================================
# Unit Tests - No device required
# ============================================================================


@pytest.mark.host
def test_lm_head_1d_config_creation():
    """Test LMHead1DConfig dataclass creation."""
    config = LMHead1DConfig(
        output_weights=[MagicMock()],
        mesh_device=MagicMock(),
        dim=4096,
    )
    assert config.dim == 4096
    assert config.lm_head_dtype == ttnn.bfloat8_b
    assert config.max_batch_size == 32


@pytest.mark.host
def test_lm_head_1d_config_defaults():
    """Test LMHead1DConfig default values."""
    config = LMHead1DConfig(output_weights=[MagicMock()])
    assert config.mesh_device is None
    assert config.program_configs is None
    assert config.compute_kernel_config is None
    assert config.lm_head_dtype == ttnn.bfloat8_b
    assert config.output_memcfg is None
    assert config.input_memcfg is None
    assert config.weights_memcfgs is None


@pytest.mark.host
def test_lm_head_1d_config_is_resolved_all_fields():
    """Test is_resolved() when all fields are set."""
    mock_device = MagicMock()
    mock_device.get_num_devices.return_value = 1

    config = LMHead1DConfig(
        output_weights=[MagicMock()],
        mesh_device=mock_device,
        dim=4096,
        program_configs=[None],
        compute_kernel_config=ttnn.init_device_compute_kernel_config(
            ttnn.device.Arch.WORMHOLE_B0,
            math_fidelity=ttnn.MathFidelity.HiFi2,
        ),
        output_split_sizes=[1024],
        output_memcfg=MagicMock(),
        input_memcfg=MagicMock(),
        weights_memcfgs=[MagicMock()],
    )
    assert config.is_resolved()


@pytest.mark.host
def test_compute_kernel_config_hifi2():
    """Test _compute_kernel_config_hifi2 returns valid config."""
    from tt_transformers.modules.lm_head.lm_head_1d import _compute_kernel_config_hifi2

    cfg = _compute_kernel_config_hifi2(ttnn.device.Arch.WORMHOLE_B0)
    assert cfg.math_fidelity == ttnn.MathFidelity.HiFi2
    assert cfg.packer_l1_acc is True
    assert cfg.fp32_dest_acc_en is False


def _pure_lm_head_config(arch):
    mesh = MagicMock()
    mesh.arch.return_value = arch
    mesh.get_num_devices.return_value = 1
    weight = LazyWeight(source=torch.empty(32, 32), device=mesh)
    return LMHead1DConfig(
        output_weights=[weight],
        mesh_device=mesh,
        dim=32,
        program_configs=[None],
        output_split_sizes=[32],
        output_memcfg=ttnn.L1_MEMORY_CONFIG,
        input_memcfg=ttnn.DRAM_MEMORY_CONFIG,
        weights_memcfgs=[ttnn.DRAM_MEMORY_CONFIG],
    )


@pytest.mark.host
@pytest.mark.parametrize("arch", [ttnn.device.Arch.WORMHOLE_B0, ttnn.device.Arch.BLACKHOLE])
def test_lm_head_arch_resolver_selects_once_without_mutation(arch):
    config = _pure_lm_head_config(arch)
    original_program_configs = config.program_configs
    original_output_weights = config.output_weights
    original_weight = config.output_weights[0]

    resolved = resolve_lm_head_1d_arch_config(config)

    assert isinstance(resolved, LMHead1DConfig)
    assert resolved is not config
    assert resolved.is_resolved()
    assert config.program_configs is original_program_configs
    assert config.output_weights is original_output_weights
    assert config.output_weights[0] is original_weight
    assert resolved.output_weights[0] is not original_weight
    assert config.compute_kernel_config is None
    assert config.mesh_device.arch.call_count == 1
    assert resolved.compute_kernel_config.math_fidelity == ttnn.MathFidelity.HiFi2
    assert resolved.compute_kernel_config.math_approx_mode is False
    assert resolved.compute_kernel_config.fp32_dest_acc_en is False
    assert resolved.compute_kernel_config.packer_l1_acc is True
    assert resolved.compute_kernel_config.dst_full_sync_en is False
    assert resolved.compute_kernel_config.throttle_level == ttnn.ThrottleLevel.NO_THROTTLE


@pytest.mark.host
def test_lm_head_explicit_common_override_is_copied():
    config = _pure_lm_head_config(ttnn.device.Arch.BLACKHOLE)
    override = ttnn.init_device_compute_kernel_config(
        ttnn.device.Arch.BLACKHOLE,
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=True,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )

    config.compute_kernel_config = override
    resolved = resolve_lm_head_1d_arch_config(config)
    assert resolved.compute_kernel_config is not override
    assert resolved.compute_kernel_config.math_fidelity == ttnn.MathFidelity.HiFi4
    assert resolved.compute_kernel_config.math_approx_mode is True
    assert resolved.compute_kernel_config.fp32_dest_acc_en is True
    assert resolved.compute_kernel_config.packer_l1_acc is False
    assert config.compute_kernel_config is override


@pytest.mark.host
def test_lm_head_arch_resolver_fails_closed(expect_error):
    config = _pure_lm_head_config(object())
    with expect_error(ValueError, "Unsupported LMHead1D architecture"):
        resolve_lm_head_1d_arch_config(config)

    config = _pure_lm_head_config(ttnn.device.Arch.BLACKHOLE)
    config.program_configs = []
    with expect_error(ValueError, "one program config"):
        resolve_lm_head_1d_arch_config(config)

    config = _pure_lm_head_config(ttnn.device.Arch.BLACKHOLE)
    config.output_split_sizes = [0]
    with expect_error(ValueError, "0 < logical"):
        resolve_lm_head_1d_arch_config(config)

    config = _pure_lm_head_config(ttnn.device.Arch.BLACKHOLE)
    config.mesh_device.get_num_devices.return_value = 3
    with expect_error(ValueError, "must be divisible"):
        resolve_lm_head_1d_arch_config(config)

    config = _pure_lm_head_config(ttnn.device.Arch.BLACKHOLE)
    config.compute_kernel_config = object()
    with expect_error(ValueError, "Invalid LMHead1D compute recipe"):
        resolve_lm_head_1d_arch_config(config)


@pytest.mark.host
def test_lm_head_resolutions_are_independent():
    config = _pure_lm_head_config(ttnn.device.Arch.BLACKHOLE)
    first = resolve_lm_head_1d_arch_config(config)
    second = resolve_lm_head_1d_arch_config(config)
    assert first is not second
    assert first.compute_kernel_config is not second.compute_kernel_config


@pytest.mark.host
def test_lm_head_weight_device_mismatch_fails_before_architecture_query(expect_error):
    config = _pure_lm_head_config(ttnn.device.Arch.BLACKHOLE)
    config.output_weights[0].device = MagicMock()
    with expect_error(ValueError, "must match output weight 0 device"):
        resolve_lm_head_1d_arch_config(config)
    assert config.mesh_device.arch.call_count == 0


@pytest.mark.host
def test_lm_head_construction_is_only_architecture_query():
    config = _pure_lm_head_config(ttnn.device.Arch.BLACKHOLE)
    module = LMHead1D.from_config(config)
    assert config.mesh_device.arch.call_count == 1

    _ = module.forward
    _ = module.config.input_memcfg
    _ = module.config.compute_kernel_config
    assert config.mesh_device.arch.call_count == 1
    assert not hasattr(module, "arch_config")


def _pure_dram_sharded_lm_head_config(
    *, grid, dim, num_devices, logical_width, physical_width, input_cores, dram_cores, per_core_n
):
    # Upstream also sets num_workers_per_dram_bank on the program config; that keyword does not
    # exist on MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig at the pinned ttnn. Nothing
    # is lost: the admission check never reads it, and the reader count these cases exercise is
    # carried by dram_cores.
    from tt_transformers.modules.lm_head.lm_head_1d import _create_dram_sharded_mem_config

    mesh = MagicMock()
    mesh.get_num_devices.return_value = num_devices
    mesh.compute_with_storage_grid_size.return_value = ttnn.CoreCoord(*grid)
    # Admission only needs source metadata; do not allocate model-sized weights.
    weight = MagicMock()
    weight.source.shape = (dim, physical_width * num_devices)
    input_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, input_cores // 8 - 1))})
    dram_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram_cores - 1, 0))})
    return LMHead1DConfig(
        output_weights=[weight],
        mesh_device=mesh,
        dim=dim,
        max_batch_size=1,
        program_configs=[
            ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                in0_block_w=dim // (TILE_SIZE * input_cores),
                per_core_M=1,
                per_core_N=per_core_n,
            )
        ],
        output_split_sizes=[logical_width],
        input_memcfg=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(input_grid, (TILE_SIZE, dim // input_cores), ttnn.ShardOrientation.ROW_MAJOR),
        ),
        weights_memcfgs=[_create_dram_sharded_mem_config(dim, physical_width, dram_grid, dram_cores=dram_cores)],
    )


@pytest.mark.host
@pytest.mark.parametrize(
    "grid,dim,num_devices,logical_width,physical_width,input_cores,dram_cores,per_core_n",
    [
        pytest.param((8, 8), 8192, 8, 8192, 8192, 32, 12, 8, id="llama33-wormhole"),
        pytest.param((11, 8), 8192, 4, 4008, 4032, 32, 8, 4, id="llama33-blackhole"),
        pytest.param((11, 8), 4096, 4, 32768, 32768, 8, 8, 64, id="llama31-qb2"),
    ],
)
def test_lm_head_output_storage_is_independent_of_inputs_and_readers(
    grid, dim, num_devices, logical_width, physical_width, input_cores, dram_cores, per_core_n
):
    # All three recipes need more output cores than they have DRAM readers; the QB2 one also
    # needs more than it has activation shards. Ported from tenstorrent/tt-metal#55922.
    config = _pure_dram_sharded_lm_head_config(
        grid=grid,
        dim=dim,
        num_devices=num_devices,
        logical_width=logical_width,
        physical_width=physical_width,
        input_cores=input_cores,
        dram_cores=dram_cores,
        per_core_n=per_core_n,
    )
    _validate_lm_head_program_configs(config)


@pytest.mark.host
@pytest.mark.parametrize("grid,dram_cores", [((8, 8), 12), ((11, 8), 8)])
@pytest.mark.parametrize("extra_tile", [0, 1])
def test_lm_head_output_storage_capacity_uses_physical_width(grid, dram_cores, extra_tile, expect_error):
    capacity = grid[0] * grid[1]
    config = _pure_dram_sharded_lm_head_config(
        grid=grid,
        dim=4096,
        num_devices=1,
        logical_width=capacity * TILE_SIZE,
        physical_width=(capacity + extra_tile) * TILE_SIZE,
        input_cores=8,
        dram_cores=dram_cores,
        per_core_n=1,
    )
    if extra_tile:
        with expect_error(ValueError, f"requires {capacity + 1} output storage cores"):
            _validate_lm_head_program_configs(config)
    else:
        _validate_lm_head_program_configs(config)


@pytest.mark.host
def test_create_dram_sharded_mem_config():
    """Test _create_dram_sharded_mem_config produces valid MemoryConfig."""
    from tt_transformers.modules.lm_head.lm_head_1d import _create_dram_sharded_mem_config

    dram_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(11, 0))})
    mc = _create_dram_sharded_mem_config(k=4096, n=16032, dram_grid=dram_grid, dram_cores=12)
    assert mc.is_sharded()
    assert mc.memory_layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED
    assert mc.buffer_type == ttnn.BufferType.DRAM


@pytest.mark.host
@pytest.mark.skip(reason="TTTv1 from_model_args compatibility is outside standalone ownership")
def test_from_model_args_rejects_galaxy(expect_error):
    """Test from_model_args raises for Galaxy devices."""
    from unittest.mock import MagicMock

    mock_args = MagicMock()
    mock_args.is_galaxy = True

    with expect_error(ValueError, "Galaxy"):
        LMHead1D.from_model_args(
            mesh_device=MagicMock(),
            args=mock_args,
            state_dict={},
            state_dict_prefix="",
            weight_cache_path="",
            max_columns_per_device=32000,
        )


# ============================================================================
# Weight helpers
# ============================================================================

_CACHED_LM_WEIGHTS: dict[str, torch.Tensor] = {}


def _get_or_init_lm_weight(key: str, dim: int, vocab_size: int) -> torch.Tensor:
    if key not in _CACHED_LM_WEIGHTS:
        logger.info(f"\033[33m[cache miss]\033[0m Initializing LM head weight for {key}")
        _CACHED_LM_WEIGHTS[key] = torch.randn(vocab_size, dim, dtype=torch.bfloat16)
    else:
        logger.info(f"\033[32m[cache hit]\033[0m Reusing cached LM head weight for {key}")
    return _CACHED_LM_WEIGHTS[key]


def _prepare_lm_head_weights(
    weight: torch.Tensor, vocab_size: int, dim: int, num_devices: int, max_columns_per_device: int
) -> list[torch.Tensor]:
    """Split LM head weight into chunks matching TTTv1 logic (non-TG path)."""
    padded_vocab_size = math.ceil(vocab_size / 32) * 32
    size_per_device = padded_vocab_size // num_devices
    num_splits = math.ceil(size_per_device / max_columns_per_device)
    split_sizes = [min(size_per_device, max_columns_per_device)] * (num_splits - 1)
    split_sizes.append(size_per_device - sum(split_sizes))

    # Transpose to (dim, vocab) and pad
    torch_w = weight.T  # (dim, vocab_size)
    if vocab_size < padded_vocab_size:
        torch_w = torch.cat([torch_w, torch.zeros(dim, padded_vocab_size - vocab_size, dtype=torch_w.dtype)], dim=-1)

    splits = []
    for i, split_size in enumerate(split_sizes):
        device_splits = []
        physical_split_size = math.ceil(split_size / TILE_SIZE) * TILE_SIZE
        for dev in range(num_devices):
            start = dev * size_per_device + sum(split_sizes[:i])
            end = start + split_size
            device_split = torch_w[:, start:end]
            if split_size < physical_split_size:
                device_split = torch.cat(
                    [device_split, torch.zeros(dim, physical_split_size - split_size, dtype=device_split.dtype)], dim=-1
                )
            device_splits.append(device_split)
        splits.append(torch.cat(device_splits, dim=-1))

    return splits


# ============================================================================
# Model names from HF to cover in tests
# ============================================================================

LLAMA_1B = "meta-llama/Llama-3.2-1B-Instruct"
LLAMA_3B = "meta-llama/Llama-3.2-3B-Instruct"
LLAMA_8B = "meta-llama/Llama-3.1-8B-Instruct"
LLAMA_11B = "meta-llama/Llama-3.2-11B-Vision-Instruct"
LLAMA_70B = "meta-llama/Llama-3.3-70B-Instruct"
MISTRAL_7B = "mistralai/Mistral-7B-Instruct-v0.3"
QWEN2_7B = "Qwen/Qwen2-7B-Instruct"
QWEN25_7B = "Qwen/Qwen2.5-7B-Instruct"
QWEN25_72B = "Qwen/Qwen2.5-72B-Instruct"
QWEN25_CODER_32B = "Qwen/Qwen2.5-Coder-32B-Instruct"
QWEN3_32B = "Qwen/Qwen3-32B"
DEEPSEEK_R1_14B = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"


_slow = pytest.mark.slow


def _list_test_cases() -> list[pytest.param]:
    # max_columns_per_device derived from TTTv1: 668 * lm_head_core_grid.num_cores
    # For simplicity, use the actual split counts from CSV
    # fmt: off
    return [
        # === Fast tests ===
        # 1x1 Llama-3.1-8B: 3 splits (dim=4096, padded_vocab=128256)
        pytest.param((1, 1), 4096, 128256, 42752, LLAMA_8B, 0.999, id="1x1-8B"),
        # 1x2 Llama-3.1-8B: 2 splits
        pytest.param((1, 2), 4096, 128256, 42752, LLAMA_8B, 0.999, id="1x2-8B"),
        # 1x8 Llama-3.1-8B: 1 split
        pytest.param((1, 8), 4096, 128256, 16032, LLAMA_8B, 0.999, id="1x8-8B"),
        # 1x8 Llama-3.3-70B: 1 split (dim=8192)
        pytest.param((1, 8), 8192, 128256, 16032, LLAMA_70B, 0.999, id="1x8-70B"),

        # === Slow tests ===
        # 1x1 Llama-3.2-1B: 3 splits (dim=2048)
        pytest.param((1, 1), 2048, 128256, 42752, LLAMA_1B, 0.999, id="1x1-1B", marks=_slow),
        # 1x1 Llama-3.2-3B: 4 splits (dim=3072)
        pytest.param((1, 1), 3072, 128256, 32064, LLAMA_3B, 0.999, id="1x1-3B", marks=_slow),
        # 1x1 Mistral-7B: 1 split (dim=4096, vocab=32768)
        pytest.param((1, 1), 4096, 32768, 32768, MISTRAL_7B, 0.999, id="1x1-Mistral-7B", marks=_slow),
        # 1x2 Llama-3.2-1B: 2 splits
        pytest.param((1, 2), 2048, 128256, 42752, LLAMA_1B, 0.999, id="1x2-1B", marks=_slow),
        # 1x2 Llama-3.2-3B: 2 splits
        pytest.param((1, 2), 3072, 128256, 32064, LLAMA_3B, 0.999, id="1x2-3B", marks=_slow),
        # 1x2 Llama-3.2-11B: 2 splits
        pytest.param((1, 2), 4096, 128256, 42752, LLAMA_11B, 0.999, id="1x2-11B", marks=_slow),
        # 1x2 Mistral-7B: 1 split
        pytest.param((1, 2), 4096, 32768, 16384, MISTRAL_7B, 0.999, id="1x2-Mistral-7B", marks=_slow),
        # 1x2 Qwen2-7B: 3 splits (dim=3584, vocab=152064)
        pytest.param((1, 2), 3584, 152064, 37408, QWEN2_7B, 0.999, id="1x2-Qwen2-7B", marks=_slow),
        # 1x2 DeepSeek-R1-14B: 3 splits (dim=5120, vocab=152064)
        pytest.param((1, 2), 5120, 152064, 26720, DEEPSEEK_R1_14B, 0.999, id="1x2-DeepSeek-R1-14B", marks=_slow),
        # 1x2 Qwen2.5-7B: 3 splits
        pytest.param((1, 2), 3584, 152064, 37408, QWEN25_7B, 0.999, id="1x2-Qwen2.5-7B", marks=_slow),
        # 1x8 Llama-3.2-1B: 1 split
        pytest.param((1, 8), 2048, 128256, 16032, LLAMA_1B, 0.999, id="1x8-1B", marks=_slow),
        # 1x8 Llama-3.2-3B: 1 split
        pytest.param((1, 8), 3072, 128256, 16032, LLAMA_3B, 0.999, id="1x8-3B", marks=_slow),
        # 1x8 Llama-3.2-11B: 1 split
        pytest.param((1, 8), 4096, 128256, 16032, LLAMA_11B, 0.999, id="1x8-11B", marks=_slow),
        # 1x8 Qwen2.5-72B: 1 split (dim=8192, vocab=152064)
        pytest.param((1, 8), 8192, 152064, 19008, QWEN25_72B, 0.999, id="1x8-Qwen2.5-72B", marks=_slow),
        # 1x8 Qwen2.5-Coder-32B: 1 split (dim=5120)
        pytest.param((1, 8), 5120, 152064, 19008, QWEN25_CODER_32B, 0.999, id="1x8-Qwen2.5-Coder-32B", marks=_slow),
        # 1x8 Qwen3-32B: 1 split (dim=5120, vocab=151936)
        pytest.param((1, 8), 5120, 151936, 18992, QWEN3_32B, 0.999, id="1x8-Qwen3-32B", marks=_slow),
        # 1x8 Mistral-7B: 1 split
        pytest.param((1, 8), 4096, 32768, 4096, MISTRAL_7B, 0.999, id="1x8-Mistral-7B", marks=_slow),
    ]
    # fmt: on


@pytest.mark.device
@pytest.mark.parametrize(
    "ttnn_mesh_device",
    [(1, 1), (1, 2), (1, 8)],
    ids=["1x1", "1x2", "1x8"],
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_shape,dim,vocab_size,max_col_per_dev,hf_model_name,pcc",
    _list_test_cases(),
)
def test_lm_head_1d_vs_reference(
    ttnn_mesh_device: ttnn.MeshDevice,
    mesh_shape,
    dim,
    vocab_size,
    max_col_per_dev,
    hf_model_name,
    pcc,
):
    """
    Test LMHead1D output shape and basic numerical correctness.

    Uses random weights split per TTTv1 logic, verifies output is non-zero
    and has correct shape.
    """
    seed = 42
    torch.manual_seed(seed)
    batch_rows = 32  # tile_padded_batch_rows for batch_size=1
    num_devices = ttnn_mesh_device.get_num_devices()

    # Get reference weight
    key = f"{hf_model_name}_{vocab_size}_{dim}"
    full_weight = _get_or_init_lm_weight(key, dim, vocab_size)

    # Reference: torch.nn.Linear (no bias), in bfloat16
    ref_linear = torch.nn.Linear(dim, vocab_size, bias=False, dtype=torch.bfloat16)
    with torch.no_grad():
        ref_linear.weight.copy_(full_weight)

    # Reference output
    torch_input = torch.randn(1, 1, batch_rows, dim, dtype=torch.bfloat16)
    with torch.no_grad():
        ref_output = ref_linear(torch_input)  # [1, 1, 32, vocab_size]

    # Split weights for TT model
    weight_splits = _prepare_lm_head_weights(full_weight, vocab_size, dim, num_devices, max_col_per_dev)

    # Create LazyWeights (cache-backed for faster repeated runs)
    ttnn.SetDefaultDevice(ttnn_mesh_device)
    cache_dir = Path(os.getenv("TT_CACHE_PATH", "model_cache/lm_head_1d"))
    lazy_weights = []
    for i, split in enumerate(weight_splits):
        lazy_weights.append(
            LazyWeight(source=split, dtype=ttnn.bfloat8_b, cache_dir_weight_name=(cache_dir, f"w_split_{i}"))
        )

    tt_model = LMHead1D(output_weights=lazy_weights)

    # Run TT model
    tt_input = LazyWeight(source=torch_input, dtype=ttnn.bfloat16)
    tt_output = tt_model.forward(tt_input)
    tt_output_torch = to_torch_auto_compose(tt_output)
    ttnn.SetDefaultDevice(None)

    # Shape checks
    assert tt_output_torch.shape[-2] == batch_rows, f"Expected batch_rows={batch_rows}, got {tt_output_torch.shape[-2]}"
    assert tt_output_torch.shape[-1] >= vocab_size, (
        f"Expected vocab cols>={vocab_size}, got {tt_output_torch.shape[-1]}. num_devices={num_devices}"
    )

    # PCC against torch reference (trim to actual vocab_size, ignore padding zeros)
    ref_trimmed = ref_output[..., :vocab_size]
    tt_trimmed = tt_output_torch[..., :vocab_size]

    passing, pcc_message = comp_pcc(ref_trimmed, tt_trimmed, pcc)
    logger.info(comp_allclose(ref_trimmed, tt_trimmed))
    logger.info(f"LMHead1D vs reference: {pcc_message}")
    assert passing, f"LMHead1D output does not meet PCC {pcc}: {pcc_message}."
    logger.info(f"LMHead1D: PASSED for {hf_model_name} (mesh={mesh_shape}, devices={num_devices})")


@pytest.mark.device
@pytest.mark.blackhole
@pytest.mark.parametrize(
    "ttnn_mesh_device,vocab_size,max_columns_per_device,expected_splits",
    [
        pytest.param((1, 1), 8192, 8192, 1, id="p150-single-split"),
        pytest.param(
            {"mesh_shape": (1, 4), "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING},
            128256,
            4008,
            8,
            id="p150x4-4008-column-splits",
        ),
    ],
    indirect=["ttnn_mesh_device"],
)
def test_lm_head_1d_blackhole_common_config_correctness_cache_and_timing(
    request, ttnn_mesh_device, require_blackhole_mesh_device, vocab_size, max_columns_per_device, expected_splits
):
    """Focused BH correctness/cache gate; timing is recorded without a fabricated threshold."""
    torch.manual_seed(2026)
    dim = 256
    batch_rows = 32
    num_devices = ttnn_mesh_device.get_num_devices()
    full_weight = torch.randn(vocab_size, dim, dtype=torch.bfloat16)
    torch_input = torch.randn(1, 1, batch_rows, dim, dtype=torch.bfloat16)
    reference = torch.nn.functional.linear(torch_input, full_weight)
    splits = _prepare_lm_head_weights(full_weight, vocab_size, dim, num_devices, max_columns_per_device)
    assert len(splits) == expected_splits
    logical_split_sizes = [max_columns_per_device] * (expected_splits - 1) + [
        vocab_size // num_devices - max_columns_per_device * (expected_splits - 1)
    ]
    assert max(logical_split_sizes) == max_columns_per_device

    compute = ttnn.init_device_compute_kernel_config(
        ttnn.device.Arch.BLACKHOLE,
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )
    common = LMHead1DConfig(
        output_weights=[LazyWeight(source=split, dtype=ttnn.bfloat8_b) for split in splits],
        mesh_device=ttnn_mesh_device,
        dim=dim,
        max_batch_size=1,
        lm_head_dtype=ttnn.bfloat16,
        output_split_sizes=logical_split_sizes,
        compute_kernel_config=compute,
    )
    model = LMHead1D.from_config(common)
    assert len(model.config.output_weights) == expected_splits
    ttnn_mesh_device.enable_program_cache()
    ttnn_mesh_device.clear_program_cache()
    request.addfinalizer(ttnn_mesh_device.disable_and_clear_program_cache)
    input_weight = LazyWeight(source=torch_input)

    def run_once():
        output = model.forward(input_weight)
        ttnn.synchronize_device(ttnn_mesh_device)
        return output

    output = run_once()
    actual = to_torch_auto_compose(output)[..., :vocab_size]
    output.deallocate(True)
    passing, pcc_message = comp_pcc(reference, actual, 0.999)
    assert passing, f"Blackhole LMHead1D PCC failed: {pcc_message}"

    cache_entries = ttnn_mesh_device.num_program_cache_entries()
    assert cache_entries > 0
    timings_ms = []
    for _ in range(3):
        start = time.perf_counter()
        output = run_once()
        timings_ms.append((time.perf_counter() - start) * 1000)
        assert ttnn_mesh_device.num_program_cache_entries() == cache_entries
        output.deallocate(True)
    logger.info(
        "BH LMHead1D measurement mesh={} dim={} vocab={} max_columns={}: warm-cache mean={:.3f} ms, samples={}",
        tuple(ttnn_mesh_device.shape),
        dim,
        vocab_size,
        max_columns_per_device,
        sum(timings_ms) / len(timings_ms),
        timings_ms,
    )


@pytest.mark.device
@pytest.mark.wormhole
@pytest.mark.parametrize(
    "ttnn_mesh_device",
    [pytest.param((1, 1), id="n150-single-split")],
    indirect=True,
)
def test_lm_head_1d_wormhole_common_config_correctness_cache_and_timing(request, ttnn_mesh_device):
    """Focused WH correctness/cache gate; timing is evidence, not a threshold."""
    torch.manual_seed(2026)
    dim = 256
    vocab_size = 8192
    batch_rows = 32
    full_weight = torch.randn(vocab_size, dim, dtype=torch.bfloat16)
    torch_input = torch.randn(1, 1, batch_rows, dim, dtype=torch.bfloat16)
    reference = torch.nn.functional.linear(torch_input, full_weight)
    splits = _prepare_lm_head_weights(full_weight, vocab_size, dim, 1, vocab_size)
    assert len(splits) == 1

    compute = ttnn.init_device_compute_kernel_config(
        ttnn.device.Arch.WORMHOLE_B0,
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )
    common = LMHead1DConfig(
        output_weights=[LazyWeight(source=splits[0], dtype=ttnn.bfloat8_b)],
        mesh_device=ttnn_mesh_device,
        dim=dim,
        max_batch_size=1,
        lm_head_dtype=ttnn.bfloat16,
        output_split_sizes=[vocab_size],
        compute_kernel_config=compute,
    )
    model = LMHead1D.from_config(common)
    assert isinstance(model.config, LMHead1DConfig)
    assert model.config is not common
    assert not hasattr(model, "arch_config")
    ttnn_mesh_device.enable_program_cache()
    ttnn_mesh_device.clear_program_cache()
    request.addfinalizer(ttnn_mesh_device.disable_and_clear_program_cache)
    input_weight = LazyWeight(source=torch_input)

    def run_once():
        output = model.forward(input_weight)
        ttnn.synchronize_device(ttnn_mesh_device)
        return output

    output = run_once()
    actual = to_torch_auto_compose(output)[..., :vocab_size]
    output.deallocate(True)
    passing, pcc_message = comp_pcc(reference, actual, 0.999)
    assert passing, f"Wormhole LMHead1D PCC failed: {pcc_message}"

    cache_entries = ttnn_mesh_device.num_program_cache_entries()
    assert cache_entries > 0
    timings_ms = []
    for _ in range(3):
        start = time.perf_counter()
        output = run_once()
        timings_ms.append((time.perf_counter() - start) * 1000)
        assert ttnn_mesh_device.num_program_cache_entries() == cache_entries
        output.deallocate(True)
    logger.info(
        "WH LMHead1D measurement mesh={} dim={} vocab={}: warm-cache mean={:.3f} ms, samples={}",
        tuple(ttnn_mesh_device.shape),
        dim,
        vocab_size,
        sum(timings_ms) / len(timings_ms),
        timings_ms,
    )


# ============================================================================
# from_model_args backward compatibility test
# ============================================================================


@pytest.mark.device
@pytest.mark.parametrize(
    "ttnn_mesh_device",
    [(1, 1), (1, 2), (1, 8)],
    ids=["1x1", "1x2", "1x8"],
    indirect=True,
)
def test_lm_head_1d_vs_reference_from_model_args(ttnn_mesh_device: ttnn.MeshDevice):
    pytest.skip(
        "TTTv1 compatibility characterization retired; standalone constructor coverage lives in tests/host/test_foundation_boundary.py"
    )
