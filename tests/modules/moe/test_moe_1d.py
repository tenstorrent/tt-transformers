# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
import torch
import ttnn

from tt_transformers.modules.moe.moe_1d import (
    MoE1D,
    MoE1DConfig,
    _largest_divisor_at_most,
    _pack_experts,
    _sparse_matmul_program_config,
    clear_fused_decode_cache,
    moe_weights_from_hf,
)
from tt_transformers.ops.moe.fused_decode import SharedExpert, round_sig

MESH_1X1 = pytest.mark.parametrize(
    "ttnn_mesh_device",
    [pytest.param({"mesh_shape": (1, 1), "trace_region_size": 32 << 20}, id="p150-1x1")],
    indirect=True,
)


@pytest.mark.host
def test_sparse_matmul_program_config_shapes():
    assert _largest_divisor_at_most(12, 8) == 6
    config = _sparse_matmul_program_config(32, 256, 2048, target_cores=16, in0_block_w_cap=64)
    assert config.compute_with_storage_grid_size == ttnn.CoreCoord(1, 8)
    assert config.in0_block_w == 64
    assert config.per_core_M == 1
    assert config.per_core_N == 1
    assert config.mcast_in0


@pytest.mark.host
@pytest.mark.parametrize("parallelism", ["tensor", "expert"])
@pytest.mark.parametrize("gate_first", [False, True])
def test_pack_experts_matches_device_shards(parallelism, gate_first):
    experts, intermediate, hidden, devices = 4, 8, 6, 2
    gate = torch.arange(experts * intermediate * hidden).reshape(experts, intermediate, hidden).float()
    up = gate + 10_000
    down = torch.arange(experts * hidden * intermediate).reshape(experts, hidden, intermediate).float()

    up_gate, packed_down = _pack_experts(
        gate,
        up,
        down,
        parallelism=parallelism,
        num_devices=devices,
        gate_first=gate_first,
    )

    assert up_gate.shape == (1, experts, hidden, 2 * intermediate)
    assert torch.equal(packed_down, round_sig(down.transpose(1, 2)).unsqueeze(0))
    first, second = (gate, up) if gate_first else (up, gate)
    first, second = round_sig(first.transpose(1, 2)), round_sig(second.transpose(1, 2))
    if parallelism == "expert":
        assert torch.equal(up_gate[0], torch.cat([first, second], -1))
    else:
        local = intermediate // devices
        for device in range(devices):
            expected = torch.cat(
                [
                    first[..., device * local : (device + 1) * local],
                    second[..., device * local : (device + 1) * local],
                ],
                -1,
            )
            assert torch.equal(up_gate[0, ..., device * 2 * local : (device + 1) * 2 * local], expected)


@pytest.mark.host
@pytest.mark.parametrize(
    "parallelism, experts, hidden, intermediate, devices",
    [
        pytest.param("tensor", 256, 2048, 512, 4, id="ornith-1.5-35b-a3b"),
        pytest.param("expert", 128, 2048, 768, 4, id="qwen3-coder-30b-a3b"),
    ],
)
def test_real_model_packed_shapes(parallelism, experts, hidden, intermediate, devices):
    gate = torch.empty(experts, intermediate, hidden, device="meta")
    up = torch.empty_like(gate)
    down = torch.empty(experts, hidden, intermediate, device="meta")
    up_gate, packed_down = _pack_experts(
        gate,
        up,
        down,
        parallelism=parallelism,
        num_devices=devices,
        gate_first=parallelism == "expert",
    )
    assert up_gate.shape == (1, experts, hidden, 2 * intermediate)
    assert packed_down.shape == (1, experts, intermediate, hidden)


@pytest.mark.host
def test_hf_weight_loader_includes_shared_expert_and_cache_names(tmp_path: Path):
    experts, intermediate, hidden, devices = 4, 8, 6, 2
    shapes = [(experts, intermediate, hidden), (experts, intermediate, hidden), (experts, hidden, intermediate)]
    gate, up, down = [torch.randn(shape) for shape in shapes]
    shared = SharedExpert(
        expert_gate=torch.randn(1, hidden),
        gate_proj=torch.randn(intermediate, hidden),
        up_proj=torch.randn(intermediate, hidden),
        down_proj=torch.randn(hidden, intermediate),
    )
    weights = moe_weights_from_hf(
        torch.randn(experts, hidden),
        gate,
        up,
        down,
        num_devices=devices,
        parallelism="tensor",
        weight_dtype=ttnn.bfloat8_b,
        shared_expert=shared,
        cache_dir=tmp_path,
        cache_prefix="layer_3",
    )
    assert weights.router.source.shape == (hidden, experts + 32)
    assert weights.shared_up_gate.source.shape == (1, 1, hidden, 2 * intermediate)
    assert weights.shared_down.source.shape == (1, 1, intermediate, hidden)
    assert weights.up_gate.cache_dir_weight_name == (tmp_path, "layer_3_up_gate")
    assert weights.shared_down.cache_dir_weight_name == (tmp_path, "layer_3_shared_down")


def _swiglu(x, gate, up, down):
    return (torch.nn.functional.silu(x @ gate.T) * (x @ up.T)) @ down.T


def _reference(x, layer, k):
    x = x.float()[0, 0]
    values, indices = torch.topk(x @ layer["router"].T, k, dim=-1)
    routing = torch.softmax(values, -1)
    experts = [round_sig(w) for w in (layer["gate"], layer["up"], layer["down"])]
    output = torch.stack(
        [
            sum(routing[t, s] * _swiglu(x[t], *(w[indices[t, s]] for w in experts)) for s in range(k))
            for t in range(x.shape[0])
        ]
    )
    if shared := layer["shared"]:
        gate = torch.sigmoid(x @ shared.expert_gate.T)
        output += gate * _swiglu(x, *(round_sig(w) for w in shared[1:]))
    return output


def _layer(seed, hidden, intermediate, experts, shared):
    g = torch.Generator().manual_seed(seed)
    randn = lambda *shape: torch.randn(*shape, generator=g) * 0.03  # noqa: E731
    return dict(
        router=randn(experts, hidden),
        gate=randn(experts, intermediate, hidden),
        up=randn(experts, intermediate, hidden),
        down=randn(experts, hidden, intermediate),
        shared=SharedExpert(
            randn(1, hidden), *(randn(intermediate, hidden) for _ in range(2)), randn(hidden, intermediate)
        )
        if shared
        else None,
    )


def _module(device, layer, k, parallelism, gate_first):
    weights = moe_weights_from_hf(
        layer["router"],
        layer["gate"],
        layer["up"],
        layer["down"],
        num_devices=1,
        parallelism=parallelism,
        weight_dtype=ttnn.bfloat8_b,
        shared_expert=layer["shared"],
        gate_first=gate_first,
    )
    config = MoE1DConfig(**weights.__dict__, k=k, parallelism=parallelism, gate_first=gate_first, mesh_device=device)
    return MoE1D.from_config(config)


def _run(device, module, x, mode):
    memory_config = ttnn.L1_MEMORY_CONFIG if mode == "decode" else ttnn.DRAM_MEMORY_CONFIG
    x_tt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config)
    return ttnn.to_torch(module(x_tt, mode)).float()[0, 0, : x.shape[-2]]


def _pcc(a, b):
    return torch.corrcoef(torch.stack([a.flatten().double(), b.flatten().double()]))[0, 1].item()


@pytest.fixture
def fresh_fused_decode_cache():
    clear_fused_decode_cache()
    yield
    clear_fused_decode_cache()


@pytest.mark.device
@pytest.mark.blackhole
@MESH_1X1
@pytest.mark.parametrize(
    "parallelism, gate_first, shared",
    [("tensor", False, False), ("tensor", False, True), ("expert", True, False)],
    ids=["tensor", "tensor-shared", "expert-gate-first"],
)
@pytest.mark.parametrize("mode, seq_len", [("decode", 1), ("decode", 32), ("prefill", 48), ("prefill", 64)])
def test_moe_1d_matches_torch(
    ttnn_mesh_device, fresh_fused_decode_cache, parallelism, gate_first, shared, mode, seq_len
):
    hidden, intermediate, experts, k = 2048, 128, 64, 8
    layer = _layer(7, hidden, intermediate, experts, shared)
    module = _module(ttnn_mesh_device, layer, k, parallelism, gate_first)
    x = torch.randn(1, 1, seq_len, hidden, generator=torch.Generator().manual_seed(8)).bfloat16()
    output = _run(ttnn_mesh_device, module, x, mode)
    pcc = _pcc(output, _reference(x, layer, k))
    assert pcc > 0.99, pcc


@pytest.mark.device
@pytest.mark.blackhole
@MESH_1X1
def test_moe_1d_layers_share_one_fused_decode(ttnn_mesh_device, fresh_fused_decode_cache):
    hidden, intermediate, experts, k = 2048, 128, 64, 8
    layers = [_layer(seed, hidden, intermediate, experts, shared=True) for seed in (9, 10)]
    modules = [_module(ttnn_mesh_device, layer, k, "tensor", False) for layer in layers]
    assert modules[0]._decode_op is modules[1]._decode_op
    x = torch.randn(1, 1, 1, hidden, generator=torch.Generator().manual_seed(11)).bfloat16()
    for index in (0, 1, 0):
        pcc = _pcc(_run(ttnn_mesh_device, modules[index], x, "decode"), _reference(x, layers[index], k))
        assert pcc > 0.99, (index, pcc)


@pytest.mark.device
@pytest.mark.blackhole
@pytest.mark.parametrize(
    "ttnn_mesh_device",
    [pytest.param({"mesh_shape": (1, 4), "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}, id="p150x4-1x4-ring")],
    indirect=True,
)
@pytest.mark.parametrize("mode, seq_len", [("decode", 1), ("decode", 32), ("prefill", 48)])
def test_moe_1d_expert_parallel_across_devices(
    ttnn_mesh_device, require_blackhole_mesh_device, fresh_fused_decode_cache, mode, seq_len
):
    """Each die holds a quarter of the experts; the reduce-scattered partials sum to the full MoE output."""
    hidden, intermediate, experts, k = 2048, 128, 64, 8
    layer = _layer(12, hidden, intermediate, experts, shared=False)
    weights = moe_weights_from_hf(
        layer["router"],
        layer["gate"],
        layer["up"],
        layer["down"],
        num_devices=4,
        parallelism="expert",
        weight_dtype=ttnn.bfloat8_b,
        gate_first=True,
    )
    config = MoE1DConfig(**weights.__dict__, k=k, parallelism="expert", gate_first=True, mesh_device=ttnn_mesh_device)
    module = MoE1D.from_config(config)
    x = torch.randn(1, 1, seq_len, hidden, generator=torch.Generator().manual_seed(13)).bfloat16()
    x_tt = ttnn.from_torch(
        x,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=ttnn_mesh_device,
        memory_config=ttnn.L1_MEMORY_CONFIG if mode == "decode" else ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(ttnn_mesh_device),
    )
    out = module(x_tt, mode)
    output = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(ttnn_mesh_device, dim=3)).float()
    pcc = _pcc(output[0, 0, :seq_len], _reference(x, layer, k))
    assert pcc > 0.99, pcc


@pytest.mark.device
@pytest.mark.blackhole
@pytest.mark.parametrize(
    "ttnn_mesh_device, parallelism",
    [
        pytest.param({"mesh_shape": (1, 1), "trace_region_size": 32 << 20}, "tensor", id="p150-1x1-tensor-shared"),
        pytest.param(
            {"mesh_shape": (1, 4), "trace_region_size": 32 << 20, "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING},
            "expert",
            id="p150x4-1x4-ring-expert",
        ),
    ],
    indirect=["ttnn_mesh_device"],
)
@pytest.mark.parametrize("mode, seq_len", [("decode", 1), ("decode", 32), ("prefill", 128)])
def test_moe_1d_replays_a_trace_with_new_inputs(
    ttnn_mesh_device, require_blackhole_mesh_device, fresh_fused_decode_cache, parallelism, mode, seq_len
):
    hidden, intermediate, experts, k = 2048, 128, 64, 8
    devices = ttnn_mesh_device.get_num_devices()
    shared = parallelism == "tensor"
    layer = _layer(14, hidden, intermediate, experts, shared=shared)
    weights = moe_weights_from_hf(
        layer["router"],
        layer["gate"],
        layer["up"],
        layer["down"],
        num_devices=devices,
        parallelism=parallelism,
        weight_dtype=ttnn.bfloat8_b,
        shared_expert=layer["shared"],
        gate_first=not shared,
    )
    config = MoE1DConfig(
        **weights.__dict__, k=k, parallelism=parallelism, gate_first=not shared, mesh_device=ttnn_mesh_device
    )
    module = MoE1D.from_config(config)
    memory_config = ttnn.L1_MEMORY_CONFIG if mode == "decode" else ttnn.DRAM_MEMORY_CONFIG
    replicate = ttnn.ReplicateTensorToMesh(ttnn_mesh_device) if devices > 1 else None

    def host(x):
        return ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=replicate)

    x_tt = host(torch.zeros(1, 1, seq_len, hidden)).to(ttnn_mesh_device, memory_config)
    module(x_tt, mode)  # compile before capture
    trace = ttnn.begin_trace_capture(ttnn_mesh_device, cq_id=0)
    out = module(x_tt, mode)
    ttnn.end_trace_capture(ttnn_mesh_device, trace, cq_id=0)
    composer = ttnn.ConcatMeshToTensor(ttnn_mesh_device, dim=3) if devices > 1 else None
    try:
        for seed in (15, 16):
            x = torch.randn(1, 1, seq_len, hidden, generator=torch.Generator().manual_seed(seed)).bfloat16()
            ttnn.copy_host_to_device_tensor(host(x), x_tt)
            ttnn.execute_trace(ttnn_mesh_device, trace, cq_id=0, blocking=True)
            output = ttnn.to_torch(out, mesh_composer=composer).float()[0, 0, :seq_len]
            pcc = _pcc(output, _reference(x, layer, k))
            assert pcc > 0.99, (seed, pcc)
    finally:
        ttnn.release_trace(ttnn_mesh_device, trace)
