# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Qwen3-Coder-30B-A3B on a four-device Blackhole mesh."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch
import ttnn
from loguru import logger

from tt_transformers.cache_environment import environment_flag, offline_mode
from tt_transformers.llm_runtime.config import fit_paged_kv_num_blocks
from tt_transformers.models.qwen3_32b import model as qwen3
from tt_transformers.models.qwen3_32b import weight_utils
from tt_transformers.modules.attention.attention_1d import Attention1D, Attention1DConfig
from tt_transformers.modules.embedding.embedding_1d import Embedding1D, Embedding1DConfig
from tt_transformers.modules.lazy_weight import LazyWeight
from tt_transformers.modules.lm_head.lm_head_1d import LMHead1D, LMHead1DConfig
from tt_transformers.modules.moe.moe_1d import (
    MoE1D,
    MoE1DConfig,
    _sparse_matmul_program_config,
    moe_weights_from_hf,
)
from tt_transformers.modules.rmsnorm.rmsnorm_1d import RMSNorm1D, RMSNorm1DConfig
from tt_transformers.modules.rope.rope_1d import Rope1DConfig, RotarySetup1D
from tt_transformers.modules.tt_ccl import get_tt_ccl

DEFAULT_HF_MODEL = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
DEFAULT_HF_REVISION = "b2cff646eb4bb1d68355c01b18ae02e7cf42d120"
SUPPORTED_CLUSTER_TYPES = (
    ttnn.cluster.ClusterType.P150_X4,
    ttnn.cluster.ClusterType.P300_X2,
)

Qwen3Coder30BA3BPagedAttentionConfig = qwen3.Qwen3_32BPagedAttentionConfig
Qwen3Coder30BA3BExecutorRuntimeConfig = qwen3.Qwen3_32BExecutorRuntimeConfig


@dataclass
class Qwen3Coder30BA3BConfig(qwen3.Qwen3_32BConfig):
    num_experts: int = 128
    num_experts_per_token: int = 8


@dataclass(frozen=True)
class Qwen3Coder30BA3BPrecisionConfig(qwen3.Qwen3_32BPrecisionConfig):
    # bfp4 in both profiles: the fused MoE decode's gate|up weight ring does not fit L1 in bfp8 at this shape
    expert_gate_up_dtype: ttnn.DataType = ttnn.bfloat4_b
    expert_down_dtype: ttnn.DataType = ttnn.bfloat4_b
    expert_prefill_gate_up_in0_block_w: int = 16
    expert_prefill_down_in0_block_w: int = 12


QWEN3_CODER_30B_A3B_ACCURACY = Qwen3Coder30BA3BPrecisionConfig()
QWEN3_CODER_30B_A3B_PERFORMANCE = Qwen3Coder30BA3BPrecisionConfig(
    wqkv_dtype=ttnn.bfloat8_b,
    wo_dtype=ttnn.bfloat8_b,
    kv_cache_dtype=ttnn.bfloat16,
    attn_decode_qkv_kernel=None,
    attn_decode_sdpa_kernel=None,
    attn_decode_wo_kernel=None,
    attn_prefill_qkv_kernel=None,
    attn_prefill_sdpa_kernel=None,
    attn_prefill_wo_kernel=None,
)


def _validate_checkpoint_config(hf_config: Any, *, num_devices: int) -> None:
    expected = {
        "model_type": "qwen3_moe",
        "num_hidden_layers": 48,
        "hidden_size": 2048,
        "intermediate_size": 6144,
        "moe_intermediate_size": 768,
        "hidden_act": "silu",
        "decoder_sparse_step": 1,
        "num_attention_heads": 32,
        "num_key_value_heads": 4,
        "head_dim": 128,
        "num_experts": 128,
        "num_experts_per_tok": 8,
        "vocab_size": 151936,
    }
    mismatches = {
        name: (getattr(hf_config, name, None), value)
        for name, value in expected.items()
        if getattr(hf_config, name, None) != value
    }
    if mismatches:
        raise ValueError(f"Unexpected Qwen3-Coder-30B-A3B geometry: {mismatches}")
    if num_devices != 4:
        raise ValueError(f"Qwen3-Coder-30B-A3B requires four devices, got {num_devices}")
    if hf_config.num_attention_heads % num_devices or hf_config.num_key_value_heads % num_devices:
        raise ValueError("Checkpoint attention heads must divide evenly across four devices")
    if not bool(getattr(hf_config, "norm_topk_prob", False)):
        raise ValueError("Qwen3-Coder-30B-A3B requires normalized top-k routing")
    if bool(getattr(hf_config, "attention_bias", False)):
        raise ValueError("Qwen3-Coder-30B-A3B requires bias-free attention")
    if bool(getattr(hf_config, "tie_word_embeddings", False)):
        raise ValueError("Qwen3-Coder-30B-A3B requires an untied LM head")
    if list(getattr(hf_config, "mlp_only_layers", ())) or bool(getattr(hf_config, "use_sliding_window", False)):
        raise ValueError("Qwen3-Coder-30B-A3B requires MoE attention layers throughout")
    if _rope_theta(hf_config) != 10_000_000:
        raise ValueError("Qwen3-Coder-30B-A3B requires rope_theta=10000000")


def _rope_theta(hf_config: Any) -> float | None:
    """rope_theta from the config, or its rope_parameters in newer transformers."""
    theta = getattr(hf_config, "rope_theta", None)
    if theta is None:
        theta = (getattr(hf_config, "rope_parameters", None) or {}).get("rope_theta")
    return None if theta is None else float(theta)


def _resolve_sku(mesh_device: ttnn.MeshDevice):
    arch = mesh_device.arch()
    cluster_type = ttnn.cluster.get_cluster_type()
    num_devices = mesh_device.get_num_devices()
    if arch != ttnn.device.Arch.BLACKHOLE or cluster_type not in SUPPORTED_CLUSTER_TYPES or num_devices != 4:
        raise ValueError(
            "Qwen3-Coder-30B-A3B supports Blackhole P150_X4/P300_X2 with four devices; "
            f"got arch={arch}, cluster_type={cluster_type}, num_devices={num_devices}"
        )
    return qwen3._resolve_qwen3_32b_sku_overlay(
        arch=arch,
        cluster_type=cluster_type,
        num_dev=num_devices,
        mesh_device=mesh_device,
    )


def _expert_weights(hf_mlp: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    packed = hf_mlp.experts.gate_up_proj.detach()
    down = hf_mlp.experts.down_proj.detach()
    if packed.ndim != 3 or packed.shape[1] % 2:
        raise ValueError("Expected experts.gate_up_proj with shape [E, 2I, H]")
    intermediate = packed.shape[1] // 2
    return packed[:, :intermediate], packed[:, intermediate:], down


def _build_moe(
    *,
    hf_mlp: Any,
    qcfg: Qwen3Coder30BA3BConfig,
    mesh_device: ttnn.MeshDevice,
    tt_ccl: Any,
    topology: ttnn.Topology,
    precision: Qwen3Coder30BA3BPrecisionConfig,
    cache_path: Path | None,
    prefix: str,
) -> MoE1D:
    gate, up, down = _expert_weights(hf_mlp)
    weights = moe_weights_from_hf(
        hf_mlp.gate.weight.detach(),
        gate,
        up,
        down,
        num_devices=qcfg.num_devices,
        parallelism="expert",
        weight_dtype=precision.expert_gate_up_dtype,
        down_dtype=precision.expert_down_dtype,
        gate_first=True,
        cache_dir=cache_path / "moe" if cache_path else None,
        cache_prefix=prefix,
    )
    moe = MoE1D.from_config(
        MoE1DConfig(
            router=weights.router,
            up_gate=weights.up_gate,
            down=weights.down,
            k=qcfg.num_experts_per_token,
            parallelism="expert",
            mesh_device=mesh_device,
            tt_ccl=tt_ccl,
            topology=topology,
            hidden=qcfg.dim,
            intermediate=qcfg.hidden_dim,
            num_experts=qcfg.num_experts,
            experts_per_device=qcfg.num_experts // qcfg.num_devices,
            gate_first=True,
            num_reduce_scatter_links=1,
            prefill_num_reduce_scatter_links=2,
            prefill_gate_up_program_config=_sparse_matmul_program_config(
                32,
                2 * qcfg.hidden_dim,
                qcfg.dim,
                target_cores=64,
                in0_block_w_cap=precision.expert_prefill_gate_up_in0_block_w,
            ),
            prefill_down_program_config=_sparse_matmul_program_config(
                32,
                qcfg.dim,
                qcfg.hidden_dim,
                target_cores=64,
                in0_block_w_cap=precision.expert_prefill_down_in0_block_w,
            ),
            prefill_compute_kernel_config=ttnn.init_device_compute_kernel_config(
                mesh_device.arch(),
                math_fidelity=ttnn.MathFidelity.LoFi,
                math_approx_mode=False,
                fp32_dest_acc_en=False,
                packer_l1_acc=False,
            ),
            # expert-sorted prefill wins from 128 tokens (at 64 the chunked sparse prefill is still faster)
            prefill_sorted_min_tokens=128,
            prefill_sorted_max_tokens=4096,
            prefill_sorted_gate_up_in0_block_w=precision.expert_prefill_gate_up_in0_block_w,
            prefill_sorted_down_in0_block_w=precision.expert_prefill_down_in0_block_w,
        )
    )
    # Upload eagerly and drop the host copies: packed fp32 experts are ~2.4 GB per layer and would otherwise
    # stay resident for every layer until the first forward pass.
    moe.load_device_weights()
    for weight in (moe.config.router, moe.config.up_gate, moe.config.down):
        weight.source = None
    return moe


class Qwen3Coder30BA3BDecoderLayer(qwen3.Qwen3_32BDecoderLayer):
    single_user: bool = False  # MoE decodes only the live row, through the fused decode

    def decode_forward(
        self,
        x: ttnn.Tensor,
        current_pos: ttnn.Tensor,
        rot_mats: tuple[ttnn.Tensor, ttnn.Tensor],
        page_table: ttnn.Tensor | None = None,
    ) -> ttnn.Tensor:
        xa = qwen3._all_gather_rmsnorm_tensor(
            self.input_layernorm,
            x,
            memory_config=self.input_layernorm.config.decode_memory_config,
        )
        attention_input = self.input_layernorm.decode_forward(xa)
        attention_output = self.self_attn.decode_forward(
            attention_input,
            current_pos,
            rot_mats,
            page_table=page_table,
        )
        hidden = ttnn.add(x, attention_output, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        hidden_full = qwen3._all_gather_rmsnorm_tensor(self.post_attention_layernorm, hidden)
        moe_input = self.post_attention_layernorm.decode_forward(hidden_full)
        if self.single_user:
            live = ttnn.slice(moe_input, (0, 0, 0, 0), (1, 1, 1, moe_input.shape[-1]))
            ttnn.deallocate(moe_input)
            moe_input = live
        if moe_input.memory_config() != self.mlp.config.decode_input_memcfg:
            converted = ttnn.to_memory_config(moe_input, self.mlp.config.decode_input_memcfg)
            ttnn.deallocate(moe_input)
            moe_input = converted
        moe_output = self.mlp.decode_forward(moe_input)
        return ttnn.add(hidden, moe_output, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def _build_lm_head(
    *,
    mesh_device: ttnn.MeshDevice,
    hf_lm_head: torch.nn.Module,
    qcfg: Qwen3Coder30BA3BConfig,
    precision: Qwen3Coder30BA3BPrecisionConfig,
    cache_path: Path | None,
) -> LMHead1D:
    columns_per_device = qcfg.vocab_size // qcfg.num_devices
    if qcfg.vocab_size % qcfg.num_devices or columns_per_device % ttnn.TILE_SIZE:
        raise ValueError("Qwen3-Coder-30B-A3B requires a tile-aligned vocabulary shard")
    source = hf_lm_head.weight.detach().to(torch.bfloat16).T.contiguous()
    weight = LazyWeight(
        source=source,
        dtype=precision.lm_head_dtype,
        device=mesh_device,
        mesh_mapper_config=ttnn.MeshMapperConfig(
            placements=[ttnn.PlacementShard(-1)],
            mesh_shape_override=ttnn.MeshShape([qcfg.num_devices]),
        ),
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        cache_dir_weight_name=(cache_path / "lm_head", "output") if cache_path else None,
    )
    return LMHead1D.from_config(
        LMHead1DConfig(
            output_weights=[weight],
            mesh_device=mesh_device,
            dim=qcfg.dim,
            max_batch_size=qcfg.max_batch_size,
            program_configs=[None],
            compute_kernel_config=qwen3._HIFI2_FP16_KERNEL,
            output_split_sizes=[columns_per_device],
            lm_head_dtype=precision.lm_head_dtype,
            output_memcfg=ttnn.DRAM_MEMORY_CONFIG,
            input_memcfg=ttnn.DRAM_MEMORY_CONFIG,
            weights_memcfgs=[ttnn.DRAM_MEMORY_CONFIG],
        )
    )


def _build_decoder_layer(
    *,
    idx: int,
    hf_layer: Any,
    qcfg: Qwen3Coder30BA3BConfig,
    mesh_device: ttnn.MeshDevice,
    tt_ccl: Any,
    topology: ttnn.Topology,
    torch_dtype: torch.dtype,
    precision: Qwen3Coder30BA3BPrecisionConfig,
    executor_mode: bool,
    paged_cfg: Qwen3Coder30BA3BPagedAttentionConfig | None,
    cache_path: Path | None,
    sku: Any,
) -> Qwen3Coder30BA3BDecoderLayer:
    prefix = f"layer{idx}"
    wqkv, wo, qn, kn, _ = weight_utils.attention_wqkv_wo_from_hf_layer(hf_layer.self_attn, qcfg.num_devices)

    def lazy(tensor: torch.Tensor, dtype: ttnn.DataType, group: str, name: str) -> LazyWeight:
        cache = (cache_path / group, name) if cache_path else None
        return qwen3._lazy(tensor, dtype=dtype, cache=cache)

    def qk_norm_config(weight: torch.Tensor | None, name: str):
        if weight is None:
            return None
        common = RMSNorm1DConfig(
            weight=lazy(weight[None, None, None, :].to(torch_dtype), ttnn.bfloat16, "attn", f"{prefix}_{name}"),
            mesh_device=mesh_device,
            eps=qcfg.rms_norm_eps,
            decode_in_sharded=False,
            decode_out_sharded=False,
            prefill_distributed=False,
            tt_ccl=tt_ccl,
        )
        return replace(common, compute_kernel_config=qwen3._HIFI4_FP32_KERNEL)

    attention = Attention1D.from_config(
        qwen3._qwen3_attention_config(
            Attention1DConfig(
                wqkv=lazy(wqkv, precision.wqkv_dtype, "attn", f"{prefix}_wqkv"),
                wo=lazy(wo, precision.wo_dtype, "attn", f"{prefix}_wo"),
                mesh_device=mesh_device,
                tt_ccl=tt_ccl,
                topology=topology,
                num_reduce_scatter_links=2,
                decode_num_reduce_scatter_links=1,
                num_all_gather_links=2,
                n_heads=qcfg.n_heads,
                n_kv_heads=qcfg.n_kv_heads,
                head_dim=qcfg.head_dim,
                max_batch_size=qcfg.max_batch_size,
                max_seq_len=qcfg.max_seq_len,
                q_norm_config=qk_norm_config(qn, "qn"),
                k_norm_config=qk_norm_config(kn, "kn"),
                use_vllm_paged_kv_cache=executor_mode,
                paged_attention_config=paged_cfg,
                kv_cache=None,
                kv_cache_dtype=precision.kv_cache_dtype,
                prefill_qkv_minimal_matmul=sku.prefill_minimal_matmul,
            ),
            sku=sku,
            precision=precision,
        )
    )

    moe = _build_moe(
        hf_mlp=hf_layer.mlp,
        qcfg=qcfg,
        mesh_device=mesh_device,
        tt_ccl=tt_ccl,
        topology=topology,
        precision=precision,
        cache_path=cache_path,
        prefix=prefix,
    )

    def build_norm(hf_norm: Any, name: str, *, moe_input: bool = False) -> RMSNorm1D:
        common = RMSNorm1DConfig(
            weight=lazy(
                weight_utils.rms_weight_torch(hf_norm).to(torch_dtype),
                ttnn.bfloat16,
                "norm",
                f"{prefix}_{name}",
            ),
            mesh_device=mesh_device,
            eps=qcfg.rms_norm_eps,
            max_batch_size=qcfg.max_batch_size,
            tt_ccl=tt_ccl,
            decode_in_sharded=not moe_input,
            decode_out_sharded=not moe_input,
            prefill_distributed=True,
        )
        return RMSNorm1D.from_config(replace(common, compute_kernel_config=qwen3._HIFI4_FP32_KERNEL))

    layer = Qwen3Coder30BA3BDecoderLayer(
        input_layernorm=build_norm(hf_layer.input_layernorm, "pre_attn"),
        self_attn=attention,
        post_attention_layernorm=build_norm(hf_layer.post_attention_layernorm, "post_attn", moe_input=True),
        mlp=moe,
    )
    layer.single_user = qcfg.max_batch_size == 1
    return layer


class Qwen3Coder30BA3B(qwen3.Qwen3_32B):
    """Qwen3-Coder-30B-A3B composed from native TTTv2 modules."""

    @classmethod
    def from_pretrained(
        cls,
        mesh_device: ttnn.MeshDevice,
        hf_model_id: str = DEFAULT_HF_MODEL,
        *,
        revision: str | None = DEFAULT_HF_REVISION,
        max_batch_size: int = 32,
        max_seq_len: int = 4096,
        num_layers: int | None = None,
        cache_dir: Path | str | None = None,
        precision: Qwen3Coder30BA3BPrecisionConfig = QWEN3_CODER_30B_A3B_ACCURACY,
        block_size: int = 32,
        max_num_blocks: int | None = None,
        executor_mode: bool = False,
        disable_batched_prefill: bool | None = None,
    ) -> Qwen3Coder30BA3B:
        from transformers import AutoConfig, AutoModelForCausalLM

        cache_path = Path(cache_dir) if cache_dir else None
        sku = _resolve_sku(mesh_device)
        num_devices = mesh_device.get_num_devices()
        tt_ccl = get_tt_ccl(mesh_device)
        local_files_only = offline_mode()
        hf_cfg = AutoConfig.from_pretrained(hf_model_id, revision=revision, local_files_only=local_files_only)
        _validate_checkpoint_config(hf_cfg, num_devices=num_devices)
        n_layers = hf_cfg.num_hidden_layers if num_layers is None else num_layers
        if not 1 <= n_layers <= hf_cfg.num_hidden_layers:
            raise ValueError(f"num_layers must be in [1, {hf_cfg.num_hidden_layers}], got {n_layers}")

        logger.info(f"Loading HF weights: {hf_model_id} (revision={revision})")
        hf = AutoModelForCausalLM.from_pretrained(
            hf_model_id,
            revision=revision,
            torch_dtype=torch.bfloat16,
            local_files_only=local_files_only,
        )
        hf.eval()
        base = hf.model
        dim = hf_cfg.hidden_size
        head_dim = hf_cfg.head_dim
        rope_len = ((max(max_seq_len * 2, 8192) + 127) // 128) * 128
        if max_num_blocks is None:
            max_num_blocks = fit_paged_kv_num_blocks(
                -(-max_seq_len // block_size) * max_batch_size, max_seq_len=max_seq_len, block_size=block_size
            )
        paged_cfg = (
            Qwen3Coder30BA3BPagedAttentionConfig(block_size=block_size, max_num_blocks=max_num_blocks)
            if executor_mode
            else None
        )
        qcfg = Qwen3Coder30BA3BConfig(
            hf_model_id=hf_model_id,
            dim=dim,
            n_heads=hf_cfg.num_attention_heads,
            n_kv_heads=hf_cfg.num_key_value_heads,
            head_dim=head_dim,
            hidden_dim=hf_cfg.moe_intermediate_size,
            vocab_size=hf_cfg.vocab_size,
            rms_norm_eps=hf_cfg.rms_norm_eps,
            rope_theta=_rope_theta(hf_cfg),
            num_hidden_layers=n_layers,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            rope_table_len=rope_len,
            num_devices=num_devices,
            mesh_device=mesh_device,
            num_experts=hf_cfg.num_experts,
            num_experts_per_token=hf_cfg.num_experts_per_tok,
        )

        embed = Embedding1D.from_config(
            Embedding1DConfig(
                weights=qwen3._lazy(
                    weight_utils.embed_tokens_torch(base.embed_tokens),
                    dtype=ttnn.bfloat16,
                    cache=(cache_path / "embedding", "tok_embeddings") if cache_path else None,
                ),
                mesh_device=mesh_device,
                embed_scale=1.0,
            )
        )
        cos, sin = weight_utils.build_rope_cos_sin_torch(base.rotary_emb, rope_len, head_dim, torch.bfloat16)
        rope_setup = RotarySetup1D.from_config(
            Rope1DConfig(
                cos_matrix=qwen3._lazy(
                    cos,
                    dtype=ttnn.bfloat16,
                    cache=(cache_path / "rope", "cos") if cache_path else None,
                ),
                sin_matrix=qwen3._lazy(
                    sin,
                    dtype=ttnn.bfloat16,
                    cache=(cache_path / "rope", "sin") if cache_path else None,
                ),
                max_batch_size=max_batch_size,
                head_dim=head_dim,
                device=mesh_device,
                use_qk_fused=False,
                core_grid=sku.attention_decode_transformation_grid,
            )
        )
        layers = [
            _build_decoder_layer(
                idx=idx,
                hf_layer=base.layers[idx],
                qcfg=qcfg,
                mesh_device=mesh_device,
                tt_ccl=tt_ccl,
                topology=sku.topology,
                torch_dtype=torch.bfloat16,
                precision=precision,
                executor_mode=executor_mode,
                paged_cfg=paged_cfg,
                cache_path=cache_path,
                sku=sku,
            )
            for idx in range(n_layers)
        ]

        final_norm_common = RMSNorm1DConfig(
            weight=qwen3._lazy(
                weight_utils.rms_weight_torch(base.norm).to(torch.bfloat16),
                dtype=ttnn.bfloat16,
                cache=(cache_path / "norm", "final") if cache_path else None,
            ),
            mesh_device=mesh_device,
            eps=hf_cfg.rms_norm_eps,
            max_batch_size=max_batch_size,
            tt_ccl=tt_ccl,
            prefill_distributed=True,
        )
        final_norm = RMSNorm1D.from_config(replace(final_norm_common, compute_kernel_config=qwen3._HIFI4_FP32_KERNEL))
        lm_head = _build_lm_head(
            mesh_device=mesh_device,
            hf_lm_head=hf.lm_head,
            qcfg=qcfg,
            precision=precision,
            cache_path=cache_path,
        )
        del hf

        model = cls(qcfg, embed, rope_setup, layers, final_norm, lm_head, mesh_device)
        if executor_mode:
            model.model_args = Qwen3Coder30BA3BExecutorRuntimeConfig(
                n_layers=n_layers,
                n_kv_heads=hf_cfg.num_key_value_heads,
                head_dim=head_dim,
                max_batch_size=max_batch_size,
                max_seq_len=max_seq_len,
                cluster_shape=list(mesh_device.shape),
                model_cache_path=cache_path,
                kv_cache_dtype=precision.kv_cache_dtype,
                disable_batched_prefill=(
                    sku.disable_batched_prefill if disable_batched_prefill is None else disable_batched_prefill
                )
                or environment_flag("DISABLE_BATCHED_PREFILL"),
                batched_prefill_batched_extract=not environment_flag("DISABLE_BATCHED_EXTRACT"),
            )
        return model
