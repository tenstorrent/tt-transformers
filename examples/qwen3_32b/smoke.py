# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Smoke and accuracy tests for ``Qwen3_32B`` on T3K.

Requires Hugging Face weights locally (or network for fresh download). Uses
``MESH_DEVICE=T3K`` only; the port raises on any other mesh.

Run (T3K, internal KV smoke, 1-layer fast iteration)::

    MESH_DEVICE=T3K HF_MODEL=Qwen/Qwen3-32B \\
      QWEN3_32B_DEMO_NUM_LAYERS=1 \\
      pytest models/common/models/qwen3_32b/demo.py -v -k prefill_smoke

Executor + paged KV::

    MESH_DEVICE=T3K HF_MODEL=Qwen/Qwen3-32B \\
      QWEN3_32B_DEMO_NUM_LAYERS=1 \\
      pytest models/common/models/qwen3_32b/demo.py -v -k executor_prefill

``Attention1D`` shards heads across the mesh: 64 / 8 = 8 attention heads per device and
8 / 8 = 1 KV head per device.
"""

from __future__ import annotations

from functools import wraps
import inspect

import os

import torch
from transformers import AutoConfig, AutoModelForCausalLM

import ttnn
from tt_transformers.device_ownership import default_device_scope
from examples.common.runtime import TemporaryPathFactory, UnsupportedConfiguration, open_mesh_device
from tt_transformers.models.qwen3_32b.executor import (
    EagerQwen3_32BExecutor,
    TracedQwen3_32BExecutor,
    run_lm_head,
    run_prefill,
)
from examples.common.greedy import greedy_argmax_from_logits, greedy_decode_one_step
from tt_transformers.models.qwen3_32b.model import Qwen3_32B
from tt_transformers.device_utils import cleanup_model_case
from examples.common.run_helpers import make_contiguous_page_table
from examples.common.comparison import comp_pcc


def _skip_unless_t3k(mesh_device: ttnn.MeshDevice, hf_model_id: str) -> None:
    """Port targets T3K only; head divisibility is satisfied (64/8, 8/8)."""
    n_dev = mesh_device.get_num_devices()
    if n_dev != 8:
        raise UnsupportedConfiguration(f'Qwen3-32B port targets T3K (8 devices) only; got {n_dev}. Set MESH_DEVICE=T3K.')
    cfg = AutoConfig.from_pretrained(hf_model_id)
    n_h, n_kv = cfg.num_attention_heads, cfg.num_key_value_heads
    if n_h % n_dev != 0 or n_kv % n_dev != 0:
        raise UnsupportedConfiguration(f'Incompatible mesh for {hf_model_id}: {n_dev} devices need num_attention_heads ({n_h}) and num_key_value_heads ({n_kv}) each divisible by {n_dev}.')


def resolve_device_params(request, galaxy_type):
    """Match ``models/tt_transformers/conftest.py`` so ``fabric_config: True`` maps to a real fabric."""
    params = getattr(request, "param", {}).copy()

    mesh_device = {"N150": (1, 1), "N300": (1, 2), "N150x4": (1, 4), "T3K": (1, 8), "TG": (8, 4)}.get(
        os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids())
    )
    is_single_device = (mesh_device == (1, 1)) if isinstance(mesh_device, tuple) else (mesh_device == 1)

    if "fabric_config" in params:
        if is_single_device:
            params["fabric_config"] = None
        elif params["fabric_config"] is True:
            params["fabric_config"] = (
                ttnn.FabricConfig.FABRIC_1D_RING if galaxy_type == "6U" else ttnn.FabricConfig.FABRIC_1D
            )

    return params




def default_hf_model_id():
    return os.environ.get("HF_MODEL", "Qwen/Qwen3-32B")




def _scoped_default_device(function):
    """Run one smoke entry point under exact-restore TTNN default-device ownership."""

    @wraps(function)
    def invoke(mesh_device, *args, **kwargs):
        owner = f"{__name__}.{function.__name__}"
        with default_device_scope(ttnn, mesh_device, owner=owner):
            return function(mesh_device, *args, **kwargs)

    return invoke


@_scoped_default_device
def run_qwen3_32b_prefill_smoke(mesh_device, hf_model_id, seq_len: int, tmp_path_factory):
    """One prefill pass (truncated layers via env) and LM head → greedy token (internal KV)."""
    _skip_unless_t3k(mesh_device, hf_model_id)
    num_layers = int(os.environ.get("QWEN3_32B_DEMO_NUM_LAYERS", "1"))
    cache = tmp_path_factory.mktemp("qwen3_32b_cache")
    model = None
    try:
        model = Qwen3_32B.from_pretrained(
            mesh_device,
            hf_model_id,
            max_batch_size=32,
            max_seq_len=max(2048, seq_len),
            num_layers=num_layers,
            cache_dir=cache,
            executor_mode=False,
        )
    except Exception as e:
        raise UnsupportedConfiguration(f'Could not build model (weights / memory): {e}')

    try:
        toks = torch.zeros(1, 1, 1, seq_len, dtype=torch.int32)
        toks[..., :4] = torch.tensor([1, 2, 3, 4], dtype=torch.int32).view(1, 1, 1, 4)
        x = ttnn.from_torch(
            toks,
            device=mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(mesh_device),
        )
        hidden = run_prefill(model, x, start_pos=0)
        logits = run_lm_head(model, hidden)
        _ = greedy_argmax_from_logits(logits, mesh_device=mesh_device)
    finally:
        cleanup_model_case(model, mesh_device)


@_scoped_default_device
def run_qwen3_32b_decode_one_step(mesh_device, hf_model_id, tmp_path_factory):
    """Prefill 128 tokens, then one greedy decode step at position 128.

    Single-user (``max_batch_size=1``): the legacy ``decode_from_token_ids`` path only
    fills one KV slot per call, so the non-paged ``paged_update_cache`` validation
    (``num_indices == batch_size``) requires the model be built at batch=1.
    """
    _skip_unless_t3k(mesh_device, hf_model_id)
    num_layers = int(os.environ.get("QWEN3_32B_DEMO_NUM_LAYERS", "1"))
    cache = tmp_path_factory.mktemp("qwen3_32b_decode_cache")
    seq_len = 128
    model = None
    try:
        model = Qwen3_32B.from_pretrained(
            mesh_device,
            hf_model_id,
            max_batch_size=1,
            max_seq_len=max(512, seq_len + 8),
            num_layers=num_layers,
            cache_dir=cache,
            executor_mode=False,
        )
    except Exception as e:
        raise UnsupportedConfiguration(f'Could not build model (weights / memory): {e}')

    try:
        toks = torch.zeros(1, 1, 1, seq_len, dtype=torch.int32)
        toks[..., :4] = torch.tensor([1, 2, 3, 4], dtype=torch.int32).view(1, 1, 1, 4)
        x = ttnn.from_torch(
            toks,
            device=mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(mesh_device),
        )
        _ = run_prefill(model, x, start_pos=0)
        _ = greedy_decode_one_step(model, token_id=64, current_pos=seq_len)
    finally:
        cleanup_model_case(model, mesh_device)


@_scoped_default_device
def run_qwen3_32b_executor_prefill_smoke(mesh_device, hf_model_id, seq_len: int, tmp_path_factory):
    """``EagerQwen3_32BExecutor`` + paged KV: prefill returns host logits."""
    _skip_unless_t3k(mesh_device, hf_model_id)
    num_layers = int(os.environ.get("QWEN3_32B_DEMO_NUM_LAYERS", "1"))
    cache = tmp_path_factory.mktemp("qwen3_32b_exec_cache")
    model = None
    try:
        model = Qwen3_32B.from_pretrained(
            mesh_device,
            hf_model_id,
            max_batch_size=1,
            max_seq_len=max(2048, seq_len),
            num_layers=num_layers,
            cache_dir=cache,
            executor_mode=True,
        )
    except Exception as e:
        raise UnsupportedConfiguration(f'Could not build model: {e}')

    ma = model.model_args
    if not (ma is not None):
        raise AssertionError('condition failed at line 207')
    block_size = 32
    max_num_blocks = (ma.max_seq_len // block_size) * ma.max_batch_size
    kv_shape = (max_num_blocks, ma.n_kv_heads // mesh_device.get_num_devices(), block_size, ma.head_dim)

    logits = None
    try:
        ex = EagerQwen3_32BExecutor(model, mesh_device)
        kv = ex.allocate_kv_cache(kv_shape, torch.bfloat16, ma.n_layers)
        page_table = make_contiguous_page_table(1, ma.max_seq_len, block_size)
        toks = torch.zeros(1, seq_len, dtype=torch.long)
        toks[0, :4] = torch.tensor([1, 2, 3, 4], dtype=torch.long)
        logits = ex.prefill_forward(toks, page_table=page_table, kv_cache=kv)
    except Exception as e:
        raise UnsupportedConfiguration(f'Executor prefill not runnable: {e}')
    finally:
        cleanup_model_case(model, mesh_device)
    # Assert OUTSIDE the try so a real shape regression fails the test instead of being
    # swallowed into a skip by the broad `except` above.
    if not (logits.shape[0] == 1 and logits.shape[-1] == model.vocab_size):
        raise AssertionError('condition failed at line 228')


@_scoped_default_device
def run_qwen3_32b_teacher_forcing_prefill_vs_hf(mesh_device, hf_model_id, tmp_path_factory):
    """Last-token logits vs HF after prefill (PCC gate, full depth).

    Must run the same number of decoder layers as ``AutoModelForCausalLM``;
    comparing a truncated TT stack (``QWEN3_32B_DEMO_NUM_LAYERS`` for smoke tests)
    to full HF logits always fails PCC. Unset ``QWEN3_32B_DEMO_NUM_LAYERS`` here,
    or set it equal to the checkpoint's ``num_hidden_layers``.
    """
    _skip_unless_t3k(mesh_device, hf_model_id)
    hf_cfg = AutoConfig.from_pretrained(hf_model_id)
    n_hf = int(hf_cfg.num_hidden_layers)
    env_layers = os.environ.get("QWEN3_32B_DEMO_NUM_LAYERS")
    if env_layers is not None:
        num_layers = int(env_layers)
        if num_layers != n_hf:
            raise UnsupportedConfiguration(f'Teacher-forcing PCC is defined against the full HF forward; unset QWEN3_32B_DEMO_NUM_LAYERS to use all {n_hf} layers (currently QWEN3_32B_DEMO_NUM_LAYERS={num_layers}).')
    else:
        num_layers = n_hf
    seq_len = 128
    cache = tmp_path_factory.mktemp("qwen3_32b_tf")
    model = None
    try:
        model = Qwen3_32B.from_pretrained(
            mesh_device,
            hf_model_id,
            max_batch_size=1,
            max_seq_len=max(512, seq_len),
            num_layers=num_layers,
            cache_dir=cache,
            executor_mode=True,
        )
    except Exception as e:
        raise UnsupportedConfiguration(f'Could not build model: {e}')

    ma = model.model_args
    block_size = 32
    max_num_blocks = (ma.max_seq_len // block_size) * ma.max_batch_size
    kv_shape = (max_num_blocks, ma.n_kv_heads // mesh_device.get_num_devices(), block_size, ma.head_dim)

    input_ids = torch.zeros(1, seq_len, dtype=torch.long)
    input_ids[0, :4] = torch.tensor([1, 2, 3, 4], dtype=torch.long)

    ok = None
    pcc = None
    try:
        hf = AutoModelForCausalLM.from_pretrained(hf_model_id, torch_dtype=torch.bfloat16)
        hf.eval()
        with torch.no_grad():
            hf_out = hf(input_ids[:, :seq_len]).logits[0, seq_len - 1, :].float()

        ex = EagerQwen3_32BExecutor(model, mesh_device)
        kv = ex.allocate_kv_cache(kv_shape, torch.bfloat16, ma.n_layers)
        page_table = make_contiguous_page_table(1, ma.max_seq_len, block_size)
        tt_logits = ex.prefill_forward(input_ids[:, :seq_len], page_table=page_table, kv_cache=kv)
        tt_vec = tt_logits[0, 0, :].float()

        ok, pcc = comp_pcc(hf_out, tt_vec, pcc=0.85)
    except Exception as e:
        raise UnsupportedConfiguration(f'Teacher-forcing PCC check not runnable: {e}')
    finally:
        cleanup_model_case(model, mesh_device)
    # Assert OUTSIDE the try so a genuine PCC regression — the whole point of this test — fails
    # instead of being converted into a skip by the broad `except` above.
    if not (ok):
        raise AssertionError(f'Prefill last-token PCC too low: {pcc}')


@_scoped_default_device
def run_qwen3_32b_eager_traced_prefill_logits_match(mesh_device, hf_model_id, tmp_path_factory):
    """``EagerQwen3_32BExecutor`` vs ``TracedQwen3_32BExecutor``: same prefill → host logits within tolerance.

    Builds two 32B models back-to-back — host RAM (~130 GB at bf16 for both HF copies) and
    device cache will be tight. Use ``QWEN3_32B_DEMO_NUM_LAYERS=1`` for fast iteration.
    """
    _skip_unless_t3k(mesh_device, hf_model_id)
    num_layers = int(os.environ.get("QWEN3_32B_DEMO_NUM_LAYERS", "1"))
    cache_a = tmp_path_factory.mktemp("qwen3_32b_par_e")
    cache_b = tmp_path_factory.mktemp("qwen3_32b_par_t")
    seq_len = 128
    m_e = m_t = None
    try:
        m_e = Qwen3_32B.from_pretrained(
            mesh_device,
            hf_model_id,
            max_batch_size=1,
            max_seq_len=512,
            num_layers=num_layers,
            cache_dir=cache_a,
            executor_mode=True,
        )
        m_t = Qwen3_32B.from_pretrained(
            mesh_device,
            hf_model_id,
            max_batch_size=1,
            max_seq_len=512,
            num_layers=num_layers,
            cache_dir=cache_b,
            executor_mode=True,
        )
    except Exception as e:
        raise UnsupportedConfiguration(f'Could not build models: {e}')

    ma = m_e.model_args
    if not (ma is not None):
        raise AssertionError('condition failed at line 340')
    block_size = 32
    max_num_blocks = (ma.max_seq_len // block_size) * ma.max_batch_size
    kv_shape = (max_num_blocks, ma.n_kv_heads // mesh_device.get_num_devices(), block_size, ma.head_dim)

    try:
        eager_ex = EagerQwen3_32BExecutor(m_e, mesh_device)
        traced_ex = TracedQwen3_32BExecutor(m_t, mesh_device)
        kv_e = eager_ex.allocate_kv_cache(kv_shape, torch.bfloat16, ma.n_layers)
        kv_t = traced_ex.allocate_kv_cache(kv_shape, torch.bfloat16, ma.n_layers)
        page_table = make_contiguous_page_table(1, ma.max_seq_len, block_size)
        toks = torch.zeros(1, seq_len, dtype=torch.long)
        toks[0, :4] = torch.tensor([1, 2, 3, 4], dtype=torch.long)
        le = eager_ex.prefill_forward(toks, page_table=page_table, kv_cache=kv_e)
        lt = traced_ex.prefill_forward(toks, page_table=page_table, kv_cache=kv_t)
        diff = (le.float() - lt.float()).abs().max().item()
        if not (diff < 0.25):
            raise AssertionError(f'eager/traced prefill logits max abs diff too large: {diff}')
    finally:
        cleanup_model_case(m_e, mesh_device)
        cleanup_model_case(m_t, mesh_device)


@_scoped_default_device
def run_qwen3_32b_numerical_divergence_vs_hf(mesh_device, hf_model_id, tmp_path_factory):
    """Per-boundary PCC of TT prefill activations vs HF; isolate the first diverging op.

    Captures activations on both sides for the same 128-token prompt:
      - embed output
      - per layer i (0..N-1): post-input-norm, attention-out, attn-residual, post-attn-norm,
        mlp-out, layer-out
      - final-norm output

    Use the legacy internal-KV path (``prefill_from_token_ids``) — same residual stream
    semantics as the executor path, simpler to instrument. Reports a CSV under
    ``QWEN3_32B_NUMDIV_OUT`` (default ``/tmp/qwen3_32b_numdiv.csv``).

    Override env:
      QWEN3_32B_NUMDIV_LAYERS=N   to limit layer count (debug).
      QWEN3_32B_NUMDIV_SEQ=S      to change seq_len (must be multiple of 128).
    """
    import csv
    import gc

    from tt_transformers.models.qwen3_32b.model import Qwen3_32BDecoderLayer, _all_gather_rmsnorm_tensor

    _skip_unless_t3k(mesh_device, hf_model_id)
    hf_cfg = AutoConfig.from_pretrained(hf_model_id)
    n_hf = int(hf_cfg.num_hidden_layers)
    num_layers = int(os.environ.get("QWEN3_32B_NUMDIV_LAYERS", n_hf))
    if num_layers > n_hf:
        num_layers = n_hf
    seq_len = int(os.environ.get("QWEN3_32B_NUMDIV_SEQ", "128"))
    if not (seq_len % 128 == 0):
        raise AssertionError(f'seq_len must be multiple of 128, got {seq_len}')

    out_csv = os.environ.get("QWEN3_32B_NUMDIV_OUT", "/tmp/qwen3_32b_numdiv.csv")
    cache = tmp_path_factory.mktemp("qwen3_32b_numdiv")

    input_ids = torch.zeros(1, seq_len, dtype=torch.long)
    input_ids[0, :4] = torch.tensor([1, 2, 3, 4], dtype=torch.long)

    # ---------- HF forward + per-layer hooks ----------
    hf_intermediates: dict[str, torch.Tensor] = {}
    hf = AutoModelForCausalLM.from_pretrained(hf_model_id, torch_dtype=torch.bfloat16)
    hf.eval()

    def _hook(name):
        def fn(_mod, _inp, out):
            t = out[0] if isinstance(out, tuple) else out
            hf_intermediates[name] = t.detach().float().cpu()

        return fn

    handles = [hf.model.embed_tokens.register_forward_hook(_hook("embed"))]
    for i, layer in enumerate(hf.model.layers[:num_layers]):
        handles.append(layer.input_layernorm.register_forward_hook(_hook(f"L{i:02d}.input_norm")))
        handles.append(layer.self_attn.register_forward_hook(_hook(f"L{i:02d}.self_attn")))
        handles.append(layer.post_attention_layernorm.register_forward_hook(_hook(f"L{i:02d}.post_attn_norm")))
        handles.append(layer.mlp.register_forward_hook(_hook(f"L{i:02d}.mlp")))
        handles.append(layer.register_forward_hook(_hook(f"L{i:02d}.layer_out")))
    handles.append(hf.model.norm.register_forward_hook(_hook("final_norm")))

    with torch.no_grad():
        if num_layers == n_hf:
            hf(input_ids)
        else:
            x = hf.model.embed_tokens(input_ids)
            position_ids = torch.arange(seq_len).unsqueeze(0)
            cos, sin = hf.model.rotary_emb(x, position_ids)
            attn_mask = torch.zeros(1, 1, seq_len, seq_len, dtype=x.dtype)
            attn_mask.masked_fill_(
                torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1), float("-inf")
            )
            h = x
            for layer in hf.model.layers[:num_layers]:
                h = layer(
                    h,
                    attention_mask=attn_mask,
                    position_ids=position_ids,
                    position_embeddings=(cos, sin),
                )[0]
            h = hf.model.norm(h)

    for h_ in handles:
        h_.remove()
    del hf
    gc.collect()

    # ---------- Build TT model ----------
    model = None
    try:
        model = Qwen3_32B.from_pretrained(
            mesh_device,
            hf_model_id,
            max_batch_size=1,
            max_seq_len=max(512, seq_len),
            num_layers=num_layers,
            cache_dir=cache,
            executor_mode=False,
        )
    except Exception as e:
        raise UnsupportedConfiguration(f'Could not build TT model: {e}')

    # ---------- Instrument TT decoder layer prefill_forward ----------
    tt_intermediates: dict[str, torch.Tensor] = {}
    orig_layer_prefill = Qwen3_32BDecoderLayer.prefill_forward
    layer_idx = [0]

    def _stash(name: str, t: ttnn.Tensor) -> None:
        # Topology metadata on intermediate tensors is unreliable here (post-RS / post-AG paths can
        # leave Shard(2) tags on tensors whose data is actually sharded on dim 3, etc.). Read each
        # device shard directly: if all shards are bit-identical, the tensor is replicated → take one;
        # otherwise concatenate along the last dim (the only fracturing axis these modules use).
        shards = ttnn.get_device_tensors(t)
        torch_shards = [ttnn.to_torch(s).detach().float().cpu() for s in shards]
        if len(torch_shards) == 1:
            host = torch_shards[0]
        elif all(torch.equal(torch_shards[0], s) for s in torch_shards[1:]):
            host = torch_shards[0]
        else:
            host = torch.cat(torch_shards, dim=-1)
        tt_intermediates[name] = host

    def patched_layer_prefill(
        self,
        x,
        rot_mats,
        *,
        user_id: int = 0,
        page_table=None,
        chunk_page_table=None,
        chunk_start_idx=None,
    ):
        i = layer_idx[0]
        r = self.input_layernorm.prefill_forward(x)
        r = _all_gather_rmsnorm_tensor(self.input_layernorm, r)
        _stash(f"L{i:02d}.input_norm", r)
        r = self.self_attn.forward(
            r,
            None,
            rot_mats,
            mode="prefill",
            user_id=user_id,
            page_table=page_table,
            chunk_page_table=chunk_page_table,
            chunk_start_idx=chunk_start_idx,
        )
        _stash(f"L{i:02d}.self_attn", r)
        h = ttnn.add(x, r, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        r2 = self.post_attention_layernorm.prefill_forward(h)
        r2 = _all_gather_rmsnorm_tensor(self.post_attention_layernorm, r2)
        _stash(f"L{i:02d}.post_attn_norm", r2)
        r2 = self.mlp.prefill_forward(r2)
        _stash(f"L{i:02d}.mlp", r2)
        out = ttnn.add(h, r2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        _stash(f"L{i:02d}.layer_out", out)
        layer_idx[0] += 1
        return out

    Qwen3_32BDecoderLayer.prefill_forward = patched_layer_prefill

    try:
        tokens_tt = ttnn.from_torch(
            input_ids.view(1, 1, 1, seq_len).to(torch.int32),
            device=mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(mesh_device),
        )
        x_embed = model.embed_prefill(tokens_tt)
        _stash("embed", x_embed)
        rot = model.rope_setup.prefill_forward(0, seq_len)
        h_tt = x_embed
        for layer in model.layers:
            h_tt = layer.prefill_forward(h_tt, rot, user_id=0, page_table=None)
        h_tt = model.norm.prefill_forward(h_tt)
        h_tt = _all_gather_rmsnorm_tensor(model.norm, h_tt)
        _stash("final_norm", h_tt)
    finally:
        Qwen3_32BDecoderLayer.prefill_forward = orig_layer_prefill

    # ---------- Compare ----------
    rows = []
    keys = sorted(set(hf_intermediates) & set(tt_intermediates))
    ordering = {"embed": 0, "final_norm": 999}
    boundary_order = ["input_norm", "self_attn", "post_attn_norm", "mlp", "layer_out"]

    def _sort_key(k: str) -> tuple:
        if k in ordering:
            return (ordering[k], 0, "")
        parts = k.split(".")
        if len(parts) == 2 and parts[0].startswith("L"):
            try:
                li = int(parts[0][1:])
                bi = boundary_order.index(parts[1]) if parts[1] in boundary_order else 99
                return (10 + li, bi, parts[1])
            except ValueError:
                pass
        return (1000, 0, k)

    keys.sort(key=_sort_key)

    print("\n=== qwen3_32b numerical divergence vs HF ===")
    print(f"  seq_len={seq_len}  num_layers={num_layers}  mesh={mesh_device.get_num_devices()}")
    print(f"  {'boundary':35s}  {'PCC':>10s}  {'max_abs_diff':>14s}  {'shape':>20s}")
    for k in keys:
        hf_t = hf_intermediates[k]
        tt_t = tt_intermediates[k]
        raw_tt_shape = tuple(tt_t.shape)
        while tt_t.dim() > hf_t.dim():
            tt_t = tt_t.squeeze(0)
        if tt_t.shape[-1] == 2 * hf_t.shape[-1]:
            tt_t = tt_t[..., : hf_t.shape[-1]]
        if tt_t.shape[-2] > hf_t.shape[-2]:
            tt_t = tt_t[..., : hf_t.shape[-2], :]
        if hf_t.shape[-2] > tt_t.shape[-2]:
            hf_t = hf_t[..., : tt_t.shape[-2], :]
        if tt_t.shape != hf_t.shape:
            print(
                f"  {k:35s}  {'SHAPE_MISMATCH':>10s}  raw_tt={raw_tt_shape}  tt={tuple(tt_t.shape)}  hf={tuple(hf_t.shape)}"
            )
            rows.append((k, float("nan"), float("nan"), tuple(tt_t.shape)))
            continue
        _, pcc = comp_pcc(hf_t, tt_t, pcc=0.99)
        max_abs = (hf_t.float() - tt_t.float()).abs().max().item()
        rows.append((k, float(pcc), float(max_abs), tuple(tt_t.shape)))
        print(f"  {k:35s}  {pcc:>10.6f}  {max_abs:>14.4e}  {str(tuple(tt_t.shape)):>20s}")

    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["boundary", "pcc", "max_abs_diff", "shape"])
        for row in rows:
            w.writerow([row[0], f"{row[1]:.6f}", f"{row[2]:.6e}", str(row[3])])
    print(f"  -> wrote {out_csv}")

    if model is not None:
        cleanup_model_case(model, mesh_device)


SMOKE_CASE_RUNNERS = {
    'qwen3-32b-prefill-smoke': run_qwen3_32b_prefill_smoke,
    'qwen3-32b-decode-one-step': run_qwen3_32b_decode_one_step,
    'qwen3-32b-executor-prefill-smoke': run_qwen3_32b_executor_prefill_smoke,
    'qwen3-32b-teacher-forcing-prefill-vs-hf': run_qwen3_32b_teacher_forcing_prefill_vs_hf,
    'qwen3-32b-eager-traced-prefill-logits-match': run_qwen3_32b_eager_traced_prefill_logits_match,
    'qwen3-32b-numerical-divergence-vs-hf': run_qwen3_32b_numerical_divergence_vs_hf,
}


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=tuple(SMOKE_CASE_RUNNERS), default=next(iter(SMOKE_CASE_RUNNERS)))
    parser.add_argument("--seq-len", type=int, default=128)
    args = parser.parse_args(argv)
    params = {"mesh_shape": (1, 8), "fabric_config": ttnn.FabricConfig.FABRIC_1D, "num_command_queues": 1}
    factory = TemporaryPathFactory()
    try:
        with open_mesh_device(params) as mesh:
            runner = SMOKE_CASE_RUNNERS[args.case]
            available = {"mesh_device": mesh, "hf_model_id": default_hf_model_id(), "seq_len": args.seq_len, "tmp_path_factory": factory}
            names = tuple(inspect.signature(runner).parameters)
            runner(**{name: available[name] for name in names})
    except UnsupportedConfiguration as error:
        parser.error(str(error))
    finally:
        factory.cleanup()


if __name__ == "__main__":
    main()
