# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""HF generation contracts exercised through actual runtime collaborators."""

from importlib import import_module
from inspect import get_annotations
from types import SimpleNamespace

import pytest
import torch
import ttnn

from tt_transformers.llm_runtime.config import PagedKVCacheConfig, TraceConfig, WarmupConfig
from tt_transformers.models.executor import ModelExecutor, ModelExecutorConfig
from tt_transformers.models.hf_generation import cleanup, generate

pytestmark = pytest.mark.host

_MODEL_PACKAGES = (
    "deepseek_r1_distill_qwen_14b",
    "llama32_1b",
    "llama32_3b",
    "llama33_70b",
    "llama3_8b",
    "mistral_7b",
    "phi4",
    "qwen25_72b",
    "qwen25_7b",
    "qwen25_coder_32b",
    "qwen2_7b",
    "qwen3_32b",
)


def _model_and_executor(package=None):
    mesh = SimpleNamespace(shape=(1, 1), get_num_devices=lambda: 1)
    paged = SimpleNamespace(block_size=32, max_num_blocks=256)
    attention = SimpleNamespace(
        n_kv_heads=8,
        head_dim=64,
        kv_cache_dtype=ttnn.bfloat8_b,
        paged_attention_config=paged,
        use_vllm_paged_kv_cache=True,
        kv_cache=None,
    )
    model = SimpleNamespace(
        config=SimpleNamespace(
            mesh_device=mesh,
            max_batch_size=4,
            max_seq_len=2048,
            vocab_size=16,
            n_layers=1,
            num_devices=1,
            block_configs=(SimpleNamespace(attention_config=attention),),
        ),
        layers=(SimpleNamespace(attention=SimpleNamespace(config=attention, kv_cache=None)),),
        iter_executor_named_modules=lambda: (),
        vocab_size=16,
        num_devices=1,
    )
    runtime = SimpleNamespace(
        model_cache_path=None,
        max_prefill_chunk_size=1024,
        trace_prefill_supported_seq_lens=(128, 1024),
        can_enable_trace=lambda length, cached=0: length in (128, 1024),
        supports_batched_prefill=True,
        disable_batched_prefill=False,
        max_prefill_batch_size=32,
        batched_prefill_batched_extract=True,
        max_context_len=2048,
        cluster_shape=[1, 1],
    )
    if package is None:
        config_type = ModelExecutorConfig

        def builder(llm, config):
            return ModelExecutor(llm.model, llm.runtime_config, config)
    else:
        module = import_module(f"tt_transformers.models.{package}.hf_generator")
        builder = next(
            value for name, value in vars(module).items() if name.startswith("build_") and name.endswith("_executor")
        )
        config_type = get_annotations(builder, eval_str=True)["config"]
    config = config_type(
        trace=TraceConfig("none"),
        warmup=WarmupConfig(prefill_seq_lens=(128, 1024)),
        paged_kv_cache=PagedKVCacheConfig(block_size=32, max_num_blocks=256, num_blocks=256, dtype=ttnn.bfloat8_b),
        device_sampling_enabled=False,
    )
    executor = builder(SimpleNamespace(model=model, runtime_config=runtime), config)
    return SimpleNamespace(
        executor=executor,
        model=model,
        runtime_config=runtime,
        tokenizer=SimpleNamespace(pad_token_id=0, eos_token_id=None),
        generation_config=SimpleNamespace(temperature=0, top_k=0, top_p=1),
    )


@pytest.mark.parametrize("package", _MODEL_PACKAGES)
@pytest.mark.host
def test_generation_uses_real_runtime_layouts_and_inactive_positions(monkeypatch, package):
    llm = _model_and_executor(package)
    executor = llm.executor
    executor.kv_cache_manager._bound_context = object()
    monkeypatch.setattr(ttnn, "synchronize_device", lambda mesh: None)
    logits = torch.arange(16, dtype=torch.float).reshape(1, 1, 16).repeat(4, 1, 1)
    prefill_logits = logits.clone()
    prefill_logits[0, 0, 2] = 20  # First row finishes immediately; second keeps decoding.
    # Keep actual model-executor dispatch, request preparation, page normalization,
    # signatures, and program registry. Replace only device invocation/readback.
    monkeypatch.setattr(
        executor.prefill_runtime, "invoke", lambda prepared, **kwargs: SimpleNamespace(value=logits, owned=None)
    )
    monkeypatch.setattr(
        executor.prefill_runtime,
        "assemble",
        lambda results, *, batch_size, **kwargs: prefill_logits[:batch_size],
    )
    monkeypatch.setattr(
        executor.decode_runtime, "invoke", lambda prepared, **kwargs: SimpleNamespace(value=logits, owned=None)
    )
    monkeypatch.setattr(executor.decode_runtime, "consume", lambda result, **kwargs: result.value)
    prepared_decode = []
    prepare = executor.decode_runtime.prepare

    def record_prepare(*args, **kwargs):
        prepared = prepare(*args, **kwargs)
        prepared_decode.append(prepared)
        return prepared

    monkeypatch.setattr(executor.decode_runtime, "prepare", record_prepare)
    prompts = torch.tensor([[0, 0, 3, 4], [2, 3, 4, 5]])
    output = generate(llm, prompts, max_new_tokens=3, eos_token_id=2)

    assert output.tolist() == [[0, 0, 3, 4, 2, 0, 0], [2, 3, 4, 5, 15, 15, 15]]
    assert prepared_decode[-1].start_pos.tolist() == [-1, 5, -1, -1]
    assert prepared_decode[-1].page_table.shape == (4, executor.page_table_layout.decode_width)
    assert prepared_decode[-1].page_table[0].count_nonzero() == 0
    assert prepared_decode[-1].page_table[1, 0].item() == 1

    # A new request can change block geometry while keeping this executor/cache.
    output = generate(llm, torch.ones((1, 65), dtype=torch.long), max_new_tokens=2, eos_token_id=None)
    assert output.shape == (1, 67)
    assert prepared_decode[-1].start_pos.tolist() == [65, -1, -1, -1]
    assert prepared_decode[-1].page_table[0, :3].tolist() == [0, 1, 2]
    assert not prepared_decode[-1].page_table[1:].any()


@pytest.mark.host
def test_failed_executor_cache_detach_defers_model_tensor_cleanup(monkeypatch):
    llm = _model_and_executor()
    manager = llm.executor.kv_cache_manager
    released = []

    class DeviceTensor:
        pass

    key, value = DeviceTensor(), DeviceTensor()
    cache = [[key, value]]
    llm.model.layers[0].attention.kv_cache = cache[0]
    llm.model.layers[0].attention.config.kv_cache = cache[0]
    manager._state = "bound"
    manager._bound_cache = cache
    manager._bound_context = object()
    manager._owned_tensors = (key, value)

    def fail_detach(cache):
        raise RuntimeError("model cache detach failed")

    llm.model.set_kv_cache = fail_detach
    monkeypatch.setattr(ttnn, "Tensor", DeviceTensor)
    monkeypatch.setattr(ttnn, "deallocate", lambda tensor: released.append(tensor))
    with pytest.raises(RuntimeError, match="model cache detach failed"):
        cleanup(llm)

    assert released == [], "the KV owner must retain borrowed tensors when model detachment fails"
    assert manager.bound_context is not None
    assert llm.model.layers[0].attention.kv_cache == cache[0]

    def detach(cache):
        assert cache is None
        llm.model.layers[0].attention.kv_cache = None
        llm.model.layers[0].attention.config.kv_cache = None

    llm.model.set_kv_cache = detach
    cleanup(llm)
    assert released == [key, value]
    assert manager.bound_context is None
    assert llm._hf_cleaned_up
