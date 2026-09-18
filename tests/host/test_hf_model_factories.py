"""Host contracts for loading each text product with exactly one executor."""

from importlib import import_module
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import ttnn
from transformers import PretrainedConfig

from tt_transformers.llm_runtime.config import PagedKVCacheConfig, TraceConfig, WarmupConfig

MODEL_PRODUCTS = {
    "deepseek_r1_distill_qwen_14b": "DeepSeekR1Qwen14B",
    "llama32_1b": "Llama32_1B",
    "llama32_3b": "Llama32_3B",
    "llama33_70b": "Llama33_70B",
    "llama3_8b": "Llama3",
    "mistral_7b": "Mistral7B",
    "phi4": "Phi4",
    "qwen25_72b": "Qwen25_72B",
    "qwen25_7b": "Qwen25",
    "qwen25_coder_32b": "Qwen25Coder32B",
    "qwen2_7b": "Qwen2",
    "qwen3_32b": "Qwen3_32B",
}

pytestmark = [pytest.mark.host, pytest.mark.model]


@pytest.fixture(params=MODEL_PRODUCTS)
def factory(request, monkeypatch):
    family = request.param
    prefix = MODEL_PRODUCTS[family]
    module = import_module(f"tt_transformers.models.{family}.hf_generator")
    model = SimpleNamespace(
        config=SimpleNamespace(max_seq_len=65, max_batch_size=2),
        layers=[SimpleNamespace(attention=SimpleNamespace(config=SimpleNamespace(kv_cache_dtype=ttnn.bfloat8_b)))],
    )
    product = getattr(module, f"{prefix}ForCausalLM")(
        model=model,
        tokenizer=SimpleNamespace(stop_tokens=[2]),
        runtime_config=SimpleNamespace(model_name="local/model", model_cache_path=None, max_context_len=65),
        instruct=True,
    )
    assert product.executor is None
    executor = SimpleNamespace(cleanup=MagicMock())
    loader = MagicMock(return_value=product)
    builder = MagicMock(return_value=executor)
    checkpoint_defaults = SimpleNamespace(max_new_tokens=7, do_sample=False)
    metadata_loader = MagicMock(return_value=checkpoint_defaults)
    monkeypatch.setattr(module, "_load_model", loader)
    builder_family = "llama3" if family == "llama3_8b" else family
    monkeypatch.setattr(module, f"build_{builder_family}_executor", builder)
    monkeypatch.setattr(module.GenerationConfig, "from_pretrained", metadata_loader)
    return SimpleNamespace(
        module=module,
        revision_kwargs={} if family == "qwen25_72b" else {"hf_revision": "exact-revision"},
        expected_revision=module.DEFAULT_HF_REVISION if family == "qwen25_72b" else "exact-revision",
        product=product,
        executor=executor,
        loader=loader,
        builder=builder,
        metadata_loader=metadata_loader,
        checkpoint_defaults=checkpoint_defaults,
        config_class=getattr(module, f"{prefix}ExecutorConfig"),
    )


def _load(factory, **kwargs):
    return factory.module.from_pretrained(
        object(),
        hf_model="local/model",
        **factory.revision_kwargs,
        max_batch_size=2,
        max_seq_len=65,
        **kwargs,
    )


@pytest.mark.host
def test_public_factory_attaches_one_executor_with_full_capacity_and_checkpoint_defaults(factory):
    result = _load(factory)

    assert result is factory.product
    assert result.executor is factory.executor
    factory.loader.assert_called_once()
    factory.builder.assert_called_once()
    loaded_product, config = factory.builder.call_args.args
    assert loaded_product is result
    assert config.trace.mode == "none"
    assert not config.device_sampling_enabled
    assert config.paged_kv_cache.block_size == 32
    assert config.paged_kv_cache.num_blocks == 6  # Two rows, each requiring three pages.
    assert config.paged_kv_cache.dtype is ttnn.bfloat8_b
    assert factory.loader.call_args.kwargs.get("hf_revision") == factory.revision_kwargs.get("hf_revision")
    assert factory.metadata_loader.call_args.args == ("local/model",)
    assert factory.metadata_loader.call_args.kwargs["revision"] == factory.expected_revision
    assert result._hf_generation_config is factory.checkpoint_defaults


@pytest.mark.parametrize("physical_blocks", [None, 5])
@pytest.mark.host
def test_explicit_executor_policy_is_resolved_without_mutating_the_callers_config(factory, physical_blocks):
    supplied = factory.config_class(
        trace=TraceConfig(mode="decode_only"),
        warmup=WarmupConfig(),
        paged_kv_cache=PagedKVCacheConfig(
            block_size=64,
            max_num_blocks=8,
            num_blocks=physical_blocks,
            dtype=ttnn.bfloat16,
        ),
        device_sampling_enabled=True,
    )

    result = _load(factory, executor_config=supplied)

    assert result.executor is factory.executor
    resolved = factory.builder.call_args.args[1]
    assert resolved.trace is supplied.trace
    assert resolved.device_sampling_enabled
    assert resolved.paged_kv_cache.num_blocks == (8 if physical_blocks is None else physical_blocks)
    assert supplied.paged_kv_cache.num_blocks == physical_blocks
    assert resolved.paged_kv_cache.dtype is ttnn.bfloat16
    page_geometry = factory.loader.call_args.kwargs["paged_attention_config"]
    assert (page_geometry.block_size, page_geometry.max_num_blocks) == (64, 8)


@pytest.mark.host
def test_executor_construction_failure_cleans_up_the_loaded_product(factory, monkeypatch):
    failure = RuntimeError("executor construction failed")
    factory.builder.side_effect = failure
    cleanup = MagicMock()
    monkeypatch.setattr(factory.product, "cleanup", cleanup)

    with pytest.raises(RuntimeError, match="executor construction failed") as caught:
        _load(factory)

    assert caught.value is failure
    cleanup.assert_called_once_with()
    factory.metadata_loader.assert_not_called()


@pytest.mark.host
def test_metadata_failure_cleans_up_after_executor_attachment_and_preserves_primary_error(factory, monkeypatch):
    failure = ValueError("invalid checkpoint generation metadata")
    cleanup_failure = RuntimeError("cleanup failed")
    factory.metadata_loader.side_effect = failure
    cleanup = MagicMock(side_effect=cleanup_failure)
    monkeypatch.setattr(factory.product, "cleanup", cleanup)

    with pytest.raises(ValueError, match="invalid checkpoint generation metadata") as caught:
        _load(factory)

    assert caught.value is failure
    assert factory.product.executor is factory.executor
    cleanup.assert_called_once_with()
    assert cleanup_failure in caught.value.cleanup_failures


@pytest.mark.host
def test_checkpoint_without_generation_metadata_keeps_a_usable_product(factory):
    factory.metadata_loader.side_effect = OSError("generation_config.json is absent")
    factory.product._hf_model_config = PretrainedConfig(eos_token_id=[2, 3], bos_token_id=1)

    result = _load(factory)

    assert result.executor is factory.executor
    assert result._hf_generation_config.eos_token_id == [2, 3]
    assert result._hf_generation_config.bos_token_id == 1


@pytest.mark.host
def test_mismatched_executor_page_geometry_is_rejected_before_loading_weights(factory):
    config = factory.config_class(
        trace=TraceConfig(mode="none"),
        warmup=WarmupConfig(),
        paged_kv_cache=PagedKVCacheConfig(block_size=64, max_num_blocks=8, dtype=ttnn.bfloat8_b),
        device_sampling_enabled=False,
    )

    with pytest.raises(ValueError, match="geometry"):
        _load(
            factory,
            executor_config=config,
            paged_attention_config=SimpleNamespace(block_size=32, max_num_blocks=8),
        )

    factory.loader.assert_not_called()
    factory.builder.assert_not_called()
