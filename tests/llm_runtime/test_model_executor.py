# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Focused contracts for the family-neutral model composition root."""

import ast
import dataclasses
import importlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import ttnn

from tt_transformers.llm_runtime.config import PagedKVCacheConfig, TraceConfig, WarmupConfig
from tt_transformers.llm_runtime.warmup import resolve_prefill_warmup_seq_lens
from tt_transformers.models import executor as executor_module
from tt_transformers.models.executor import ModelExecutor, ModelExecutorConfig

_REPOSITORY_ROOT = Path(__file__).parents[2]
_EXECUTOR_PATH = _REPOSITORY_ROOT / "src/tt_transformers/models/executor.py"
_MODELS_ROOT = _EXECUTOR_PATH.parent


def _config(*, device_sampling_enabled: bool = False) -> ModelExecutorConfig:
    return ModelExecutorConfig(
        trace=TraceConfig(mode="none"),
        warmup=WarmupConfig(),
        paged_kv_cache=PagedKVCacheConfig(
            block_size=32,
            max_num_blocks=128,
            num_blocks=128,
            dtype=ttnn.bfloat8_b,
        ),
        device_sampling_enabled=device_sampling_enabled,
    )


@pytest.mark.host
def test_common_executor_has_no_concrete_model_dependencies_or_dispatch() -> None:
    tree = ast.parse(_EXECUTOR_PATH.read_text())
    imports = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module is not None}
    imported_names = {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    assert not any(module.startswith("tt_transformers.models.") for module in imports)
    assert not any(name.startswith("tt_transformers.models.") for name in imported_names)

    control_flow = [
        node.test for node in ast.walk(tree) if isinstance(node, (ast.If, ast.IfExp, ast.While, ast.Assert))
    ]
    dispatch_names = {"model_name", "model_id", "provider_id", "checkpoint_path", "model_version"}
    assert not any(
        isinstance(node, ast.Name) and node.id in dispatch_names
        for expression in control_flow
        for node in ast.walk(expression)
    )

    torch_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "torch"
    ]
    assert torch_calls == []


@pytest.mark.host
def test_model_layer_has_only_the_approved_family_modules_and_readmes() -> None:
    assert (_MODELS_ROOT / "llama3_executor.py").is_file()
    assert (_MODELS_ROOT / "qwen2_executor.py").is_file()
    assert not (_MODELS_ROOT / "qwen3_executor.py").exists()

    model_directories = sorted(
        path for path in _MODELS_ROOT.iterdir() if path.is_dir() and (path / "model.py").is_file()
    )
    assert len(model_directories) == 12
    for path in model_directories:
        assert (path / "hf_generator.py").is_file()
        assert (path / "vllm_generator.py").is_file()
        assert not any((path / name).exists() for name in ("hf_adaptor.py", "executor.py", "generator.py"))
    assert all((_REPOSITORY_ROOT / "examples" / path.name / "README.md").is_file() for path in model_directories)


@pytest.mark.host
@pytest.mark.parametrize(
    "relative_path",
    (
        "deepseek_r1_distill_qwen_14b/hf_generator.py",
        "mistral_7b/hf_generator.py",
        "phi4/hf_generator.py",
    ),
)
def test_direct_composition_examples_do_not_depend_on_shared_family_executors(relative_path: str) -> None:
    tree = ast.parse((_MODELS_ROOT / relative_path).read_text())
    imports = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    assert "tt_transformers.models.executor" not in imports
    assert "tt_transformers.models.llama3_executor" not in imports
    assert "tt_transformers.models.qwen2_executor" not in imports


@pytest.mark.host
def test_common_config_is_frozen_and_rejects_non_exact_nested_config_types(expect_error) -> None:
    config = _config()
    with expect_error(AttributeError, "cannot assign to field"):
        config.device_sampling_enabled = True

    class TraceConfigSubclass(TraceConfig):
        pass

    with expect_error(TypeError, "trace must be exactly TraceConfig"):
        ModelExecutorConfig(
            trace=TraceConfigSubclass(mode="none"),
            warmup=config.warmup,
            paged_kv_cache=config.paged_kv_cache,
            device_sampling_enabled=False,
        )


@pytest.mark.host
def test_sampling_state_inputs_are_an_optional_owned_pair(expect_error) -> None:
    with expect_error(ValueError, "must be supplied together"):
        ModelExecutor(
            None,
            None,
            _config(device_sampling_enabled=True),
            sampling_state_controller=object(),
        )
    with expect_error(ValueError, "requires device sampling"):
        ModelExecutor(
            None,
            None,
            _config(),
            sampling_state_controller=object(),
            sampling_state=object(),
        )


@pytest.mark.host
@pytest.mark.parametrize(
    ("enable_trace", "expected"),
    [(True, ["prime", "coordinator"]), (False, ["coordinator", "prime"])],
)
def test_prefill_warmup_policy_controls_order_through_one_continuation(enable_trace, expected) -> None:
    events = []

    def policy(executor, default_warmup, *, kv_cache, can_sample_on_device, enable_trace):
        assert executor is target
        assert kv_cache is cache
        assert can_sample_on_device
        if enable_trace:
            events.append("prime")
        default_warmup()
        if not enable_trace:
            events.append("prime")

    cache = object()
    target = object.__new__(ModelExecutor)
    target._terminal = False
    target.prefill_runtime = SimpleNamespace(transient_orphan_count=0)
    target.decode_runtime = SimpleNamespace(transient_orphan_count=0)
    target._prefill_warmup = policy
    target.warmup = SimpleNamespace(
        warmup_prefill=lambda **kwargs: events.append("coordinator"),
    )

    target.warmup_model_prefill(
        kv_cache=cache,
        can_sample_on_device=True,
        enable_trace=enable_trace,
    )

    assert events == expected


@pytest.mark.host
def test_request_state_and_execution_target_are_forwarded_by_identity() -> None:
    target = object.__new__(ModelExecutor)
    target._terminal = False
    target.prefill_runtime = SimpleNamespace(transient_orphan_count=0)
    target.decode_runtime = SimpleNamespace(transient_orphan_count=0)
    target._validate_bound_cache = MagicMock()
    target._ensure_sampling_for = MagicMock()
    target._prefill_execution = MagicMock()
    target._request_state_fields = ("prompt_tokens", "output_tokens", "slot_remap")

    values = {name: object() for name in ("tokens", "page_table", "prompt_tokens", "output_tokens", "slot_remap")}
    target.compile_prefill(
        tokens=values["tokens"],
        page_table=values["page_table"],
        prompt_tokens=values["prompt_tokens"],
        output_tokens=values["output_tokens"],
        slot_remap=values["slot_remap"],
    )

    forwarded = target._prefill_execution.compile_prefill.call_args.kwargs
    for name, value in values.items():
        assert forwarded[name] is value


def _split_prefill_target(*, traceable: bool) -> ModelExecutor:
    target = object.__new__(ModelExecutor)
    target._terminal = False
    target.prefill_runtime = SimpleNamespace(transient_orphan_count=0, can_trace=MagicMock(return_value=traceable))
    target.decode_runtime = SimpleNamespace(transient_orphan_count=0)
    target._validate_bound_cache = MagicMock()
    target._ensure_sampling_for = MagicMock()
    target._request_state_fields = ()
    target.config = SimpleNamespace(trace=TraceConfig(mode="all"))
    target.traced_executor = MagicMock(name="traced")
    target.traced_prefill_execution = target.traced_executor
    target.eager_executor = MagicMock(name="eager")
    target._prefill_execution = target.traced_prefill_execution
    return target


@pytest.mark.host
@pytest.mark.parametrize("traceable", [True, False])
@pytest.mark.parametrize("method", ["compile_prefill", "prefill_forward"])
def test_prefill_without_an_execution_selects_by_trace_eligibility(method, traceable) -> None:
    # Serving compiles and runs a prefill with no explicit target. An uncovered bucket
    # must reach the eager executor rather than the traced compiler, and an eligible one
    # must still take the trace.
    target = _split_prefill_target(traceable=traceable)

    getattr(target, method)(tokens=object(), page_table=object())

    selected = target.traced_prefill_execution if traceable else target.eager_executor
    other = target.eager_executor if traceable else target.traced_prefill_execution
    getattr(selected, method).assert_called_once()
    getattr(other, method).assert_not_called()


@pytest.mark.host
@pytest.mark.parametrize("method", ["compile_prefill", "prefill_forward"])
def test_prefill_explicit_execution_wins_over_selection(method) -> None:
    target = _split_prefill_target(traceable=False)
    explicit = MagicMock(name="explicit")

    getattr(target, method)(tokens=object(), page_table=object(), execution=explicit)

    getattr(explicit, method).assert_called_once()
    target.prefill_runtime.can_trace.assert_not_called()


def _runtime(**overrides):
    fields = dict(max_prefill_chunk_size=4096, max_seq_len=8192, trace_prefill_supported_seq_lens=(128, 1024))
    return SimpleNamespace(**{**fields, **overrides})


@pytest.mark.host
def test_prefill_warmup_lengths_cover_the_bucket_ladder_whenever_trace_is_configured() -> None:
    resolve = resolve_prefill_warmup_seq_lens
    # Trace activation forbids a later compile under both trace modes, so the full ladder
    # up to the chunk cap warms; the traced buckets are a subset of it.
    for mode in ("all", "decode_only"):
        assert resolve(_runtime(), TraceConfig(mode=mode)) == (128, 1024, 2048, 4096)
    # Without trace a later compile is allowed, so the traced buckets are enough.
    assert resolve(_runtime(), TraceConfig(mode="none")) == (128, 1024)
    assert resolve(_runtime(trace_prefill_supported_seq_lens=()), TraceConfig(mode="none")) == (128,)
    # An empty traced set still warms every bucket eagerly.
    assert resolve(_runtime(trace_prefill_supported_seq_lens=()), TraceConfig(mode="all")) == (
        128,
        1024,
        2048,
        4096,
    )
    # A model's declared list is authoritative.
    assert resolve(_runtime(prefill_warmup_seq_lens=(128, 8192)), TraceConfig(mode="all")) == (128, 8192)


@pytest.mark.host
def test_layout_refresh_preserves_owner_and_sampling_state_identity(monkeypatch) -> None:
    state = object()
    layout = object()
    prefill_config = SimpleNamespace(
        max_batch_size=4,
        max_prefill_chunk_size=2048,
        device_sampling_enabled=True,
        can_enable_trace=lambda *_: True,
        supports_batched_prefill=True,
        disable_batched_prefill=True,
        max_prefill_batch_size=4,
        batched_prefill_batched_extract=True,
        trace_capture_prime_sequence_lengths=(128,),
        sampling_state_controller=object(),
        sampling_state=state,
    )
    decode_config = SimpleNamespace(
        lane_capacity=4,
        device_sampling_enabled=True,
        force_greedy_top_k=True,
        sampling_state_controller=prefill_config.sampling_state_controller,
        sampling_state=state,
    )
    warmup_config = SimpleNamespace(warmup=object(), prefill_sequence_lengths=(128,))
    resolved_prefill = SimpleNamespace(page_table_layout=layout, sampling_state=state)
    resolved_decode = SimpleNamespace(page_table_layout=layout, sampling_state=state)
    resolved_warmup = SimpleNamespace(page_table_layout=layout)
    prefill_resolve = MagicMock(return_value=resolved_prefill)
    decode_resolve = MagicMock(return_value=resolved_decode)
    warmup_resolve = MagicMock(return_value=resolved_warmup)
    monkeypatch.setattr(executor_module.PrefillRuntimeConfig, "resolve", prefill_resolve)
    monkeypatch.setattr(executor_module.DecodeRuntimeConfig, "resolve", decode_resolve)
    monkeypatch.setattr(executor_module.WarmupCoordinatorConfig, "resolve", warmup_resolve)

    target = object.__new__(ModelExecutor)
    target.model = object()
    target.output_reader = object()
    target.config = SimpleNamespace(trace=object())
    target.prefill_runtime = SimpleNamespace(config=prefill_config)
    target.decode_runtime = SimpleNamespace(config=decode_config)
    target.warmup = SimpleNamespace(config=warmup_config)
    target._resolve_page_table_layout = lambda: layout
    owners = (target.prefill_runtime, target.decode_runtime, target.warmup)

    target._refresh_page_table_layout()

    assert (target.prefill_runtime, target.decode_runtime, target.warmup) == owners
    assert target.page_table_layout is layout
    assert all(owner.config.page_table_layout is layout for owner in owners)
    assert target.prefill_runtime.config.sampling_state is state
    assert target.decode_runtime.config.sampling_state is state
    assert prefill_resolve.call_args.kwargs["trace_capture_prime_sequence_lengths"] == (128,)
    assert prefill_resolve.call_args.kwargs["sampling_state_controller"] is prefill_config.sampling_state_controller
    assert prefill_resolve.call_args.kwargs["sampling_state"] is state
    assert decode_resolve.call_args.kwargs["sampling_state_controller"] is prefill_config.sampling_state_controller
    assert decode_resolve.call_args.kwargs["sampling_state"] is state


@pytest.mark.host
def test_cleanup_is_ordered_retryable_idempotent_and_terminal(expect_error) -> None:
    events = []
    failing = {"reader", "trace"}

    class _Owner:
        def __init__(self, name):
            self.name = name

        def action(self, *args):
            events.append(self.name)
            if self.name in failing:
                raise RuntimeError(self.name)

        cleanup = action
        drain = action
        drain_external_outputs = action
        cleanup_transients = action
        release = action

    target = object.__new__(ModelExecutor)
    target._terminal = False
    target._cleaned_up = False
    target._owner_name = "TestExecutor"
    target.decode_runtime = _Owner("decode")
    target.output_reader = _Owner("reader")
    target.prefill_runtime = _Owner("prefill")
    target.trace_compiler = _Owner("trace")
    target.program_compiler = _Owner("program")
    target.config = SimpleNamespace(device_sampling_enabled=True)
    target.sampling_state_controller = _Owner("sampling-state")
    target.sampling_state = object()
    target.model = SimpleNamespace(sampling=_Owner("sampling"))
    target.kv_cache_manager = _Owner("kv")

    expected = ["decode", "reader", "prefill", "decode", "trace", "program", "sampling-state", "sampling", "kv"]
    with expect_error(RuntimeError, "reader") as raised:
        target.cleanup()
    assert events == expected
    assert [str(error) for error in raised.value.cleanup_failures] == ["trace"]
    assert target.terminal
    assert not target._cleaned_up

    failing.clear()
    target.cleanup()
    target.cleanup()
    assert events == expected * 2
    assert target._cleaned_up


_MODEL_IDS = tuple(sorted(path.parent.name for path in _MODELS_ROOT.glob("*/hf_generator.py")))


@pytest.mark.host
@pytest.mark.parametrize("model_id", _MODEL_IDS)
def test_every_runtime_config_declares_what_the_warmup_ladder_reads(model_id) -> None:
    # The default warmup ladder is bounded by the chunk cap and max_seq_len; a runtime
    # config without either cannot resolve its warmup lengths.
    module = importlib.import_module(f"tt_transformers.models.{model_id}.hf_generator")
    runtime_configs = [
        value
        for name, value in vars(module).items()
        if name.endswith("RuntimeConfig") and dataclasses.is_dataclass(value) and value.__module__ == module.__name__
    ]
    assert runtime_configs, f"{model_id} defines no runtime config dataclass"
    for runtime_config in runtime_configs:
        fields = {field.name for field in dataclasses.fields(runtime_config)}
        assert {"max_prefill_chunk_size", "max_seq_len", "trace_prefill_supported_seq_lens"} <= fields
