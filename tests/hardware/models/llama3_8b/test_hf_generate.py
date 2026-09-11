# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Public HF-style generation against the established direct-executor path."""

from __future__ import annotations

import os
import time
from functools import partial
from types import SimpleNamespace

import pytest
import torch
import ttnn
from examples.common.run_helpers import run_perf_benchmark
from examples.common.runtime import open_mesh_device
from examples.common.trace_region_sizes import resolve_trace_region_size
from tests.support.marker_policy import selected_topology_marks

from tt_transformers.device_ownership import default_device_scope
from tt_transformers.llm_runtime.config import PagedKVCacheConfig, TraceConfig, WarmupConfig
from tt_transformers.mesh_utils import make_contiguous_page_table
from tt_transformers.models.llama3_8b.hf_generator import DEFAULT_HF_MODEL, Llama3ExecutorConfig, from_pretrained

pytestmark = [pytest.mark.timeout(3600), *selected_topology_marks(pytest)]

_MAX_SEQ_LEN = 1024
_NEW_TOKENS = 8


@pytest.mark.device
@pytest.mark.model
@pytest.mark.slow
def test_hf_style_demo_cli(capsys):
    from examples.llama3_8b import demo

    if os.environ.get("MESH_DEVICE", "").strip().upper() not in {"N150", "P150"}:
        pytest.skip("Set MESH_DEVICE=N150 or P150 for the simple text demo")
    demo.main(
        [
            "--prompt",
            "What is the capital of France? Answer in one short sentence.",
            "--max-new-tokens",
            "16",
            "--max-seq-len",
            "1024",
        ]
    )
    output = capsys.readouterr().out
    assert "Paris" in output, output
    print("HF_SIMPLE_DEMO: public loading, chat template, generate, decode, and cleanup completed")
    print("HF_SIMPLE_DEMO_TEXT:", output.strip().splitlines()[-1])


@pytest.fixture
def generation_model(request):
    selected = os.environ.get("MESH_DEVICE", "").strip().upper()
    if selected not in {"N150", "P150"}:
        pytest.skip("Set MESH_DEVICE=N150 or P150 for the single-device HF generation gate")
    if ttnn.device.is_blackhole() != (selected == "P150"):
        pytest.fail(f"MESH_DEVICE={selected} does not match the actual architecture")
    max_batch_size, traced = request.param if isinstance(request.param, tuple) else (request.param, False)
    trace_region_size = resolve_trace_region_size("llama3.1-8b", selected) if traced else 0
    executor_config = None
    if traced:
        executor_config = Llama3ExecutorConfig(
            trace=TraceConfig(mode="all"),
            warmup=WarmupConfig(prefill_seq_lens=(128,), prefill_batch_sizes=(1,)),
            paged_kv_cache=PagedKVCacheConfig(
                block_size=32,
                max_num_blocks=_MAX_SEQ_LEN // 32,
                num_blocks=_MAX_SEQ_LEN // 32,
                dtype=ttnn.bfloat8_b,
            ),
            device_sampling_enabled=True,
        )
    # This context propagates test-body failures on both architectures. The
    # legacy generic WH fixture can turn an exception crossing yield into skip.
    with open_mesh_device(
        {"mesh_shape": (1, 1), "trace_region_size": trace_region_size, "num_command_queues": 1}
    ) as mesh:
        with default_device_scope(ttnn, mesh, owner="test_hf_generate"):
            mesh.enable_program_cache()
            model = from_pretrained(
                mesh,
                hf_model=os.environ.get("HF_MODEL", DEFAULT_HF_MODEL),
                max_batch_size=max_batch_size,
                max_seq_len=_MAX_SEQ_LEN,
                optimizations="performance",
                executor_config=executor_config,
            )
            try:
                yield model
            finally:
                model.cleanup()
                model.cleanup()
                ttnn.synchronize_device(mesh)
                print("HF_GENERATE_TEARDOWN: executor/model cleaned; borrowed mesh still usable")


def _chat_inputs(model, text):
    return model.tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )


def _generate(model, inputs):
    output = model.generate(**inputs, max_new_tokens=_NEW_TOKENS, do_sample=False, eos_token_id=None)
    assert output.device.type == "cpu"
    assert output.dtype == torch.long
    assert output.shape == (inputs["input_ids"].shape[0], inputs["input_ids"].shape[1] + _NEW_TOKENS)
    assert torch.equal(output[:, : inputs["input_ids"].shape[1]], inputs["input_ids"])
    assert ((output >= 0) & (output < model.model.vocab_size)).all()
    return output


def _demo_reference(model, inputs, *, eager=False):
    """Borrow the already-allocated handle; do not allocate a second KV cache."""
    tokens = inputs["input_ids"]
    config = model.executor.paged_kv_cache_config
    page_table = make_contiguous_page_table(model.model.config.max_batch_size, _MAX_SEQ_LEN, config.block_size)
    target = model.executor
    if eager:
        # Preserve executor ownership/readback while selecting its eager target
        # for every compile and forward call. The benchmark uses host argmax.
        target = SimpleNamespace(
            mesh_device=model.executor.mesh_device,
            cluster_shape=model.executor.cluster_shape,
            **{
                name: partial(getattr(model.executor, name), execution=model.executor.eager_execution)
                for name in ("compile_prefill", "compile_decode", "prefill_forward", "decode_forward")
            },
        )
    return run_perf_benchmark(
        target,
        tokens=tokens,
        kv_cache=model.executor.kv_cache_manager._bound_cache,
        page_table=page_table,
        prompt_lens=inputs["attention_mask"].sum(dim=-1),
        max_batch_size=model.model.config.max_batch_size,
        num_decode_tokens=_NEW_TOKENS - 1,
        sampling_params=None,
    )


@pytest.mark.parametrize("generation_model", [1], indirect=True, ids=["batch1"])
@pytest.mark.device
@pytest.mark.model
@pytest.mark.slow
def test_generate_matches_demo_and_reuses_model(generation_model):
    model = generation_model
    inputs = _chat_inputs(model, "What is the capital of France? Answer in one short sentence.")
    first = _generate(model, inputs)
    reference = _demo_reference(model, inputs)
    assert first[:, inputs["input_ids"].shape[1] :].tolist() == reference.generated_token_ids

    _generate(model, _chat_inputs(model, "Name three colors."))
    start = time.perf_counter()
    repeated = _generate(model, inputs)
    elapsed = time.perf_counter() - start
    assert torch.equal(first, repeated), "an intervening request changed greedy generation"
    print(f"HF_GENERATE: public_vs_demo=exact repeated_call=exact warm_request_seconds={elapsed:.6f}")
    print(f"HF_GENERATE_DEMO: TTFT_ms={reference.ttft_ms:.3f} TPOT_ms={reference.decode_latency_mean_ms:.3f}")
    print("HF_GENERATE_TEXT:", model.tokenizer.decode(first[0, inputs["input_ids"].shape[1] :]))


@pytest.mark.parametrize("generation_model", [32], indirect=True, ids=["batch3-capacity32"])
@pytest.mark.device
@pytest.mark.model
@pytest.mark.slow
def test_generate_left_padded_batch(generation_model):
    model = generation_model
    prompts = ["Name a primary color.", "What is the capital of France?", "Count from one to three."]
    rows = [_chat_inputs(model, prompt)["input_ids"][0] for prompt in prompts]
    width = max(row.numel() for row in rows)
    pad = model.tokenizer.pad_token_id
    if pad is None:
        pad = model.tokenizer.eos_token_id
    tokens = torch.full((len(rows), width), pad, dtype=torch.long)
    mask = torch.zeros_like(tokens)
    for index, row in enumerate(rows):
        tokens[index, -row.numel() :] = row
        mask[index, -row.numel() :] = 1
    inputs = {"input_ids": tokens, "attention_mask": mask}
    output = _generate(model, inputs)
    assert torch.equal(output, _generate(model, inputs))

    # The existing demo consumes compact prompts plus explicit row lengths.
    compact = torch.zeros_like(tokens)
    for index, row in enumerate(rows):
        compact[index, : row.numel()] = row
    reference = _demo_reference(model, {"input_ids": compact, "attention_mask": mask})
    assert output[:, width:].tolist() == reference.generated_token_ids
    print("HF_GENERATE_BATCH: active_rows=3 capacity=32 left_padding=preserved public_vs_demo=exact")


@pytest.mark.parametrize("generation_model", [(1, True)], indirect=True, ids=["batch1-trace-all-device-greedy"])
@pytest.mark.device
@pytest.mark.model
@pytest.mark.slow
def test_generate_trace_device_greedy_matches_eager_host(generation_model):
    model = generation_model
    inputs = _chat_inputs(model, "What is the capital of France? Answer in one short sentence.")
    assert inputs["input_ids"].shape[1] <= 128, "the gate covers the explicit 128-token prefill bucket"
    assert model.executor.config.trace.mode == "all"
    assert model.executor.config.device_sampling_enabled
    traced = model.executor.traced_executor

    output = _generate(model, inputs)
    after_generate = traced.runtime_summary()
    assert after_generate["trace_replays_by_operation"]["prefill"] >= 1
    assert after_generate["trace_replays_by_operation"]["decode"] >= _NEW_TOKENS - 1
    evidence = traced.recent_prefill_replay_evidence
    assert evidence and all(item.sampling_path in {"argmax", "topk"} for item in evidence)
    assert all(item.execution == "trace_replay" for item in evidence)

    reference = _demo_reference(model, inputs, eager=True)
    after_reference = traced.runtime_summary()
    assert after_reference["trace_replays_by_operation"] == after_generate["trace_replays_by_operation"]
    assert after_reference["eager_prefill_executions"] > after_generate["eager_prefill_executions"]
    assert output[:, inputs["input_ids"].shape[1] :].tolist() == reference.generated_token_ids

    start = time.perf_counter()
    repeated = _generate(model, inputs)
    elapsed = time.perf_counter() - start
    assert torch.equal(output, repeated), "eager reference execution changed subsequent traced generation"
    after_repeat = traced.runtime_summary()
    assert after_repeat["eager_prefill_executions"] == after_reference["eager_prefill_executions"]
    assert (
        after_repeat["trace_replays_by_operation"]["prefill"] > after_reference["trace_replays_by_operation"]["prefill"]
    )
    assert (
        after_repeat["trace_replays_by_operation"]["decode"] > after_reference["trace_replays_by_operation"]["decode"]
    )
    assert after_repeat["strict_coverage_misses"] == 0
    print(
        f"HF_GENERATE_TRACE: device_greedy_vs_eager_host=exact repeated_call=exact warm_request_seconds={elapsed:.6f}"
    )
    print(f"HF_GENERATE_TRACE_COUNTS: {after_repeat['trace_replays_by_operation']}")
    print(f"HF_GENERATE_EAGER: TTFT_ms={reference.ttft_ms:.3f} TPOT_ms={reference.decode_latency_mean_ms:.3f}")
