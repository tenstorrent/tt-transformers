# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Exercise public generation semantics without opening a TT device."""

from types import SimpleNamespace

import pytest
import torch

from tt_transformers.models.hf_generation import cleanup, generate

pytestmark = pytest.mark.host


class FakeExecutor:
    """Autoregressive next-token model with observable cache and slot routing."""

    def __init__(self, *, batch=4, blocks=128, device_sampling=False, trace="none"):
        self.config = SimpleNamespace(
            device_sampling_enabled=device_sampling,
            trace=SimpleNamespace(mode=trace, prefill_enabled=trace == "all", decode_enabled=trace != "none"),
        )
        self.paged_kv_cache_config = SimpleNamespace(block_size=4, num_blocks=blocks)
        self.kv_cache_manager = SimpleNamespace(bound_context=None)
        self.batch = batch
        self.terminal = False
        self.allocations = 0
        self.calls = []
        self.failure = None
        self.sampling_state_controller = None
        self.eager_execution = object()

    def allocate_kv_cache(self):
        assert self.allocations == 0, "a second physical allocation is invalid"
        self.allocations += 1
        self.kv_cache_manager.bound_context = object()
        self.cache = [[object(), object()]]
        return self.cache

    def compile_prefill(self, **kwargs):
        self.calls.append(("compile_prefill", kwargs))

    def compile_decode(self, **kwargs):
        self.calls.append(("compile_decode", kwargs))

    def _output(self, tokens, sampling_params):
        targets = (tokens + 1) % 10
        if sampling_params is not None:
            return targets, None
        logits = torch.full((len(tokens), 1, 12), -20.0)
        logits[:, :, 10:] = 1000.0  # Padded vocabulary must never win selection.
        logits[torch.arange(len(tokens)), 0, targets] = 10.0
        return logits

    def prefill_forward(self, **kwargs):
        self.calls.append(("prefill", kwargs))
        if self.failure is not None:
            raise self.failure
        tokens = kwargs["tokens"]
        lengths = kwargs["prompt_lens"]
        return self._output(tokens[torch.arange(len(lengths)), lengths - 1], kwargs["sampling_params"])

    def decode_forward(self, tokens, start_pos, **kwargs):
        assert tokens.shape == start_pos.shape == (self.batch,)
        assert kwargs["page_table"].shape[0] == self.batch
        self.calls.append(("decode", dict(tokens=tokens.clone(), start_pos=start_pos.clone(), **kwargs)))
        result = self._output(tokens, kwargs["sampling_params"])
        return result if kwargs["sampling_params"] is not None else (result, None)

    def warmup_model_prefill(self, **kwargs):
        self.calls.append(("warmup_prefill", kwargs))

    def warmup_model_decode(self, **kwargs):
        self.calls.append(("warmup_decode", kwargs))

    def cleanup(self):
        self.calls.append(("cleanup", {}))
        self.terminal = True


def make_llm(**executor_options):
    executor = FakeExecutor(**executor_options)
    return SimpleNamespace(
        model=SimpleNamespace(config=SimpleNamespace(max_batch_size=executor.batch, max_seq_len=32, vocab_size=10)),
        executor=executor,
        runtime_config=SimpleNamespace(max_context_len=32),
        tokenizer=SimpleNamespace(eos_token_id=9, pad_token_id=0),
        generation_config=SimpleNamespace(max_decode_tokens=4, temperature=0.0, top_k=5, top_p=1.0, stop_token_ids=()),
    )


def calls(llm, name):
    return [kwargs for operation, kwargs in llm.executor.calls if operation == name]


@pytest.mark.host
def test_ragged_left_padding_prompt_preservation_eos_and_physical_slots():
    llm = make_llm()
    inputs = torch.tensor([[0, 1, 7], [0, 0, 3]], dtype=torch.int32)
    mask = torch.tensor([[0, 1, 1], [0, 0, 1]])
    output = generate(llm, inputs, mask, max_new_tokens=4)
    assert output.tolist() == [[0, 1, 7, 8, 9, 0, 0], [0, 0, 3, 4, 5, 6, 7]]
    assert output.dtype == torch.long and output.device.type == "cpu"
    prefill = calls(llm, "prefill")[0]
    assert prefill["tokens"].tolist() == [[1, 7], [3, 0]]
    assert prefill["prompt_lens"].tolist() == [2, 1]
    assert prefill["empty_slots"] == [0, 1]
    assert prefill["page_table"].tolist() == [[0, 1], [2, 3]]
    assert [call["start_pos"].tolist() for call in calls(llm, "decode")] == [
        [2, 1, -1, -1],
        [-1, 2, -1, -1],
        [-1, 3, -1, -1],
    ]
    assert inputs.tolist() == [[0, 1, 7], [0, 0, 3]]


@pytest.mark.host
def test_no_mask_infers_distinct_configured_pad_and_supports_right_padding():
    llm = make_llm()
    output = generate(llm, torch.tensor([[3, 0, 0], [1, 2, 3]]), max_new_tokens=1)
    assert output.tolist() == [[3, 0, 0, 4], [1, 2, 3, 4]]
    assert calls(llm, "prefill")[0]["prompt_lens"].tolist() == [1, 3]


@pytest.mark.host
def test_eos_as_pad_is_not_inferred_without_a_mask():
    llm = make_llm()
    llm.tokenizer.pad_token_id = 9
    generate(llm, torch.tensor([[9, 2]]), max_new_tokens=1)
    assert calls(llm, "prefill")[0]["prompt_lens"].tolist() == [2]


@pytest.mark.host
def test_explicit_pad_override_controls_mask_inference():
    llm = make_llm()
    generate(llm, torch.tensor([[5, 5, 0, 2]]), max_new_tokens=1, pad_token_id=5)
    assert calls(llm, "prefill")[0]["tokens"].tolist() == [[0, 2]]


@pytest.mark.host
def test_checkpoint_unsupported_defaults_fail_before_allocation():
    llm = make_llm()
    llm._hf_generation_config = {"repetition_penalty": 1.2}
    with pytest.raises(ValueError, match="Unsupported checkpoint generation option.*repetition_penalty"):
        generate(llm, torch.tensor([[1]]))
    assert llm.executor.allocations == 0


@pytest.mark.host
def test_multiple_eos_and_default_pad_eos_fallback():
    llm = make_llm()
    llm.tokenizer.pad_token_id = None
    output = generate(llm, torch.tensor([[7], [3]]), max_new_tokens=4, eos_token_id=[8, 5])
    assert output.tolist() == [[7, 8, 8], [3, 4, 5]]


@pytest.mark.host
def test_explicit_none_disables_eos_and_checkpoint_defaults_are_immutable():
    llm = make_llm()
    llm._hf_generation_config = {"max_new_tokens": 3, "do_sample": False, "eos_token_id": [8, 9]}
    assert generate(llm, torch.tensor([[7]])).tolist() == [[7, 8]]
    assert generate(llm, torch.tensor([[7]]), eos_token_id=None).tolist() == [[7, 8, 9, 0]]
    assert llm._hf_generation_config == {"max_new_tokens": 3, "do_sample": False, "eos_token_id": [8, 9]}
    assert llm.generation_config.max_decode_tokens == 4


@pytest.mark.host
def test_repeated_calls_reuse_allocation_and_use_new_prompt_positions():
    llm = make_llm()
    first = generate(llm, torch.tensor([[1, 2]]), max_new_tokens=3)
    generate(llm, torch.tensor([[6]]), max_new_tokens=2)
    again = generate(llm, torch.tensor([[1, 2]]), max_new_tokens=3)
    assert torch.equal(first, again)
    assert llm.executor.allocations == 1
    assert [p["prompt_lens"].tolist() for p in calls(llm, "prefill")] == [[2], [1], [2]]
    assert all(p["reset_batch"] for p in calls(llm, "decode"))


@pytest.mark.host
def test_zero_budget_does_not_allocate_or_compile():
    llm = make_llm()
    tokens = torch.tensor([[1, 2]])
    output = generate(llm, tokens, max_new_tokens=0)
    assert torch.equal(output, tokens) and output is not tokens
    assert llm.executor.allocations == 0 and not llm.executor.calls


@pytest.mark.parametrize("device_sampling", [False, True])
@pytest.mark.host
def test_first_token_eos_does_not_decode(device_sampling):
    llm = make_llm(device_sampling=device_sampling)
    assert generate(llm, torch.tensor([[8]]), max_new_tokens=6).tolist() == [[8, 9]]
    assert not calls(llm, "decode")


@pytest.mark.parametrize(
    "options,match",
    [
        ({"max_new_tokens": -1}, "max_new_tokens"),
        ({"max_new_tokens": True}, "max_new_tokens"),
        ({"max_new_tokens": 32}, "context capacity"),
        ({"top_k": -1}, "top_k"),
        ({"top_p": 0.0}, "top_p"),
        ({"top_p": float("nan")}, "top_p"),
        ({"temperature": float("inf")}, "temperature"),
        ({"do_sample": True, "temperature": 0.0}, "temperature"),
        ({"do_sample": "yes"}, "do_sample"),
        ({"eos_token_id": 10}, "eos_token_id"),
        ({"eos_token_id": True}, "eos_token_id"),
        ({"pad_token_id": -1}, "pad_token_id"),
    ],
)
@pytest.mark.host
def test_invalid_options_fail_before_allocation(options, match):
    llm = make_llm()
    with pytest.raises(ValueError, match=match):
        generate(llm, torch.tensor([[1]]), **options)
    assert llm.executor.allocations == 0


@pytest.mark.parametrize(
    "tokens,mask,match",
    [
        (torch.tensor([1]), None, "rank-two"),
        (torch.tensor([[1.0]]), None, "int32 or int64"),
        (torch.tensor([[-1]]), None, "vocabulary"),
        (torch.tensor([[10]]), None, "vocabulary"),
        (torch.ones(5, 1, dtype=torch.long), None, "batch size"),
        (torch.tensor([[1, 2]]), torch.tensor([[0, 0]]), "unmasked"),
        (torch.tensor([[1, 2]]), torch.tensor([[1, 2]]), "zero and one"),
        (torch.tensor([[1, 2]]), torch.tensor([[1]]), "same shape"),
        (torch.tensor([[1, 2, 3]]), torch.tensor([[1, 0, 1]]), "contiguous"),
    ],
)
@pytest.mark.host
def test_invalid_inputs_fail_before_allocation(tokens, mask, match):
    llm = make_llm()
    with pytest.raises(ValueError, match=match):
        generate(llm, tokens, mask)
    assert llm.executor.allocations == 0


@pytest.mark.host
def test_unknown_generate_options_are_not_ignored():
    llm = make_llm()
    with pytest.raises(TypeError, match="num_beams, return_dict_in_generate"):
        generate(llm, torch.tensor([[1]]), num_beams=2, return_dict_in_generate=True)
    assert llm.executor.allocations == 0


@pytest.mark.host
def test_context_checks_use_prompt_lengths_and_physical_kv_capacity():
    llm = make_llm(blocks=1)
    with pytest.raises(ValueError, match="KV blocks"):
        generate(llm, torch.tensor([[1], [2]]), max_new_tokens=1)
    assert llm.executor.allocations == 0
    # A wide, padded input does not consume TT positions for its left padding.
    tokens = torch.zeros((1, 40), dtype=torch.long)
    tokens[0, -1] = 2
    assert generate(llm, tokens, max_new_tokens=1).shape == (1, 41)


@pytest.mark.host
def test_host_sampling_matches_hf_filtering_and_seeded_draw(monkeypatch):
    from transformers.generation.logits_process import (
        TemperatureLogitsWarper,
        TopKLogitsWarper,
        TopPLogitsWarper,
    )

    llm = make_llm()
    logits = torch.tensor([[0.2, 0.3, 1.4, 1.4, -0.6, 2.0, 0.1, -1.5, 1.0, 0.5]])
    monkeypatch.setattr(llm.executor, "prefill_forward", lambda **kwargs: logits[:, None, :])
    scores = logits.clone()
    for processor in (TemperatureLogitsWarper(0.8), TopKLogitsWarper(5), TopPLogitsWarper(0.8)):
        scores = processor(torch.tensor([[1]]), scores)
    torch.manual_seed(23)
    expected = torch.multinomial(scores.softmax(-1), 1)
    torch.manual_seed(23)
    actual = generate(llm, torch.tensor([[1]]), max_new_tokens=1, do_sample=True, temperature=0.8, top_k=5, top_p=0.8)
    assert actual[:, 1:].tolist() == expected.tolist()
    assert llm.generation_config.temperature == 0.0


@pytest.mark.host
def test_device_greedy_and_host_sampling_have_explicit_routes():
    llm = make_llm(device_sampling=True)
    assert generate(llm, torch.tensor([[1]]), max_new_tokens=3).tolist() == [[1, 2, 3, 4]]
    assert calls(llm, "prefill")[-1]["sampling_params"].temperature == 0
    assert calls(llm, "decode")[-1]["sampling_params"].top_k == 1
    generate(llm, torch.tensor([[2]]), max_new_tokens=2, do_sample=True, top_k=1)
    assert calls(llm, "prefill")[-1]["sampling_params"] is None
    assert calls(llm, "decode")[-1]["sampling_params"] is None


@pytest.mark.host
def test_trace_warmup_happens_once_and_coverage_failure_propagates(monkeypatch):
    llm = make_llm(trace="decode_only")
    generate(llm, torch.tensor([[2]]), max_new_tokens=2)
    generate(llm, torch.tensor([[2]]), max_new_tokens=2)
    assert len(calls(llm, "warmup_prefill")) == 1
    assert [c["enable_trace"] for c in calls(llm, "warmup_decode")] == [False, True]
    monkeypatch.setattr(
        llm.executor, "compile_prefill", lambda **kwargs: (_ for _ in ()).throw(RuntimeError("trace coverage missing"))
    )
    with pytest.raises(RuntimeError, match="trace coverage missing"):
        generate(llm, torch.tensor([[1, 2, 3]]), max_new_tokens=2)
    assert not llm._hf_generating


@pytest.mark.host
def test_failure_does_not_poison_request_flag_or_replace_cache():
    llm = make_llm()
    llm.executor.failure = RuntimeError("prefill failed")
    with pytest.raises(RuntimeError, match="prefill failed"):
        generate(llm, torch.tensor([[1]]), max_new_tokens=2)
    assert not llm._hf_generating
    llm.executor.failure = None
    assert generate(llm, torch.tensor([[2]]), max_new_tokens=2).tolist() == [[2, 3, 4]]
    assert llm.executor.allocations == 1


@pytest.mark.host
def test_cleanup_orders_executor_before_model_keeps_mesh_and_is_terminal(monkeypatch):
    import tt_transformers.device_utils

    llm = make_llm()
    events = []
    monkeypatch.setattr(llm.executor, "cleanup", lambda: events.append("executor"))
    monkeypatch.setattr(tt_transformers.device_utils, "cleanup_object_graph", lambda model: events.append("model"))
    cleanup(llm)
    cleanup(llm)
    assert events == ["executor", "model"]
    with pytest.raises(RuntimeError, match="terminal"):
        generate(llm, torch.tensor([[1]]))


@pytest.mark.host
def test_cleanup_defers_model_release_after_executor_failure_and_retries(monkeypatch):
    import tt_transformers.device_utils

    llm = make_llm()
    first, second = RuntimeError("executor cleanup"), RuntimeError("model cleanup")
    model_releases = []

    def release_model(model):
        model_releases.append(model)
        raise second

    monkeypatch.setattr(llm.executor, "cleanup", lambda: (_ for _ in ()).throw(first))
    monkeypatch.setattr(tt_transformers.device_utils, "cleanup_object_graph", release_model)
    with pytest.raises(RuntimeError, match="executor cleanup") as raised:
        cleanup(llm)
    assert raised.value is first
    assert model_releases == []
    assert llm._hf_terminal and not getattr(llm, "_hf_cleaned_up", False)
    monkeypatch.setattr(llm.executor, "cleanup", lambda: None)
    with pytest.raises(RuntimeError, match="model cleanup") as raised:
        cleanup(llm)
    assert raised.value is second
    assert model_releases == [llm.model]
    monkeypatch.setattr(tt_transformers.device_utils, "cleanup_object_graph", lambda model: None)
    cleanup(llm)
    assert llm._hf_cleaned_up


@pytest.mark.host
def test_model_operation_lock_excludes_generation_and_cleanup(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    llm = make_llm()
    admitted, finish = Event(), Event()
    original = llm.executor.prefill_forward

    def blocked_prefill(**kwargs):
        admitted.set()
        assert finish.wait(timeout=5)
        return original(**kwargs)

    monkeypatch.setattr(llm.executor, "prefill_forward", blocked_prefill)
    with ThreadPoolExecutor(max_workers=1) as pool:
        first = pool.submit(generate, llm, torch.tensor([[1]]), max_new_tokens=1)
        assert admitted.wait(timeout=5)
        try:
            with pytest.raises(RuntimeError, match="another operation"):
                generate(llm, torch.tensor([[2]]), max_new_tokens=1)
            with pytest.raises(RuntimeError, match="another operation"):
                cleanup(llm)
        finally:
            finish.set()
        assert first.result(timeout=5).tolist() == [[1, 2]]
    assert llm.executor.allocations == 1


@pytest.mark.host
def test_success_and_failure_reset_sampling_request_state():
    llm = make_llm(device_sampling=True)
    resets = []
    llm.executor.sampling_state = object()
    llm.executor.sampling_state_controller = SimpleNamespace(reset=lambda state: resets.append(state))
    generate(llm, torch.tensor([[1]]), max_new_tokens=1)
    assert resets == [llm.executor.sampling_state] * 2
    llm.executor.failure = RuntimeError("failed")
    with pytest.raises(RuntimeError, match="failed"):
        generate(llm, torch.tensor([[2]]), max_new_tokens=1)
    assert resets == [llm.executor.sampling_state] * 4


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.host
def test_invalid_active_logits_fail_and_release_request_lock(monkeypatch, value):
    llm = make_llm()
    monkeypatch.setattr(llm.executor, "prefill_forward", lambda **kwargs: torch.full((1, 1, 10), value))
    with pytest.raises(ValueError, match="invalid logits"):
        generate(llm, torch.tensor([[1]]), max_new_tokens=1)
    assert not llm._hf_generating
    assert llm._hf_operation_lock.acquire(blocking=False)
    llm._hf_operation_lock.release()


@pytest.mark.host
def test_page_table_disjoint_active_allocations_with_unused_lane_rows():
    llm = make_llm(blocks=3)
    generate(llm, torch.tensor([[1, 2, 3, 4, 5], [0, 0, 0, 0, 2]]), max_new_tokens=2)
    page_table = calls(llm, "decode")[0]["page_table"]
    assert page_table.tolist() == [[0, 1], [2, 0], [0, 0], [0, 0]]
    # Only the initialized prefix is exposed; positions suppress both inactive
    # physical rows and the unused second block of the shorter active prompt.
    assert calls(llm, "decode")[0]["start_pos"].tolist() == [5, 1, -1, -1]


@pytest.mark.host
def test_real_checkpoint_generation_config_accepts_unset_fields(monkeypatch):
    from transformers import GenerationConfig

    import tt_transformers.models.hf_generation as helper

    llm = make_llm()
    llm._hf_generation_config = GenerationConfig.from_dict(
        {
            "bos_token_id": 128000,
            "do_sample": True,
            "eos_token_id": [128001, 128008, 128009],
            "temperature": 0.6,
            "top_p": 0.9,
            "transformers_version": "4.43.0",
        }
    )
    before = llm._hf_generation_config.to_dict()
    selected = []
    original = helper._select

    def select(logits, **kwargs):
        selected.append(kwargs)
        return original(logits, **kwargs)

    monkeypatch.setattr(helper, "_select", select)
    # Token IDs are from the real Llama checkpoint; disable EOS for this fake
    # ten-token model while retaining its real generation-default representation.
    assert generate(llm, torch.tensor([[1]]), max_new_tokens=1, eos_token_id=None).tolist() == [[1, 2]]
    assert selected[0]["do_sample"] is True
    assert selected[0]["temperature"] == 0.6
    assert selected[0]["top_p"] == 0.9
    assert selected[0]["top_k"] == GenerationConfig._get_default_generation_params()["top_k"] == 50
    assert llm._hf_generation_config.to_dict() == before


@pytest.mark.host
def test_real_unset_checkpoint_sampling_fields_use_hf_effective_defaults(monkeypatch):
    from transformers import GenerationConfig

    import tt_transformers.models.hf_generation as helper

    llm = make_llm(device_sampling=True)
    llm.generation_config.temperature = 0.7
    llm.generation_config.top_k = 3
    llm.generation_config.top_p = 0.2
    llm._hf_generation_config = GenerationConfig()
    defaults = GenerationConfig._get_default_generation_params()
    for name in ("do_sample", "temperature", "top_k", "top_p"):
        assert helper._default(llm, name) == defaults[name]
    # Unset do_sample is greedy even if legacy defaults requested sampling.
    assert generate(llm, torch.tensor([[1]]), max_new_tokens=1).tolist() == [[1, 2]]
    assert calls(llm, "prefill")[-1]["sampling_params"].temperature == 0.0
    # An explicit checkpoint temperature alone also does not turn sampling on.
    llm._hf_generation_config = GenerationConfig.from_dict({"temperature": 0.6})
    generate(llm, torch.tensor([[2]]), max_new_tokens=1)
    assert calls(llm, "prefill")[-1]["sampling_params"].temperature == 0.0


@pytest.mark.parametrize("name,value", [("num_beams", 2), ("repetition_penalty", 1.2), ("output_scores", True)])
@pytest.mark.host
def test_real_checkpoint_explicit_unsupported_options_still_fail(name, value):
    from transformers import GenerationConfig

    llm = make_llm()
    llm._hf_generation_config = GenerationConfig.from_dict({name: value})
    with pytest.raises(ValueError, match=f"Unsupported checkpoint generation option.*{name}"):
        generate(llm, torch.tensor([[1]]), max_new_tokens=1)
    assert llm.executor.allocations == 0
