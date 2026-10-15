# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Check sequential Llama target logits without opening a TT device."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from tt_transformers.models.llama32_1b.speculation import verify_logits


def _arguments(batch=4, drafts=3):
    width = drafts + 1
    return {
        "tokens": torch.arange(1, batch * width + 1, dtype=torch.int32).reshape(batch, width),
        "start_pos": 3 + torch.arange(batch)[:, None] * 4 + torch.arange(width)[None, :],
        "page_table": torch.arange(8, dtype=torch.int32).repeat(batch, 1),
        "num_valid_drafts": torch.full((batch,), drafts, dtype=torch.int32),
        "accepted_counts": torch.full((batch,), width, dtype=torch.int32),
        "batch_size": batch,
        "vocab_size": 19,
        "max_seq_len": 32,
        "block_size": 4,
        "num_blocks": 16,
    }


def _target_logits(tokens, positions, vocab_size):
    rows = torch.arange(tokens.shape[0], dtype=torch.float32)
    values = tokens.float() * 5 + positions.float() * 3 + rows * 7
    return values[:, None, None] + torch.arange(vocab_size, dtype=torch.float32)[None, None, :] / 32


class _Target:
    def __init__(self, batch, vocab_size, *, tuple_output=True):
        self.calls = []
        self.output = torch.empty((batch, 1, vocab_size), dtype=torch.float32)
        self.tuple_output = tuple_output

    def __call__(self, *, tokens, start_pos, page_table):
        self.calls.append((tokens.clone(), start_pos.clone(), page_table.clone()))
        self.output.copy_(_target_logits(tokens, start_pos, self.output.shape[-1]))
        return (self.output, None) if self.tuple_output else self.output


def _assert_rejected_before_decode(arguments, *, error=ValueError, match):
    target = _Target(arguments["batch_size"], arguments["vocab_size"])
    originals = {name: value.clone() for name, value in arguments.items() if isinstance(value, torch.Tensor)}
    with pytest.raises(error, match=match):
        verify_logits(**arguments, decode=target)
    assert target.calls == []
    for name, original in originals.items():
        assert torch.equal(arguments[name], original), name


@pytest.mark.host
@pytest.mark.parametrize("batch", (1, 2, 4))
@pytest.mark.parametrize("drafts", (1, 2, 3))
def test_supported_batch_and_draft_length_preserve_each_target_distribution(batch, drafts):
    arguments = _arguments(batch, drafts)
    target = _Target(batch, arguments["vocab_size"])

    result = verify_logits(**arguments, decode=target)

    assert result.shape == (batch, drafts + 1, arguments["vocab_size"])
    assert result.device.type == "cpu"
    assert result.dtype == torch.float32
    expected = (
        arguments["tokens"].float()[..., None] * 5
        + arguments["start_pos"].float()[..., None] * 3
        + torch.arange(batch).float()[:, None, None] * 7
        + torch.arange(arguments["vocab_size"]).float()[None, None, :] / 32
    )
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    torch.testing.assert_close(result.softmax(-1), expected.softmax(-1), rtol=0, atol=0)
    assert len(target.calls) == drafts + 1


@pytest.mark.host
@pytest.mark.parametrize("tuple_output", (False, True))
def test_logits_survive_target_output_storage_reuse(tuple_output):
    arguments = _arguments(2, 3)
    target = _Target(2, arguments["vocab_size"], tuple_output=tuple_output)

    result = verify_logits(**arguments, decode=target)
    expected_anchor = _target_logits(arguments["tokens"][:, 0], arguments["start_pos"][:, 0], 19)[:, 0]
    expected_final = _target_logits(arguments["tokens"][:, -1], arguments["start_pos"][:, -1], 19)[:, 0]
    target.output.fill_(9999)

    torch.testing.assert_close(result[:, 0], expected_anchor, rtol=0, atol=0)
    torch.testing.assert_close(result[:, -1], expected_final, rtol=0, atol=0)
    assert result.untyped_storage().data_ptr() != target.output.untyped_storage().data_ptr()


@pytest.mark.host
def test_mixed_drafts_mask_inactive_tokens_positions_and_output_columns():
    arguments = _arguments()
    arguments["num_valid_drafts"] = torch.tensor([3, 1, 0, 0], dtype=torch.int32)
    arguments["start_pos"][1, 2:] = -1
    arguments["start_pos"][2, 1:] = -1
    arguments["start_pos"][3] = -1
    arguments["tokens"][1, 2:] = -1
    arguments["tokens"][2, 1:] = -1
    arguments["tokens"][3] = -1
    arguments["page_table"][3] = -1
    target = _Target(4, arguments["vocab_size"])

    result = verify_logits(**arguments, decode=target)

    assert [positions.tolist() for _, positions, _ in target.calls] == [
        [3, 7, 11, -1],
        [4, 8, -1, -1],
        [5, -1, -1, -1],
        [6, -1, -1, -1],
    ]
    assert [tokens.tolist() for tokens, _, _ in target.calls] == [
        [1, 5, 9, 0],
        [2, 6, 0, 0],
        [3, 0, 0, 0],
        [4, 0, 0, 0],
    ]
    assert torch.count_nonzero(result[1, 2:]) == 0
    assert torch.count_nonzero(result[2, 1:]) == 0
    assert torch.count_nonzero(result[3]) == 0
    for tokens, positions, page_table in target.calls:
        assert torch.equal(tokens[positions < 0], torch.zeros_like(tokens[positions < 0]))
        assert torch.equal(page_table, arguments["page_table"])


@pytest.mark.host
def test_zero_drafts_execute_only_anchor_after_previous_full_acceptance():
    arguments = _arguments()
    arguments["num_valid_drafts"].zero_()
    arguments["tokens"][:, 1:] = -1
    arguments["start_pos"][:, 1:] = -1
    target = _Target(4, arguments["vocab_size"])

    result = verify_logits(**arguments, decode=target)

    assert len(target.calls) == 1
    assert arguments["accepted_counts"].tolist() == [4, 4, 4, 4]
    torch.testing.assert_close(result[:, :1], target.output, rtol=0, atol=0)
    assert torch.count_nonzero(result[:, 1:]) == 0


@pytest.mark.host
def test_all_padding_rows_execute_no_target_forward():
    arguments = _arguments()
    arguments["num_valid_drafts"].zero_()
    arguments["start_pos"].fill_(-1)
    arguments["tokens"].fill_(-1)
    arguments["page_table"].fill_(-1)
    target = _Target(4, arguments["vocab_size"])

    result = verify_logits(**arguments, decode=target)

    assert target.calls == []
    assert result.shape == (4, 4, 19)
    assert torch.count_nonzero(result) == 0


@pytest.mark.host
@pytest.mark.parametrize(
    ("field", "index", "value", "message"),
    [
        ("tokens", (3, 3), -1, "valid tokens"),
        ("tokens", (3, 3), 19, "valid tokens"),
        ("start_pos", (3, 3), 21, "consecutive"),
        ("num_valid_drafts", 3, -1, "num_valid_drafts"),
        ("num_valid_drafts", 3, 4, "num_valid_drafts"),
        ("accepted_counts", 3, 0, "accepted_counts"),
        ("accepted_counts", 3, 5, "accepted_counts"),
    ],
)
def test_invalid_later_candidate_is_rejected_before_any_target_mutation(field, index, value, message):
    arguments = _arguments()
    arguments[field][index] = value
    _assert_rejected_before_decode(arguments, match=message)


@pytest.mark.host
def test_inactive_later_position_is_rejected_before_any_target_mutation():
    arguments = _arguments()
    arguments["num_valid_drafts"][3] = 1
    _assert_rejected_before_decode(arguments, match="inactive start_pos")


@pytest.mark.host
def test_padding_row_with_drafts_is_rejected_before_any_target_mutation():
    arguments = _arguments()
    arguments["start_pos"][3] = -1
    _assert_rejected_before_decode(arguments, match="padding rows")


@pytest.mark.host
def test_maximum_sequence_length_is_checked_before_any_target_mutation():
    arguments = _arguments()
    arguments["start_pos"][3] = torch.tensor([29, 30, 31, 32])
    _assert_rejected_before_decode(arguments, match="max_seq_len")


@pytest.mark.host
@pytest.mark.parametrize("field", ("tokens", "start_pos", "page_table", "num_valid_drafts", "accepted_counts"))
def test_floating_side_tensor_is_rejected_before_any_target_mutation(field):
    arguments = _arguments()
    arguments[field] = arguments[field].float()
    _assert_rejected_before_decode(arguments, error=TypeError, match=field)


@pytest.mark.host
@pytest.mark.parametrize("field", ("num_valid_drafts", "accepted_counts"))
def test_side_tensor_shape_is_checked_before_any_target_mutation(field):
    arguments = _arguments()
    arguments[field] = arguments[field][:, None]
    _assert_rejected_before_decode(arguments, match=field)


@pytest.mark.host
@pytest.mark.parametrize("physical_page", (-1, 16))
def test_later_allocated_page_is_checked_before_any_target_mutation(physical_page):
    arguments = _arguments()
    arguments["page_table"][3, 4] = physical_page
    _assert_rejected_before_decode(arguments, match="unallocated KV blocks")


@pytest.mark.host
def test_missing_later_page_is_checked_before_any_target_mutation():
    arguments = _arguments()
    arguments["page_table"] = arguments["page_table"][:, :4]
    _assert_rejected_before_decode(arguments, match="too narrow")


@pytest.mark.host
def test_unused_page_table_tail_does_not_require_allocation():
    arguments = _arguments(1, 1)
    arguments["page_table"][:, 2:] = -1
    target = _Target(1, arguments["vocab_size"])

    result = verify_logits(**arguments, decode=target)

    assert result.shape == (1, 2, 19)
    assert len(target.calls) == 2


@pytest.mark.host
def test_full_slot_permutation_does_not_change_host_logits_or_inputs():
    arguments = _arguments()
    arguments["slot_remap"] = torch.tensor([3, 0, 2, 1], dtype=torch.int32)
    original_remap = arguments["slot_remap"].clone()

    remapped = verify_logits(**arguments, decode=_Target(4, 19))
    plain = verify_logits(
        **{name: value for name, value in arguments.items() if name != "slot_remap"}, decode=_Target(4, 19)
    )

    torch.testing.assert_close(remapped, plain, rtol=0, atol=0)
    assert torch.equal(arguments["slot_remap"], original_remap)


@pytest.mark.host
@pytest.mark.parametrize(
    "slot_remap",
    [
        torch.tensor([-1, 0, 1, 2]),
        torch.tensor([0, 0, 2, 3]),
        torch.tensor([0, 1, 2, 4]),
        torch.tensor([0, 1, 2]),
        torch.tensor([[0, 1, 2, 3]]),
    ],
)
def test_invalid_slot_permutation_is_rejected_before_any_target_mutation(slot_remap):
    arguments = _arguments()
    arguments["slot_remap"] = slot_remap
    _assert_rejected_before_decode(arguments, match="slot_remap")


@pytest.mark.host
def test_noninteger_slot_permutation_is_rejected_before_any_target_mutation():
    arguments = _arguments()
    arguments["slot_remap"] = torch.arange(4, dtype=torch.float32)
    _assert_rejected_before_decode(arguments, error=TypeError, match="slot_remap")


@pytest.mark.host
def test_next_outer_verification_uses_supplied_position_after_rejection():
    arguments = _arguments(1, 3)
    arguments["start_pos"][0] = torch.tensor([7, 8, 9, 10])
    target = _Target(1, arguments["vocab_size"])
    verify_logits(**arguments, decode=target)
    arguments["start_pos"][0] = torch.tensor([8, 9, 10, 11])
    arguments["accepted_counts"].fill_(1)
    arguments["tokens"][0] = torch.tensor([13, 14, 15, 16])

    result = verify_logits(**arguments, decode=target)

    assert [int(positions[0]) for _, positions, _ in target.calls] == [7, 8, 9, 10, 8, 9, 10, 11]
    expected_anchor = _target_logits(torch.tensor([13]), torch.tensor([8]), 19)
    torch.testing.assert_close(result[:, :1], expected_anchor, rtol=0, atol=0)


@pytest.mark.host
@pytest.mark.parametrize("output", (None, torch.zeros(1, 2, 19), torch.zeros(1, 1, 19, dtype=torch.int64)))
def test_target_output_contract_rejects_missing_wrong_shape_or_integer_logits(output):
    arguments = _arguments(1, 1)
    with pytest.raises(TypeError, match="CPU floating logits"):
        verify_logits(**arguments, decode=lambda **kwargs: output)


@pytest.fixture
def generator_runtime(monkeypatch):
    monkeypatch.setenv("MESH_DEVICE", "N150")
    ttnn = pytest.importorskip("ttnn", reason="generator contract tests require the optional real TTNN runtime")
    contract = pytest.importorskip(
        "vllm_tt_plugin.spec_decode", reason="generator tests require the real plugin contract"
    )
    from tt_transformers.llm_runtime.config import PagedKVCacheConfig, TraceConfig
    from tt_transformers.llm_runtime.vllm_adapter import VLLMAdapter, VLLMAdapterConfig
    from tt_transformers.models.llama32_1b.vllm_generator import Llama32_1BGenerator

    return SimpleNamespace(
        ttnn=ttnn,
        contract=contract,
        PagedKVCacheConfig=PagedKVCacheConfig,
        TraceConfig=TraceConfig,
        VLLMAdapter=VLLMAdapter,
        VLLMAdapterConfig=VLLMAdapterConfig,
        Llama32_1BGenerator=Llama32_1BGenerator,
    )


def _launch_configuration():
    return SimpleNamespace(
        model_config=SimpleNamespace(model="meta-llama/Llama-3.2-1B", hf_config=SimpleNamespace()),
        speculative_config=SimpleNamespace(method="ngram"),
        scheduler_config=SimpleNamespace(async_scheduling=False),
        cache_config=SimpleNamespace(enable_prefix_caching=False),
        additional_config={"tt": {"trace_mode": "none", "optimizations": "performance"}},
        parallel_config=SimpleNamespace(data_parallel_size=1),
    )


def _generator_with_real_adapter(runtime, *, batch=2):
    cache_config = runtime.PagedKVCacheConfig(
        block_size=4, max_num_blocks=16, num_blocks=16, dtype=runtime.ttnn.bfloat16
    )
    adapter = runtime.VLLMAdapter(
        runtime.VLLMAdapterConfig.resolve(
            trace=runtime.TraceConfig(mode="all"),
            paged_kv_cache=cache_config,
            expected_num_layers=1,
            model_kv_cache_dtype=runtime.ttnn.bfloat16,
            request_state_fields=("prompt_tokens", "output_tokens", "slot_remap"),
        )
    )
    output_target = _Target(batch, 19)
    target_calls = []
    target_returns = []

    def decode_forward(**kwargs):
        target_calls.append(kwargs)
        output = output_target(tokens=kwargs["tokens"], start_pos=kwargs["start_pos"], page_table=kwargs["page_table"])
        target_returns.append(output)
        return output

    target = SimpleNamespace(
        model=SimpleNamespace(
            config=SimpleNamespace(max_batch_size=batch, vocab_size=19, max_seq_len=32, num_devices=1)
        ),
        config=SimpleNamespace(paged_kv_cache=cache_config),
        eager_execution=object(),
        decode_forward=decode_forward,
        calls=target_calls,
        returns=target_returns,
        output=output_target.output,
    )
    return runtime.Llama32_1BGenerator(target, adapter), target


def _generator_verify_arguments(batch=2, drafts=3):
    return {
        name: value
        for name, value in _arguments(batch, drafts).items()
        if name in ("tokens", "start_pos", "page_table", "num_valid_drafts", "accepted_counts")
    } | {"enable_trace": False, "spec_mode": "logits", "read_from_device": True}


@pytest.mark.host
@pytest.mark.parametrize("batch", (1, 2, 4))
@pytest.mark.parametrize("drafts", (1, 2, 3))
def test_generator_admission_returns_real_logits_spec_plan_at_all_nine_points(generator_runtime, batch, drafts):
    outcome = generator_runtime.Llama32_1BGenerator.spec_plan(_launch_configuration(), batch, drafts)

    assert isinstance(outcome, generator_runtime.contract.SpecPlan)
    assert outcome.effective_k == drafts
    assert outcome.accept_modes == ("logits",)
    assert outcome.supports_narrow_decode is False
    assert outcome.lanes_per_request == 1
    assert outcome.extra_bytes_per_seq == 0
    assert outcome.extra_bytes_per_token == 0
    assert outcome.drafter_state == "internal"
    assert outcome.block_width == 1 + drafts


@pytest.mark.host
@pytest.mark.parametrize(
    ("model_name", "hf_model_name"),
    [
        ("meta-llama/Llama-3.2-3B", "meta-llama/Llama-3.2-3B"),
        ("meta-llama/Llama-3.2-1B-Instruct", "meta-llama/Llama-3.2-1B"),
        ("meta-llama/Llama-3.2-1B", "meta-llama/Llama-3.2-1B-Instruct"),
        ("/models/renamed-checkpoint", "/models/renamed-checkpoint"),
    ],
)
def test_generator_admission_rejects_unqualified_or_instruct_checkpoint(generator_runtime, model_name, hf_model_name):
    configuration = _launch_configuration()
    configuration.model_config.model = model_name
    configuration.model_config.hf_config._name_or_path = hf_model_name

    outcome = generator_runtime.Llama32_1BGenerator.spec_plan(configuration, 2, 3)

    assert isinstance(outcome, generator_runtime.contract.SpecReject)
    assert "base checkpoint" in outcome.reason


@pytest.mark.host
def test_generator_admission_accepts_local_checkpoint_with_explicit_base_identity(generator_runtime):
    configuration = _launch_configuration()
    configuration.model_config.model = "/models/llama32-base/snapshots/pinned-revision"
    configuration.model_config.hf_config._name_or_path = "meta-llama/Llama-3.2-1B"

    outcome = generator_runtime.Llama32_1BGenerator.spec_plan(configuration, 2, 3)

    assert isinstance(outcome, generator_runtime.contract.SpecPlan)


@pytest.mark.host
def test_generator_admission_accepts_offline_resolved_base_snapshot(generator_runtime, monkeypatch, tmp_path):
    configuration = _launch_configuration()
    revision = "4e20de362430cd3b72f300e6b0f18e50e7166e08"
    snapshot = tmp_path / "models--meta-llama--Llama-3.2-1B" / "snapshots" / revision
    configuration.model_config.model = str(snapshot)
    configuration.model_config.hf_config._name_or_path = str(snapshot)
    configuration.model_config.revision = revision
    lookups = []

    def cached_config(model, filename, *, revision):
        lookups.append((model, filename, revision))
        return str(snapshot / "config.json")

    monkeypatch.setattr("huggingface_hub.try_to_load_from_cache", cached_config)

    outcome = generator_runtime.Llama32_1BGenerator.spec_plan(configuration, 2, 3)

    assert isinstance(outcome, generator_runtime.contract.SpecPlan)
    assert lookups == [("meta-llama/Llama-3.2-1B", "config.json", revision)]


@pytest.mark.host
@pytest.mark.parametrize("cached", (None, "/cache/base/snapshots/revision/config.json"))
def test_generator_admission_rejects_snapshot_without_matching_base_cache(generator_runtime, monkeypatch, cached):
    configuration = _launch_configuration()
    configuration.model_config.model = "/models/unknown/snapshots/revision"
    configuration.model_config.hf_config._name_or_path = configuration.model_config.model
    monkeypatch.setattr("huggingface_hub.try_to_load_from_cache", lambda *args, **kwargs: cached)

    outcome = generator_runtime.Llama32_1BGenerator.spec_plan(configuration, 2, 3)

    assert isinstance(outcome, generator_runtime.contract.SpecReject)
    assert "base checkpoint" in outcome.reason


@pytest.mark.host
@pytest.mark.parametrize("mesh_device", (None, "N300", "T3K", "(1, 2)", "invalid"))
def test_generator_admission_rejects_unqualified_mesh(generator_runtime, monkeypatch, mesh_device):
    if mesh_device is None:
        monkeypatch.delenv("MESH_DEVICE")
    else:
        monkeypatch.setenv("MESH_DEVICE", mesh_device)
    outcome = generator_runtime.Llama32_1BGenerator.spec_plan(_launch_configuration(), 2, 3)
    assert isinstance(outcome, generator_runtime.contract.SpecReject)
    assert "1x1 MESH_DEVICE" in outcome.reason


@pytest.mark.host
@pytest.mark.parametrize(
    ("batch", "drafts", "reason"),
    [
        (3, 1, "max_num_seqs"),
        (8, 1, "max_num_seqs"),
        (True, 1, "max_num_seqs"),
        (1, 0, "K="),
        (1, 4, "K="),
        (1, True, "K="),
    ],
)
def test_generator_admission_rejects_unsupported_capacity(generator_runtime, batch, drafts, reason):
    outcome = generator_runtime.Llama32_1BGenerator.spec_plan(_launch_configuration(), batch, drafts)

    assert isinstance(outcome, generator_runtime.contract.SpecReject)
    assert reason in outcome.reason


@pytest.mark.host
@pytest.mark.parametrize(
    ("section", "name", "value", "reason"),
    [
        ("scheduler_config", "async_scheduling", True, "synchronous"),
        ("cache_config", "enable_prefix_caching", True, "prefix caching"),
        ("speculative_config", "method", "custom_class", "ngram"),
        ("parallel_config", "data_parallel_size", 2, "one model instance"),
    ],
)
def test_generator_admission_rejects_unqualified_execution_modes(generator_runtime, section, name, value, reason):
    configuration = _launch_configuration()
    setattr(getattr(configuration, section), name, value)

    outcome = generator_runtime.Llama32_1BGenerator.spec_plan(configuration, 2, 3)

    assert isinstance(outcome, generator_runtime.contract.SpecReject)
    assert reason in outcome.reason


@pytest.mark.host
@pytest.mark.parametrize(
    ("tt_config", "reason"),
    [
        ({}, "trace_mode"),
        ({"trace_mode": "all"}, "trace_mode"),
        ({"trace_mode": "decode_only"}, "trace_mode"),
        ({"trace_mode": "none"}, "optimizations"),
        ({"trace_mode": "none", "optimizations": "accuracy"}, "optimizations"),
        ({"trace_mode": "none", "optimizations": "performance", "sample_on_device_mode": "all"}, "host sampling"),
        (
            {"trace_mode": "none", "optimizations": "performance", "sample_on_device_mode": "decode_only"},
            "host sampling",
        ),
    ],
)
def test_generator_admission_rejects_unqualified_trace_sampling_and_lane_configuration(
    generator_runtime, tt_config, reason
):
    configuration = _launch_configuration()
    configuration.additional_config = {"tt": tt_config}

    outcome = generator_runtime.Llama32_1BGenerator.spec_plan(configuration, 2, 3)

    assert isinstance(outcome, generator_runtime.contract.SpecReject)
    assert reason in outcome.reason


@pytest.mark.host
def test_generator_admission_rejects_resolved_multiple_lanes(generator_runtime):
    configuration = _launch_configuration()
    configuration.additional_config["_tt_resolved_lane_count"] = 2

    outcome = generator_runtime.Llama32_1BGenerator.spec_plan(configuration, 2, 3)

    assert isinstance(outcome, generator_runtime.contract.SpecReject)
    assert "one model instance" in outcome.reason


@pytest.mark.host
@pytest.mark.parametrize("read_from_device", (False, True))
def test_generator_decode_dispatches_real_helper_and_real_verify_output(generator_runtime, read_from_device):
    generator, target = _generator_with_real_adapter(generator_runtime)
    arguments = _generator_verify_arguments()
    arguments["read_from_device"] = read_from_device
    arguments["slot_remap"] = torch.tensor([1, 0], dtype=torch.int32)
    kv_cache = object()

    outcome = generator.decode_forward(**arguments, kv_cache=kv_cache, reset_batch=True)

    assert isinstance(outcome, generator_runtime.contract.VerifyOutput)
    assert outcome.spec_mode == "logits"
    assert outcome.logits.shape == (2, 4, 19)
    assert len(target.calls) == 4
    expected_anchor = _target_logits(arguments["tokens"][:, 0], arguments["start_pos"][:, 0], 19)
    torch.testing.assert_close(outcome.logits[:, :1], expected_anchor, rtol=0, atol=0)
    for column, call in enumerate(target.calls):
        assert call["execution"] is target.eager_execution
        assert call["kv_cache"] is kv_cache
        assert call["sampling_params"] is None
        assert call["reset_batch"] is True
        assert call["read_from_device"] is True
        assert call["tokens"].dtype == torch.long
        assert call["start_pos"].dtype == torch.long
        assert call["page_table"].dtype == torch.int32
        assert torch.equal(call["tokens"], arguments["tokens"][:, column])
        assert torch.equal(call["start_pos"], arguments["start_pos"][:, column])
        assert "slot_remap" not in call
        assert "accepted_counts" not in call
        assert "num_valid_drafts" not in call
        assert "spec_mode" not in call
    target.output.fill_(-9999)
    torch.testing.assert_close(outcome.logits[:, :1], expected_anchor, rtol=0, atol=0)


@pytest.mark.host
def test_generator_ordinary_decode_retains_normalization_delegation_and_output_identity(generator_runtime):
    generator, target = _generator_with_real_adapter(generator_runtime)
    tokens = torch.tensor([[1], [2]], dtype=torch.int32)
    positions = torch.tensor([3, 7], dtype=torch.int32)
    page_table = torch.arange(8, dtype=torch.int64).repeat(2, 1)
    kv_cache = object()
    slot_remap = torch.tensor([1, 0], dtype=torch.int32)
    prompt_tokens = [[1, 2], [3, 4]]
    output_tokens = [[5], [6]]

    outcome = generator.decode_forward(
        tokens=tokens,
        start_pos=positions,
        page_table=page_table,
        enable_trace=False,
        kv_cache=kv_cache,
        slot_remap=slot_remap,
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
    )

    assert outcome is target.returns[0]
    assert len(target.calls) == 1
    call = target.calls[0]
    assert call["execution"] is target.eager_execution
    assert call["kv_cache"] is kv_cache
    assert call["slot_remap"] is slot_remap
    assert call["prompt_tokens"] is prompt_tokens
    assert call["output_tokens"] is output_tokens
    assert call["tokens"].shape == (2,)
    assert call["tokens"].dtype == torch.long
    assert call["start_pos"].dtype == torch.long
    assert call["page_table"].dtype == torch.int32
    assert not isinstance(outcome, generator_runtime.contract.VerifyOutput)


@pytest.mark.host
@pytest.mark.parametrize(
    ("changes", "error", "message"),
    [
        ({"spec_mode": "argmax_ids"}, ValueError, "spec_mode"),
        ({"spec_mode": "fused_sample"}, ValueError, "spec_mode"),
        ({"enable_trace": True}, ValueError, "eager execution"),
        ({"sampling_params": object()}, ValueError, "host logits"),
        ({"read_from_device": "false"}, TypeError, "read_from_device"),
        ({"num_valid_drafts": None}, TypeError, "num_valid_drafts"),
        ({"accepted_counts": None}, TypeError, "accepted_counts"),
        ({"num_valid_drafts": None, "accepted_counts": None}, TypeError, "num_valid_drafts"),
        ({"spec_mode": None}, ValueError, "require spec_mode"),
        ({"reload_inputs": True}, TypeError, "reload_inputs"),
    ],
)
def test_generator_rejects_invalid_speculative_call_before_target_execution(generator_runtime, changes, error, message):
    generator, target = _generator_with_real_adapter(generator_runtime)
    arguments = _generator_verify_arguments() | changes
    originals = {name: value.clone() for name, value in arguments.items() if isinstance(value, torch.Tensor)}

    with pytest.raises(error, match=message):
        generator.decode_forward(**arguments)

    assert target.calls == []
    for name, original in originals.items():
        assert torch.equal(arguments[name], original), name


@pytest.mark.host
def test_generator_rejects_malformed_later_candidate_before_target_execution(generator_runtime):
    generator, target = _generator_with_real_adapter(generator_runtime)
    arguments = _generator_verify_arguments()
    arguments["tokens"][1, -1] = 19

    with pytest.raises(ValueError, match="valid tokens"):
        generator.decode_forward(**arguments)

    assert target.calls == []


@pytest.mark.host
def test_generator_rejects_multiple_target_devices_before_execution(generator_runtime):
    generator, target = _generator_with_real_adapter(generator_runtime)
    target.model.config.num_devices = 2
    with pytest.raises(ValueError, match="single-device target"):
        generator.decode_forward(**_generator_verify_arguments())
    assert target.calls == []
