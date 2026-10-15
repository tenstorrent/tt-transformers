# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Check pressure references against exact committed resume prefixes on the host."""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / ".github/scripts/qualify_llama32_ngram.py"
_SPEC = importlib.util.spec_from_file_location("llama_ngram_pressure_qualification", _SCRIPT)
qualification = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(qualification)


def _event(record, count, ordinal=0):
    history = record["prompt"] + record["token_ids"][:count]
    return {
        "request_id": record["response"]["id"] + "-0-1234abcd",
        "num_prompt_tokens": len(record["prompt"]),
        "num_output_tokens": count,
        "num_computed_tokens": len(history) - 1,
        "all_token_ids": history,
        "post_all_token_ids": history.copy(),
        "drop_stale_output": False,
        "num_preemptions": ordinal,
        "num_in_flight_tokens": 0,
        "num_output_placeholders": 0,
        "num_stale_output_tokens": 0,
        "post_status": "PREEMPTED",
        "post_num_computed_tokens": 0,
        "post_num_preemptions": ordinal + 1,
    }


@pytest.fixture
def pressure():
    prompts = [[11, 12, 13], [21, 22, 23], [31, 32, 33], [41, 42, 43]]
    values = [
        [-2.0] * 40 + [-2.5] * 40 + [-3.0] * 80,
        [-4.0] * 160,
        [-5.0] * 60 + [-6.0] * 100,
        [-7.0] * 160,
    ]
    records = []
    for row, (prompt, logprobs) in enumerate(zip(prompts, values)):
        records.append(
            {
                "prompt": prompt,
                "controls": {
                    "max_tokens": 160,
                    "temperature": 0.7,
                    "allowed_token_ids": [71 + row],
                    "seed": 901 + row,
                    "logprobs": 1,
                    "return_tokens_as_token_ids": True,
                },
                "token_ids": [71 + row] * 160,
                "logprobs": logprobs,
                "response": {"id": f"cmpl-pressure-{row}"},
            }
        )
    return {
        "records": records,
        "events": [
            _event(records[0], 40),
            _event(records[0], 40, 1),
            _event(records[2], 60),
            _event(records[0], 80, 2),
        ],
        "preemptions": 4,
    }


@pytest.mark.host
def test_pressure_segments_cover_every_output_once_and_retain_repeated_preemptions(pressure):
    original = copy.deepcopy(pressure)
    segments = qualification.pressure_segments(pressure["records"], pressure["events"], 4, 3)
    assert segments == [
        {"request_index": 0, "start": 0, "stop": 40},
        {"request_index": 0, "start": 40, "stop": 80},
        {"request_index": 0, "start": 80, "stop": 160},
        {"request_index": 1, "start": 0, "stop": 160},
        {"request_index": 2, "start": 0, "stop": 60},
        {"request_index": 2, "start": 60, "stop": 160},
        {"request_index": 3, "start": 0, "stop": 160},
    ]
    for row in range(4):
        covered = [
            output for item in segments if item["request_index"] == row for output in range(item["start"], item["stop"])
        ]
        assert covered == list(range(160))
    assert pressure == original
    assert len(pressure["events"]) == pressure["preemptions"] == 4


@pytest.mark.parametrize("count", [0, -1, True, 3, 4.5, float("nan"), float("inf")])
@pytest.mark.host
def test_pressure_segments_require_one_event_for_each_measured_preemption(pressure, count):
    with pytest.raises(AssertionError):
        qualification.pressure_segments(pressure["records"], pressure["events"], count, 3)


@pytest.mark.parametrize(
    "field,value",
    [
        ("num_prompt_tokens", 4),
        ("num_output_tokens", -1),
        ("num_output_tokens", 160),
        ("num_output_tokens", 40.0),
        ("num_output_tokens", True),
        ("num_computed_tokens", -1),
        ("num_preemptions", -1),
        ("num_preemptions", True),
        ("num_in_flight_tokens", 1),
        ("num_output_placeholders", 1),
        ("num_stale_output_tokens", 1),
        ("num_stale_output_tokens", False),
        ("post_status", "RUNNING"),
        ("post_num_computed_tokens", 1),
        ("post_num_preemptions", 3),
        ("post_num_preemptions", 1.0),
        ("drop_stale_output", True),
        ("drop_stale_output", 0),
    ],
)
@pytest.mark.host
def test_pressure_segments_reject_invalid_or_async_backend_metadata(pressure, field, value):
    pressure["events"][0][field] = value
    with pytest.raises(AssertionError):
        qualification.pressure_segments(pressure["records"], pressure["events"], 4, 3)


@pytest.mark.parametrize("failure", ["missing_anchor", "wrong_token", "post_history", "unrelated_id", "partial_id"])
@pytest.mark.host
def test_pressure_segments_require_exact_committed_history_and_request_identity(pressure, failure):
    event = pressure["events"][0]
    if failure == "missing_anchor":
        event["all_token_ids"] = [11, 12, 13] + [71] * 39
    elif failure == "wrong_token":
        event["all_token_ids"][-1] = 72
    elif failure == "post_history":
        event["post_all_token_ids"][-1] = 72
    elif failure == "unrelated_id":
        event["request_id"] = "cmpl-some-other-pressure-0-1234abcd"
    else:
        event["request_id"] += "-extra"
    with pytest.raises(AssertionError):
        qualification.pressure_segments(pressure["records"], pressure["events"], 4, 3)


@pytest.mark.host
def test_pressure_segments_reject_ambiguous_http_response_ids(pressure):
    pressure["records"][1]["response"]["id"] = pressure["records"][0]["response"]["id"]
    with pytest.raises(AssertionError):
        qualification.pressure_segments(pressure["records"], pressure["events"], 4, 3)


@pytest.mark.parametrize("failure", ["decreasing_history", "counter_gap", "duplicate_counter"])
@pytest.mark.host
def test_pressure_segments_require_per_request_scheduler_order(pressure, failure):
    if failure == "decreasing_history":
        pressure["events"][1] = _event(pressure["records"][0], 39, 1)
    elif failure == "counter_gap":
        pressure["events"][1]["num_preemptions"] = 2
        pressure["events"][1]["post_num_preemptions"] = 3
    else:
        pressure["events"][1]["num_preemptions"] = 0
        pressure["events"][1]["post_num_preemptions"] = 1
    with pytest.raises(AssertionError):
        qualification.pressure_segments(pressure["records"], pressure["events"], 4, 3)


@pytest.mark.host
def test_pressure_segments_reject_preemption_before_any_output(pressure):
    events = [_event(pressure["records"][0], 0)]
    with pytest.raises(AssertionError):
        qualification.pressure_segments(pressure["records"], events, 1, 3)


@pytest.mark.parametrize("k,start", [(0, 157), (3, 148)])
@pytest.mark.host
def test_pressure_resume_must_cover_prefill_and_three_later_submissions(pressure, k, start):
    with pytest.raises(AssertionError):
        qualification.pressure_segments(pressure["records"], [_event(pressure["records"][0], start)], 1, k)


@pytest.mark.parametrize("k,start", [(0, 156), (3, 147)])
@pytest.mark.host
def test_pressure_resume_accepts_exact_minimum_post_resume_coverage(pressure, k, start):
    segments = qualification.pressure_segments(pressure["records"], [_event(pressure["records"][0], start)], 1, k)
    assert {"request_index": 0, "start": start, "stop": 160} in segments


@pytest.fixture
def reference_harness(tmp_path, monkeypatch):
    args = SimpleNamespace(output=tmp_path / "qualification", port=8123)
    expected = [
        ([11, 12, 13], 71, 40, -2.0, 901),
        ([11, 12, 13] + [71] * 40, 71, 40, -2.5, 901),
        ([11, 12, 13] + [71] * 80, 71, 80, -3.0, 901),
        ([21, 22, 23], 72, 160, -4.0, 902),
        ([31, 32, 33], 73, 60, -5.0, 903),
        ([31, 32, 33] + [73] * 60, 73, 100, -6.0, 903),
        ([41, 42, 43], 74, 160, -7.0, 904),
    ]
    state = {"launches": [], "calls": [], "closed": False, "reference_preemptions": 0}

    @contextmanager
    def server(server_args, batch, k):
        state["launches"].append((server_args.output, batch, k))
        directory = server_args.output / f"B{batch}-K{k}"
        directory.mkdir(parents=True)
        qualification.reset_reference_comparisons(directory)
        state["directory"] = directory
        try:
            yield directory, directory / "server.log"
        finally:
            state["closed"] = True

    class Client:
        def __init__(self, url, model):
            assert url == "http://127.0.0.1:8123" and model == "meta-llama/Llama-3.2-1B"

        def metrics(self):
            return SimpleNamespace(preemptions=state["reference_preemptions"] if state["calls"] else 0)

        def complete(self, prompt, **controls):
            wanted_prompt, token, count, logprob, seed = expected[len(state["calls"])]
            assert prompt == wanted_prompt
            assert controls == {
                "max_tokens": count,
                "temperature": 0.7,
                "allowed_token_ids": [token],
                "seed": seed,
                "logprobs": 1,
                "return_tokens_as_token_ids": True,
            }
            state["calls"].append((prompt.copy(), controls.copy()))
            return SimpleNamespace(
                status=200,
                token_ids=[token] * count,
                completion_tokens=count,
                finish_reason="length",
                body={"choices": [{"logprobs": {"token_logprobs": [logprob] * count}}]},
            )

    def assert_full_length(result, count):
        assert result.status == 200
        assert result.completion_tokens == len(result.token_ids) == count
        assert result.finish_reason == "length"

    def delta(before, after, k):
        assert k == 1
        return SimpleNamespace(preemptions=after.preemptions - before.preemptions)

    monkeypatch.setattr(qualification, "server", server)
    return args, (Client, delta, assert_full_length, Mock()), state


@pytest.mark.parametrize("k", [0, 3])
@pytest.mark.host
def test_pressure_reference_replays_entire_prefix_including_anchor_on_ordinary_same_batch(
    pressure, reference_harness, k
):
    args, helpers, state = reference_harness
    original_output = args.output
    qualification.exercise_pressure_reference(args, 4, k, pressure, helpers)
    assert args.output == original_output
    assert state["launches"] == [(original_output / f"B4-K{k}-pressure-reference", 4, 0)]
    assert len(state["calls"]) == 7
    resumed_prompt, controls = state["calls"][1]
    assert resumed_prompt == [11, 12, 13] + [71] * 40
    assert resumed_prompt[-1] == pressure["records"][0]["token_ids"][39]
    assert len(resumed_prompt) == 43 and controls["max_tokens"] == 40
    comparisons = list((state["directory"] / "comparisons").glob("*.json"))
    assert len(comparisons) == 7
    assert all(json.loads(path.read_text())["max_abs_difference"] == 0 for path in comparisons)
    saved = json.loads((state["directory"] / "results.json").read_text())
    assert saved["pressure"]["events"] == pressure["events"]
    assert len(saved["segments"]) == 7 and saved["pressure"]["preemptions"] == 4
    assert state["closed"]


@pytest.mark.host
def test_pressure_reference_rejects_any_reference_preemption(pressure, reference_harness):
    args, helpers, state = reference_harness
    state["reference_preemptions"] = 1
    with pytest.raises(AssertionError):
        qualification.exercise_pressure_reference(args, 4, 3, pressure, helpers)
    assert len(state["calls"]) == 7
    assert not (state["directory"] / "results.json").exists()
    assert state["closed"]


@pytest.mark.host
def test_corrupt_resumed_logprob_fails_strict_comparison_and_preserves_evidence(pressure, reference_harness):
    args, helpers, state = reference_harness
    pressure["records"][0]["logprobs"][40] = -2.503
    with pytest.raises(AssertionError, match="raw logprobs differ"):
        qualification.exercise_pressure_reference(args, 4, 3, pressure, helpers)
    artifact = state["directory"] / "comparisons/pressure-request-0-outputs-40-to-80.json"
    saved = json.loads(artifact.read_text())
    assert saved["actual"]["token_ids"] == saved["expected"]["token_ids"] == [71] * 40
    assert saved["actual"]["logprobs"][0] == -2.503
    assert saved["expected"]["logprobs"][0] == -2.5
    assert saved["max_abs_difference"] == pytest.approx(0.003)
    assert saved["expected"]["prompt"] == [11, 12, 13] + [71] * 40
    assert not (state["directory"] / "results.json").exists()
    assert state["closed"]


@pytest.mark.host
def test_qualify_http_cannot_return_pressure_success_before_reference_validation(pressure, monkeypatch):
    args, tokenizer, helpers, ordinary = object(), object(), object(), object()
    response = {"pressure": pressure}
    execute = Mock(return_value=response)
    failure = AssertionError("resumed reference differs")
    validate = Mock(side_effect=failure)
    monkeypatch.setattr(qualification, "exercise_http", execute)
    monkeypatch.setattr(qualification, "exercise_pressure_reference", validate)
    with pytest.raises(AssertionError) as raised:
        qualification.qualify_http(args, 4, 3, tokenizer, helpers, ordinary)
    assert raised.value is failure
    execute.assert_called_once_with(args, 4, 3, tokenizer, helpers, ordinary)
    validate.assert_called_once_with(args, 4, 3, pressure, helpers)


@pytest.mark.host
def test_qualify_http_returns_after_reference_gate_and_skips_absent_pressure(pressure, monkeypatch):
    args, tokenizer, helpers = object(), object(), object()
    response = {"pressure": pressure}
    validate = Mock()
    monkeypatch.setattr(qualification, "exercise_http", Mock(return_value=response))
    monkeypatch.setattr(qualification, "exercise_pressure_reference", validate)
    assert qualification.qualify_http(args, 4, 0, tokenizer, helpers) is response
    validate.assert_called_once_with(args, 4, 0, pressure, helpers)
    response["pressure"] = None
    validate.reset_mock()
    assert qualification.qualify_http(args, 2, 0, tokenizer, helpers) is response
    validate.assert_not_called()


@pytest.mark.host
def test_main_cannot_write_full_success_when_pressure_reference_gate_fails(tmp_path, pressure, monkeypatch):
    modules = {}
    for name in ("ttnn", "ttnn.device", "vllm", "tt_transformers", "vllm_tt_plugin"):
        module = ModuleType(name)
        module.__file__ = "/test/source.py"
        module.__version__ = "test"
        modules[name] = module
    platform = ModuleType("vllm.platforms")
    platform.current_platform = SimpleNamespace(device_type="tt")
    hub = ModuleType("huggingface_hub")
    hub.try_to_load_from_cache = Mock()
    transformers = ModuleType("transformers")
    transformers.AutoTokenizer = SimpleNamespace(from_pretrained=Mock(return_value=object()))
    spec_client = ModuleType("spec_client")
    for name in ("SpecServer", "acceptance_delta", "assert_full_length_completion", "widest_decode_batch_size"):
        setattr(spec_client, name, Mock())
    for module in (platform, hub, transformers, spec_client):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(qualification.importlib, "import_module", lambda name: modules[name])
    monkeypatch.setattr(qualification.importlib.metadata, "version", lambda name: "test")
    monkeypatch.setattr(qualification, "git_revision", lambda path: "a" * 40)
    for name in ("source_provenance", "package_source", "runtime_provenance"):
        monkeypatch.setattr(qualification, name, Mock(return_value={}))
    monkeypatch.setattr(qualification, "require_cached_checkpoint", Mock())
    execute = Mock(side_effect=lambda args, batch, *rest: {"pressure": pressure if batch == 4 else None})
    monkeypatch.setattr(qualification, "exercise_http", execute)
    failure = AssertionError("pressure reference gate failed")
    validate = Mock(side_effect=failure)
    monkeypatch.setattr(qualification, "exercise_pressure_reference", validate)
    device_run = Mock(side_effect=AssertionError("device work must not start after failed ordinary qualification"))
    monkeypatch.setattr(qualification.subprocess, "run", device_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(_SCRIPT),
            "--plugin-root",
            str(tmp_path / "plugin"),
            "--plugin-revision",
            "a" * 40,
            "--checkpoint-revision",
            "b" * 40,
            "--output",
            str(tmp_path / "evidence"),
        ],
    )
    with pytest.raises(AssertionError) as raised:
        qualification.main()
    assert raised.value is failure
    assert execute.call_count == 3 and validate.call_count == 1
    device_run.assert_not_called()
    assert not (tmp_path / "evidence/success.json").exists()
    assert not (tmp_path / "evidence/partial.json").exists()
