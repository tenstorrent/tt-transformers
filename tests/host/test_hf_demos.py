# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Run each text demo through only the public tokenizer/generate/cleanup API."""

from contextlib import contextmanager
from importlib import import_module
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import ttnn
from examples.common.runtime import UnsupportedConfiguration, mesh_device_parameters_from_env

_FAMILIES = (
    "llama3_8b",
    "llama32_1b",
    "llama32_3b",
    "llama33_70b",
    "qwen2_7b",
    "qwen25_7b",
    "qwen25_72b",
    "qwen25_coder_32b",
    "qwen3_32b",
    "deepseek_r1_distill_qwen_14b",
    "mistral_7b",
    "phi4",
)


class _PublicModel:
    """A demo cannot reach executor, tensor-model, or synthetic device attributes."""

    __slots__ = ("tokenizer", "generate", "cleanup")

    def __init__(self, tokenizer, generate, cleanup):
        self.tokenizer = tokenizer
        self.generate = generate
        self.cleanup = cleanup


@pytest.fixture(params=_FAMILIES)
def demo_case(request):
    family = request.param
    demo = import_module(f"examples.{family}.demo")
    factory = import_module(f"tt_transformers.models.{family}.hf_generator").from_pretrained
    return SimpleNamespace(family=family, demo=demo, factory=factory)


@pytest.fixture
def public_flow(demo_case, monkeypatch):
    monkeypatch.delenv("HF_MODEL", raising=False)
    events = []
    inputs = {
        "input_ids": torch.tensor([[17, 23, 41]], dtype=torch.long),
        "attention_mask": torch.ones((1, 3), dtype=torch.long),
    }
    continuation = torch.tensor([[53, 59]], dtype=torch.long)
    outputs = torch.cat((inputs["input_ids"], continuation), dim=1)

    def tokenize(*args, **kwargs):
        events.append("tokenize")
        return inputs

    def generate(**kwargs):
        events.append("generate")
        return outputs

    def decode(*args, **kwargs):
        events.append("decode")
        return ["Generated answer."]

    tokenizer = SimpleNamespace(apply_chat_template=Mock(side_effect=tokenize), batch_decode=Mock(side_effect=decode))
    model = _PublicModel(tokenizer, Mock(side_effect=generate), Mock(side_effect=lambda: events.append("cleanup")))
    loader = Mock(return_value=model)
    monkeypatch.setattr(demo_case.demo, "from_pretrained", loader)
    return SimpleNamespace(
        case=demo_case,
        model=model,
        loader=loader,
        inputs=inputs,
        continuation=continuation,
        events=events,
        mesh=object(),
    )


@pytest.mark.host
def test_demo_imports_the_actual_public_factory(demo_case):
    assert demo_case.demo.from_pretrained is demo_case.factory


@pytest.mark.host
def test_run_uses_cpu_chat_inputs_and_decodes_only_continuation(public_flow):
    flow = public_flow
    messages = [{"role": "user", "content": "Explain paged attention."}]
    original_input = flow.inputs["input_ids"].clone()

    result = flow.case.demo.run(
        flow.mesh, messages, hf_model="organization/checkpoint", max_new_tokens=7, max_seq_len=256
    )

    assert result == "Generated answer."
    flow.loader.assert_called_once_with(
        flow.mesh, hf_model="organization/checkpoint", max_batch_size=1, max_seq_len=256
    )
    flow.model.tokenizer.apply_chat_template.assert_called_once_with(
        messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
    )
    flow.model.generate.assert_called_once()
    generation_options = flow.model.generate.call_args.kwargs
    assert set(generation_options) == {"input_ids", "attention_mask", "max_new_tokens", "do_sample"}
    assert generation_options["input_ids"] is flow.inputs["input_ids"]
    assert generation_options["attention_mask"] is flow.inputs["attention_mask"]
    assert generation_options["input_ids"].device.type == "cpu"
    assert generation_options["max_new_tokens"] == 7
    assert generation_options["do_sample"] is False
    flow.model.tokenizer.batch_decode.assert_called_once()
    decoded = flow.model.tokenizer.batch_decode.call_args
    assert len(decoded.args) == 1 and torch.equal(decoded.args[0], flow.continuation)
    assert decoded.kwargs == {"skip_special_tokens": True}
    assert torch.equal(flow.inputs["input_ids"], original_input)
    assert flow.events == ["tokenize", "generate", "decode", "cleanup"]
    flow.model.cleanup.assert_called_once_with()


@pytest.mark.host
def test_run_preserves_the_public_factory_default_checkpoint(public_flow):
    flow = public_flow
    expected_default = import_module(f"tt_transformers.models.{flow.case.family}.hf_generator").DEFAULT_HF_MODEL
    assert flow.case.demo.run(flow.mesh, [{"role": "user", "content": "Hello"}]) == "Generated answer."
    assert flow.loader.call_args.kwargs["hf_model"] == expected_default
    assert flow.model.generate.call_args.kwargs["max_new_tokens"] == 40
    assert flow.loader.call_args.kwargs["max_seq_len"] == 2048


@pytest.mark.host
@pytest.mark.parametrize("override", [None, "organization/explicit-checkpoint"])
def test_checkpoint_argument_takes_precedence_over_environment(public_flow, monkeypatch, override):
    flow = public_flow
    monkeypatch.setenv("HF_MODEL", "organization/environment-checkpoint")
    flow.case.demo.run(flow.mesh, [{"role": "user", "content": "Hello"}], hf_model=override)
    assert flow.loader.call_args.kwargs["hf_model"] == (override or "organization/environment-checkpoint")


@pytest.mark.host
@pytest.mark.parametrize("stage", ["tokenize", "generate", "decode"])
def test_run_cleans_the_public_model_when_request_processing_fails(public_flow, stage):
    flow = public_flow
    error = RuntimeError(f"{stage} failed")
    operation = {
        "tokenize": flow.model.tokenizer.apply_chat_template,
        "generate": flow.model.generate,
        "decode": flow.model.tokenizer.batch_decode,
    }[stage]
    operation.side_effect = error

    with pytest.raises(RuntimeError, match=f"{stage} failed") as raised:
        flow.case.demo.run(flow.mesh, [{"role": "user", "content": "Hello"}])

    assert raised.value is error
    flow.model.cleanup.assert_called_once_with()
    assert flow.events[-1] == "cleanup"


@pytest.mark.host
def test_cli_prints_decoded_text_and_closes_mesh_after_model_cleanup(public_flow, monkeypatch, capsys):
    flow = public_flow
    parameters = {"mesh_shape": (1, 1), "trace_region_size": 0, "num_command_queues": 1}
    monkeypatch.setattr(flow.case.demo, "mesh_device_parameters_from_env", lambda: parameters)

    @contextmanager
    def open_mesh(received):
        assert received is parameters
        flow.events.append("mesh-enter")
        try:
            yield flow.mesh
        finally:
            flow.events.append("mesh-exit")

    monkeypatch.setattr(flow.case.demo, "open_mesh_device", open_mesh)
    flow.case.demo.main(
        [
            "--prompt",
            "Explain KV caching.",
            "--hf-model",
            "organization/checkpoint",
            "--max-new-tokens",
            "6",
            "--max-seq-len",
            "512",
        ]
    )

    assert capsys.readouterr().out == "Generated answer.\n"
    flow.loader.assert_called_once_with(
        flow.mesh, hf_model="organization/checkpoint", max_batch_size=1, max_seq_len=512
    )
    assert flow.model.tokenizer.apply_chat_template.call_args.args == (
        [{"role": "user", "content": "Explain KV caching."}],
    )
    assert flow.model.generate.call_args.kwargs["max_new_tokens"] == 6
    assert flow.events == ["mesh-enter", "tokenize", "generate", "decode", "cleanup", "mesh-exit"]


@pytest.mark.host
def test_cli_failure_closes_mesh_after_cleaning_model(public_flow, monkeypatch, capsys):
    flow = public_flow
    monkeypatch.setattr(flow.case.demo, "mesh_device_parameters_from_env", lambda: {})

    @contextmanager
    def open_mesh(_parameters):
        flow.events.append("mesh-enter")
        try:
            yield flow.mesh
        finally:
            flow.events.append("mesh-exit")

    monkeypatch.setattr(flow.case.demo, "open_mesh_device", open_mesh)
    flow.model.generate.side_effect = RuntimeError("generation failed")
    with pytest.raises(RuntimeError, match="generation failed"):
        flow.case.demo.main(["--prompt", "Hello"])
    assert flow.events == ["mesh-enter", "tokenize", "cleanup", "mesh-exit"]
    assert capsys.readouterr().out == ""
    flow.model.cleanup.assert_called_once_with()


@pytest.mark.host
@pytest.mark.parametrize(
    "selection,shape,fabric",
    [
        ("N150", (1, 1), None),
        ("N300", (1, 2), ttnn.FabricConfig.FABRIC_1D),
        ("T3K", (1, 8), ttnn.FabricConfig.FABRIC_1D_RING),
        ("P150", (1, 1), None),
        ("P300", (1, 2), ttnn.FabricConfig.FABRIC_1D_RING),
        (" p150x4 ", (1, 4), ttnn.FabricConfig.FABRIC_1D_RING),
        ("TG", (4, 8), ttnn.FabricConfig.FABRIC_1D),
    ],
)
def test_mesh_selection_resolves_shape_and_fabric_without_opening_hardware(monkeypatch, selection, shape, fabric):
    monkeypatch.setenv("MESH_DEVICE", selection)
    parameters = mesh_device_parameters_from_env()
    assert parameters["mesh_shape"] == shape
    assert parameters["trace_region_size"] == 0
    assert parameters["num_command_queues"] == 1
    if fabric is None:
        assert "fabric_config" not in parameters
    else:
        assert parameters["fabric_config"] == fabric


@pytest.mark.host
@pytest.mark.parametrize("selection", [None, "", "unsupported"])
def test_mesh_selection_requires_a_supported_explicit_environment(monkeypatch, selection):
    if selection is None:
        monkeypatch.delenv("MESH_DEVICE", raising=False)
    else:
        monkeypatch.setenv("MESH_DEVICE", selection)
    with pytest.raises(UnsupportedConfiguration, match="Set MESH_DEVICE"):
        mesh_device_parameters_from_env()
