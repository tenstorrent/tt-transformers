"""Host-only contracts for immutable default Hugging Face checkpoint revisions."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
CHECKPOINTS = {
    "llama32_1b": ("meta-llama/Llama-3.2-1B-Instruct", "9213176726f574b556790deb65791e0c5aa438b6"),
    "llama32_3b": ("meta-llama/Llama-3.2-3B-Instruct", "0cb88a4f764b7a12671c53f0838cd831a0843b95"),
    "llama33_70b": ("meta-llama/Llama-3.3-70B-Instruct", "6f6073b423013f6a7d4d9f39144961bfbfbc386b"),
    "llama3_8b": ("meta-llama/Llama-3.1-8B-Instruct", "0e9e39f249a16976918f6564b8830bc894c89659"),
    "mistral_7b": ("mistralai/Mistral-7B-Instruct-v0.3", "c170c708c41dac9275d15a8fff4eca08d52bab71"),
    "qwen25_7b": ("Qwen/Qwen2.5-7B-Instruct", "a09a35458c702b33eeacc393d103063234e8bc28"),
    "qwen2_7b": ("Qwen/Qwen2-7B-Instruct", "f2826a00ceef68f0f2b946d945ecc0477ce4450c"),
}


def _adaptor(model: str):
    return importlib.import_module(f"tt_transformers.models.{model}.hf_adaptor")


@pytest.mark.host
@pytest.mark.parametrize("model", CHECKPOINTS)
def test_default_revision_is_exact_and_custom_model_ids_remain_unforced(model):
    adaptor = _adaptor(model)
    default_model, default_revision = CHECKPOINTS[model]
    assert adaptor.DEFAULT_HF_MODEL == default_model
    assert adaptor.DEFAULT_HF_REVISION == default_revision
    assert adaptor._resolve_hf_revision(default_model, None) == default_revision
    assert adaptor._resolve_hf_revision("custom/compatible-model", None) is None
    assert adaptor._resolve_hf_revision("custom/compatible-model", "custom-revision") == "custom-revision"


@pytest.mark.host
@pytest.mark.parametrize("model", CHECKPOINTS)
def test_tokenizer_loader_threads_default_custom_and_explicit_revisions(monkeypatch, model):
    adaptor = _adaptor(model)
    default_model, default_revision = CHECKPOINTS[model]
    calls = []

    class Tokenizer:
        eos_token_id = 1
        stop_tokens = None

        @staticmethod
        def convert_tokens_to_ids(_token):
            return 2

    def fake_from_pretrained(hf_model, **kwargs):
        calls.append((hf_model, kwargs))
        return Tokenizer()

    monkeypatch.setattr(adaptor.AutoTokenizer, "from_pretrained", fake_from_pretrained)
    adaptor.load_tokenizer(default_model)
    adaptor.load_tokenizer("custom/compatible-model")
    adaptor.load_tokenizer("custom/compatible-model", hf_revision="custom-revision")

    assert [(hf_model, kwargs["revision"]) for hf_model, kwargs in calls] == [
        (default_model, default_revision),
        ("custom/compatible-model", None),
        ("custom/compatible-model", "custom-revision"),
    ]


@pytest.mark.host
@pytest.mark.parametrize("model", CHECKPOINTS)
@pytest.mark.parametrize(
    ("requested_model", "requested_revision", "expected_revision"),
    (
        (None, None, "default"),
        ("custom/compatible-model", None, None),
        ("custom/compatible-model", "custom-revision", "custom-revision"),
    ),
)
def test_config_and_model_hf_calls_receive_the_resolved_revision(
    monkeypatch,
    model,
    requested_model,
    requested_revision,
    expected_revision,
):
    adaptor = _adaptor(model)
    default_model, default_revision = CHECKPOINTS[model]
    expected_model = requested_model or default_model
    expected_revision = default_revision if expected_revision == "default" else expected_revision
    monkeypatch.delenv("HF_MODEL", raising=False)

    class StopAtHfCall(RuntimeError):
        pass

    seen = []

    def config_loader(hf_model, **kwargs):
        seen.append(("config", hf_model, kwargs.get("revision")))
        if model == "llama3_8b":
            raise StopAtHfCall
        return SimpleNamespace(num_attention_heads=8, num_key_value_heads=8)

    def model_loader(hf_model, **kwargs):
        seen.append(("model", hf_model, kwargs.get("revision")))
        raise StopAtHfCall

    monkeypatch.setattr(adaptor.AutoConfig, "from_pretrained", config_loader)
    monkeypatch.setattr(adaptor.AutoModelForCausalLM, "from_pretrained", model_loader)
    if hasattr(adaptor, "_validate_checkpoint_config"):
        monkeypatch.setattr(adaptor, "_validate_checkpoint_config", lambda _config: None)
    if model == "llama33_70b":
        monkeypatch.setattr(adaptor, "_resolve_supported_sku", lambda **_kwargs: "P150x4")
        monkeypatch.setattr(adaptor, "_llama33_70b_ccl_topology", lambda _mesh: None)
        monkeypatch.setattr(adaptor.ttnn, "cluster", SimpleNamespace(get_cluster_type=lambda: "test-cluster"))

    mesh = SimpleNamespace(
        get_num_devices=lambda: 4 if model == "llama33_70b" else 2,
        arch=lambda: "test-arch",
        shape=(1, 2),
    )
    kwargs = {"max_batch_size": 1, "max_seq_len": 128}
    if requested_model is not None:
        kwargs["hf_model"] = requested_model
    if requested_revision is not None:
        kwargs["hf_revision"] = requested_revision

    with pytest.raises(StopAtHfCall):
        adaptor.from_pretrained(mesh, **kwargs)
    assert seen[0] == ("config", expected_model, expected_revision)

    if model == "llama3_8b":
        seen.clear()
        converted_kwargs = {
            "head_dim": 1,
            "n_heads": 1,
            "n_kv_heads": 1,
            "n_layers": 1,
        }
        if requested_revision is not None:
            converted_kwargs["hf_revision"] = requested_revision
        with pytest.raises(StopAtHfCall):
            adaptor.load_converted_state_dict(expected_model, **converted_kwargs)
    assert seen[-1] == ("model", expected_model, expected_revision)


def _call_name(node: ast.Call) -> str:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
        return f"{node.func.value.id}.{node.func.attr}"
    return ""


def _uses_revision(call: ast.Call, keyword: str = "revision") -> bool:
    return any(
        item.arg == keyword
        and isinstance(item.value, ast.Name)
        and item.value.id == "hf_revision"
        or item.arg is None
        and isinstance(item.value, ast.Name)
        and item.value.id == "load_kwargs"
        for item in call.keywords
    )


@pytest.mark.host
@pytest.mark.parametrize("model", CHECKPOINTS)
def test_config_model_cache_preflight_and_tokenizer_share_resolved_revision(model):
    path = ROOT / f"src/tt_transformers/models/{model}/hf_adaptor.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    loader = functions["from_pretrained"]
    calls = {_call_name(node): node for node in ast.walk(loader) if isinstance(node, ast.Call)}

    assert any(
        isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "hf_revision" for target in node.targets)
        and isinstance(node.value, ast.Call)
        and _call_name(node.value) == "_resolve_hf_revision"
        for node in loader.body
    )
    assert _uses_revision(calls["AutoConfig.from_pretrained"])
    if model == "llama33_70b":
        load_kwargs = next(
            node.value
            for node in loader.body
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "load_kwargs" for target in node.targets)
        )
        assert isinstance(load_kwargs, ast.Dict)
        assert any(
            isinstance(key, ast.Constant)
            and key.value == "revision"
            and isinstance(value, ast.Name)
            and value.id == "hf_revision"
            for key, value in zip(load_kwargs.keys, load_kwargs.values)
        )
    if model != "llama3_8b":
        assert _uses_revision(calls["AutoModelForCausalLM.from_pretrained"])
    tokenizer_call = calls["load_tokenizer"]
    assert (
        _uses_revision(tokenizer_call, "hf_revision")
        or len(tokenizer_call.args) >= 2
        and isinstance(tokenizer_call.args[1], ast.Name)
        and tokenizer_call.args[1].id == "hf_revision"
    )
    cache_call = calls["_model_cache_path" if model == "llama3_8b" else "_cache_path"]
    assert _uses_revision(cache_call, "hf_revision")

    cache_function = functions["_model_cache_path" if model == "llama3_8b" else "_cache_path"]
    preflight = next(
        node
        for node in ast.walk(cache_function)
        if isinstance(node, ast.Call) and _call_name(node) == "report_model_preflight"
    )
    assert _uses_revision(preflight, "hf_revision")

    if model == "llama3_8b":
        converted_loader = functions["load_converted_state_dict"]
        model_call = next(
            node
            for node in ast.walk(converted_loader)
            if isinstance(node, ast.Call) and _call_name(node) == "AutoModelForCausalLM.from_pretrained"
        )
        assert _uses_revision(model_call)
