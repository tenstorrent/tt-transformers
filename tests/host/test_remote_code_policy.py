# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Host-only checks for the Hugging Face remote-code execution policy."""

from __future__ import annotations

import ast
import importlib
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
QWEN_ADAPTORS = (
    "qwen25_72b",
    "qwen25_coder_32b",
    "qwen3_32b",
)


class _FakeTokenizer:
    eos_token_id = 151645

    def convert_tokens_to_ids(self, token):
        return {"<|im_end|>": 151645, "<|im_start|>": 151644}[token]


def _is_literal_true(node: ast.AST | None) -> bool:
    return isinstance(node, ast.Constant) and node.value is True


@pytest.mark.host
def test_all_production_from_pretrained_calls_forbid_implicit_remote_code():
    calls = []
    forced = []
    for path in sorted((ROOT / "src/tt_transformers").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and (
                (isinstance(node.func, ast.Attribute) and node.func.attr == "from_pretrained")
                or (isinstance(node.func, ast.Name) and node.func.id == "from_pretrained")
            ):
                calls.append((path, node.lineno))
                if any(
                    keyword.arg == "trust_remote_code" and _is_literal_true(keyword.value)
                    for keyword in node.keywords
                ):
                    forced.append((path, node.lineno))
            if isinstance(node, ast.Dict) and any(
                isinstance(key, ast.Constant)
                and key.value == "trust_remote_code"
                and _is_literal_true(value)
                for key, value in zip(node.keys, node.values)
            ):
                forced.append((path, node.lineno))

    assert calls, "the audit must cover the production from_pretrained surface"
    assert forced == []


@pytest.mark.host
def test_transformers_5_12_has_builtin_qwen_loader_registrations():
    import transformers
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES
    from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
    from transformers.models.auto.tokenization_auto import TOKENIZER_MAPPING_NAMES

    assert transformers.__version__ == "5.12.1"
    assert {name: CONFIG_MAPPING_NAMES[name] for name in ("qwen2", "qwen3")} == {
        "qwen2": "Qwen2Config",
        "qwen3": "Qwen3Config",
    }
    assert {name: MODEL_FOR_CAUSAL_LM_MAPPING_NAMES[name] for name in ("qwen2", "qwen3")} == {
        "qwen2": "Qwen2ForCausalLM",
        "qwen3": "Qwen3ForCausalLM",
    }
    assert {name: TOKENIZER_MAPPING_NAMES[name] for name in ("qwen2", "qwen3")} == {
        "qwen2": "Qwen2Tokenizer",
        "qwen3": "Qwen2Tokenizer",
    }


@pytest.mark.host
@pytest.mark.parametrize("model_name", QWEN_ADAPTORS)
def test_qwen_tokenizer_loaders_do_not_opt_in_by_default(monkeypatch, model_name):
    adaptor = importlib.import_module(f"tt_transformers.models.{model_name}.hf_adaptor")
    observed = []

    def fake_from_pretrained(*args, **kwargs):
        observed.append(kwargs)
        return _FakeTokenizer()

    monkeypatch.setattr(adaptor.AutoTokenizer, "from_pretrained", fake_from_pretrained)
    adaptor.load_tokenizer("local/model", "pinned-revision")

    assert len(observed) == 1
    assert observed[0].get("trust_remote_code", False) is False


@pytest.mark.host
def test_qwen25_72b_config_and_model_loaders_do_not_forward_remote_code(monkeypatch):
    adaptor = importlib.import_module("tt_transformers.models.qwen25_72b.hf_adaptor")
    observed = []
    config = SimpleNamespace(
        num_hidden_layers=80,
        hidden_size=8192,
        num_attention_heads=64,
        num_key_value_heads=8,
        intermediate_size=29568,
        vocab_size=152064,
        tie_word_embeddings=False,
    )

    def fake_config_loader(*args, **kwargs):
        observed.append(("config", kwargs))
        return config

    def fake_model_loader(*args, **kwargs):
        observed.append(("model", kwargs))
        raise RuntimeError("stop after patched HF loaders")

    monkeypatch.setattr(adaptor.AutoConfig, "from_pretrained", fake_config_loader)
    monkeypatch.setattr(adaptor.AutoModelForCausalLM, "from_pretrained", fake_model_loader)
    mesh = SimpleNamespace(get_num_devices=lambda: 8)

    with pytest.raises(RuntimeError, match="stop after patched HF loaders"):
        adaptor.from_pretrained(mesh)

    assert [loader for loader, _kwargs in observed] == ["config", "model"]
    assert all(kwargs.get("trust_remote_code", False) is False for _loader, kwargs in observed)


@pytest.mark.host
def test_existing_llama3_remote_code_opt_in_remains_explicit_and_defaults_false(monkeypatch):
    adaptor = importlib.import_module("tt_transformers.models.llama3_8b.hf_adaptor")
    assert inspect.signature(adaptor.load_tokenizer).parameters["trust_remote_code"].default is False
    assert inspect.signature(adaptor.load_converted_state_dict).parameters["trust_remote_code"].default is False
    observed = []

    def fake_from_pretrained(*args, **kwargs):
        observed.append(kwargs)
        return _FakeTokenizer()

    monkeypatch.setattr(adaptor.AutoTokenizer, "from_pretrained", fake_from_pretrained)
    adaptor.load_tokenizer("local/model")
    adaptor.load_tokenizer("local/model", trust_remote_code=True)

    assert [kwargs["trust_remote_code"] for kwargs in observed] == [False, True]
