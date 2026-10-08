# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Contract tests for the architecture-level vLLM entry points.

The vLLM TT plugin registers the classes in ``tt_transformers.vllm_registry``
by dotted path and reads some of their attributes before any model exists.
These tests pin that contract without ``ttnn`` or ``vllm``: generator sources
are read as text, and generator classes are replaced by fakes where a call
has to be routed.
"""

from __future__ import annotations

import ast
import importlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tt_transformers import vllm_registry
from tt_transformers.vllm_registry import (
    GeneratorEntry,
    LlamaForCausalLM,
    MistralForCausalLM,
    Phi3ForCausalLM,
    Qwen2ForCausalLM,
    Qwen3ForCausalLM,
    UnsupportedModelError,
    resolve_hf_id,
)

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "src" / "tt_transformers" / "models"

DISPATCHERS = (LlamaForCausalLM, Qwen2ForCausalLM, Qwen3ForCausalLM, MistralForCausalLM, Phi3ForCausalLM)

# The dotted paths the plugin registers. Changing one breaks every plugin
# release that names it, so a rename must fail here first.
PUBLIC_SYMBOLS = (
    "tt_transformers.vllm_registry:LlamaForCausalLM",
    "tt_transformers.vllm_registry:Qwen2ForCausalLM",
    "tt_transformers.vllm_registry:Qwen3ForCausalLM",
    "tt_transformers.vllm_registry:MistralForCausalLM",
    "tt_transformers.vllm_registry:Phi3ForCausalLM",
)

# Every model_capabilities key the plugin reads, and what an architecture class
# does with it. "declared" keys must be on every architecture class; the others
# must be absent from it, for the reason given.
PLUGIN_CAPABILITY_KEYS = {
    "supports_prefix_caching": "declared",
    "supports_async_decode": "declared",
    "supports_sample_on_device": "declared",
    "accepts_trace_mode": "declared",
    "fabric_config": "absent: generators of one architecture need different fabrics",
    "max_device_top_k": "absent: read from the generator instance",
    "supports_device_penalties": "absent: read from the generator instance; absent means supported",
    "supports_chunked_prefill": "absent: no generator declares it",
    "output_tokens_per_step": "absent: every generator emits one token per step",
    "tt_adaptive_block_output": "absent: block-output only",
    "tt_adaptive_block_max_prompt_tokens": "absent: block-output only",
    "tt_block_kv_extent_tokens": "absent: block-output only",
    "supports_device_grammar": "absent: no generator declares it",
    "supports_spec_decode": "absent: no generator declares it",
    "supports_async_spec_decode": "absent: no generator declares it",
    "spec_requirements": "absent: speculative decoding only",
    "spec_hidden_handoff": "absent: speculative decoding only",
}


def _all_entries() -> list[tuple[type, GeneratorEntry]]:
    return [(dispatcher, entry) for dispatcher in DISPATCHERS for entry in dispatcher.generators]


def _generator_source(entry: GeneratorEntry) -> Path:
    module_name, _, _ = entry.generator.partition(":")
    assert module_name == f"tt_transformers.models.{entry.package}.vllm_generator"
    return MODELS / entry.package / "vllm_generator.py"


def _generator_class(entry: GeneratorEntry) -> ast.ClassDef:
    class_name = entry.generator.partition(":")[2]
    tree = ast.parse(_generator_source(entry).read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return node
    raise AssertionError(f"{entry.generator} is not defined")


def _declared_capabilities(entry: GeneratorEntry) -> dict[str, object]:
    """Return the generator's literal model_capabilities; non-literal values map to ``...``."""

    for node in _generator_class(entry).body:
        if (
            isinstance(node, ast.Assign)
            and [getattr(target, "id", None) for target in node.targets] == ["model_capabilities"]
            and isinstance(node.value, ast.Dict)
        ):
            declared: dict[str, object] = {}
            for key, value in zip(node.value.keys, node.value.values):
                assert isinstance(key, ast.Constant), "model_capabilities keys must be literal"
                try:
                    declared[key.value] = ast.literal_eval(value)
                except ValueError:
                    declared[key.value] = ...
            return declared
    raise AssertionError(f"{entry.generator} declares no model_capabilities")


def _initialize_signature(entry: GeneratorEntry) -> ast.arguments:
    for node in _generator_class(entry).body:
        if isinstance(node, ast.FunctionDef) and node.name == "initialize_vllm_model":
            return node.args
    raise AssertionError(f"{entry.generator} has no initialize_vllm_model")


def _fake_generator(entry: GeneratorEntry, calls: list[tuple[str, tuple, dict]]) -> type:
    class FakeGenerator:
        @classmethod
        def get_max_tokens_all_users(cls, **kwargs):
            calls.append((entry.hf_id, (), kwargs))
            return kwargs["max_model_len"] + 1

        @classmethod
        def initialize_vllm_model(cls, *args, **kwargs):
            calls.append((entry.hf_id, args, kwargs))
            return SimpleNamespace(served_by=entry.hf_id)

    return FakeGenerator


@pytest.fixture
def routed(monkeypatch) -> list[tuple[str, tuple, dict]]:
    """Replace generator imports with fakes that record each call's hf_id."""

    calls: list[tuple[str, tuple, dict]] = []
    monkeypatch.setattr(GeneratorEntry, "load", lambda entry: _fake_generator(entry, calls))
    return calls


@pytest.mark.host
def test_table_matches_support_manifests() -> None:
    manifests = {
        path.parent.name: json.loads(path.read_text(encoding="utf-8"))["model"]
        for path in sorted((ROOT / "examples").glob("*/support.json"))
    }
    served = {path.parent.name for path in MODELS.glob("*/vllm_generator.py")}

    table = {entry.package: entry.hf_id for _, entry in _all_entries()}

    assert len(table) == len(_all_entries()), "a package appears under more than one architecture"
    assert set(table) == served
    for package, hf_id in table.items():
        assert manifests[package]["package"] == package
        assert manifests[package]["hf_id"] == hf_id


@pytest.mark.host
@pytest.mark.parametrize(("dispatcher", "entry"), _all_entries(), ids=lambda value: getattr(value, "package", None))
def test_table_names_an_existing_generator_class(dispatcher, entry) -> None:
    _generator_class(entry)


@pytest.mark.host
@pytest.mark.parametrize("symbol", PUBLIC_SYMBOLS)
def test_public_symbols_resolve_by_dotted_path(symbol: str) -> None:
    module_name, _, class_name = symbol.partition(":")
    resolved = getattr(importlib.import_module(module_name), class_name)

    assert resolved.architecture == class_name
    assert callable(resolved.initialize_vllm_model)
    assert callable(resolved.get_max_tokens_all_users)


@pytest.mark.host
def test_import_is_cheap_and_has_no_side_effects() -> None:
    code = (
        "import sys, tt_transformers.vllm_registry; "
        "heavy = sorted(m for m in ('ttnn', 'vllm', 'torch', 'transformers') if m in sys.modules); "
        "models = sorted(m for m in sys.modules if m.startswith('tt_transformers.models')); "
        "print(heavy, models)"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)

    assert result.stdout.strip() == "[] []"
    assert not hasattr(vllm_registry, "register")


@pytest.mark.host
@pytest.mark.parametrize(
    ("name_or_path", "expected"),
    [
        ("meta-llama/Llama-3.1-8B-Instruct", "meta-llama/Llama-3.1-8B-Instruct"),
        (
            "/cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a1",
            "meta-llama/Llama-3.1-8B-Instruct",
        ),
        ("/cache/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/abc/", "Qwen/Qwen2.5-7B-Instruct"),
        ("relative/models--microsoft--phi-4/snapshots/rev", "microsoft/phi-4"),
        ("/weights/Llama-3.1-8B-Instruct", None),
        ("/weights/meta-llama/Llama-3.1-8B-Instruct", None),
        ("./meta-llama/Llama-3.1-8B-Instruct", None),
        ("/cache/hub/models--Llama-3.1-8B-Instruct/snapshots/rev", None),
        ("Llama-3.1-8B-Instruct", None),
        ("", None),
        (None, None),
    ],
)
def test_resolve_hf_id(name_or_path, expected) -> None:
    assert resolve_hf_id(name_or_path) == expected


@pytest.mark.host
@pytest.mark.parametrize(("dispatcher", "entry"), _all_entries(), ids=lambda value: getattr(value, "package", None))
def test_select_by_id_and_by_snapshot(dispatcher, entry) -> None:
    org, name = entry.hf_id.split("/")
    snapshot = f"/hub/models--{org}--{name}/snapshots/0123abcd"

    assert dispatcher.select(entry.hf_id) is entry
    assert dispatcher.select(snapshot) is entry


@pytest.mark.host
@pytest.mark.parametrize("dispatcher", DISPATCHERS, ids=lambda d: d.architecture)
@pytest.mark.parametrize(
    "name_or_path",
    [
        "meta-llama/Llama-3.1-70B-Instruct",
        "meta-llama/llama-3.1-8b-instruct",
        "/weights/Llama-3.1-8B-Instruct",
        None,
    ],
)
def test_unknown_checkpoint_names_the_supported_ids(dispatcher, name_or_path) -> None:
    with pytest.raises(UnsupportedModelError) as raised:
        dispatcher.select(name_or_path)

    message = str(raised.value)
    assert dispatcher.architecture in message
    assert repr(name_or_path) in message
    for hf_id in dispatcher.supported_hf_ids():
        assert hf_id in message


@pytest.mark.host
def test_a_checkpoint_is_served_only_under_its_own_architecture() -> None:
    with pytest.raises(UnsupportedModelError):
        Qwen2ForCausalLM.select("Qwen/Qwen3-32B")
    with pytest.raises(UnsupportedModelError):
        LlamaForCausalLM.select("mistralai/Mistral-7B-Instruct-v0.3")


@pytest.mark.host
@pytest.mark.parametrize("dispatcher", DISPATCHERS, ids=lambda d: d.architecture)
def test_class_capabilities_are_the_intersection_of_the_generators(dispatcher) -> None:
    declared = [_declared_capabilities(entry) for entry in dispatcher.generators]
    shared = {key for key in declared[0] if all(caps.get(key) is True for caps in declared)}

    assert dispatcher.model_capabilities == {key: True for key in shared}


@pytest.mark.host
@pytest.mark.parametrize("dispatcher", DISPATCHERS, ids=lambda d: d.architecture)
def test_class_capabilities_cover_every_plugin_key(dispatcher) -> None:
    declared_keys = {key for key, policy in PLUGIN_CAPABILITY_KEYS.items() if policy == "declared"}

    assert set(dispatcher.model_capabilities) == declared_keys
    assert "fabric_config" not in dispatcher.model_capabilities
    assert "max_device_top_k" not in dispatcher.model_capabilities


@pytest.mark.host
def test_every_generator_capability_key_is_known() -> None:
    for _, entry in _all_entries():
        unknown = set(_declared_capabilities(entry)) - set(PLUGIN_CAPABILITY_KEYS)
        assert not unknown, f"{entry.generator} declares keys this registry does not account for: {unknown}"


@pytest.mark.host
def test_architecture_classes_do_not_share_a_capability_dict() -> None:
    assert len({id(dispatcher.model_capabilities) for dispatcher in DISPATCHERS}) == len(DISPATCHERS)


@pytest.mark.host
@pytest.mark.parametrize(("dispatcher", "entry"), _all_entries(), ids=lambda value: getattr(value, "package", None))
def test_get_max_tokens_all_users_routes_by_model_name(routed, dispatcher, entry) -> None:
    org, name = entry.hf_id.split("/")
    for model_name in (entry.hf_id, f"/hub/models--{org}--{name}/snapshots/rev"):
        budget = dispatcher.get_max_tokens_all_users(
            model_name=model_name, num_devices=2, tt_data_parallel=1, max_model_len=4096, max_num_seqs=8
        )
        assert budget == 4097
        assert routed[-1] == (
            entry.hf_id,
            (),
            {
                "model_name": model_name,
                "num_devices": 2,
                "tt_data_parallel": 1,
                "max_model_len": 4096,
                "max_num_seqs": 8,
            },
        )


@pytest.mark.host
@pytest.mark.parametrize("dispatcher", DISPATCHERS, ids=lambda d: d.architecture)
def test_get_max_tokens_all_users_rejects_an_unknown_model(routed, dispatcher) -> None:
    with pytest.raises(UnsupportedModelError):
        dispatcher.get_max_tokens_all_users(model_name="org/unknown", max_model_len=4096)
    assert routed == []


@pytest.mark.host
@pytest.mark.parametrize(("dispatcher", "entry"), _all_entries(), ids=lambda value: getattr(value, "package", None))
@pytest.mark.parametrize("optimizations", [None, "performance", "accuracy"])
def test_initialize_returns_the_selected_generator_instance(routed, dispatcher, entry, optimizations) -> None:
    hf_config = SimpleNamespace(_name_or_path=entry.hf_id)
    mesh = object()

    # The plugin's loader call: positional config, device and batch; keyword rest.
    model = dispatcher.initialize_vllm_model(
        hf_config, mesh, 32, max_seq_len=4096, tt_data_parallel=1, optimizations=optimizations
    )

    assert model.served_by == entry.hf_id
    hf_id, args, kwargs = routed[-1]
    assert hf_id == entry.hf_id
    assert args == (hf_config, mesh, 32, 4096)
    expected = {"n_layers": None, "tt_data_parallel": 1}
    if optimizations is not None:
        expected["optimizations"] = optimizations
    assert kwargs == expected


@pytest.mark.host
@pytest.mark.parametrize("dispatcher", DISPATCHERS, ids=lambda d: d.architecture)
def test_initialize_rejects_an_unknown_model_before_importing_a_generator(routed, dispatcher) -> None:
    with pytest.raises(UnsupportedModelError):
        dispatcher.initialize_vllm_model(SimpleNamespace(_name_or_path="org/unknown"), object(), 32, 4096)
    with pytest.raises(UnsupportedModelError):
        dispatcher.initialize_vllm_model(SimpleNamespace(), object(), 32, 4096)
    assert routed == []


@pytest.mark.host
@pytest.mark.parametrize(("dispatcher", "entry"), _all_entries(), ids=lambda value: getattr(value, "package", None))
def test_generator_accepts_what_the_dispatcher_passes(dispatcher, entry) -> None:
    """Every generator takes the forwarded keywords and defaults to performance precision.

    The dispatcher drops ``optimizations=None``, so the generator's own default
    decides the precision when the deployment names none. Pinning that default
    keeps "no override" meaning the same precision for every model.
    """

    signature = _initialize_signature(entry)
    names = [arg.arg for arg in signature.args]
    assert names[:5] == ["cls", "hf_config", "mesh_device", "max_batch_size", "max_seq_len"]
    assert {"n_layers", "tt_data_parallel", "optimizations"} <= set(names)

    defaults = dict(zip(names[-len(signature.defaults) :], signature.defaults))
    assert ast.literal_eval(defaults["optimizations"]) == "performance"
