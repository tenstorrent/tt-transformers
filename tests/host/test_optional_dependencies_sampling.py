# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Host-only characterization for lazy model exports and canonical sampling."""

from __future__ import annotations

import ast
import importlib
import os
import subprocess
import sys
import types
from dataclasses import MISSING, fields
from pathlib import Path

import pytest

from tt_transformers.sampling.sampling_params import SamplingParams

pytestmark = pytest.mark.host

ROOT = Path(__file__).resolve().parents[2]
MODEL_NAMES = (
    "deepseek_r1_distill_qwen_14b",
    "llama32_1b",
    "llama32_3b",
    "llama33_70b",
    "llama3_8b",
    "mistral_7b",
    "phi4",
    "qwen25_72b",
    "qwen25_7b",
    "qwen25_coder_32b",
    "qwen2_7b",
    "qwen3_32b",
)


def _run_isolated(source: str) -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    result = subprocess.run(
        [sys.executable, "-c", source],
        cwd="/tmp",
        env=environment,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _literal_assignment(tree: ast.Module, name: str):
    node = next(
        node
        for node in tree.body
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        and any(
            isinstance(target, ast.Name) and target.id == name
            for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
        )
    )
    return ast.literal_eval(node.value)


@pytest.mark.host
@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_model_initializer_is_lazy_and_export_map_matches_all(model_name):
    path = ROOT / "src/tt_transformers/models" / model_name / "__init__.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    exports = _literal_assignment(tree, "__all__")
    import_roots = {(node.module or "").split(".", 1)[0] for node in tree.body if isinstance(node, ast.ImportFrom)}
    assert not ({"transformers", "tqdm", "tt_transformers"} & import_roots)

    export_maps = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "_EXPORTS" for target in node.targets)
    ]
    if exports:
        assert len(export_maps) == 1
        export_map = ast.literal_eval(export_maps[0].value)
        assert list(export_map) == exports
        assert all(module.startswith(f"tt_transformers.models.{model_name}.") for module, _ in export_map.values())
    else:
        assert model_name == "llama3_8b"
        assert export_maps == []


@pytest.mark.host
def test_all_model_packages_import_with_optional_dependencies_blocked():
    packages = ", ".join(repr(f"tt_transformers.models.{name}") for name in MODEL_NAMES)
    _run_isolated(
        f"""
import importlib
import importlib.abc
import sys

class BlockOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] in {{"transformers", "tqdm"}}:
            raise ModuleNotFoundError(f"blocked optional dependency: {{fullname}}", name=fullname)
        return None

sys.meta_path.insert(0, BlockOptional())
for package_name in ({packages},):
    package = importlib.import_module(package_name)
    assert not any(name.split(".", 1)[0] in {{"transformers", "tqdm"}} for name in sys.modules)
    assert set(package.__dict__).isdisjoint(package.__all__)
"""
    )


@pytest.mark.host
def test_optional_hf_exports_load_only_when_accessed():
    optional_exports = {
        "deepseek_r1_distill_qwen_14b": "DeepSeekR1Qwen14BForCausalLM",
        "llama32_1b": "Llama32_1BForCausalLM",
        "llama32_3b": "Llama32_3BForCausalLM",
        "llama33_70b": "Llama33_70BForCausalLM",
        "mistral_7b": "Mistral7BForCausalLM",
        "phi4": "Phi4ForCausalLM",
        "qwen25_72b": "Qwen25_72BForCausalLM",
        "qwen25_7b": "Qwen25ForCausalLM",
        "qwen25_coder_32b": "Qwen25Coder32BForCausalLM",
        "qwen2_7b": "Qwen2ForCausalLM",
    }
    cases = ", ".join(repr(item) for item in optional_exports.items())
    _run_isolated(
        f"""
import importlib
import importlib.abc
import sys

class BlockOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] in {{"transformers", "tqdm"}}:
            raise ModuleNotFoundError(f"blocked optional dependency: {{fullname}}", name=fullname)
        return None

sys.meta_path.insert(0, BlockOptional())
for model_name, export_name in ({cases},):
    package = importlib.import_module(f"tt_transformers.models.{{model_name}}")
    assert export_name not in package.__dict__
    try:
        getattr(package, export_name)
    except ModuleNotFoundError as error:
        assert error.name.split(".", 1)[0] in {{"transformers", "tqdm"}}
    else:
        raise AssertionError(f"optional export {{model_name}}.{{export_name}} loaded without its dependency")
"""
    )


@pytest.mark.host
@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_core_model_module_has_no_eager_optional_import(model_name):
    path = ROOT / "src/tt_transformers/models" / model_name / "model.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    optional = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            optional.extend(
                alias.name for alias in node.names if alias.name.split(".", 1)[0] in {"transformers", "tqdm"}
            )
        elif isinstance(node, ast.ImportFrom) and (node.module or "").split(".", 1)[0] in {"transformers", "tqdm"}:
            optional.append(node.module)
    assert optional == []


@pytest.mark.host
def test_sampling_params_has_one_identity_and_exact_defaults(monkeypatch):
    fake_ttnn = types.ModuleType("ttnn")
    fake_ttnn.Tensor = type("Tensor", (), {})
    monkeypatch.setitem(sys.modules, "ttnn", fake_ttnn)

    package = importlib.import_module("tt_transformers.sampling")

    # The TTTv1 generator carried a second SamplingParams dataclass, so this used to assert that
    # the two identities agreed. That module is gone; sampling_params is the only definition left.
    assert package.SamplingParams is SamplingParams

    definitions = {field.name: field.default for field in fields(SamplingParams)}
    assert definitions == {
        "temperature": MISSING,
        "top_k": MISSING,
        "top_p": MISSING,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "repetition_penalty": 1.0,
        "seed": None,
        "enable_log_probs": False,
        "num_logprobs": 0,
    }


@pytest.mark.host
def test_sampling_package_no_longer_exports_the_tttv1_surface():
    """The generator/TTSampling/TTPenalties surface was removed; nothing here consumed it.

    Its parameter helpers duplicated ``modules.sampling.params``, which owns them now. tt-metal
    keeps its own copy of the removed modules under ``models/common/sampling`` and is unaffected.
    """
    package = importlib.import_module("tt_transformers.sampling")

    assert set(package.__all__) == {"LogProbsCalculator", "LogProbsResult", "SamplingParams", "split_list"}
    for removed in ("SamplingGenerator", "TTSampling", "TTPenalties", "slice_sampling_params", "SeedManager"):
        with pytest.raises(AttributeError):
            getattr(package, removed)
    for module in ("generator", "tt_sampling", "tt_penalties"):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(f"tt_transformers.sampling.{module}")
