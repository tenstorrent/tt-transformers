# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Host characterization for the reusable-foundation public boundary."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tt_transformers.modules.mode import Mode, normalize_mode

pytestmark = pytest.mark.host

_REPO_ROOT = Path(__file__).resolve().parents[2]

_DIRECT_CONSTRUCTOR_PARAMETERS = {
    ("attention/attention_1d.py", "Attention1D"): (
        "self", "wqkv", "wo", "n_heads", "n_kv_heads", "head_dim", "max_batch_size", "max_seq_len"
    ),
    ("embedding/embedding_1d.py", "Embedding1D"): ("self", "weights", "embed_scale"),
    ("lm_head/lm_head_1d.py", "LMHead1D"): ("self", "output_weights"),
    ("mlp/mlp_1d.py", "MLP1D"): ("self", "w1", "w2", "w3"),
    ("mlp/mlp_2d.py", "MLP2D"): ("self", "w1", "w2", "w3"),
    ("rmsnorm/rmsnorm_1d.py", "RMSNorm1D"): ("self", "weight"),
    ("rmsnorm/rmsnorm_2d.py", "RMSNorm2D"): ("self", "weight"),
    ("rope/rope_1d.py", "RotarySetup1D"): ("self", "cos_matrix", "sin_matrix", "max_batch_size"),
    ("sampling/penalties_1d.py", "Penalties1D"): ("self", "vocab_size", "mesh_device", "kwargs"),
    ("sampling/sampling_1d.py", "Sampling1D"): ("self", "vocab_size", "mesh_device", "kwargs"),
}


def _class_definition(relative_path: str, class_name: str) -> ast.ClassDef:
    path = _REPO_ROOT / "src/tt_transformers/modules" / relative_path
    tree = ast.parse(path.read_text(), filename=str(path))
    return next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)


@pytest.mark.host
@pytest.mark.parametrize("module_key,expected_parameters", _DIRECT_CONSTRUCTOR_PARAMETERS.items())
def test_retained_direct_and_config_constructor_contract(module_key, expected_parameters):
    class_definition = _class_definition(*module_key)
    methods = {
        node.name: node
        for node in class_definition.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    constructor = methods["__init__"]
    parameters = [argument.arg for argument in constructor.args.posonlyargs + constructor.args.args]
    parameters.extend(argument.arg for argument in constructor.args.kwonlyargs)
    if constructor.args.vararg is not None:
        parameters.append(constructor.args.vararg.arg)
    if constructor.args.kwarg is not None:
        parameters.append(constructor.args.kwarg.arg)

    assert tuple(parameters) == expected_parameters
    assert "from_config" in methods
    assert "from_model_args" not in methods


@pytest.mark.host
def test_legacy_rope_adapters_are_not_public():
    methods = {
        node.name
        for node in _class_definition("rope/rope_1d.py", "RotarySetup1D").body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert "get_rot_idxs" not in methods
    assert "get_rot_mats" not in methods


@pytest.mark.host
@pytest.mark.parametrize(
    "mode,expected_path",
    [
        ("decode", "decode"),
        (Mode.DECODE, "decode"),
        ("prefill", "prefill"),
        (Mode.PREFILL, "prefill"),
    ],
)
def test_mlp_mode_dispatch_accepts_neutral_values_and_strings(mode, expected_path):
    assert normalize_mode(mode) == expected_path


@pytest.mark.host
def test_models_common_utils_logprobs_alias_is_closed():
    sampling_path = _REPO_ROOT / "src/tt_transformers/modules/sampling/sampling_1d.py"
    sampling_tree = ast.parse(sampling_path.read_text(), filename=str(sampling_path))
    sampling_imports = {
        node.module
        for node in ast.walk(sampling_tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    assert "tt_transformers.sampling.logprobs" in sampling_imports
    assert "models.common.utils" not in sampling_imports

    compatibility_path = _REPO_ROOT / "src/tt_transformers/sampling/logprobs.py"
    compatibility_tree = ast.parse(compatibility_path.read_text(), filename=str(compatibility_path))
    compatibility_imports = {
        node.module
        for node in ast.walk(compatibility_tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    assert compatibility_imports == {"tt_transformers.sampling.tt_log_probs"}
