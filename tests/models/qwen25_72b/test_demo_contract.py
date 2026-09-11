# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

_DEMO_PATH = "examples/qwen25_72b/demo.py"
_HARDWARE_DEMO_PATH = "tests/hardware/models/qwen25_72b/test_demo.py"
_DEMO_SOURCE = "\n".join(
    (Path(_DEMO_PATH).read_text(encoding="utf-8"), Path(_HARDWARE_DEMO_PATH).read_text(encoding="utf-8"))
)
_DEMO_TREE = ast.parse(_DEMO_SOURCE, filename=f"{_DEMO_PATH}+{_HARDWARE_DEMO_PATH}")


def _function(name):
    if name == "_ttnn_mesh_device_param_from_env":
        name = "ttnn_mesh_device_param_from_env"
    return next(node for node in _DEMO_TREE.body if isinstance(node, ast.FunctionDef) and node.name == name)


def _calls(function_name, called_name):
    return [
        node
        for node in ast.walk(_function(function_name))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == called_name
    ]


@pytest.mark.host
@pytest.mark.model
def test_demo_case_manifest_is_preserved():
    decorators = [node for node in _function("test_qwen25_72b").decorator_list if isinstance(node, ast.Call)]
    test_config = next(node for node in decorators if ast.literal_eval(node.args[0]) == "test_config")
    optimizations = next(node for node in decorators if ast.literal_eval(node.args[0]) == "optimizations")
    assert [ast.literal_eval(element.keywords[0].value) for element in test_config.args[1].elts] == [
        "token-accuracy",
        "batch-1",
        "batch-32",
        "batch-32-ci",
        "eval-32",
        "ci-b1-DP-2",
        "ci-b1-DP-4",
        "ci-b1-DP-8",
        "ci-b1-DP-16",
        "ci-b1-DP-32",
    ]
    assert ast.literal_eval(optimizations.args[1]) == ["performance", "accuracy"]


@pytest.mark.host
@pytest.mark.model
def test_demo_keeps_qwen72_trace_region_and_ring_fabric():
    assert '"trace_region_size": 70_000_000' in _DEMO_SOURCE
    assert "ttnn.FabricConfig.FABRIC_1D_RING" in _DEMO_SOURCE


@pytest.mark.host
@pytest.mark.model
def test_demo_uses_model_owned_runtime_provider_and_shared_helpers():
    imports = [ast.unparse(node) for node in _DEMO_TREE.body if isinstance(node, (ast.Import, ast.ImportFrom))]
    assert any("tt_transformers.models.qwen25_72b.hf_generator" in statement for statement in imports)
    assert any("tt_transformers.models.qwen25_72b.hf_generator" in statement for statement in imports)
    assert any("examples.common.run_helpers" in statement for statement in imports)
    assert all("tt_transformers.models.executor" not in statement for statement in imports)
    assert all("AutoConfig" not in statement and "AutoTokenizer" not in statement for statement in imports)
    assert not any(
        node.name == "assert_no_special_tokens" for node in _DEMO_TREE.body if isinstance(node, ast.FunctionDef)
    )


@pytest.mark.host
@pytest.mark.model
def test_supported_tp8_model_build_failures_are_not_converted_to_skips():
    create_model = _function("create_model")
    assert not any(isinstance(node, ast.Try) for node in ast.walk(create_model))
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "pytest"
        and node.func.attr == "skip"
        for node in ast.walk(create_model)
    )


@pytest.mark.host
@pytest.mark.model
@pytest.mark.parametrize("data_parallel", [2, 4, 8, 16, 32])
def test_every_dp_case_skips_before_submesh_or_model_construction(data_parallel, expect_error):
    namespace = {
        "pytest": pytest,
        "ttnn": SimpleNamespace(MeshDevice=object),
        "_MIN_TP_DEVICES": 8,
        "UnsupportedConfiguration": pytest.skip.Exception,
    }
    function = _function("_dp_or_skip")
    exec(compile(ast.Module(body=[function], type_ignores=[]), _DEMO_PATH, "exec"), namespace)
    mesh = SimpleNamespace(get_num_devices=lambda: 8)
    with expect_error(pytest.skip.Exception, f"DP-{data_parallel}"):
        namespace["_dp_or_skip"](mesh, data_parallel)
    run_dp = _function("_run_dp_smoke")
    calls = [
        node.func.id for node in ast.walk(run_dp) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    ]
    assert calls[0] == "_dp_or_skip"
    assert "create_dp_submeshes" not in calls[: calls.index("_skip_below_min_tp_devices")]


@pytest.mark.host
@pytest.mark.model
def test_demo_allocates_kv_cache_without_model_shape_arguments():
    for function_name in ("_run_token_accuracy", "_run_perf_benchmark", "_run_eval_repeat_batch32"):
        allocations = [
            node
            for node in ast.walk(_function(function_name))
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "allocate_kv_cache"
        ]
        assert allocations
        assert all(not call.args and not call.keywords for call in allocations)


@pytest.mark.host
@pytest.mark.model
def test_perf_registers_actual_prefill_before_closed_world_trace_activation():
    tokenization = _calls("_run_perf_benchmark", "tokenize_prompts")[0]
    warmup = _calls("_run_perf_benchmark", "_warmup_demo_executor")[0]
    benchmark = _calls("_run_perf_benchmark", "run_perf_benchmark")[0]
    assert tokenization.lineno < warmup.lineno < benchmark.lineno
    keywords = {keyword.arg: ast.unparse(keyword.value) for keyword in warmup.keywords}
    assert keywords["prefill_compile_case"] == "(input_tokens, prompt_lens)"


@pytest.mark.host
@pytest.mark.model
def test_eval_uses_decode_only_trace_and_registers_representative_prefill_eagerly():
    create = _calls("_run_eval_repeat_batch32", "create_executor")[0]
    create_keywords = {keyword.arg: keyword.value for keyword in create.keywords}
    assert ast.literal_eval(create_keywords["trace_mode"]) == "decode_only"
    warmup = _calls("_run_eval_repeat_batch32", "_warmup_demo_executor")[0]
    warmup_keywords = {keyword.arg: ast.unparse(keyword.value) for keyword in warmup.keywords}
    assert warmup_keywords["prefill_compile_case"] == "representative_prefill"
