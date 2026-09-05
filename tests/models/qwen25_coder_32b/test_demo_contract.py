# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

_DEMO_PATH = "examples/qwen25_coder_32b/demo.py"
_HARDWARE_DEMO_PATH = "tests/hardware/models/qwen25_coder_32b/test_demo.py"
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
    decorators = [node for node in _function("test_qwen25_coder_32b").decorator_list if isinstance(node, ast.Call)]
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
def test_demo_keeps_coder_trace_region_and_fabric():
    assert '"trace_region_size": 50_000_000' in _DEMO_SOURCE
    assert "ttnn.FabricConfig.FABRIC_1D" in _DEMO_SOURCE


@pytest.mark.host
@pytest.mark.model
def test_demo_uses_model_owned_runtime_compatibility_wrappers():
    imports = [ast.unparse(node) for node in _DEMO_TREE.body if isinstance(node, (ast.Import, ast.ImportFrom))]
    assert any("tt_transformers.models.qwen25_coder_32b.executor" in statement for statement in imports)
    assert "EagerQwen25Coder32BExecutor" in _DEMO_SOURCE
    assert "TracedQwen25Coder32BExecutor" in _DEMO_SOURCE
    assert "Qwen25Coder32B.from_pretrained" in _DEMO_SOURCE


@pytest.mark.host
@pytest.mark.model
def test_supported_tp8_model_build_failures_are_reported_as_skips_for_demo_usability():
    create_model = _function("create_model")
    assert any(isinstance(node, ast.Try) for node in ast.walk(create_model))
    assert any(
        isinstance(node, ast.Raise)
        and isinstance(node.exc, ast.Call)
        and isinstance(node.exc.func, ast.Name)
        and node.exc.func.id == "UnsupportedConfiguration"
        for node in ast.walk(create_model)
    )
    wrapper = ast.unparse(_function("test_qwen25_coder_32b"))
    assert "except example.UnsupportedConfiguration" in wrapper
    assert "pytest.skip(str(error))" in wrapper


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
    if data_parallel == 8:
        namespace["_dp_or_skip"](mesh, data_parallel)
    else:
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
def test_demo_allocates_kv_cache_with_vllm_shape_compatibility_arguments():
    for function_name in ("_run_token_accuracy", "_run_perf_benchmark", "_run_eval_repeat_batch32"):
        allocations = [
            node
            for node in ast.walk(_function(function_name))
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "allocate_kv_cache"
        ]
        assert allocations
        assert all(call.args or call.keywords for call in allocations)


@pytest.mark.host
@pytest.mark.model
def test_perf_and_eval_use_traced_model_owned_wrapper():
    assert _calls("_run_perf_benchmark", "TracedQwen25Coder32BExecutor")
    assert _calls("_run_eval_repeat_batch32", "TracedQwen25Coder32BExecutor")
