"""Host-only contracts for executable example cache and TTNN ownership policy."""

import ast
import inspect
import os
import subprocess
import sys
from functools import wraps
from pathlib import Path

import pytest

from tt_transformers.device_ownership import default_device_scope

ROOT = Path(__file__).resolve().parents[2]
SMOKE_PATHS = (
    ROOT / "examples/qwen25_coder_32b/smoke.py",
    ROOT / "examples/qwen3_32b/smoke.py",
)


@pytest.mark.host
def test_benchmark_helper_does_not_require_private_tt_metal_infra_in_generic_ci():
    environment = os.environ.copy()
    environment["CI"] = "true"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from examples.common import benchmarking_utils as helper; assert helper.IS_CI_ENV is False",
        ],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def _calls(path: Path):
    return [node for node in ast.walk(ast.parse(path.read_text(), filename=str(path))) if isinstance(node, ast.Call)]


@pytest.mark.host
def test_all_executable_examples_have_no_forced_remote_cwd_cache_or_unscoped_default_device():
    violations = []
    for path in sorted((ROOT / "examples").rglob("*.py")):
        for call in _calls(path):
            if any(
                keyword.arg == "trust_remote_code"
                and isinstance(keyword.value, ast.Constant)
                and keyword.value.value is True
                for keyword in call.keywords
            ):
                violations.append((path, call.lineno, "trust_remote_code=True"))
            if (
                isinstance(call.func, ast.Name)
                and call.func.id == "Path"
                and call.args
                and isinstance(call.args[0], ast.Constant)
                and call.args[0].value == "model_cache"
            ):
                violations.append((path, call.lineno, 'Path("model_cache")'))
            if isinstance(call.func, ast.Attribute) and call.func.attr in {"SetDefaultDevice", "GetDefaultDevice"}:
                violations.append((path, call.lineno, call.func.attr))
    assert violations == []


@pytest.mark.host
def test_demo_cache_helpers_use_standalone_policy_and_preserve_tt_cache_topology():
    helpers = []
    for path in sorted((ROOT / "examples").glob("*/demo.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        function = next(
            (
                node
                for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == "lazy_weight_cache_dir_for_demo"
            ),
            None,
        )
        if function is None:
            continue
        helpers.append(path.parent.name)
        calls = [
            node
            for node in ast.walk(function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "resolve_model_cache_path"
        ]
        assert len(calls) == 1
        keywords = {keyword.arg: keyword.value for keyword in calls[0].keywords}
        assert ast.literal_eval(keywords["append_topology_to_tt_cache_path"]) is True
        assert isinstance(keywords["hf_revision"], ast.Name) and keywords["hf_revision"].id == "DEMO_HF_REVISION"
        assert isinstance(keywords["mesh_device"], ast.Name) and keywords["mesh_device"].id == "mesh_device"
    assert len(helpers) == 11


@pytest.mark.host
@pytest.mark.parametrize("path", SMOKE_PATHS, ids=lambda path: path.parent.name)
def test_qwen_smoke_runners_are_scoped_and_cli_signature_aware(path):
    tree = ast.parse(path.read_text(), filename=str(path))
    imports_scope = any(
        isinstance(node, ast.ImportFrom)
        and node.module == "tt_transformers.device_ownership"
        and any(alias.name == "default_device_scope" for alias in node.names)
        for node in tree.body
    )
    assert imports_scope
    runners = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name.startswith("run_")]
    assert len(runners) == 6
    assert all(
        any(
            isinstance(decorator, ast.Name) and decorator.id == "_scoped_default_device"
            for decorator in node.decorator_list
        )
        for node in runners
    )
    assert "tuple(inspect.signature(runner).parameters)" in path.read_text()


@pytest.mark.host
def test_qwen25_coder_smoke_uses_attested_cache_and_t3k_ring(monkeypatch, tmp_path):
    path = ROOT / "examples/qwen25_coder_32b/smoke.py"
    source = path.read_text()
    tree = ast.parse(source, filename=str(path))
    helper = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_weight_cache_dir")
    namespace = {"os": os, "Path": Path}
    exec(compile(ast.Module(body=[helper], type_ignores=[]), str(path), "exec"), namespace)

    class RefuseTemporaryCache:
        def mktemp(self, _name):
            raise AssertionError("attested TT_CACHE_PATH must win")

    monkeypatch.setenv("TT_CACHE_PATH", str(tmp_path / "qwen25_coder_32b"))
    cache = namespace["_weight_cache_dir"](RefuseTemporaryCache(), "unused")
    assert cache == tmp_path / "qwen25_coder_32b" / "T3K"
    assert cache.is_dir()
    assert 'os.environ.get("MESH_DEVICE") == "T3K"' in source
    assert "ttnn.FabricConfig.FABRIC_1D_RING" in source


@pytest.mark.host
@pytest.mark.parametrize("path", SMOKE_PATHS, ids=lambda path: path.parent.name)
def test_qwen_smoke_scope_restores_exact_prior_device_on_success_and_failure(path):
    tree = ast.parse(path.read_text(), filename=str(path))
    helper = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_scoped_default_device"
    )

    class FakeTTNN:
        def __init__(self):
            self.current = object()
            self.events = []

        def GetDefaultDevice(self):
            return self.current

        def SetDefaultDevice(self, device):
            self.current = device
            self.events.append(device)

    fake = FakeTTNN()
    previous = fake.current
    namespace = {
        "__name__": "example_smoke_probe",
        "wraps": wraps,
        "default_device_scope": default_device_scope,
        "ttnn": fake,
    }
    exec(compile(ast.Module(body=[helper], type_ignores=[]), str(path), "exec"), namespace)

    observed = []

    def subject(mesh_device, value, *, fail=False):
        observed.append(fake.current)
        if fail:
            raise RuntimeError("body failure")
        return value

    wrapped = namespace["_scoped_default_device"](subject)
    device = object()
    assert inspect.signature(wrapped) == inspect.signature(subject)
    assert wrapped(device, 7) == 7
    assert fake.current is previous and observed[-1] is device
    with pytest.raises(RuntimeError, match="body failure"):
        wrapped(device, 8, fail=True)
    assert fake.current is previous and observed[-1] is device
