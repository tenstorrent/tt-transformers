from __future__ import annotations

import ast
import errno
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tt_transformers import cache_environment as cache_policy
from tt_transformers.cache_environment import (
    CacheIdentity,
    ENVIRONMENT_SPECS,
    environment_flag,
    environment_int,
    model_preflight_report,
    offline_mode,
    parse_bool,
    resolve_model_cache,
)
from tt_transformers.cache_environment import report_model_preflight as shared_report_model_preflight


ROOT = Path(__file__).resolve().parents[2]


PHASE0_KEYS = {
    "CI",
    "DISABLE_BATCHED_EXTRACT",
    "DISABLE_BATCHED_PREFILL",
    "DISABLE_MINIMAL_MATMUL",
    "DISABLE_PREFILL_AG_BF8",
    "DISABLE_PREFILL_REDUCE_BF8",
    "HF_HUB_OFFLINE",
    "HF_MODEL",
    "MAX_PREFILL_CHUNK_SIZE",
    "MESH_DEVICE",
    "QWEN25_CODER_32B_DEMO_NUM_LAYERS",
    "QWEN25_CODER_32B_NUMDIV_LAYERS",
    "QWEN25_CODER_32B_NUMDIV_OUT",
    "QWEN25_CODER_32B_NUMDIV_SEQ",
    "QWEN3_32B_DEMO_NUM_LAYERS",
    "QWEN3_32B_NUMDIV_LAYERS",
    "QWEN3_32B_NUMDIV_OUT",
    "QWEN3_32B_NUMDIV_SEQ",
    "TRANSFORMERS_OFFLINE",
    "TT_CACHE_FALLBACK_PATH",
    "TT_CACHE_PATH",
}


@pytest.mark.host
@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on", True])
def test_normalized_true_values(value):
    assert parse_bool(value, name="FLAG") is True


@pytest.mark.host
@pytest.mark.parametrize("value", ["", "0", "false", "FALSE", "no", "off", False])
def test_normalized_false_values(value):
    assert parse_bool(value, name="FLAG", default=True) is False


@pytest.mark.host
def test_invalid_boolean_is_not_silently_truthy():
    with pytest.raises(ValueError, match="FLAG must be one of"):
        parse_bool("sometimes", name="FLAG")


@pytest.mark.host
def test_offline_policy_is_explicit_and_not_derived_from_ci():
    assert offline_mode({"CI": "true"}) is False
    assert offline_mode({"HF_HUB_OFFLINE": "1"}) is True
    assert offline_mode({"TRANSFORMERS_OFFLINE": "yes"}) is True
    assert offline_mode({"TT_TRANSFORMERS_OFFLINE": "on"}) is True
    assert offline_mode({"CI": "true", "HF_HUB_OFFLINE": "0"}) is False


@pytest.mark.host
def test_typed_inventory_covers_all_phase0_keys():
    assert PHASE0_KEYS <= ENVIRONMENT_SPECS.keys()
    assert environment_flag("DISABLE_BATCHED_PREFILL", {"DISABLE_BATCHED_PREFILL": "0"}) is False
    assert environment_int("MAX_PREFILL_CHUNK_SIZE", {"MAX_PREFILL_CHUNK_SIZE": "4"}) == 4


@pytest.mark.host
def test_default_cache_is_home_scoped_and_never_cwd_relative(tmp_path):
    resolution = resolve_model_cache(
        hf_model_id="owner/model",
        topology="N300",
        create_mode="never",
        environ={"HOME": str(tmp_path / "home")},
    )
    assert resolution.path == (
        tmp_path
        / "home/.cache/tt-transformers/owner/model"
        / f"identity-1-{resolution.cache_identity.digest}"
        / "N300"
    )
    assert resolution.source == "user_cache"
    assert resolution.identity_applied_to_path is True
    assert resolution.release_warning is None
    assert resolution.path != resolution.legacy_cwd_path


@pytest.mark.host
def test_xdg_and_standalone_cache_roots_keep_model_topology_suffix(tmp_path):
    xdg = resolve_model_cache(
        hf_model_id="owner/model",
        topology="T3K",
        create_mode="never",
        environ={"XDG_CACHE_HOME": str(tmp_path / "xdg")},
    )
    configured = resolve_model_cache(
        hf_model_id="owner/model",
        topology="T3K",
        create_mode="never",
        environ={"TT_TRANSFORMERS_CACHE": str(tmp_path / "configured")},
    )
    assert xdg.path == (
        tmp_path / "xdg/tt-transformers/owner/model" / f"identity-1-{xdg.cache_identity.digest}" / "T3K"
    )
    assert configured.path == (
        tmp_path / "configured/owner/model" / f"identity-1-{configured.cache_identity.digest}" / "T3K"
    )


@pytest.mark.host
def test_explicit_cache_dir_and_legacy_tt_cache_path_remain_exact(tmp_path):
    argument = resolve_model_cache(
        hf_model_id="owner/model",
        topology="N300",
        cache_dir=tmp_path / "argument",
        create_mode="never",
        environ={},
    )
    legacy_env = resolve_model_cache(
        hf_model_id="owner/model",
        topology="N300",
        create_mode="never",
        environ={"TT_CACHE_PATH": str(tmp_path / "legacy")},
    )
    llama_env = resolve_model_cache(
        hf_model_id="owner/model",
        topology="P150",
        append_topology_to_tt_cache_path=True,
        create_mode="never",
        environ={"TT_CACHE_PATH": str(tmp_path / "legacy")},
    )
    assert argument.path == tmp_path / "argument"
    assert legacy_env.path == tmp_path / "legacy"
    assert llama_env.path == tmp_path / "legacy/P150"
    assert not argument.identity_applied_to_path
    assert not legacy_env.identity_applied_to_path
    assert not llama_env.identity_applied_to_path
    assert "release qualification warning" in argument.release_warning
    assert "release qualification warning" in legacy_env.release_warning


@pytest.mark.host
def test_permission_fallback_preserves_llama_layout(monkeypatch, tmp_path):
    configured = tmp_path / "readonly/P150"
    fallback = tmp_path / "fallback/model/P150"

    def fake_mkdir(path, *, parents, exist_ok):
        assert parents and exist_ok
        if path == configured:
            raise OSError(errno.EACCES, "denied")
        assert path == fallback

    monkeypatch.setattr(Path, "mkdir", fake_mkdir)
    resolution = resolve_model_cache(
        hf_model_id="owner/model",
        topology="P150",
        append_topology_to_tt_cache_path=True,
        create_mode="explicit_only",
        permission_fallback=True,
        environ={
            "TT_CACHE_PATH": str(tmp_path / "readonly"),
            "TT_CACHE_FALLBACK_PATH": str(tmp_path / "fallback"),
        },
    )
    assert resolution.path == fallback
    assert resolution.fallback_from == configured
    assert resolution.source == "permission_fallback"


@pytest.mark.host
def test_cache_identity_is_complete_stable_and_sensitive():
    inputs = dict(
        hf_model_id="owner/model",
        hf_revision="abc123",
        architecture="wormhole_b0",
        topology="N300",
        dtype_inputs={"weights": "bfloat8_b"},
        layout_inputs={"weights": "TILE_LAYOUT"},
        sharding_inputs={"mesh_shape": (1, 2), "num_devices": 2},
    )
    first = CacheIdentity.resolve(**inputs)
    second = CacheIdentity.resolve(**inputs)
    assert first.digest == second.digest
    changed_inputs = (
        {"hf_revision": "different"},
        {"architecture": "blackhole"},
        {"topology": "T3K"},
        {"dtype_inputs": {"weights": "bfloat16"}},
        {"layout_inputs": {"weights": "ROW_MAJOR_LAYOUT"}},
        {"sharding_inputs": {"mesh_shape": (2, 1), "num_devices": 2}},
        {"conversion_schema": "tttv2-lazy-weight-v2"},
    )
    assert all(first.digest != CacheIdentity.resolve(**{**inputs, **change}).digest for change in changed_inputs)
    assert first.tt_transformers_version
    assert first.ttnn_version
    assert first.conversion_schema


@pytest.mark.host
def test_implicit_namespace_invalidates_on_checkpoint_dtype_architecture_and_versions(monkeypatch, tmp_path):
    versions = {"tt-transformers": "1.0", "ttnn": "0.77.0"}
    monkeypatch.setattr(cache_policy, "_installed_version", lambda distribution, _fallback: versions[distribution])

    def resolve(*, revision="revision-a", dtype="bfloat8_b", arch="wormhole"):
        mesh = SimpleNamespace(arch=lambda: arch, shape=(1, 2), get_num_devices=lambda: 2)
        return resolve_model_cache(
            hf_model_id="owner/model",
            hf_revision=revision,
            topology="N300",
            mesh_device=mesh,
            dtype=dtype,
            create_mode="never",
            environ={"HOME": str(tmp_path / "home")},
        )

    baseline = resolve()
    assert baseline.path.name == "N300"
    assert baseline.path.parent.name == f"identity-1-{baseline.cache_identity.digest}"
    assert len({baseline.path, resolve(revision="revision-b").path, resolve(dtype="bfloat16").path, resolve(arch="blackhole").path}) == 4
    versions["tt-transformers"] = "1.1"
    package_changed = resolve()
    versions["ttnn"] = "0.78.0"
    ttnn_changed = resolve()
    assert len({baseline.path, package_changed.path, ttnn_changed.path}) == 3


@pytest.mark.host
def test_explicit_paths_do_not_change_when_identity_inputs_change(tmp_path):
    mesh = SimpleNamespace(arch=lambda: "wormhole", shape=(1, 2), get_num_devices=lambda: 2)
    paths = {
        resolve_model_cache(
            hf_model_id="owner/model",
            hf_revision=revision,
            topology="N300",
            mesh_device=mesh,
            dtype=dtype,
            cache_dir=tmp_path / "established",
            create_mode="never",
            environ={},
        ).path
        for revision, dtype in (("revision-a", "bfloat8_b"), ("revision-b", "bfloat16"))
    }
    assert paths == {tmp_path / "established"}


@pytest.mark.host
def test_preflight_is_secret_redacted_and_reports_identity(tmp_path):
    mesh = SimpleNamespace(arch=lambda: "blackhole", shape=(2, 2), get_num_devices=lambda: 4)
    report = model_preflight_report(
        hf_model_id="owner/model",
        hf_revision="revision",
        cache_path=tmp_path / "established",
        topology="P150x4",
        mesh_device=mesh,
        dtype="bfloat8_b",
        cache_dir=tmp_path / "established",
        environ={
            "HOME": str(tmp_path / "home"),
            "CI": "true",
            "HF_TOKEN": "never-report-this",
            "TT_TRANSFORMERS_OFFLINE": "true",
        },
    )
    serialized = json.dumps(report, sort_keys=True)
    assert "never-report-this" not in serialized
    assert report["environment"]["HF_TOKEN"] == "<redacted>"
    assert report["offline"] is True
    assert report["ci"] is True
    assert report["cache"]["identity_applied_to_path"] is False
    assert "not identity-namespaced" in report["cache"]["release_warning"]
    assert report["cache_identity"]["hf_model_id"] == "owner/model"
    assert report["cache_identity"]["hf_revision"] == "revision"
    assert report["cache_identity"]["architecture"] == "blackhole"
    assert report["cache_identity"]["topology"] == "P150x4"
    assert report["cache_identity"]["sha256"]


@pytest.mark.host
def test_implicit_preflight_confirms_the_exact_identity_namespace(tmp_path):
    mesh = SimpleNamespace(arch=lambda: "wormhole", shape=(1, 2), get_num_devices=lambda: 2)
    environ = {"HOME": str(tmp_path / "home")}
    resolution = resolve_model_cache(
        hf_model_id="owner/model",
        hf_revision="revision",
        topology="N300",
        mesh_device=mesh,
        dtype="bfloat8_b",
        create_mode="never",
        environ=environ,
    )
    report = model_preflight_report(
        hf_model_id="owner/model",
        hf_revision="revision",
        cache_path=resolution.path,
        topology="N300",
        mesh_device=mesh,
        dtype="bfloat8_b",
        environ=environ,
    )
    assert report["cache"]["identity_applied_to_path"] is True
    assert report["cache"]["release_warning"] is None
    assert report["cache_identity"]["sha256"] == resolution.cache_identity.digest


@pytest.mark.host
def test_all_twelve_hf_adaptors_use_the_shared_policy():
    adaptors = sorted((ROOT / "src/tt_transformers/models").glob("*/hf_adaptor.py"))
    assert len(adaptors) == 12
    for path in adaptors:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        imported_policy_names = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module == "tt_transformers.cache_environment"
            for alias in node.names
        }
        assert {"offline_mode", "report_model_preflight", "resolve_model_cache"} <= imported_policy_names
        assert 'Path("model_cache")' not in source
        assert "os.getenv" not in source
        assert "os.environ" not in source
        preflight_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "report_model_preflight"
        ]
        assert len(preflight_calls) == 1
        assert "cache_resolution" in {keyword.arg for keyword in preflight_calls[0].keywords}
        cache_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "resolve_model_cache"
        ]
        assert len(cache_calls) == 1
        cache_keywords = {keyword.arg for keyword in cache_calls[0].keywords}
        assert {"hf_model_id", "hf_revision", "topology", "mesh_device", "dtype"} <= cache_keywords


@pytest.mark.host
@pytest.mark.parametrize(
    ("model", "num_devices", "expected_topology", "cache_kwargs"),
    (
        ("deepseek_r1_distill_qwen_14b", 2, "N300", {}),
        ("llama32_1b", 2, "N300", {}),
        ("llama32_3b", 8, "T3K", {}),
        ("llama33_70b", 4, "P150x4", {"sku": "P150x4"}),
        ("llama3_8b", 1, "P150", {}),
        ("mistral_7b", 4, "TP4", {}),
        ("phi4", 2, "N300", {}),
        ("qwen25_72b", 8, "T3K", {}),
        ("qwen25_7b", 2, "N300", {}),
        ("qwen25_coder_32b", 8, "T3K", {}),
        ("qwen2_7b", 2, "N300", {}),
        ("qwen3_32b", 4, "P150x4", {"sku": "P150x4"}),
    ),
)
def test_adaptor_default_cache_keeps_pinned_topology_suffix(
    monkeypatch, tmp_path, model, num_devices, expected_topology, cache_kwargs
):
    path = ROOT / f"src/tt_transformers/models/{model}/hf_adaptor.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    function_name = "_model_cache_path" if model == "llama3_8b" else "_cache_path"
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == function_name)
    namespace = {
        "Path": Path,
        "get_device_name": lambda _mesh: "P150",
        "report_model_preflight": shared_report_model_preflight,
        "resolve_model_cache": resolve_model_cache,
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    monkeypatch.delenv("TT_CACHE_PATH", raising=False)
    monkeypatch.setenv("TT_TRANSFORMERS_CACHE", str(tmp_path / "cache"))
    mesh = SimpleNamespace(
        arch=lambda: "test-arch",
        shape=(1, num_devices),
        get_num_devices=lambda: num_devices,
    )
    if model == "llama3_8b":
        resolved = namespace[function_name]("owner/model", mesh, **cache_kwargs)
    else:
        resolved = namespace[function_name]("owner/model", mesh, None, **cache_kwargs)
    assert resolved.name == expected_topology
    assert resolved.parent.name.startswith("identity-1-")
    assert resolved.parent.parent == tmp_path / "cache/owner/model"
