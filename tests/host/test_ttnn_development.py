# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Host regression tests for editable/native and fixed-wheel development runtimes."""

from __future__ import annotations

import base64
import hashlib
import importlib.metadata
import io
import json
import subprocess
import tarfile
import zipfile
from pathlib import Path

import pytest

from tools.ttnn_dev_support import build, common, environment
from tools.ttnn_dev_support import probe as import_probe


def git(checkout, *args):
    return subprocess.check_output(["git", "-C", str(checkout), *args], text=True).strip()


def repository(path):
    path.mkdir()
    git(path, "init", "-q")
    git(path, "config", "user.email", "tests@example.invalid")
    git(path, "config", "user.name", "Runtime tests")
    (path / ".gitignore").write_text("build/\n")
    (path / "ttnn/ttnn/operations").mkdir(parents=True)
    (path / "ttnn/ttnn/__init__.py").write_text("value = 1\n")
    (path / "ttnn/ttnn/operations/example.py").write_text("value = 1\n")
    (path / "native.cpp").write_text("int value = 1;\n")
    git(path, "add", ".")
    git(path, "commit", "-qm", "Initial")
    return path


def runtime(tmp_path, *, mode="source"):
    root = tmp_path / "runtime"
    (root / "dependencies").mkdir(parents=True)
    lock = root / "dependencies/requirements.txt"
    lock.write_text("example==1 --hash=sha256:" + "a" * 64 + "\n")
    data = {
        "mode": mode,
        "metal": {"sha": "a" * 40},
        "python": {"abi": "cpython-310", "minor": "3.10"},
        "recipe": {"name": "test"},
        "ttnn": {"version": "0.78.0.dev1"},
        "dependencies": {"directory": "dependencies", "extras": [], "requirements": ["torch==2.11.0"]},
        "toolchain": {"compiler_sha256": "b" * 64},
        "source": {
            "checkout": str(tmp_path / "checkout"),
            "build_dir": str(tmp_path / "build"),
            "native_fingerprint": "c" * 64,
            "outputs": {"_ttnn.so": "d" * 64},
        },
        "files": [common.artifact_entry(root, lock)],
    }
    manifest = root / "runtime.json"
    common.save_runtime(manifest, data)
    return manifest, data


@pytest.mark.parametrize("relative", ["ttnn/ttnn/__init__.py", "ttnn/ttnn/operations/example.py"])
@pytest.mark.host
def test_python_edits_are_live_without_claiming_a_new_native_build(tmp_path, relative):
    checkout = repository(tmp_path / "metal")
    before = common.source_state(checkout)
    (checkout / relative).write_text("value = 2\n")
    after = common.source_state(checkout)
    assert after["dirty"]
    assert after["fingerprint"] != before["fingerprint"]
    assert after["native_fingerprint"] == before["native_fingerprint"]


@pytest.mark.host
def test_native_edits_require_rebuild(tmp_path):
    checkout = repository(tmp_path / "metal")
    before = common.source_state(checkout)
    (checkout / "native.cpp").write_text("int value = 2;\n")
    assert common.source_state(checkout)["native_fingerprint"] != before["native_fingerprint"]


@pytest.mark.host
def test_untracked_build_inputs_are_attributed(tmp_path):
    checkout = repository(tmp_path / "metal")
    before = common.source_state(checkout)
    (checkout / "new-config.json").write_text('{"value": 1}')
    after = common.source_state(checkout)
    assert after["dirty"]
    assert "new-config.json" in after["untracked"]
    assert after["native_fingerprint"] != before["native_fingerprint"]


@pytest.mark.host
def test_dirty_submodule_contents_affect_parent_identity(tmp_path):
    child = repository(tmp_path / "child")
    parent = repository(tmp_path / "parent")
    git(parent, "-c", "protocol.file.allow=always", "submodule", "add", "-q", str(child), "dep")
    git(parent, "commit", "-qam", "Submodule")
    before = common.source_state(parent)
    (parent / "dep/native.cpp").write_text("int value = 3;\n")
    first = common.source_state(parent)
    (parent / "dep/native.cpp").write_text("int value = 4;\n")
    second = common.source_state(parent)
    assert first["dirty"] and second["dirty"]
    assert len({before["fingerprint"], first["fingerprint"], second["fingerprint"]}) == 3


@pytest.mark.host
def test_runtime_manifest_detects_changed_artifacts_and_metadata(tmp_path):
    manifest, data = runtime(tmp_path)
    assert common.load_runtime(manifest)["ttnn"]["version"] == "0.78.0.dev1"
    changed = json.loads(manifest.read_text())
    changed["ttnn"]["version"] = "0.77.0"
    manifest.write_text(json.dumps(changed))
    with pytest.raises(common.DevError, match="fingerprint mismatch"):
        common.load_runtime(manifest)
    common.save_runtime(manifest, data)
    (manifest.parent / "dependencies/requirements.txt").write_text("changed")
    with pytest.raises(common.DevError, match="missing or modified"):
        common.load_runtime(manifest)


@pytest.mark.host
def test_runtime_requires_fingerprinted_wheel_and_toolchain(tmp_path):
    manifest, data = runtime(tmp_path, mode="wheel")
    data["ttnn"]["wheel"] = "unverified.whl"
    data["toolchain"]["archive"] = "sfpi.tar.gz"
    common.save_runtime(manifest, data)
    with pytest.raises(common.DevError, match="Wheel/toolchain"):
        common.load_runtime(manifest)


@pytest.mark.parametrize("relative", ["../outside", "/absolute"])
@pytest.mark.host
def test_manifest_paths_cannot_escape_bundle(tmp_path, relative):
    with pytest.raises(common.DevError, match="escapes"):
        common.within(tmp_path, relative)


@pytest.mark.host
def test_symlink_cannot_escape_bundle(tmp_path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "outside").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(common.DevError, match="escapes"):
        common.within(bundle, "outside/file")


@pytest.mark.host
def test_source_runtime_refuses_missing_build_identity(tmp_path):
    manifest, data = runtime(tmp_path)
    data["source"]["outputs"] = {}
    common.save_runtime(manifest, data)
    with pytest.raises(common.DevError, match="native build outputs"):
        common.load_runtime(manifest)


@pytest.mark.host
def test_rebuild_cannot_run_while_runtime_is_in_use(tmp_path):
    path = tmp_path / "runtime.lock"
    with common.runtime_lock(path, exclusive=False):
        with pytest.raises(common.DevError, match="Runtime is in use"):
            with common.runtime_lock(path, exclusive=True):
                pytest.fail("Exclusive rebuild lock was granted")
    with common.runtime_lock(path, exclusive=True):
        pass


@pytest.mark.host
def test_ref_resolution_peels_annotated_tags(monkeypatch):
    monkeypatch.setattr(
        build, "command", lambda *a, **k: "a" * 40 + "\trefs/tags/v1\n" + "b" * 40 + "\trefs/tags/v1^{}"
    )
    assert build.resolve_ref("v1") == "b" * 40


@pytest.mark.host
def test_ref_resolution_refuses_ambiguity_and_shell_fragments(monkeypatch):
    monkeypatch.setattr(
        build, "command", lambda *a, **k: "a" * 40 + "\trefs/tags/main\n" + "b" * 40 + "\trefs/heads/main"
    )
    with pytest.raises(common.DevError, match="ambiguous"):
        build.resolve_ref("main")
    with pytest.raises(common.DevError, match="Use a Git"):
        build.resolve_ref("main; echo unexpected")


@pytest.mark.host
def test_development_requirements_replace_only_ttnn(tmp_path):
    (tmp_path / "pyproject.toml").write_text(
        '[project]\ndependencies=["ttnn==0.77.0", "torch==2.11.0", "transformers==5.12.1"]\n'
        '[project.optional-dependencies]\ntest=["pytest==9.0.3"]\n'
    )
    assert environment.requirements(tmp_path, ["test"]) == ["pytest==9.0.3", "torch==2.11.0", "transformers==5.12.1"]
    with pytest.raises(common.DevError, match="Unknown project extra"):
        environment.requirements(tmp_path, ["absent"])


@pytest.mark.host
def test_wrong_python_abi_profile_is_rejected(monkeypatch):
    monkeypatch.setattr(common, "command", lambda *a, **k: json.dumps({"minor": "3.9", "machine": "x86_64"}))
    with pytest.raises(common.DevError, match="CPython 3.10/3.12"):
        common.interpreter_info("python")


@pytest.mark.host
def test_native_build_does_not_replace_an_existing_build_tree(tmp_path, monkeypatch):
    checkout = repository(tmp_path / "metal")
    existing = tmp_path / "existing"
    existing.mkdir()
    (checkout / "build").symlink_to(existing, target_is_directory=True)
    settings = {"checkout": str(checkout), "build_dir": str(tmp_path / "new")}
    monkeypatch.setattr(build, "command", lambda *a, **k: pytest.fail("Build should not start"))
    with pytest.raises(common.DevError, match="already selects another build"):
        build.native_build(settings, tmp_path / "venv")
    assert (checkout / "build").resolve() == existing


@pytest.mark.host
def test_native_build_refuses_source_changes_during_compilation(tmp_path, monkeypatch):
    checkout = repository(tmp_path / "metal")
    settings = {
        "checkout": str(checkout),
        "build_dir": str(tmp_path / "native"),
        "work": str(tmp_path),
        "jobs": 2,
        "build_type": "Release",
        "distributed": False,
    }
    monkeypatch.setattr(build, "command", lambda *a, **k: (checkout / "native.cpp").write_text("int value = 5;\n"))
    with pytest.raises(common.DevError, match="changed during compilation"):
        build.native_build(settings, tmp_path / "venv")


@pytest.mark.parametrize("kind", ["traversal", "symlink"])
@pytest.mark.host
def test_toolchain_extraction_rejects_unsafe_members(tmp_path, kind):
    archive = tmp_path / "sfpi.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        member = tarfile.TarInfo("../outside" if kind == "traversal" else "compiler")
        if kind == "symlink":
            member.type = tarfile.SYMTYPE
            member.linkname = "/outside"
        tar.addfile(member, io.BytesIO(b""))
    with pytest.raises(common.DevError, match="escapes|Unexpected"):
        environment.extract_toolchain(archive, tmp_path / "extract")


@pytest.mark.host
def test_installed_wheel_is_verified_against_its_record(tmp_path):
    package = tmp_path / "site-packages/ttnn"
    package.mkdir(parents=True)
    data = b"native extension contents"
    (package / "_ttnn.so").write_bytes(data)
    checksum = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode().rstrip("=")
    wheel = tmp_path / "ttnn.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("ttnn-1.dist-info/RECORD", f"ttnn/_ttnn.so,sha256={checksum},{len(data)}\n")
    environment.verify_installed_wheel(wheel, package)
    (package / "_ttnn.so").write_bytes(b"changed native extension")
    with pytest.raises(common.DevError, match="Installed wheel file changed"):
        environment.verify_installed_wheel(wheel, package)


@pytest.mark.host
def test_launcher_replaces_stale_runtime_and_cache_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/stale/metal")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/stale/build/lib")
    monkeypatch.setenv("TT_METAL_RUNTIME_ROOT", "/stale/metal")
    monkeypatch.setenv("TT_CACHE_PATH", "/stale/weights")
    settings = {
        "venv": str(tmp_path / "venv"),
        "schema_version": 1,
        "runtime": str(tmp_path / "runtime.json"),
        "project": str(tmp_path / "project"),
        "runtime_root": str(tmp_path / "selected-root"),
    }
    config = tmp_path / "environment.json"
    common.write_json(config, settings)
    env = environment.run_environment(config, {"mode": "wheel"}, identity="abc")
    assert env["TT_METAL_RUNTIME_ROOT"] == settings["runtime_root"]
    assert "LD_LIBRARY_PATH" not in env
    assert env["PYTHONPATH"].split(":")[0] == str(tmp_path / "project/src")
    assert "/cache/abc/" in env["TT_CACHE_PATH"]


@pytest.mark.host
def test_doctor_refuses_stale_native_inputs_before_importing_ttnn(tmp_path, monkeypatch):
    manifest, data = runtime(tmp_path)
    config = manifest.parent / "environment.json"
    common.write_json(
        config,
        {
            "schema_version": 1,
            "runtime": str(manifest),
            "venv": str(tmp_path / "venv"),
            "project": str(tmp_path),
            "runtime_root": data["source"]["checkout"],
        },
    )
    monkeypatch.setattr(environment, "runtime_python", lambda path: Path("python"))
    monkeypatch.setattr(environment, "interpreter_info", lambda path: data["python"])
    monkeypatch.setattr(environment, "requirements", lambda *a: data["dependencies"]["requirements"])
    monkeypatch.setattr(environment, "source_state", lambda path: {"native_fingerprint": "changed"})
    monkeypatch.setattr(environment, "command", lambda *a, **k: pytest.fail("No imports before freshness check"))
    with pytest.raises(common.DevError, match="Native tt-metal inputs changed"):
        environment.doctor(config)


@pytest.mark.host
def test_source_egg_info_is_not_treated_as_an_installed_distribution(tmp_path, monkeypatch):
    source = tmp_path / "src"
    egg = source / "tt_transformers.egg-info"
    egg.mkdir(parents=True)
    (egg / "PKG-INFO").write_text("Metadata-Version: 2.1\nName: tt-transformers\nVersion: 2.0.0.dev0\n")
    installed = tmp_path / "site-packages"
    installed.mkdir()
    monkeypatch.syspath_prepend(str(source))
    monkeypatch.setattr(import_probe.site, "getsitepackages", lambda: [str(installed)])
    assert importlib.metadata.version("tt-transformers") == "2.0.0.dev0"
    assert import_probe.installed_project_version() is None
    # Use a separate environment, as the real probe runs in a fresh interpreter.
    # importlib.metadata caches directory entries on some Python/filesystem pairs.
    populated = tmp_path / "populated-site-packages"
    info = populated / "tt_transformers-2.0.0.dev0.dist-info"
    info.mkdir(parents=True)
    (info / "METADATA").write_text((egg / "PKG-INFO").read_text())
    monkeypatch.setattr(import_probe.site, "getsitepackages", lambda: [str(populated)])
    assert import_probe.installed_project_version() == "2.0.0.dev0"


@pytest.mark.host
def test_native_inventory_uses_compiler_and_cmake_dependency_graph(tmp_path, monkeypatch):
    checkout = tmp_path / "metal"
    checkout.mkdir()
    (checkout / "CMakeLists.txt").write_text("project(example)")
    header = tmp_path / "external.hpp"
    header.write_text("constexpr int value = 1;")
    build_dir = tmp_path / "build"
    reply = build_dir / ".cmake/api/v1/reply"
    reply.mkdir(parents=True)
    common.write_json(reply / "cmakeFiles.json", {"inputs": [{"path": "CMakeLists.txt"}]})
    common.write_json(reply / "index-1.json", {"reply": {"cmakeFiles-v1": {"jsonFile": "cmakeFiles.json"}}})
    monkeypatch.setattr(build, "command", lambda *a, **k: f"output.o: #deps 1\n    {header}\n")
    inputs = build.native_inputs(checkout, build_dir)
    assert str(header) in inputs
    assert str(checkout / "CMakeLists.txt") in inputs
    assert inputs[str(header)]["sha256"] == common.file_hash(header)


@pytest.mark.host
def test_explicit_tag_ref_is_peeled(monkeypatch):
    def resolve(argv, **kwargs):
        assert "refs/tags/v1^{}" in argv
        return "a" * 40 + "\trefs/tags/v1\n" + "b" * 40 + "\trefs/tags/v1^{}"

    monkeypatch.setattr(build, "command", resolve)
    assert build.resolve_ref("refs/tags/v1") == "b" * 40


@pytest.mark.host
def test_container_executor_preserves_paths_and_replaces_executor(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from tools import ttnn_dev

    image = "example/image@sha256:" + "a" * 64
    argv = ["build", "--output", str(tmp_path / "out"), "--executor=local-container", "--image", image]
    args = ttnn_dev.parser().parse_args(argv)
    calls = []
    monkeypatch.setattr(ttnn_dev.shutil, "which", lambda name: "/usr/bin/docker")
    original_run = subprocess.run

    def run(command, **kwargs):
        if command[0] == "/usr/bin/docker":
            calls.append(command)
            return SimpleNamespace(returncode=0)
        return original_run(command, **kwargs)

    monkeypatch.setattr(ttnn_dev.subprocess, "run", run)
    assert ttnn_dev.container_execution(args, argv) == 0
    assert "--executor=native" in calls[0]
    assert f"type=bind,src={tmp_path / 'out'},dst={tmp_path / 'out'}" in calls[0]
    assert image in calls[0]


@pytest.mark.host
def test_container_executor_mounts_worktree_git_metadata(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from tools import ttnn_dev

    parent = repository(tmp_path / "parent")
    checkout = tmp_path / "worktree"
    git(parent, "worktree", "add", "--detach", str(checkout), "HEAD")
    image = "example/image@sha256:" + "a" * 64
    argv = [
        "attach",
        "--tt-metal-checkout",
        str(checkout),
        "--output",
        str(tmp_path / "out"),
        "--executor",
        "local-container",
        "--image",
        image,
    ]
    args = ttnn_dev.parser().parse_args(argv)
    calls = []
    original_run = subprocess.run

    def run(command, **kwargs):
        if command[0] == "/usr/bin/docker":
            calls.append(command)
            return SimpleNamespace(returncode=0)
        return original_run(command, **kwargs)

    monkeypatch.setattr(ttnn_dev.shutil, "which", lambda name: "/usr/bin/docker")
    monkeypatch.setattr(ttnn_dev.subprocess, "run", run)
    assert ttnn_dev.container_execution(args, argv) == 0
    common_dir = parent / ".git"
    assert f"type=bind,src={common_dir},dst={common_dir}" in calls[0]


@pytest.mark.host
def test_ci_dispatch_pins_ref_and_rejects_the_wrong_artifact(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from tools import ttnn_dev

    args = ttnn_dev.parser().parse_args(
        ["build", "--executor", "ci", "--output", str(tmp_path), "--workflow-ref", "feature"]
    )
    monkeypatch.setattr(ttnn_dev, "resolve_ref", lambda ref: "a" * 40)
    monkeypatch.setattr(ttnn_dev, "interpreter_info", lambda *a: pytest.fail("CI must not require local target Python"))

    def dispatch(argv, **kwargs):
        payload = json.loads(kwargs["input"])
        assert payload["ref"] == "feature"
        assert payload["inputs"]["metal_ref"] == "a" * 40
        assert payload["inputs"]["request_id"]
        return SimpleNamespace(returncode=0, stdout='{"workflow_run_id": 42}', stderr="")

    monkeypatch.setattr(ttnn_dev.subprocess, "run", dispatch)
    monkeypatch.setattr(ttnn_dev, "command", lambda *a, **k: "")
    monkeypatch.setattr(ttnn_dev, "load_runtime", lambda path: {"mode": "wheel", "metal": {"sha": "b" * 40}})
    with pytest.raises(common.DevError, match="different source commit"):
        ttnn_dev.dispatch_build(args)


@pytest.mark.host
def test_toolchain_tree_hash_matches_a_dereferenced_export(tmp_path):
    source = tmp_path / "source"
    (source / "lib").mkdir(parents=True)
    (source / "lib/header.hpp").write_text("value")
    (source / "lib64").symlink_to(source / "lib", target_is_directory=True)
    archive = tmp_path / "sfpi.tar.gz"
    with tarfile.open(archive, "w:gz", dereference=True) as tar:
        for path in sorted(source.iterdir()):
            tar.add(path, arcname=path.name)
    target = tmp_path / "installed"
    environment.extract_toolchain(archive, target)
    assert common.tree_hash(source) == common.tree_hash(target)
    (target / "lib64/header.hpp").write_text("changed")
    assert common.tree_hash(source) != common.tree_hash(target)


@pytest.mark.host
def test_git_trust_is_scoped_to_the_selected_checkout(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(common, "command", lambda argv, **kwargs: calls.append(argv) or "revision")
    assert common.git(tmp_path, "rev-parse", "HEAD") == "revision"
    assert calls == [["git", "-c", f"safe.directory={tmp_path.resolve()}", "-C", tmp_path, "rev-parse", "HEAD"]]


@pytest.mark.host
def test_effective_source_requirements_cannot_be_bypassed_by_pip_check(tmp_path):
    lock = tmp_path / "requirements.txt"
    lock.write_text("torch==2.11.0+cpu --hash=sha256:" + "a" * 64 + "\n")
    environment.verify_installed_dependencies(lock, {"torch": "2.11.0+cpu"})
    with pytest.raises(common.DevError, match="Installed torch differs"):
        environment.verify_installed_dependencies(lock, {"torch": "2.7.1+cpu"})
