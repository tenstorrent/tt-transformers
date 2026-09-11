#!/usr/bin/env python3
"""Probe a non-editable tt-transformers wheel installation without extras."""

from __future__ import annotations

import argparse
import ast
import importlib
import importlib.abc
import importlib.metadata
import importlib.util
import json
import pathlib
import site
import sys

BLOCKED_OPTIONAL = {"pytest"}
EXPECTED_VERSIONS = {
    "tt-transformers": "2.0.0.dev0",
    "ttnn": "0.77.0",
    "torch": "2.11.0+cpu",
    "loguru": "0.6.0",
    "transformers": "5.12.1",
}
MODEL_FAMILIES = (
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
REQUIRED_MODULES = (
    "tt_transformers",
    "tt_transformers.cache_environment",
    "tt_transformers.device_ownership",
    "tt_transformers.device_utils",
    "tt_transformers.mesh_utils",
    "tt_transformers.tensor_utils",
    "tt_transformers.modules",
    "tt_transformers.modules.attention.attention_1d",
    "tt_transformers.modules.embedding.embedding_1d",
    "tt_transformers.modules.lazy_buffer",
    "tt_transformers.modules.lazy_weight",
    "tt_transformers.modules.lightweightmodule",
    "tt_transformers.modules.lm_head.lm_head_1d",
    "tt_transformers.modules.mlp.mlp_1d",
    "tt_transformers.modules.mlp.mlp_2d",
    "tt_transformers.modules.rmsnorm.rmsnorm_1d",
    "tt_transformers.modules.rmsnorm.rmsnorm_2d",
    "tt_transformers.modules.rope.rope_1d",
    "tt_transformers.modules.sampling.params",
    "tt_transformers.modules.sampling.penalties_1d",
    "tt_transformers.modules.sampling.sampling_1d",
    "tt_transformers.modules.sampling.sampling_state_1d",
    "tt_transformers.modules.sampling.seed_manager_1d",
    "tt_transformers.modules.tt_ccl",
    "tt_transformers.sampling",
    "tt_transformers.sampling.sampling_params",
    "tt_transformers.llm_runtime",
    "tt_transformers.llm_runtime.config",
    "tt_transformers.llm_runtime.decode",
    "tt_transformers.llm_runtime.execution",
    "tt_transformers.llm_runtime.lane_group",
    "tt_transformers.llm_runtime.output_reader",
    "tt_transformers.llm_runtime.paged_kv_cache",
    "tt_transformers.llm_runtime.program_compiler",
    "tt_transformers.llm_runtime.tensor_resources",
    "tt_transformers.llm_runtime.trace_compiler",
    "tt_transformers.llm_runtime.vllm_adapter",
    "tt_transformers.llm_runtime.warmup",
    "tt_transformers.llm_runtime.prefill.config",
    "tt_transformers.llm_runtime.prefill.inputs",
    "tt_transformers.llm_runtime.prefill.plan",
    "tt_transformers.llm_runtime.prefill.postprocess",
    "tt_transformers.llm_runtime.prefill.result_collector",
    "tt_transformers.llm_runtime.prefill.runtime",
    "tt_transformers.llm_runtime.prefill.sampling_helpers",
    "tt_transformers.llm_runtime.prefill.sequence_runner",
    "tt_transformers.llm_runtime.prefill.signatures",
    "tt_transformers.llm_runtime.prefill.trace",
    "tt_transformers.models",
    "tt_transformers.models.executor",
    "tt_transformers.models.hf_generation",
    "tt_transformers.models.llama3_executor",
    "tt_transformers.models.qwen2_executor",
)


class BlockOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        del path, target
        if fullname.split(".", 1)[0] in BLOCKED_OPTIONAL:
            raise ImportError(f"blocked optional dependency: {fullname}")
        return None


def assert_optional_absent() -> None:
    for distribution in BLOCKED_OPTIONAL:
        try:
            installed = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            installed = None
        if installed is not None:
            raise AssertionError(f"optional distribution unexpectedly installed: {distribution}=={installed}")
        if importlib.util.find_spec(distribution) is not None:
            raise AssertionError(f"optional import unexpectedly resolvable: {distribution}")


def audit_installed_source(package_root: pathlib.Path) -> dict[str, int]:
    python_files = sorted(package_root.rglob("*.py"))
    for path in python_files:
        source = path.read_text(encoding="utf-8")
        if any(marker in source for marker in ("/localdev/", "/home/gwang/", "/tmp/gwang/")):
            raise AssertionError(f"workspace absolute path in installed source: {path}")
        tree = ast.parse(source, filename=str(path))
        for node in ast.walk(tree):
            modules = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules = [node.module]
            for module in modules:
                root = module.split(".", 1)[0]
                if root == "models" or root in {"tests", "examples", "pytest"}:
                    raise AssertionError(f"forbidden installed import {path}:{node.lineno}: {module}")
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in {"SetDefaultDevice", "GetDefaultDevice"}
                and path.name != "device_ownership.py"
            ):
                raise AssertionError(f"unscoped default-device access {path}:{node.lineno}")
    return {"python_files": len(python_files)}


def probe() -> dict[str, object]:
    assert_optional_absent()
    sys.meta_path.insert(0, BlockOptional())
    modules = list(REQUIRED_MODULES)
    for family in MODEL_FAMILIES:
        modules.extend(
            (
                f"tt_transformers.models.{family}",
                f"tt_transformers.models.{family}.model",
                f"tt_transformers.models.{family}.hf_generator",
                f"tt_transformers.models.{family}.vllm_generator",
            )
        )
    for module in modules:
        importlib.import_module(module)

    package = importlib.import_module("tt_transformers")
    package_file = pathlib.Path(package.__file__).resolve()
    package_root = package_file.parent
    site_roots = [pathlib.Path(entry).resolve() for entry in site.getsitepackages()]
    if not any(package_file.is_relative_to(root) for root in site_roots):
        raise AssertionError(f"package is not under site-packages: {package_file}")
    versions = {name: importlib.metadata.version(name) for name in EXPECTED_VERSIONS}
    if versions != EXPECTED_VERSIONS:
        raise AssertionError(f"version mismatch: {versions}")
    loaded_roots = {module.split(".", 1)[0] for module in sys.modules}
    if BLOCKED_OPTIONAL & loaded_roots:
        raise AssertionError(f"optional imports loaded: {sorted(BLOCKED_OPTIONAL & loaded_roots)}")

    import ttnn

    from tt_transformers import device_utils
    from tt_transformers.cache_environment import model_preflight_report
    from tt_transformers.device_ownership import (
        DEFAULT_DEVICE_FALLBACK_LEDGER,
        active_default_device_owners,
        compatibility_default_device,
        default_device_scope,
    )
    from tt_transformers.llm_runtime import tensor_resources
    from tt_transformers.models.executor import ModelExecutor
    from tt_transformers.tensor_utils import program_config_to_dict

    cleanup = {
        device_utils: ("cleanup_ttnn_value", "cleanup_object_graph", "cleanup_model_case", "cleanup_dp_model_case"),
        tensor_resources: (
            "attach_cleanup_failures",
            "best_effort_deallocate_owned_tensors",
            "raise_cleanup_failures",
            "release_orphans",
        ),
    }
    for module, names in cleanup.items():
        for name in names:
            if not callable(getattr(module, name)):
                raise AssertionError(f"cleanup surface is not callable: {module.__name__}.{name}")
    for value in (default_device_scope, compatibility_default_device, ModelExecutor.cleanup):
        if not callable(value):
            raise AssertionError(f"ownership surface is not callable: {value}")
    if len(DEFAULT_DEVICE_FALLBACK_LEDGER) != 10 or active_default_device_owners():
        raise AssertionError("default-device ownership state mismatch")

    class Mesh:
        shape = (1, 2)

        def arch(self):
            return "wormhole_b0"

        def get_num_devices(self):
            return 2

    report = model_preflight_report(
        hf_model_id="owner/model",
        hf_revision="revision",
        cache_path=pathlib.Path("/tmp/cache"),
        topology="N300",
        mesh_device=Mesh(),
        dtype="bfloat8_b",
        environ={"CI": "true", "HF_TOKEN": "secret"},
    )
    if report["offline"] or report["environment"]["HF_TOKEN"] != "<redacted>":
        raise AssertionError("cache preflight policy mismatch")
    if report["cache_identity"]["ttnn_version"] != "0.77.0":
        raise AssertionError("cache identity did not report installed TTNN")
    sdpa = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
        q_chunk_size=256,
        k_chunk_size=256,
    )
    serialized_sdpa = program_config_to_dict(sdpa)
    if serialized_sdpa["compute_with_storage_grid_size"] != {"x": 8, "y": 8}:
        raise AssertionError(f"TTNN 0.77 program-config serialization mismatch: {serialized_sdpa}")

    distribution = importlib.metadata.distribution("tt-transformers")
    distribution_root = pathlib.Path(distribution.locate_file("")).resolve()
    if not any(distribution_root.is_relative_to(root) for root in site_roots):
        raise AssertionError(f"distribution files are not under site-packages: {distribution_root}")
    source_audit = audit_installed_source(package_root)
    return {
        "python": sys.version.split()[0],
        "package_file": str(package_file),
        "distribution_root": str(distribution_root),
        "versions": versions,
        "imported_modules": len(modules),
        "model_families": len(MODEL_FAMILIES),
        "optional_distributions_absent": sorted(BLOCKED_OPTIONAL),
        "optional_modules_loaded": [],
        "cleanup_ownership": "pass",
        "cache_device_preflight": "pass",
        "ttnn_077_program_config_serialization": "pass",
        "static_boundary": "pass",
        **source_audit,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()
    result = probe()
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
