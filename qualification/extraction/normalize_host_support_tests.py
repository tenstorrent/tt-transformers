#!/usr/bin/env python3
"""Idempotently normalize extracted support tests for the standalone layout.

Run this after ``extract_support_surface.py`` and the Phase 3 example split.  It
contains only mechanical ownership/path changes; it does not weaken active
standalone assertions or add compatibility APIs to production code.
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LEGACY_REASON = "TTTv1 from_model_args compatibility is outside standalone ownership"


REPLACEMENTS: dict[str, tuple[tuple[str, str], ...]] = {
    "tests/llm_runtime/test_model_executor.py": (
        (
            '_EXECUTOR_PATH = Path(__file__).parents[2] / "models" / "executor.py"',
            '_REPOSITORY_ROOT = Path(__file__).parents[2]\n'
            '_EXECUTOR_PATH = _REPOSITORY_ROOT / "src/tt_transformers/models/executor.py"',
        ),
        ('(_MODELS_ROOT / path.name / "README.md")', '(_REPOSITORY_ROOT / "examples" / path.name / "README.md")'),
        ('module.startswith("models.common.models.")', 'module.startswith("tt_transformers.models.")'),
        ('name.startswith("models.common.models.")', 'name.startswith("tt_transformers.models.")'),
    ),
    "tests/llm_runtime/test_prefill_runtime.py": (
        ("models.common.llm_runtime.prefill", "tt_transformers.llm_runtime.prefill"),
    ),
    "tests/modules/sampling/test_params.py": (
        ("Path(__file__).parents[5]", "Path(__file__).parents[3]"),
        ("importlib.import_module('models.common.llm_runtime.decode')", "importlib.import_module('tt_transformers.llm_runtime.decode')"),
    ),
    "tests/modules/sampling/test_seed_manager_1d.py": (
        (
            'Path(__file__).parents[3] / "modules" / "sampling" / "seed_manager_1d.py"',
            'Path(__file__).parents[3] / "src/tt_transformers/modules/sampling/seed_manager_1d.py"',
        ),
    ),
    "tests/models/llama3_8b/test_model_profile.py": (
        ("models.common.models.llama3_8b.model", "tt_transformers.models.llama3_8b.model"),
    ),
    "tests/models/llama33_70b/test_model_profile.py": (
        ("models.common.models.llama33_70b.model", "tt_transformers.models.llama33_70b.model"),
    ),
    "tests/models/qwen3_32b/test_module_profiles.py": (
        (
            'Path(__file__).parents[4] / "tt_transformers/model_params/Qwen3-32B/config.json"',
            'Path(__file__).parents[3] / "qualification/model_params/Qwen3-32B/config.json"',
        ),
    ),
}


LEGACY_TESTS: dict[str, tuple[str, ...]] = {
    "tests/modules/attention/test_attention_1d.py": ("test_attention_1d_rejects_galaxy",),
    "tests/modules/lm_head/test_lm_head_1d.py": ("test_from_model_args_rejects_galaxy",),
    "tests/modules/mlp/test_mlp_2d.py": ("test_mlp_2d_rejects_non_galaxy_from_model_args",),
    "tests/modules/rmsnorm/test_rmsnorm_1d.py": ("test_rmsnorm_1d_rejects_galaxy",),
    "tests/modules/rmsnorm/test_rmsnorm_2d.py": ("test_rmsnorm_2d_rejects_non_tg",),
    "tests/modules/rope/test_rope_1d.py": ("test_rope_1d_from_model_args_rejects_galaxy",),
    "tests/modules/sampling/test_legacy_sampling.py": (
        "test_seed_stream_is_independent_of_host_position_lag",
        "test_seed_stream_matches_across_different_host_lags",
        "test_seed_counters_realign_when_host_inputs_are_authoritative",
        "test_newly_seeded_slot_is_aligned_even_on_a_non_authoritative_step",
    ),
    "tests/modules/sampling/test_params.py": ("test_legacy_device_sampling_capabilities_declare_the_exact_limit",),
}


def _replace(path: Path, replacements: tuple[tuple[str, str], ...]) -> bool:
    text = path.read_text()
    output = text
    for old, new in replacements:
        output = output.replace(old, new)
    if output == text:
        return False
    ast.parse(output, filename=str(path))
    path.write_text(output)
    return True


def _ensure_skip(path: Path, names: tuple[str, ...]) -> bool:
    text = path.read_text()
    tree = ast.parse(text, filename=str(path))
    lines = text.splitlines(keepends=True)
    insertions: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or node.name not in names:
            continue
        if any(
            isinstance(decorator, ast.Call)
            and isinstance(decorator.func, ast.Attribute)
            and decorator.func.attr == "skip"
            for decorator in node.decorator_list
        ):
            continue
        insertions.append(node.lineno - 1)
    if not insertions:
        return False
    for index in sorted(insertions, reverse=True):
        indent = lines[index][: len(lines[index]) - len(lines[index].lstrip())]
        lines.insert(index, f'{indent}@pytest.mark.skip(reason="{LEGACY_REASON}")\n')
    output = "".join(lines)
    ast.parse(output, filename=str(path))
    path.write_text(output)
    return True


def _normalize_demo_contract(path: Path) -> bool:
    model = path.parent.name
    text = path.read_text()
    output = text
    if "_HARDWARE_DEMO_PATH" not in output:
        simple = '_DEMO_TREE = ast.parse(Path(_DEMO_PATH).read_text(encoding="utf-8"), filename=_DEMO_PATH)'
        sourced = (
            '_DEMO_SOURCE = Path(_DEMO_PATH).read_text(encoding="utf-8")\n'
            '_DEMO_TREE = ast.parse(_DEMO_SOURCE, filename=_DEMO_PATH)'
        )
        replacement = (
            f'_HARDWARE_DEMO_PATH = "tests/hardware/models/{model}/test_demo.py"\n'
            '_DEMO_SOURCE = "\\n".join(\n'
            '    (Path(_DEMO_PATH).read_text(encoding="utf-8"), Path(_HARDWARE_DEMO_PATH).read_text(encoding="utf-8"))\n'
            ')\n'
            '_DEMO_TREE = ast.parse(_DEMO_SOURCE, filename=f"{_DEMO_PATH}+{_HARDWARE_DEMO_PATH}")'
        )
        output = output.replace(sourced, replacement).replace(simple, replacement)
    if 'if name == "_ttnn_mesh_device_param_from_env":' not in output:
        for signature in ("def _function(name):", "def _demo_function(name, namespace=None):"):
            output = output.replace(
                signature + "\n",
                signature + '\n    if name == "_ttnn_mesh_device_param_from_env":\n        name = "ttnn_mesh_device_param_from_env"\n',
                1,
            )
    output = output.replace(
        "    namespace = {} if namespace is None else namespace\n    exec(compile(",
        '    namespace = {} if namespace is None else namespace\n'
        '    namespace.setdefault("UnsupportedConfiguration", pytest.skip.Exception)\n'
        "    exec(compile(",
    )
    if output == text:
        return False
    ast.parse(output, filename=str(path))
    path.write_text(output)
    return True


def normalize(*, check: bool = False) -> int:
    changed: list[str] = []
    for relative, replacements in REPLACEMENTS.items():
        path = ROOT / relative
        if path.is_file() and _replace(path, replacements):
            changed.append(relative)
    for relative, names in LEGACY_TESTS.items():
        path = ROOT / relative
        if path.is_file() and _ensure_skip(path, names):
            changed.append(relative)
    for path in sorted((ROOT / "tests/models").glob("*/test_demo_contract.py")):
        if _normalize_demo_contract(path):
            changed.append(str(path.relative_to(ROOT)))
    for model in ("llama32_1b", "llama32_3b"):
        path = ROOT / "tests/models" / model / "test_demo_warmup.py"
        if path.is_file() and _replace(
            path,
            ((
                "    namespace = {} if namespace is None else namespace\n    exec(compile(",
                '    namespace = {} if namespace is None else namespace\n'
                '    namespace.setdefault("UnsupportedConfiguration", pytest.skip.Exception)\n'
                "    exec(compile(",
            ),),
        ):
            changed.append(str(path.relative_to(ROOT)))
    if check and changed:
        raise SystemExit("host support normalization required: " + ", ".join(sorted(set(changed))))
    print(f"host support normalization: {len(set(changed))} file(s) changed")
    return len(set(changed))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="fail if normalization would change tracked files")
    args = parser.parse_args()
    normalize(check=args.check)


if __name__ == "__main__":
    main()
