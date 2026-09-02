#!/usr/bin/env python3
"""Extract the pinned TTTv2 runtime and shared executors.

The source repository is read through Git objects only. The sole source-code
transformations are import-module namespace rewrites plus the explicitly
characterized Phase 3 active-caller migrations in apply_boundary_closures().
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


PINNED_REVISION = "00748e6ac7b65f50e5c2af07f6e7c1c535c7f4c0"
DEFAULT_SOURCE_REPO = Path("/localdev/gwang/tt-metal")
DEFAULT_DESTINATION_ROOT = Path(__file__).resolve().parents[1]
PHASE3_BOUNDARY_CLOSURE_EDITS = 3


@dataclass(frozen=True)
class Extraction:
    source: str
    destination: str
    blob: str


EXTRACTIONS = (
    Extraction(
        "models/common/llm_runtime/config.py",
        "src/tt_transformers/llm_runtime/config.py",
        "34e767bf49743e4d1df23139121d31ad5e4aa0ae",
    ),
    Extraction(
        "models/common/llm_runtime/decode.py",
        "src/tt_transformers/llm_runtime/decode.py",
        "8d1685b68ddb1c41bbf7c31df0259c22cd46e825",
    ),
    Extraction(
        "models/common/llm_runtime/execution.py",
        "src/tt_transformers/llm_runtime/execution.py",
        "5be975228764c390d23d4b856a29cbf6bf3d5738",
    ),
    Extraction(
        "models/common/llm_runtime/lane_group.py",
        "src/tt_transformers/llm_runtime/lane_group.py",
        "4cffe71d197231578ff5d2214bc3946ca6cd0c23",
    ),
    Extraction(
        "models/common/llm_runtime/output_reader.py",
        "src/tt_transformers/llm_runtime/output_reader.py",
        "2635a5b5117ecff139570624830ecf93c3e30df3",
    ),
    Extraction(
        "models/common/llm_runtime/paged_kv_cache.py",
        "src/tt_transformers/llm_runtime/paged_kv_cache.py",
        "ad4e774390d0ee09ee76f56133f070aca176870b",
    ),
    Extraction(
        "models/common/llm_runtime/prefill/config.py",
        "src/tt_transformers/llm_runtime/prefill/config.py",
        "ffe15b61d0446a0712cf6a744a4fa7a43f784027",
    ),
    Extraction(
        "models/common/llm_runtime/prefill/inputs.py",
        "src/tt_transformers/llm_runtime/prefill/inputs.py",
        "edf86ca70500bd543f5d6878ac72fee51f61170d",
    ),
    Extraction(
        "models/common/llm_runtime/prefill/plan.py",
        "src/tt_transformers/llm_runtime/prefill/plan.py",
        "fa0c05779aae434f288ace93415cb2f3b71474cf",
    ),
    Extraction(
        "models/common/llm_runtime/prefill/postprocess.py",
        "src/tt_transformers/llm_runtime/prefill/postprocess.py",
        "c7ab39d4803fc0be6b13356094a32970aa80afb1",
    ),
    Extraction(
        "models/common/llm_runtime/prefill/result_collector.py",
        "src/tt_transformers/llm_runtime/prefill/result_collector.py",
        "03df6a1828f4678ed43f2a0820ed38eb537c35fb",
    ),
    Extraction(
        "models/common/llm_runtime/prefill/runtime.py",
        "src/tt_transformers/llm_runtime/prefill/runtime.py",
        "1bd3b99ff81774af505e66cc8da8bca84d28d939",
    ),
    Extraction(
        "models/common/llm_runtime/prefill/sampling_helpers.py",
        "src/tt_transformers/llm_runtime/prefill/sampling_helpers.py",
        "7ecdb1c92baf703b59259df69d7bffd1cb6928f3",
    ),
    Extraction(
        "models/common/llm_runtime/prefill/sequence_runner.py",
        "src/tt_transformers/llm_runtime/prefill/sequence_runner.py",
        "1f7523f79faaa9f18208cf3c035341fe444887ee",
    ),
    Extraction(
        "models/common/llm_runtime/prefill/signatures.py",
        "src/tt_transformers/llm_runtime/prefill/signatures.py",
        "6f5449e9a2dfeb9f45831e40dc12c1ec1d949a0b",
    ),
    Extraction(
        "models/common/llm_runtime/prefill/trace.py",
        "src/tt_transformers/llm_runtime/prefill/trace.py",
        "499343b812a421953b163f4d995d6b3e96340b82",
    ),
    Extraction(
        "models/common/llm_runtime/program_compiler.py",
        "src/tt_transformers/llm_runtime/program_compiler.py",
        "3ac6ef89614f269861505fe2f5789273a108aa02",
    ),
    Extraction(
        "models/common/llm_runtime/tensor_resources.py",
        "src/tt_transformers/llm_runtime/tensor_resources.py",
        "16d96d7b6e42d66951307947450181e46a3168dd",
    ),
    Extraction(
        "models/common/llm_runtime/trace_compiler.py",
        "src/tt_transformers/llm_runtime/trace_compiler.py",
        "fb48777d15eef1595da21029a377b7b46cb361fd",
    ),
    Extraction(
        "models/common/llm_runtime/vllm_adapter.py",
        "src/tt_transformers/llm_runtime/vllm_adapter.py",
        "5ce2eafa479b73734766445971babac25d870845",
    ),
    Extraction(
        "models/common/llm_runtime/warmup.py",
        "src/tt_transformers/llm_runtime/warmup.py",
        "47b1792045c2152271fe0f74646948b04ef4c1c7",
    ),
    Extraction(
        "models/common/models/executor.py",
        "src/tt_transformers/models/executor.py",
        "bd1d4a8607aa27a75ad63109b046bcc41cd47d11",
    ),
    Extraction(
        "models/common/models/llama3_executor.py",
        "src/tt_transformers/models/llama3_executor.py",
        "9c84008b6b551a97ed2a156aa1d2a4d8994a457a",
    ),
    Extraction(
        "models/common/models/qwen2_executor.py",
        "src/tt_transformers/models/qwen2_executor.py",
        "321d10bd5c1165eecac5ec30d9185d5e1b71321a",
    ),
)


# Longest prefix first. These are namespace moves only; no symbol, signature,
# statement ordering, or executable expression is changed.
REWRITE_RULES = (
    ("models.common.sampling.sampling_params", "tt_transformers.sampling.sampling_params"),
    ("models.common.llm_runtime", "tt_transformers.llm_runtime"),
    ("models.common.models.executor", "tt_transformers.models.executor"),
    ("models.common.modules", "tt_transformers.modules"),
    ("models.common.sampling", "tt_transformers.sampling.sampling_params"),
)

FROM_IMPORT = re.compile(r"^(?P<prefix>\s*from\s+)(?P<module>[A-Za-z_][\w.]*)(?P<suffix>\s+import\b)")
PLAIN_IMPORT = re.compile(r"^(?P<prefix>\s*import\s+)(?P<module>[A-Za-z_][\w.]*)(?P<suffix>(?:\s+as\s+\w+)?\s*(?:#.*)?(?:\r?\n)?$)")


def git(source_repo: Path, *args: str, input_text: str | None = None) -> str:
    return subprocess.check_output(
        ["git", "-C", str(source_repo), *args],
        input=input_text,
        text=True,
        stderr=subprocess.PIPE,
    )


def rewrite_module(module: str) -> str:
    for old, new in REWRITE_RULES:
        if module == old or module.startswith(old + "."):
            return new + module[len(old) :]
    if module == "models" or module.startswith("models."):
        raise ValueError(f"legacy import has no approved rewrite rule: {module}")
    return module


def imported_modules(tree: ast.AST) -> list[str]:
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module)
    return modules


class NormalizeImports(ast.NodeTransformer):
    """Normalize only legacy import modules for an AST equivalence check."""

    def visit_Import(self, node: ast.Import) -> ast.AST:
        for alias in node.names:
            alias.name = rewrite_module(alias.name)
        return node

    def visit_ImportFrom(self, node: ast.ImportFrom) -> ast.AST:
        if node.module:
            node.module = rewrite_module(node.module)
        return node


def rewrite_import_namespaces(source: str, filename: str) -> tuple[str, list[tuple[str, str]]]:
    original_tree = ast.parse(source, filename=filename)
    expected_tree = NormalizeImports().visit(ast.parse(source, filename=filename))
    ast.fix_missing_locations(expected_tree)
    changes: list[tuple[str, str]] = []
    output_lines: list[str] = []

    for line in source.splitlines(keepends=True):
        match = FROM_IMPORT.match(line) or PLAIN_IMPORT.match(line)
        if match is None:
            output_lines.append(line)
            continue
        old_module = match.group("module")
        new_module = rewrite_module(old_module)
        if new_module == old_module:
            output_lines.append(line)
            continue
        output_lines.append(
            line[: match.start("module")] + new_module + line[match.end("module") :]
        )
        changes.append((old_module, new_module))

    rewritten = "".join(output_lines)
    rewritten_tree = ast.parse(rewritten, filename=filename)
    if ast.dump(expected_tree, include_attributes=False) != ast.dump(rewritten_tree, include_attributes=False):
        raise ValueError(f"non-import or incomplete transformation detected in {filename}")

    original_legacy = [module for module in imported_modules(original_tree) if module.startswith("models.")]
    if len(changes) != len(original_legacy):
        raise ValueError(
            f"expected {len(original_legacy)} rewritten import statements in {filename}, got {len(changes)}"
        )
    remaining = [module for module in imported_modules(rewritten_tree) if module.startswith("models.")]
    if remaining:
        raise ValueError(f"legacy imports remain in {filename}: {remaining}")
    return rewritten, changes


def apply_boundary_closures(source: str, filename: str) -> str:
    """Apply exact active-caller migrations required by removed TTTv1 adapters."""

    if filename != "models/common/llm_runtime/decode.py":
        return source
    replacements = (
        (
            "from tt_transformers.modules.sampling.seed_manager_1d import SeedManager1D\n",
            "from tt_transformers.modules.sampling.seed_manager_1d import SeedManager1D\n"
            "from tt_transformers.modules.rope.rope_1d import prepare_rot_idxs\n",
        ),
        (
            "rotary = config.model.rope_setup.get_rot_idxs(nonnegative, on_host=True)",
            "rotary = prepare_rot_idxs(config.model.rope_setup.config, nonnegative, on_host=True)",
        ),
        (
            "rot_mats = model.rope_setup.get_rot_mats(inputs.rotary_indices)",
            "rot_mats = model.rope_setup.decode_forward(inputs.rotary_indices)",
        ),
    )
    for old, new in replacements:
        if source.count(old) != 1:
            raise ValueError(f"expected exactly one characterized decode caller pattern: {old!r}")
        source = source.replace(old, new)
    ast.parse(source, filename=filename)
    return source


def git_blob_sha(source_repo: Path, content: str) -> str:
    return git(source_repo, "hash-object", "--stdin", input_text=content).strip()


def extract(source_repo: Path, destination_root: Path, *, check: bool) -> dict[str, object]:
    resolved_revision = git(source_repo, "rev-parse", "--verify", f"{PINNED_REVISION}^{{commit}}").strip()
    if resolved_revision != PINNED_REVISION:
        raise ValueError(f"pinned revision resolved to {resolved_revision}")
    if len(EXTRACTIONS) != 24:
        raise ValueError(f"manifest must contain 24 files, got {len(EXTRACTIONS)}")
    runtime_count = sum(item.source.startswith("models/common/llm_runtime/") for item in EXTRACTIONS)
    executor_count = len(EXTRACTIONS) - runtime_count
    if (runtime_count, executor_count) != (21, 3):
        raise ValueError(f"manifest split must be 21 runtime + 3 executors, got {runtime_count} + {executor_count}")

    report_rows: list[dict[str, object]] = []
    total_changes = 0
    for item in EXTRACTIONS:
        source = git(source_repo, "show", f"{PINNED_REVISION}:{item.source}")
        actual_blob = git_blob_sha(source_repo, source)
        if actual_blob != item.blob:
            raise ValueError(f"blob mismatch for {item.source}: {actual_blob} != {item.blob}")
        rewritten, changes = rewrite_import_namespaces(source, item.source)
        rewritten = apply_boundary_closures(rewritten, item.source)
        destination = destination_root / item.destination
        if check:
            if not destination.is_file():
                raise ValueError(f"missing extracted destination: {destination}")
            actual = destination.read_text(encoding="utf-8")
            if actual != rewritten:
                raise ValueError(f"destination drift: {destination}")
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(rewritten, encoding="utf-8")
        total_changes += len(changes)
        report_rows.append(
            {
                "source": item.source,
                "destination": item.destination,
                "source_blob": item.blob,
                "rewritten_sha256": hashlib.sha256(rewritten.encode()).hexdigest(),
                "namespace_rewrite_count": len(changes),
                "namespace_rewrites": [f"{old} -> {new}" for old, new in changes],
            }
        )

    return {
        "revision": PINNED_REVISION,
        "mode": "check" if check else "write",
        "runtime_files": runtime_count,
        "shared_executor_files": executor_count,
        "namespace_rewrite_statements": total_changes,
        "phase3_boundary_closure_edits": PHASE3_BOUNDARY_CLOSURE_EDITS,
        "files": report_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-repo", type=Path, default=DEFAULT_SOURCE_REPO)
    parser.add_argument("--destination-root", type=Path, default=DEFAULT_DESTINATION_ROOT)
    parser.add_argument("--check", action="store_true", help="verify destinations without writing")
    args = parser.parse_args()
    result = extract(args.source_repo, args.destination_root, check=args.check)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
