#!/usr/bin/env python3
"""Extract the twelve concrete model packages from the pinned Git snapshot."""

from __future__ import annotations

import argparse
import ast
import copy
import csv
import hashlib
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_REPO = Path("/localdev/gwang/tt-metal")
SOURCE_REVISION = "00748e6ac7b65f50e5c2af07f6e7c1c535c7f4c0"
INVENTORY = ROOT / "qualification/provenance/source_inventory.csv"
MODEL_ROOT = "models/common/models/"
MODELS = {
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
}
REMOTE_CODE_POLICY_SOURCES = frozenset(
    {
        "models/common/models/qwen25_72b/hf_adaptor.py",
        "models/common/models/qwen25_coder_32b/hf_adaptor.py",
        "models/common/models/qwen3_32b/hf_adaptor.py",
    }
)
GENERATED_MODEL_INITIALIZERS = {
    ROOT / "src/tt_transformers/models/llama3_8b/__init__.py",
}
PHASE3_HF_ADAPTOR_SHA256 = {
    (
        "models/common/models/deepseek_r1_distill_qwen_14b/hf_adaptor.py"
    ): "1616f482dcfdaf86b6a48d828d1e485be73e8aafd615a0cd38c7ab75b5b13b13",
    "models/common/models/llama32_1b/hf_adaptor.py": "ebd25fbbbc59fc992501bcfe4f4aa3197fd228b8d8ed4cc73e159168718e7277",
    "models/common/models/llama32_3b/hf_adaptor.py": "eff67a497dcec66184e3c71a9a0ad4eb2441930672c918bb250be16a8897898d",
    (
        "models/common/models/llama33_70b/hf_adaptor.py"
    ): "a92ed0dcb28858b9bdcab263b1f5533394bc079ca817de721845059bea821321",
    "models/common/models/llama3_8b/hf_adaptor.py": "b90c03265897bd272d9bea1334c5ec02ae7ec5c141b6248cfa56e7a6eb036f5a",
    "models/common/models/mistral_7b/hf_adaptor.py": "4b717376ec592d9953a832652b69733ace31ce089099b81918cda41416dadc6e",
    "models/common/models/phi4/hf_adaptor.py": "c2b5c281e67e40ef27d450e22a1a5a667de66e1c6e9dee6a0486df41acac1f28",
    "models/common/models/qwen25_72b/hf_adaptor.py": "c3f55cdd27118b84c95885915951d15d5d3f99cf45d8de602d197b4687c1a083",
    "models/common/models/qwen25_7b/hf_adaptor.py": "ebf725620ae35c2f8ad58098ece22edb874c88550fac04544c60ffa6196709d5",
    (
        "models/common/models/qwen25_coder_32b/hf_adaptor.py"
    ): "a9b6ac012c728c7417fc6a10975ae1f92ed3ba5536aa94874c05cf2126431648",
    "models/common/models/qwen2_7b/hf_adaptor.py": "a010f3f91a3e47811bdd8b80dbda89e1acbc312907f3a36c985ad8fd49abbfec",
    "models/common/models/qwen3_32b/hf_adaptor.py": "1c2a6d8e280d3045dec2b299e09e163cd17a6e8c7034d3038de916b4e0c60c07",
}

CHECKPOINT_PIN_TRANSFORMS = {
    "models/common/models/llama32_1b/hf_adaptor.py": (
        "meta-llama/Llama-3.2-1B-Instruct",
        "9213176726f574b556790deb65791e0c5aa438b6",
    ),
    "models/common/models/llama32_3b/hf_adaptor.py": (
        "meta-llama/Llama-3.2-3B-Instruct",
        "0cb88a4f764b7a12671c53f0838cd831a0843b95",
    ),
    "models/common/models/llama33_70b/hf_adaptor.py": (
        "meta-llama/Llama-3.3-70B-Instruct",
        "6f6073b423013f6a7d4d9f39144961bfbfbc386b",
    ),
    "models/common/models/llama3_8b/hf_adaptor.py": (
        "meta-llama/Llama-3.1-8B-Instruct",
        "0e9e39f249a16976918f6564b8830bc894c89659",
    ),
    "models/common/models/mistral_7b/hf_adaptor.py": (
        "mistralai/Mistral-7B-Instruct-v0.3",
        "c170c708c41dac9275d15a8fff4eca08d52bab71",
    ),
    "models/common/models/qwen25_7b/hf_adaptor.py": (
        "Qwen/Qwen2.5-7B-Instruct",
        "a09a35458c702b33eeacc393d103063234e8bc28",
    ),
    "models/common/models/qwen2_7b/hf_adaptor.py": (
        "Qwen/Qwen2-7B-Instruct",
        "f2826a00ceef68f0f2b946d945ecc0477ce4450c",
    ),
}
OPTIONAL_REVISION_API_EXTENSIONS = {
    "models/common/models/llama32_1b/hf_adaptor.py",
    "models/common/models/llama32_3b/hf_adaptor.py",
    "models/common/models/llama33_70b/hf_adaptor.py",
    "models/common/models/llama3_8b/hf_adaptor.py",
}

NAMESPACE_REWRITES = (
    ("models.common.lightweightmodule", "tt_transformers.modules.lightweightmodule"),
    ("models.common.llm_runtime", "tt_transformers.llm_runtime"),
    ("models.common.tensor_utils", "tt_transformers.tensor_utils"),
    ("models.common.device_utils", "tt_transformers.device_utils"),
    ("models.common.sampling.sampling_params", "tt_transformers.sampling.sampling_params"),
    ("models.common.sampling", "tt_transformers.sampling.sampling_params"),
    ("models.common.modules", "tt_transformers.modules"),
    ("models.common.models", "tt_transformers.models"),
    ("models.tt_transformers.tt.generator", "tt_transformers.mesh_utils"),
)


@dataclass(frozen=True)
class InventoryRow:
    source_path: str
    destination_path: Path
    blob_sha: str


def remove_llama3_legacy_forward(text: str, source_path: str) -> str:
    """Remove the characterized TTTv1 dispatcher, and nothing else."""

    if source_path != "models/common/models/llama3_8b/model.py":
        return text
    tree = ast.parse(text, filename=source_path)
    matches = [
        child
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Llama3Transformer1D"
        for child in node.body
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name == "forward"
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one Llama3Transformer1D.forward in {source_path}, found {len(matches)}")
    method = matches[0]
    if "backward compatibility" not in (ast.get_docstring(method) or ""):
        raise ValueError("refusing to remove an uncharacterized Llama3Transformer1D.forward")
    lines = text.splitlines(keepends=True)
    start = method.lineno - 1
    end = method.end_lineno
    if end < len(lines) and not lines[end].strip():
        end += 1
    del lines[start:end]
    rewritten = "".join(lines)
    ast.parse(rewritten, filename=source_path)
    return rewritten


def pin_standalone_checkpoint_revision(text: str, source_path: str) -> str:
    """Record the approved standalone default ID/revision transform."""

    checkpoint = CHECKPOINT_PIN_TRANSFORMS.get(source_path)
    if checkpoint is None:
        return text
    model_id, revision = checkpoint
    revision_assignment = f'DEFAULT_HF_REVISION = "{revision}"'
    if re.search(r"^DEFAULT_HF_REVISION = .+$", text, flags=re.MULTILINE):
        return re.sub(
            r"^DEFAULT_HF_REVISION = .+$",
            revision_assignment,
            text,
            count=1,
            flags=re.MULTILINE,
        )
    model_assignment = f'DEFAULT_HF_MODEL = "{model_id}"'
    if model_assignment in text:
        return text.replace(model_assignment, f"{model_assignment}\n{revision_assignment}", 1)
    marker = "@dataclass(frozen=True)\nclass RopeScaling:"
    if source_path.endswith("/llama3_8b/hf_adaptor.py") and marker in text:
        return text.replace(
            marker,
            f"{model_assignment}\n{revision_assignment}\n\n\n{marker}",
            1,
        )
    raise ValueError(f"unable to apply checkpoint pin transform to {source_path}")


def close_qwen3_optional_import(text: str, source_path: str) -> str:
    """Keep Qwen3 core importable until its explicit HF loader is called."""

    if source_path != "models/common/models/qwen3_32b/model.py":
        return text
    eager = "from transformers import AutoConfig, AutoModelForCausalLM\n"
    if text.count(eager) != 1:
        raise ValueError("expected one eager Qwen3 transformers import")
    text = text.replace(eager, "")
    loader_start = (
        "        ttnn.SetDefaultDevice(mesh_device)\n"
        "        cache_path = Path(cache_dir) if cache_dir else None\n"
    )
    if text.count(loader_start) != 1:
        raise ValueError("expected one Qwen3 from_pretrained loader body")
    return text.replace(
        loader_start,
        "        from transformers import AutoConfig, AutoModelForCausalLM\n\n"
        "        cache_path = Path(cache_dir) if cache_dir else None\n",
    )


def remove_model_default_device_mutation(text: str, source_path: str) -> str:
    owners = {
        "models/common/models/qwen25_72b/model.py",
        "models/common/models/qwen25_coder_32b/model.py",
    }
    if source_path not in owners:
        return text
    mutation = "    ttnn.SetDefaultDevice(mesh_device)\n"
    if text.count(mutation) != 1:
        raise ValueError(f"expected one model default-device mutation in {source_path}")
    return text.replace(mutation, "")


def remove_forced_remote_code(text: str, source_path: str) -> str:
    """Remove only the pinned Qwen forced remote-code opt-ins."""

    if source_path not in REMOTE_CODE_POLICY_SOURCES:
        return text
    direct = "        trust_remote_code=True,\n"
    forwarded = '        "trust_remote_code": True,\n'
    expected = 2 if source_path == "models/common/models/qwen25_72b/hf_adaptor.py" else 1
    found = text.count(direct) + text.count(forwarded)
    if found != expected:
        raise ValueError(f"expected {expected} forced remote-code opt-ins in {source_path}, found {found}")
    rewritten = text.replace(direct, "").replace(forwarded, "")
    ast.parse(rewritten, filename=source_path)
    return rewritten


def ordered_explicit_all(tree: ast.Module) -> list[str]:
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if not any(isinstance(target, ast.Name) and target.id == "__all__" for target in targets):
            continue
        value = node.value
        if not isinstance(value, (ast.List, ast.Tuple)):
            raise ValueError("model package __all__ must be a literal list/tuple")
        return [
            item.value
            for item in value.elts
            if isinstance(item, ast.Constant) and isinstance(item.value, str)
        ]
    raise ValueError("model package initializer has no explicit __all__")


def lazy_model_initializer(text: str, source_path: str) -> str:
    """Replace eager package re-exports with cached module-level lazy exports."""

    if not source_path.endswith("/__init__.py"):
        return text
    tree = ast.parse(text, filename=source_path)
    exports = ordered_explicit_all(tree)
    origins: dict[str, tuple[str, str]] = {}
    for node in tree.body:
        if not isinstance(node, ast.ImportFrom) or node.level:
            continue
        for alias in node.names:
            origins[alias.asname or alias.name] = (node.module or "", alias.name)
    missing = [name for name in exports if name not in origins]
    if missing:
        raise ValueError(f"model initializer exports lack import origins: {missing}")

    source_lines = text.splitlines()
    header: list[str] = []
    for line in source_lines:
        if line.startswith("#") or (not line and header):
            header.append(line)
            continue
        break
    while header and not header[-1]:
        header.pop()

    lines = [
        *header,
        "",
        '"""Lazy model-local exports; optional HF dependencies load on access."""',
        "",
        "from __future__ import annotations",
        "",
        "from importlib import import_module",
        "",
        "_EXPORTS = {",
    ]
    for name in exports:
        module, attribute = origins[name]
        lines.append(f"    {name!r}: ({module!r}, {attribute!r}),")
    lines.extend(
        [
            "}",
            "",
            "__all__ = [",
            *(f"    {name!r}," for name in exports),
            "]",
            "",
            "",
            "def __getattr__(name: str):",
            "    try:",
            "        module_name, attribute = _EXPORTS[name]",
            "    except KeyError as error:",
            "        raise AttributeError(f\"module {__name__!r} has no attribute {name!r}\") from error",
            "    value = getattr(import_module(module_name), attribute)",
            "    globals()[name] = value",
            "    return value",
            "",
            "",
            "def __dir__():",
            "    return sorted(set(globals()) | set(__all__))",
            "",
        ]
    )
    rewritten = "\n".join(lines)
    ast.parse(rewritten, filename=source_path)
    return rewritten


def generated_empty_initializer() -> bytes:
    return b'''"""Llama 3 8B model package; no package-level exports are declared."""

__all__: list[str] = []
'''


def git_output(*args: str, binary: bool = False) -> bytes | str:
    result = subprocess.run(
        ["git", "-C", str(SOURCE_REPO), *args],
        check=True,
        capture_output=True,
        text=not binary,
    )
    return result.stdout


def selected_rows() -> list[InventoryRow]:
    with INVENTORY.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))

    selected: list[InventoryRow] = []
    for row in rows:
        relative = row["source_path"].removeprefix(MODEL_ROOT)
        model = relative.split("/", 1)[0]
        if row["category"] != "production_model" or model not in MODELS:
            continue
        destination = row["destination_path"]
        if ";" in destination or not destination.startswith("src/tt_transformers/models/"):
            raise ValueError(f"unexpected production-model destination: {destination}")
        selected.append(
            InventoryRow(
                source_path=row["source_path"],
                destination_path=ROOT / destination,
                blob_sha=row["git_blob_sha"],
            )
        )

    # The inventory's 74 production-model rows include three shared executors,
    # which are extracted by the runtime lane rather than this concrete lane.
    if len(selected) != 71:
        raise ValueError(f"expected 71 concrete production-model rows, found {len(selected)}")
    if {row.source_path.removeprefix(MODEL_ROOT).split("/", 1)[0] for row in selected} != MODELS:
        raise ValueError("the selected inventory does not cover exactly the twelve model packages")
    return sorted(selected, key=lambda row: row.source_path)


def expected_content(row: InventoryRow) -> bytes:
    actual_blob = str(git_output("rev-parse", f"{SOURCE_REVISION}:{row.source_path}")).strip()
    if actual_blob != row.blob_sha:
        raise ValueError(f"blob mismatch for {row.source_path}: {actual_blob} != {row.blob_sha}")
    source = bytes(git_output("cat-file", "blob", f"{SOURCE_REVISION}:{row.source_path}", binary=True))
    text = remove_llama3_legacy_forward(source.decode("utf-8"), row.source_path)
    text = pin_standalone_checkpoint_revision(text, row.source_path)
    for old, new in NAMESPACE_REWRITES:
        text = text.replace(old, new)
    text = close_qwen3_optional_import(text, row.source_path)
    text = remove_model_default_device_mutation(text, row.source_path)
    text = remove_forced_remote_code(text, row.source_path)
    text = lazy_model_initializer(text, row.source_path)
    return text.encode("utf-8")


def verify_phase3_hf_adaptor(path: Path, source_path: str, actual: bytes, pinned_expected: bytes) -> str | None:
    """Verify exact policy-normalized content while retaining pinned API provenance."""

    expected_hash = PHASE3_HF_ADAPTOR_SHA256[source_path]
    actual_hash = hashlib.sha256(actual).hexdigest()
    if actual_hash != expected_hash:
        return (
            f"content mismatch {path.relative_to(ROOT)} "
            f"(expected Phase 3 sha256 {expected_hash}, got {actual_hash})"
        )
    source = actual.decode("utf-8")
    if any(
        token in source
        for token in ('Path("model_cache")', "os.getenv", "os.environ", "SetDefaultDevice", "GetDefaultDevice")
    ):
        return f"cache/environment policy regression {path.relative_to(ROOT)}"
    tree = ast.parse(source, filename=str(path))
    checkpoint = CHECKPOINT_PIN_TRANSFORMS.get(source_path)
    if checkpoint is not None:
        expected_model, expected_revision = checkpoint
        assignments = {
            target.id: node.value.value
            for node in tree.body
            if isinstance(node, (ast.Assign, ast.AnnAssign))
            for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
            if isinstance(target, ast.Name)
            and target.id in {"DEFAULT_HF_MODEL", "DEFAULT_HF_REVISION"}
            and isinstance(node.value, ast.Constant)
        }
        if assignments != {
            "DEFAULT_HF_MODEL": expected_model,
            "DEFAULT_HF_REVISION": expected_revision,
        }:
            return f"default checkpoint identity regression in {path.relative_to(ROOT)}"
        functions = {
            node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
        }
        if "_resolve_hf_revision" not in functions:
            return f"missing model-aware revision resolver in {path.relative_to(ROOT)}"
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if (
                node.func.attr == "from_pretrained"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in {"AutoConfig", "AutoModelForCausalLM", "AutoTokenizer"}
                and not any(
                    (
                        keyword.arg == "revision"
                        and isinstance(keyword.value, ast.Name)
                        and keyword.value.id == "hf_revision"
                    )
                    or (
                        keyword.arg is None
                        and isinstance(keyword.value, ast.Name)
                        and keyword.value.id == "load_kwargs"
                    )
                    for keyword in node.keywords
                )
            ):
                return f"unversioned HF load in {path.relative_to(ROOT)}:{node.lineno}"
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and any(
            keyword.arg == "trust_remote_code"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value is True
            for keyword in node.keywords
        ):
            return f"forced remote-code loader opt-in in {path.relative_to(ROOT)}:{node.lineno}"
        if isinstance(node, ast.Dict) and any(
            isinstance(key, ast.Constant)
            and key.value == "trust_remote_code"
            and isinstance(value, ast.Constant)
            and value.value is True
            for key, value in zip(node.keys, node.values)
        ):
            return f"forced remote-code forwarded opt-in in {path.relative_to(ROOT)}:{node.lineno}"
    policy_imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == "tt_transformers.cache_environment"
        for alias in node.names
    }
    if not {"offline_mode", "report_model_preflight", "resolve_model_cache"} <= policy_imports:
        return f"missing shared cache/environment policy import in {path.relative_to(ROOT)}"
    preflight_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "report_model_preflight"
    ]
    if len(preflight_calls) != 1:
        return f"expected one preflight report in {path.relative_to(ROOT)}, found {len(preflight_calls)}"
    if not any(keyword.arg == "cache_resolution" for keyword in preflight_calls[0].keywords):
        return f"preflight does not report the exact cache resolution in {path.relative_to(ROOT)}"
    cache_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "resolve_model_cache"
    ]
    if len(cache_calls) != 1:
        return f"expected one shared cache resolution in {path.relative_to(ROOT)}, found {len(cache_calls)}"
    cache_keywords = {keyword.arg for keyword in cache_calls[0].keywords}
    if not {"hf_model_id", "hf_revision", "topology", "mesh_device", "dtype"} <= cache_keywords:
        return f"cache identity inputs are incomplete in {path.relative_to(ROOT)}"
    pinned_tree = ast.parse(pinned_expected.decode("utf-8"), filename=source_path)
    pinned_api = next(
        node for node in pinned_tree.body if isinstance(node, ast.FunctionDef) and node.name == "from_pretrained"
    )
    actual_api = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "from_pretrained"
    )
    pinned_args = copy.deepcopy(pinned_api.args)
    actual_args = copy.deepcopy(actual_api.args)
    if source_path in OPTIONAL_REVISION_API_EXTENSIONS:
        revision_index = next(
            (index for index, argument in enumerate(actual_args.kwonlyargs) if argument.arg == "hf_revision"),
            None,
        )
        if revision_index is None or not (
            isinstance(actual_args.kw_defaults[revision_index], ast.Constant)
            and actual_args.kw_defaults[revision_index].value is None
        ):
            return f"hf_revision must be an optional keyword in {path.relative_to(ROOT)}"
        actual_args.kwonlyargs.pop(revision_index)
        actual_args.kw_defaults.pop(revision_index)
    elif source_path == "models/common/models/mistral_7b/hf_adaptor.py":
        actual_index = next(
            index for index, argument in enumerate(actual_args.kwonlyargs) if argument.arg == "hf_revision"
        )
        pinned_index = next(
            index for index, argument in enumerate(pinned_args.kwonlyargs) if argument.arg == "hf_revision"
        )
        actual_args.kw_defaults[actual_index] = copy.deepcopy(pinned_args.kw_defaults[pinned_index])
    if ast.dump(pinned_args, include_attributes=False) != ast.dump(actual_args, include_attributes=False):
        return f"public from_pretrained signature drift in {path.relative_to(ROOT)}"
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write",
        action="store_true",
        help="rewrite destinations from the pinned snapshot plus approved transforms",
    )
    parser.add_argument(
        "--scope",
        choices=("all", "optional-deps", "remote-code"),
        default="all",
        help="verify all model files or a narrow approved-transform scope",
    )
    args = parser.parse_args()

    failures: list[str] = []
    rows = selected_rows()
    if args.scope == "optional-deps":
        rows = [
            row
            for row in rows
            if row.source_path.endswith("/__init__.py")
            or row.source_path == "models/common/models/qwen3_32b/model.py"
        ]
    elif args.scope == "remote-code":
        rows = [row for row in rows if row.source_path in REMOTE_CODE_POLICY_SOURCES]
    for row in rows:
        expected = expected_content(row)
        destination = row.destination_path
        is_phase3_adaptor = row.source_path in PHASE3_HF_ADAPTOR_SHA256
        if args.write and not is_phase3_adaptor:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(expected)
        if not destination.exists():
            failures.append(f"missing {destination.relative_to(ROOT)}")
            continue
        actual = destination.read_bytes()
        if is_phase3_adaptor:
            failure = verify_phase3_hf_adaptor(destination, row.source_path, actual, expected)
            if failure:
                failures.append(failure)
        elif actual != expected:
            failures.append(
                f"content mismatch {destination.relative_to(ROOT)} "
                f"(expected sha256 {hashlib.sha256(expected).hexdigest()}, "
                f"got {hashlib.sha256(actual).hexdigest()})"
            )

    for destination in GENERATED_MODEL_INITIALIZERS if args.scope != "remote-code" else ():
        expected = generated_empty_initializer()
        if args.write:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(expected)
        if not destination.exists():
            failures.append(f"missing {destination.relative_to(ROOT)}")
        elif destination.read_bytes() != expected:
            failures.append(f"content mismatch {destination.relative_to(ROOT)}")

    if failures:
        print("\n".join(failures))
        return 1
    if args.scope == "optional-deps":
        print("verified 12 pinned optional-dependency transforms plus 1 generated package initializer")
    elif args.scope == "remote-code":
        print("verified 3 pinned remote-code policy transforms")
    else:
        print("verified 71 pinned concrete-model files plus 1 generated package initializer")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
