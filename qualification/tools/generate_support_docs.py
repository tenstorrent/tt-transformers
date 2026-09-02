#!/usr/bin/env python3
"""Generate deterministic experimental support docs from support manifests."""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MODELS = (
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
BEGIN = "<!-- BEGIN GENERATED SUPPORT -->"
END = "<!-- END GENERATED SUPPORT -->"


FACTS = {
    "deepseek_r1_distill_qwen_14b": {
        "title": "DeepSeek-R1-Distill-Qwen-14B",
        "purpose": "Run the concrete 14B distilled-Qwen tensor model through direct eager/traced execution, teacher-forced accuracy, throughput, and DP smoke paths.",
        "limits": ["Demo cases cover active batch 1 or 32", "standard/CI sequence budgets are 1024/2048; retained DP smokes reach 4096", "TP1 is rejected; supported model lanes are TP2, TP4, or TP8", "N300 accuracy eval-32 and batch-32-ci are explicitly DRAM-infeasible"],
        "sampling": "Host and on-device top-k paths exist; the demo default is on_device_topk.",
        "unsupported": ["N150/TP1", "N300 accuracy eval-32 and batch-32-ci", "any geometry not declared in support.json"],
    },
    "llama32_1b": {
        "title": "Llama-3.2-1B-Instruct",
        "purpose": "Exercise the concrete small Llama tensor model, shared Llama executor, accuracy, performance, and DP lane routing.",
        "limits": ["Demo cases cover active batch 1 or 32", "standard/CI sequence budgets are 1024/2048; DP smokes reach 4096", "TP1, TP2, and TP8 model paths are declared; TP4 DP lanes are deliberately rejected", "N150 batch-32-ci at sequence 2048 is not enabled", "traced prefill is Q128 on N150 and Q128/Q1024 on N300/T3K"],
        "sampling": "Host and on-device paths exist; the demo default is host sampling.",
        "unsupported": ["TP4 lanes", "N150 batch-32-ci sequence 2048", "any geometry not declared in support.json"],
    },
    "llama32_3b": {
        "title": "Llama-3.2-3B-Instruct",
        "purpose": "Exercise the concrete 3B Llama tensor model, shared Llama executor, accuracy, performance, and DP lane routing.",
        "limits": ["Demo cases cover active batch 1 or 32", "standard/CI sequence budgets are 1024/2048; DP smokes reach 4096", "TP1, TP2, and TP8 paths are declared; TP4 DP lanes are rejected", "N150 has decode tracing but no traced-prefill bucket; N300/T3K declare Q128/Q1024"],
        "sampling": "Host and on-device paths exist; the demo default is host sampling.",
        "unsupported": ["TP4 lanes", "traced prefill on N150", "any geometry not declared in support.json"],
    },
    "llama33_70b": {
        "title": "Llama-3.3-70B-Instruct",
        "purpose": "Run the concrete 70B Llama model on its full tensor-parallel lane for token accuracy, repeat-batch determinism, trace, and throughput characterization.",
        "limits": ["Demo cases cover active batch 1 or 32", "P150x4 contract buckets are 4096 for token accuracy and 1024 for eval/performance cases", "one model replica requires the full declared lane; every collected DP>1 case skips", "P150x4 long-prefill coverage is not established"],
        "sampling": "Host and on-device top-k paths exist; capability contracts distinguish trace none/decode_only/all.",
        "unsupported": ["DP greater than 1", "noncanonical P150x4 orientations", "unproven 32K context", "any geometry not declared in support.json"],
    },
    "llama3_8b": {
        "title": "Llama-3.1-8B-Instruct",
        "purpose": "Run the concrete Llama-3.1-8B tensor model across Wormhole and declared Blackhole paths, including accuracy, performance, DP, and seeded cross-cardinality diagnostics.",
        "limits": ["Demo cases cover active batch 1 or 32", "standard/CI budgets are 1024/2048; the DP4 case reaches 4096", "Blackhole contract buckets are 1024, with P150x4 DP4 at 4096", "device sampling keeps Blackhole prefill sequential unless the controlled invariance experiment proves otherwise"],
        "sampling": "Host and on-device top-k paths exist; seeded device-sampling cross-cardinality diagnostics are retained.",
        "unsupported": ["TTTv1 DRAM prefetcher", "P100 and 128K P300 acceptance", "development two-P150 stand-in as real-P300 evidence", "any geometry not declared in support.json"],
    },
    "mistral_7b": {
        "title": "Mistral-7B-Instruct-v0.3",
        "purpose": "Run the concrete Mistral tensor model through direct executor accuracy, throughput, determinism, and DP smoke paths.",
        "limits": ["Demo cases cover active batch 1 or 32", "standard/CI budgets are 1024/2048; retained DP smokes reach 4096", "N300 DP2 and T3K DP8 use one-device lanes; other retained DP factors skip"],
        "sampling": "Host and on-device top-k paths exist; the demo default is host sampling.",
        "unsupported": ["DP layouts other than the declared one-device lanes", "any geometry not declared in support.json"],
    },
    "phi4": {
        "title": "Phi-4",
        "purpose": "Run the concrete Phi-4 tensor model through direct executor accuracy, performance, determinism, and TP2 lane smoke paths.",
        "limits": ["Demo cases cover active batch 1 or 32", "standard/CI budgets are 1024/2048; DP smoke metadata reaches 4096", "ordinary execution is N300 TP2; physical T3K is admitted only as four TP2 lanes", "N150 exceeds the source L1 capacity guard"],
        "sampling": "Host and on-device top-k paths exist.",
        "unsupported": ["N150", "ordinary T3K/TG tensor parallelism", "DP layouts other than T3K DP4/TP2", "any geometry not declared in support.json"],
    },
    "qwen25_72b": {
        "title": "Qwen2.5-72B-Instruct",
        "purpose": "Run the concrete 72B Qwen2.5 model on its full T3K TP8 lane for accuracy, performance, and determinism characterization.",
        "limits": ["Demo cases cover active batch 1 or 32", "standard/CI sequence budgets are 1024/2048", "weights and KV require T3K TP8", "all retained DP cases capacity-skip"],
        "sampling": "Host and on-device top-k paths exist; the performance path defaults to on_device_topk.",
        "unsupported": ["non-T3K meshes", "DP greater than 1", "any geometry not declared in support.json"],
    },
    "qwen25_7b": {
        "title": "Qwen2.5-7B-Instruct",
        "purpose": "Run the concrete Qwen2.5-7B model on TP2 lanes for accuracy, throughput, determinism, and T3K DP4 smoke coverage.",
        "limits": ["Demo cases cover active batch 1 or 32", "standard/CI budgets are 1024/2048", "ordinary execution is N300 TP2; T3K DP4 partitions into four TP2 lanes", "N150 overflows the source L1 capacity guard"],
        "sampling": "Host and on-device top-k paths exist; the demo default is host sampling.",
        "unsupported": ["N150", "ordinary T3K/TG TP", "DP factors other than T3K DP4", "any geometry not declared in support.json"],
    },
    "qwen25_coder_32b": {
        "title": "Qwen2.5-Coder-32B-Instruct",
        "purpose": "Run the concrete 32B coder model on T3K TP8, with end-to-end and focused smoke/PCC diagnostic CLIs.",
        "limits": ["End-to-end cases cover active batch 1 or 32", "standard/CI sequence budgets are 1024/2048", "weights and KV require T3K TP8", "all retained DP cases capacity-skip", "focused smoke prefill uses sequence 128 by default"],
        "sampling": "Host and on-device top-k paths exist; the performance path defaults to on_device_topk.",
        "unsupported": ["non-T3K meshes", "DP greater than 1", "any geometry not declared in support.json"],
    },
    "qwen2_7b": {
        "title": "Qwen2-7B-Instruct",
        "purpose": "Run the concrete Qwen2-7B model on TP2 lanes for accuracy, throughput, determinism, and T3K DP4 smoke coverage.",
        "limits": ["Demo cases cover active batch 1 or 32", "standard/CI budgets are 1024/2048", "ordinary execution is N300 TP2; T3K DP4 partitions into four TP2 lanes", "N150 and ordinary TP8 fail source capacity/head-divisibility guards"],
        "sampling": "Host and on-device top-k paths exist; the demo default is host sampling.",
        "unsupported": ["N150", "ordinary T3K/TG TP", "DP factors other than T3K DP4", "any geometry not declared in support.json"],
    },
    "qwen3_32b": {
        "title": "Qwen3-32B",
        "purpose": "Run the concrete Qwen3 model on T3K TP8 or declared Blackhole P150x4 TP4, including accuracy, trace, determinism, seeded batching, and focused smoke diagnostics.",
        "limits": ["Demo cases cover active batch 1 or 32", "P150x4 buckets are 4096 for token accuracy, 1024 for eval, and 2048 for batch-32-ci", "model lanes require at least TP4; all DP cases capacity-skip", "P150x4 batched prefill remains disabled unless exact-token cross-cardinality invariance is proven"],
        "sampling": "Host and on-device top-k paths exist; P150x4 includes a seeded exact-token cross-cardinality diagnostic.",
        "unsupported": ["P100/P150/P300 single/two-die model paths", "DP greater than 1", "noncanonical P150x4 orientations", "any geometry not declared in support.json"],
    },
}


def candidate_software() -> dict:
    text = (ROOT / "pyproject.toml").read_text()
    version = re.search(r'^version = "([^"]+)"', text, re.MULTILINE).group(1)
    pins = dict(re.findall(r'"([A-Za-z0-9_-]+)==([^";]+)"', text))
    return {
        "tt_transformers": version,
        "ttnn": pins["ttnn"],
        "python": ["3.10", "3.12"],
        "torch": pins["torch"],
        "transformers": pins["transformers"],
    }


def hardware_table(rows: list[dict]) -> str:
    lines = ["| Architecture | Physical SKU/system declaration | Mesh | TP | DP |", "| --- | --- | --- | ---: | ---: |"]
    for row in rows:
        lines.append(f"| {row['architecture']} | {row['sku']} | {row['mesh']} | {row['tp']} | {row['dp']} |")
    return "\n".join(lines)


def install_command() -> str:
    return "python -m pip install -e '.[examples,test]'"


def mesh_environment(row: dict) -> str:
    text = f"{row['sku']} {row['mesh']}".lower()
    if "t3k" in text:
        return "T3K"
    if "n300" in text:
        return "N300"
    if "n150" in text:
        return "N150"
    if "p150_x4" in text or "p150x4" in text or int(row["tp"]) == 4:
        return "P150x4"
    if "p300" in text:
        return "P300"
    if "p150" in text:
        return "P150"
    return "SET_ME"


def generated_block(model: str, manifest: dict) -> str:
    facts = FACTS[model]
    checkpoint = manifest["model"]
    revision = checkpoint["hf_revision"]
    revision_text = revision if revision != "UNPINNED" else "**UNPINNED — release blocker; provider default may move**"
    mesh = manifest["hardware"][0]
    limits = "\n".join(f"- {item}." if not item.endswith(".") else f"- {item}" for item in facts["limits"])
    gaps = [*facts["unsupported"], *manifest.get("known_gaps", [])]
    gaps_text = "\n".join(f"- {item}." if not item.endswith(".") else f"- {item}" for item in dict.fromkeys(gaps))
    mesh_env = mesh_environment(mesh)
    run = (
        f"HF_HOME=/path/to/hf-cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "
        f"MESH_DEVICE={mesh_env} HF_MODEL={checkpoint['hf_id']} "
        f"python -m examples.{model}.demo --case token-accuracy --optimizations performance"
    )
    extra = ""
    if model in {"qwen25_coder_32b", "qwen3_32b"}:
        extra = f"\nFocused smoke:\n\n```bash\nMESH_DEVICE=T3K HF_MODEL={checkpoint['hf_id']} python -m examples.{model}.smoke --case {model.replace('_', '-')}-prefill-smoke\n```\n"
    return f"""{BEGIN}

## Standalone support contract

### Purpose

{facts['purpose']}

### Status and checkpoint

- Status: **experimental**.
- Implementation: **concrete and runnable**; this is not evidence of qualification.
- Hugging Face ID: `{checkpoint['hf_id']}`.
- Hugging Face revision: {revision_text}.

### Candidate software tuple

This is the declared qualification candidate, not a passing verdict:

- `tt-transformers=={manifest['software']['tt_transformers']}`
- `ttnn=={manifest['software']['ttnn']}`
- Python `{', '.join(manifest['software']['python'])}`
- `torch=={manifest['software']['torch']}`
- `transformers=={manifest['software']['transformers']}`

### Declared hardware geometry

{hardware_table(manifest['hardware'])}

Only these source-declared rows are candidates. No row has passing hardware evidence at the pinned extraction revision.

### Proven limits and features

{limits}
- Device sampling: {facts['sampling']}
- Trace support is case/topology-specific; use the exact hardware test parameters and capability manifest rather than extrapolating a generic trace claim.

### Checkpoint, cache, and offline requirements

- Obtain the authorized HF config, tokenizer/chat template, and complete checkpoint separately; weights are not shipped.
- Set a writable `TT_CACHE_PATH` (the example appends topology exactly once), or use the versioned standalone root selected by `TT_TRANSFORMERS_CACHE`, `XDG_CACHE_HOME`, or the user cache.
- Examples use built-in Transformers loading behavior; `trust_remote_code` is not enabled by default.
- For offline runs set `HF_HOME`, `HF_HUB_OFFLINE=1`, and `TRANSFORMERS_OFFLINE=1` after populating the exact snapshot.
- Reference asset root: `{manifest['assets']['reference_root']}`.

### Install, run, and collect

```bash
{install_command()}
```

Representative run using the first declared geometry:

```bash
{run}
```
{extra}
Collect the equivalent hardware gate without running it:

```bash
PYTHONPATH=src MESH_DEVICE={mesh_env} pytest --collect-only -q tests/hardware/models/{model}/test_demo.py
```

### Correctness criterion

`token-accuracy` compares teacher-forced predictions with the committed `.refpt` and enforces the source-declared top-1/top-5 floor when one exists. Performance cases enforce a floor only when a complete source target exists; otherwise measurements are observational. Eval cases require cross-batch consistency, DP smokes require every lane to complete without invalid special-token output, and any explicit source capacity guard remains a skip rather than support evidence.

### Validation and evidence

- Last standalone hardware validation date: **none**.
- Validated git SHA: **none**.
- Evidence: **none attributable to pinned source revision `00748e6ac7b65f50e5c2af07f6e7c1c535c7f4c0`**.
- [Machine-readable manifest](support.json)
- [Hardware gate](../../tests/hardware/models/{model}/test_demo.py)
- [Pinned support baseline](../../qualification/analysis/support/support_baseline.md)
- [Phase 3 boundary evidence](../../qualification/extraction/support_boundary.md)
- [Historical evidence ledger](../../qualification/analysis/support/hardware_evidence.csv)

### Known and unsupported gaps

{gaps_text}

{END}
"""


def replace_generated(readme: Path, block: str) -> None:
    text = readme.read_text()
    if BEGIN in text:
        before, tail = text.split(BEGIN, 1)
        _old, after = tail.split(END, 1)
        text = before.rstrip() + "\n\n" + block.rstrip() + "\n\n" + after.lstrip("\n")
    else:
        lines = text.splitlines()
        if lines and lines[0].startswith("# "):
            text = lines[0] + "\n\n" + block + "\n" + "\n".join(lines[1:]).lstrip()
        else:
            text = block + "\n" + text
    readme.write_text(text.rstrip() + "\n")


def matrix_rows(manifests: dict[str, dict]) -> list[str]:
    rows = []
    for model in MODELS:
        data = manifests[model]
        rev = data["model"]["hf_revision"]
        hardware = "; ".join(f"{x['architecture']} {x['sku']} mesh={x['mesh']} TP{x['tp']}/DP{x['dp']}" for x in data["hardware"])
        rows.append(
            f"| [{model}](examples/{model}/README.md) | experimental | concrete; runnable CLI | `{data['model']['hf_id']}` | "
            + (f"`{rev}`" if rev != "UNPINNED" else "**UNPINNED blocker**")
            + f" | {hardware} | none at pinned SHA |"
        )
    return rows


def write_matrices(manifests: dict[str, dict]) -> None:
    header = [
        "| Example | Status | Implementation | HF ID | Revision | Declared candidate geometry | Evidence |",
        "| --- | --- | --- | --- | --- | --- | --- |",
        *matrix_rows(manifests),
    ]
    matrix = "\n".join(header)
    support = f"""# TT Transformers support matrix

Concrete implementation and a runnable CLI mean the source is present and callable. They do **not** mean qualified. Every current row is experimental because no hardware result is attributable to the pinned extraction SHA.

Candidate software tuple: `tt-transformers==0.1.0.dev0`, `ttnn==0.77.0`, Python 3.10/3.12, `torch==2.11.0`, `transformers==5.12.1`.

{matrix}

Evidence policy and details:

- [Pinned support baseline](qualification/analysis/support/support_baseline.md)
- [Hardware evidence ledger](qualification/analysis/support/hardware_evidence.csv)
- [Phase 3 support-boundary report](qualification/extraction/support_boundary.md)
- [Example overview](examples/README.md)

An empty evidence cell would be ambiguous, so every row explicitly says that pinned-SHA evidence is absent. Historical observations at other SHAs are context only.
"""
    (ROOT / "SUPPORT.md").write_text(support)
    example_header = [line.replace("](examples/", "](") for line in header]
    example_matrix = "\n".join(example_header)
    overview = f"""# Examples and model support

All twelve examples are concrete and runnable, but **experimental**. No example is qualified and no hardware pass is attributable to the pinned source revision.

Install the candidate environment with:

```bash
{install_command()}
```

{example_matrix}

Each model README contains exact run/collection commands, checkpoint/cache requirements, proven source limits, unsupported configurations, and evidence links. Machine-readable truth lives in each `support.json`; validate all documentation with:

```bash
python -B qualification/tools/validate_support_docs.py
```
"""
    (ROOT / "examples/README.md").write_text(overview)


def main() -> None:
    software = candidate_software()
    manifests = {}
    for model in MODELS:
        path = ROOT / "examples" / model / "support.json"
        manifest = json.loads(path.read_text())
        manifest["software"] = software
        manifest["limits"] = {"source_proven": FACTS[model]["limits"]}
        manifest["features"] = {"device_sampling": FACTS[model]["sampling"], "trace": "case_and_topology_specific"}
        manifest["assets"].update(
            {
                "hf_assets_external": True,
                "offline_environment": ["HF_HOME", "HF_HUB_OFFLINE=1", "TRANSFORMERS_OFFLINE=1"],
                "tt_cache_writable": True,
                "remote_code_default": False,
                "cwd_relative_cache_default": False,
                "cache_resolver": "tt_transformers.cache_environment.resolve_model_cache_path",
                "tt_cache_path_semantics": "append topology exactly once",
            }
        )
        manifest["known_gaps"] = list(dict.fromkeys([*manifest["known_gaps"], *FACTS[model]["unsupported"]]))
        path.write_text(json.dumps(manifest, indent=2) + "\n")
        manifests[model] = manifest
        replace_generated(ROOT / "examples" / model / "README.md", generated_block(model, manifest))
    write_matrices(manifests)
    print("generated SUPPORT.md, examples/README.md, and 12 README support contracts")


if __name__ == "__main__":
    main()
