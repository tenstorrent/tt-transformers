# Phase 3 public API policy gate

Status: **machine-complete; 34 model-family exposure decisions require human review**

Authority inputs:

- `qualification/analysis/public_symbols.csv` — 1,760 source symbol rows;
- `qualification/analysis/imports.csv` — cross-module import evidence;
- `qualification/provenance/source_inventory.csv` — source/destination ownership;
- extracted Python ASTs under `src/tt_transformers`.

Machine-readable outputs:

- `qualification/analysis/public_api_policy.csv`;
- `qualification/analysis/public_api_removals.csv`;
- `qualification/analysis/check_public_api_policy.py`.

## Policy definitions

| Status | Contract |
|---|---|
| `supported_public` | Reusable module/runtime construction or protocol surface. Its module-qualified identity and signature are compatibility-controlled. This does not imply a root re-export or hardware support. |
| `model_local_public` | Concrete-model package surface. Its model-local identity and signature are compatibility-controlled, but it is not promoted to the root package. |
| `compatibility_only` | Legacy source compatibility surface with an approved removal and explicit replacement in the removal ledger. |
| `private_by_policy` | Public-looking name without export/import evidence, or orchestration relocated outside the installed package. Access is incidental and has no compatibility promise. |

A public class's non-underscore methods, properties, and constructor inherit that class's status. This preserves externally imported construction and protocol signatures without treating every internal dataclass or helper as API.

## Classification counts

| Status | Total | Modules | LLM runtime | Models |
|---|---:|---:|---:|---:|
| `supported_public` | 367 | 146 | 221 | 0 |
| `model_local_public` | 950 | 0 | 0 | 950 |
| `compatibility_only` | 13 | 12 | 0 | 1 |
| `private_by_policy` | 430 | 40 | 62 | 328 |
| **Total** | **1,760** | **198** | **283** | **1,279** |

All 228 rows marked explicit by source `__all__` are `model_local_public`. No explicit export is classified private or compatibility-only.

The policy does not add eager imports. The checker asserts that `src/tt_transformers/__init__.py` still exports only `__version__`; concrete model packages remain lazy from the root.

## Evidence rules

Classification is deterministic:

1. Preserve every explicit source `__all__` row.
2. Promote a top-level symbol imported by another in-scope production module.
3. Inherit a promoted class's status across its public constructor/protocol members.
4. Keep model-layer promotions model-local.
5. Default unexported and unimported public-looking names to `private_by_policy`.
6. Assign approved TTTv1 adapters/factories to `compatibility_only` and record replacements separately.
7. Treat pytest/demo orchestration split to top-level examples/tests as `private_by_policy`, not installed package API.

The `evidence` and `rationale` columns record the rule used for each row. Source and destination path, qualname, kind, and signatures provide both symbol identities.

## Signature and extraction gate

The checker accounts for every source row exactly once:

| Result | Count | Meaning |
|---|---:|---|
| `exact` | 1,563 | Extracted AST signature is byte-identical to the source inventory signature. |
| `namespace_rewrite` | 1 | Explicit re-export contract is unchanged after the mechanical namespace owner rewrite. |
| `lazy_export_rewrite` | 152 | Model-package `__all__` identity is preserved through a lazy export map rather than an eager import. |
| `compatible_optional_hf_revision` | 9 | A reviewed HF adaptor adds only an optional `hf_revision=None` keyword. |
| `compatible_hf_revision_default` | 2 | A reviewed Mistral adaptor changes the optional revision default to `None`, preventing a custom model ID from inheriting the default model's pin. |
| `immutable_default_hf_revision` | 1 | The Mistral default revision value is pinned to the reviewed immutable commit. |
| `compatible_llama33_performance_profile` | 1 | The Llama 3.3 performance value adds exactly `prefill_minimal_matmul=True`, preserving its minimal-matmul TTFT policy while accuracy remains linear. |
| `approved_removal` | 31 | Source production identity is absent only through the reviewed removal/relocation ledger. |
| **Total** | **1,760** | No unexplained absence or signature mismatch. |

The Llama 3.3 compatibility rule is fail-closed: it matches only
`models/common/models/llama33_70b/model.py::LLAMA33_70B_PERFORMANCE` and only
the complete expected destination constructor expression. Any other path,
qualname, or value still fails as unapproved signature drift. The extraction
rationale and deterministic transform are recorded in
`qualification/extraction/models.md`.

`public_api_removals.csv` contains:

- 13 `compatibility_removed` rows: ten module `from_model_args` factories, two `RotarySetup1D` adapters, and `Llama3Transformer1D.forward`;
- 18 `relocated_nonpackage` rows from two hybrid pytest demos split into explicit example and hardware-test owners.

Each removal row has a replacement identity and approval evidence.

## Human review: 34 family inconsistencies

These rows remain `private_by_policy` pending a deliberate cross-family decision. Analogous symbols are explicit/imported in some model families but merely unexported candidates in another.

| Candidate shape | Count |
|---|---:|
| Generator or generator/generation config classes | 17 |
| `DEFAULT_HF_MODEL` / `DEFAULT_HF_REVISION` values | 10 |
| `build_*_generator` functions | 7 |
| **Total** | **34** |

Affected families include DeepSeek-R1-Distill-Qwen-14B, Llama 3.2 1B/3B, Llama 3 8B, Mistral 7B, Phi-4, Qwen 2/2.5/3, and Qwen 2.5 Coder. Every exact case is flagged with `human_review=yes` and a review reason in `public_api_policy.csv`.

To list them:

```bash
python3 - <<'PY'
import csv
with open('qualification/analysis/public_api_policy.csv', newline='') as stream:
    for row in csv.DictReader(stream):
        if row['human_review'] == 'yes':
            print(row['source_path'], row['source_qualname'], row['source_kind'])
PY
```

Recommended decision: either expose a uniform generator/HF construction surface from each model package's `__all__`, or narrow the currently explicit families. Resolve this as a dedicated API change; do not expand the root package.

## Machine check

Run from the repository root:

```bash
python3 qualification/analysis/check_public_api_policy.py
```

Expected output:

```text
verified 1760 unique policy rows
status {'compatibility_only': 13, 'model_local_public': 950, 'private_by_policy': 430, 'supported_public': 367}
signature {'approved_removal': 31, 'compatible_hf_revision_default': 2, 'compatible_llama33_performance_profile': 1, 'compatible_optional_hf_revision': 9, 'exact': 1563, 'immutable_default_hf_revision': 1, 'lazy_export_rewrite': 152, 'namespace_rewrite': 1}
removals {'compatibility_removed': 13, 'relocated_nonpackage': 18}
human_review 34
root_exports ['__version__']
```

The checker deterministically rebuilds classifications from source symbol, import, provenance, and extracted AST inventories. It fails on missing, duplicate, or unclassified rows; unknown status; unapproved signature drift or absence; lost explicit exports; removal-ledger drift; or root eager-export expansion.

To reproduce the CSVs without editing source:

```bash
python3 qualification/analysis/check_public_api_policy.py --emit policy > /tmp/public_api_policy.csv
python3 qualification/analysis/check_public_api_policy.py --emit removals > /tmp/public_api_removals.csv
diff -u qualification/analysis/public_api_policy.csv /tmp/public_api_policy.csv
diff -u qualification/analysis/public_api_removals.csv /tmp/public_api_removals.csv
```

## Scope boundary

This is an API identity and signature policy gate. It does not prove runtime semantics, model correctness, optional-dependency importability, TTNN compatibility, or hardware support. It intentionally makes no changes to `src/`, tests, examples, the main migration work log, or hardware state.
