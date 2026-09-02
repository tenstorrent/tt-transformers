# Phase 2 concrete-model extraction

Source repository: `/localdev/gwang/tt-metal`

Source revision: `00748e6ac7b65f50e5c2af07f6e7c1c535c7f4c0`

Status: mechanical extraction complete; boundary closure and runtime
qualification pending

## Extracted scope

`tools/extract_model_packages.py` selected and verified 71 pinned production
files across all twelve concrete model packages. The three shared executor
files included in the inventory's 74 `production_model` rows are owned by the
runtime extraction lane and are deliberately not duplicated here.

| Package | Files |
| --- | ---: |
| `deepseek_r1_distill_qwen_14b` | 6 |
| `llama32_1b` | 6 |
| `llama32_3b` | 6 |
| `llama33_70b` | 6 |
| `llama3_8b` | 4 |
| `mistral_7b` | 6 |
| `phi4` | 6 |
| `qwen25_72b` | 7 |
| `qwen25_7b` | 6 |
| `qwen25_coder_32b` | 6 |
| `qwen2_7b` | 6 |
| `qwen3_32b` | 6 |

Model READMEs, hardware demos, tests, and reference assets are owned by the
support extraction lane. In particular, the two package-local hybrid Qwen
demos are not installed as production model modules.

## Mechanical namespace rules

The extractor verifies each inventory blob SHA, reads the file directly from
the pinned Git object, and rewrites only these module prefixes:

- `models.common.modules` → `tt_transformers.modules`
- `models.common.llm_runtime` → `tt_transformers.llm_runtime`
- `models.common.models` → `tt_transformers.models`
- `models.common.sampling.sampling_params` → `tt_transformers.sampling.sampling_params`
- aggregate `models.common.sampling.SamplingParams` → `tt_transformers.sampling.sampling_params.SamplingParams`
- lightweight/tensor/device helpers → their assigned standalone owners
- `models.tt_transformers.tt.generator` → `tt_transformers.mesh_utils`

The Phase 2 mechanical pass changed no class name, function signature, model
tuning, environment policy, cache policy, checkpoint revision, or execution
behavior. Phase 3/7 subsequently normalized the twelve HF adaptors under the
typed policy documented in `cache_environment.md`; the extractor now verifies
their reviewed backward-compatible revision extensions, complete cache-identity
inputs, exact preflight resolution, and exact policy-normalized hashes.

## Verification

From the repository root:

```bash
python tools/extract_model_packages.py
PYTHONPYCACHEPREFIX=/tmp/gwang/tt-transformers-model-pyc \
  python -m compileall -q src/tt_transformers/models
rg -n '(^|\s)(from|import) models\.' src/tt_transformers/models
git diff --check
```

The deterministic extractor verified all 71 destination contents, syntax
compilation passed, the legacy-import scan produced no matches, and
`git diff --check` passed.

The repository-wide boundary checker passes after the integrated Phase 3
closure. Hardware behavior qualification remains a later gate.
