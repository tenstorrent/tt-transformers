# Phase 2 concrete-model extraction

Source repository: `/localdev/gwang/tt-metal`

Source revision: `00748e6ac7b65f50e5c2af07f6e7c1c535c7f4c0`

Status: mechanical extraction and reviewed deterministic policy transforms
complete; hardware qualification remains a separate evidence boundary

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

## Llama 3.3 folded-prefill profile policy

Post-qualification W6 triage identified one model-owned operator-policy delta
in `llama33_70b/model.py`. The pinned source selected minimal matmul solely from
the absence of `DISABLE_MINIMAL_MATMUL`, which made the accuracy recipe change
operator family when a folded batched prefill crossed the common modules'
`seq_len > 128` threshold. The reviewed standalone transform instead records
the choice in the immutable precision recipe:

- `LLAMA33_70B_ACCURACY` inherits `prefill_minimal_matmul=False`, keeping folded
  QKV/W2 on `ttnn.linear` for the strict batch-one-versus-batched oracle;
- `LLAMA33_70B_PERFORMANCE` sets `prefill_minimal_matmul=True`, retaining the
  minimal-matmul TTFT tradeoff;
- `_resolve_llama33_70b_profile` resolves the final value as the profile choice
  **and** absence of `DISABLE_MINIMAL_MATMUL`, so the environment remains a
  global force-off and cannot force the accuracy profile on.

`tools/extract_model_packages.py` reproduces the exact pinned-source edits and
refuses source drift in each changed block. It reparses the result and asserts
the default-false dataclass field, accuracy's unoverridden linear policy,
performance's explicit true policy, and the exact profile-and-environment
force-off expression before comparing destination bytes.

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
