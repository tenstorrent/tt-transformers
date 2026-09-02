# Phase 3 optional-dependency and canonical-sampling cleanup

Status: **complete for the approved transforms; no hardware used**

Source baseline: `/localdev/gwang/tt-metal` at `00748e6ac7b65f50e5c2af07f6e7c1c535c7f4c0`.

## Outcome

- `tt_transformers.sampling.sampling_params.SamplingParams` is the only dataclass owner.
- `sampling.generator` imports that owner; its fields/defaults and helper behavior are unchanged.
- `tt_transformers.sampling.SamplingParams` lazily resolves to the same class.
- Eleven source model-package initializers preserve their exact `__all__` order through cached lazy export maps.
- Llama3-8B, whose pinned source had no package initializer or package exports, has an empty `__all__` initializer and gains no public surface.
- Qwen3-32B's core `model.py` imports Transformers only inside its explicit `from_pretrained` loader.
- Importing the root, models root, any concrete model package, or any core `model.py` source no longer eagerly requires `transformers` or `tqdm`.
- No `hf_adaptor.py` body was edited.

## Canonical SamplingParams contract

The canonical frozen dataclass retains the exact field order and defaults:

| Field | Default |
|---|---|
| `temperature` | required |
| `top_k` | required |
| `top_p` | required |
| `presence_penalty` | `0.0` |
| `frequency_penalty` | `0.0` |
| `repetition_penalty` | `1.0` |
| `seed` | `None` |
| `enable_log_probs` | `False` |
| `num_logprobs` | `0` |

`SAMPLING_PARAM_FIELDS` is still derived with `dataclasses.fields`. Focused tests prove `broadcast_sampling_params`, `slice_sampling_params`, and `chunk_sampling_params` return the canonical class and preserve representative list/scalar behavior.

## Lazy model export policy

Each non-empty model initializer now contains only standard-library imports, an `_EXPORTS` map, the unchanged ordered `__all__`, cached `__getattr__`, and `__dir__`.

Accessing a core symbol imports only its owning model/executor module. Accessing an HF or generator symbol imports its owner on demand, so missing optional dependencies fail at the access point rather than package import.

The root package and `tt_transformers.models` root remain non-eager. The public API policy counts are unchanged:

- 367 `supported_public`;
- 950 `model_local_public`;
- 13 `compatibility_only`;
- 430 `private_by_policy`.

The API signature checker now records 152 model-package rows as `lazy_export_rewrite`, 1 remaining namespace-only re-export, 1,576 exact signatures, and 31 approved removals. All 228 explicit source exports remain `model_local_public`.

## Deterministic extractors

`tools/extract_foundation.py` verifies the three pinned canonical-sampling files and reproduces removal of the duplicate dataclass plus the canonical lazy export. It also verifies the pinned source blobs and exact final hashes for the three bounded post-extraction sampling adaptations: scalar policy expansion without scalar-seed broadcast, batched prefill admission/request-order seed publication, and conservative survivor-history invalidation.

`tools/extract_model_packages.py` now reproduces:

- lazy conversion of all eleven pinned model `__init__.py` files;
- the Qwen3 core optional-import move;
- the already-approved Llama3 dispatcher removal;
- the generated empty Llama3-8B package initializer.

The model extractor's `--scope optional-deps` gate verifies these 12 pinned transforms plus the generated initializer independently of concurrent cache-policy edits in `hf_adaptor.py`. The full extractor remains the aggregate lane and was not used to overwrite cache-agent work.

## Focused host tests

`tests/host/test_optional_dependencies_sampling.py` contains 28 cases:

- 12 initializer/export-map checks;
- isolated import of all 12 packages with `transformers` and `tqdm` blocked;
- on-access optional-export failure checks for all 10 packages that explicitly export HF symbols;
- 12 core-model AST checks for no eager optional imports;
- canonical class identity/default and pure helper checks.

The sampling-generator tests inject only minimal TTNN/sampling module stubs; no device behavior is simulated or claimed.

## Validation

```bash
python3 tools/extract_foundation.py
python3 tools/extract_model_packages.py --scope optional-deps
python3 tools/check_import_boundaries.py src/tt_transformers
python3 qualification/analysis/check_public_api_policy.py
env PYTHONPYCACHEPREFIX=/tmp/gwang/tttv2_optional_sampling_pycache \
  python3 -m compileall -q src/tt_transformers \
  tools/extract_model_packages.py tools/extract_foundation.py \
  tests/host/test_optional_dependencies_sampling.py
PYTHONPATH=src pytest -q --confcutdir=tests/host \
  tests/host/test_optional_dependencies_sampling.py
```

Results:

```text
verified 3 pinned canonical-sampling files, 3 exact-hash standalone sampling adaptations, and Phase 4 tensor_utils normalization
verified 12 pinned optional-dependency transforms plus 1 generated package initializer
static import-boundary checker: exit 0
public API policy checker: exit 0
compileall: exit 0
............................                                             [100%]
28 passed in 2.94s
```

## Evidence boundary

This is host import, identity, helper, signature, and extractor evidence. It does not exercise HF loading, TTNN tensor semantics, model correctness, cache behavior, or hardware. No hardware/remotes, examples/support files, main work log, or HF adaptor body was changed.
