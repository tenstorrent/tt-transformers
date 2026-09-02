# Phase 3/7 cache and environment policy

Status: implemented across all twelve HF adaptors; implicit defaults are identity-namespaced and established explicit cache paths remain compatible.

Source characterization: `qualification/analysis/environment_variables.csv` and `qualification/analysis/runtime_assumptions.csv` at pinned `tt-metal` revision `00748e6ac7b65f50e5c2af07f6e7c1c535c7f4c0`.

## Result

`src/tt_transformers/cache_environment.py` now owns typed environment parsing, explicit offline selection, cache-root resolution, versioned cache identity, compatibility-path reporting, and secret-redacted preflight output.

The policy is integrated into these twelve adaptors with backward-compatible optional revision extensions where the pinned source lacked one:

1. `deepseek_r1_distill_qwen_14b`;
2. `llama32_1b`;
3. `llama32_3b`;
4. `llama33_70b`;
5. `llama3_8b`;
6. `mistral_7b`;
7. `phi4`;
8. `qwen25_72b`;
9. `qwen25_7b`;
10. `qwen25_coder_32b`;
11. `qwen2_7b`;
12. `qwen3_32b`.

Every adaptor now:

- obtains `local_files_only` from `offline_mode()`;
- reads boolean tuning controls through `environment_flag()`;
- resolves cache directories through `resolve_model_cache()` with checkpoint revision, mesh architecture/topology/sharding, dtype, layout, and conversion-schema inputs;
- emits one `report_model_preflight()` call with checkpoint, cache, architecture/topology, and conversion identity inputs;
- has no direct `os.getenv`, `os.environ`, or `Path("model_cache")` use.

## Typed environment policy

The registry includes all 21 keys found in Phase 0. Their declared kinds are boolean, integer, path, or string. The eight Qwen demo/number-divergence keys are marked non-production but remain typed so the inventory is complete.

The standalone policy adds:

- `TT_TRANSFORMERS_OFFLINE`: explicit package-level offline switch;
- `TT_TRANSFORMERS_CACHE`: explicit standalone default cache root;
- `XDG_CACHE_HOME`: standard user cache-root input.

Boolean values are normalized case-insensitively:

```text
true:  1, true, yes, on
false: empty, 0, false, no, off
```

An unrecognized value raises `ValueError`; it is never treated as true merely because the string is non-empty. This fixes prior inconsistencies such as `bool(os.getenv(...))`, where `"0"` disabled nothing.

Offline mode is true only when at least one of these explicit controls is true:

- `TT_TRANSFORMERS_OFFLINE`;
- `HF_HUB_OFFLINE`;
- `TRANSFORMERS_OFFLINE`.

`CI` is parsed and reported but does not select offline mode by itself. This avoids coupling checkpoint-network policy to a generic execution context.

HF model selection and numeric tuning are also typed where the adaptors consume them:

- Llama 3.1-8B's `HF_MODEL` remains a string fallback;
- `MAX_PREFILL_CHUNK_SIZE` is parsed as an integer;
- `DISABLE_BATCHED_PREFILL` and `DISABLE_BATCHED_EXTRACT` use the normalized boolean parser.

## Cache-root resolution

For the eleven adaptors with a `cache_dir` argument, resolution order is:

1. explicit `cache_dir`, used as the exact final directory;
2. legacy `TT_CACHE_PATH`, used as the exact final directory;
3. `TT_TRANSFORMERS_CACHE/<hf-model-id>/identity-1-<digest>/<topology>`;
4. `XDG_CACHE_HOME/tt-transformers/<hf-model-id>/identity-1-<digest>/<topology>`;
5. `$HOME/.cache/tt-transformers/<hf-model-id>/identity-1-<digest>/<topology>`.

Thus the implicit default no longer depends on process CWD and cannot silently reuse tensors from a different package, TTNN, checkpoint, conversion schema, architecture, topology, dtype, layout, or sharding identity. The topology remains the final path segment.

Llama 3.1-8B preserves its distinct compatibility contract:

- `TT_CACHE_PATH/<device-name>` remains the configured path;
- only an explicit legacy path is created eagerly;
- `EROFS`, `EACCES`, or `EPERM` still falls back to `TT_CACHE_FALLBACK_PATH/<model-basename>/<device-name>`;
- fallback defaults to `/tmp/tttv2_model_cache`;
- the implicit standalone default remains lazily created by tensor-cache use, matching the pinned behavior.

Topology suffixes are unchanged:

| Model | Preserved suffix policy |
|---|---|
| DeepSeek-R1-Distill-Qwen-14B | TP2 `N300`, TP4 `N150x4`, TP8 `T3K` |
| Llama 3.2 1B/3B | 1 device `N150`, 2 `N300`, 8 `T3K` |
| Llama 3.3 70B | resolved SKU string |
| Llama 3.1 8B | `get_device_name(mesh_device)` |
| Mistral 7B | 1 `N150`, 2 `N300`, 8 `T3K`, otherwise `TP<n>` |
| Phi-4 | `N300` |
| Qwen2/2.5 7B | `N300` |
| Qwen2.5 72B/Coder 32B | `T3K` |
| Qwen3 32B | resolved SKU string |

## Warm-cache compatibility and migration

The identity namespace is intentionally not applied to established overrides. A prior CWD or hardware cache such as:

```text
<old-cwd>/model_cache/<hf-model-id>/<topology>
```

is reported as `legacy_cwd_path`, but it is never scanned, moved, copied, deleted, or materialized by the policy. Users and hardware jobs preserve an established warm cache exactly with either:

```text
cache_dir=<existing-final-directory>
```

or:

```text
TT_CACHE_PATH=<existing-final-directory>
```

For Llama 3.1-8B, `TT_CACHE_PATH` continues to mean the parent of the device-name suffix, as before.

For both overrides, preflight reports `identity_applied_to_path: false` and emits a release-qualification warning. The complete identity is still reported so evidence can bind an override to its exact inputs; the path is never described as versioned. Callers relying on an old implicit default must opt into that exact path or migrate it themselves.

## Versioned cache identity

`CacheIdentity` has schema version 1 and includes all required invalidation inputs:

- `tt-transformers` distribution version;
- TTNN distribution version;
- exact HF model ID;
- HF revision, or the explicit `unversioned` marker;
- tensor conversion schema (`tttv2-lazy-weight-v1`);
- device architecture;
- preserved topology/SKU suffix;
- requested weight dtype inputs;
- default layout inputs;
- mesh shape and device-count sharding inputs.

The canonical sorted JSON representation is SHA-256 hashed. Implicit defaults insert `identity-<schema-version>-<digest>` immediately before the preserved topology suffix and preflight reports `identity_applied_to_path: true` only when that exact component is present. Changes to either software version, HF revision, architecture, topology, dtype, layout, or sharding inputs select a different implicit namespace.

The directory identity records the requested model weight dtype, default tile layout, and mesh shape/device count. Individual `LazyWeight` filenames retain their finer-grained dtype, layout, mapper, memory-config, and source-shape fingerprints. A semantic conversion-recipe change must bump `CONVERSION_SCHEMA`; this keeps the directory identity model-neutral without discarding the established per-tensor invalidation layer.

No sidecar is created, read, or validated during resolution. Sidecar validation for established explicit caches remains deferred because adding or requiring metadata there could mutate, reject, or trigger cold replacement of valuable hardware warm caches. Release evidence must bind those override paths to the preflight identity instead.

## Secret-redacted preflight

Each adaptor reports:

- effective offline and CI context;
- HF model ID and revision;
- resolved cache path and its source (`cache_dir`, `TT_CACHE_PATH`, standalone/XDG/user root, or permission fallback);
- whether the exact identity digest is present in the path and an explicit release warning when it is not;
- the old CWD compatibility path;
- full cache identity and digest;
- only environment keys known to this policy.

Environment names containing token, password, secret, credential, API-key, or access-key markers are emitted only as `<redacted>`. Host tests serialize the report and prove that a supplied `HF_TOKEN` value does not appear.

## Extractor drift policy

`tools/extract_model_packages.py` still verifies every pinned source blob. For the twelve intentionally policy-normalized HF adaptors it additionally:

- pins the exact normalized destination SHA-256;
- rejects direct CWD-cache or direct environment access;
- requires the shared policy imports and exactly one preflight call;
- requires every adaptor cache resolution to pass HF ID/revision, topology, mesh, and dtype identity inputs;
- compares the current `from_pretrained` AST arguments to the pinned source signature;
- does not overwrite an existing policy-normalized adaptor during `--write`.

All other concrete-model files remain reproducible through the prior pinned transformations. The concurrently introduced Qwen3 local Transformers import and lazy package initializers remain preserved.

## Validation

Focused host tests:

```bash
cd /localdev/gwang/tt_transformers
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/localdev/gwang/tt_transformers/src \
  python -m pytest -q --confcutdir=tests/host \
  tests/host/test_cache_environment.py
```

Result: `38 passed`.

Coverage includes:

- normalized true/false/error parsing;
- offline policy independent of `CI`;
- all 21 characterized keys in the typed registry;
- identity-namespaced standalone, XDG, and user default roots with no CWD dependence;
- invalidation on package/TTNN/checkpoint/dtype/architecture identity changes;
- exact `cache_dir` and `TT_CACHE_PATH` compatibility across identity changes, with explicit warnings;
- Llama path suffix and permission fallback;
- complete/stable/sensitive cache identity;
- secret-redacted preflight;
- static integration of all twelve adaptors;
- all twelve preserved topology suffixes.

The complete host-policy directory also passes without loading the hardware
conftest:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/localdev/gwang/tt_transformers/src \
  python -m pytest -q --confcutdir=tests/host tests/host
```

Result: `87 passed`.

Static and syntax commands:

```bash
PYTHONDONTWRITEBYTECODE=1 python tools/check_import_boundaries.py src/tt_transformers
PYTHONDONTWRITEBYTECODE=1 python tools/extract_model_packages.py
cache_compile=$(mktemp -d /tmp/gwang/tttv2-cache-environment-pyc.XXXXXX)
PYTHONPYCACHEPREFIX="$cache_compile" python -m compileall -q -f \
  src/tt_transformers/cache_environment.py \
  src/tt_transformers/models \
  tools/check_import_boundaries.py \
  tools/extract_model_packages.py \
  tests/host/test_cache_environment.py
```

Results: checker exit 0, extractor drift check exit 0, compile exit 0.

No model was constructed, no checkpoint was loaded, no model tensor cache was rematerialized by validation, and no TT hardware or reset command was used.
