# TTNN 0.77 substantive host-failure triage

Status: substantive production failure fixed; final full-host suite passes.

Final candidate SHA: `73d414f8b826a7da982df8c8229d4ac41ed8ba33`

Environment:

- interpreter: `/tmp/gwang/ttnn-077-symbol-probe.rGt2zl/venv/bin/python`;
- CPython 3.10.19;
- `ttnn==0.77.0`;
- `torch==2.11.0+cpu`;
- `loguru==0.6.0`;
- `pytest==9.0.3`;
- source import: `PYTHONPATH=src`.

No TT device, checkpoint, cache, remote host, or reset command was used.

## Initial targeted result

Command:

```bash
PYTHONPATH=src /tmp/gwang/ttnn-077-symbol-probe.rGt2zl/venv/bin/python \
  -m pytest -m host -q \
  tests/models/llama33_70b/test_model_profile.py \
  tests/models/llama3_8b/test_model_profile.py \
  tests/models/qwen3_32b/test_module_profiles.py \
  tests/modules/test_tensor_utils.py
```

Result before this lane's fix: 7 failed, 63 passed, 3 deselected.

## Failure classification

| Failure | Count | Classification | Treatment |
|---|---:|---|---|
| Llama 3.3 profile monkeypatches `models.common.models...` | 2 | obsolete extraction characterization | Owned by concurrent stale-path support lane; no source change here |
| Llama 3.1 profile monkeypatches `models.common.models...` | 1 | obsolete extraction characterization | Owned by concurrent stale-path support lane; no source change here |
| Qwen3 checked-in config lookup uses removed root `model_params/...` | 1 | obsolete extraction/path characterization | Owned by concurrent stale-path support lane; no source change here |
| Test constructs `MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(num_workers_per_dram_bank=...)` | 2 parameter cases | TTNN 0.77 signature difference / obsolete test expectation | Use the supported 0.77 constructor contract; production never passes this field |
| `program_config_to_dict(SDPAProgramConfig)` calls broken `repr` | 1 | standalone implementation defect exposed by TTNN 0.77 binding semantics | Serialize readable public data attributes rather than depending on repr |

The first four failures are specifically excluded from this lane so they are not papered over by duplicate edits. The last three were substantive and are closed below.

## TTNN signature evidence

TTNN 0.77 reports this supported constructor:

```text
MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
    *, in0_block_w: int, per_core_M: int, per_core_N: int,
    fused_activation: UnaryWithParam | None = None
)
```

`num_workers_per_dram_bank` is absent. A production search found no use of that argument; only the characterization test attempted it. The test now exercises the actual supported program config and retains round-trip checks for `in0_block_w`, `per_core_M`, `per_core_N`, and `fused_activation`.

TTNN 0.77's `SDPAProgramConfig` has no `to_json`. Its binding exposes readable, non-callable fields:

```text
compute_with_storage_grid_size = CoreCoord(8, 8)
sub_core_grids = None
q_chunk_size = 256
k_chunk_size = 256
exp_approx_mode = None
max_cores_per_head_batch = 16
```

Calling `repr(config)` raises a `TypeError` while converting the bound C++ `std::string`. This is a binding semantic difference, but relying only on repr when public fields exist was an implementation defect in the standalone diagnostic serializer.

## Production adaptation

`tt_transformers.tensor_utils.program_config_to_dict` retains its existing `to_json` fast path. For configs without `to_json`, it now:

1. enumerates public names;
2. skips inaccessible or callable values;
3. recursively preserves JSON scalars, mappings, and sequences;
4. uses nested `to_json` where a field type supports it (for example `CoreCoord` becomes `{"x": 8, "y": 8}`);
5. falls back to string conversion for other public field values;
6. uses a guarded type marker only when an opaque object has neither readable fields nor a usable repr.

This does not vendor TTNN, guess a missing field, or relax the serialized values under test. The resulting SDPA dictionary contains all six supported public fields plus the exact type name.

## Focused verification

Tensor utility file:

```bash
PYTHONPATH=src /tmp/gwang/ttnn-077-symbol-probe.rGt2zl/venv/bin/python \
  -m pytest -m host -q tests/modules/test_tensor_utils.py
```

Result: `15 passed`.

Substantive target set with the four concurrently owned stale tests excluded:

```bash
PYTHONPATH=src /tmp/gwang/ttnn-077-symbol-probe.rGt2zl/venv/bin/python \
  -m pytest -m host -q \
  tests/models/llama33_70b/test_model_profile.py \
  tests/models/llama3_8b/test_model_profile.py \
  tests/models/qwen3_32b/test_module_profiles.py \
  tests/modules/test_tensor_utils.py \
  -k 'not test_decoder_builder_writes_explicit_recipes_on_common_configs and \
      not test_transformer_block_consumes_only_common_configs and \
      not test_checked_in_qwen_config_retains_intermediate_size_25600'
```

Result: `65 passed, 7 deselected`. Four deselections are support-lane tests and three are non-host tests filtered by `-m host`.

No `tests/llm_runtime` failure remained in the subsequent full-host run.

## Extractor and static checks

`tools/extract_foundation.py` now verifies:

- the pinned `models/common/tensor_utils.py` blob `382e92e35d38fb80d60b80e2cd9206538b8fe417`;
- the exact Phase 4 normalized standalone SHA-256 `b74028873bd694498b274010f2003d532fda9de8a271e4a9633c53b72dc7e85c`;
- its existing three canonical-sampling foundation destinations.

Commands:

```bash
PYTHONDONTWRITEBYTECODE=1 python tools/extract_foundation.py
PYTHONDONTWRITEBYTECODE=1 python tools/check_import_boundaries.py src/tt_transformers
triage_cache=$(mktemp -d /tmp/gwang/ttnn077-host-triage-pyc.XXXXXX)
PYTHONPYCACHEPREFIX="$triage_cache" python -m compileall -q -f \
  src/tt_transformers/tensor_utils.py \
  tools/extract_foundation.py \
  tests/modules/test_tensor_utils.py
```

Results: extractor pass, static boundary pass, compile pass.

## Full-host tracking

The first post-fix full command was:

```bash
PYTHONPATH=src /tmp/gwang/ttnn-077-symbol-probe.rGt2zl/venv/bin/python -m pytest -m host -q
```

It produced `1978 passed, 74 failed, 18 skipped, 6790 deselected, 81 subtests passed`. The remaining named failures were in the concurrently owned demo/stale-path/legacy/capability set; there were no runtime-suite failures and the substantive serialization failures were gone.

After the support lane settled its stale demo/path/legacy expectations and
added the hardware-runner host policies, exact full runs passed on both
supported interpreters:

```text
CPython 3.10.19: 2170 passed, 28 skipped, 6791 deselected,
                 5 warnings, 81 subtests passed, exit 0
CPython 3.12.13: 2170 passed, 28 skipped, 6791 deselected,
                 5 warnings, 81 subtests passed, exit 0
```

There were zero failures and zero errors. The 28 skips are explicit
hardware-selection/no-device skips and approved retired TTTv1 compatibility
cases. The five warnings are existing pytest empty-regex warnings in runtime
configuration tests, not TTNN semantic failures.

The 3.12 process emitted shutdown-only nanobind diagnostics after pytest's
successful result: 10 leaked instances, 36 leaked types, and 330 leaked
functions, ending with a likely binding reference-counting issue. This is
recorded as TTNN binding-lifecycle feedback, not hidden and not counted as a
pytest failure. Full evidence: `host-ttnn-0.77.md` and retained output
`/tmp/gwang/tttv2-73d-py312-host-final.log`.

No runtime or concrete-model production adaptation was required. The three
model-profile candidates were confirmed as obsolete path/namespace
characterization, and every `tests/llm_runtime` host test passed.
