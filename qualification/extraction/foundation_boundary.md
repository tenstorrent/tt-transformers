# Phase 3 reusable-foundation boundary closure

Source baseline: `/localdev/gwang/tt-metal` at
`00748e6ac7b65f50e5c2af07f6e7c1c535c7f4c0`

Phase 2 report: `qualification/extraction/foundation.md`

Status: **complete for the reusable foundation; no hardware used**

## Outcome

- Removed all ten characterized module `from_model_args` factories.
- Removed the module-owned `RotarySetup1D.get_rot_idxs` and
  `RotarySetup1D.get_rot_mats` compatibility adapters.
- Replaced the active TTTv1 `Mode` import with
  `tt_transformers.modules.mode.Mode` and a neutral `normalize_mode`
  helper.
- Kept direct constructors and every `from_config` constructor.
- Redirected the two real runtime RoPE adapter calls to
  `prepare_rot_idxs` and `decode_forward`.
- Kept `models.common.utils` closed through
  `tt_transformers.sampling.logprobs`.
- Removed all `models.*` imports from the reusable foundation.
- Added 16 host characterization cases.
- Did not run or reset hardware.

## Caller scans before removal

### Module factories

The extracted production scan found no call to any of the ten factories in
`src/tt_transformers/models` or `src/tt_transformers/llm_runtime`. Concrete
models already use direct construction or `from_config`.

The only extracted calls were in legacy-only module tests. Seventeen test
functions were removed with the retired API:

- `test_attention_1d_vs_reference_from_model_args`
- `test_attention_1d_rejects_galaxy`
- `test_embedding_1d_vs_reference_from_model_args`
- `test_lm_head_1d_vs_reference_from_model_args`
- `test_from_model_args_rejects_galaxy` in the LM-head suite
- `test_mlp_1d_vs_reference_from_model_args`
- `test_mlp_2d_vs_reference_from_model_args`
- `test_mlp_2d_rejects_non_galaxy_from_model_args`
- `test_rmsnorm_1d_vs_reference_from_model_args`
- `test_rmsnorm_1d_rejects_galaxy`
- `test_rmsnorm_2d_vs_reference_from_model_args`
- `test_rmsnorm_2d_rejects_non_tg`
- `test_rope_1d_vs_reference_from_model_args`
- `test_rope_1d_from_model_args_rejects_galaxy`
- `TestPenalties1D.test_from_model_args` and
  `TestPenalties1D.test_rejects_galaxy`
- the four factory-specific methods in `TestSampling1D`:
  `test_from_model_args`, `test_from_model_args_with_galaxy_num_links`,
  `test_from_model_args_with_sampling_ag_config`, and
  `test_rejects_galaxy`

The pre-removal AST scan found and removed exactly these 20 test functions or
class-owned test methods.

### RoPE adapters

The production caller scan found exactly two calls, both in
`src/tt_transformers/llm_runtime/decode.py`:

| Old call | Retained core call | Equivalence |
|---|---|---|
| `rope_setup.get_rot_idxs(nonnegative, on_host=True)` | `prepare_rot_idxs(rope_setup.config, nonnegative, on_host=True)` | The removed adapter was a direct delegate. |
| `rope_setup.get_rot_mats(inputs.rotary_indices)` | `rope_setup.decode_forward(inputs.rotary_indices)` | The input is already a TTNN rotation-index tensor, so the removed adapter directly called `decode_forward`. |

One retained module hardware test similarly changed
`rope.get_rot_mats(rot_idxs_host)` to
`rope.decode_forward(rot_idxs_host)`. Other `get_rot_mats` matches in the
attention suites belong to a test-local `RotarySetupHelper`, not
`RotarySetup1D`, and were correctly left unchanged.

Concrete model callers already use `RotarySetup1D.from_config`,
`prepare_rot_idxs`, `decode_forward`, or `get_both_trans_mats`.

## Removed factories

| Class/symbol | Pinned source | Original line | Original signature | Retained construction |
|---|---|---:|---|---|
| `Attention1D.from_model_args` | `models/common/modules/attention/attention_1d.py` | 1377 | `(cls, mesh_device, tt_ccl, args, state_dict, weight_cache_path, layer_num: int, transformation_mats: dict[str, ttnn.Tensor], paged_attention_config=None, use_paged_kv_cache: bool=False)` | Direct `__init__` and `from_config` |
| `Embedding1D.from_model_args` | `models/common/modules/embedding/embedding_1d.py` | 153 | `(cls, mesh_device, args, weight_cache_path, state_dict, dtype, embed_scale: float=1.0)` | Direct `__init__` and `from_config` |
| `LMHead1D.from_model_args` | `models/common/modules/lm_head/lm_head_1d.py` | 170 | `(cls, mesh_device, args, state_dict, state_dict_prefix, weight_cache_path, max_columns_per_device, dtype=None, model_config=None, tt_ccl=None)` | Direct `__init__` and `from_config` |
| `MLP1D.from_model_args` | `models/common/modules/mlp/mlp_1d.py` | 479 | `(cls, mesh_device, tt_ccl, args, state_dict, weight_cache_path, layer_num: int, dtype=None, model_config=None, state_dict_prefix: Optional[str]=None, prefetcher=None)` | Direct `__init__` and `from_config` |
| `MLP2D.from_model_args` | `models/common/modules/mlp/mlp_2d.py` | 483 | `(cls, mesh_device, tt_ccl, args, state_dict, weight_cache_path, layer_num: int, state_dict_prefix: Optional[str]=None)` | Direct `__init__` and `from_config` |
| `RMSNorm1D.from_model_args` | `models/common/modules/rmsnorm/rmsnorm_1d.py` | 339 | `(cls, mesh_device, tt_ccl, args, state_dict, weight_cache_path, layer_num: int, weight_key: str, state_dict_prefix: Optional[str]=None, sharded_program_config=None, sharded_output_config=None)` | Direct `__init__` and `from_config` |
| `RMSNorm2D.from_model_args` | `models/common/modules/rmsnorm/rmsnorm_2d.py` | 277 | `(cls, mesh_device, tt_ccl, args, state_dict, weight_cache_path, layer_num: int, weight_key: str, state_dict_prefix: Optional[str]=None)` | Direct `__init__` and `from_config` |
| `RotarySetup1D.from_model_args` | `models/common/modules/rope/rope_1d.py` | 280 | `(cls, device: Any, args, model_name: str='unknown')` | Direct `__init__` and `from_config` |
| `Penalties1D.from_model_args` | `models/common/modules/sampling/penalties_1d.py` | 148 | `(cls, mesh_device, args) -> Penalties1D` | Direct `__init__` and `from_config` |
| `Sampling1D.from_model_args` | `models/common/modules/sampling/sampling_1d.py` | 674 | `(cls, mesh_device, tt_ccl, args, model_config=None) -> Sampling1D` | Direct `__init__` and `from_config` |

No source from `models.tt_transformers.tt.model_config` or
`models.tt_transformers.tt.rope` was copied. Their only foundation consumers
were inside the removed factories.

## Neutral Mode ownership

The pinned `Mode` values are preserved exactly:

```python
class Mode(Enum):
    DECODE = "decode"
    PREFILL = "prefill"
```

The owner is now `src/tt_transformers/modules/mode.py`. MLP dispatch accepts
either the neutral enum or the existing `"decode"`/`"prefill"` strings.
`normalize_mode` only unwraps the enum; the pre-existing string dispatch
behavior is unchanged.

The provenance row for `models/tt_transformers/tt/common.py` was updated from
the provisional `src/tt_transformers/models/mode.py`/RoPE-scaling destinations
to `src/tt_transformers/modules/mode.py`. The RoPE-scaling helper was not
copied because its only caller was the removed
`RotarySetup1D.from_model_args` bridge.

## Characterization tests

`tests/host/test_foundation_boundary.py` is host-only and does not require a
TTNN wheel to collect. It checks:

- exact direct-constructor parameter names for all ten retained classes;
- presence of `from_config` and absence of `from_model_args`;
- absence of both retired RoPE adapters;
- enum and string normalization for decode/prefill dispatch;
- the narrow `sampling.logprobs` owner and absence of
  `models.common.utils`.

Existing direct/config-based module hardware tests remain in place. Only tests
whose purpose was the removed TTTv1 factory were deleted.

## Reproducible validation

### Caller and namespace closure

```bash
rg -n 'def from_model_args|\.from_model_args\(|models\.' \
  src/tt_transformers/modules \
  src/tt_transformers/sampling \
  src/tt_transformers/device_utils.py \
  src/tt_transformers/tensor_utils.py \
  src/tt_transformers/mesh_utils.py

rg -n 'rope_setup\.(get_rot_idxs|get_rot_mats)' \
  src/tt_transformers/models src/tt_transformers/llm_runtime
```

Result: both commands return no matches.

### Static boundary checker

```bash
python3 tools/check_import_boundaries.py src/tt_transformers
```

Result: exit code 0, no output.

### Syntax

```bash
env PYTHONPYCACHEPREFIX=/tmp/gwang/tttv2_foundation_boundary_pycache \
  python3 -m compileall -q \
  src/tt_transformers/modules \
  src/tt_transformers/sampling \
  src/tt_transformers/device_utils.py \
  src/tt_transformers/tensor_utils.py \
  src/tt_transformers/mesh_utils.py \
  src/tt_transformers/llm_runtime/decode.py \
  tests/host/test_foundation_boundary.py
```

Result: exit code 0, no output.

### Focused host characterization

```bash
PYTHONPATH=src pytest -q --confcutdir=tests/host \
  tests/host/test_foundation_boundary.py
```

Result:

```text
................                                                         [100%]
16 passed in 0.06s
```

`--confcutdir=tests/host` keeps this dependency-free host characterization
independent of the separately migrating hardware fixture stack.

## Concurrent provenance completeness correction

During this goal, the full pinned qualification/support-root audit identified
21 previously unassigned `models/common/readiness_check/` files:

- 15 tools/assets now map to `qualification/readiness/`;
- six tests now map to `tests/qualification/readiness/`.

Every readiness blob was verified. The provenance inventory now contains 364
unique rows, 338 assigned destinations, and the same 26 explicit MoE-only
exclusions. The support extraction lane was notified to extract these newly
assigned rows.
