# Phase 3 TTNN default-device ownership

Status: installed production default-device ownership closed and statically enforced.

Characterization source: `qualification/analysis/runtime_assumptions.csv` at pinned `tt-metal` revision `00748e6ac7b65f50e5c2af07f6e7c1c535c7f4c0`.

## Outcome

There is no direct `SetDefaultDevice` or `GetDefaultDevice` call anywhere under `src/tt_transformers` except the single owner module `device_ownership.py`.

The Phase 0 source inventory contained 38 writes:

| Source classification | Count | Standalone treatment |
|---|---:|---|
| HF adaptor construction | 11 | removed; every provider/model config receives `mesh_device` explicitly |
| provider-neutral Qwen model builders | 3 | removed; builders already thread `mesh_device` into every owned config/resource |
| two package-local Qwen hybrid demo/test modules | 24 | relocated outside installed production during extraction; not part of `src/tt_transformers` or this production acceptance gate |
| total | 38 | all installed-production writes eliminated |

Llama 3.1-8B's HF adaptor already had no global write; it was audited with the other eleven.

The ten characterized reusable-module reads remain only as explicit compatibility fallbacks. Every direct call was replaced by `compatibility_default_device(ttnn, owner=...)`, which rejects unregistered owners and raises a clear explicit-`mesh_device` error when no scoped default is active.

## Explicit propagation

All twelve integrated provider paths now construct against their `mesh_device` argument without mutating process state. The removed writes were not replaced by long-lived scopes. This is intentional: the extracted config builders already set module `mesh_device` fields or attach the device to their `LazyWeight` values, and the runtime passes the model's explicit device into cache, prefill, decode, sampling, and output owners.

Public model/adaptor signatures did not change. `tools/extract_model_packages.py` compares every current `from_pretrained` argument AST with its pinned source signature and verifies exact normalized adaptor hashes.

The three removed provider-neutral writes were:

- `build_qwen25_72b_model`;
- `build_qwen25_coder_32b_model`;
- `Qwen3_32B.from_pretrained`.

Their pinned extractor transforms now remove exactly one characterized write per source file. Qwen3's concurrent local Transformers import remains preserved.

## Scoped compatibility owner

`default_device_scope(ttnn_api, device, owner=...)` is the only approved writer. It:

- captures the exact previous object, including `None`;
- holds one process-wide re-entrant lock for the complete scope;
- supports same-thread nesting and restores in LIFO order;
- serializes different threads so another owner cannot observe or replace the temporary default;
- restores after normal completion or a body exception;
- makes a best-effort exact restoration if `SetDefaultDevice` mutates and then raises;
- maintains a thread-local owner stack for diagnostics;
- rejects `None` devices and empty owners.

Standalone construction does not need this scope after explicit propagation. It exists only for callers that deliberately exercise one of the audited legacy convenience constructors.

## Retained fallback ledger

The ledger is code-owned as `DEFAULT_DEVICE_FALLBACK_LEDGER`. Its ten entries match the ten production call sites exactly:

| Owner | Explicit values tried before fallback | Compatibility reason |
|---|---|---|
| `modules.attention._attention_mesh_device` | `config.mesh_device`, then `config.wqkv.device` | simple/power config compatibility |
| `modules.embedding._resolve_embedding1d_config` | config and weight device inference | legacy simple constructor |
| `modules.lm_head._derive_lm_head_mesh_device` | config and weight device inference | legacy simple constructor |
| `modules.mlp._resolve_mlp1d_mesh` | `config.mesh_device`, then `config.w1.device` | legacy simple constructor |
| `modules.mlp._resolve_mlp2d_config` | config and weight device inference | legacy 2D constructor |
| `modules.rmsnorm._derive_rmsnorm_mesh_device` | config and weight device inference | legacy 1D constructor |
| `modules.rmsnorm._resolve_2d_config` | config and weight device inference | legacy 2D constructor |
| `modules.rope._resolve_rope_config` | config and `LazyWeight` device inference | legacy simple constructor |
| `modules.sampling._resolve_penalties1d_config` | `config.mesh_device` | legacy convenience constructor |
| `modules.sampling._resolve_sampling1d_config` | `config.mesh_device` | legacy convenience constructor |

The fallback is read-only. It does not silently install a device and cannot introduce a cross-owner mutation. With no active scoped compatibility default, each site fails rather than guessing.

## Static enforcement

`tools/check_import_boundaries.py` rejects any direct `SetDefaultDevice` or `GetDefaultDevice` call outside `src/tt_transformers/device_ownership.py`.

The host ledger test independently parses every production Python file and asserts:

- zero direct accesses outside the owner module;
- every `compatibility_default_device` call has a literal owner;
- the sorted call-site owners equal the sorted ledger owners exactly;
- the ledger contains ten unique entries with non-empty rationales.

The model extractor additionally rejects default-device symbols in all twelve normalized HF adaptors and reproduces the three model-builder removals.

## Focused fake-TTNN proof

`tests/host/test_device_ownership.py` uses no TTNN installation or hardware. Its fake binding covers:

1. exact normal restoration;
2. restoration after a body exception;
3. nested LIFO restoration and owner-stack state;
4. repeated construction without retained ownership;
5. restoration when a binding mutates and then raises during `SetDefaultDevice`;
6. two-thread serialization with proof that the second owner cannot enter or observe the first owner's default;
7. registered/unregistered/missing-default fallback behavior;
8. exact ledger size and uniqueness;
9. full production AST/ledger correspondence.

Focused result:

```text
9 passed
```

The complete host-policy suite also passes:

```bash
cd /localdev/gwang/tt_transformers
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/localdev/gwang/tt_transformers/src \
  python -m pytest -q --confcutdir=tests/host tests/host
```

Result: `98 passed` at final integrated validation.

## Compile, checker, and extractor evidence

```bash
ownership_cache=$(mktemp -d /tmp/gwang/tttv2-device-ownership-pyc.XXXXXX)
PYTHONPYCACHEPREFIX="$ownership_cache" python -m compileall -q -f \
  src/tt_transformers/device_ownership.py \
  src/tt_transformers/modules \
  src/tt_transformers/models \
  tools/check_import_boundaries.py \
  tools/extract_model_packages.py \
  tests/host/test_device_ownership.py
PYTHONDONTWRITEBYTECODE=1 python tools/check_import_boundaries.py src/tt_transformers
PYTHONDONTWRITEBYTECODE=1 python tools/extract_model_packages.py
```

Results:

- compile exit 0 with bytecode directed under `/tmp/gwang`;
- static checker exit 0 with no output;
- extractor drift check exit 0 for 71 pinned concrete-model files plus the generated package initializer.

No model or checkpoint was loaded, no cache was read or written, and no TT hardware, device-open, or reset command was used.
