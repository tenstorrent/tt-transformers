# TTNN 0.77.0 CPython 3.12 symbol qualification

Status: **passed for symbol presence only**

## Result

The published `ttnn==0.77.0` CPython 3.12 wheel imports in a fresh
CPython 3.12.13 environment, and all 14 unstable/private API paths used by the
pinned TTTv2 source resolve:

- 13 `ttnn.experimental` operations;
- 1 private `ttnn._ttnn.tensor.dump_tensor_flatbuffer` function;
- 60 total occurrences in the pinned production-source analysis;
- 0 missing paths.

The machine-readable probe result is
`qualification/reports/ttnn-0.77.0-symbols-py312.json`.

## Isolated environment

| Item | Verified value |
|---|---|
| Managed interpreter | `/tmp/gwang/uv-python/cpython-3.12.13-linux-x86_64-gnu/bin/python3.12` |
| Python | `3.12.13` |
| Temporary venv | `/tmp/gwang/tttv2-py312.1MKAuO/venv` |
| TTNN distribution | `0.77.0` |
| TTNN wheel tag | `cp312-cp312-manylinux_2_34_x86_64` |
| Torch distribution | `2.11.0+cpu` (CPU local build of release `2.11.0`) |
| Torch wheel tag | `cp312-cp312-manylinux_2_28_x86_64` |
| Loguru distribution | `0.6.0` |
| Dependency consistency | `pip check`: no broken requirements |
| Platform | `Linux-4.18.0-553.el8_10.x86_64-x86_64-with-glibc2.35` |

The final checks ran from `/tmp/gwang` with `PYTHONPATH` unset. The observed
`sys.path` contained only the current empty-path entry, the managed CPython
standard library, and the temporary venv's `site-packages`. It contained
neither a `tt-metal` checkout nor `tt_transformers/src`. Module files for
TTNN, Torch, and Loguru all resolved inside the temporary venv.

No `tt-transformers` wheel or source package was built or installed in this
lane.

## Symbol results

| API | Source occurrences | Present | Object type | Inspectable signature |
|---|---:|---|---|---|
| `ttnn._ttnn.tensor.dump_tensor_flatbuffer` | 1 | yes | `nb_func` | `(*args, **kwargs)` |
| `ttnn.experimental.all_gather_async` | 31 | yes | `FastOperation` | `(*function_args, **function_kwargs)` |
| `ttnn.experimental.all_gather_matmul_async` | 1 | yes | `FastOperation` | `(*function_args, **function_kwargs)` |
| `ttnn.experimental.minimal_matmul` | 3 | yes | `FastOperation` | `(*function_args, **function_kwargs)` |
| `ttnn.experimental.nlp_concat_heads` | 1 | yes | `FastOperation` | `(*function_args, **function_kwargs)` |
| `ttnn.experimental.nlp_concat_heads_decode` | 1 | yes | `FastOperation` | `(*function_args, **function_kwargs)` |
| `ttnn.experimental.nlp_create_qkv_heads` | 1 | yes | `FastOperation` | `(*function_args, **function_kwargs)` |
| `ttnn.experimental.nlp_create_qkv_heads_decode` | 1 | yes | `FastOperation` | `(*function_args, **function_kwargs)` |
| `ttnn.experimental.paged_fill_cache` | 6 | yes | `FastOperation` | `(*function_args, **function_kwargs)` |
| `ttnn.experimental.paged_fused_update_cache` | 1 | yes | `FastOperation` | `(*function_args, **function_kwargs)` |
| `ttnn.experimental.paged_update_cache` | 2 | yes | `FastOperation` | `(*function_args, **function_kwargs)` |
| `ttnn.experimental.reduce_scatter_minimal_async` | 6 | yes | `FastOperation` | `(*function_args, **function_kwargs)` |
| `ttnn.experimental.rotary_embedding_llama` | 4 | yes | `FastOperation` | `(*function_args, **function_kwargs)` |
| `ttnn.experimental.rotary_embedding_llama_fused_qk` | 1 | yes | `FastOperation` | `(*function_args, **function_kwargs)` |

The 13 experimental operations expose the generic wrapper signature
`(*function_args, **function_kwargs)`. On this CPython 3.12 wheel, the private
nanobind function exposes `(*args, **kwargs)`; that differs from the
non-inspectable result seen in the CPython 3.10 probe, but does not establish a
semantic difference.

## Commands

```bash
PY312=/tmp/gwang/uv-python/cpython-3.12.13-linux-x86_64-gnu/bin/python3.12
QUAL_DIR=/tmp/gwang/tttv2-py312.1MKAuO

"$PY312" -m venv "$QUAL_DIR/venv"
"$QUAL_DIR/venv/bin/python" -m pip install \
  --index-url https://download.pytorch.org/whl/cpu \
  torch==2.11.0
"$QUAL_DIR/venv/bin/python" -m pip install \
  --index-url https://pypi.org/simple \
  ttnn==0.77.0 loguru==0.6.0
"$QUAL_DIR/venv/bin/python" -m pip check

cd /tmp/gwang
env -u PYTHONPATH "$QUAL_DIR/venv/bin/python" \
  /localdev/gwang/tt_transformers/qualification/tools/probe_ttnn_symbols.py
```

The probe exited 0.

## Evidence boundary

This result establishes only that the named Python attributes exist in the
published wheel and that importing TTNN succeeds in the isolated host
environment. It does **not** establish:

- accepted arguments beyond generic wrapper introspection;
- output shapes, dtypes, layouts, memory configurations, or numerical behavior;
- trace, cache, sampling, or cleanup semantics;
- compatibility with extracted TTTv2 call sites;
- firmware, driver, Wormhole, Blackhole, mesh, SKU, or performance support.

No API was invoked with tensors, no TT device was opened, and no hardware or
remote machine was used. Semantic call probes and the ordered hardware matrix
remain separate qualification gates.
