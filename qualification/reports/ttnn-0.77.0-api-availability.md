# TTNN 0.77.0 comprehensive host API availability

Status: **all current module-level TTNN paths are present; semantics and hardware remain unqualified**

Evidence:

- `qualification/tools/probe_all_ttnn_symbols.py`
- `qualification/reports/ttnn-0.77.0-api-availability.csv`
- `qualification/reports/ttnn-0.77.0-api-availability.json`

## Result

The current `src/` AST contains 160 unique maximal module-rooted `ttnn.*` paths across 4,393 import/attribute occurrences. Both published `ttnn==0.77.0` wheels expose all 160 paths without constructing a TTNN object or calling a TTNN function.

| Python | Public paths | Experimental paths | Private paths | Available | Missing |
|---|---:|---:|---:|---:|---:|
| 3.10.19 | 146 | 13 | 1 | 160 | 0 |
| 3.12.13 | 146 | 13 | 1 | 160 | 0 |

The inventory includes functions, operations, enums and enum members, tensor/layout/dtype constants, memory/program/config classes, mesh constructors, trace functions, and the bare module import—not just the previously tracked unstable APIs.

Both probes used `env -u PYTHONPATH` from `/tmp/gwang`. Their `sys.path` records contain the probe script directory, managed standard library, and isolated venv `site-packages`, but no `tt_transformers/src` or `tt-metal` checkout.

Source snapshot SHA256: `3814705a56bcf5bdd3ed88578e203d6d9e428f329289facb9bc2215366fb4e5f`.

## Previously tracked unstable APIs

All 14 paths from `qualification/analysis/ttnn_unstable_uses.csv` are included in the 160-path inventory and present on both interpreters:

- 13 `ttnn.experimental.*` operations;
- `ttnn._ttnn.tensor.dump_tensor_flatbuffer` (private).

This closes only their symbol-presence question. The private/experimental status and required device semantics are unchanged.

## Signature evidence

| Python | Informative signatures | Generic wrapper signatures | Not inspectable |
|---|---:|---:|---:|
| 3.10.19 | 15 | 72 | 73 |
| 3.12.13 | 15 | 103 | 42 |

Thirty-five paths have different introspection representations across interpreters. The differences are nanobind/Enum introspection shapes—for example unavailable versus `(*args, **kwargs)` signatures—not presence differences. No call was made, so none is evidence of a semantic incompatibility.

Under the migration-plan taxonomy, the 145 generic or unavailable signatures per interpreter remain **semantic or signature difference from the source environment: signature evidence incomplete**. They are not classified as missing APIs.

## Typed instance and dynamic-return members

`ttnn.*` module resolution cannot prove members accessed through runtime objects. The tool separately records 204 high-confidence sites inferred from explicit TTNN annotations, grouped into 36 receiver/member identities:

| Class-level result | Unique identities | Source occurrences | Interpretation |
|---|---:|---:|---|
| Descriptor available | 29 | 188 | Class surface exists; instance/device semantics not tested. |
| Complex/union annotation | 5 | 13 | Not safely attributable to one TTNN class without runtime dataflow. |
| Abstract descriptor absent | 2 | 3 | `DeviceComputeKernelConfig.fp32_dest_acc_en` and `.dst_full_sync_en`; resolved through official source evidence below. |

The complex group includes unions such as `ttnn.Tensor | LazyWeight`, `Optional[ttnn.MeshDevice]`, and `dict[str, ttnn.Tensor]`; it is evidence-bound rather than a missing-symbol classification.

## Official v0.77.0 source fallback

No module-level path was unresolved, so source fallback was required only for the two abstract descriptor cases.

- Local official tag: `v0.77.0`
- Commit: `9f9cd4fd590f4b606bd0981a4fe0b6403eb38ec9`
- `ttnn/cpp/ttnn-nanobind/operations/core.cpp` blob `4c9fb6ce8379e36309b0d6d95ba919a58f3c05c9`, lines 47–117
- `ttnn/cpp/ttnn/operations/core/compute_kernel/compute_kernel_config.hpp` blob `56839a6ec135fa30f1f1fdf2729ba1a882c5ff8d`, lines 14–34

The binding intentionally exposes `DeviceComputeKernelConfig` as an abstract placeholder and documents `WormholeComputeKernelConfig` as the concrete instantiable class. The concrete binding has read/write `fp32_dest_acc_en` and `dst_full_sync_en` fields, and the C++ type aliases all architecture names to `ComputeKernelConfig`.

Therefore the two absent abstract descriptors are classified as **semantic or signature difference from the source environment (abstract annotation versus concrete runtime value)**, not missing public APIs. Actual returned-instance behavior still requires non-host qualification.

## Migration taxonomy

| Classification | Count/verdict |
|---|---|
| Missing public TTNN API | 0 module paths |
| Missing experimental/private TTNN API | 0 module paths |
| Semantic or signature difference from source environment | 145 incomplete signatures per interpreter; 35 cross-interpreter introspection differences; 2 abstract descriptor interpretations |
| TTTv2 packaging defect | 0 observed by this probe |
| TTTv2 implementation defect | 0 established by this probe |
| Hardware/firmware/driver mismatch | not evaluated |
| Unsupported model geometry | not evaluated |

## Reproduction

```bash
PY310=/tmp/gwang/tttv2-deps-py310.0JvRwC/venv/bin/python
PY312=/tmp/gwang/tttv2-deps-py312.yi8BCf/venv/bin/python
PROBE=/localdev/gwang/tt_transformers/qualification/tools/probe_all_ttnn_symbols.py

cd /tmp/gwang
env -u PYTHONPATH "$PY310" "$PROBE" --compact > /tmp/ttnn-api-py310.json
env -u PYTHONPATH "$PY312" "$PROBE" --compact > /tmp/ttnn-api-py312.json
env -u PYTHONPATH "$PY310" "$PROBE" --compact --format csv > /tmp/ttnn-api-py310.csv
env -u PYTHONPATH "$PY312" "$PROBE" --compact --format csv > /tmp/ttnn-api-py312.csv
```

Omit `--compact` to emit the full per-site JSON inventory for all 4,393
import/attribute occurrences. Compact mode retains every unique path, count,
context summary, probe result, and typed-instance aggregate used by the
checked-in artifacts.

The checked-in JSON combines the two compact JSON reports, cross-interpreter signature comparison, unstable-API cross-reference, taxonomy summary, and official-source fallback. The checked-in CSV is the header plus all rows from the 3.10 CSV and the data rows from the 3.12 CSV.

## Concurrent provenance correction

A host test also proved one missing source asset dependency: `models/tt_transformers/model_params/Qwen3-32B/config.json` (blob `12ea4a36c6ac093af8d8dbc3bd435ae8b67067d6`) is now assigned to `qualification/model_params/Qwen3-32B/config.json`. The full pinned common-test audit found no other `model_params` path reference. This is qualification data; production runtime does not load it. The support extraction lane was notified.

The corrected provenance inventory contains 366 rows with the original 26
MoE-only exclusions unchanged.

## Evidence boundary

This is AST inventory, host attribute lookup, class-descriptor lookup, and official source/binding evidence. It does not establish accepted arguments, constructor behavior, tensor results, dtype/layout/memory semantics, trace/cache/sampling behavior, model correctness, performance, firmware/driver compatibility, or hardware/SKU support.

No TTNN callable or constructor was invoked, no device query was made, and no hardware or remote system was used. No `src`, test, example, `pyproject.toml`, main compatibility report, or main work-log file was edited.
