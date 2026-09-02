# Isolated TTNN 0.77 host validation

Validation date: 2026-09-02 (UTC)

This report records host-only validation of the extracted standalone support and test surface. No TT device, remote machine, server, or hardware test was used.

## Environments and commands

| Python | Interpreter | Direct host-test tuple |
|---|---|---|
| 3.10.19 | `/tmp/gwang/ttnn-077-symbol-probe.rGt2zl/venv/bin/python` | TTNN 0.77.0, Torch 2.11.0+cpu, Loguru 0.6.0, Transformers 5.12.1, pytest 9.0.3, jsonschema 4.26.0, pytz 2026.3.post1 |
| 3.12.13 | `/tmp/gwang/tttv2-deps-py312.yi8BCf/venv/bin/python` | TTNN 0.77.0, Torch 2.11.0+cpu, Loguru 0.6.0, Transformers 5.12.1, pytest 9.0.3, jsonschema 4.26.0, pytz 2026.3.post1 |

The final audited wheel
`dist/tt_transformers-0.1.0.dev0-py3-none-any.whl` was installed non-editably
in both environments. The repository remained the test/support source, but no
`PYTHONPATH` or editable package path was used. Exact final commands:

```bash
TT_TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /tmp/gwang/ttnn-077-symbol-probe.rGt2zl/venv/bin/python -m pytest -m host -q
TT_TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /tmp/gwang/tttv2-deps-py312.yi8BCf/venv/bin/python -m pytest -m host -q
```

## Result

Both final commands completed successfully with the same exact test totals:

| Python | Passed | Skipped | Deselected | Passing subtests | Failures/errors | Process exit |
|---|---:|---:|---:|---:|---:|---:|
| 3.10.19 | 2,153 | 28 | 6,791 | 81 | 0 | 0 |
| 3.12.13 | 2,153 | 28 | 6,791 | 81 | 0 | 0 |

The previously recorded five pytest warnings are empty-regex warnings emitted by `pytest.raises` in `tests/llm_runtime/test_config.py`; they do not indicate collection or execution failure.

### Python 3.12 TTNN binding feedback

After pytest had reported its successful result and exit status, CPython 3.12 emitted:

```text
nanobind: leaked 8 instances!
nanobind: leaked 36 types!
nanobind: leaked 330 functions!
nanobind: this is likely caused by a reference counting issue in the binding code.
```

The complete retained source-layout reproduction is
`/tmp/gwang/tttv2-py312-host-final.log`; the same diagnostic class and counts
were reproduced after non-editable wheel installation. The diagnostics occur
during interpreter teardown and do not change pytest's exit code 0. They are not counted as a test failure or collection error.

This is recorded as **TTNN 0.77 Python 3.12 binding-lifecycle feedback**, not hidden output and not a TTTv2 correctness pass for binding cleanup. The retained actionable gap is to reproduce the shutdown diagnostic in a minimal TTNN import/config-object process and resolve the reference ownership in TTNN/nanobind.

The 28 skips are accounted for as follows:

- 12 hardware demo wrappers fail closed at module collection because `MESH_DEVICE` is unset. These tests are deselected by `-m host`; their module-level environment checks still execute during collection. A skip is not hardware evidence.
- 1 W6 correctness test requires `MESH_DEVICE=T3K`.
- 9 parametrized checks exercise removed TTTv1 `from_model_args`/Galaxy topology compatibility entry points and are explicitly retired by the compatibility ledger: Attention1D (1), LMHead1D (1), MLP2D (4), RMSNorm1D (1), RMSNorm2D (1), and Rope1D (1).
- 4 seed-stream checks exercise excluded TTTv1 `Generator` state, not the standalone `SamplingGenerator`/`SeedManager` ownership boundary.
- 2 capability checks address legacy `generator_vllm.py` files which were not extracted into the standalone repository.

The MLP2D 8x4 topology-bug probe is explicitly classified as `device` +
`wormhole`; it is deselected by the host command and therefore does not probe
PCI devices. The final taxonomy audit reports 1,452 test functions, 1,198 host,
254 device, and no errors.

## Defects closed

- Pytest taxonomy insertion now preserves module docstrings and every `from __future__` import before adding marker imports.
- Pytest uses importlib import mode, preventing collisions among model tests with identical basenames.
- The vLLM matrix test resolves validators from `qualification/tools`; readiness server collection treats `requests` and `openai` as optional imports and produces runtime dependency errors only when their functions are invoked.
- Qwen smoke examples import their public greedy helpers from `examples/common/greedy.py`.
- Model-executor, prefill, sampling, qualification-schema, manifest, trace-size, model-target, and seed-manager paths now resolve against the standalone layout.
- Eleven generator import paths and the remaining model-profile monkeypatch paths use `tt_transformers.*`.
- The pinned Qwen3 configuration asset (`git blob 12ea4a36c6ac093af8d8dbc3bd435ae8b67067d6`) is retained at `qualification/model_params/Qwen3-32B/config.json`; production code does not load it.
- Ten demo-contract modules now validate the public example plus hardware-wrapper pair. Runnable examples remain pytest-free, while wrapper decorators, fixtures, parameters, and skip conversion remain under `tests/hardware`.
- Three Blackhole capability manifests point at their standalone hardware wrappers and carry regenerated hashes for those exact wrapper files.
- Obsolete TTTv1 generator/factory characterization is explicitly skipped with a reason instead of inventing compatibility APIs in the standalone package.

## Progression and triage

The first exact isolated run stopped during collection with 52 errors, 2 skips, and 6756 deselections. After the collection boundary fixes, the first run to reach execution reported 1914 passed, 113 failed, 14 skipped, 6790 deselected, and 30 collection errors. The remaining failures were separated into support migration defects and implementation/TTNN candidates. Focused reruns established:

- qualification/runtime/legacy-generator support set: 591 passed, 4 intentional legacy skips, 21 deselected, 81 subtests passed;
- all split-demo contract modules: 192 passed;
- final stale path/demo/legacy subset: 288 passed, 2 intentional legacy skips, 3 deselected;
- exact full host suites on both supported interpreters: clean, as recorded above.

Because both exact final runs have no failure or error, these host environments leave no production or TTNN test failure to reassign. The Python 3.12 shutdown diagnostic remains explicit binding feedback. These results are host/static evidence only and must not be treated as model correctness, performance, architecture, SKU, mesh, TP, DP, or hardware qualification evidence.
