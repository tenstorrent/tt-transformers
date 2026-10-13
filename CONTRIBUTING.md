# Contributing

Bugs are reported via [GitHub Issues](https://github.com/tenstorrent/tt-transformers/issues).
Bug fixes and new functionality are submitted via GitHub Pull Requests, which
are reviewed on a weekly cadence. By participating in this project, you agree
to abide by our [Code of Conduct](CODE_OF_CONDUCT.md).

This guide covers the development setup, the architecture every change must
fit, the steps for adding a model, and the checks to run before review. Each
section links to the detailed guide for its area.

## Set up a development environment

Use CPython 3.10 or 3.12 on Linux. For a normal editable checkout:

```bash
python -m pip install -e '.[examples,test,dev]'
pre-commit install
```

Maintainers and CI should prefer the hash-locked environments documented in
`constraints/README.md`.

To develop against a tt-metal checkout or a TTNN wheel built from a particular
commit, follow [Custom TTNN development](docs/ttnn-development.md). It provides
isolated source-linked and fixed-wheel environments while preserving release pins.

## Architecture

TT Transformers is a library of TTNN transformer building blocks, not an
end-to-end serving framework. Code is organized in layers. Each layer may use
the layers below it, never the layers above.

```text
HF and vLLM entry points         models/<model>/hf_generator.py, vllm_generator.py
                │
                ▼
model-owned executor policy      models/executor.py, family facades, model executors
                │
                ▼
model-neutral runtime owners     llm_runtime
                │
                ▼
reusable modules and sampling    modules, sampling
                │
                ▼
               TTNN
```

| Package | Owns | Read |
|---|---|---|
| `modules` | Attention, MLP, RMSNorm, RoPE, embedding, LM head, device sampling, lazy weights and buffers, collectives. | [Modules guide](src/tt_transformers/modules/README.md) |
| `sampling` | Model-neutral sampling parameters, penalties, log-probabilities, and seeds. | [Sampling guide](src/tt_transformers/sampling/README.md) |
| `llm_runtime` | Prefill/decode planning, program and trace compilation, KV-cache ownership, warmup, output reads, data-parallel lanes, and cleanup. | [Runtime guide](src/tt_transformers/llm_runtime/README.md) |
| `models` | Checkpoint adaptation, model tensor layout, executor policy, and the HF and vLLM entry points. | [Model guide](src/tt_transformers/models/README.md) |

The [architecture overview](docs/architecture.md) and the
[package guide](src/tt_transformers/README.md) describe the layering in more
detail.

### Design rules

Reviews hold every change to these rules. The linked guides explain each one.

- **Library, not framework.** Callers control the execution flow. Modules are
  building blocks, not an orchestration layer.
- **Resolve static choices at construction.** Topology, program configuration,
  trace mode, and other static policy become frozen config values when an
  object is built. `forward()` is a straight line of compute with no if-else on
  static configuration.
- **Keep policy at its own layer.** Model-specific decisions stay in `models`.
  Do not add a generic aggregate executor to `llm_runtime`; share composition in
  `models/executor.py` only when several models demonstrably share the
  lifecycle.
- **Give every device resource one owner.** Tensors, traces, programs, and the
  KV cache have explicit owners. Cleanup is ordered, safe after partial
  initialization, and safe to call twice.
- **Fail instead of degrading silently.** A selected trace that has no coverage
  raises; it does not fall back to eager execution. Unsupported options raise
  errors instead of being ignored.
- **Load weights lazily.** Wrap weights in `LazyWeight` and mutable state in
  `LazyBuffer` so device allocation happens on first use.
- **Prefer unit tests.** Fast module and runtime tests come before end-to-end
  tests.

## Repository boundaries

- Production code may import only `tt_transformers`, declared third-party
  dependencies, and the Python standard library.
- Production code must not import repository `tests`, `examples`, or
  `qualification` modules.
- Examples are repository-only and may use `examples.common` plus documented
  qualification helpers.
- Test-only utilities belong under `tests/support`.
- New model support must update its `support.json`, documentation, and tests
  without claiming unexecuted hardware coverage.

## Add or change a model

A model is a package under `src/tt_transformers/models/<model>/` plus a
repository example under `examples/<model>/`. Use
[`llama3_8b`](src/tt_transformers/models/llama3_8b/) as the reference, and
follow its file names, class names, and function names so every model can be
read the same way.

### 1. Model package

| File | Responsibility |
|---|---|
| `model.py` | Compose `modules` into the model's TTNN tensor computation and layouts. |
| `hf_generator.py` | Load the checkpoint and tokenizer, build one executor, and expose `from_pretrained()` and `.generate()`. |
| `vllm_generator.py` | Implement the [vLLM generator interface](docs/vllm-interface.md) and delegate execution to the executor. |
| `weight_utils.py` | Convert model-specific weights, when a separate helper is needed. |
| `__init__.py` | Declare the package's intended import surface. |

Name the vLLM generator class `<Family>Generator`, its construction config
`<Family>GeneratorConfig`, and its builder `build_<model>_generator`, as
`llama3_8b`, `phi4`, and the Qwen packages do.

### 2. Executor

Pick the smallest executor that fits:

1. the shared [`ModelExecutor`](src/tt_transformers/models/executor.py);
2. an existing family facade, such as
   [`llama3_executor.py`](src/tt_transformers/models/llama3_executor.py) or
   [`qwen2_executor.py`](src/tt_transformers/models/qwen2_executor.py); or
3. a model-owned executor composed directly from `llm_runtime` modules, when
   the model's orchestration differs. See the
   [Phi-4 example](examples/phi4/README.md).

The executor owns device tensors, traces, and the KV cache. The generators stay
thin: they adapt arguments, choose eager or traced execution, and delegate.
The [runtime guide](src/tt_transformers/llm_runtime/README.md#reusing-the-toolkit-for-another-model)
lists what a tensor model must provide to the runtime.

### 3. vLLM generator

The [vLLM generator interface](docs/vllm-interface.md) is the contract between
this repository and the [TT vLLM plugin](https://github.com/tenstorrent/vllm-tt-plugin).
It defines the methods the plugin calls, their order, their arguments and
return values, the `model_capabilities` flags, and how to register a generator
with the plugin. A generator that implements it can be served without plugin
changes.

Declare a `model_capabilities` flag only when the model supports that behavior
on every geometry its support manifest declares.

### 4. Example, manifest, and tests

- `examples/<model>/README.md` contains the support-contract headings that
  `qualification/tools/validate_support_docs.py` checks.
- `examples/<model>/support.json` records the checkpoint revision, software
  tuple, declared hardware geometry, and known limits.
- `examples/<model>/demo.py` follows the public flow: `from_pretrained()`,
  `apply_chat_template()`, `.generate()`, decode, and `cleanup()`.
- `examples/<model>/benchmark.py` holds the accuracy and performance
  workloads.
- `tests/models/<model>/` holds host tests, and `tests/hardware/models/`
  holds the hardware tests. See the [test guide](tests/README.md).
- Add the model to the table in the [model guide](src/tt_transformers/models/README.md#model-directory)
  and to [SUPPORT.md](SUPPORT.md).

## Changes that span the vLLM plugin

The plugin reads `model_capabilities` from the generator class and gates
behavior on those declarations. When a change needs a capability that the
plugin does not read yet, or a plugin change starts reading one that no model
declares, the two changes land as a pair:

- open the plugin change in [tenstorrent/vllm-tt-plugin](https://github.com/tenstorrent/vllm-tt-plugin)
  and update its
  [`MODEL_CAPABILITIES.md`](https://github.com/tenstorrent/vllm-tt-plugin/blob/main/docs/MODEL_CAPABILITIES.md);
- declare the capability in the affected `vllm_generator.py` files here; and
- link each pull request from the other.

## Before requesting review

```bash
pre-commit run --all-files
python -m pytest -m host
python tools/check_import_boundaries.py
python qualification/tools/validate_support_docs.py
python qualification/tools/run_hardware_matrix.py --validate
python tools/validate_lockfiles.py
python -m build
python -m twine check --strict dist/*
```

Hardware tests must identify the exact Git SHA, machine and physical SKU,
logical mesh, environment, selector, exit status, metrics, and teardown state.
One TT process may run per physical host. A skipped hardware test is not
evidence of support.

## Pull requests

Keep commits focused and explain user-visible behavior, validation performed,
and any remaining limitation. Never commit model checkpoints, caches,
credentials, machine-local logs, or migration work logs.
