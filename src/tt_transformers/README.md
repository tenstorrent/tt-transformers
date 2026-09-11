# TT Transformers package

`tt_transformers` provides reusable transformer operators, shared inference
runtime components, and concrete model implementations on TTNN. Applications
can compose individual modules or use a model's executor and generation
interface. See the [software architecture](../../docs/architecture.md) for the
layering and resource ownership principles.

## Architecture and package layout

| Package | Responsibility | Use it when |
|---|---|---|
| [`modules`](modules/README.md) | Attention, embedding, LM-head, MLP, RMSNorm, RoPE, device sampling, lazy tensors, and collectives. | Building or tuning reusable tensor operations. |
| [`sampling`](sampling/README.md) | Model-neutral sampling parameters, penalties, log-probabilities, generation, and seed utilities. | Sharing sampling behavior across models. |
| [`llm_runtime`](llm_runtime/README.md) | Prefill/decode planning, KV-cache coordination, program and trace compilation, output ownership, warmup, and cleanup. | Managing execution and resources across inference calls. |
| [`models`](models/README.md) | Checkpoint adaptation, model-specific tensor layouts, executor policy, and generator interfaces. | Composing the shared layers for a particular model. |

Model implementations compose reusable modules to express tensor computation.
Their executors combine model-specific policy with the shared runtime's
resource owners. Both modules and runtime components use TTNN for device
tensors and execution. Sampling utilities supply shared behavior to these
paths. Model-specific policy stays in `models` so the lower layers can be
reused by other models.

For example, the [Llama 3.1 8B implementation](models/llama3_8b/) separates
checkpoint loading and configuration in `hf_adaptor.py` from tensor operations
in `model.py`. Its `executor.py` constructs the execution target using shared
Llama-family policy and runtime owners. Its `generator.py` exposes generation
calls and delegates prefill, decode, and cleanup to that target. See the
[model guide](models/README.md#model-package-layout) for the common file layout
and the [prefill guide](llm_runtime/prefill/README.md) for execution planning.

Static choices such as topology and program configuration are resolved during
construction. Device, tensor, trace, and cache resources have explicit owners;
cleanup must be ordered, retryable, and safe after partial initialization.

## Release compatibility

Every tt-transformers release uses a publicly available TTNN package, with an
exact version pinned in that release's [`pyproject.toml`](../../pyproject.toml).
The current checkout pins `ttnn==0.77.0`. Installing the released package
resolves this dependency through the normal Python package installation flow;
a tt-metal source checkout or a private TTNN build is not required.

Use the dependency metadata and
[hash-locked environments](../../constraints/README.md) from the same
tt-transformers release or tag. Consult the [support matrix](../../SUPPORT.md)
and model manifests for the validated hardware and workloads; a TTNN version
pin alone does not qualify every model configuration.

Developers can also test a selected tt-metal commit or edit TTNN alongside
tt_transformers using the [TTNN development guide](../../docs/ttnn-development.md).
These custom runtimes use separate development environments. A development
build does not change the release dependency pin or establish release support
for that TTNN revision.

## Package boundaries

Only symbols intentionally exported from package `__init__.py` files should be
treated as convenient import surfaces. Model APIs remain experimental until
their support manifests are promoted. The [model lifecycle policy](models/README.md#model-lifecycle)
explains how deprecated implementations will be preserved under `legacy/`.

Repository [examples](../../examples/README.md), [tests](../../tests/README.md),
and [qualification tooling](../../qualification/README.md) are not installed by
the wheel and must never be imported by production package code.
