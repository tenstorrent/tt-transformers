# Package map

`tt_transformers` is organized into four primary layers:

- [`modules`](modules/README.md): reusable attention, embedding, LM-head, MLP,
  RMSNorm, RoPE, sampling, lazy tensor, and collective building blocks.
- [`sampling`](sampling/README.md): model-neutral sampling parameters,
  penalties, log-probability, generation, and seed utilities.
- [`llm_runtime`](llm_runtime/README.md): prefill/decode execution, KV-cache,
  program/trace compilation, output ownership, warmup, and cleanup.
- `models`: twelve model packages plus shared Llama- and Qwen-family executor
  policy.

The prefill subsystem has its own
[`llm_runtime/prefill` guide](llm_runtime/prefill/README.md).

Model packages normally contain:

```text
hf_adaptor.py   checkpoint/tokenizer adaptation
model.py        model-specific TTNN tensor model
executor.py     composition with shared runtime owners
generator.py    direct/vLLM-facing generation facade
weight_utils.py model-specific weight conversion
```

Only symbols intentionally exported from package `__init__.py` files should be
treated as convenient import surfaces. Model APIs remain experimental and may
change until their support manifests are promoted.

Device, tensor, trace, and cache resources must have explicit owners. Cleanup
must be ordered, retryable, and safe when initialization is partial.

Repository examples, tests, and qualification tooling are not installed by the
wheel and must never be imported by production package code.
