# Architecture

TT Transformers is a library of TTNN transformer building blocks rather than
an end-to-end serving framework.

## Layers

```text
HF generation and vLLM entry points
                │
                ▼
      model-owned executor policy
                │
                ▼
    model-neutral llm_runtime owners
                │
                ▼
 reusable modules and sampling utilities
                │
                ▼
               TTNN
```

- `tt_transformers.modules` owns reusable tensor operators and lazy weight or
  buffer materialization.
- `tt_transformers.sampling` owns model-neutral sampling parameters,
  penalties, log-probabilities, and seed management.
- `tt_transformers.llm_runtime` owns prefill/decode planning, program and trace
  compilation, KV-cache coordination, output ownership, warmup, and cleanup.
- `tt_transformers.models` owns checkpoint adaptation, model-specific tensor
  layout, executor policy, and generator facades. Each concrete model keeps
  its tensor implementation in `model.py`, its executor-backed HF interface
  in `hf_generator.py`, and its vLLM interface in `vllm_generator.py`.

Static choices such as topology and program configuration are resolved during
construction. Device resources have explicit owners and ordered, idempotent
cleanup paths. Production code never imports repository examples, tests, or
qualification helpers.

All model APIs are experimental until their support manifests say otherwise.

## Repository entry points

`examples/<model>/demo.py` is the application-facing example. It calls the
model's public `hf_generator.from_pretrained()`, applies the tokenizer's chat
template, calls `.generate()`, decodes the continuation, and cleans up the
model. It supplies an open mesh; the model owns executor construction, KV
storage, input staging, and the generation loop.

`examples/<model>/benchmark.py` contains the detailed accuracy, performance,
tracing, and DP workloads. These tools explicitly configure execution to
measure particular runtime paths. Existing hardware accuracy/performance
wrappers delegate to `benchmark.py`; the simple demo remains independent of
benchmark helpers and reference artifacts.
