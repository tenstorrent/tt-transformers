# Architecture

TT Transformers is a library of TTNN transformer building blocks rather than
an end-to-end serving framework.

## Layers

```text
model adaptor and model-owned generator
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
  layout, executor policy, and generator facades.

Static choices such as topology and program configuration are resolved during
construction. Device resources have explicit owners and ordered, idempotent
cleanup paths. Production code never imports repository examples, tests, or
qualification helpers.

All model APIs are experimental until their support manifests say otherwise.
