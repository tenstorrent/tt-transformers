# Changelog

All notable changes to this project will be documented here.

## 2.0.0.dev0

- Establish `tt-transformers` as the standalone package for TTTv2 modules,
  runtime components, sampling, and twelve experimental model families.
- Add reproducible package builds and CPython 3.10/3.12 host validation.
- Add serialized Wormhole and Blackhole hardware qualification policy.
- Require `ttnn==0.79.0`. Trace capture now acknowledges its allocations to
  ttnn's trace allocation tracker (tenstorrent/tt-metal#53735).
- Rename the warmup override `trace_prefill_warmup_seq_lens` to
  `prefill_warmup_seq_lens`.
- Remove the TTTv1 sampler surface from `tt_transformers.sampling`
  (`generator`, `tt_sampling`, `tt_penalties`). Use
  `tt_transformers.modules.sampling`.
- With any trace mode, a prefill bucket that has no captured trace is served
  eager instead of failing. The first eager request per bucket logs one
  warning, and the load logs which buckets are traced.
- A traced configuration whose warmup lengths leave a servable prefill bucket
  uncompiled is refused at load with a `ValueError`.
- A `max_seq_len` that isn't a prefill bucket boundary now warms the bucket it
  pads to.
- Default KV-cache sizing leaves room for the decode page table. A paged KV
  cache narrower than the decode page table is refused at load with a
  `ValueError`, instead of failing on the first decode step.
- On-device greedy sampling breaks an exact logit tie by the lowest token id,
  as host argmax does. Before, the pick depended on the batch slot and could
  vary between runs.

This is a developer preview. No model is promoted beyond the support status in
`SUPPORT.md` and its example manifest.
