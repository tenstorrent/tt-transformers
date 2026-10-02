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
- Warmup compiles the prefix-cached variant of every prefill bucket a cached
  prompt can reach, including the top one. A configuration whose `max_seq_len`
  is itself the top bucket gains one such case per sampling path, captured as a
  trace where that bucket is traced; the Llama-3.1-8B trace region grows to
  70 MB on N300 and to 100 MB on P300, P150x4 and P300x2 for it.
- The Qwen3-32B trace region on P150x4 and P300x2 grows to 200 MB. Its traced
  batch demos capture about 171 MB of traces there.
- The Llama-3.3-70B trace region on P150x4 and P300x2 grows to 400 MB. Its
  traced batch-32 demos capture about 304 MB of traces there.
- Default KV-cache sizing leaves room for the decode page table. A paged KV
  cache narrower than the decode page table is refused at load with a
  `ValueError`, instead of failing on the first decode step.
- A cached tensor larger than 32 MiB is copied into process memory on the
  host before it is uploaded. Uploading it straight from the cache file could
  stall indefinitely with ttnn 0.79.0 on Blackhole with the IOMMU enabled.
- On-device greedy sampling breaks an exact logit tie by the lowest token id,
  as host argmax does. Before, the pick depended on the batch slot and could
  vary between runs.

This is a developer preview. No model is promoted beyond the support status in
`SUPPORT.md` and its example manifest.
