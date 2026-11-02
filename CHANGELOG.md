# Changelog

All notable changes to this project will be documented here.

## 2.0.0.dev0

- `get_device_name` gives a Quasar device the Blackhole names (`P150`, `P300`,
  `P150x4`, `P150x8`, `BHGLX`), so code keyed on those names runs on Quasar.
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
  trace where that bucket is traced; the Llama-3.1-8B N300 trace region grows
  to 70 MB for it.
  The Qwen3-32B T3K trace region grows from 120 MB to 250 MB: its batch-32
  cases gain three prefix-cached 1024 traces and a penalised twin of every
  sampled warmup case, 23 traces in all, and need about 210 MB.
- Default KV-cache sizing leaves room for the decode page table. A paged KV
  cache narrower than the decode page table is refused at load with a
  `ValueError`, instead of failing on the first decode step.
- On-device greedy sampling breaks an exact logit tie by the lowest token id,
  as host argmax does. Before, the pick depended on the batch slot and could
  vary between runs.
- vLLM serving of Qwen2.5-72B (T3K) and Qwen2.5-7B (N300) prefills prompts in
  the 128-token bucket in fixed waves of four rows, padding a partial wave, and
  warms exactly that shape. Under many concurrent short prompts, time to first
  token falls by more than half against one prompt per pass; a lone short
  prompt waits 85–106 ms longer. Greedy output on these prompts is
  deterministic but is the four-row pass's text, so it differs from one-row
  prefill and from TTTv1: pinned greedy texts for these two models move once.

This is a developer preview. No model is promoted beyond the support status in
`SUPPORT.md` and its example manifest.
