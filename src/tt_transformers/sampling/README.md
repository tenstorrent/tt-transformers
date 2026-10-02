# Sampling values and helpers

`tt_transformers.sampling` carries the **shared value types** for sampling plus the
log-probability calculator. It is not the sampler.

The on-device sampler lives in [`tt_transformers.modules.sampling`](../modules/README.md):
`Sampling1D` (`sampling/sampling_1d.py`), `Penalties1D` (`sampling/penalties_1d.py`),
`SeedManager1D`, and the parameter helpers in `sampling/params.py`.

## What is here

| File | Purpose |
|---|---|
| `sampling_params.py` | `SamplingParams` — the one canonical dataclass (temp, top_k, top_p, penalties, seed, log-probs) |
| `tt_log_probs.py` | `LogProbsCalculator` / `LogProbsResult` — log-softmax across a sharded vocabulary (global max / sum-exp reduction across devices) |
| `logprobs.py` | Re-export shim for `LogProbsCalculator` |
| `vocab_padding.py` | Vocabulary padding geometry shared by the sampler and the LM head |
| `_utils.py` | `filter_none`, `split_list` |

```python
from tt_transformers.sampling import SamplingParams

params = SamplingParams(temperature=0.7, top_k=32, top_p=0.9)
```

`SamplingParams` has exactly one definition. vLLM has its own duck-type-compatible
`TTSamplingParams`; the two are interchangeable at the call sites that accept either.

To build, slice, broadcast or chunk parameters, import from
`tt_transformers.modules.sampling.params` — `prepare_sampling_params`,
`slice_sampling_params`, `format_sampling_params`, `place_prepared_sampling_params`.

## Removed: the TTTv1 sampler surface

`generator.py` (`SamplingGenerator`, `SeedManager`, and parameter helpers), `tt_sampling.py`
(`TTSampling`) and `tt_penalties.py` (`TTPenalties`) were **removed**. Nothing in this package
imported them, and their parameter helpers were a second implementation of what
`modules/sampling/params.py` already owns — including a second `SamplingParams` dataclass, which
is why this package now pins a single canonical identity.

tt-metal keeps and uses **its own copy** of those modules under `models/common/sampling/`, with its
own importers. Nothing there imported this package, so the removal does not affect it. The two
copies were already diverging.

If you are looking for the removed API, use `models.common.sampling` in tt-metal, or the v2
modules here.

## `data_parallel` vs `sampling_dp`

Different concepts, easily confused:

- **`data_parallel`** lives above this package: multiple TT model instances / submeshes processing
  different requests in parallel.
- **`sampling_dp`** lives inside the sampler: one model instance with multiple independent sampling
  groups, usually one per mesh row. Params, seeds and penalty state are flattened to
  `max_batch_size * sampling_dp`, then row-sharded onto the device.

## Pitfalls

Written against the removed TTTv1 sampler. They describe device behaviour rather than any one
implementation, so they are retained here as reference — but they have **not** been re-verified
against `modules/sampling`.

**`padded_vocab_size` vs `vocab_size`**: device offsets for global token IDs must use the *padded*
vocab size, to match how the LM head shards logits across devices. Using the unpadded `vocab_size`
shifts token IDs from devices 1+ and produces garbled output.

**Padded vocab logits**: if the LM head pads output weights beyond the real tokenizer vocabulary,
the sampler must mask those padded token IDs before force-argmax or local top-k. Zero-padded
LM-head weights give legal sharded matmul shapes; they are not a sampling mask.

**Batched prefill + on-device sampling**: only valid when the runtime prefill compute layout matches
the sampling-group layout. A model with `sampling_dp > 1` that does not expose a row-sharded
batched-prefill input contract must fall back to sequential prefill for correctness.

**Trace invalidation**: changing force-argmax state invalidates captured traces. Force-argmax is
triggered by k=1, p=1.0, temp=1.0 — note that p=1.0 means "no top-p filtering", distinct from the
internal initialization default of p=0.
