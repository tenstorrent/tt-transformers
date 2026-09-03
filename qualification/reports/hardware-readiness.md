# Hardware qualification evidence

Status: **full matrix green — 42/42 passed**

The attributable release-candidate revision is
`73d414f8b826a7da982df8c8229d4ac41ed8ba33` on branch
`tttv2-standalone-migration`.

The canonical evidence index is
`qualification/evidence/hardware/73d414f8b826a7da982df8c8229d4ac41ed8ba33/index.json`.
Its SHA-256 is
`4e98b62c34fb9f8744e00624c091dc9de18b3f32c5cd74f2ad2be1ad26274d7a`.

## Final-SHA result

The 42-node matrix has 42 same-SHA execution records, and all 42 passed. The
host split is 23 records on `wh-lb-42` and 19 on `bh-qb-05`.

| Stage | Matrix nodes | Executed | Passed | Failed | Deferred |
|---|---:|---:|---:|---:|---:|
| Focused reusable modules | 30 | 30 | 30 | 0 | 0 |
| Runtime trace/order correctness | 1 | 1 | 1 | 0 | 0 |
| One-layer/model smoke | 3 | 3 | 3 | 0 | 0 |
| Token-accuracy/e2e | 8 | 8 | 8 | 0 | 0 |
| **Total** | **42** | **42** | **42** | **0** | **0** |

All stage groups passed: modules 30/30, runtime 1/1, smoke 3/3, and e2e 8/8.

| Architecture / mesh | Passed |
|---|---:|
| Wormhole / logical N150 on T3K | 9 |
| Wormhole / physical N300 pair on T3K | 6 |
| Wormhole / full T3K | 8 |
| Blackhole / logical single P150 on physical P150_X4 quietbox | 8 |
| Blackhole / full-host P150_X4 quietbox | 11 |

Across the 42 records there were no functional, hardware-lifecycle,
pre-device, or missing-acceptance classifications and no resets. Each record
reports `reset.performed=false` and `reset.automatic=false`.

## Strict W6 pass

Priority 17, `wh-t3k-runtime-trace-order`, executed all four capture/sampling
order cases on the full T3K and passed the unchanged strict logits, trace, KV,
replay, sampling, resume, chunk, and cache assertions. The accuracy profile
kept folded QKV/W2 prefill on `ttnn.linear`; the runner environment contains no
`DISABLE_MINIMAL_MATMUL` or W6 threshold override. Pytest reported four passes
in 490.54 seconds, and the runner classified the node `passed` with exit 0.
The evidence record is:

`qualification/evidence/hardware/73d414f8b826a7da982df8c8229d4ac41ed8ba33/wh-lb-42/20260903T124923.309445Z-wh-t3k-runtime-trace-order.json`

## Noncanonical accuracy-TTFT diagnostic

Four separate Llama-3.3-70B accuracy-profile performance cells ran at the
superseded candidate `2883a949860d749adc2ed1af5525b27a9a547505`, not at
the current candidate. Their compact summary is
`qualification/evidence/diagnostics/2883a949860d749adc2ed1af5525b27a9a547505/accuracy-ttft/summary.json`.

- TTFT passed 4/4, ranging from 86.7 to 87.2 ms against the
  tolerance-adjusted 105 ms ceiling.
- Throughput produced `performance_floor_failure` in 4/4 cells: host
  batch-32 7.9 tok/s/u, on-device-top-k batch-32 12.2 tok/s/u, host
  batch-32-ci 7.7 tok/s/u, and on-device-top-k batch-32-ci 11.9 tok/s/u.
- Every process and fixture teardown completed cleanly; there were zero
  hardware-lifecycle failures and zero resets.

These four cells are noncanonical performance diagnostics. They are excluded
from the current 42-record index and do not change its all-pass correctness
result. The throughput observation remains attributable only to
`2883a949860d749adc2ed1af5525b27a9a547505`.

The earlier relaxed one-order W6 run at
`d7677f822356e839f707a6447fd0abc89e620d56` remains non-qualifying history
associated with the superseded b24 strict failure. It is excluded from the
current canonical index and pass count and does not qualify the current W6
result.

## Logical single-P150 execution boundary

The following nodes all passed:

| Priority | Node | Stage |
|---:|---|---|
| 24 | `bh-p150-rmsnorm-decode` | module |
| 25 | `bh-p150-lm-head` | module |
| 26 | `bh-p150-mlp-decode` | module |
| 27 | `bh-p150-mlp-prefill` | module |
| 28 | `bh-p150-attention-prefill` | module |
| 29 | `bh-p150-attention-decode` | module |
| 30 | `bh-p150-attention-paged-transition` | module |
| 38 | `bh-p150-llama3-8b-token-accuracy` | e2e |

These are specifically the matrix's `MESH_DEVICE=P150` nodes. They ran as
logical 1x1 executions on the physical four-board P150_X4 `bh-qb-05`
quietbox. `MESH_DEVICE=P150` was the only topology-selection variable;
`TT_VISIBLE_DEVICES` was unset, so TTNN selected the board. They are valid
logical single-P150 regression results, but are not standalone-P150 product
evidence and are not full-host P150_X4 evidence.

## Physical evidence boundary

The Wormhole records were produced on `wh-lb-42` with an eight-device T3K
consisting of four physical N300 left/right pairs. Its inventory recorded a
1x8 system mesh, healthy DRAM on all eight devices, TT-KMD 2.4.1, firmware
bundle 18.12.1.0, and `TT_VISIBLE_DEVICES` unset.

`N150` rows are logical one-chip regression submeshes on that physical T3K;
they are not standalone-N150 product evidence. `N300` rows select a physical
left/right N300 board pair within the T3K.

The 19 Blackhole records were produced on `bh-qb-05`, a physical P150_X4
quietbox with four physical p150b boards and a 2x2 system mesh. Its final-SHA
inventory recorded the full-host P150_X4 topology and `TT_VISIBLE_DEVICES`
unset. Eight records selected a logical 1x1 P150 mesh; the other 11 exercised
the full P150_X4 mesh. None are physical-P300 evidence.

The two independent physical hosts may run one node each concurrently. The
records overlap in time across Wormhole and Blackhole while remaining
serialized within each host, consistent with the matrix's host-scoped lock.

## Evidence integrity and interpretation

The canonical index contains one entry per executed node with exact branch and
full SHA, physical inventory, machine, mesh, selector, stage, exit code,
classification, metric count, teardown status, reset state, and content hashes
for both the JSON record and stdout log. Its embedded matrix identity is:

- path: `qualification/manifests/hardware-matrix.json`
- schema version: 1
- node count: 42
- SHA-256: `1d04716dd79cf3c5ab6a7a2ac251224fe326ad64ef19fa440c06187118b9d7de`

Every executed record says `process_exited; fixture teardown not independently
hardware-verified`. This wording is retained verbatim: process exit is
recorded, but independent post-fixture hardware verification is not claimed.

`qualification/analysis/support/hardware_evidence.csv` lists all 42
attributable current-candidate records individually. Current same-SHA passing
evidence is eligible for the pinned baseline. Superseded-candidate rows are
ineligible.

The ledger also preserves the superseded exact-SHA candidates
`2883a949860d749adc2ed1af5525b27a9a547505`,
`b24eabe35c8f2c73f45493da40e5a6351eb0ec2d`, and
`ba7abefba4484689c953ac53fe8810322db1d184`, plus older revisions
`00748e6ac7b65f50e5c2af07f6e7c1c535c7f4c0` and
`b1a75d474ee44f583c32c5e6279c7026907553b9` as historical, ineligible context.
Their observations do not transfer to the final candidate SHA.

This report records qualification evidence only. It does not promote any
support manifest or change its `experimental` status. Six model families have
only partial subset evidence; for Llama-3.1-8B that evidence is limited to the
one logical-P150 token-accuracy gate. No whole-model or whole-geometry support
claim follows from this ledger.
