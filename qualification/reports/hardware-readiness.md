# Hardware qualification evidence

Status: **all accessible nodes green — 34 passed and 8 topology-specific nodes deferred**

The attributable release-candidate revision is
`2883a949860d749adc2ed1af5525b27a9a547505` on branch
`tttv2-standalone-migration`.

The canonical evidence index is
`qualification/evidence/hardware/2883a949860d749adc2ed1af5525b27a9a547505/index.json`.
Its SHA-256 is
`ec9a540a8f31762078b592909e02cdb03e99384f9ddf8f955966fa41e42af07b`.

## Final-SHA result

The 42-node matrix has 34 same-SHA execution records, and all 34 passed. Eight
single-P150 nodes were not run because their required `bh-lb-11` physical host
role was unavailable.

| Stage | Matrix nodes | Executed | Passed | Failed | Deferred |
|---|---:|---:|---:|---:|---:|
| Focused reusable modules | 30 | 23 | 23 | 0 | 7 |
| Runtime trace/order correctness | 1 | 1 | 1 | 0 | 0 |
| One-layer/model smoke | 3 | 3 | 3 | 0 | 0 |
| Token-accuracy/e2e | 8 | 7 | 7 | 0 | 1 |
| **Total** | **42** | **34** | **34** | **0** | **8** |

All executed stage groups passed: modules 23/23, runtime 1/1, smoke 3/3, and
e2e 7/7.

| Architecture / mesh | Executed result | Deferred |
|---|---:|---:|
| Wormhole / logical N150 on T3K | 9 passed | 0 |
| Wormhole / physical N300 pair on T3K | 6 passed | 0 |
| Wormhole / full T3K | 8 passed | 0 |
| Blackhole / single P150 on development loudbox | 0 | 8 |
| Blackhole / physical P150_X4 quietbox | 11 passed | 0 |

Across the 34 records there were no functional, hardware-lifecycle,
pre-device, or missing-acceptance classifications and no resets. Each record
reports `reset.performed=false` and `reset.automatic=false`.

## Strict W6 pass

Priority 17, `wh-t3k-runtime-trace-order`, executed all four capture/sampling
order cases on the full T3K and passed the unchanged strict logits, trace, KV,
replay, sampling, resume, chunk, and cache assertions. The accuracy profile
kept folded QKV/W2 prefill on `ttnn.linear`; the runner environment contains no
`DISABLE_MINIMAL_MATMUL` or W6 threshold override. Pytest reported four passes
in 343.05 seconds, and the runner classified the node `passed` with exit 0.
The evidence record is:

`qualification/evidence/hardware/2883a949860d749adc2ed1af5525b27a9a547505/wh-lb-42/20260903T030639.828366Z-wh-t3k-runtime-trace-order.json`

## Noncanonical accuracy-TTFT diagnostic

Four separate Llama-3.3-70B accuracy-profile performance cells ran after the
canonical sweep. Their compact summary is
`qualification/evidence/diagnostics/2883a949860d749adc2ed1af5525b27a9a547505/accuracy-ttft/summary.json`.

- TTFT passed 4/4, ranging from 86.7 to 87.2 ms against the
  tolerance-adjusted 105 ms ceiling.
- Throughput produced `performance_floor_failure` in 4/4 cells: host
  batch-32 7.9 tok/s/u, on-device-top-k batch-32 12.2 tok/s/u, host
  batch-32-ci 7.7 tok/s/u, and on-device-top-k batch-32-ci 11.9 tok/s/u.
- Every process and fixture teardown completed cleanly; there were zero
  hardware-lifecycle failures and zero resets.

These four cells are noncanonical performance diagnostics. They are excluded
from the 34-record index and do not change its all-pass correctness result.

The earlier relaxed one-order W6 run at
`d7677f822356e839f707a6447fd0abc89e620d56` remains non-qualifying history
associated with the superseded b24 strict failure. It is excluded from the
current canonical index and pass count and does not qualify the current W6
result.

## Deferred single-P150 scope

The following nodes were not executed:

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

These are specifically the matrix's `MESH_DEVICE=P150` nodes. They require a
single P150 die selected on the eight-P150 development loudbox and list only
`bh-lb-11` in their machine pool. The available `bh-qb-05` machine is a
physical four-board P150_X4 quietbox; its topology and provenance cannot be
reinterpreted as the required development-loudbox P150 scope. The eight absent
records are `deferred_not_run`, neither passes nor failures.

## Physical evidence boundary

The Wormhole records were produced on `wh-lb-42` with an eight-device T3K
consisting of four physical N300 left/right pairs. Its inventory recorded a
1x8 system mesh, healthy DRAM on all eight devices, TT-KMD 2.4.1, firmware
bundle 18.12.1.0, and `TT_VISIBLE_DEVICES` unset.

`N150` rows are logical one-chip regression submeshes on that physical T3K;
they are not standalone-N150 product evidence. `N300` rows select a physical
left/right N300 board pair within the T3K.

The Blackhole records were produced on `bh-qb-05`, a physical P150_X4
quietbox with four physical p150b boards and a 2x2 system mesh. Its final-SHA
inventory recorded the full-host P150_X4 topology and
`TT_VISIBLE_DEVICES` unset; these records are not physical-P300 evidence.

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
- SHA-256: `e5e54f1a216164c2036913313d0775454357ce5c6a82dc09c61caaba3c890db4`

Every executed record says `process_exited; fixture teardown not independently
hardware-verified`. This wording is retained verbatim: process exit is
recorded, but independent post-fixture hardware verification is not claimed.

`qualification/analysis/support/hardware_evidence.csv` lists all 34 attributable
current-candidate records and all eight deferred nodes individually. Current
same-SHA passing evidence is eligible for the pinned baseline. Deferred and
superseded-candidate rows are ineligible.

The ledger also preserves the superseded exact-SHA candidates
`b24eabe35c8f2c73f45493da40e5a6351eb0ec2d` and
`ba7abefba4484689c953ac53fe8810322db1d184`, plus older revisions
`00748e6ac7b65f50e5c2af07f6e7c1c535c7f4c0` and
`b1a75d474ee44f583c32c5e6279c7026907553b9` as historical, ineligible context.
Their observations do not transfer to the final candidate SHA.

This report records qualification evidence only. It does not promote any
support manifest, change its `experimental` status, or convert deferred scope
into a support claim.
