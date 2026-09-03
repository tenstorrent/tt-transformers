# Hardware qualification evidence

Status: **not fully green — 33 passed, 1 functional blocker, and 8 topology-specific nodes deferred**

The attributable release-candidate revision is
`b24eabe35c8f2c73f45493da40e5a6351eb0ec2d` on branch
`tttv2-standalone-migration`.

The canonical evidence index is
`qualification/evidence/hardware/b24eabe35c8f2c73f45493da40e5a6351eb0ec2d/index.json`.
Its SHA-256 is
`8bbf3e6340d4015753a24d5107ae4183f9de57c931ed6d4eb111af769e9c040c`.

## Final-SHA result

The 42-node matrix has 34 same-SHA execution records. Of those, 33 passed and
one is a Wormhole W6 functional failure. Eight single-P150 nodes were not run
because their required `bh-lb-11` physical host role was unavailable.

| Stage | Matrix nodes | Executed | Passed | Functional failure | Deferred |
|---|---:|---:|---:|---:|---:|
| Focused reusable modules | 30 | 23 | 23 | 0 | 7 |
| Runtime trace/order correctness | 1 | 1 | 0 | 1 | 0 |
| One-layer/model smoke | 3 | 3 | 3 | 0 | 0 |
| Token-accuracy/e2e | 8 | 7 | 7 | 0 | 1 |
| **Total** | **42** | **34** | **33** | **1** | **8** |

Thus, all executed reusable-module, smoke, and e2e records passed: modules
23/23, smoke 3/3, and e2e 7/7. The runtime gate is 0/1.

| Architecture / mesh | Executed result | Deferred |
|---|---:|---:|
| Wormhole / logical N150 on T3K | 9 passed | 0 |
| Wormhole / physical N300 pair on T3K | 6 passed | 0 |
| Wormhole / full T3K | 7 passed, 1 functional failure | 0 |
| Blackhole / single P150 on development loudbox | 0 | 8 |
| Blackhole / physical P150_X4 quietbox | 11 passed | 0 |

Across the 34 records there were no hardware-lifecycle failures, pre-device
failures, missing-acceptance classifications, or resets. Each record reports
`reset.performed=false` and `reset.automatic=false`.

## W6 functional blocker

Priority 17, `wh-t3k-runtime-trace-order`, executed all four capture/sampling
order cases on the full T3K and failed strict logits parity in each case. The
recorded row-0 maximum absolute difference was 1.5 against a 1.0 limit, and
top-5 overlap was 3 against a minimum of 4.

The runner classified this as `functional_failure`, not
`hardware_lifecycle_failure`. The process exited and no reset was performed.
Its evidence record is:

`qualification/evidence/hardware/b24eabe35c8f2c73f45493da40e5a6351eb0ec2d/wh-lb-42/20260903T001610.842135Z-wh-t3k-runtime-trace-order.json`

This blocker does not invalidate the separately passing module, smoke, or e2e
records, but it prevents a fully green hardware-qualification verdict.

### Non-qualifying relaxed diagnostic

One order was also run at diagnostic candidate
`d7677f822356e839f707a6447fd0abc89e620d56` with only the known
cross-geometry logits oracle relaxed. It passed the later trace, KV, replay,
sampling, resume, chunk, and cache invariants in 176.12 seconds. This result is
diagnostic context only: it is excluded from the canonical index and pass count,
does not qualify W6, and does not supersede or weaken the official
strict functional failure above.

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
same-SHA functional evidence is eligible for the pinned baseline even when it
records a blocker; eligibility means attributable evidence, not a passing
result. Deferred and superseded-candidate rows are ineligible.

The ledger also preserves the earlier exact-SHA candidate
`ba7abefba4484689c953ac53fe8810322db1d184` and older revisions
`00748e6ac7b65f50e5c2af07f6e7c1c535c7f4c0` and
`b1a75d474ee44f583c32c5e6279c7026907553b9` as historical, ineligible context.
Their observations do not transfer to the final candidate SHA.

This report records qualification evidence only. It does not promote any
support manifest, change its `experimental` status, or convert deferred scope
into a support claim.
