# Phase 0 Support Characterization Work Log

## Checkpoint 1 — scope and source freeze verified

- Timestamp: 2026-09-02 UTC
- Dedicated goal created for this support-characterization lane.
- Read `TTTV2_MIGRATION_PLAN.md` and the complete TTTv2 hardware instruction manual.
- Verified the read-only source checkout is clean for tracked files, on branch
  `gongyu/tttv2_bh_support`, at the required full revision
  `00748e6ac7b65f50e5c2af07f6e7c1c535c7f4c0`.
- Scope is inventory/evidence analysis only. No TT tests, remote connections,
  resets, source extraction, or source-tree edits will be performed.
- Historical results will be accepted only when their recorded tested-code SHA
  is exactly the pinned revision. Results documented for `b1a75d474ee...` or
  other revisions will remain context, not pinned-revision evidence.

Next: enumerate in-scope test files, fixtures, markers, and collection-visible
test definitions from the pinned tree without executing pytest.

## Checkpoint 2 — static test, fixture, and marker census complete

- Timestamp: 2026-09-02 UTC
- Added a deterministic static inventory generator guarded by the pinned SHA
  and a clean tracked source worktree.
- Inventoried 111 in-scope Python test files and 1,368 source-level `test_*`
  definitions: 279 module, 404 LLM-runtime, 364 model, 47 demo, 77
  qualification, 195 shared-support, and 2 host-support definitions.
- Excluded the three MoE test files together with the explicitly excluded MoE
  implementation boundary.
- Inventoried 63 fixture definitions from the relevant test files and their
  three ancestor `conftest.py` files, plus 23 registered or source-observed
  marker tokens.
- Each test/file row records the exact pinned Git blob SHA. The inventory is
  static: parametrized source definitions are deliberately not expanded into
  collected node IDs, and device-fixture detection is a signal rather than a
  claim that an unmarked test is host-only.
- Outputs: `test_inventory.csv`, `test_file_inventory.csv`,
  `fixture_inventory.csv`, `marker_inventory.csv`, and the reproducible
  `generate_test_inventory.py`.

Next: characterize each model package, exact checkpoint IDs, model assets and
cache assumptions, and declared architecture/topology/parallelism coverage.

## Checkpoint 3 — model, asset, and topology characterization complete

- Timestamp: 2026-09-02 UTC
- Confirmed all twelve packages are concrete products with model, HF adaptor,
  executor, vLLM generator, tests, and a runnable pytest demo.
- Recorded exact source-defined HF IDs and revisions: five products pin a
  revision and seven float at the provider default.
- Recorded committed prompt/reference dependencies, cache precedence and
  writability requirements, offline-loading differences, `trust_remote_code`
  use, and the model-specific topology-appending behavior.
- Normalized Wormhole/Blackhole physical SKU, logical mesh, TP, and DP rows.
  Unsupported retained DP IDs and development stand-ins are not labeled as
  supported.
- Recorded the three Blackhole manifests as pre-acceptance contracts rather
  than result evidence.
- Outputs: `model_support.csv`, `hardware_coverage.csv`, and
  `asset_assumptions.json`.

## Checkpoint 4 — historical evidence attribution complete

- Timestamp: 2026-09-02 UTC
- Exact full/short-SHA searches found no checked-in hardware log, report, or
  Git note attributable to the pinned revision.
- Preserved representative manual results from
  `b1a75d474ee44f583c32c5e6279c7026907553b9` as ineligible context only.
  Git confirms that revision is not an ancestor of the pinned source, so its
  evidence cannot transfer.
- Output: `hardware_evidence.csv`; pinned verdict is `no attributable evidence`,
  which is neither pass nor failure.

Next: validate deterministic regeneration and all machine-readable artifacts,
then complete the support baseline.

## Checkpoint 5 — baseline validated and complete

- Timestamp: 2026-09-02 UTC
- Re-ran the guarded generator and verified byte-identical hashes for all five
  generated inventories.
- Parsed every CSV and JSON artifact and asserted expected row/model counts.
- Recomputed and matched every recorded Git blob SHA against the pinned commit.
- Confirmed exactly twelve model defaults, five pinned HF revisions, and no
  evidence row eligible for the pinned hardware baseline.
- Parsed the generator as Python and found no trailing whitespace in the
  support artifact tree.
- Final report: `support_baseline.md`.
- No pytest, TT hardware process, remote connection, reset, source extraction,
  or edit outside `qualification/analysis/support/` was performed by this lane.
