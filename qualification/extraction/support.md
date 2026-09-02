# Phase 2 support-surface extraction report

## Outcome

Tests, examples, model documentation, reference assets, and qualification
support were mechanically extracted from pinned `tt-metal` revision
`00748e6ac7b65f50e5c2af07f6e7c1c535c7f4c0`. This lane wrote only under
`tests/`, `examples/`, and `qualification/`; it did not modify
`src/tt_transformers`, run pytest, connect to remote machines, or open/reset TT
hardware.

The corrected authority in `qualification/provenance/source_inventory.csv`
has 364 rows. Exactly 204 source rows assign at least one
`tests/`, `examples/`, or `qualification/` destination. Semicolon expansion
produces 221 source/destination assignments and 218 distinct destinations.

| Source category | Rows |
| --- | ---: |
| test | 101 |
| qualification tool | 29 |
| qualification asset | 20 |
| model documentation | 12 |
| hybrid test demo | 12 |
| source-CI manifest | 7 |
| example support | 6 |
| qualification manifest | 5 |
| boundary support | 4 |
| test fixture | 3 |
| hybrid model demo | 2 |
| qualification documentation | 1 |
| test support | 1 |
| qualification schema | 1 |

The destinations comprise 134 distinct test paths, 35 example paths, and 49
qualification paths. Existing `tests/host` scaffold tests were preserved.
Existing `examples/README.md` was extended rather than replaced, and the
newer `qualification/schemas/support-manifest.schema.json` was preserved next
to the extracted `bh_required_capabilities.schema.json`.

### Support-inventory reconciliation

The earlier support census identified 21 readiness-check assets, including six
test files, that initially lacked destinations. The provenance authority now
assigns all of them to `qualification/readiness/` or
`tests/qualification/readiness/`. Their blobs are verified in the normal copy
manifest, and the temporary `support_unassigned.csv` ledger has been removed.

## Copy and provenance verification

`extract_support_surface.py` reads bytes with `git cat-file blob`, not from the
source worktree. Before copying each row it verifies that
`REV:source_path` resolves to the inventory's Git blob SHA. It refuses an
unexpected pre-existing target on its first run and restricts all writes to
the three authorized top-level roots.

`support_copy_manifest.csv` contains one record per source/destination
assignment with:

- source path, pinned Git blob SHA, and source Git mode;
- category, disposition, and boundary-cleanup flag;
- destination path, size, SHA-256, and recreated mode;
- source blob size;
- whether the destination is still an exact raw blob copy; and
- every mechanical transformation or deferred split.

At extraction time, 75 assignments remained byte-for-byte raw copies. The
other 125 received only deterministic import rewrites, fixture concatenation,
or both. Reference `.refpt` files and the compressed corpus are unchanged
binary blob copies. Both inventory rows targeting the Llama-3.1-8B test
reference resolve to the same source blob, so the destination collision is
lossless. The three source-executable shell tools retain mode `0755`; regular
files are recreated as `0644`.

The extraction is reproducible: rerunning the script recreates the same 218
destinations and verification manifest. It does not own or rewrite
`examples/README.md`, the standalone support-manifest schema, provenance,
analysis, reports, or production package files.

Phase 3 resolution is recorded separately in `support_boundary.md` and
`support_boundary_manifest.json`.

## Mechanical import rewrites

The extraction rewrote 656 import statements whose ownership was already
unambiguous, including:

- `models.common.modules` to `tt_transformers.modules`;
- `models.common.llm_runtime` to `tt_transformers.llm_runtime`;
- `models.common.models` to `tt_transformers.models`;
- `models.common.sampling` to `tt_transformers.sampling`;
- narrow device/tensor helpers to their standalone owners;
- validation, metrics, benchmark, model-target, and trace-size helpers to
  `qualification.tools`;
- retained test helpers to `tests.support` or `tests.integration`; and
- the standalone `Mode` import to `tt_transformers.modules.mode`.

Symbol-level split sources were not globally redirected. In particular,
`models.tt_transformers.tt.common` still owns `encode_prompt_hf` and
`get_padded_prefill_len` in the extracted snapshots because the inventory does
not assign those functions an unambiguous standalone owner. The broad legacy
model-config and fixture helpers are likewise deferred.

## Whole-snapshot split handling

Phase 2 preserves behavior and source intent; it does not perform semantic
redesign. Every split source was therefore copied as a whole pinned snapshot
to each assigned support destination, with the split recorded as deferred.

In particular:

- all twelve hybrid pytest demos were copied whole to both
  `examples/<model>/demo.py` and
  `tests/hardware/models/<model>/test_demo.py`;
- Qwen2.5-Coder-32B and Qwen3-32B package-local demos were copied whole to
  both their example smoke and hardware smoke destinations;
- Llama-3.1-8B demo helpers and local prompt/reference assets were copied to
  both assigned destinations;
- the full broad helper snapshots remain in `tests/integration` or
  `tests/support` while narrow production ownership is handled separately;
- source-CI YAML/shell snapshots and the two repository-wide model target
  YAML files remain unfiltered; and
- `tests/conftest.py` is a mechanical concatenation of the complete root,
  `models`, and common-test fixture snapshots, with pinned path/blob segment
  headers.

There are 53 distinct destinations with a Phase 2 deferred semantic
split/filter. Phase 3 subsequently resolved all 53; see
`support_boundary.md`. No skip, xfail, parameter ID, reference tensor, or
hardware gate was deleted.

## Syntax and collection limits

All 171 Phase 2 Python files under `tests/`, `examples/`, and
`qualification/tools/` parse successfully with `ast.parse`.

Pytest collection was deliberately not attempted. The merged conftest still
imports tt-metal-wide fixture support, initializes architecture policy during
collection, and includes hardware ownership behavior. Collection would also
fail in an isolated standalone environment until production extraction and
Phase 3 boundary cleanup supply or remove all legacy owners. A collection skip
would not constitute hardware evidence in any event.

No behavioral, host, or hardware test result is claimed by this extraction.

## Phase 2 legacy gaps (resolved in Phase 3)

Phase 2 recorded 21 executable import lines as unresolved:

- hybrid demos and two reference generators import `encode_prompt_hf` and one
  demo also imports `get_padded_prefill_len` from
  `models.tt_transformers.tt.common`;
- the reference-output generator imports TTTv1 `ModelArgs`;
- merged `tests/conftest.py` imports the legacy trace-region configuration and
  `tests.scripts.common`; and
- these imports require symbol-level replacement or fixture reduction rather
  than a safe module rename.

Legacy filesystem assumptions also remained in the Phase 2 snapshot. Most
tests still name old `models/common/tests/demos`,
`models/tt_transformers/tests/reference_outputs`, and
`models/tt_transformers/demo/sample_prompts` paths. Their assets have been
copied to standalone destinations, but changing the path strings now would
alter source-contract assertions and hybrid behavior. Phase 3 updated the
example and test copies together, including validator node IDs and capability
manifest paths; active legacy executable-import count is now zero.

The extracted Blackhole capability manifests and validator still describe
their historical source node IDs. They are preserved evidence contracts, not
standalone acceptance results. Model-target and source-CI snapshots contain
unrelated tt-metal entries until their recorded split is performed.

## Reproduction

From the repository root:

```text
python qualification/extraction/extract_support_surface.py
```

This performs no imports from the extracted code and no test or hardware
execution. Independent verification should parse the manifest, recompute each
destination SHA-256, resolve every source blob at the pinned revision, assert
all 204 authority rows and 221 assignments are covered, and parse all Python
files without importing them.
