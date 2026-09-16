# Source provenance

The initial standalone package was extracted from `tt-metal` commit
`00748e6ac7b65f50e5c2af07f6e7c1c535c7f4c0`.

`source_inventory.csv` maps the reviewed source scope to its standalone
destination, or identifies migration-only context archived with the audit tag.
MoE is the only implementation subsystem excluded from that initial extraction;
the Galaxy port below adds two further exclusion categories of its own.

`issue_migration.md` records the recreation of the `tt_transformer v2.1.0`
milestone and its open issues from `tt-metal`, and the back-pointer work still
outstanding on the `tt-metal` originals.

Detailed migration work logs, generated analyses, and intermediate evidence
are intentionally not part of the active repository. The pre-cleanup audit
tree is preserved by Git tag `tttv2-migration-audit-54648bc`.

## Second event: the Galaxy 8x4 and 2D-module port

The ledger records two provenance events. Rows 1-366 are the initial
extraction above. The remaining 97 rows are a later port from a different
`tt-metal` branch, `apbernal/tttv2_wh_glx_2d_modules_milestone_c` at commit
`0036376d342bbd9de50ca8ea284b578838c44b29`.

That branch is not a descendant of the initial extraction base. The two
lineages share merge base `bc8b7d73b7650450b1c07bf93a4c813b1dc93a9f` and then
diverge for three weeks, so the port is a merge across a fork rather than a
file move. Its scope was taken from the three-way split of `models/common/**`
at that merge base: 104 paths changed only on the branch, 15 changed on both
sides, and 113 changed only upstream. The 113 were left untouched.

What the port adds is the Galaxy 8x4 mesh support, which was absent from the
initial extraction base rather than excluded by it, plus the five remaining 2D
tensor-parallel module variants alongside the `mlp_2d` and `rmsnorm_2d` the
initial extraction already carried, and the 2D prefetcher they stage weights
through.

Its two exclusion categories are:

- `excluded_condemned_runner` (16 rows) - the `GalaxyDirectRunner` execution
  path, retired by an operator decision of 2026-09-02, and the demos and suites
  that cannot run without it. The `GalaxySamplingPolicy` dataclass defined
  inside the runner is carried separately, as
  `src/tt_transformers/models/galaxy/sampling_policy.py`.
- `excluded_deferred_performance` (3 rows) - the paired-throughput arm, whose
  gate thresholds were measured on hardware this repository's qualification
  policy does not yet describe.

Both are `excluded` rather than `archived`: the files are real source a future
contributor may want, not migration context.

### Rows the port supersedes

22 rows from the initial extraction name a file the port then merged or
overwrote. Those rows are left as written, because they record the initial
extraction truthfully; their `git_blob_sha` is the blob as extracted and is not
the current content of the destination. They break down as eight `llm_runtime`
runtime sources and three of its test sources, four module sources and two
module-test sources, four 1D test sources, and the test requirements now folded
into `pyproject.toml`. Where such a row carries `boundary_cleanup=true`, the
extraction had transformed the file in a way that appears in neither lineage's
history, so the port had to reapply that transformation by hand after merging.

### What the port does not claim

The device suites land as files only. `tests/hardware/hardware-matrix.json`
declares no Galaxy, TG or 32-chip node, so nothing the port adds has been
re-qualified here, and no gate in this repository executes it. Every Galaxy
claim behind this code was measured on the `tt-metal` branch named above.

Two of the ported suites compared the executor against tensors recorded by the
condemned runner; because the runner wrapped the same model object the executor
drives, those gates never validated Galaxy numerics, and they are retired
rather than regenerated. See `docs/validation.md`.

The scope limit on this event is the same as the first: the ledger inventories
source-tree files that were candidates for the package. The porting author's
working notes, logs and evidence trees were never candidates and are not
listed.
