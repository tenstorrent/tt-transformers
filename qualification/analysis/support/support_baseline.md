# TTTv2 Phase 0 Test, Example, and Support Baseline

## Verdict

The pinned source contains twelve concrete model products and broad test/demo
coverage, but it does **not** contain hardware evidence attributable to the
pinned revision `00748e6ac7b65f50e5c2af07f6e7c1c535c7f4c0`. Source presence,
parameterized coverage, a capability contract, or a skipped test is therefore
not a support claim.

The source checkout was clean for tracked files and exactly at the pinned SHA
while this inventory was generated. No pytest process, TT device process,
remote connection, or hardware reset was performed.

## Inventory boundary and counts

The static census includes all Python tests below `models/common/tests/`
except the explicitly excluded `modules/moe/` subtree, plus readiness checks
and `models/test_tttv2_validate_vllm_matrix.py`. It also includes `demo.py`
files because those are intentionally invoked as pytest entry points even
though their filenames do not start with `test_`.

| Category | Files | Source-level `test_*` definitions |
| --- | ---: | ---: |
| Module | 19 | 279 |
| Common LLM runtime | 18 | 404 |
| Model | 39 | 364 |
| Demo | 14 | 47 |
| Qualification/readiness | 9 | 77 |
| Shared support | 11 | 195 |
| Host support | 1 | 2 |
| **Total** | **111** | **1,368** |

These are source definitions, not collected pytest nodes. Parameterization is
not expanded, and dynamically generated cases are not guessed. Every test and
file row carries its Git blob SHA. The reproducible generator refuses a source
checkout at another SHA or with tracked changes.

The accompanying support-asset census has 72 tracked files spanning runnable
demos, prompt/reference assets, readiness tools, capability manifests,
validators, hardware-gate tooling, TTTv2 documentation, and the transitive
cache-entry counter required by the device fixtures.

## Fixtures and markers

There are 63 relevant fixture definitions across the selected tests and their
ancestor `conftest.py` files. Important behavior includes:

- root `device`, `mesh_device`, `t3k_single_board_mesh_device`,
  `pcie_mesh_device`, `bh_1d_mesh_device`, and `bh_2d_mesh_device` ownership;
- architecture selection synthesized by `pytest_generate_tests` from
  `silicon_arch_name` and `silicon_arch_<arch>` fixture names;
- the common module-scoped `ttnn_mesh_device`, which filters invalid mesh
  cross-products, requires explicit Blackhole `MESH_DEVICE`, verifies physical
  four-die cluster type, owns fabric setup/teardown, and serializes access with
  `/tmp/tt_device.lock`;
- Blackhole-selected fixture exceptions remain failures, while legacy
  non-opted-in paths may translate device setup exceptions to skips;
- autouse garbage collection, default-device restoration, minimum-grid
  checking, graph reporting, and test timestamps.

The marker census contains 23 registered or source-observed tokens. The
actively important tokens are `slow`, `timeout`, `usefixtures`, `xfail`, and
the root-registered CI/performance/grid/device markers. Built-in pytest tokens
such as `parametrize`, `skip`, `skipif`, and `usefixtures` are reported even
though they need no local registration.

The current suite does not implement the migration plan's intended explicit
host/device/model/architecture/SKU marker taxonomy. Hardware selection is
mostly encoded in fixture names, environment variables, parameter IDs, and
runtime skips. This makes static scheduling and support claims less reliable;
the standalone suite should add explicit markers without treating skips as
evidence.

## Twelve model products

All twelve packages have a concrete tensor model, HF adaptor, executor,
vLLM-facing generator, model tests, and a runnable pytest demo. Eleven are
covered by the generic executor-binding characterization; Llama-3.1-8B has its
own dedicated runtime/model integration suites. Qwen2.5-Coder-32B and
Qwen3-32B additionally contain package-local diagnostic demos.

That is a runnable-source statement, not a claim that checkpoints are locally
available or that any geometry passed at this SHA. The normalized per-model
status is in `model_support.csv`; topology rows are in
`hardware_coverage.csv`.

Five runtime defaults pin a 40-character HF revision:

- DeepSeek-R1-Distill-Qwen-14B;
- Phi-4;
- Qwen2.5-72B-Instruct;
- Qwen2.5-Coder-32B-Instruct; and
- Qwen3-32B.

Seven float at the provider default: Llama-3.2-1B, Llama-3.2-3B,
Llama-3.3-70B, Llama-3.1-8B, Mistral-7B, Qwen2.5-7B, and Qwen2-7B. Mistral
spells this as an explicit `None`; the Llama-3.1-8B adaptor requires an
argument or `HF_MODEL` while its demo supplies the exact default ID.

Only Llama-3.1-8B, Llama-3.3-70B, and Qwen3-32B have Blackhole capability
manifests. Those manifests declare `pre_acceptance` requirements; they are not
pass reports. Required Blackhole coverage is:

- Llama-3.1-8B: P150 TP1/DP1, P150x4 TP4/DP1 and TP1/DP4, plus an additive
  real-P300 TP1/DP2 leg; the two-P150 P300 stand-in is development-only.
- Llama-3.3-70B: logical P150x4 TP4/DP1.
- Qwen3-32B: logical P150x4 TP4/DP1.

Physical provenance must remain explicit. A P150_X4 quietbox result is not a
physical P300_X2 result even when both satisfy the code's logical P150x4
profile.

Wormhole coverage is model-specific. The principal patterns are N150/N300/T3K
for the small Llama and Mistral products, TP2-only lanes for Qwen2/Qwen2.5-7B
and Phi-4, T3K TP8-only for Qwen2.5-72B and Qwen2.5-Coder-32B, and T3K TP8 for
the two large BH-enabled models. Several retained `ci-b1-DP-*` IDs
intentionally skip because the resulting TP lane is invalid or cannot hold the
model. `hardware_coverage.csv` records admitted and required TP/DP rows rather
than inferring support from the presence of a parameter ID.

## Model and test asset assumptions

Each product loads HF config, tokenizer/chat template, and a complete
`AutoModelForCausalLM` state dict, generally at `torch.bfloat16`, then writes
converted tensors to a model/topology cache. The intended package does not
ship weights, tokenizers, firmware, or drivers.

Common cache precedence is explicit `cache_dir`, then `TT_CACHE_PATH`, then a
cwd-relative `model_cache/<HF_MODEL>/<topology>`. Llama-3.1-8B additionally
falls back to `TT_CACHE_FALLBACK_PATH` or `/tmp/tttv2_model_cache` on a
permission failure. Callers must know whether a model appends its topology;
DP demos also construct lane-specific sibling cache paths.

Every model default has a committed `.refpt`, but eleven of the twelve
artifacts contain no model/revision provenance. Qwen2-7B alone records HF
revision `f2826a00ceef68f0f2b946d945ecc0477ce4450c`; its runtime remains
unpinned. Most demos also depend on prompt/reference files under the legacy
`models/tt_transformers` tree, an extraction dependency that must be migrated
or replaced. Qwen2.5-72B, Qwen2.5-Coder-32B, and Qwen3-32B enable
`trust_remote_code=True`.

The vLLM qualification runner expects external, immutable expectations that
pin the HF snapshot/ref, verified files, revisions, environment, and TT cache
root. The three checked-in capability manifests do not themselves supply
those model snapshot revisions.

See `asset_assumptions.json` for exact IDs, revisions, paths, environment
inputs, and cache rules.

## Pinned-revision evidence ledger

An exact full/short-SHA search of the pinned tree and migration workspace found
no hardware log or report naming
`00748e6ac7b65f50e5c2af07f6e7c1c535c7f4c0` as the tested code. Git notes
also contain no entry for it. Consequently the pinned hardware baseline is
`no attributable evidence`, not pass, fail, or deferred.

The hardware manual records passes and failures for
`b1a75d474ee44f583c32c5e6279c7026907553b9`, including Blackhole module and
model observations. That SHA is different from, and is not an ancestor of,
the pinned revision, so none of those results transfer. The manual also names
other historical or remote checkout SHAs; branch identity never substitutes
for an identical full SHA.

`hardware_evidence.csv` preserves representative older results solely as
ineligible context and gives the exact exclusion reason. A future pinned
baseline must run with synchronized clean checkouts and record the full SHA,
machine, physical SKU, mesh, environment/cache, exact node, exit status,
metrics, teardown, reset state, and log path.

## Machine-readable artifacts

- `test_inventory.csv`: one row per source test definition.
- `test_file_inventory.csv`: file-level counts and signals.
- `fixture_inventory.csv`: fixture ownership, scope, autouse, and params.
- `marker_inventory.csv`: marker registration and source occurrence.
- `support_asset_inventory.csv`: demos, readiness/qualification assets, and
  relevant documentation.
- `model_support.csv`: twelve-package concrete/runnable status and exact HF
  defaults.
- `hardware_coverage.csv`: normalized architecture/SKU/mesh/TP/DP declarations.
- `hardware_evidence.csv`: pinned evidence verdict and excluded other-SHA
  observations.
- `asset_assumptions.json`: checkpoint, cache, prompt/reference, and environment
  assumptions.
- `generate_test_inventory.py`: deterministic regeneration tool.

## Phase 0 implications

The migration can treat the source/test census as frozen, but it cannot claim
a pinned hardware baseline yet. Before promotion, preserve the model-specific
capacity skips, add explicit scheduling markers, pin the seven floating HF
revisions, move legacy prompt/reference dependencies into the standalone
boundary, give every reference artifact model/revision provenance, and collect
fresh same-SHA host and serialized hardware evidence.
