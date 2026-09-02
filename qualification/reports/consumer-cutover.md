# Consumer cutover readiness

## Verdict

Consumer cutover is **not complete**. The standalone package has replacement
paths, but pinned `tt-metal` and the inspected local vLLM TT plugin still refer
to the in-tree `models.common` product. Removal of the in-tree implementation
would currently break model, training, test, CI, and plugin consumers.

This assessment was read-only outside this repository. No tt-metal/vLLM file,
test, server, remote machine, or TT device was modified or run.

Inspected identities:

- tt-metal: branch `gongyu/tttv2_bh_support`, clean tracked tree, pinned SHA
  `00748e6ac7b65f50e5c2af07f6e7c1c535c7f4c0`;
- local vLLM: `/localdev/gwang/vllm`, branch
  `gongyu/tttv2_sibling_model_vllm_registration`, clean tracked tree, SHA
  `607831803c938c9dc4d92ef0b02a76ba622315fd`;
- no `/localdev/gwang/vllm_duo/vllm` Git checkout was available locally.

`consumer_import_sites.csv` records 209 concrete sites/gaps:

| Kind | Sites |
| --- | ---: |
| tt-metal Python imports outside the extracted product | 64 |
| tt-metal CI/test-path references | 107 |
| tt-metal qualification tools/contracts | 9 |
| vLLM standalone-generator registrations | 12 |
| vLLM selector/error-contract references | 15 |
| Missing vLLM registration-test gate | 1 |
| Missing vLLM package dependency contract | 1 |

Every CSV row names the repository/SHA/site/line, owner, exact replacement
contract, shim decision/duration, gate, order, and external status. All rows
remain `not_cut_over`.

## Key blockers

### vLLM plugin

The TT plugin has all twelve TTTv2 model IDs in `platform.py`, but every value
still imports `models.common.models.<model>.generator`. Its `pyproject.toml`
depends only on `tblib`; it has no declared `tt-transformers` dependency.

The plugin defaults Llama, Qwen2, Qwen3, and Mistral selectors to the legacy
`tt_transformers` implementation. TTTv2 remains opt-in through
`tt_transformers_v2`; Phi is registered only when its selector is explicitly
set. No plugin test at the inspected SHA mentions `tt_transformers_v2` or
verifies all twelve generator registrations. This missing gate must be added
before switching imports or defaults.

Required registry replacements are mechanical, for example:

```text
models.common.models.llama3_8b.generator:Llama3Generator
  -> tt_transformers.models.llama3_8b.generator:Llama3Generator
```

The plugin must pin `tt-transformers==<approved-release>` exactly (with its
lockfile/wheel hash), not rely on a sibling tt-metal checkout.

### tt-metal consumers

The consumer surface is broader than the twelve model directories. There are
64 external import sites across:

- current model demos such as Gemma, GPT-OSS, Qwen VL, Llama Galaxy,
  DeepSeek, and multimodal products;
- legacy `models.tt_transformers` implementation and tests;
- BGE-M3 consumers of `LazyWeight`;
- tt-train GRPO rollout code;
- Emule integration tests;
- experimental Quasar products; and
- one TTNN test that imports model-owned sampling code.

Most can change `models.common.sampling` to `tt_transformers.sampling`, or
`models.common.modules.<name>` to `tt_transformers.modules.<name>`, under the
same exact package pin.

One experimental Quasar generator imports legacy
`EagerLlamaExecutor`/`TracedLlamaExecutor` symbols that the standalone package
does not export. It requires an explicit refactor to the public generator and
executor contract; prefix substitution is not sufficient and no compatibility
claim is made.

The 107 CI references still invoke in-tree tests and coverage paths. They must
switch atomically to an exact standalone release checkout/package and its
`tests/`, `examples/`, or `qualification/` paths. A filesystem path shim is not
appropriate for CI.

### Qualification duplication

Nine `models/tttv2_*` tools/contracts remain in tt-metal. After consumers move,
tt-metal should retain only a short forwarding notice or orchestration wrapper
for one release. The schemas, validators, manifests, and hardware runner must
come from the same exact standalone version as the model code they grade.

## One-way dependency rule

The only permitted dependency graph is:

```text
vLLM TT plugin ─┐
tt-metal models ├──> tt-transformers (exact released version) ───> TTNN
tt-train/Emule ─┘

TTNN core ──X──> tt-transformers
```

The audit found no production import/reference from `ttnn/ttnn` or `ttnn/cpp`
to either `tt_transformers` or the in-tree `models.common` product. No proposed
CSV replacement introduces one.

The sole `tests/ttnn/.../test_tiebreak_input_adjust.py` dependency is a test
layer violation, not permission to add a TTNN dependency. Before cutover it
must either move to standalone sampling tests or become a TTNN-owned primitive
test with no model-package import. TTNN build/runtime/test dependency metadata
must never add `tt-transformers`.

## Ordered cutover

1. **Approve and publish a qualified standalone release.** Freeze the exact
   `tt-transformers`, TTNN, Python, Torch, Transformers, and HF revisions. Do
   not use `0.1.0.dev0` as an external compatibility promise.
2. **Add gates before switching.** Add vLLM tests for all twelve registry
   values, selector/fail-closed behavior, and imports with no tt-metal checkout.
   Relocate the TTNN sampling test and resolve the Quasar symbol mismatch.
3. **Update the vLLM plugin.** Add the exact package dependency and replace all
   twelve generator paths. Keep the `tt_transformers_v2` selector spelling for
   overlap; do not change defaults in the same step.
4. **Migrate tt-metal/tt-train/Emule imports by owner.** Change every inventoried
   shared sampling/module import to the exact released package, running the
   owner's existing tests plus isolated import probes.
5. **Switch CI and qualification atomically.** Install the same standalone
   version and invoke that version's tests/manifests/tools. Reject mixed code
   and qualification versions.
6. **Run representative serialized hardware gates.** Cover Wormhole and
   Blackhole model/serving paths at the exact package/plugin/TTNN versions.
   These runs were not performed in this assessment.
7. **Switch documented/default selectors.** Only after explicit TTTv2 plugin
   paths pass should the plugin default move from legacy to standalone. Retain
   the old selector alias for one plugin release.
8. **Replace active tt-metal implementation with forwarding shims.** Shims live
   in tt-metal, import the exact installed standalone package, warn, and contain
   no copied implementation. Retain for one tt-metal release after all listed
   consumers migrate.
9. **Remove implementation and later shims.** Delete the active duplicate only
   when the zero-reference/split-brain gates pass. Remove forwarding shims in
   the following announced release.

## Compatibility-shim policy

The broad `models.common.sampling` and narrow module imports (`lazy_weight`,
`tt_ccl`) have many downstream consumers, so a temporary forwarding shim is
practical. The shim must:

- be owned by tt-metal, not shipped as a second `models` namespace in the
  standalone wheel;
- contain imports only, with no copied classes/functions;
- verify/report the installed `tt-transformers` version;
- warn with the final removal release; and
- expire after one tt-metal release once all CSV sites are migrated.

The vLLM plugin needs no Python import shim after it pins the package; only the
environment selector spelling remains temporarily. CI paths and the Quasar
symbol mismatch must be changed directly rather than hidden behind shims.

## Split-brain prevention gates

Before removing any in-tree code, require all of the following:

1. `git grep` finds no non-shim `models.common.{sampling,modules,llm_runtime,models}`
   imports and no in-tree TTTv2 test/qualification paths.
2. vLLM's twelve registry values all start with `tt_transformers.models.` and
   its dependency metadata pins the same approved release used by the job.
3. A clean environment with no tt-metal source on `sys.path` imports every
   registered generator and reports each `__file__` beneath the installed
   standalone distribution.
4. `importlib.metadata.version("tt-transformers")`, the plugin lock, test
   report, support manifest, and qualification runner all report one identical
   version.
5. Runtime rejects simultaneous loading of a standalone model and an in-tree
   implementation; `sys.modules` must not contain active
   `models.common.models.*` TTTv2 modules.
6. The standalone static boundary checker confirms the package imports TTNN,
   while TTNN production directories contain no reverse import or dependency.
7. Wheel contents contain no old `models.*` namespace and tt-metal shims contain
   no implementation bodies.
8. Source provenance/blob inventory matches the released tag and hardware
   evidence names that exact tag/SHA.

Any failure blocks duplicate removal. Branch-name equality, an editable sibling
checkout, a skipped test, or evidence from another SHA does not satisfy these
gates.

## Required gates by stage

- **Pre-switch host:** CSV audit, plugin registry tests, isolated twelve-model
  imports, dependency-direction scan, manifest/schema validation.
- **Per-consumer:** existing owner unit/integration tests under the exact package
  pin, with no implicit tt-metal source import.
- **Plugin:** selector/unknown-model tests followed by representative real
  server smoke for Llama, Qwen, Mistral, Phi, DP, trace, cache, and cleanup.
- **CI:** dry-run/syntax validation followed by scheduled hardware jobs using
  one model code and qualification version.
- **Removal:** zero-reference scan, wheel namespace inspection, shim-body audit,
  and repeat clean-environment imports.

This report does not claim those external changes or gates have occurred.

## Reproduction

```bash
python -B qualification/tools/audit_consumer_cutover.py
```

The tool verifies the pinned tt-metal revision, reads vLLM at its actual local
HEAD, recreates `consumer_import_sites.csv`, and fails if it detects a proposed
production TTNN reverse dependency.
