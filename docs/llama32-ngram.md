# Llama 3.2 1B with CPU n-gram proposals

`Llama32_1BGenerator` computes target logits for CPU n-gram proposals from vLLM.
`Llama32_1BGenerator.decode_forward` executes candidate columns sequentially with
host logits. `vllm_tt_plugin.spec_accept.accept_sampled_drafts` selects accepted
drafts and a correction or bonus token. `Llama32_1BGenerator` supplies a
sequential correctness reference.

The initial speculative scope is the base checkpoint
`meta-llama/Llama-3.2-1B`, synchronous scheduling, physical batch 1, 2, or 4,
one logical 1x1 mesh and one execution lane, requested draft width 1, 2, or 3, host sampling, disabled
prefix caching, `trace_mode="none"`, and `optimizations="performance"`.
Device qualification remains unverified until the complete device command in
`llama32-ngram.yml` or `llama32-ngram-source.yml` succeeds at the exact
published revisions.

## Launch

Install standalone tt-transformers and the plugin into the same environment.
When using the source environment from `docs/ttnn-development.md`, expose this
checkout with `PYTHONPATH="$PWD/src:$PWD"` instead of installing a
tt-transformers distribution into that environment.
Follow the checked-out plugin's `README.md` for vLLM installation. Pin a plugin
revision containing logits acceptance, initially
`61c40e6334bf28467693ca530ced66beabd96998`. Record the complete Python package
tuple and the exact checkpoint, tt-transformers, plugin, and tt-metal revisions.
Populate an isolated Hugging Face cache with the authorized base checkpoint at
the selected revision. The Hugging Face cache's `refs/main` must select that
revision because `hf_generator._load_model` resolves the configured model ID.

From the tt-transformers checkout root, export the source path and explicit
architecture override for every server process:

```bash
export PYTHONPATH="$PWD/src:$PWD"
export TT_MODEL_CLASS_OVERRIDES='TTLlamaForCausalLM=tt_transformers.models.llama32_1b.vllm_generator:Llama32_1BGenerator'
export MESH_DEVICE='(1, 1)'
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TT_TRANSFORMERS_OFFLINE=1
python "$PLUGIN_ROOT/examples/server_example_tt.py" \
  --model meta-llama/Llama-3.2-1B --revision "$CHECKPOINT_REVISION" \
  --max_num_seqs 4 --max_model_len 1024 --block_size 32 \
  --no-async-scheduling --no-enable-prefix-caching \
  --additional-config '{"tt":{"trace_mode":"none","optimizations":"performance"}}' \
  --speculative-config '{"method":"ngram","num_speculative_tokens":3,"prompt_lookup_min":2,"prompt_lookup_max":4}'
```

Omit `sample_on_device_mode` to select host sampling. Do not pass an explicit
null value because the plugin validates supplied `sample_on_device_mode` values.
The startup log must show `TTLlamaForCausalLM` registered to
`tt_transformers.models.llama32_1b.vllm_generator:Llama32_1BGenerator`.

## Durable device qualification

The GitHub Actions workflow `Llama 3.2 1B sampled ngram qualification` in
`.github/workflows/llama32-ngram.yml` runs on
`tt-ubuntu-2204-N150-viommu-stable` or
`tt-ubuntu-2204-N300-viommu-stable`, selected by the `runner` input. Both runner
pools execute the same logical 1x1 mesh. Dispatch `llama32-ngram.yml` with complete
checkpoint, plugin, and tt-metal commit SHAs and a `runtime_run_id` from the
existing `TTNN development runtime` workflow. The `ttnn-runtime` bundle must
match the supplied tt-metal SHA. `ttnn_dev.py env` installs that fixed wheel
and its matching SFPI toolchain, and `doctor` plus a JIT tensor operation
check the runtime assets before the plugin installation. The repository must
supply an authorized `HF_TOKEN` secret for the gated base checkpoint.
`llama32-ngram.yml` uses an isolated model cache and permits only one
qualification job per runner pool at a time.

`llama32-ngram.yml` and `llama32-ngram-source.yml` run the same complete
qualification sequence:

1. `.github/scripts/qualify_llama32_ngram.py` imports TTNN, `ttnn.device`, and vLLM in one
   interpreter and requires TT platform activation.
2. `.github/scripts/qualify_llama32_ngram.py` starts ordinary servers at batch 1, 2, and 4 with
   the explicit standalone adapter, host sampling, and eager execution.
3. `test_candidate_logits_and_inactive_kv` compares every valid target column
   with ordinary target execution for all nine batch/draft-width combinations.
4. `test_candidate_logits_and_inactive_kv` reads every layer's KV tensors and requires exact
   preservation outside valid candidate positions, including zero-draft and
   padding rows with unallocated page-table suffixes.
5. `test_rejection_correction_and_committed_kv` directs first rejection,
   partial acceptance, and bonus cases through sampled acceptance with top-k 1
   and one allowed token per request. The allowed-token constraint makes each
   directed case exact when `Llama32_1BGenerator` returns tied logits.
   `test_rejection_correction_and_committed_kv` executes the committed prefix
   against a separate physical KV region. The first two forwards after a
   correction must match ordinary execution against that separate KV region,
   including at a KV page boundary.
6. `.github/scripts/qualify_llama32_ngram.py` starts synchronous n-gram servers
   for all nine batch/draft-width combinations. Repeated token prompts and an
   allowed-token constraint provide a deterministic integration oracle. vLLM
   metrics must show proposals and accepted drafts, and `TT submissions` log
   records must show logits verifies. Concurrent greedy and sampled requests
   vary seed, temperature, top-k, top-p, prompt length, and output length.
   `.github/scripts/qualify_llama32_ngram.py` also checks penalties, allowed
   tokens, bad words, minimum tokens, structured output, logprobs, EOS, stream
   cancellation, and immediate request reuse. Greedy
   controlled outputs and raw logprobs must match the ordinary HTTP baseline.
   An identical sampled request schedule must reproduce its token IDs.
7. `.github/scripts/qualify_llama32_ngram.py` starts four requests whose combined
   KV use exceeds the shared 1024-token pool. The ordinary batch-4 launch and the
   speculative batch-4/draft-3 launch must report a positive preemption
   counter delta and return the full constrained token sequences after resume.

`test_candidate_logits_and_inactive_kv` requires exact logits, exact softmax distributions,
and exact committed KV because both executions use the same eager graph,
precision, physical batch, and token prefix. An exact comparison failure needs
investigation before any tolerance changes.

`.github/scripts/qualify_llama32_ngram.py` installs a private, child-process
observer for `Scheduler.finish_requests` during HTTP qualification. The observer
calls `Scheduler.finish_requests` unchanged, then records request status and
scheduler ownership. The cancellation check requires a named request to reach
`FINISHED_ABORTED` before its output limit and to leave `Scheduler.requests`,
`Scheduler.running`, and the waiting queues before replacement admission.
Frontend disconnect logs alone do not prove backend cancellation in vLLM 0.30.
The observer does not change the installed vLLM package or normal server launches.

Run `.github/scripts/qualify_llama32_ngram.py` on a reserved, provisioned
N150-compatible device:

```bash
MESH_DEVICE=N150 python .github/scripts/qualify_llama32_ngram.py \
  --plugin-root "$PLUGIN_ROOT" --plugin-revision "$PLUGIN_REVISION" \
  --checkpoint-revision "$CHECKPOINT_REVISION" --output generated/llama32-ngram
```

Plan for 45 to 165 minutes after environment and checkpoint preparation. The
success signal is an exit status of zero, passing `device-matrix.xml`, and
`generated/llama32-ngram/success.json`. Run a lab qualification command as
background work and capture its output. The GitHub Actions artifact includes
package versions, exact Git revisions, launch arguments, server logs, final
metrics, HTTP responses, hardware inventory, and teardown status. Installed
package metadata can lack a VCS commit. `provenance.json` labels that package's
source revision as unverified when installed VCS metadata is absent.

## Prepared source runner

`.github/workflows/llama32-ngram-source.yml` runs the complete qualification
against an already prepared source environment. `llama32-ngram-source.yml`
uses shell steps and pinned checkout/upload actions. `llama32-ngram-source.yml`
does not install packages, rebuild TTNN, reserve devices, or start a container.

`workflow_dispatch` is for an explicitly authorized, trusted workflow revision
on a dedicated self-hosted Linux x64 runner. The operator selects a reviewed
tt-transformers revision and a trusted environment activation script.
`llama32-ngram-source.yml` has no pull-request, push, or schedule trigger.
The operator reserves and provisions hardware, registers a temporary runner,
and removes the temporary runner outside `llama32-ngram-source.yml`. The
operator starts the temporary runner after any existing device job finishes.

The prepared runner must have:

- Exclusive access to the reserved physical device and no other TT device job.
  The `runner_label` input must identify that dedicated runner.
- A source-built TTNN environment, vLLM, vllm-tt-plugin, and the repository's
  host-test dependencies. One `source "$ENVIRONMENT_SCRIPT" "$ENVIRONMENT_NAME"`
  command must select the complete environment and its `TT_METAL_HOME`.
- A clean vllm-tt-plugin checkout at `plugin_revision` and a clean tt-metal
  checkout at `tt_metal_revision`. The source-built `ttnn` package and native
  extension must resolve inside that tt-metal checkout.
- Read access to the complete cached base checkpoint at `checkpoint_revision`.
  The Hugging Face cache's `refs/main` must select `checkpoint_revision`.
  Converted-weight, TTNN, and TT-Metal kernel caches must be writable.
- `python`, `tt-smi`, `git`, and `pgrep` available after environment activation.
  The prepared environment must provide `pip` or `uv` for dependency checks.
  GitHub Actions checkout and artifact upload must work from the runner.

`llama32-ngram-source.yml` accepts seven inputs: `runner_label`,
`environment_script`, `environment_name`, `plugin_checkout`, `plugin_revision`,
`tt_metal_revision`, and `checkpoint_revision`. GitHub Actions passes each
input through an environment variable. `llama32-ngram-source.yml` sources the
quoted `environment_script` once and supplies the quoted `environment_name`
as one argument. `llama32-ngram-source.yml` then sets `PYTHONPATH` to the fresh
Actions checkout's `src` and repository root, the selected tt-metal checkout's
real `ttnn` package before the tt-metal repository root, and the selected
vllm-tt-plugin checkout's `src`. Import-origin checks reject stale standalone
or plugin packages before host contracts start.

The prepared candidate uses source commit
`de546d3b146758714d900f11b218c8f9c805f410` from tt-metal tag `v0.79.0`.
The prepared candidate reports TTNN distribution version `0.79.0rc2`; the
qualification records that actual version without rewriting package metadata.
The prepared source profile also retains NumPy 1.26.4 and OpenCV 4.11.0.86.
The tt-metal source environment requires NumPy below 2, while vLLM's declared
OpenCV minimum selects a release that requires NumPy 2. The prepared profile's
FiftyOne 0.25.2 also declares an SSE dependency that conflicts with vLLM's
installed SSE package. `llama32-ngram-source.yml` records the complete package
metadata check and its exit status as diagnostics. Exact import origins,
ordinary serving, and every host, device, and HTTP correctness gate remain
mandatory. The Llama text qualification does not qualify computer vision or
FiftyOne benchmarking in this source environment.
For the prepared `llama-ngram` profile, dispatch with:

```bash
gh workflow run llama32-ngram-source.yml \
  --repo "$QUALIFICATION_REPOSITORY" --ref "$REVIEWED_SOURCE_REF" \
  -f runner_label=tt-llama32-prepared \
  -f environment_script=/localdev/viktorpus/dotfiles/tt/tt-env.sh \
  -f environment_name=llama-ngram \
  -f plugin_checkout=/localdev/viktorpus/llama-ngram-qualification/vllm-tt-plugin \
  -f plugin_revision=61c40e6334bf28467693ca530ced66beabd96998 \
  -f tt_metal_revision=de546d3b146758714d900f11b218c8f9c805f410 \
  -f checkpoint_revision=4e20de362430cd3b72f300e6b0f18e50e7166e08
```

`llama32-ngram-source.yml` requires the checked-out tt-transformers SHA to match
the dispatched GitHub SHA. `llama32-ngram-source.yml` also checks the exact
plugin and tt-metal SHAs, tracked-file cleanliness, source import origins,
TT platform activation, and the base checkpoint revision. The focused host
contracts must execute without skips. `llama32-ngram-source.yml` also runs the
complete host selector and records its skips in `full-host.xml`.
`.github/scripts/qualify_llama32_ngram.py`
then runs ordinary serving at B=1/2/4, all 18 real-model device tests, and all
nine sampled n-gram B/K combinations. The prepared-source path does not pass
`--skip-device-matrix`.

Plan for 45 to 165 minutes after preflight. The success signal is a green
workflow with non-skipped host contracts, all 18 device cases, and
`generated/llama32-ngram-source/qualification/success.json`.
`llama32-ngram-source.yml` uploads `prepared-source.json`, package versions,
module origins, the native extension checksum, host and device JUnit reports,
the exact qualifier command, server logs, HTTP results, metrics, hardware
inventory, and teardown state even after a failed gate. A teardown check
reports any remaining engine process and fails `llama32-ngram-source.yml`
without resetting hardware or terminating an unrelated process.

The real-model HTTP suite compares deterministic request controls with
ordinary target execution. The plugin's host and synthetic device suites
retain precise small-distribution statistical checks. Seeded speculative and
ordinary decoding can consume different random draws, so sampled text equality
between those two schedules is not an acceptance criterion. All device claims
remain unverified until the complete qualification command succeeds.
