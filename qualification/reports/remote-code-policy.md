# Remote-code loading policy

Status: **forced Hugging Face remote-code execution removed**

Production model loading now relies on the built-in Qwen2/Qwen3 support in the
declared `transformers==5.12.1` dependency. No Qwen adaptor opts into executing
code supplied by a model repository by default.

## Approved pinned transforms

All source blobs are from the read-only commit
`00748e6ac7b65f50e5c2af07f6e7c1c535c7f4c0` and are verified by
`tools/extract_model_packages.py --scope remote-code`.

| Pinned source (blob SHA) | Destination | Transform |
|---|---|---|
| `models/common/models/qwen25_72b/hf_adaptor.py` (`b1d6a9143511bb0b5685b356083facb3deb0b0e5`) | `src/tt_transformers/models/qwen25_72b/hf_adaptor.py` | Removed the tokenizer's literal `True` keyword and the shared literal forwarded to config/model loaders. |
| `models/common/models/qwen25_coder_32b/hf_adaptor.py` (`341d7c6fabe9b9bfa72eaf4f26a0cc2e5afdf123`) | `src/tt_transformers/models/qwen25_coder_32b/hf_adaptor.py` | Removed the tokenizer's literal `True` keyword. |
| `models/common/models/qwen3_32b/hf_adaptor.py` (`2b28398329500af2007b2f5fa678dc002d507528`) | `src/tt_transformers/models/qwen3_32b/hf_adaptor.py` | Removed the tokenizer's literal `True` keyword. |

This is four removed forced values across three adaptors: three direct
`AutoTokenizer.from_pretrained` keywords and one Qwen2.5-72B kwargs mapping used
by both `AutoConfig` and `AutoModelForCausalLM`. Public signatures did not expose
a Qwen remote-code option, so no new opt-in parameter was added.

The extractor performs those removals mechanically, checks the pinned blobs,
and verifies exact policy-normalized destination hashes. The static import
checker rejects both a literal call keyword and a forwarded dictionary value
set to `True` anywhere in the model layer.

## Complete production call audit

The post-change AST audit covers 51 calls named `from_pretrained` in 26 source
files:

| Callee form | Calls |
|---|---:|
| `AutoConfig.from_pretrained` | 13 |
| `AutoModelForCausalLM.from_pretrained` | 12 |
| `AutoTokenizer.from_pretrained` | 12 |
| `Qwen3_32B.from_pretrained` | 1 |
| Model-local imported `from_pretrained` | 13 |
| **Total** | **51** |

There are zero literal `trust_remote_code=True` call keywords and zero
`{"trust_remote_code": True}` forwarding dictionaries. The two remaining
keyword occurrences are the existing Llama 3 tokenizer and converted-state
loaders forwarding an explicit caller parameter. Both public parameters retain
their pre-existing `False` default; tests prove the default and explicit-opt-in
paths separately.

Transformers 5.12.1's local auto registries expose built-in Qwen2 and Qwen3
config, causal-LM, and tokenizer mappings. Qwen2.5 uses the Qwen2 architecture
mapping. The registry check performs no repository access or model load.

## Host characterization

`tests/host/test_remote_code_policy.py`:

- parses every production Python file and audits every `from_pretrained` call
  plus forwarded kwargs dictionaries;
- monkeypatches the tokenizer loaders in all three affected adaptors and
  observes that their default calls do not pass a true opt-in;
- monkeypatches Qwen2.5-72B's config and causal-LM loaders, proving the shared
  kwargs do not forward an opt-in;
- verifies the existing Llama 3 public opt-in remains explicit and defaults to
  false; and
- verifies the Transformers 5.12.1 built-in Qwen2/Qwen3 registrations.

The patched loaders return inert test doubles or deliberately stop execution.
No network request, model repository, checkpoint, weight conversion, or TT
device operation is reached.

## Verification

```bash
cd /localdev/gwang/tt_transformers

python tools/extract_model_packages.py --scope remote-code
python tools/check_import_boundaries.py src/tt_transformers
python qualification/tools/audit_test_taxonomy.py --json

PYTHONPATH=src:. \
  /tmp/gwang/tttv2-deps-py310.0JvRwC/venv/bin/python \
  -m pytest -q --confcutdir=tests/host tests/host/test_remote_code_policy.py

PYTHONPYCACHEPREFIX=/tmp/gwang/tt-transformers-remote-code-py310 \
  /tmp/gwang/tttv2-deps-py310.0JvRwC/venv/bin/python \
  -m compileall -q src tools tests/host/test_remote_code_policy.py qualification

PYTHONPYCACHEPREFIX=/tmp/gwang/tt-transformers-remote-code-py312 \
  /tmp/gwang/tttv2-deps-py312.yi8BCf/venv/bin/python \
  -m compileall -q src tools tests/host/test_remote_code_policy.py qualification
```

Observed results:

- pinned extractor policy scope: pass, 3 transforms;
- full static boundary checker: pass;
- explicit test taxonomy audit: pass;
- focused host suite: 7 passed in 3.97 seconds;
- CPython 3.10 and 3.12 full-tree compilation: pass.

## Evidence boundary

This establishes default argument policy, deterministic extraction, static
enforcement, built-in registry availability, and patched host call behavior. It
does not establish remote checkpoint compatibility, model correctness, weight
loading, numerical behavior, performance, or hardware support. No network,
checkpoint, hardware, or remote machine was used.
