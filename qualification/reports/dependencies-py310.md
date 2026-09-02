# Dependency-group qualification — CPython 3.10.19

Status: **passed for the refreshed isolated dependency groups and imports**

Machine-readable evidence: `qualification/reports/dependencies-py310.json`

## Verdict

The current base, examples, and test groups resolve together in the existing
isolated CPython 3.10.19 venv. `pip check` reports no broken
requirements, and all eleven direct distribution/import roots load.

Host collection established `pytz` as a direct test requirement and support
schema validation requires `jsonschema`. The observed bounded resolutions are
`pytz==2026.3.post1` and `jsonschema==4.26.0` on both interpreters.

This remains dependency/install/import evidence only: `tt-transformers` was
not built or installed and no source or hardware behavior was exercised.

## Environment

| Item | Value |
|---|---|
| Python | `3.10.19` |
| Venv | `/tmp/gwang/tttv2-deps-py310.0JvRwC/venv` |
| Platform | `Linux-4.18.0-553.el8_10.x86_64-x86_64-with-glibc2.35` |
| Indexes | PyPI primary; PyTorch CPU supplemental |
| `PYTHONPATH` during probe | unset |
| Checkout on `sys.path` | no |
| `tt-transformers` installed | no |
| `pip check` | no broken requirements |

## Requested groups

- Base: `ttnn==0.77.0`, `torch==2.11.0+cpu`, `loguru==0.6.0`
- Examples: `transformers==5.12.1`, `tqdm==4.66.3`
- Test: `pytest==9.0.3`, `pytest-cov==7.0.0`,
  `pytest-timeout==2.4.0`, `huggingface-hub>=0.30`,
  `jsonschema>=4.23,<5`, `pytz>=2024.1`, plus the example pins

## Direct imports and wheel identities

| Distribution | Resolved version | Wheel tag(s) | Import root |
|---|---|---|---|
| `ttnn` | `0.77.0` | `cp310-cp310-manylinux_2_34_x86_64` | `ttnn` |
| `torch` | `2.11.0+cpu` | `cp310-cp310-manylinux_2_28_x86_64` | `torch` |
| `loguru` | `0.6.0` | `py3-none-any` | `loguru` |
| `transformers` | `5.12.1` | `py3-none-any` | `transformers` |
| `tqdm` | `4.66.3` | `py3-none-any` | `tqdm` |
| `pytest` | `9.0.3` | `py3-none-any` | `pytest` |
| `pytest-cov` | `7.0.0` | `py3-none-any` | `pytest_cov` |
| `pytest-timeout` | `2.4.0` | `py3-none-any` | `pytest_timeout` |
| `huggingface-hub` | `1.29.0` | `py3-none-any` | `huggingface_hub` |
| `jsonschema` | `4.26.0` | `py3-none-any` | `jsonschema` |
| `pytz` | `2026.3.post1` | `py2-none-any`, `py3-none-any` | `pytz` |

## Exact transitive freeze (65 distributions)

```text
annotated-doc==0.0.5
anyio==4.14.2
attrs==26.1.0
certifi==2026.7.22
click==8.5.0
contourpy==1.3.2
coverage==7.16.0
cycler==0.12.1
exceptiongroup==1.3.1
filelock==3.32.5
fonttools==4.64.0
fsspec==2026.7.0
graphviz==0.21
h11==0.16.0
hf-xet==1.6.0
httpcore==1.0.9
httpx==0.28.1
huggingface_hub==1.29.0
idna==3.19
iniconfig==2.3.0
Jinja2==3.1.6
jsonschema==4.26.0
jsonschema-specifications==2025.9.1
kiwisolver==1.5.1
loguru==0.6.0
markdown-it-py==4.2.0
MarkupSafe==3.0.3
matplotlib==3.10.9
mdurl==0.1.2
ml_dtypes==0.5.4
mpmath==1.3.0
networkx==3.4.2
numpy==1.26.4
packaging==26.3
pandas==2.3.3
pillow==12.3.0
pip==23.0.1
pluggy==1.6.0
Pygments==2.21.0
pyparsing==3.3.2
pytest==9.0.3
pytest-cov==7.0.0
pytest-timeout==2.4.0
python-dateutil==2.9.0.post0
pytz==2026.3.post1
PyYAML==6.0.3
referencing==0.37.0
regex==2026.9.3
rich==15.0.0
rpds-py==0.30.0
safetensors==0.8.0
seaborn==0.13.2
setuptools==79.0.1
shellingham==1.5.4
six==1.17.0
sympy==1.14.0
tokenizers==0.22.2
tomli==2.4.1
torch==2.11.0+cpu
tqdm==4.66.3
transformers==5.12.1
ttnn==0.77.0
typer==0.27.2
typing_extensions==4.16.0
tzdata==2026.3
```

## Resolver note

A first discarded CPython 3.10 attempt made the CPU index primary and failed
while resolving normalized dependency metadata. The successful environment
uses PyPI primary plus the CPU index as supplemental. This is an index-ordering
issue, not a conflict in the qualified tuple.

## Interpreter differences

Direct requirements and their resolved versions are identical. Unconstrained
transitives differ by interpreter:

- CPython 3.10 selects `contourpy==1.3.2`, `matplotlib==3.10.9`,
  `networkx==3.4.2`, `pandas==2.3.3`, `rpds-py==0.30.0`,
  and compatibility packages `exceptiongroup`, `tomli`, and `tzdata`.
- CPython 3.12 selects `contourpy==1.3.3`, `matplotlib==3.11.1`,
  `networkx==3.6.1`, `pandas==3.0.5`, and
  `rpds-py==2026.6.3`.

Both resolve `pytz==2026.3.post1`; neither difference set is a conflict.

## Reproduction

```bash
PYTHON=/usr/local/share/uv/cpython-3.10.19-linux-x86_64-gnu/bin/python3.10
QUAL_DIR=/tmp/gwang/tttv2-deps-py310.0JvRwC

"$PYTHON" -m venv "$QUAL_DIR/venv"
"$QUAL_DIR/venv/bin/python" -m pip install \
  --index-url https://pypi.org/simple \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  ttnn==0.77.0 torch==2.11.0+cpu loguru==0.6.0 \
  transformers==5.12.1 tqdm==4.66.3 \
  pytest==9.0.3 pytest-cov==7.0.0 pytest-timeout==2.4.0 \
  'huggingface-hub>=0.30' 'jsonschema>=4.23,<5' 'pytz>=2024.1'
"$QUAL_DIR/venv/bin/python" -m pip check
"$QUAL_DIR/venv/bin/python" -m pip freeze --all
```

## Evidence boundary

No repository source, tests, examples, `pyproject.toml`, main log/report, or
hardware state was changed by this refresh.
