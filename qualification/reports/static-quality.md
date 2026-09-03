# Static-quality baseline

Status: failing extraction baseline; cleanup is deferred to Phase 9 after
hardware parity, exactly as required by the migration plan.

The audit used isolated declared-floor tools: Ruff 0.11.0, Black 25.1.0, and
mypy 1.15.0. No automatic fixes or formatting were applied because that would
mix broad mechanical churn with the pre-hardware behavior baseline and break
the reviewed provenance normalizations.

| Gate | Current result |
| --- | --- |
| Ruff lint | 2,158 findings; legacy CLI-comparable count reports 628 fixable, while JSON classifies 505 safe and 154 unsafe fix applications |
| Ruff format | 167 files would change; 204 already formatted |
| mypy | 489 errors in 99 of 132 package files using the qualified dependency environment |
| Black | did not complete the 132-file package check within the bounded 30-second run |

Ruff's largest classes are 1,447 line-length findings, 226 import-order
findings, 398 Python-modernization findings, and 55 deprecated-import findings.
Seventeen undefined-name findings require real review and must not be folded
into an indiscriminate auto-fix.

The legacy CLI-comparable fixable count and the JSON applicability counts use
different Ruff aggregations, so `628` is not expected to equal `505 + 154`.
Both views are retained explicitly instead of conflating them.

The mypy run used the qualified Python 3.10 dependency interpreter so Torch and
other installed packages were visible. TTNN 0.77 lacks a `py.typed` marker or
complete stubs; those import-untyped diagnostics are external typing feedback,
while the remaining Optional, TypedDict, assignment, and lifecycle-owner
diagnostics are Phase 9 package debt.

`static-quality-baseline.json` records exact counts, tool versions, and the
current 132-file source digest. Validate it with:

```bash
python -B qualification/tools/validate_static_quality_baseline.py
```

This baseline prevents the debt from being described as green; it is not a
waiver or a replacement for clean lint/format/type gates. Phase 9 must select
one formatter, fix reviewed categories in independently testable changes,
update extraction normalizers where required, and make zero-regression quality
checks mandatory only after parity is established.
