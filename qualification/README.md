# Repository qualification support

`qualification` contains repository-only utilities and data used by examples,
hardware policy, and release-readiness clients. It is not installed by the
`tt-transformers` wheel and production code must not import it.

## Contents

- `readiness/`: deterministic direct/vLLM readiness clients and schemas
- `schemas/`: support, capability, and hardware-evidence schemas
- `tools/`: hardware matrix, support/capability validators, reference-output
  generators, and vLLM readiness helpers

Active matrix and model configuration lives beside its consumers under
`tests/hardware` and `examples/config`. Test-only tensor comparison utilities
live in `tests/support`; repository example helpers live in `examples/common`.

## Common host-safe commands

```bash
python qualification/tools/audit_test_taxonomy.py
python qualification/tools/validate_support_docs.py
python qualification/tools/validate_bh_required_capabilities.py
python qualification/tools/run_hardware_matrix.py --validate
```

The commands above do not open a device. Hardware execution requires explicit
matrix attestation, inventory, `MESH_DEVICE`, and one-process-per-host
serialization. The runner never resets hardware automatically.

Raw execution logs belong in immutable CI/release artifacts, not in the source
tree. User-facing current results are summarized in `docs/validation.md`,
`SUPPORT.md`, and `examples/README.md`.
