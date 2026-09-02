#!/bin/bash
set -u

run_t3000_tttv2_fast_unit_tests() {
  local fail=0
  pytest --tb=short tests/support tests/llm_runtime tests/models || fail=1
  pytest tests/modules/mlp/test_mlp_1d.py || fail=1
  pytest tests/modules/rmsnorm/test_rmsnorm_1d.py || fail=1
  pytest tests/modules/rope/test_rope_1d.py || fail=1
  pytest tests/modules/lm_head/test_lm_head_1d.py || fail=1
  pytest tests/modules/attention/test_attention_1d.py || fail=1
  pytest tests/modules/embedding/test_embedding_1d.py || fail=1
  pytest tests/modules/sampling/test_penalties_1d.py || fail=1
  pytest tests/modules/sampling/test_sampling_1d.py || fail=1
  return "${fail}"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  run_t3000_tttv2_fast_unit_tests
fi
