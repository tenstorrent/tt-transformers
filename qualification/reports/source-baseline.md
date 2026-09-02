# Pinned source baseline

Source: `/localdev/gwang/tt-metal`

Branch: `gongyu/tttv2_bh_support`

Revision: `00748e6ac7b65f50e5c2af07f6e7c1c535c7f4c0`

Date: 2026-09-02 UTC

## Static results

- The forbidden public Wormhole/Blackhole/Architecture 1D config-wrapper scan
  produced no matches in Llama 3.1 8B, Llama 3.3 70B, or Qwen3 32B.
- The execution-path `is_blackhole()` and `get_arch_name()` scan produced no
  matches in the shared RMSNorm, LM head, MLP, or attention 1D modules.
- Syntax compilation of shared modules and the three BH model packages passed
  with bytecode redirected outside the read-only source tree.
- `git diff --check` passed.

## Baseline discrepancies and blockers

The historical WH-only zero-diff check against
`6de73d1ec382279f640f4b01c52076ca5737e06c` fails at the pinned migration
revision because all nine named WH-only package trees include later shared
runtime consolidation and documentation changes. The result is not treated as
same-SHA WH regression evidence.

The coordinating checkout has no `python_env`. Collection under `/opt/venv`
stops while loading the root conftest because that environment's incomplete
`ttnn` namespace lacks `ttnn.device`. No test or device fixture is reached and
no host pass is claimed.

Historical hardware results from other SHAs are context only. Standalone
hardware qualification will require destination code to be committed and
synchronized to every participating remote checkout under the authoritative
hardware manual before any device process starts.
