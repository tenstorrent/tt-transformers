# Qwen3.8 operation library

Provenance: this code comes from the tt-metal Qwen3.6 bring-up.

This model-scoped library supplies three Blackhole device operations
used by Qwen3.8 that are not available in stock TTNN 0.79.0:

- `gdn_decode_step`
- `gdn_spec_step`
- `qkv_causal_conv1d_silu_tile`

The Python launchers use `ttnn.generic_op`. The packaged kernels are copied
byte-for-byte from the Unified V1 tt-metal revision
`547841b0109bf57f020355287841cd4f38eec98b` and are also byte-identical to
tt-metal `d1ed117ac894`.

`qkv_causal_conv1d_silu_tile` runs on stock `ttnn==0.79.0`.

`gdn_decode_step` and `gdn_spec_step` use `KernelDescriptor.BuildOptLevel` to
preserve the native `Os` data-movement and `O2` fused-compute settings.
`ttnn` 0.79.0 does not expose that binding, so these two operations raise a
`RuntimeError` on 0.79.0. They are optional and off by default; the default
Qwen3.8 path does not call them. Device qualification covers
binary fit, bit-exact
output/state/window behavior, trace replay, address rebinding, signed zero,
and warmed latency against the native V1 operations.

No TTNN or tt-metal runtime binaries are included in this library.
