# E05 — Does the residual survive more than one layer?

**Hardware: one 32-device Blackhole Galaxy. Specification, not a patch** — see *Why there is no
patch* below. **Depends on [E04](../E04-module-capability-gates/).**

**The most dangerous correctness item in the port.** A single-layer module test passes and an
80-layer model is quietly wrong.

---

## Why it exists

The non-fused distributed RMSNorm **does not write the residual sum back in place**. So relying
on the fused path on Blackhole *"silently drops each layer's `ff_out` from the residual
stream"*, and the reference's own words for its visibility are *"only visible across >1 layer"*.

The reference's answer is `unfuse_res_add`: do the add explicitly, and move the residual dtype to
bf16 with bf8 norm output on the no-prefetcher path. `GalaxyCapabilities.has_fused_residual_norm`
is already `False` on Blackhole for this reason — what is untested is whether the explicit add
actually happens and produces the right numbers.

## Hypothesis

`RMSNorm2D` on a Blackhole Galaxy mesh, run over **at least two layers**, keeps the residual
stream: layer *n*'s `ff_out` is present in layer *n+1*'s input.

## Why there is no patch

The shape of the fix depends on what [E04](../E04-module-capability-gates/) reveals about where
`_resolve_2d_config` stops on a Blackhole mesh. Writing a patch now would be guessing at the
call site, and a plausible-looking patch for an unmeasured failure is worse than none — it
invites someone to apply it and believe the result.

What is *not* a guess is the test, and that is what this specifies.

## What to build

A Blackhole Galaxy RMSNorm2D suite, modelled on
`tests/modules/rmsnorm/test_rmsnorm_2d_wh_galaxy.py`, with three properties the Wormhole suite
does not need:

1. **At least two layers, from the start.** One layer cannot see this defect. This is the single
   most important property of the test and the reason it exists.
2. **Compared against a CPU reference**, not against another TT path. The provenance record
   already retired two ported suites for comparing an executor against a wrapper of the same
   model object; do not repeat that shape here. `torch.nn.RMSNorm` is what the Wormhole suite
   uses and what the 1D suites compare against.
3. **Byte-identical across three fresh processes.** The fused-RMSNorm stats CB binds to the
   first core of the norm input shard grid and aliases whatever the allocator left there — PCC of
   0.0977 / 0.1394 / 0.1701 / 0.9999 has been observed across processes *on an unchanged test*.
   Three PCC passes prove much less than one byte-identical triple.

The dtype recipe to use, from the reference's no-prefetcher path: **bf16 residual, bf8 norm
output**. That is a recipe, not a tuning choice.

## Run

```bash
# One node id per process.
MESH_DEVICE=BHGLX pytest tests/modules/rmsnorm/test_rmsnorm_2d_bh_galaxy.py::<multi_layer_id> -sv
# Three times, in fresh processes, and diff the saved outputs with torch.equal.
```

## What each outcome means

**Residual preserved across layers, byte-identical across processes.** The dtype recipe and the
explicit add are right. This is the single most valuable green result in Blackhole bring-up,
because it is the defect that would otherwise be found at model scale, where the search space is
80 layers wide.

**Layer 1 correct, layer 2 wrong by roughly one `ff_out`.** The exact predicted failure. The
explicit residual add is not happening. Check that the fused path is not being selected —
`has_fused_residual_norm` is `False` on the Blackhole descriptor, so a fused path still running
means the capability is not reaching the module, which is an E04 wiring defect rather than a
numerics one.

**PCC varies across processes.** Not this defect — that is the stats-CB aliasing trap. It means
the stats buffer is not on the first core of the norm input shard grid. `RMSNorm2D`'s own
`_require_fused_stats_placement` rejects a mismatch, so if it is passing and the numbers still
swing, the placement rule itself is wrong on this architecture.

**Everything passes at bf16 but not at the production dtype.** Record both. A dtype-sensitive
residual is a real finding and it changes what the model-level recipe can use.

---

## Result

*Not run.*
