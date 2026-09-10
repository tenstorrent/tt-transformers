# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Model-owned Llama-3.3-70B Galaxy `(8, 4)` execution composition and cleanup root.

This is the Milestone C executor for the reconstructed 2D tensor model. It composes
the common runtime (`src/tt_transformers/llm_runtime`) exactly as
`src/tt_transformers/models/llama33_70b/executor.py` composes it for the 1D model, and it
is the resource and cleanup root for one execution lane.

Two things make this executor longer than its 1D sibling, and both of them are
*model-owned adaptation* rather than new mechanics:

**1. The runtime's model contract is 1D-shaped, and the Galaxy graph is not.**
`PrefillRuntime` and `DecodeRuntime` call

```text
model.embed_prefill(tokens)                       model.embed_decode(tokens)
model.prefill_forward(x, rot_mats, user_id=…,     model.decode_forward(x, positions,
                      get_last_token=-1, …)                            rot_mats, page_table=…)
model.post_process_prefill_output(hidden, last)   model.gather_and_untilize_logits(logits)
model.rope_setup.{load_device_weights, cos_matrix, get_rot_idxs, decode_forward}
model.prepare_prefill_rot_mats(position_indices)
```

`Llama33_70BGalaxyTransformer2D` exposes a different, mode-explicit contract
(`activate(mode)`, `prefill_forward(..., sequence_length=…, user_ids=…)`,
`project_prefill_logits`, `prepare_prefill_rot_mats(start_pos, seq_len)`). The
adaptation lives here, in the model package, so that `llm_runtime` keeps zero
Galaxy, Llama, 2D-mesh or `(8, 4)` knowledge. See `_GalaxyRuntimeModelView`.

**2. Two placements differ from the 1D convention, and both were qualified at
Milestone B in `GalaxyDirectRunner`.**

*Decode positions and the decode page table.* `DecodeRuntime._prepare_inputs_host`
maps both with `ShardTensor2dMesh(dims=(None, None))` — replicated. The Galaxy
decode graph attends to one mesh column's users on each device, so
`paged_update_cache` and the paged decode SDPA need the device-local table to carry
exactly `users_per_column` rows and the positions to carry that column's users:
`dims=(None, 0)`. A replicated device tensor cannot be turned into a
column-sharded one on device, because slicing a different range per device is not
expressible in one SPMD op. So the executor stages the Galaxy-placed pair at the
operation boundary, from the host request it was handed, and the view consumes
those instead of the runtime's. The runtime's own two small tensors are still
allocated and released by the runtime; nothing in `llm_runtime` changes.

*Logits composition.* `result_collector.concat_host_output` and
`decode._concat_host_output` concatenate mesh **columns** along the vocabulary
axis. On Galaxy the vocabulary is sharded over the eight mesh **rows** and
replicated over the four columns; composing it along the wrong axis is finding
D-B23, and `collectives.compose_galaxy_logits` is the qualified composition that
carries the measurement. This view composes with it. Decode returns the composed
host tensor — both runtime readers accept `torch.Tensor` and pass it straight
through — while prefill re-stages the composed row as a replicated device tensor,
because the runtime untilizes and slices prefill logits on device before reading
them. A device-side all-gather over the mesh-row axis is the trace-compatible
successor and needs a new persistent CCL resource in `galaxy/plans.py`; that is
not this job's to add.

**Out of scope here, deliberately.** No `generator.py`, no vLLM adapter, no lane
group. Batched (concat-32) prefill is out of Milestone C: this executor resolves
`supports_batched_prefill=False` and `post_process_batched_prefill_output` raises.
Tracing is `c-trace`'s job — the trace collaborators are constructed exactly as the
1D executor constructs them, over this same eager executor, and nothing here reads
`trace_mode` in a hot path.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, replace
from typing import Any

import torch
import ttnn

from tt_transformers.llm_runtime.config import PagedKVCacheConfig, PageTableLayout, TraceConfig, WarmupConfig
from tt_transformers.llm_runtime.decode import DecodeRuntime, DecodeRuntimeConfig
from tt_transformers.llm_runtime.execution import EagerExecutor, TracedExecutor
from tt_transformers.llm_runtime.output_reader import OutputReader
from tt_transformers.llm_runtime.paged_kv_cache import PagedKVCacheManager
from tt_transformers.llm_runtime.prefill.config import PrefillRuntimeConfig
from tt_transformers.llm_runtime.prefill.runtime import PrefillRuntime
from tt_transformers.llm_runtime.program_compiler import ProgramCompiler
from tt_transformers.llm_runtime.tensor_resources import attach_cleanup_failures
from tt_transformers.llm_runtime.trace_compiler import TraceCompiler
from tt_transformers.llm_runtime.warmup import WarmupCoordinator, WarmupCoordinatorConfig
from tt_transformers.models.galaxy.collectives import (
    compose_galaxy_logits,
    compose_galaxy_sampled_tokens,
    deallocate_if_allocated,
)
from tt_transformers.models.galaxy.recipes import GALAXY_MESH_SHAPE
from tt_transformers.models.llama33_70b_galaxy.model import Llama33_70BGalaxyTransformer2D
from tt_transformers.sampling.sampling_params import SamplingParams

#: One tile row block. The runtime's prefill readback addresses the last token by
#: its row inside this block, so a projected prefill result must present 32 rows.
_TILE_SIZE = 32


@dataclass(frozen=True)
class Llama33_70BGalaxyExecutorConfig:
    """Immutable aggregate policy paired with one model-owned Galaxy executor."""

    trace: TraceConfig
    warmup: WarmupConfig
    paged_kv_cache: PagedKVCacheConfig
    device_sampling_enabled: bool = False
    #: Milestone C prefills one row at a time. This resolves the runtime's
    #: batched-prefill capability to False, so the planner never buckets rows
    #: into a concat-32 wave. It is a policy value, not a fallback switch.
    sequential_prefill_only: bool = True

    def __post_init__(self) -> None:
        nested_configs = (
            ("trace", self.trace, TraceConfig),
            ("warmup", self.warmup, WarmupConfig),
            ("paged_kv_cache", self.paged_kv_cache, PagedKVCacheConfig),
        )
        for name, value, expected_type in nested_configs:
            if type(value) is not expected_type:
                raise TypeError(f"{name} must be exactly {expected_type.__name__}")
        for name in ("device_sampling_enabled", "sequential_prefill_only"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be bool")


class _GalaxyRopeView:
    """Present `RotarySetup2D` through the runtime's rope-setup contract.

    The runtime reads `cos_matrix.shape[2]` for the rotary capacity check, stages
    decode rotary indices with `get_rot_idxs`, and turns them into cos/sin with
    `decode_forward`. `RotarySetup2D` owns all three under those exact names, so
    this view is a pass-through.

    It did not used to be. At the port's merge base the runtime asked for the
    cos/sin step as `get_rot_mats`, and this view existed mainly to rename
    `RotarySetup2D.decode_forward` to it. Upstream has since renamed the runtime's
    expectation to `decode_forward` -- the 2D module's own name all along -- so the
    rename is gone. Keeping only the old name here raised `AttributeError:
    '_GalaxyRopeView' object has no attribute 'decode_forward'` on the first decode
    step of every Galaxy model.
    """

    def __init__(self, rope: Any):
        self._rope = rope

    @property
    def cos_matrix(self) -> Any:
        return self._rope.cos_matrix

    @property
    def sin_matrix(self) -> Any:
        return self._rope.sin_matrix

    @property
    def config(self) -> Any:
        return self._rope.config

    def load_device_weights(self) -> None:
        self._rope.load_device_weights()

    def get_rot_idxs(self, position_idxs: Any, on_host: bool = False) -> Any:
        return self._rope.get_rot_idxs(position_idxs, on_host=on_host)

    def decode_forward(self, rot_idxs: Any) -> list[Any]:
        return self._rope.decode_forward(rot_idxs)


class _GalaxySamplingConfigView:
    """Add the runtime's required `allow_force_argmax` to `Sampling2DConfig`.

    The runtime requires the attribute to exist and to be a bool - it raises
    `TypeError: model.sampling.config.allow_force_argmax must be bool` at
    `llm_runtime/prefill/config.py:173` and reads it again in
    `DecodeRuntimeConfig.resolve` - and `Sampling2DConfig` does not carry it,
    because `Sampling2D` has no separate argmax program: a greedy row is encoded
    as ``top_k=1, top_p=0`` through the same chain
    (`sampling_2d.py::_update_call_buffers`). `False` is therefore the accurate
    answer as well as the one that keeps the runtime on the single chain this
    sampler implements. **This is the reason no Galaxy executor had ever been
    constructed with `device_sampling_enabled=True`:** without it, construction
    fails before any device work.

    Every other attribute is the sampler's own, read through.
    """

    allow_force_argmax = False

    def __init__(self, config: Any):
        self._config = config

    def __getattr__(self, name: str) -> Any:
        return getattr(self._config, name)


class _GalaxySamplingView:
    """Present `Sampling2D` through the runtime's 1D sampling contract.

    Two things differ between `Sampling1D`, which `DecodeRuntime` was written
    against, and `Sampling2D`, which this model owns. Neither is a behaviour
    difference; both are currency.

    **Keyword names.** `DecodeRuntime._sample_device` calls
    ``decode_forward(logits, k=…, p=…, temp=…)``; `Sampling2D.decode_forward`
    takes keyword-only ``top_k=``, ``top_p=``, ``temperature=``. Without this view
    a sampled decode through this executor raises `TypeError` before it reaches
    silicon, which is why no Milestone C evidence for device-sampled decode
    *through an executor* existed before this job.

    **Where the per-slot controls live.** `Sampling1D` reads K/P/T as device
    tensors the runtime stages per call. `Sampling2D` owns its own persistent
    control buffers and refreshes them from **host** values inside
    `prepare_call` - four host-to-device writes plus a fresh random seed
    (`sampling_2d.py::_update_call_buffers`). A trace capture region cannot
    contain a host-to-device write, and a replay would never repeat one, so the
    refresh is hoisted to the operation boundary
    (`_refresh_device_sampling`) and the captured body runs with
    ``update_buffers=False``. The runtime's own K/P/T device tensors are still
    allocated and released by the runtime; they are simply not the currency this
    sampler reads - exactly as with the Galaxy-placed decode positions and page
    table.

    The value inversion is deliberate too. `format_sampling_params` inverts
    temperature for `Sampling1D` and `Sampling2D._update_call_buffers` inverts it
    again for itself, so this view is refreshed from the caller's **raw**
    `SamplingParams`, never from the runtime's already-formatted values.
    """

    def __init__(self, sampling: Any):
        self._sampling = sampling
        self._config = _GalaxySamplingConfigView(sampling.config)
        self._call: tuple[Any, Any, Any, Any] | None = None

    @property
    def config(self) -> Any:
        return self._config

    @property
    def sampling(self) -> Any:
        return self._sampling

    def refresh(self, call: tuple[Any, Any, Any, Any]) -> None:
        """Write one request's controls into the sampler's own buffers."""

        top_k, top_p, temperature, seed = call
        self._sampling.prepare_call(
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            seed=seed,
            forced_argmax=False,
            slot_ids=None,
            update_buffers=True,
        )
        self._call = call

    def clear(self) -> None:
        self._call = None

    def decode_forward(self, logits: Any, *, k: Any = None, p: Any = None, temp: Any = None, tt_out_tok: Any = None):
        """Run the sampling chain on already-refreshed controls."""

        if self._call is None:
            raise RuntimeError("device sampling controls were not refreshed at the operation boundary")
        top_k, top_p, temperature, seed = self._call
        return self._sampling.decode_forward(
            logits,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            seed=seed,
            forced_argmax=False,
            slot_ids=None,
            tt_out_tok=tt_out_tok,
            update_buffers=False,
        )


class _GalaxyRuntimeModelView:
    """Adapt the Galaxy 2D graph to the common runtime's model contract.

    Every method here delegates to `Llama33_70BGalaxyTransformer2D`; the view owns
    no device state except the transients it creates and releases inside one call.
    Mode activation is **not** done here — it is an operation-boundary action the
    executor performs before delegating, so no module hot path carries a mode
    branch.
    """

    def __init__(self, model: Llama33_70BGalaxyTransformer2D, *, device_sampling_enabled: bool):
        self._model = model
        self.rope_setup = _GalaxyRopeView(model.rope_setup)
        self._device_sampling_enabled = bool(device_sampling_enabled)
        self._sampling_view = (
            _GalaxySamplingView(model.sampling)
            if self._device_sampling_enabled and model.sampling is not None
            else None
        )
        #: Galaxy-placed decode inputs staged by the executor for one call.
        self._decode_positions: Any = None
        self._decode_page_table: Any = None
        #: Row inside the final tile block that a cached/chunked request's last
        #: token occupies, bound by the executor from the public request.
        self._prefill_last_row: int | None = None
        #: Whether this view launches the persistent prefetch sender itself, at
        #: the top of the decode graph body. See `set_sender_launch_in_graph`.
        self._sender_launch_in_graph = False
        #: The owner's answer to "is a trace active?", which decides whether the
        #: prefill hidden state handed to `post_process_prefill_output` may be
        #: released. See `set_trace_active_probe`.
        self._trace_active_probe: Any = None

    # -- identity the runtime validates -----------------------------------

    @property
    def model(self) -> Llama33_70BGalaxyTransformer2D:
        return self._model

    @property
    def config(self) -> Any:
        return self._model.config

    @property
    def vocab_size(self) -> int:
        return int(self._model.vocab_size)

    @property
    def num_devices(self) -> int:
        return int(self._model.num_devices)

    @property
    def mesh_device(self) -> Any:
        return self._model.mesh_device

    @property
    def sampling(self) -> Any:
        # Reported only when this lane resolved device sampling. The runtime
        # validates `allow_force_argmax` against this attribute, so presenting a
        # sampler that the executor will not use would misdescribe the lane. It is
        # reported through `_GalaxySamplingView` because `Sampling2D` does not
        # answer to the runtime's 1D keyword names.
        return self._sampling_view

    def iter_executor_named_modules(self) -> Iterator[tuple[str, Any]]:
        return self._model.iter_executor_named_modules()

    # -- decode input staging owned by the model --------------------------

    def bind_decode_inputs(self, *, positions: Any, page_table: Any) -> None:
        """Install the Galaxy-placed decode pair for the next decode body."""

        self._decode_positions = positions
        self._decode_page_table = page_table

    def clear_decode_inputs(self) -> None:
        self._decode_positions = None
        self._decode_page_table = None

    def set_trace_active_probe(self, probe: Any) -> None:
        """Install the owner's answer to "is a trace active?".

        It decides one thing: whether `post_process_prefill_output` may release
        the hidden state the runtime handed it. **With a trace active it must
        not** - `PrefillTraceRuntime.finish` names that tensor
        `trace_owned_hidden_output` and every replay rewrites it, so freeing it
        kills the *next* replay (`logs/a1_L_decode32.log`, a segmentation fault
        in `rms_norm_pre_all_gather` on the second of thirty-two prefills).

        **Without one it must**, and that is not a preference either: releasing
        it as soon as the final norm has consumed it is what every caller had
        before tracing, and holding it instead moves the peak - which on this
        model moves an allocation, which is the Llama L1 address-clash class.
        `logs/n3_L_exec_prefill128.log` is that regression measured: the
        executor's own prefill claim, argmax 1709 against a freshly recomputed
        reference's 8703, with logits of magnitude 2.9e14.

        `None` means "no trace can be active", which is every caller before
        tracing existed.
        """

        if probe is not None and not callable(probe):
            raise TypeError("trace active probe must be callable or None")
        self._trace_active_probe = probe

    def _trace_is_active(self) -> bool:
        probe = self._trace_active_probe
        return bool(probe()) if probe is not None else False

    def set_sender_launch_in_graph(self, enabled: bool) -> None:
        """Launch the persistent prefetch sender inside the decode body.

        The executor turns this on together with
        `GalaxyResources.set_prefetch_sender_launch_site("graph")`, so exactly
        one of the two sites launches. See that method for why a traced decode
        needs the launch recorded rather than dispatched at the boundary.
        """

        self._sender_launch_in_graph = bool(enabled)

    def bind_prefill_last_row(self, row: int | None) -> None:
        """Install the last token's row inside its tile block for this call.

        The runtime hands the cached/chunked prefill body the tile-block bounds as
        device tensors and the row index only when device sampling is on, but the
        readback always addresses row ``(last_token - cached_tokens) % 32``. That
        value is derivable on the host from the public request, which the executor
        has, so the executor binds it at the operation boundary rather than the
        view reading a device tensor back.
        """

        self._prefill_last_row = None if row is None else int(row)

    def clear_prefill_last_row(self) -> None:
        self._prefill_last_row = None

    # -- staging helpers the runtime calls --------------------------------

    def embed_prefill(self, tokens: Any) -> Any:
        return self._model.embed_prefill(_as_token_row(tokens))

    def embed_decode(self, tokens: Any) -> Any:
        return self._model.embed_decode(_as_token_row(tokens))

    def prepare_prefill_rot_mats(self, position_indices: Any) -> list[Any]:
        """Return prefill cos/sin for a runtime-staged position-index tensor.

        `RotarySetup2D.prefill_forward(start_pos, seq_len)` slices its tilized
        table copy and is the path Milestone B qualified; the runtime supplies the
        positions as a *device* tensor instead, so the start position is read back
        from its first element. The runtime builds that tensor as
        ``arange(start_pos, start_pos + seq_len)``, so one element plus the shape
        determine the range exactly.

        **A device-side gather was tried first and is not equivalent.**
        `ttnn.embedding` over the same table needs no readback and would be
        trace-compatible, but it moved the KV that prefill writes: with the gather,
        `logs/i4_pagedkv_l1.log` reports the first layer's K at PCC 0.907 against
        the same request through `GalaxyDirectRunner` while the logits agreed at
        0.9994 — and K is the one tensor of the pair that passes through RoPE.
        `scratch/test_rope_gather_probe.py` measures the two directly. Until that
        is understood, this executor uses the qualified slice: a four-byte read per
        prefill step is a smaller price than an unqualified rotary.
        """

        # Through the model's own API, which is what `GalaxyDirectRunner` calls,
        # so the two paths cannot drift apart in how they build cos/sin.
        return self._model.prepare_prefill_rot_mats(
            _first_position(position_indices),
            int(position_indices.shape[-1]),
        )

    # -- graph bodies -----------------------------------------------------

    def _validate_chunk_start(self, chunk_start: int) -> None:
        """Apply the recipe's chunk-alignment guard to a host chunk start.

        `Attention2D._validate_prefill` only checks `metadata.chunk_start`, and
        this view hands the module the device tensor instead so that the eager and
        traced bodies are one program. The check therefore happens here, on the
        host value the caller did supply, rather than being lost.
        """

        alignment = int(getattr(getattr(self._model, "geometry", None), "chunk_alignment", 0) or 0)
        if chunk_start < 0 or (alignment and chunk_start % alignment):
            raise ValueError(
                f"chunk_start {chunk_start} must be non-negative and aligned to chunk_alignment {alignment}"
            )

    def prefill_forward(
        self,
        x_embed: Any,
        rot_mats: Any,
        *,
        user_id: Any = 0,
        page_table: Any = None,
        chunk_page_table: Any = None,
        chunk_start_idx: int | None = None,
        get_last_token: int = -1,
        batch_size: int | None = None,
        chunk_start_idx_tensor: Any = None,
        last_token_slice: Any = None,
        last_token_index: Any = None,
    ) -> Any:
        """Run one prefill invocation and return hidden state or a logits block.

        The runtime uses two shapes of prefill body. A regular single request asks
        for the hidden state and post-processes it in a separate call; a
        cached/chunked request passes `last_token_slice` and expects the body to
        return the last-token logits itself. Both are served here.
        """

        if int(get_last_token) != -1:
            raise ValueError("the Galaxy prefill body extracts its last token after the graph, not inside it")
        user_ids = tuple(int(value) for value in user_id) if isinstance(user_id, (list, tuple)) else (int(user_id),)
        rows = int(batch_size) if batch_size is not None else len(user_ids)
        if rows != 1 or len(user_ids) != 1:
            raise NotImplementedError(
                "batched (concat-32) prefill is out of Milestone C scope; this executor prefills one row at a time"
            )
        tokens = int(x_embed.shape[-2])
        if tokens % rows:
            raise ValueError(f"{tokens} prefill tokens do not divide into {rows} rows")
        sequence_length = tokens // rows
        chunked = chunk_start_idx is not None or chunk_start_idx_tensor is not None
        # The module accepts either the host start or its device tensor, never
        # both (`attention_2d.py:902`), and **the device tensor is what both paths
        # use.** That is not a preference: the two forms are two different
        # `ttnn.transformer.chunked_scaled_dot_product_attention` programs, so an
        # eager warmup that compiled the host form leaves the trace capture body
        # asking for a program the ttnn cache has never seen -
        #   TT_FATAL @ mesh_workload.cpp:153 !is_capturing_trace
        #   Cannot load new binaries during trace capture.
        # (`logs/t1_l1_traced_prefill.log`). A captured graph cannot carry a host
        # start, so unifying has to go this way round; TTTv1 does the same, giving
        # `ttnn_prefill_forward` a device `chunk_start_idx` for its compile run
        # and its capture run alike (`generator.py:1120-1170`).
        #
        # The host value is still the one the recipe's chunk alignment is stated
        # against, so the guard `_validate_prefill` would have applied to it is
        # applied here instead of being dropped with it.
        if chunk_start_idx is not None:
            self._validate_chunk_start(int(chunk_start_idx))
        hidden = self._model.prefill_forward(
            x_embed,
            list(rot_mats),
            sequence_length=sequence_length,
            user_ids=user_ids,
            page_table=page_table,
            chunk_page_table=chunk_page_table,
            chunk_start=None if chunk_start_idx_tensor is not None else chunk_start_idx,
            chunk_start_tensor=chunk_start_idx_tensor,
            prefix_user_id=user_ids[0] if chunked else None,
            return_hidden_state=True,
        )
        if last_token_slice is None:
            return hidden
        if self._prefill_last_row is None:
            raise RuntimeError(
                "a cached or chunked Galaxy prefill needs its last-token row bound at the operation boundary"
            )
        try:
            return self._project_tile_block(
                hidden,
                _tile_block_start(last_token_slice),
                sequence_length,
                row=self._prefill_last_row,
            )
        finally:
            deallocate_if_allocated(hidden)

    def post_process_prefill_output(
        self,
        hidden: Any,
        last_token: int,
        *,
        last_token_slice: Any = None,
        last_token_index: Any = None,
    ) -> Any:
        """Normalize, project and present one prefill row's logits.

        Returns a replicated `[1, 1, 32, vocab_size]` TILE tensor whose row
        `last_token % 32` holds the requested token's logits. The runtime
        untilizes it, slices that row, reads it, and composes it with the
        column-concatenating reader — which is correct for a replicated
        full-vocabulary tensor and is not correct for the row-sharded LM head
        output. Composition therefore happens here, with the qualified
        `compose_galaxy_logits`.
        """

        last_token = int(last_token)
        sequence_length = int(hidden.shape[-2])
        if not 0 <= last_token < sequence_length:
            raise ValueError(f"prefill last token {last_token} is outside {sequence_length} tokens")
        row = last_token % _TILE_SIZE
        logits = None
        try:
            (logits,) = self._model.project_prefill_logits(
                hidden,
                rows=1,
                sequence_length=sequence_length,
                token_indices=(last_token,),
                # The runtime owns this hidden state. Under tracing it *is* the
                # captured graph's persistent output buffer, reused by every
                # replay, and releasing it kills the next one; without a trace
                # the historical early release is what the memory layout was
                # qualified against. See `set_trace_active_probe`.
                release_hidden=not self._trace_is_active(),
            )
            composed = compose_galaxy_logits(
                logits,
                mesh_device=self._model.mesh_device,
                vocab_size=self.vocab_size,
            )
        finally:
            deallocate_if_allocated(logits)
        return self._present_logits_block(composed[:1, :], row)

    def post_process_batched_prefill_output(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(
            "batched (concat-32) prefill extraction is out of Milestone C scope for the Galaxy executor"
        )

    def decode_forward(self, x_embed: Any, current_pos: Any, rot_mats: Any, page_table: Any = None) -> Any:
        """Run one physical-batch-32 decode step on the Galaxy-placed inputs.

        `current_pos` and `page_table` arrive replicated from the runtime's
        staging and are ignored in favour of the column-sharded pair the executor
        staged; see this module's docstring.
        """

        if self._decode_positions is None or self._decode_page_table is None:
            raise RuntimeError("Galaxy decode inputs were not staged for this call")
        if self._sender_launch_in_graph:
            # TTTv1 launches `ttnn.dram_prefetcher` here, at the top of the
            # decode body, and not at the operation boundary
            # (`models/demos/llama3_70b_galaxy/tt/llama_model.py:790-808`). A
            # capture region records this call, so every replay dispatches the
            # sender in the recorded order against the matmuls that consume its
            # global-CB credits; a boundary launch is dispatched *for real* while
            # the graph that would consume it is only being recorded.
            self._model.resources.launch_prefetch_sender()
        return self._model.decode_forward(
            x_embed,
            self._decode_positions,
            rot_mats,
            self._decode_page_table,
        )

    def gather_and_untilize_logits(self, logits: Any) -> Any:
        """Return the decode logits still on device, for the runtime to read.

        **This used to compose on host here, and it cannot.** The method is called
        from inside `DecodeRuntime._run_body`, and that body is the exact region
        `TraceCompiler.capture_all` records between `ttnn.begin_trace_capture` and
        `ttnn.end_trace_capture`. A `ttnn.to_torch` there is a queue read inside a
        capture region; and even where such a read is tolerated it is never
        *replayed*, because `TraceCompiler.replay` hands the caller back
        `artifact.outputs` — so every replay would return the one host tensor
        composed at capture time, and eager-vs-traced parity would be measuring a
        stale constant.

        The composition therefore moves one step later, behind the runtime's own
        output read, where it runs identically for the eager and the traced path:
        see `compose_decode_logits`. TTTv1 splits the graph at the same seam
        (`models/demos/llama3_70b_galaxy/tt/generator.py`, `on_device_logits`).
        No number changes: the same tensor is composed by the same
        `compose_galaxy_logits`.
        """

        return logits

    def compose_decode_logits(self, value: Any) -> torch.Tensor:
        """Compose row-sharded decode logits into `[1, 1, 32, vocab]` on host.

        `DecodeRuntime._convert_logits` probes the model for this method, exactly
        as `PrefillInputStager.stage_device_inputs` probes it for
        `prepare_prefill_rot_mats`, and keeps its previous column-concatenating
        composition for a model that does not provide one. Galaxy has to provide
        one: the vocabulary is sharded over mesh **rows** and replicated over the
        four columns, so concatenating columns along the vocabulary axis is
        finding D-B23. `compose_galaxy_logits` is the composition that carries the
        Milestone B measurement.
        """

        composed = compose_galaxy_logits(
            value,
            mesh_device=self._model.mesh_device,
            vocab_size=self.vocab_size,
        )
        rows = int(self._model.config.max_batch_size)
        if int(composed.shape[0]) < rows:
            raise ValueError(f"composed {composed.shape[0]} decode rows, expected {rows}")
        return composed[:rows, :].reshape(1, 1, rows, -1)

    def compose_decode_tokens(self, value: Any) -> torch.Tensor:
        """Compose device-sampled decode tokens into one flat vector of users.

        Probed by `DecodeRuntime._normalize_host_output` for the same reason and
        on the same terms as `compose_decode_logits`. `ttnn.sampling`'s output
        inherits its labels from the all-gather that feeds it, so the runtime's
        generic column-concatenating composition reads the wrong axis;
        `compose_galaxy_sampled_tokens` is the qualified one.
        """

        return compose_galaxy_sampled_tokens(
            value,
            mesh_device=self._model.mesh_device,
            users=int(self._model.config.max_batch_size),
        )

    # -- private ----------------------------------------------------------

    def _project_tile_block(self, hidden: Any, block_start: int, sequence_length: int, *, row: int) -> Any:
        """Project a chunk's last token and present it in its tile block row."""

        block_start = int(block_start)
        row = int(row)
        if not 0 <= row < _TILE_SIZE:
            raise ValueError(f"prefill last-token row {row} is outside one tile block")
        last_token = block_start + row
        if block_start < 0 or last_token >= sequence_length:
            raise ValueError(f"tile block row {last_token} is outside {sequence_length} prefill tokens")
        outputs: tuple[Any, ...] = ()
        try:
            outputs = self._model.project_prefill_logits(
                hidden,
                rows=1,
                sequence_length=sequence_length,
                token_indices=(last_token,),
            )
            composed = compose_galaxy_logits(
                outputs[0],
                mesh_device=self._model.mesh_device,
                vocab_size=self.vocab_size,
            )
        finally:
            for value in outputs:
                deallocate_if_allocated(value)
        return self._present_logits_block(composed[:1, :], row)

    def _present_logits_block(self, row_logits: torch.Tensor, row: int | None) -> Any:
        """Stage `[1, 1, 32, vocab]` replicated TILE logits from one host row."""

        vocab = int(row_logits.shape[-1])
        block = torch.zeros((1, 1, _TILE_SIZE, vocab), dtype=torch.bfloat16)
        if row is None:
            block[0, 0, :, :] = row_logits[0].to(torch.bfloat16)
        else:
            block[0, 0, int(row), :] = row_logits[0].to(torch.bfloat16)
        return ttnn.from_torch(
            block,
            device=self._model.mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self._model.mesh_device),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )


class Llama33_70BGalaxyExecutor:
    """Compose every runtime owner for one Galaxy Llama-3.3-70B execution lane.

    Construction wires one `PrefillRuntime`, `DecodeRuntime`, `ProgramCompiler`
    and `EagerExecutor` over one full-mesh `Llama33_70BGalaxyTransformer2D`, one
    `PagedKVCacheManager`, one `OutputReader` and one `WarmupCoordinator`.
    Trace-enabled configurations add one `TraceCompiler` and one `TracedExecutor`
    over that exact eager instance.

    The caller resolves and allocates the paged KV cache, warms or compiles
    programs, then calls `prefill_forward` and `decode_forward`. `cleanup` is the
    deterministic release root and makes the executor terminal. The model, its
    `Prefetcher2D` and its Galaxy CCL resources are **borrowed**: the executor
    activates the mode context each operation needs and synchronizes the active
    mode during cleanup, but the loader handle that built the model closes it.
    That is what makes repeated startup/serve/cleanup cycles over one model
    possible.
    """

    requires_prefill_trace_warmup = True

    def __init__(
        self,
        model: Llama33_70BGalaxyTransformer2D,
        runtime_config: Any,
        config: Llama33_70BGalaxyExecutorConfig,
    ) -> None:
        if not isinstance(config, Llama33_70BGalaxyExecutorConfig):
            raise TypeError("config must be a Llama33_70BGalaxyExecutorConfig")
        if not callable(getattr(model, "iter_executor_named_modules", None)):
            raise TypeError("model must provide iter_executor_named_modules()")
        if not callable(getattr(model, "paged_kv_contract", None)):
            raise TypeError("model must provide paged_kv_contract()")
        if not callable(getattr(model, "activate", None)):
            raise TypeError("model must provide activate(mode)")
        if not callable(getattr(runtime_config, "can_enable_trace", None)):
            raise TypeError("runtime_config must provide can_enable_trace()")
        mesh_device = getattr(getattr(model, "config", None), "mesh_device", None)
        if mesh_device is None:
            raise ValueError("model.config.mesh_device is required")
        if tuple(int(value) for value in mesh_device.shape) != GALAXY_MESH_SHAPE:
            raise ValueError(f"the Galaxy executor requires a {GALAXY_MESH_SHAPE} mesh")
        if config.device_sampling_enabled and model.sampling is None:
            raise ValueError("device sampling requires a resolved Sampling2D on the model")
        if config.sequential_prefill_only and tuple(config.warmup.prefill_batch_sizes) != (1,):
            # A warmup batch size denotes a physical padded prefill wave. With
            # batching resolved off, any value above one would plan that many
            # identical single-row requests and compile the same program again,
            # which reads as coverage it is not.
            raise ValueError("sequential prefill requires warmup.prefill_batch_sizes == (1,)")

        self.model = model
        self.runtime_config = runtime_config
        self.model_args = runtime_config
        self.config = config
        self.mesh_device = mesh_device
        self.cache_path = getattr(runtime_config, "model_cache_path", None)
        self.geometry = model.geometry
        # The common runtime's warmup plan covers one cached (prefix) prefill case
        # per configured length, and until now its prefix was fixed at one
        # page-table block. On this mesh a block is 32 tokens and `Attention2D`
        # refuses any `chunk_start` that is not a multiple of the recipe's
        # `chunk_alignment`, which is 128 because it *is* the flash-SDPA chunk
        # size (`recipes.py:651`). So the default plan asked for a prefix this
        # model cannot serve and warmup died host-side before any device work:
        #
        #   ValueError: chunk_start must be non-negative and aligned to
        #               chunk_alignment            (attention_2d.py:908, D-C16)
        #
        # The alignment is the model's own property, so resolving the warmup
        # prefix against it is model-owned work. The runtime keeps its previous
        # default for every caller that leaves the field unset.
        chunk_alignment = int(getattr(self.geometry, "chunk_alignment", 0) or 0)
        if chunk_alignment > 0 and config.warmup.cached_prefill_tokens is None:
            config = replace(config, warmup=replace(config.warmup, cached_prefill_tokens=chunk_alignment))
            self.config = config
        self._terminal = False
        self._cleaned_up = False
        self._sampling_buffers_loaded = False
        self._runtime_configuration_sealed = False
        self._active_mode: str | None = None
        #: True only for the duration of an activation the trace compiler asked
        #: for. Read by the prefetch owner; see `_activate_for_capture`.
        self._capture_activation = False
        #: Built lazily on the first warmup, so an executor constructed
        #: without Galaxy resources (host tests) never asks for one.
        self._capture_frees: Any = None
        #: Persistent Galaxy-placed decode positions and page table; see
        #: `_ensure_decode_inputs`.
        self._decode_inputs: tuple[Any, Any] | None = None

        self.runtime_model = _GalaxyRuntimeModelView(
            model,
            device_sampling_enabled=config.device_sampling_enabled,
        )
        self.kv_cache_manager = PagedKVCacheManager(model.paged_kv_contract(), config.paged_kv_cache)
        self.page_table_layout = self._resolve_page_table_layout()
        self.output_reader = OutputReader(mesh_device)
        self.prefill_runtime = PrefillRuntime(self._resolve_prefill_config(self.page_table_layout))
        self.decode_runtime = DecodeRuntime(self._resolve_decode_config(self.page_table_layout))
        self.program_compiler = ProgramCompiler(mesh_device, lambda: self.kv_cache_manager.bound_context)
        self.eager_executor = EagerExecutor(
            prefill=self.prefill_runtime,
            decode=self.decode_runtime,
            program_compiler=self.program_compiler,
        )
        self.trace_compiler: TraceCompiler | None = None
        self.traced_executor: TracedExecutor | None = None
        if config.trace.mode != "none":
            self.trace_compiler = TraceCompiler(self.program_compiler)
            self.traced_executor = TracedExecutor(
                eager=self.eager_executor,
                trace_compiler=self.trace_compiler,
                trace_mode=config.trace.mode,
                # One capture region records one operation's programs, and on
                # this model prefill and decode do not share a device context:
                # `GalaxyExecutionResources.activate` loads a *different*
                # sub-device manager per mode, sets a different stall group, and
                # starts the persistent prefetcher for decode only
                # (`prefetcher_2d.py::_configure_mode_resources`). A decode
                # program dispatched under the prefill manager is
                # `TT_FATAL @ program.cpp:2205 Kernel group cores do not match
                # sub device cores` - D-C8's signature - and that abort inside a
                # multi-sub-device program leaves the mesh un-drainable. So the
                # trace compiler is given this executor's own operation-boundary
                # action to run before each capture, outside the capture region.
                capture_context=self._activate_for_capture,
            )
            # A capture region cannot contain a host barrier, and the Galaxy
            # collectives perform four of them inside the decode graph
            # (`collectives.py`: `gather_users`, `reduce_create_qkv_heads`, and
            # both `_all_reduce` branches). `ProgramCompiler` already publishes
            # the exact bracket - it is set around every capture region by
            # `TraceCompiler.capture_all` - so the model's resources are told how
            # to ask, and skip the barrier only there. Eager execution keeps it.
            self.model.resources.set_capture_probe(lambda: self.program_compiler.trace_capture_in_progress)
            # A second, narrower probe, for a fault the barrier one cannot cover.
            # `ttnn.dram_prefetcher` is a persistent sender that blocks on
            # global-CB credits from the decode graph, and
            # `Prefetcher2D.activate("decode")` dispatches it. Under capture that
            # graph is *recorded*, so the sender is stranded on the prefetch
            # sender cores (`x in {0, 4}`) - and the prefill subdevice is the
            # whole compute grid, so the next blocking finish in prefill mode
            # waits on a worker-completion count those cores can never reach.
            # That is the `ttnn.end_trace_capture` hang in
            # `tttv2_milestone_c_evidence/trace/logs/s4_l1_prefill_capture_hang.gdb.txt`.
            #
            # It cannot be the same probe: the launch happens in the activation
            # `capture_all` performs *before* `ttnn.begin_trace_capture`, when no
            # capture region is open yet. Nor may the barrier probe be widened to
            # the whole capture phase - `capture_all` stages real device work in
            # that window, and those barriers are what keep the host from
            # freeing a buffer a running program still reads
            # (`collectives.py:917`, `:956`). So this probe answers the exact
            # question instead: is this activation the trace compiler's?
            self.model.resources.set_prefetch_capture_probe(lambda: self._capture_activation)
            # And the view is told when a trace is active, which is the only
            # condition under which it may not free the prefill hidden state.
            self.runtime_model.set_trace_active_probe(lambda: self.trace_compiler.trace_active)
            # The launch *site* is deliberately left at the boundary. TTTv1
            # launches `ttnn.dram_prefetcher` from inside `forward`
            # (`models/demos/llama3_70b_galaxy/tt/llama_model.py:790-808`) and
            # `Prefetcher2D.set_sender_launch_site("graph")` exists so this
            # executor can do the same - but measured on silicon it does **not**
            # fix the traced-replay instability
            # (`tttv2_milestone_c_evidence/trace/logs/e6_Q_logits.log`: non-finite
            # traced decode logits with the graph site selected), so enabling it
            # would be an unmeasured behaviour change to shared 2D module code.
            # The seam and its tests stay; the switch stays off until something
            # measures a difference.
        self.eager_execution = self.eager_executor
        self.traced_prefill_execution = (
            self.traced_executor if config.trace.prefill_enabled and self.traced_executor is not None else None
        )
        self.traced_decode_execution = (
            self.traced_executor if config.trace.decode_enabled and self.traced_executor is not None else None
        )
        self._prefill_execution = self.traced_prefill_execution or self.eager_executor
        self._decode_execution = self.traced_decode_execution or self.eager_executor

        prefill_sequence_lengths = getattr(runtime_config, "trace_prefill_warmup_seq_lens", ())
        if not prefill_sequence_lengths:
            prefill_sequence_lengths = getattr(runtime_config, "trace_prefill_supported_seq_lens", (128,))
        self.warmup = WarmupCoordinator(
            config=WarmupCoordinatorConfig.resolve(
                warmup=config.warmup,
                trace=config.trace,
                prefill=self.prefill_runtime.config,
                decode=self.decode_runtime.config,
                prefill_sequence_lengths=tuple(int(value) for value in prefill_sequence_lengths),
            ),
            execution=self.traced_executor or self.eager_executor,
            ensure_sampling_buffers=self._ensure_sampling_buffers,
            validate_bound_cache=self._validate_bound_cache,
            # The coordinator compiles one decode program per configured
            # sampling path, and a decode operation boundary here dispatches the
            # persistent `ttnn.dram_prefetcher` sender for exactly one traversal
            # of the decode graph. So every program after the first needs its own
            # boundary, and this is the executor handing the coordinator the same
            # activation that `compile_decode` and `decode_forward` perform per
            # call. See `WarmupCoordinator.warmup_decode` and
            # `tttv2_milestone_c_evidence/trace/logs/w1_L_sampling_tb.log`.
            reestablish_operation_boundary=self.activate,
        )

    # Public model execution API

    @property
    def model_config(self) -> Any:
        return self.model.config

    @property
    def cluster_shape(self) -> list[int]:
        return list(self.mesh_device.shape)

    @property
    def paged_kv_cache_config(self) -> PagedKVCacheConfig:
        return self.kv_cache_manager.config

    @property
    def terminal(self) -> bool:
        return self._terminal

    @property
    def active_mode(self) -> str | None:
        """Return the prefetcher/CCL mode this executor last activated."""

        return self._active_mode

    @property
    def already_warmed_up_prefill(self) -> bool:
        return self.warmup.already_warmed_up_prefill

    def configure_paged_kv_cache(self, config: PagedKVCacheConfig) -> None:
        """Resolve the physical KV geometry before the first allocation.

        The runtime's late-resolution step is documented as "only ``num_blocks``
        becomes final", and on a 1D model that is all it takes. On Galaxy it is
        not: `Attention2D.bind_kv_cache` validates a bound cache against the
        block count its **own** metadata declares, so a physical pool smaller
        than the construction ceiling is refused at binding —

            ValueError: paged KV cache shape must be (2048, 1, 32, 128),
                        got (95, 1, 32, 128)                (`logs/i9_shrink_l1.log`)

        The model therefore has to be told the physical count, not just the
        ceiling. That is the model-owned half of this step and it lives here.
        Resolving may only shrink: the ceiling was a construction-time capacity
        bound, and narrowing it to the physical pool is what makes the bound
        cache, the module metadata and the page-table geometry describe one
        geometry.
        """

        self._ensure_active()
        if self._runtime_configuration_sealed:
            raise RuntimeError("runtime configuration is sealed")
        if not isinstance(config, PagedKVCacheConfig):
            raise TypeError("config must be a PagedKVCacheConfig")
        current = self.kv_cache_manager.config
        if current.is_resolved():
            raise RuntimeError("paged KV cache configuration is already resolved")
        if config.dtype != current.dtype:
            raise ValueError("resolved paged KV cache cannot change dtype")
        if config.memory_config != current.memory_config:
            raise ValueError("resolved paged KV cache cannot change memory_config")
        if config.block_size != current.block_size:
            raise ValueError("resolved paged KV cache cannot change block_size")
        if not config.is_resolved():
            raise ValueError("resolved paged KV cache must contain num_blocks")
        physical = int(config.num_blocks)
        if physical > current.max_num_blocks:
            raise ValueError(
                f"resolved paged KV capacity {physical} exceeds the construction ceiling {current.max_num_blocks}"
            )
        resolved = PagedKVCacheConfig(
            block_size=int(config.block_size),
            max_num_blocks=physical,
            dtype=config.dtype,
            memory_config=config.memory_config,
            num_blocks=physical,
        )
        if physical == current.max_num_blocks:
            # Nothing about the model's per-layer metadata moves, so the manager
            # keeps its identity and its already-validated model contract.
            self.kv_cache_manager.configure(resolved)
        else:
            # The model's paged metadata moves, and `GalaxyPagedKVContract`
            # snapshots that metadata at construction. The manager owns no device
            # resource before `allocate()`, so the honest move is to rebuild it
            # against the model's updated contract rather than to let a stale
            # snapshot validate the replacement.
            self.model.configure_paged_attention(block_size=resolved.block_size, max_num_blocks=physical)
            self.kv_cache_manager = PagedKVCacheManager(self.model.paged_kv_contract(), resolved)
        self.config = replace(self.config, paged_kv_cache=resolved)
        self._refresh_page_table_layout()

    def allocate_kv_cache(self) -> list[list[Any]]:
        """Allocate and bind the model-owned paged KV cache."""

        self._ensure_active()
        if not self.kv_cache_manager.config.is_resolved():
            raise RuntimeError("Paged KV cache capacity must be resolved before allocation")
        self._seal_runtime_configuration()
        return self.kv_cache_manager.allocate()

    def compile_prefill(
        self,
        *,
        tokens: torch.Tensor,
        page_table: torch.Tensor,
        prompt_lens: torch.Tensor | None = None,
        start_pos: torch.Tensor | None = None,
        empty_slots: Sequence[int] | None = None,
        kv_cache: Any = None,
        sampling_params: Any = None,
        execution: EagerExecutor | TracedExecutor | None = None,
    ) -> Any:
        """Compile prefill on the supplied eager or traced execution target."""

        self._ensure_active()
        self._validate_bound_cache(kv_cache)
        self._ensure_sampling_for(sampling_params)
        self.activate("prefill")
        self.runtime_model.bind_prefill_last_row(_prefill_last_row(tokens, prompt_lens, start_pos))
        try:
            return (execution or self._prefill_execution).compile_prefill(
                tokens=tokens,
                page_table=page_table,
                prompt_lens=prompt_lens,
                start_pos=start_pos,
                empty_slots=empty_slots,
                sampling_params=sampling_params,
            )
        finally:
            self.runtime_model.clear_prefill_last_row()

    def compile_decode(
        self,
        *,
        tokens: torch.Tensor,
        start_pos: torch.Tensor,
        page_table: torch.Tensor,
        kv_cache: Any = None,
        sampling_params: Any = None,
        reset_batch: bool = False,
        execution: EagerExecutor | TracedExecutor | None = None,
    ) -> Any:
        """Compile decode on the supplied eager or traced execution target."""

        self._ensure_active()
        self._validate_bound_cache(kv_cache)
        self._ensure_sampling_for(sampling_params)
        self.activate("decode")
        self._refresh_decode_inputs(start_pos=start_pos, page_table=page_table)
        self._refresh_device_sampling(sampling_params)
        return (execution or self._decode_execution).compile_decode(
            tokens=tokens,
            start_pos=start_pos,
            page_table=page_table,
            sampling_params=sampling_params,
            reset_batch=reset_batch,
        )

    def prefill_forward(
        self,
        tokens: torch.Tensor,
        page_table: torch.Tensor,
        *,
        prompt_lens: torch.Tensor | None = None,
        start_pos: torch.Tensor | None = None,
        empty_slots: Sequence[int] | None = None,
        kv_cache: Any = None,
        sampling_params: Any = None,
        execution: EagerExecutor | TracedExecutor | None = None,
    ) -> Any:
        """Validate ownership, activate the prefill context, and run one call."""

        self._ensure_active()
        self._validate_bound_cache(kv_cache)
        self._ensure_sampling_for(sampling_params)
        self.activate("prefill")
        self.runtime_model.bind_prefill_last_row(_prefill_last_row(tokens, prompt_lens, start_pos))
        try:
            return (execution or self._prefill_execution).prefill_forward(
                tokens=tokens,
                page_table=page_table,
                prompt_lens=prompt_lens,
                start_pos=start_pos,
                empty_slots=empty_slots,
                sampling_params=sampling_params,
            )
        finally:
            self.runtime_model.clear_prefill_last_row()

    def decode_forward(
        self,
        tokens: torch.Tensor,
        start_pos: torch.Tensor,
        page_table: torch.Tensor,
        *,
        kv_cache: Any = None,
        sampling_params: Any = None,
        reset_batch: bool = False,
        read_from_device: bool = True,
        execution: EagerExecutor | TracedExecutor | None = None,
    ) -> Any:
        """Validate ownership, activate the decode context, and run one call."""

        self._ensure_active()
        self._validate_bound_cache(kv_cache)
        self._ensure_sampling_for(sampling_params)
        self.activate("decode")
        self._refresh_decode_inputs(start_pos=start_pos, page_table=page_table)
        self._refresh_device_sampling(sampling_params)
        return (execution or self._decode_execution).decode_forward(
            tokens=tokens,
            start_pos=start_pos,
            page_table=page_table,
            sampling_params=sampling_params,
            reset_batch=reset_batch,
            read_from_device=read_from_device,
        )

    def can_trace_prefill(
        self,
        *,
        tokens: torch.Tensor,
        prompt_lens: torch.Tensor | None = None,
        start_pos: torch.Tensor | None = None,
        empty_slots: Sequence[int] | None = None,
    ) -> bool:
        """Classify whether the prefill request can use this lane's trace."""

        if self.traced_executor is None or not self.config.trace.prefill_enabled:
            return False
        return self.prefill_runtime.can_trace(
            tokens=tokens,
            prompt_lens=prompt_lens,
            start_pos=start_pos,
        )

    def read_decode_output(self, tt_out: Any, *, async_read: bool = False) -> Any:
        self._ensure_active()
        return self.decode_runtime.read_decode_output(tt_out=tt_out, async_read=async_read)

    def process_decode_output_host(self, tt_out: Any, *, is_tokens: bool = False) -> tuple[Any, Any]:
        self._ensure_active()
        return self.decode_runtime.process_decode_output_host(tt_out=tt_out, is_tokens=is_tokens)

    def warmup_model_prefill(
        self,
        *,
        kv_cache: Any,
        can_sample_on_device: bool = False,
        enable_trace: bool = False,
    ) -> None:
        self._ensure_active()
        self.activate("prefill")
        # Every warmup plan includes a cached (prefix) prefill case, and a cached
        # request's body extracts its own last token. Warmup builds each case with
        # its whole padded length present and no partial prompt, so the last
        # token's row inside its tile block is `(length - 1) % 32` for every case.
        self.runtime_model.bind_prefill_last_row(self._warmup_prefill_last_row())
        try:
            # F-1: no DRAM address may be recycled inside a trace capture body.
            # `capture_all` fires from inside whichever warmup call completes the
            # configured plan last, so both calls hold the same region.
            with self._capture_free_hold():
                return self.warmup.warmup_prefill(
                    kv_cache=kv_cache,
                    can_sample_on_device=can_sample_on_device,
                    enable_trace=enable_trace,
                )
        finally:
            self.runtime_model.clear_prefill_last_row()

    def warmup_model_decode(
        self,
        *,
        kv_cache: Any,
        max_batch_size: int | None = None,
        num_blocks: int | None = None,
        can_sample_on_device: bool = False,
        enable_trace: bool = False,
    ) -> None:
        self._ensure_active()
        lane_capacity = int(self.decode_runtime.config.lane_capacity)
        blocks = int(num_blocks) if num_blocks is not None else int(self.page_table_layout.decode_width)
        self.activate("decode")
        page_table = torch.zeros((lane_capacity, blocks), dtype=torch.int32)
        self._refresh_decode_inputs(
            start_pos=torch.zeros(lane_capacity, dtype=torch.long),
            page_table=page_table,
        )
        if can_sample_on_device:
            self._refresh_device_sampling(_warmup_greedy_sampling_params(lane_capacity))
        # F-1, as in `warmup_model_prefill`: see `_capture_free_hold`.
        with self._capture_free_hold():
            return self.warmup.warmup_decode(
                kv_cache=kv_cache,
                max_batch_size=lane_capacity if max_batch_size is None else int(max_batch_size),
                num_blocks=blocks,
                can_sample_on_device=can_sample_on_device,
                enable_trace=enable_trace,
            )

    def _capture_free_hold(self) -> Any:
        """Hold this model's capture-body DRAM frees for the whole capture phase.

        A buffer freed inside a capture body hands its address back to the
        allocator while the graph being recorded still writes into it, so a later
        allocation in the same body can be placed on top of a Galaxy
        collective's fabric-write destination. That is defect F-1; the resource
        owner implements the hold, because it is the only thing that knows both
        when a capture region is open and which buffers it lends out. See
        `src/tt_transformers/models/galaxy/capture_frees.py` for the silicon arms.

        Inert when tracing is off, and inert outside a capture region, so this
        wraps warmup unconditionally rather than branching on `trace_mode`.
        """

        resources = getattr(self.model, "resources", None)
        build = getattr(resources, "defer_capture_frees", None)
        if not callable(build):
            return contextlib.nullcontext()
        if self._capture_frees is None:
            self._capture_frees = build()
        return self._capture_frees

    def _activate_for_capture(self, mode: str) -> Any:
        """Establish one operation's context on the trace compiler's behalf.

        Identical to `activate` except that it is marked as a capture
        activation, which is what stops the persistent prefetch sender being
        dispatched with no consumer. See `set_prefetch_capture_probe`.
        """

        self._capture_activation = True
        try:
            return self.activate(mode)
        finally:
            self._capture_activation = False

    def activate(self, mode: str) -> Any:
        """Activate one operation's prefetcher/CCL context at its boundary."""

        if mode not in ("prefill", "decode"):
            raise ValueError(f"unsupported Galaxy execution mode: {mode!r}")
        context = self.model.activate(mode)
        self._active_mode = mode
        return context

    def synchronize(self) -> None:
        """Wait for the active mode's outstanding device work."""

        if self._active_mode is not None:
            self.model.synchronize(self._active_mode)

    def cleanup(self) -> None:
        """Release runtime, trace, program and KV resources in order.

        Outstanding device work is drained first, so nothing the runtime is about
        to deallocate is still referenced by a running program. The borrowed
        prefetcher/CCL context is left as the model found it; the model's own
        owner closes it.
        """

        self._terminal = True
        if self._cleaned_up:
            return

        failures: list[BaseException] = []
        actions = [
            self.synchronize,
            self.runtime_model.clear_decode_inputs,
            self.runtime_model.clear_prefill_last_row,
            self.decode_runtime.drain_external_outputs,
            self.output_reader.drain,
            self.prefill_runtime.cleanup,
            self.decode_runtime.cleanup_transients,
        ]
        if self.trace_compiler is not None:
            actions.append(self.trace_compiler.cleanup)
        # After the trace is released: the captured decode program reads this pair.
        actions.append(self._release_decode_inputs)
        actions.append(self._clear_capture_probe)
        actions.append(self.program_compiler.cleanup)
        if self.config.device_sampling_enabled:
            actions.append(self.model.sampling.release)
        actions.append(self.kv_cache_manager.release)

        for action in actions:
            try:
                action()
            except BaseException as error:  # noqa: BLE001 - collect, then raise the first
                failures.append(error)
        if failures:
            _raise_cleanup_failures(failures, "Llama33_70BGalaxyExecutor")
        self._cleaned_up = True

    # Private implementation

    def _clear_capture_probe(self) -> None:
        """Return the borrowed model's resources to unconditional barriers."""

        if self.trace_compiler is None:
            return
        resources = getattr(self.model, "resources", None)
        if resources is None:
            return
        if callable(getattr(resources, "set_capture_probe", None)):
            resources.set_capture_probe(None)
        if callable(getattr(resources, "set_prefetch_capture_probe", None)):
            resources.set_prefetch_capture_probe(None)
        if callable(getattr(resources, "set_prefetch_sender_launch_site", None)):
            resources.set_prefetch_sender_launch_site("boundary")
        if callable(getattr(self.runtime_model, "set_sender_launch_in_graph", None)):
            self.runtime_model.set_sender_launch_in_graph(False)
        if callable(getattr(self.runtime_model, "set_trace_active_probe", None)):
            self.runtime_model.set_trace_active_probe(None)

    def _decode_input_mapper(self) -> Any:
        return ttnn.ShardTensor2dMesh(self.mesh_device, dims=(None, 0), mesh_shape=GALAXY_MESH_SHAPE)

    def _ensure_decode_inputs(self) -> tuple[Any, Any]:
        """Allocate this lane's persistent Galaxy-placed decode pair once.

        **The pair used to be allocated and freed inside one call, and tracing
        forbids that.** A trace bakes the buffer addresses its capture body read,
        so a pair released at the end of the capturing call leaves every replay
        reading a freed allocation. It also breaks capture *ordering*:
        `WarmupCoordinator._maybe_capture` fires `capture_all()` from whichever
        public warmup call completes the configured plan second, and under
        `TraceConfig(mode="all")` that can be `warmup_prefill` — where no decode
        call is in progress at all and the decode capture body would raise
        "Galaxy decode inputs were not staged for this call".

        Making the pair persistent is also what keeps the active row count and the
        runtime's slot assignment *out of* program and trace identity: they reach
        the graph only as the contents of these two tensors, refreshed per call by
        `_refresh_decode_inputs`. The geometry that decides their shape - lane
        capacity and the resolved decode page-table width - is physical and fixed
        for the executor's life, which is exactly the property the trace key needs.

        The eager path uses the same pair, because there is only one path.
        """

        if self._decode_inputs is not None:
            return self._decode_inputs
        lane_capacity = int(self.decode_runtime.config.lane_capacity)
        width = int(self.page_table_layout.decode_width)
        mapper = self._decode_input_mapper()
        positions = ttnn.from_torch(
            torch.zeros(lane_capacity, dtype=torch.int32),
            device=self.mesh_device,
            mesh_mapper=mapper,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        try:
            table = ttnn.from_torch(
                torch.zeros((lane_capacity, width), dtype=torch.int32),
                device=self.mesh_device,
                mesh_mapper=mapper,
                dtype=self.model.kv_specs[0].page_table_dtype,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        except BaseException:
            deallocate_if_allocated(positions)
            raise
        self._decode_inputs = (positions, table)
        self.runtime_model.bind_decode_inputs(positions=positions, page_table=table)
        return self._decode_inputs

    def _refresh_decode_inputs(self, *, start_pos: torch.Tensor, page_table: torch.Tensor) -> None:
        """Write one call's positions and page table into the persistent pair."""

        positions, table = self._ensure_decode_inputs()
        lane_capacity = int(self.decode_runtime.config.lane_capacity)
        row = _require_row(start_pos, lane_capacity, "decode start_pos")
        host_table = _require_decode_page_table(page_table, lane_capacity, int(self.page_table_layout.decode_width))
        mapper = self._decode_input_mapper()
        host_positions = ttnn.from_torch(
            row.to(torch.int32),
            device=None,
            mesh_mapper=mapper,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        host_page_table = ttnn.from_torch(
            host_table,
            device=None,
            mesh_mapper=mapper,
            dtype=self.model.kv_specs[0].page_table_dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        ttnn.copy_host_to_device_tensor(host_positions, positions)
        ttnn.copy_host_to_device_tensor(host_page_table, table)

    def _release_decode_inputs(self) -> None:
        """Release the persistent decode pair; idempotent, and after the trace."""

        pair, self._decode_inputs = self._decode_inputs, None
        if pair is None:
            return
        self.runtime_model.clear_decode_inputs()
        for value in pair:
            deallocate_if_allocated(value)

    def _warmup_prefill_last_row(self) -> int | None:
        """Return the tile-block row every configured warmup case's last token has."""

        lengths = set(int(value) for value in self.warmup.config.prefill_sequence_lengths)
        # The Q128 sampled path primes the four tile ends of a 128-token prefill.
        if 128 in lengths and self.config.device_sampling_enabled:
            lengths.update((32, 64, 96, 128))
        rows = {(length - 1) % _TILE_SIZE for length in lengths}
        return rows.pop() if len(rows) == 1 else None

    def _resolve_prefill_config(self, layout: PageTableLayout) -> PrefillRuntimeConfig:
        runtime_config = self.runtime_config
        supports_batched = (
            False
            if self.config.sequential_prefill_only
            else bool(getattr(runtime_config, "supports_batched_prefill", False))
        )
        return PrefillRuntimeConfig.resolve(
            model=self.runtime_model,
            output_reader=self.output_reader,
            page_table_layout=layout,
            max_batch_size=int(self.model.config.max_batch_size),
            max_prefill_chunk_size=int(runtime_config.max_prefill_chunk_size),
            device_sampling_enabled=self.config.device_sampling_enabled,
            can_enable_trace=runtime_config.can_enable_trace,
            supports_batched_prefill=supports_batched,
            disable_batched_prefill=self.config.sequential_prefill_only
            or bool(getattr(runtime_config, "disable_batched_prefill", False)),
            max_prefill_batch_size=1
            if self.config.sequential_prefill_only
            else int(getattr(runtime_config, "max_prefill_batch_size", 1)),
            batched_prefill_batched_extract=False
            if self.config.sequential_prefill_only
            else bool(getattr(runtime_config, "batched_prefill_batched_extract", False)),
        )

    def _resolve_decode_config(self, layout: PageTableLayout) -> DecodeRuntimeConfig:
        return DecodeRuntimeConfig.resolve(
            model=self.runtime_model,
            output_reader=self.output_reader,
            lane_capacity=int(self.model.config.max_batch_size),
            page_table_layout=layout,
            device_sampling_enabled=self.config.device_sampling_enabled,
            force_greedy_top_k=self.config.warmup.include_decode_top_k,
        )

    def _resolve_page_table_layout(self) -> PageTableLayout:
        kv_config = self.kv_cache_manager.config
        physical_num_blocks = kv_config.num_blocks or kv_config.max_num_blocks
        return PageTableLayout.resolve(
            block_size=int(kv_config.block_size),
            model_max_sequence_length=int(self.model.config.max_seq_len),
            physical_num_blocks=int(physical_num_blocks),
            max_prefill_chunk_size=min(
                int(self.runtime_config.max_prefill_chunk_size),
                int(self.model.config.max_seq_len),
            ),
        )

    def _refresh_page_table_layout(self) -> None:
        """Install the final physical geometry across every resolved config."""

        layout = self._resolve_page_table_layout()
        prefill_config = self._resolve_prefill_config(layout)
        decode_config = self._resolve_decode_config(layout)
        warmup_config = WarmupCoordinatorConfig.resolve(
            warmup=self.warmup.config.warmup,
            trace=self.config.trace,
            prefill=prefill_config,
            decode=decode_config,
            prefill_sequence_lengths=self.warmup.config.prefill_sequence_lengths,
        )
        self.prefill_runtime.config = prefill_config
        self.decode_runtime.config = decode_config
        self.warmup.config = warmup_config
        self.page_table_layout = layout

    def _seal_runtime_configuration(self) -> None:
        self.warmup.seal_configuration()
        self._runtime_configuration_sealed = True

    def _ensure_sampling_for(self, sampling_params: Any) -> None:
        if sampling_params is None:
            return
        if not self.config.device_sampling_enabled:
            raise ValueError("sampling parameters were supplied while device sampling is disabled")
        self._ensure_sampling_buffers()

    def _refresh_device_sampling(self, sampling_params: Any) -> None:
        """Refresh `Sampling2D`'s persistent controls outside any trace region.

        See `_GalaxySamplingView`: the sampler's control buffers are refreshed by
        host-to-device writes, and `_update_call_buffers` also draws a fresh
        random seed per call. Neither may sit inside a captured graph, so both
        happen here, at the operation boundary, before the capture body or the
        replay that follows.
        """

        if sampling_params is None or not self.config.device_sampling_enabled:
            return
        self._ensure_sampling_buffers()
        sampler = self.runtime_model.sampling
        if sampler is None:
            raise RuntimeError("device sampling is enabled but this lane resolved no sampler")
        sampler.refresh(_galaxy_sampling_call(sampling_params, int(self.decode_runtime.config.lane_capacity)))

    def _ensure_sampling_buffers(self) -> None:
        if self._sampling_buffers_loaded or not self.config.device_sampling_enabled:
            return
        if self.trace_compiler is not None and self.trace_compiler.trace_active:
            raise RuntimeError("cannot materialize sampling buffers after trace activation")
        self.model.sampling.load_device_buffers()
        self._sampling_buffers_loaded = True

    def _validate_bound_cache(self, kv_cache: Any) -> None:
        if self.kv_cache_manager.bound_context is None:
            raise RuntimeError("Paged KV cache must be allocated and bound before execution")
        if kv_cache is not None:
            self.kv_cache_manager.validate_borrowed_handle(kv_cache)

    def _ensure_active(self) -> None:
        if self._terminal:
            raise RuntimeError("Llama33_70BGalaxyExecutor is terminal; construct a new executor")
        if self.prefill_runtime.transient_orphan_count or self.decode_runtime.transient_orphan_count:
            raise RuntimeError("Llama33_70BGalaxyExecutor has unreleased transient resources; clean up this executor")


def build_llama33_70b_galaxy_executor(
    llm: Any,
    config: Llama33_70BGalaxyExecutorConfig,
) -> Llama33_70BGalaxyExecutor:
    """Build one executor around an already-loaded Galaxy Llama handle."""

    return Llama33_70BGalaxyExecutor(llm.model, llm.runtime_config, config)


def default_galaxy_paged_kv_cache_config(model: Any, dtype: Any = None) -> PagedKVCacheConfig:
    """Return the `PagedKVCacheConfig` matching a built model's paged geometry."""

    spec = model.kv_specs[0]
    paged = spec.paged_attention_config
    if paged is None:
        raise ValueError("the Galaxy executor requires a paged KV model")
    return PagedKVCacheConfig(
        block_size=int(paged.block_size),
        max_num_blocks=int(paged.max_num_blocks),
        dtype=spec.kv_cache_dtype if dtype is None else dtype,
        num_blocks=int(paged.max_num_blocks),
    )


def _prefill_last_row(
    tokens: torch.Tensor,
    prompt_lens: torch.Tensor | None,
    start_pos: torch.Tensor | None,
) -> int | None:
    """Return the shared tile-block row of every request row's last token.

    The runtime addresses a cached or chunked prefill result at row
    ``(last_token_index - cached_tokens) % 32``, and derives the last token from
    ``prompt_lens`` and the cached prefix from ``start_pos`` exactly this way.
    Returns ``None`` when the public call's rows disagree, which leaves the view
    to refuse rather than to guess.
    """

    rows = int(tokens.shape[0])
    width = int(tokens.shape[-1])
    lengths = [width] * rows if prompt_lens is None else [int(value) for value in prompt_lens.reshape(-1)]
    cached = [0] * rows if start_pos is None else [int(value) for value in start_pos.reshape(-1)]
    if len(lengths) != rows or len(cached) != rows:
        return None
    candidates = {(length - 1 - num_cached) % _TILE_SIZE for length, num_cached in zip(lengths, cached)}
    return candidates.pop() if len(candidates) == 1 else None


def _as_token_row(tokens: Any) -> Any:
    """Return a rank-2 `[1, n]` view of a runtime-staged index tensor."""

    shape = tuple(int(value) for value in tokens.shape)
    if len(shape) == 2:
        return tokens
    return ttnn.reshape(tokens, ttnn.Shape((1, shape[-1])))


def _first_position(position_indices: Any) -> int:
    """Read the first entry of a runtime-staged prefill position-index tensor."""

    replicas = ttnn.get_device_tensors(position_indices)
    values = ttnn.to_torch(replicas[0] if replicas else position_indices).reshape(-1)
    return int(values[0])


def _tile_block_start(last_token_slice: Any) -> int:
    """Read the tile-block start out of the runtime's slice-bound tensor."""

    start = last_token_slice[0] if isinstance(last_token_slice, (tuple, list)) else last_token_slice
    values = ttnn.to_torch(ttnn.get_device_tensors(start)[0]).reshape(-1)
    return int(values[2])


def _warmup_greedy_sampling_params(size: int) -> SamplingParams:
    """Return the greedy controls the runtime's own decode warmup plan builds."""

    return SamplingParams(
        temperature=[0.0] * int(size),
        top_k=[1] * int(size),
        top_p=[1.0] * int(size),
    )


def _galaxy_sampling_call(sampling_params: Any, size: int) -> tuple[Any, Any, Any, Any]:
    """Spread one raw request's controls over every physical slot.

    Raw, not `format_sampling_params`-formatted: `Sampling2D` performs its own
    greedy encoding and its own temperature inversion, so handing it the
    runtime's formatted values would invert twice.
    """

    size = int(size)

    def spread(value: Any, default: Any) -> list[Any]:
        if value is None:
            return [default] * size
        if isinstance(value, torch.Tensor):
            value = [value.item()] if value.ndim == 0 else value.tolist()
        if not isinstance(value, (list, tuple)):
            value = [value]
        values = list(value)
        if len(values) == 1:
            values = values * size
        if len(values) < size:
            values = values + [default] * (size - len(values))
        return values[:size]

    return (
        spread(sampling_params.top_k, 1),
        spread(sampling_params.top_p, 1.0),
        spread(sampling_params.temperature, 0.0),
        spread(getattr(sampling_params, "seed", None), None),
    )


def _require_row(values: torch.Tensor, expected: int, name: str) -> torch.Tensor:
    row = values.reshape(-1)
    if int(row.numel()) != expected:
        raise ValueError(f"{name} must hold {expected} entries, got {int(row.numel())}")
    return row


def _require_decode_page_table(page_table: torch.Tensor, rows: int, width: int) -> torch.Tensor:
    """Return the `[rows, width]` int32 decode table the Galaxy graph reads.

    The Galaxy decode SDPA derives each slot's KV length from the table's row
    width, so the table is presented at the resolved decode width: narrower
    request tables are zero-extended and wider ones are refused rather than
    silently narrowed.
    """

    if not isinstance(page_table, torch.Tensor) or page_table.ndim != 2:
        raise ValueError("decode page_table must be a rank-2 torch.Tensor")
    if int(page_table.shape[0]) != rows:
        raise ValueError(f"decode page_table must have {rows} rows, got {int(page_table.shape[0])}")
    actual = int(page_table.shape[1])
    if actual > width:
        raise ValueError(f"decode page_table width {actual} exceeds the resolved decode width {width}")
    table = torch.zeros((rows, width), dtype=torch.int32)
    table[:, :actual] = page_table.to(torch.int32)
    return table


def _raise_cleanup_failures(failures: list[BaseException], owner: str) -> None:
    primary, *additional = failures
    attach_cleanup_failures(
        primary,
        additional,
        note=f"{owner} cleanup also encountered {{count}} failure(s)",
    )
    raise primary


__all__ = [
    "Llama33_70BGalaxyExecutor",
    "Llama33_70BGalaxyExecutorConfig",
    "build_llama33_70b_galaxy_executor",
    "default_galaxy_paged_kv_cache_config",
]
