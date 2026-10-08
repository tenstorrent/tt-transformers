# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Architecture-level vLLM entry points for the vLLM TT plugin.

vLLM resolves a checkpoint's architecture (``LlamaForCausalLM``, ...) to one
registered class, but one architecture is served here by several model
packages. Each class in this module is that one registered class for one text
architecture: when vLLM loads the model it picks the package's generator by
the exact Hugging Face id and returns that generator's instance unchanged, so
everything vLLM reads from the live model comes from the concrete generator.

The plugin registers these classes by dotted path, for example
``tt_transformers.vllm_registry:LlamaForCausalLM``. Their module and class
names are a stable interface: renaming or adding a model package changes the
tables below, never these names.

Importing this module is cheap and has no side effects. It imports neither
``ttnn`` nor ``vllm``; a generator module, and with it ``ttnn``, is imported
only when a model is selected. Nothing here registers anything with vLLM.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, ClassVar

from tt_transformers.cache_environment import hf_repo_id

__all__ = [
    "GeneratorEntry",
    "LlamaForCausalLM",
    "MistralForCausalLM",
    "Phi3ForCausalLM",
    "Qwen2ForCausalLM",
    "Qwen3ForCausalLM",
    "UnsupportedModelError",
    "resolve_hf_id",
]


class UnsupportedModelError(ValueError):
    """Raised when a checkpoint does not name a model this package serves."""


@dataclass(frozen=True)
class GeneratorEntry:
    """One servable checkpoint and the generator class that serves it."""

    hf_id: str
    package: str
    generator: str  # "module.path:ClassName", imported only when selected

    def load(self) -> Any:
        module_name, _, class_name = self.generator.partition(":")
        return getattr(importlib.import_module(module_name), class_name)


def _entry(hf_id: str, package: str, class_name: str) -> GeneratorEntry:
    return GeneratorEntry(hf_id, package, f"tt_transformers.models.{package}.vllm_generator:{class_name}")


resolve_hf_id = hf_repo_id


# Boolean capabilities read by the plugin from the registered class before the
# model exists. Each architecture class declares the keys that every one of its
# generators declares True; the generator instance carries its exact set at
# runtime. Value-typed keys stay on the generators: ``max_device_top_k`` is read
# from the instance, and ``fabric_config`` differs between the generators of one
# architecture, so serving configuration supplies it per deployment.
_SHARED_CAPABILITIES: dict[str, Any] = {
    "supports_prefix_caching": True,
    "supports_async_decode": True,
    "supports_sample_on_device": True,
    "accepts_trace_mode": True,
}


class _ArchitectureDispatcher:
    """Select a generator by exact Hugging Face id for one vLLM architecture."""

    architecture: ClassVar[str]
    generators: ClassVar[tuple[GeneratorEntry, ...]]
    model_capabilities: ClassVar[dict[str, Any]] = _SHARED_CAPABILITIES

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        cls.model_capabilities = dict(cls.model_capabilities)

    @classmethod
    def supported_hf_ids(cls) -> tuple[str, ...]:
        return tuple(entry.hf_id for entry in cls.generators)

    @classmethod
    def select(cls, name_or_path: str | None) -> GeneratorEntry:
        """Return the entry serving ``name_or_path``, or raise naming the supported ids."""

        hf_id = resolve_hf_id(name_or_path)
        for entry in cls.generators:
            if entry.hf_id == hf_id:
                return entry
        supported = ", ".join(cls.supported_hf_ids())
        raise UnsupportedModelError(
            f"tt_transformers does not serve {name_or_path!r} as {cls.architecture}. "
            f"Supported checkpoints: {supported}. Pass one of these ids as the model, "
            "or a Hugging Face cache snapshot of one."
        )

    @classmethod
    def get_max_tokens_all_users(
        cls,
        model_name: str = "",
        num_devices: int = 1,
        tt_data_parallel: int = 1,
        max_model_len: int = 0,
        max_num_seqs: int = 1,
    ) -> int:
        """Delegate KV-cache sizing to the generator that serves ``model_name``."""

        return (
            cls.select(model_name)
            .load()
            .get_max_tokens_all_users(
                model_name=model_name,
                num_devices=num_devices,
                tt_data_parallel=tt_data_parallel,
                max_model_len=max_model_len,
                max_num_seqs=max_num_seqs,
            )
        )

    @classmethod
    def initialize_vllm_model(
        cls,
        hf_config,
        mesh_device,
        max_batch_size,
        max_seq_len,
        n_layers=None,
        tt_data_parallel=1,
        optimizations=None,
        **kwargs,
    ):
        """Build and return the selected generator's own vLLM model.

        ``optimizations=None`` means "no override" and leaves the generator's
        default precision in place; ``"performance"`` and ``"accuracy"`` are
        passed through.
        """

        generator = cls.select(getattr(hf_config, "_name_or_path", None)).load()
        if optimizations is not None:
            kwargs["optimizations"] = optimizations
        return generator.initialize_vllm_model(
            hf_config,
            mesh_device,
            max_batch_size,
            max_seq_len,
            n_layers=n_layers,
            tt_data_parallel=tt_data_parallel,
            **kwargs,
        )


class LlamaForCausalLM(_ArchitectureDispatcher):
    architecture = "LlamaForCausalLM"
    generators = (
        _entry("meta-llama/Llama-3.2-1B-Instruct", "llama32_1b", "Llama32_1BGenerator"),
        _entry("meta-llama/Llama-3.2-3B-Instruct", "llama32_3b", "Llama32_3BGenerator"),
        _entry("meta-llama/Llama-3.1-8B-Instruct", "llama3_8b", "Llama3Generator"),
        _entry("meta-llama/Llama-3.3-70B-Instruct", "llama33_70b", "Llama33_70BGenerator"),
    )


class Qwen2ForCausalLM(_ArchitectureDispatcher):
    architecture = "Qwen2ForCausalLM"
    generators = (
        _entry("Qwen/Qwen2-7B-Instruct", "qwen2_7b", "Qwen2Generator"),
        _entry("Qwen/Qwen2.5-7B-Instruct", "qwen25_7b", "Qwen25Generator"),
        _entry("Qwen/Qwen2.5-Coder-32B-Instruct", "qwen25_coder_32b", "Qwen25Coder32BGenerator"),
        _entry("Qwen/Qwen2.5-72B-Instruct", "qwen25_72b", "Qwen25_72BGenerator"),
        _entry(
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
            "deepseek_r1_distill_qwen_14b",
            "DeepSeekR1Qwen14BGenerator",
        ),
    )


class Qwen3ForCausalLM(_ArchitectureDispatcher):
    architecture = "Qwen3ForCausalLM"
    generators = (_entry("Qwen/Qwen3-32B", "qwen3_32b", "Qwen3_32BGenerator"),)


class MistralForCausalLM(_ArchitectureDispatcher):
    architecture = "MistralForCausalLM"
    generators = (_entry("mistralai/Mistral-7B-Instruct-v0.3", "mistral_7b", "Mistral7BGenerator"),)


class Phi3ForCausalLM(_ArchitectureDispatcher):
    architecture = "Phi3ForCausalLM"
    generators = (_entry("microsoft/phi-4", "phi4", "Phi4Generator"),)
