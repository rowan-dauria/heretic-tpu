# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

# Placeholder for the Model facade described in docs/DESIGN.md ("Model facade").
# It fixes the plugin-facing API so that modules written against it can be imported
# and tested with fakes before the facade itself is implemented.

from contextlib import AbstractContextManager
from typing import TYPE_CHECKING, Any, Callable, NamedTuple, TypeAlias

import jax
import jax.numpy as jnp
import numpy as np
from transformers import PreTrainedTokenizerBase

from .config import ExportStrategy, Settings
from .utils import Prompt

if TYPE_CHECKING:
    from transformers import GenerationConfig, PretrainedConfig

Array: TypeAlias = np.ndarray | jax.Array

# Component name -> (inputs [L, M, N, d_in], outputs [L, M, N, d_out]), model dtype.
ModuleIO: TypeAlias = dict[str, tuple[Array, Array]]


class LMState(NamedTuple):
    engine: Any
    params: Any
    lora: Any
    hf_config: "PretrainedConfig"
    generation_config: "GenerationConfig"
    pad_id: int
    next_key: Callable[[], jax.Array]


def is_out_of_memory(e: BaseException) -> bool:
    raise NotImplementedError


class Model:
    settings: Settings
    tokenizer: PreTrainedTokenizerBase
    dtype: jnp.dtype
    revision_kwargs: dict[str, str]
    trusted_models: set[str]
    lora_rank: int | None

    def __init__(self, settings: Settings):
        raise NotImplementedError

    def get_layers(self) -> list[int]:
        raise NotImplementedError

    def get_abliterable_components(self) -> list[str]:
        raise NotImplementedError

    def get_module_count(self, component: str) -> int:
        raise NotImplementedError

    def get_base_weights(self, component: str) -> jax.Array:
        raise NotImplementedError

    def apply_lora(self, lora_rank: int) -> None:
        raise NotImplementedError

    def get_lora(self, component: str) -> tuple[jax.Array, jax.Array]:
        raise NotImplementedError

    def set_lora(self, component: str, A: Array, B: Array) -> None:
        raise NotImplementedError

    def reset_model(self) -> bool:
        raise NotImplementedError

    def lora_disabled(self) -> AbstractContextManager[None]:
        raise NotImplementedError

    def get_responses(
        self,
        prompts: list[Prompt],
        skip_special_tokens: bool = False,
    ) -> list[str]:
        raise NotImplementedError

    def get_responses_batched(
        self,
        prompts: list[Prompt],
        skip_special_tokens: bool = False,
    ) -> list[str]:
        raise NotImplementedError

    def get_logits(self, prompts: list[Prompt]) -> Array:
        raise NotImplementedError

    def get_logits_batched(self, prompts: list[Prompt]) -> Array:
        raise NotImplementedError

    def get_residuals(
        self,
        prompts: list[Prompt],
        winsorization_quantile: float = 1.0,
    ) -> Array:
        raise NotImplementedError

    def get_residuals_batched(
        self,
        prompts: list[Prompt],
        winsorization_quantile: float = 1.0,
    ) -> Array:
        raise NotImplementedError

    def get_residuals_mean(
        self,
        prompts: list[Prompt],
        winsorization_quantile: float = 1.0,
    ) -> np.ndarray:
        raise NotImplementedError

    def get_module_io(self, prompts: list[Prompt]) -> ModuleIO:
        raise NotImplementedError

    def get_module_io_batched(self, prompts: list[Prompt]) -> ModuleIO:
        raise NotImplementedError

    def stream_chat_response(self, chat: list[dict[str, str]]) -> str:
        raise NotImplementedError

    def lm_eval_state(self) -> LMState:
        raise NotImplementedError

    def save_merged(self, directory: str) -> None:
        raise NotImplementedError

    def save_adapter(self, directory: str) -> None:
        raise NotImplementedError

    def push_to_hub(
        self,
        repo_id: str,
        *,
        private: bool,
        token: str,
        strategy: ExportStrategy,
    ) -> None:
        raise NotImplementedError
