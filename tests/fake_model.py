# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

"""
An in-memory stand-in for the model facade (`heretic_tpu.model.Model`), for testing
modifiers without the engine.

It implements the parts of the facade API that modifiers use, with the semantics
docs/DESIGN.md gives them ("Model facade" and "Parameter layout, components and
LoRA"), on weights held in memory instead of a checkpoint. No forward pass exists:
residuals and module inputs are synthetic (or supplied by the test), and module
outputs are computed from the weights and the current adapters.

Import with `from tests.fake_model import FakeModel, fake_settings`.

API
---

`fake_settings(**fields) -> Settings`
    Heretic settings for tests, built without reading the command line, the
    environment or config.toml. `model` defaults to "fake/model" and `seed` to 0.

`FakeModel(settings, weights, *, residuals=None, module_io=None)`
    `weights` maps each abliterable component to its base weights
    `[L, M, d_out, d_in]` (NumPy, including ml_dtypes.bfloat16, or JAX); the model
    dtype is theirs. Every component must have the same L, and d_out is the hidden
    size D.

    `residuals(prompts) -> [N, L + 1, D]` gives the per-prompt residuals (float32)
    that `get_residuals_mean` averages. By default they are standard normal,
    determined by the text of each prompt.

    `module_io(prompts) -> ModuleIO` replaces the default module I/O, for example
    with I/O captured from a real model. By default the inputs of every module are
    standard normal rounded to the model dtype, determined by the text of each
    prompt, and the outputs are those of the adapted module (see below).

`FakeModel.random(settings, *, layer_count=4, module_count=1, shapes=None,
dtype=jnp.bfloat16, weight_seed=0, **kwargs) -> FakeModel`
    A model with random weights N(0, 1 / d_in) rounded to `dtype`. `shapes` maps
    components to `(d_out, d_in)` and defaults to
    `{"attn.o_proj": (64, 64), "mlp.down_proj": (64, 128)}`. `kwargs` are passed
    to the constructor.

Members, as in the facade: `settings`, `dtype`, `lora_rank`, `get_layers()`,
`get_abliterable_components()`, `get_module_count(component)`,
`get_base_weights(component)`, `apply_lora(rank)`, `get_lora(component)`,
`set_lora(component, A, B)`, `reset_model()`, `get_residuals_mean(prompts,
winsorization_quantile=1.0)` and `get_module_io_batched(prompts)`.

* `apply_lora(r)` initialises `A [L, M, r, d_in]` like PEFT (uniform in
  ±1/sqrt(d_in)) from `fold_in(fold_in(key(seed), 1), i)` for the i-th component in
  sorted order, and `B [L, M, d_out, r]` with zeros, both float32.
  `reset_model()` does the same at the current rank and returns True. Setting
  `fast_reset = False` makes it take the slow path instead: the adapters are
  dropped, `lora_rank` becomes None, and it returns False.
* `set_lora` checks shapes, casts to float32 and counts its calls per component in
  `set_lora_calls`.
* Module outputs are computed like the engine's adapted modules:
  `base(x) = x @ Wᵀ` with float32 accumulation, rounded to the model dtype, and with
  adapters `(base(x).astype(f32) + (x.astype(f32) @ Aᵀ) @ Bᵀ).astype(dtype)`.
* `get_residuals_mean` winsorises each prompt's residuals per layer as the facade
  does, accumulates in float64 and returns float32.
* With `settings.offload_outputs_to_cpu` true (the default), module I/O is NumPy;
  otherwise it is JAX arrays.
"""

import hashlib
from collections.abc import Callable, Mapping
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from heretic_tpu.config import Settings
from heretic_tpu.model import Array, ModuleIO
from heretic_tpu.utils import Prompt

DEFAULT_SHAPES = {"attn.o_proj": (64, 64), "mlp.down_proj": (64, 128)}


def fake_settings(**fields: Any) -> Settings:
    # model_construct skips the settings sources (command line, environment and
    # config.toml), which would otherwise pick up pytest's arguments.
    return Settings.model_construct(**({"model": "fake/model", "seed": 0} | fields))


def prompt_rng(prompt: Prompt, purpose: str) -> np.random.Generator:
    """A random generator determined by the text of a prompt (not by Python's
    per-process string hashing)."""

    digest = hashlib.sha256(f"{purpose}\0{prompt.system}\0{prompt.user}".encode())
    return np.random.default_rng(int.from_bytes(digest.digest()[:8], "little"))


class FakeModel:
    def __init__(
        self,
        settings: Settings,
        weights: Mapping[str, Array],
        *,
        residuals: Callable[[list[Prompt]], np.ndarray] | None = None,
        module_io: Callable[[list[Prompt]], ModuleIO] | None = None,
    ):
        self.settings = settings
        self.weights = {
            component: jnp.asarray(weights[component]) for component in sorted(weights)
        }
        self.residuals = residuals or self.random_residuals
        self.module_io = module_io or self.adapted_module_io

        first = next(iter(self.weights.values()))
        self.dtype = first.dtype
        self.layer_count = first.shape[0]
        self.hidden_size = first.shape[2]
        for component, W in self.weights.items():
            assert W.ndim == 4, component
            assert W.dtype == self.dtype, component
            assert W.shape[0] == self.layer_count, component
            assert W.shape[2] == self.hidden_size, component

        self.lora_rank: int | None = None
        self.adapters: dict[str, tuple[jax.Array, jax.Array]] | None = None
        self.fast_reset = True
        self.set_lora_calls = {component: 0 for component in self.weights}

    @classmethod
    def random(
        cls,
        settings: Settings,
        *,
        layer_count: int = 4,
        module_count: int = 1,
        shapes: Mapping[str, tuple[int, int]] | None = None,
        dtype: Any = jnp.bfloat16,
        weight_seed: int = 0,
        **kwargs: Any,
    ) -> "FakeModel":
        rng = np.random.default_rng(weight_seed)
        weights = {}
        for component, (d_out, d_in) in (shapes or DEFAULT_SHAPES).items():
            W = rng.standard_normal((layer_count, module_count, d_out, d_in))
            W /= np.sqrt(d_in)
            weights[component] = W.astype(dtype)
        return cls(settings, weights, **kwargs)

    # Structure

    def get_layers(self) -> list[int]:
        return list(range(self.layer_count))

    def get_abliterable_components(self) -> list[str]:
        return list(self.weights)

    def get_module_count(self, component: str) -> int:
        return self.weights[component].shape[1]

    def get_base_weights(self, component: str) -> jax.Array:
        return self.weights[component]

    # Adapters

    def apply_lora(self, lora_rank: int) -> None:
        root = jax.random.key(self.settings.seed)
        self.adapters = {}
        for i, (component, W) in enumerate(self.weights.items()):
            layer_count, module_count, d_out, d_in = W.shape
            bound = 1 / np.sqrt(d_in)
            A = jax.random.uniform(
                jax.random.fold_in(jax.random.fold_in(root, 1), i),
                (layer_count, module_count, lora_rank, d_in),
                jnp.float32,
                -bound,
                bound,
            )
            B = jnp.zeros((layer_count, module_count, d_out, lora_rank), jnp.float32)
            self.adapters[component] = (A, B)
        self.lora_rank = lora_rank

    def get_lora(self, component: str) -> tuple[jax.Array, jax.Array]:
        assert self.adapters is not None, "apply_lora() has not been called"
        return self.adapters[component]

    def set_lora(self, component: str, A: Array, B: Array) -> None:
        old_A, old_B = self.get_lora(component)
        if A.shape != old_A.shape or B.shape != old_B.shape:
            raise ValueError(
                f"Adapter shapes {A.shape}, {B.shape} for {component} do not match "
                f"{old_A.shape}, {old_B.shape}"
            )
        self.adapters[component] = (
            jnp.asarray(A, jnp.float32),
            jnp.asarray(B, jnp.float32),
        )
        self.set_lora_calls[component] += 1

    def reset_model(self) -> bool:
        if not self.fast_reset:
            self.adapters = None
            self.lora_rank = None
            return False
        assert self.lora_rank is not None, "apply_lora() has not been called"
        self.apply_lora(self.lora_rank)
        return True

    # Inference

    def random_residuals(self, prompts: list[Prompt]) -> np.ndarray:
        return np.stack(
            [
                prompt_rng(prompt, "residuals")
                .standard_normal((self.layer_count + 1, self.hidden_size))
                .astype(np.float32)
                for prompt in prompts
            ]
        )

    def get_residuals_mean(
        self,
        prompts: list[Prompt],
        winsorization_quantile: float = 1.0,
    ) -> np.ndarray:
        residuals = np.asarray(self.residuals(prompts), np.float32)
        if 0 <= winsorization_quantile < 1:
            thresholds = np.quantile(
                np.abs(residuals),
                winsorization_quantile,
                axis=-1,
                keepdims=True,
                method="linear",
            ).astype(np.float32)
            residuals = np.clip(residuals, -thresholds, thresholds)
        return (np.sum(residuals, axis=0, dtype=np.float64) / len(prompts)).astype(
            np.float32
        )

    def adapted_module_io(self, prompts: list[Prompt]) -> ModuleIO:
        module_io = {}
        for component, W in self.weights.items():
            layer_count, module_count, _, d_in = W.shape
            # [L, M, N, d_in], rounded to the model dtype.
            inputs = np.stack(
                [
                    prompt_rng(prompt, component).standard_normal(
                        (layer_count, module_count, d_in)
                    )
                    for prompt in prompts
                ],
                axis=2,
            ).astype(self.dtype)
            module_io[component] = (inputs, self.adapted_outputs(component, inputs))
        return module_io

    def adapted_outputs(self, component: str, inputs: np.ndarray) -> np.ndarray:
        """The outputs `[L, M, N, d_out]` of a component's adapted modules."""

        # Products of model-dtype values are exact in float32, so float32
        # matrix products give the base output with float32 accumulation.
        x = inputs.astype(np.float32)
        W = np.asarray(self.weights[component], np.float32)
        outputs = (x @ np.swapaxes(W, -1, -2)).astype(inputs.dtype)
        if self.adapters is not None:
            A, B = (np.asarray(m) for m in self.adapters[component])
            delta = (x @ np.swapaxes(A, -1, -2)) @ np.swapaxes(B, -1, -2)
            outputs = (outputs.astype(np.float32) + delta).astype(inputs.dtype)
        return outputs

    def get_module_io_batched(self, prompts: list[Prompt]) -> ModuleIO:
        module_io = self.module_io(prompts)
        if self.settings.offload_outputs_to_cpu:
            return {
                component: (np.asarray(inputs), np.asarray(outputs))
                for component, (inputs, outputs) in module_io.items()
            }
        return {
            component: (jnp.asarray(inputs), jnp.asarray(outputs))
            for component, (inputs, outputs) in module_io.items()
        }
