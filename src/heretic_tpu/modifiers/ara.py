# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

# Arbitrary-Rank Ablation (ARA) (Weidmann 2026)
# See https://github.com/p-e-w/heretic/pull/211 for more information.

import functools
from dataclasses import asdict, dataclass
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array, lax
from optuna import Trial
from pydantic import (
    BaseModel,
    Field,
    PositiveInt,
)

from heretic_tpu.backend import lbfgs, linalg
from heretic_tpu.config import DatasetSpecification, SingleDatasetSpecification
from heretic_tpu.modifier import Context, Modifier, Serializable
from heretic_tpu.utils import format_dataset_specification, print

# The largest neighbour count that suggest_parameters samples. ara_optimise always
# ranks this many nearest neighbours and averages a traced prefix of them,
# so that every neighbour count shares one compiled program.
NEIGHBOR_COUNT_MAX = 15


@dataclass
class Parameters(Serializable):
    start_layer_index: int
    end_layer_index: int
    preserve_good_behavior_weight: float
    steer_bad_behavior_weight: float
    overcorrect_relative_weight: float
    neighbor_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_presentation_dict(self) -> dict[str, str]:
        return {
            name: (f"{value:.4f}" if isinstance(value, float) else f"{value}")
            for name, value in asdict(self).items()
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Serializable":
        return Parameters(**data)


class Settings(BaseModel):
    good_prompts: DatasetSpecification = Field(
        default=SingleDatasetSpecification(
            dataset="mlabonne/harmless_alpaca",
            split="train[:400]",
            column="text",
        ),
        description="Dataset of prompts that tend to produce desirable responses.",
    )

    bad_prompts: DatasetSpecification = Field(
        default=SingleDatasetSpecification(
            dataset="mlabonne/harmful_behaviors",
            split="train[:400]",
            column="text",
        ),
        description="Dataset of prompts that tend to produce undesirable responses.",
    )

    preserve_row_magnitudes: bool = Field(
        default=True,
        description=(
            "Whether to renormalize the rows of the modified matrices to preserve "
            "the original matrices' row magnitudes. This is believed to improve "
            'intelligence retention (see Lai 2025, "Magnitude-Preserving Orthogonal Ablation").'
        ),
    )

    lora_rank: PositiveInt = Field(
        default=50,
        description=(
            "The rank of the LoRA adapter to use. "
            'While mathematically, ARA is of "arbitrary" rank, experiments have shown that '
            "singular values tend to drop rapidly after a few dozen dimensions, and approximating "
            "the full transformation with a LoRA has many practical advantages."
        ),
    )

    n_optimization_steps: PositiveInt = Field(
        default=5,
        description="Number of (outer) L-BFGS optimization steps to perform.",
    )

    learning_rate: float = Field(
        default=1.0,
        description="Learning rate to use in the L-BFGS optimizer.",
    )

    max_iter: PositiveInt = Field(
        default=20,
        description="Maximum number of (inner) iterations to perform per (outer) L-BFGS optimization step.",
    )

    history_size: PositiveInt = Field(
        default=10,
        description="Number of past updates to store for approximating the Hessian matrix in the L-BFGS optimizer.",
    )

    print_loss: bool = Field(
        default=False,
        description="Whether to print the loss value for each L-BFGS optimization step.",
    )


class ObjectiveData(NamedTuple):
    """
    The arrays that ARA's objective for a module depends on (built by
    `objective_data`).
    """

    W_base: Array  # [d_out, d_in], the base weight in float32
    W_row_norms_squared: Array  # [d_out]
    W_row_norms: Array  # [d_out]
    good_input: Array  # [N_good, d_in]
    good_output: Array  # [N_good, d_out]
    bad_input: Array  # [N_bad, d_in]
    bad_output: Array  # [N_bad, d_out]
    # The outputs of the base weight, good_input @ W_baseᵀ and bad_input @ W_baseᵀ.
    good_base_output: Array  # [N_good, d_out]
    bad_base_output: Array  # [N_bad, d_out]
    # [3]: preserve_good_behavior_weight, steer_bad_behavior_weight
    # and overcorrect_relative_weight.
    loss_weights: Array
    neighbor_count: Array  # int32 []


def objective_data(
    W_base: Array,
    good_input: Array,
    good_output: Array,
    bad_input: Array,
    bad_output: Array,
    loss_weights: Array,
    neighbor_count: Array,
) -> ObjectiveData:
    """
    The objective's data for a module, with the quantities that do not depend on
    the adapter computed once.
    """

    W_row_norms_squared = jnp.sum(W_base * W_base, axis=1)

    return ObjectiveData(
        W_base=W_base,
        W_row_norms_squared=W_row_norms_squared,
        # Pre-calculate the original row norms to preserve them.
        # See https://huggingface.co/blog/grimjim/norm-preserving-biprojected-abliteration
        W_row_norms=jnp.sqrt(W_row_norms_squared),
        good_input=good_input,
        good_output=good_output,
        bad_input=bad_input,
        bad_output=bad_output,
        good_base_output=jnp.matmul(
            good_input, W_base.T, precision=lax.Precision.HIGHEST
        ),
        bad_base_output=jnp.matmul(
            bad_input, W_base.T, precision=lax.Precision.HIGHEST
        ),
        loss_weights=loss_weights,
        neighbor_count=neighbor_count,
    )


# The objective function at the heart of ARA.
# `k_max_good` and `k_max_bad` are static upper bounds of the traced
# `neighbor_count` (see linalg.knn_mean).
def ara_loss(
    good_output: Array,
    bad_output: Array,
    new_good_output: Array,
    new_bad_output: Array,
    loss_weights: Array,
    neighbor_count: Array,
    k_max_good: int,
    k_max_bad: int,
) -> Array:
    (
        preserve_good_behavior_weight,
        steer_bad_behavior_weight,
        overcorrect_relative_weight,
    ) = loss_weights

    # The outputs for "good" prompts should change as little as possible.
    preserve_good_behavior = jnp.mean((new_good_output - good_output) ** 2)

    steer_bad_behavior = (
        # Pull the outputs for "bad" prompts towards
        # the original outputs for "good" prompts.
        jnp.mean(
            linalg.knn_mean(
                new_bad_output,
                good_output,
                neighbor_count,
                k_max_good,
            )
        )
        # Push the outputs for "bad" prompts away from
        # the original outputs for "bad" prompts.
        # In combination with the above, this overcorrects
        # away from the original residuals, which results
        # in stronger steering that can overcome more complex
        # refusal mechanisms.
        + overcorrect_relative_weight
        * -jnp.mean(
            linalg.knn_mean(
                new_bad_output,
                bad_output,
                neighbor_count,
                k_max_bad,
            )
        )
    )

    return (
        preserve_good_behavior_weight * preserve_good_behavior
        + steer_bad_behavior_weight * steer_bad_behavior
    )


def _sqrt(squared: Array) -> Array:
    # The square root with a zero (not NaN) gradient where its argument is not
    # positive, as linalg.normalize's norm has at a zero row. The squared norm of a
    # row of W_eff can round below zero only where that row (nearly) vanishes.
    positive = squared > 0
    return jnp.where(positive, jnp.sqrt(jnp.where(positive, squared, 1)), 0)


def objective(
    x: Array,
    data: ObjectiveData,
    *,
    preserve_row_magnitudes: bool,
    k_max_good: int,
    k_max_bad: int,
) -> Array:
    """
    ARA's loss for a module, as a function of its flattened LoRA matrices
    `x = concat(A.ravel(), B.ravel())`.
    """

    d_out, d_in = data.W_base.shape
    rank = x.size // (d_in + d_out)
    A = x[: rank * d_in].reshape(rank, d_in)
    B = x[rank * d_in :].reshape(d_out, rank)

    def matmul(a: Array, b: Array) -> Array:
        return jnp.matmul(a, b, precision=lax.Precision.HIGHEST)

    # Upstream forms the effective weight W_eff = W_base + B @ A, optionally
    # normalises its rows to the original row norms, and multiplies the inputs by it.
    # The same outputs are computed here through the adapter's rank-r structure,
    # which only reassociates the float32 arithmetic. The inputs' products with
    # W_base are fixed for the module (see objective_data), and forming W_eff and
    # multiplying the inputs by it in every evaluation and its gradient cost
    # O(N d_in d_out) rather than O((N + d_out) d_in r): for a 4B model's down_proj,
    # about 90 GFLOP per evaluation instead of 7 (besides the k-NN distances).
    #
    # Outputs using the effective weight: inputs @ W_effᵀ.
    new_good_output = data.good_base_output + matmul(matmul(data.good_input, A.T), B.T)
    new_bad_output = data.bad_base_output + matmul(matmul(data.bad_input, A.T), B.T)

    if preserve_row_magnitudes:
        # Normalising row i of W_eff to unit length and scaling it by the original
        # norm ‖w_i‖ scales output i by ‖w_i‖ / max(‖w_eff_i‖, 1e-12)
        # (F.normalize's clamp), where
        # ‖w_eff_i‖² = ‖w_i‖² + B_i · (2 W_base Aᵀ + B A Aᵀ)_i.
        W_eff_row_norms = _sqrt(
            data.W_row_norms_squared
            + jnp.sum(
                B * (2 * matmul(data.W_base, A.T) + matmul(B, matmul(A, A.T))),
                axis=1,
            )
        )
        scale = data.W_row_norms / jnp.maximum(W_eff_row_norms, 1e-12)
        new_good_output = new_good_output * scale
        new_bad_output = new_bad_output * scale

    return ara_loss(
        data.good_output,
        data.bad_output,
        new_good_output,
        new_bad_output,
        data.loss_weights,
        data.neighbor_count,
        k_max_good,
        k_max_bad,
    )


@functools.partial(
    jax.jit,
    static_argnames=[
        "preserve_row_magnitudes",
        "max_iter",
        "history_size",
        "learning_rate",
        "k_max_good",
        "k_max_bad",
    ],
    donate_argnames=["A", "B", "state"],
)
def ara_optimise(
    A: Array,
    B: Array,
    W_stack: Array,
    layer: Array,
    module: Array,
    good_in: Array,
    good_out: Array,
    bad_in: Array,
    bad_out: Array,
    loss_weights: Array,
    k: Array,
    state: lbfgs.LBFGSState,
    *,
    preserve_row_magnitudes: bool,
    max_iter: int,
    history_size: int,
    learning_rate: float,
    k_max_good: int,
    k_max_bad: int,
) -> tuple[Array, Array, lbfgs.LBFGSState, Array]:
    """
    One (outer) L-BFGS optimisation step on ARA's objective for a module,
    as `torch.optim.LBFGS.step` with the strong Wolfe line search performs it.

    `A [r, d_in]` and `B [d_out, r]` are the module's LoRA matrices (float32),
    and `state` is the optimizer state it carries across steps. These three are
    donated. `W_stack` holds the base weights of the component `[L, M, d_out, d_in]`,
    of which the module's is `W_stack[layer, module]`. The module I/O
    (`good_in [N_good, d_in]`, `good_out [N_good, d_out]`, `bad_in [N_bad, d_in]`,
    `bad_out [N_bad, d_out]`) is float32, `loss_weights` holds the weights of
    `ara_loss` and `k` is the neighbour count (int32), where
    `1 <= k <= k_max_good <= N_good` and `k <= k_max_bad <= N_bad`.

    Only the keyword arguments are static, so a single compiled program serves
    every module of a component, every trial and every neighbour count.

    Returns the new A, B and state, and the loss before the step.
    """

    # We need the base weight in float32 (the dtype of the adapters)
    # to compute the outputs and row norms of the effective weight.
    W_base = W_stack[layer, module].astype(A.dtype)

    data = objective_data(W_base, good_in, good_out, bad_in, bad_out, loss_weights, k)

    # Like PyTorch's optimisers, L-BFGS works on the parameters
    # flattened and concatenated in order.
    x, state, loss = lbfgs.step(
        functools.partial(
            objective,
            preserve_row_magnitudes=preserve_row_magnitudes,
            k_max_good=k_max_good,
            k_max_bad=k_max_bad,
        ),
        jnp.concatenate([A.ravel(), B.ravel()]),
        state,
        data,
        lr=learning_rate,
        max_iter=max_iter,
        history_size=history_size,
    )

    return x[: A.size].reshape(A.shape), x[A.size :].reshape(B.shape), state, loss


class ARA(Modifier[Parameters]):
    settings: Settings

    @property
    def reproducible(self) -> bool:
        return True

    @property
    def modifier_name(self) -> str:
        return "Arbitrary-Rank Ablation (ARA)"

    def init(self, ctx: Context) -> None:
        model = ctx.get_model()

        print()
        print(
            f"Loading good prompts from [bold]{format_dataset_specification(self.settings.good_prompts)}[/]..."
        )
        good_prompts = ctx.load_prompts(self.settings.good_prompts)
        print(f"* [bold]{len(good_prompts)}[/] prompts loaded")

        print()
        print(
            f"Loading bad prompts from [bold]{format_dataset_specification(self.settings.bad_prompts)}[/]..."
        )
        bad_prompts = ctx.load_prompts(self.settings.bad_prompts)
        print(f"* [bold]{len(bad_prompts)}[/] prompts loaded")

        print()
        print("Obtaining module I/O for good prompts...")
        self.good_module_io = model.get_module_io_batched(good_prompts)

        print()
        print("Obtaining module I/O for bad prompts...")
        self.bad_module_io = model.get_module_io_batched(bad_prompts)

        # LoRA B matrices are initialized to zero by default in PEFT,
        # so we don't need to do anything manually.
        model.apply_lora(self.settings.lora_rank)

    def suggest_parameters(self, ctx: Context, trial: Trial) -> Parameters:
        layer_count = len(ctx.get_model().get_layers())

        start_layer_index = trial.suggest_int(
            "start_layer_index",
            0,
            layer_count // 2,
        )
        end_layer_index = trial.suggest_int(
            "end_layer_index",
            layer_count // 2,
            layer_count,
        )
        preserve_good_behavior_weight = trial.suggest_float(
            "preserve_good_behavior_weight",
            0.0,
            1.0,
        )
        steer_bad_behavior_weight = trial.suggest_float(
            "steer_bad_behavior_weight",
            0.0001,
            1.0,
            log=True,
        )
        overcorrect_relative_weight = trial.suggest_float(
            "overcorrect_relative_weight",
            0.0,
            1.3,
        )
        neighbor_count = trial.suggest_int(
            "neighbor_count",
            1,
            NEIGHBOR_COUNT_MAX,
        )

        return Parameters(
            start_layer_index=start_layer_index,
            end_layer_index=end_layer_index,
            preserve_good_behavior_weight=preserve_good_behavior_weight,
            steer_bad_behavior_weight=steer_bad_behavior_weight,
            overcorrect_relative_weight=overcorrect_relative_weight,
            neighbor_count=neighbor_count,
        )

    def modify_model(self, ctx: Context, parameters: Parameters) -> None:
        model = ctx.get_model()

        loss_weights = np.array(
            [
                parameters.preserve_good_behavior_weight,
                parameters.steer_bad_behavior_weight,
                parameters.overcorrect_relative_weight,
            ],
            np.float32,
        )
        neighbor_count = parameters.neighbor_count

        components = model.get_abliterable_components()
        # Optimisation starts from the reset adapters (B = 0),
        # which modules outside the layer range keep.
        reset_adapters = {
            component: model.get_lora(component) for component in components
        }
        optimised_adapters: dict[str, dict[tuple[int, int], tuple[Array, Array]]] = {
            component: {} for component in components
        }

        for layer_index in range(
            parameters.start_layer_index,
            parameters.end_layer_index,
        ):
            for component in components:
                good_inputs, good_outputs = self.good_module_io[component]
                bad_inputs, bad_outputs = self.bad_module_io[component]
                good_count = good_inputs.shape[2]
                bad_count = bad_inputs.shape[2]

                # torch.topk raises for these in upstream.
                if not 1 <= neighbor_count <= min(good_count, bad_count):
                    raise ValueError(
                        f"neighbor_count is {neighbor_count}, but must be between 1 and "
                        f"the number of good and bad prompts ({min(good_count, bad_count)})"
                    )

                # A single compiled program serves every neighbour count
                # up to NEIGHBOR_COUNT_MAX.
                k_max = max(neighbor_count, NEIGHBOR_COUNT_MAX)

                reset_A, reset_B = reset_adapters[component]
                W_stack = model.get_base_weights(component)

                for module_index in range(model.get_module_count(component)):
                    # We optimize the LoRA weights A and B.
                    lora_A = reset_A[layer_index, module_index]
                    lora_B = reset_B[layer_index, module_index]
                    state = lbfgs.init(
                        lora_A.size + lora_B.size,
                        self.settings.history_size,
                    )
                    # Every step returns committed arrays, and JAX compiles separate
                    # programs for committed and uncommitted arguments, so the first
                    # step's are committed too (B is replicated in every plan).
                    lora_A, lora_B, state = jax.device_put(
                        (lora_A, lora_B, state),
                        lora_B.sharding,
                    )

                    # Move I/O tensors to the device, in float32.
                    good_input = jnp.asarray(
                        good_inputs[layer_index, module_index], jnp.float32
                    )
                    good_output = jnp.asarray(
                        good_outputs[layer_index, module_index], jnp.float32
                    )
                    bad_input = jnp.asarray(
                        bad_inputs[layer_index, module_index], jnp.float32
                    )
                    bad_output = jnp.asarray(
                        bad_outputs[layer_index, module_index], jnp.float32
                    )

                    for step in range(self.settings.n_optimization_steps):
                        lora_A, lora_B, state, loss = ara_optimise(
                            lora_A,
                            lora_B,
                            W_stack,
                            np.int32(layer_index),
                            np.int32(module_index),
                            good_input,
                            good_output,
                            bad_input,
                            bad_output,
                            loss_weights,
                            np.int32(neighbor_count),
                            state,
                            preserve_row_magnitudes=self.settings.preserve_row_magnitudes,
                            max_iter=self.settings.max_iter,
                            history_size=self.settings.history_size,
                            learning_rate=self.settings.learning_rate,
                            k_max_good=min(k_max, good_count),
                            k_max_bad=min(k_max, bad_count),
                        )
                        if self.settings.print_loss:
                            print(
                                f"\\[{layer_index}/{component}/{module_index}] Step: {step + 1}, Loss: {loss.item():.6f}"
                            )

                    # Finish the module before starting the next one, so that only
                    # one module's I/O is on the device at a time.
                    jax.block_until_ready((lora_A, lora_B))
                    optimised_adapters[component][layer_index, module_index] = (
                        lora_A,
                        lora_B,
                    )

        for component, adapters in optimised_adapters.items():
            if not adapters:
                continue

            # Every module's adapters, optimised or reset, are stacked, so that the
            # stacked arrays have the same shapes in every trial, whatever its layer
            # range, and compile only once.
            reset_A, reset_B = reset_adapters[component]
            layer_count, module_count = reset_A.shape[:2]
            modules = [
                adapters[layer_index, module_index]
                if (layer_index, module_index) in adapters
                else (
                    reset_A[layer_index, module_index],
                    reset_B[layer_index, module_index],
                )
                for layer_index in range(layer_count)
                for module_index in range(module_count)
            ]
            model.set_lora(
                component,
                jnp.stack([A for A, _ in modules]).reshape(reset_A.shape),
                jnp.stack([B for _, B in modules]).reshape(reset_B.shape),
            )

    def reset_model(self, ctx: Context) -> None:
        model = ctx.get_model()
        fast_path = model.reset_model()
        if not fast_path:
            model.apply_lora(self.settings.lora_rank)
