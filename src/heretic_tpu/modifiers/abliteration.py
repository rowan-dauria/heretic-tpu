# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import functools
import math
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, cast

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

from heretic_tpu.backend import linalg
from heretic_tpu.config import DatasetSpecification, SingleDatasetSpecification
from heretic_tpu.modifier import Context, Modifier, Serializable
from heretic_tpu.utils import format_dataset_specification, print


@dataclass
class WeightDistribution:
    max_weight: float
    max_weight_position: float
    min_weight: float
    min_weight_distance: float


@dataclass
class Parameters(Serializable):
    direction_index: float | None
    weight_distributions: dict[str, WeightDistribution]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_presentation_dict(self) -> dict[str, str]:
        parameters = {}

        parameters["direction_index"] = (
            "per layer"
            if (self.direction_index is None)
            else f"{self.direction_index:.2f}"
        )

        for component, weight_distribution in self.weight_distributions.items():
            for name, value in asdict(weight_distribution).items():
                parameters[f"{component}.{name}"] = f"{value:.2f}"

        return parameters

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Serializable":
        return Parameters(
            direction_index=data["direction_index"],
            weight_distributions={
                component: WeightDistribution(**weight_distribution)
                for component, weight_distribution in data[
                    "weight_distributions"
                ].items()
            },
        )


class RowNormalization(str, Enum):
    NONE = "none"
    PRE = "pre"
    # POST = "post"  # Theoretically possible, but provides no advantage.
    FULL = "full"


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

    orthogonalize_direction: bool = Field(
        default=True,
        description=(
            "Whether to adjust the residual directions so that only the component that is "
            "orthogonal to the good direction is subtracted during abliteration."
        ),
    )

    row_normalization: RowNormalization = Field(
        default=RowNormalization.FULL,
        description=(
            "How to apply row normalization of the weights. Options: "
            '"none" (no normalization), '
            '"pre" (compute LoRA adapter relative to row-normalized weights), '
            '"full" (like "pre", but renormalizes to preserve original row magnitudes).'
        ),
    )

    full_normalization_lora_rank: PositiveInt = Field(
        default=3,
        description=(
            'The rank of the LoRA adapter to use when "full" row normalization is used. '
            "Row magnitude preservation is approximate due to non-linear effects, "
            "and this determines the rank of that approximation. Higher ranks produce "
            "larger output files and may slow down evaluation."
        ),
    )

    winsorization_quantile: float = Field(
        default=1.0,
        description=(
            "The symmetric winsorization to apply to the per-prompt, per-layer residual vectors, "
            "expressed as the quantile to clamp to (between 0 and 1). Disabled by default. "
            'This can tame so-called "massive activations" that occur in some models. '
            "Example: winsorization_quantile = 0.95 computes the 0.95-quantile of the absolute values "
            "of the components, then clamps the magnitudes of all components to that quantile."
        ),
    )


# F.normalize(x, p=2, dim=axis) for the residual directions,
# which are computed on the host in the dtype of x (float32).
def normalize(x: np.ndarray, axis: int) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=axis, keepdims=True), 1e-12)


# torch.lerp with a scalar weight, in the dtype of start and end.
# Like PyTorch, it switches between two formulas at 0.5 for accuracy.
def lerp(start: np.ndarray, end: np.ndarray, weight: float) -> np.ndarray:
    weight = start.dtype.type(weight)
    if abs(weight) < 0.5:
        return start + weight * (end - start)
    else:
        return end - (end - start) * (1 - weight)


@functools.partial(jax.jit, static_argnames=["row_normalization"])
def get_adapters(
    W: Array,
    residual_directions: Array,
    weights: Array,
    apply: Array,
    A_reset: Array,
    key: Array,
    *,
    row_normalization: RowNormalization,
) -> tuple[Array, Array]:
    """
    Computes the abliteration LoRA adapters of all modules of a component.

    `W` holds the base weights of the component's modules `[L, M, d_out, d_in]`
    in the model dtype, `residual_directions` the direction to ablate in each
    layer `[L, d_out]`, `weights` the ablation weight of each layer `[L]`,
    `apply` whether to ablate each layer `[L]`, and `A_reset` the reset LoRA
    A matrices `[L, M, r, d_in]`. `key` seeds the randomised SVD that "full"
    row normalisation uses.

    Returns the LoRA matrices `(A [L, M, r, d_in], B [L, M, d_out, r])`
    in float32. Modules that are not ablated keep `A_reset` and get `B = 0`.
    """

    layer_count, module_count, d_out, d_in = W.shape
    r = A_reset.shape[2]

    def ablate(
        W: Array,
        v: Array,
        weight: Array,
        A_reset: Array,
    ) -> tuple[Array, Array]:
        # LoRA abliteration: delta W = -lambda * v * (v^T W)
        # lora_B = -lambda * v
        # lora_A = v^T W
        W = W.astype(jnp.float32)

        if row_normalization == RowNormalization.FULL:
            # Keep a reference to the original weight matrix so we can subtract it later.
            W_org = W

        if row_normalization != RowNormalization.NONE:
            # Get the row norms.
            W_row_norms = jnp.linalg.norm(W, axis=1, keepdims=True)
            # Normalize the weight matrix along the rows.
            W = linalg.normalize(W, axis=1)

        # Calculate lora_A = v^T W
        # v is (d_out,), W is (d_out, d_in)
        # v @ W -> (d_in,)
        lora_A = jnp.matmul(v, W, precision=lax.Precision.HIGHEST).reshape(1, -1)

        # Calculate lora_B = -weight * v
        # v is (d_out,)
        lora_B = (-weight * v).reshape(-1, 1)

        if row_normalization == RowNormalization.PRE:
            # Make the LoRA adapter apply to the original weight matrix.
            lora_B = W_row_norms * lora_B
        elif row_normalization == RowNormalization.FULL:
            # Approximates https://huggingface.co/blog/grimjim/norm-preserving-biprojected-abliteration
            W = W + jnp.matmul(lora_B, lora_A, precision=lax.Precision.HIGHEST)
            # Normalize the adjusted weight matrix along the rows.
            W = linalg.normalize(W, axis=1)
            # Restore the original row norms of the weight matrix.
            W = W * W_row_norms
            # Subtract the original matrix to turn W into a delta.
            W = W - W_org
            # Use a low-rank SVD to get an approximation of the matrix.
            # svd_lowrank is randomised. Upstream reseeds immediately before every
            # call, so every module uses the same key.
            U, S, V = linalg.svd_lowrank(W, q=2 * r + 4, niter=6, key=key)

            # Truncate it to the part we want to store in the LoRA adapter.
            U = U[:, :r]
            S = S[:r]
            Vh = V[:, :r].T
            # Transfer it into the LoRA adapter components. Split the singular values
            # evenly between the two components to keep their norms balanced and avoid
            # potential issues with numerical stability.
            sqrt_S = jnp.sqrt(S)
            # U @ diag(sqrt_S) and diag(sqrt_S) @ Vh, which are exact scalings.
            lora_B = U * sqrt_S
            lora_A = sqrt_S[:, None] * Vh

        return lora_A, lora_B

    def skip(
        W: Array,
        v: Array,
        weight: Array,
        A_reset: Array,
    ) -> tuple[Array, Array]:
        # Leave the adapter at identity, without touching the weights.
        return A_reset, jnp.zeros((d_out, r), jnp.float32)

    def process_module(module: tuple[Array, ...]) -> tuple[Array, Array]:
        W, v, weight, apply, A_reset = module
        return lax.cond(apply, ablate, skip, W, v, weight, A_reset)

    # The modules are processed one at a time: lax.map without a batch size is
    # a scan, inside which lax.cond runs only the selected branch. (With a batch
    # size, lax.map would vmap the body, which turns the conditional into a select
    # that runs both branches.) So at most one module's float32 working set is
    # live, the weights are never converted to float32 as a whole, and modules
    # that are not ablated cost nothing.
    A, B = lax.map(
        process_module,
        (
            W.reshape(-1, d_out, d_in),
            jnp.repeat(residual_directions, module_count, axis=0),
            jnp.repeat(weights, module_count),
            jnp.repeat(apply, module_count),
            A_reset.reshape(-1, r, d_in),
        ),
    )

    return A.reshape(A_reset.shape), B.reshape(layer_count, module_count, d_out, r)


class Abliteration(Modifier[Parameters]):
    settings: Settings

    @property
    def reproducible(self) -> bool:
        return True

    @property
    def modifier_name(self) -> str:
        if (
            self.settings.orthogonalize_direction
            and self.settings.row_normalization == RowNormalization.FULL
        ):
            return "Magnitude-Preserving Orthogonal Ablation (MPOA)"
        elif self.settings.orthogonalize_direction:
            return "Projected Abliteration"
        else:
            return "Abliteration"

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
        print("Calculating per-layer residual directions...")

        print("* Obtaining residual mean for good prompts...")
        good_means = model.get_residuals_mean(
            good_prompts,
            winsorization_quantile=self.settings.winsorization_quantile,
        )
        print("* Obtaining residual mean for bad prompts...")
        bad_means = model.get_residuals_mean(
            bad_prompts,
            winsorization_quantile=self.settings.winsorization_quantile,
        )

        self.residual_directions = normalize(bad_means - good_means, axis=1)

        if self.settings.orthogonalize_direction:
            # Implements https://huggingface.co/blog/grimjim/projected-abliteration
            # Adjust the residual directions so that only the component that is
            # orthogonal to the good direction is subtracted during abliteration.
            good_directions = normalize(good_means, axis=1)
            projection_vector = np.sum(
                self.residual_directions * good_directions,
                axis=1,
            )
            self.residual_directions = (
                self.residual_directions - projection_vector[:, None] * good_directions
            )
            self.residual_directions = normalize(self.residual_directions, axis=1)

        if self.settings.row_normalization != RowNormalization.FULL:
            # Rank 1 is sufficient for directional ablation without renormalization.
            self.lora_rank = 1
        else:
            # Row magnitude preservation introduces nonlinear effects.
            self.lora_rank = self.settings.full_normalization_lora_rank

        # LoRA B matrices are initialized to zero by default in PEFT,
        # so we don't need to do anything manually.
        model.apply_lora(self.lora_rank)

    def suggest_parameters(self, ctx: Context, trial: Trial) -> Parameters:
        model = ctx.get_model()

        direction_scope = trial.suggest_categorical(
            "direction_scope",
            [
                "global",
                "per layer",
            ],
        )

        last_layer_index = len(model.get_layers()) - 1

        # Discrimination between "harmful" and "harmless" inputs is usually strongest
        # in layers slightly past the midpoint of the layer stack. See the original
        # abliteration paper (https://arxiv.org/abs/2406.11717) for a deeper analysis.
        #
        # Note that we always sample this parameter even though we only need it for
        # the "global" direction scope. The reason is that multivariate TPE doesn't
        # work with conditional or variable-range parameters.
        direction_index = trial.suggest_float(
            "direction_index",
            0.4 * last_layer_index,
            0.9 * last_layer_index,
        )

        if direction_scope == "per layer":
            direction_index = None

        weight_distributions = {}

        for component in model.get_abliterable_components():
            # The parameter ranges are based on experiments with various models
            # and much wider ranges. They are not set in stone and might have to be
            # adjusted for future models.
            #
            # The MLP gets a negative lower bound that is then clamped to 0, so the
            # optimizer can fully disable its ablation. The clamp puts a positive
            # probability mass on exactly 0 (the continuous sampler would otherwise
            # reach 0 with probability zero). Ablating the MLP is often unnecessary for
            # removing refusals and tends to damage model intelligence more than
            # ablating the attention output, so on many models the optimum is to leave
            # it (mostly) untouched. See issue #202.
            max_weight_lower_bound = -0.25 if component == "mlp.down_proj" else 0.8
            max_weight = max(
                0.0,
                trial.suggest_float(
                    f"{component}.max_weight",
                    max_weight_lower_bound,
                    1.5,
                ),
            )
            max_weight_position = trial.suggest_float(
                f"{component}.max_weight_position",
                0.6 * last_layer_index,
                1.0 * last_layer_index,
            )
            # For sampling purposes, min_weight is expressed as a fraction of max_weight,
            # again because multivariate TPE doesn't support variable-range parameters.
            # The value is transformed into the actual min_weight value below.
            min_weight = trial.suggest_float(
                f"{component}.min_weight",
                0.0,
                1.0,
            )
            min_weight_distance = trial.suggest_float(
                f"{component}.min_weight_distance",
                1.0,
                max(0.6 * last_layer_index, 1.0),
            )

            weight_distributions[component] = WeightDistribution(
                max_weight=max_weight,
                max_weight_position=max_weight_position,
                min_weight=(min_weight * max_weight),
                min_weight_distance=min_weight_distance,
            )

        return Parameters(
            direction_index=direction_index,
            weight_distributions=weight_distributions,
        )

    def modify_model(self, ctx: Context, parameters: Parameters) -> None:
        model = ctx.get_model()
        layer_count = len(model.get_layers())

        if parameters.direction_index is None:
            # The index must be shifted by 1 because the first element
            # of residual_directions is the direction for the embeddings.
            residual_directions = self.residual_directions[1:]
        else:
            # The index must be shifted by 1 because the first element
            # of residual_directions is the direction for the embeddings.
            weight, index = math.modf(parameters.direction_index + 1)
            residual_direction = normalize(
                lerp(
                    self.residual_directions[int(index)],
                    self.residual_directions[int(index) + 1],
                    weight,
                ),
                axis=0,
            )
            # Every layer gets the same direction, so that both direction scopes
            # share one compiled program.
            residual_directions = np.broadcast_to(
                residual_direction,
                (layer_count, residual_direction.size),
            )

        # svd_lowrank is randomised. Upstream reseeds immediately before every call
        # so restoring a trial is independent of RNG history. A key derived from the
        # seed alone does the same.
        key = jax.random.fold_in(jax.random.key(self.heretic_settings.seed), 2)

        # Note that some implementations of abliteration also orthogonalize
        # the embedding matrix, but it's unclear if that has any benefits.
        for component in model.get_abliterable_components():
            weight_distribution = parameters.weight_distributions[component]

            weights = np.zeros(layer_count, np.float32)
            apply = np.zeros(layer_count, bool)

            for layer_index in range(layer_count):
                # Type inference fails here for some reason.
                distance = cast(
                    float, abs(layer_index - weight_distribution.max_weight_position)
                )

                # Don't orthogonalize layers that are more than
                # min_weight_distance away from max_weight_position.
                if distance > weight_distribution.min_weight_distance:
                    continue

                # Interpolate linearly between max_weight and min_weight
                # over min_weight_distance.
                weight = weight_distribution.max_weight + (
                    distance / weight_distribution.min_weight_distance
                ) * (weight_distribution.min_weight - weight_distribution.max_weight)

                # A weight of 0 disables this component's ablation. reset_model() has
                # already left the adapter at identity, so skip the otherwise wasteful
                # decomposition (which would also be operating on a zero matrix).
                if weight == 0:
                    continue

                weights[layer_index] = weight
                apply[layer_index] = True

            A_reset, _ = model.get_lora(component)
            lora_A, lora_B = get_adapters(
                model.get_base_weights(component),
                residual_directions,
                weights,
                apply,
                A_reset,
                key,
                row_normalization=self.settings.row_normalization,
            )
            model.set_lora(component, lora_A, lora_B)

    def reset_model(self, ctx: Context) -> None:
        model = ctx.get_model()
        fast_path = model.reset_model()
        if not fast_path:
            model.apply_lora(self.lora_rank)
