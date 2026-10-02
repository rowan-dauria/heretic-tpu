# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

"""
The ARA modifier against upstream's implementation.

Upstream optimises the LoRA matrices of each module with torch.optim.LBFGS on its
objective; `upstream_optimise` ports that loop to PyTorch. In float64 the port
follows PyTorch's trajectory to rounding. In float32 no two implementations agree
on the trajectory: at B = 0 every bad output is its own nearest neighbour at a
distance that is mostly rounding noise, and the k-NN terms switch neighbours
discontinuously, so single-ulp differences grow into O(1) changes of A and B (see
tests/backend/test_lbfgs.py). Float32 runs are therefore compared by the losses they
reach.
"""

import functools
import re
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest

from heretic_tpu.backend import lbfgs
from heretic_tpu.config import SingleDatasetSpecification
from heretic_tpu.model import ModuleIO
from heretic_tpu.modifiers import ara
from heretic_tpu.modifiers.ara import (
    ARA,
    NEIGHBOR_COUNT_MAX,
    ObjectiveData,
    Parameters,
    Settings,
    ara_optimise,
)
from heretic_tpu.plugin import Context
from heretic_tpu.utils import Prompt, batchify, load_prompts
from tests.backend.test_precision import float32_dot_precisions
from tests.fake_model import FakeModel, fake_settings
from tests.test_abliteration import class_members, relative_error

UPSTREAM_MODULE = Path(__file__).parents[1] / "heretic/src/heretic/modifiers/ara.py"
PORT_MODULE = Path(__file__).parents[1] / "src/heretic_tpu/modifiers/ara.py"

GOOD_PROMPTS = [Prompt("You are a helpful assistant.", f"Good {i}") for i in range(40)]
BAD_PROMPTS = [Prompt("You are a helpful assistant.", f"Bad {i}") for i in range(30)]

GOOD_DATASET = "mlabonne/harmless_alpaca"
BAD_DATASET = "mlabonne/harmful_behaviors"


def serve_prompts(
    monkeypatch: pytest.MonkeyPatch,
    good_prompts: list[Prompt],
    bad_prompts: list[Prompt],
) -> None:
    """Makes the modifier load the given prompts as its good and bad prompts."""

    prompts = {GOOD_DATASET: good_prompts, BAD_DATASET: bad_prompts}
    monkeypatch.setattr(
        "heretic_tpu.plugin.load_prompts",
        lambda settings, specification: prompts[specification.dataset],
    )


@pytest.fixture
def fake_prompts(monkeypatch: pytest.MonkeyPatch) -> None:
    serve_prompts(monkeypatch, GOOD_PROMPTS, BAD_PROMPTS)


def make_modifier(model: FakeModel, **settings: Any) -> tuple[ARA, Context]:
    """Initialises the modifier on a model and resets the model, as main.py does."""

    modifier = ARA(heretic_settings=model.settings, settings=Settings(**settings))
    ctx = Context(settings=model.settings, model=model)  # ty:ignore[invalid-argument-type]
    modifier.init(ctx)
    modifier.reset_model(ctx)
    return modifier, ctx


def module_data(
    modifier: ARA,
    model: FakeModel,
    component: str,
    layer_index: int,
    module_index: int,
) -> tuple[np.ndarray, ...]:
    """The base weight and the module I/O of a module, in float32."""

    module = (layer_index, module_index)
    good_inputs, good_outputs = modifier.good_module_io[component]
    bad_inputs, bad_outputs = modifier.bad_module_io[component]
    return tuple(
        np.asarray(x[module], np.float32)
        for x in (
            model.get_base_weights(component),
            good_inputs,
            good_outputs,
            bad_inputs,
            bad_outputs,
        )
    )


def upstream_optimise(
    data: tuple[np.ndarray, ...],
    A: np.ndarray,
    B: np.ndarray,
    parameters: Parameters,
    settings: Settings,
    dtype: Any,
) -> tuple[list[float], np.ndarray, np.ndarray]:
    """
    Upstream's optimisation of a module in PyTorch, in the given dtype, from the
    given A and B. `data` holds the base weight and the module I/O. Returns the loss
    of every step and the final A and B.
    """

    torch = pytest.importorskip("torch")
    F = torch.nn.functional

    # Arrays fetched from a TPU can have a column-major layout, which torch.tensor
    # keeps, but LBFGS needs contiguous parameters (as PEFT's are).
    def tensor(x: np.ndarray, **kwargs: Any) -> Any:
        return torch.tensor(np.ascontiguousarray(x, dtype), **kwargs)

    W_base, good_input, good_output, bad_input, bad_output = map(tensor, data)
    W_row_norms = torch.linalg.vector_norm(W_base, dim=1, keepdim=True)
    lora_A = tensor(A, requires_grad=True)
    lora_B = tensor(B, requires_grad=True)

    def mean_distances_to_knn(a: Any, b: Any, k: int) -> Any:
        distances = torch.cdist(a, b)
        nearest_distances, _ = distances.topk(k, dim=1, largest=False)
        return nearest_distances.mean(1)

    def objective(A: Any, B: Any) -> Any:
        W_eff = W_base + (B @ A)
        if settings.preserve_row_magnitudes:
            W_eff = F.normalize(W_eff, p=2, dim=1) * W_row_norms
        new_good_output = good_input @ W_eff.T
        new_bad_output = bad_input @ W_eff.T
        preserve_good_behavior = ((new_good_output - good_output) ** 2).mean()
        steer_bad_behavior = (
            mean_distances_to_knn(
                new_bad_output, good_output, parameters.neighbor_count
            ).mean()
            + parameters.overcorrect_relative_weight
            * -mean_distances_to_knn(
                new_bad_output, bad_output, parameters.neighbor_count
            ).mean()
        )
        return (
            parameters.preserve_good_behavior_weight * preserve_good_behavior
            + parameters.steer_bad_behavior_weight * steer_bad_behavior
        )

    optimizer = torch.optim.LBFGS(
        [lora_A, lora_B],
        lr=settings.learning_rate,
        max_iter=settings.max_iter,
        history_size=settings.history_size,
        line_search_fn="strong_wolfe",
    )

    def closure() -> Any:
        optimizer.zero_grad()
        loss = objective(lora_A, lora_B)
        loss.backward()
        return loss

    losses = [
        optimizer.step(closure).item() for _ in range(settings.n_optimization_steps)
    ]
    return losses, lora_A.detach().numpy(), lora_B.detach().numpy()


def float64_loss(
    A: np.ndarray,
    B: np.ndarray,
    data: tuple[np.ndarray, ...],
    parameters: Parameters,
    preserve_row_magnitudes: bool,
) -> float:
    """ARA's loss computed in float64 (with distances from the Gram matrix, which is
    exact enough in float64)."""

    W_base, good_input, good_output, bad_input, bad_output = (
        np.asarray(x, np.float64) for x in data
    )
    W_eff = W_base + np.asarray(B, np.float64) @ np.asarray(A, np.float64)
    if preserve_row_magnitudes:
        W_eff *= np.linalg.norm(W_base, axis=1, keepdims=True) / np.linalg.norm(
            W_eff, axis=1, keepdims=True
        )
    new_good_output = good_input @ W_eff.T
    new_bad_output = bad_input @ W_eff.T

    def mean_distance_to_knn(b: np.ndarray) -> float:
        squared_distances = (
            np.sum(new_bad_output**2, axis=1)[:, None]
            + np.sum(b**2, axis=1)[None, :]
            - 2 * new_bad_output @ b.T
        )
        distances = np.sqrt(np.maximum(squared_distances, 0))
        nearest = np.sort(distances, axis=1)[:, : parameters.neighbor_count]
        return float(np.mean(nearest))

    weights = np.array(
        [
            parameters.preserve_good_behavior_weight,
            parameters.steer_bad_behavior_weight,
            parameters.overcorrect_relative_weight,
        ],
        np.float32,
    ).astype(np.float64)
    return weights[0] * np.mean((new_good_output - good_output) ** 2) + weights[1] * (
        mean_distance_to_knn(good_output)
        - weights[2] * mean_distance_to_knn(bad_output)
    )


def random_module(
    rng: np.random.Generator,
    d_out: int,
    d_in: int,
    good_count: int,
    bad_count: int,
    rank: int,
) -> tuple[tuple[np.ndarray, ...], np.ndarray, np.ndarray]:
    """
    A module's base weight and I/O, rounded to bfloat16 like captured I/O, with the
    bad inputs offset from the good ones, and adapters initialised as by apply_lora
    (A uniform, B zero).
    """

    def bf16(x: np.ndarray) -> np.ndarray:
        return x.astype(ml_dtypes.bfloat16).astype(np.float32)

    W_base = bf16(rng.standard_normal((d_out, d_in)) / np.sqrt(d_in))
    good_input = bf16(rng.standard_normal((good_count, d_in)))
    bad_input = bf16(rng.standard_normal((bad_count, d_in)) + 0.5)
    data = (
        W_base,
        good_input,
        bf16(good_input @ W_base.T),
        bad_input,
        bf16(bad_input @ W_base.T),
    )
    A = rng.uniform(-1, 1, (rank, d_in)) / np.sqrt(d_in)
    return data, A.astype(np.float32), np.zeros((d_out, rank), np.float32)


def test_unchanged_from_upstream() -> None:
    # Settings, parameters, their presentation, the search space, the prompt and
    # module I/O handling of init, and the modifier name are upstream's.
    if not UPSTREAM_MODULE.is_file():
        pytest.skip("the upstream source is not checked out")

    names = {
        "Parameters": None,
        "Settings": None,
        "ARA": [
            "reproducible",
            "modifier_name",
            "init",
            "suggest_parameters",
            "reset_model",
        ],
    }
    upstream = class_members(UPSTREAM_MODULE, names)
    port = class_members(PORT_MODULE, names)
    assert len(upstream) == 7

    # The upper bound of the neighbour count became a shared constant.
    assert NEIGHBOR_COUNT_MAX == 15
    port["ARA.suggest_parameters"] = port["ARA.suggest_parameters"].replace(
        "NEIGHBOR_COUNT_MAX", "15"
    )
    assert port == upstream


@pytest.mark.parametrize("preserve_row_magnitudes", [True, False])
# There are 12 bad prompts, so k_max_bad = 12 and the k-NN ranks all bad outputs.
# (With k = 12 the problem is so ill-conditioned that the float64 trajectories
# drift apart at the 1e-7 level within five steps.)
@pytest.mark.parametrize("neighbor_count", [1, 5, 11])
def test_ara_optimise_matches_torch_in_float64(
    preserve_row_magnitudes: bool,
    neighbor_count: int,
) -> None:
    rng = np.random.default_rng(neighbor_count)
    data, A, B = random_module(rng, 48, 40, 40, 12, 4)
    parameters = Parameters(
        start_layer_index=0,
        end_layer_index=1,
        preserve_good_behavior_weight=0.5,
        steer_bad_behavior_weight=0.3,
        overcorrect_relative_weight=0.8,
        neighbor_count=neighbor_count,
    )
    settings = Settings(preserve_row_magnitudes=preserve_row_magnitudes)

    losses_desired, A_desired, B_desired = upstream_optimise(
        data, A, B, parameters, settings, np.float64
    )

    with jax.default_device(jax.devices("cpu")[0]), jax.enable_x64(True):
        W_base, good_input, good_output, bad_input, bad_output = (
            jnp.asarray(x, jnp.float64) for x in data
        )
        lora_A = jnp.asarray(A, jnp.float64)
        lora_B = jnp.asarray(B, jnp.float64)
        state = jax.tree.map(
            lambda leaf: (
                leaf.astype(jnp.float64) if leaf.dtype == jnp.float32 else leaf
            ),
            lbfgs.init(lora_A.size + lora_B.size, settings.history_size),
        )
        loss_weights = jnp.array(
            [
                parameters.preserve_good_behavior_weight,
                parameters.steer_bad_behavior_weight,
                parameters.overcorrect_relative_weight,
            ],
            jnp.float64,
        )
        losses = []
        for _ in range(settings.n_optimization_steps):
            lora_A, lora_B, state, loss = ara_optimise(
                lora_A,
                lora_B,
                W_base[None, None],
                np.int32(0),
                np.int32(0),
                good_input,
                good_output,
                bad_input,
                bad_output,
                loss_weights,
                np.int32(neighbor_count),
                state,
                preserve_row_magnitudes=preserve_row_magnitudes,
                max_iter=settings.max_iter,
                history_size=settings.history_size,
                learning_rate=settings.learning_rate,
                k_max_good=NEIGHBOR_COUNT_MAX,
                k_max_bad=12,
            )
            losses.append(float(loss))
        lora_A, lora_B = np.asarray(lora_A), np.asarray(lora_B)

    np.testing.assert_allclose(
        losses, losses_desired, rtol=0, atol=1e-7 * abs(losses_desired[0])
    )
    assert losses_desired[-1] < 0.5 * losses_desired[0]
    assert relative_error(lora_B @ lora_A, B_desired @ A_desired) < 5e-5


def test_modify_model_matches_torch_in_float32(
    fake_prompts: None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    model = FakeModel.random(
        fake_settings(),
        layer_count=3,
        shapes={"attn.o_proj": (48, 40), "mlp.down_proj": (48, 96)},
    )
    modifier, ctx = make_modifier(model, lora_rank=4, print_loss=True)
    settings = modifier.settings
    reset_adapters = {
        component: tuple(np.asarray(x) for x in model.get_lora(component))
        for component in model.get_abliterable_components()
    }
    parameters = Parameters(
        start_layer_index=1,
        end_layer_index=3,
        preserve_good_behavior_weight=0.5,
        steer_bad_behavior_weight=0.3,
        overcorrect_relative_weight=0.8,
        neighbor_count=5,
    )

    capsys.readouterr()
    modifier.modify_model(ctx, parameters)
    printed_losses = {
        (int(layer_index), component, int(step)): float(loss)
        for layer_index, component, step, loss in re.findall(
            r"\[(\d+)/(.+?)/0\] Step: (\d+), Loss: (\S+)",
            capsys.readouterr().out,
        )
    }
    assert len(printed_losses) == 2 * 2 * settings.n_optimization_steps

    for component, (A_reset, B_reset) in reset_adapters.items():
        assert model.set_lora_calls[component] == 1
        A, B = (np.asarray(x) for x in model.get_lora(component))

        # Layer 0 is outside the range and keeps the reset adapter.
        assert np.array_equal(A[0], A_reset[0])
        assert not B[0].any()

        for layer_index in [1, 2]:
            data = module_data(modifier, model, component, layer_index, 0)
            losses_desired, A_desired, B_desired = upstream_optimise(
                data,
                A_reset[layer_index, 0],
                B_reset[layer_index, 0],
                parameters,
                settings,
                np.float32,
            )

            # The first loss is that of the reset adapter. It agrees up to the
            # rounding noise of the distances between bad outputs at B = 0.
            assert printed_losses[layer_index, component, 1] == pytest.approx(
                losses_desired[0], rel=1e-5
            )

            # The trajectories diverge, but reach similar losses.
            loss_at = functools.partial(
                float64_loss,
                data=data,
                parameters=parameters,
                preserve_row_magnitudes=True,
            )
            initial_loss = loss_at(A_reset[layer_index, 0], B_reset[layer_index, 0])
            loss = loss_at(A[layer_index, 0], B[layer_index, 0])
            loss_desired = loss_at(A_desired, B_desired)
            assert loss_desired < initial_loss
            assert abs(loss - loss_desired) <= 0.1 * (initial_loss - loss_desired)


def test_neighbor_count_must_fit_prompts(fake_prompts: None) -> None:
    model = FakeModel.random(fake_settings(), layer_count=2)
    modifier, ctx = make_modifier(model, lora_rank=2, n_optimization_steps=1)

    def parameters(start: int, end: int, neighbor_count: int) -> Parameters:
        return Parameters(
            start_layer_index=start,
            end_layer_index=end,
            preserve_good_behavior_weight=0.5,
            steer_bad_behavior_weight=0.1,
            overcorrect_relative_weight=1.0,
            neighbor_count=neighbor_count,
        )

    # There are 30 bad prompts.
    with pytest.raises(ValueError, match="neighbor_count"):
        modifier.modify_model(ctx, parameters(0, 2, 31))
    with pytest.raises(ValueError, match="neighbor_count"):
        modifier.modify_model(ctx, parameters(0, 2, 0))

    # An empty layer range leaves the adapters alone.
    modifier.modify_model(ctx, parameters(1, 1, 31))
    assert set(model.set_lora_calls.values()) == {0}

    # Neighbour counts above NEIGHBOR_COUNT_MAX are not in the search space,
    # but are still computed correctly.
    modifier.modify_model(ctx, parameters(1, 2, 30))
    assert set(model.set_lora_calls.values()) == {1}


def test_slow_reset_reapplies_adapters(fake_prompts: None) -> None:
    model = FakeModel.random(fake_settings(), layer_count=2)
    modifier, ctx = make_modifier(model, lora_rank=7)
    model.fast_reset = False
    modifier.reset_model(ctx)
    assert model.lora_rank == 7


def test_one_compile_for_all_neighbor_counts(fake_prompts: None) -> None:
    # Shapes no other test uses, so that every compile is counted.
    model = FakeModel.random(
        fake_settings(),
        layer_count=3,
        shapes={"attn.o_proj": (24, 20), "mlp.down_proj": (24, 36)},
    )
    modifier, ctx = make_modifier(
        model,
        lora_rank=2,
        n_optimization_steps=1,
        max_iter=2,
    )

    cache_size = ara_optimise._cache_size()
    rng = np.random.default_rng(0)
    for neighbor_count in range(1, NEIGHBOR_COUNT_MAX + 1):
        modifier.reset_model(ctx)
        modifier.modify_model(
            ctx,
            Parameters(
                start_layer_index=int(rng.integers(0, 2)),
                end_layer_index=3,
                preserve_good_behavior_weight=float(rng.uniform(0, 1)),
                steer_bad_behavior_weight=float(rng.uniform(1e-4, 1)),
                overcorrect_relative_weight=float(rng.uniform(0, 1.3)),
                neighbor_count=neighbor_count,
            ),
        )

    # One program per component shape.
    assert ara_optimise._cache_size() - cache_size == 2


def ara_optimise_arguments(
    preserve_row_magnitudes: bool = True,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Arguments of ara_optimise for a module of a bfloat16 component."""

    layer_count, d_out, d_in, rank, good_count, bad_count = 4, 96, 160, 8, 40, 30
    zeros = functools.partial(jnp.zeros, dtype=jnp.float32)
    arguments = (
        zeros((rank, d_in)),
        zeros((d_out, rank)),
        jnp.zeros((layer_count, 1, d_out, d_in), jnp.bfloat16),
        np.int32(2),
        np.int32(0),
        zeros((good_count, d_in)),
        zeros((good_count, d_out)),
        zeros((bad_count, d_in)),
        zeros((bad_count, d_out)),
        zeros(3),
        np.int32(5),
        lbfgs.init(rank * (d_in + d_out), 10),
    )
    static = {
        "preserve_row_magnitudes": preserve_row_magnitudes,
        "max_iter": 20,
        "history_size": 10,
        "learning_rate": 1.0,
        "k_max_good": NEIGHBOR_COUNT_MAX,
        "k_max_bad": NEIGHBOR_COUNT_MAX,
    }
    return arguments, static


def test_no_large_constants() -> None:
    # Arrays closed over by a jitted function become constants of its program.
    arguments, static = ara_optimise_arguments()
    text = ara_optimise.lower(*arguments, **static).as_text()

    shapes = re.findall(r"stablehlo\.constant .*: tensor<([^>]*)>", text)
    assert shapes
    for shape in shapes:
        assert np.prod([int(size) for size in shape.split("x")[:-1]]) <= 16, shape


@pytest.mark.parametrize("preserve_row_magnitudes", [True, False])
def test_float32_dots_request_highest_precision(
    device: jax.Device,
    preserve_row_magnitudes: bool,
) -> None:
    # Precision (1) of docs/DESIGN.md. The default precision is forced back to
    # DEFAULT, so that a dot that does not request HIGHEST itself shows up.
    arguments, static = ara_optimise_arguments(preserve_row_magnitudes)
    objective = functools.partial(
        ara.objective,
        preserve_row_magnitudes=preserve_row_magnitudes,
        k_max_good=NEIGHBOR_COUNT_MAX,
        k_max_bad=NEIGHBOR_COUNT_MAX,
    )
    A, B, W_stack, *_ = arguments
    data = ObjectiveData(
        W_stack[0, 0].astype(jnp.float32),
        jnp.ones((B.shape[0], 1)),
        *arguments[5:11],
    )

    with jax.default_matmul_precision("default"):
        texts = [
            ara_optimise.lower(*arguments, **static).as_text(),
            jax.jit(jax.value_and_grad(objective))
            .lower(jnp.concatenate([A.ravel(), B.ravel()]), data)
            .as_text(),
        ]
    for text in texts:
        precisions = float32_dot_precisions(text)
        assert precisions
        assert set(precisions) == {("HIGHEST", "HIGHEST")}


def float64_loss_and_gradient(
    A: np.ndarray,
    B: np.ndarray,
    data: tuple[np.ndarray, ...],
    loss_weights: np.ndarray,
    neighbor_count: int,
    preserve_row_magnitudes: bool,
) -> tuple[float, np.ndarray, np.ndarray]:
    """ARA's loss and its gradients with respect to A and B, in float64, from
    explicit differences."""

    W_base, good_input, good_output, bad_input, bad_output = data
    preserve_weight, steer_weight, overcorrect_weight = loss_weights

    W = W_base + B @ A
    W_row_norms = np.linalg.norm(W_base, axis=1, keepdims=True)
    norms = np.linalg.norm(W, axis=1, keepdims=True)
    W_eff = W / norms * W_row_norms if preserve_row_magnitudes else W
    new_good_output = good_input @ W_eff.T
    new_bad_output = bad_input @ W_eff.T

    residuals = new_good_output - good_output
    loss = preserve_weight * np.mean(residuals**2)
    gradient = preserve_weight * 2 / residuals.size * residuals.T @ good_input

    for weight, outputs in [
        (steer_weight, good_output),
        (-steer_weight * overcorrect_weight, bad_output),
    ]:
        differences = new_bad_output[:, None, :] - outputs[None, :, :]
        distances = np.linalg.norm(differences, axis=-1)
        nearest = np.argsort(distances, axis=1)[:, :neighbor_count]
        rows = np.arange(len(new_bad_output))[:, None]
        loss += weight * np.mean(distances[rows, nearest])
        # The mean distance's gradient is the mean of the unit difference vectors.
        selected = np.zeros_like(distances)
        selected[rows, nearest] = weight / nearest.size
        unit_differences = differences / distances[..., None]
        gradient += np.einsum("ij,ijd->di", selected, unit_differences) @ bad_input

    if preserve_row_magnitudes:
        # Through the row normalisation: project out the row direction.
        gradient *= W_row_norms
        directions = W / norms
        gradient = (
            gradient - directions * np.sum(gradient * directions, axis=1, keepdims=True)
        ) / norms
    return float(loss), B.T @ gradient, gradient @ A.T


def objective_inputs(
    data: tuple[np.ndarray, ...],
    loss_weights: np.ndarray,
    neighbor_count: int,
) -> ObjectiveData:
    W_base, good_input, good_output, bad_input, bad_output = (
        jnp.asarray(x, jnp.float32) for x in data
    )
    return ObjectiveData(
        W_base=W_base,
        W_row_norms=jnp.linalg.norm(W_base, axis=1, keepdims=True),
        good_input=good_input,
        good_output=good_output,
        bad_input=bad_input,
        bad_output=bad_output,
        loss_weights=jnp.asarray(loss_weights, jnp.float32),
        neighbor_count=jnp.int32(neighbor_count),
    )


@pytest.mark.parametrize("preserve_row_magnitudes", [True, False])
def test_loss_and_gradient_match_float64(
    device: jax.Device,
    preserve_row_magnitudes: bool,
) -> None:
    # Precision (2) of docs/DESIGN.md: discriminates only on the TPU. B is far
    # enough from 0 that the k nearest neighbours are well separated.
    rng = np.random.default_rng(0)
    data, A, _ = random_module(rng, 64, 128, 48, 40, 8)
    B = (0.3 * rng.standard_normal((64, 8)) / np.sqrt(8)).astype(np.float32)
    loss_weights = np.array([0.5, 0.3, 0.6], np.float32)
    neighbor_count = 5

    loss, gradient = jax.jit(
        jax.value_and_grad(
            functools.partial(
                ara.objective,
                preserve_row_magnitudes=preserve_row_magnitudes,
                k_max_good=NEIGHBOR_COUNT_MAX,
                k_max_bad=NEIGHBOR_COUNT_MAX,
            )
        )
    )(
        jnp.concatenate([jnp.asarray(A).ravel(), jnp.asarray(B).ravel()]),
        objective_inputs(data, loss_weights, neighbor_count),
    )

    loss_desired, gradient_A, gradient_B = float64_loss_and_gradient(
        A.astype(np.float64),
        B.astype(np.float64),
        tuple(x.astype(np.float64) for x in data),
        loss_weights.astype(np.float64),
        neighbor_count,
        preserve_row_magnitudes,
    )
    gradient_desired = np.concatenate([gradient_A.ravel(), gradient_B.ravel()])

    assert float(loss) == pytest.approx(loss_desired, rel=1e-5)
    error = np.linalg.norm(np.asarray(gradient, np.float64) - gradient_desired)
    assert error / np.linalg.norm(gradient_desired) < 5e-5


@pytest.mark.parametrize("preserve_row_magnitudes", [True, False])
def test_small_adapter_changes_change_the_loss(
    device: jax.Device,
    preserve_row_magnitudes: bool,
) -> None:
    # Precision (2) of docs/DESIGN.md: a change of B whose effect on W is 1e-6 ‖W‖
    # changes the loss by grad·δ. Where B @ A is rounded to bfloat16 in the
    # products with W_eff, it is lost. The k-NN terms are disabled: their distances
    # come from the Gram matrix, as in torch.cdist, whose cancellation near
    # coincident points is larger in float32 than such a change.
    rng = np.random.default_rng(1)
    data, A, _ = random_module(rng, 64, 128, 48, 40, 8)
    W_base, good_input, _, bad_input, _ = data
    # Exact float32 outputs, so that the loss is that of B @ A alone.
    data = (W_base, good_input, good_input @ W_base.T, bad_input, bad_input @ W_base.T)
    loss_weights = np.array([0.5, 0.0, 0.0], np.float32)

    # B @ A is 1 % of W.
    B = rng.standard_normal((64, 8))
    B *= 1e-2 * np.linalg.norm(W_base) / np.linalg.norm(B @ A)
    B = B.astype(np.float32)

    _, _, gradient_B = float64_loss_and_gradient(
        A.astype(np.float64),
        B.astype(np.float64),
        tuple(x.astype(np.float64) for x in data),
        loss_weights.astype(np.float64),
        1,
        preserve_row_magnitudes,
    )
    delta = gradient_B / np.linalg.norm(gradient_B @ A) * 1e-6 * np.linalg.norm(W_base)

    loss = jax.jit(
        functools.partial(
            ara.objective,
            preserve_row_magnitudes=preserve_row_magnitudes,
            k_max_good=NEIGHBOR_COUNT_MAX,
            k_max_bad=NEIGHBOR_COUNT_MAX,
        )
    )
    inputs = objective_inputs(data, loss_weights, 1)

    def loss_at(B: np.ndarray) -> float:
        x = jnp.concatenate(
            [jnp.asarray(A).ravel(), jnp.asarray(B, jnp.float32).ravel()]
        )
        return float(loss(x, inputs))

    change = (loss_at(B + delta) - loss_at(B - delta)) / 2
    assert change == pytest.approx(np.sum(gradient_B * delta), rel=0.05)


def capture_module_io(
    repo_id: str,
    prompts: dict[str, list[Prompt]],
) -> tuple[dict[str, np.ndarray], dict[str, ModuleIO]]:
    """
    The weights of the abliterable modules of a checkpoint, and their I/O for each
    list of prompts, captured with transformers in bfloat16 on the CPU as upstream
    captures it: at the last prompt position, with the prompts formatted by the
    chat template and left-padded.
    """

    torch = pytest.importorskip("torch")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(repo_id, dtype=torch.bfloat16)
    layers = model.model.layers

    def modules(layer: Any) -> dict[str, Any]:
        return {
            "attn.o_proj": layer.self_attn.o_proj,
            "mlp.down_proj": layer.mlp.down_proj,
        }

    def to_numpy(tensors: list[Any]) -> np.ndarray:
        # [L, M, N, d] in bfloat16. The values are bfloat16, so the casts are exact.
        array = torch.stack(tensors).float().numpy()
        return array[:, None].astype(ml_dtypes.bfloat16)

    weights = {
        component: to_numpy(
            [modules(layer)[component].weight.detach() for layer in layers]
        )
        for component in modules(layers[0])
    }

    module_io = {}
    for name, prompt_list in prompts.items():
        captured = {component: ([], []) for component in weights}
        hooks = []
        for layer in layers:
            for component, module in modules(layer).items():

                def hook(
                    module: Any,
                    inputs: Any,
                    output: Any,
                    values: tuple[list[Any], list[Any]] = captured[component],
                ) -> None:
                    values[0].append(inputs[0][:, -1, :])
                    values[1].append(output[:, -1, :])

                hooks.append(module.register_forward_hook(hook))

        batches = 0
        with torch.no_grad():
            for batch in batchify(prompt_list, 50):
                chats = [
                    [
                        {"role": "system", "content": prompt.system},
                        {"role": "user", "content": prompt.user},
                    ]
                    for prompt in batch
                ]
                texts = tokenizer.apply_chat_template(
                    chats, add_generation_prompt=True, tokenize=False
                )
                inputs = tokenizer(
                    texts,
                    return_tensors="pt",
                    padding=True,
                    return_token_type_ids=False,
                )
                model(**inputs)
                batches += 1
        for hook in hooks:
            hook.remove()

        # The hooks ran layer by layer for each batch.
        module_io[name] = {
            component: tuple(
                to_numpy(
                    [
                        torch.cat(values[layer_index :: len(layers)])
                        for layer_index in range(len(layers))
                    ]
                )
                for values in captured_io
            )
            for component, captured_io in captured.items()
        }
        assert batches * len(layers) == len(captured["attn.o_proj"][0])
    return weights, module_io


@pytest.mark.tpu
@pytest.mark.slow
def test_small_steering_weight_makes_progress(monkeypatch: pytest.MonkeyPatch) -> None:
    # Precision (3) of docs/DESIGN.md: ARA on module I/O captured from a real
    # bfloat16 checkpoint with steer_bad_behavior_weight = 1e-3, where updates
    # below half a bfloat16 ulp of W must not vanish. With such a small weight the
    # first L-BFGS step either takes off or ends the optimisation of a module
    # (also in PyTorch), and which modules take off depends on rounding, so the
    # TPU run is compared with the CPU run over all modules of the model.
    try:
        tpu = jax.devices("tpu")[0]
    except RuntimeError:
        pytest.skip("no tpu device")

    settings = fake_settings()
    specifications = {
        "good": SingleDatasetSpecification(
            dataset=GOOD_DATASET, split="train[:200]", column="text"
        ),
        "bad": SingleDatasetSpecification(
            dataset=BAD_DATASET, split="train[:200]", column="text"
        ),
    }
    prompts = {
        name: load_prompts(settings, specification)
        for name, specification in specifications.items()
    }
    serve_prompts(monkeypatch, prompts["good"], prompts["bad"])
    weights, module_io = capture_module_io(
        "HuggingFaceTB/SmolLM2-135M-Instruct", prompts
    )

    def captured_module_io(prompt_list: list[Prompt]) -> ModuleIO:
        return module_io["good" if prompt_list == prompts["good"] else "bad"]

    layer_count = len(next(iter(weights.values())))
    parameters = Parameters(
        start_layer_index=0,
        end_layer_index=layer_count,
        preserve_good_behavior_weight=0.5,
        steer_bad_behavior_weight=1e-3,
        overcorrect_relative_weight=1.0,
        neighbor_count=5,
    )

    adapters = {}
    for device in [tpu, jax.devices("cpu")[0]]:
        with jax.default_device(device):
            model = FakeModel(settings, weights, module_io=captured_module_io)
            modifier, ctx = make_modifier(model)
            # The same on both devices (JAX's random numbers do not depend on
            # the backend).
            reset_adapters = {
                component: tuple(np.asarray(x) for x in model.get_lora(component))
                for component in weights
            }
            modifier.modify_model(ctx, parameters)
            adapters[device.platform] = {
                component: tuple(np.asarray(x) for x in model.get_lora(component))
                for component in weights
            }

    # The summed losses of all modules, and the number of modules for which
    # ‖B @ A‖ / ‖W‖ > 1e-3.
    losses = {"initial": 0.0, "tpu": 0.0, "cpu": 0.0}
    progress = {"tpu": 0, "cpu": 0}
    for component in weights:
        for layer_index in range(layer_count):
            data = module_data(modifier, model, component, layer_index, 0)
            A_reset, B_reset = (x[layer_index, 0] for x in reset_adapters[component])
            losses["initial"] += float64_loss(A_reset, B_reset, data, parameters, True)
            for platform in progress:
                A, B = (x[layer_index, 0] for x in adapters[platform][component])
                losses[platform] += float64_loss(A, B, data, parameters, True)
                if np.linalg.norm(B @ A) / np.linalg.norm(data[0]) > 1e-3:
                    progress[platform] += 1

    results = f"losses {losses}, modules with progress {progress}"
    decrease = losses["initial"] - losses["cpu"]
    # In float32 on the CPU, ARA takes off on most modules and reduces the summed
    # loss by more than its initial value.
    assert progress["cpu"] >= layer_count, results
    assert decrease > losses["initial"] > 0, results
    # So it does on the TPU, within the variation between float32 runs.
    assert abs(losses["tpu"] - losses["cpu"]) <= 0.25 * decrease, results
    assert abs(progress["tpu"] - progress["cpu"]) <= 0.25 * progress["cpu"], results
