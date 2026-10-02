# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

"""
The abliteration modifier against upstream's implementation.

Upstream computes the residual directions with PyTorch and the adapters module by
module in a Python loop. `upstream_adapters` ports that loop to PyTorch on the
weights of a FakeModel, and the port's adapters are compared with it. With "full"
row normalisation the randomised SVDs draw different random matrices, so only the
products B @ A (the merged weight deltas) are compared. They agree to about 1e-4
of the delta because its spectrum decays fast: a rank-one term dominates, and the
normalisation adds a perturbation whose singular values fall off with the squared
direction entries.
"""

import ast
import functools
import math
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from heretic_tpu.config import Settings as HereticSettings
from heretic_tpu.modifiers.abliteration import (
    Abliteration,
    Parameters,
    RowNormalization,
    Settings,
    WeightDistribution,
    get_adapters,
)
from heretic_tpu.plugin import Context
from heretic_tpu.utils import Prompt
from tests.backend.test_precision import float32_dot_precisions
from tests.fake_model import FakeModel, fake_settings

UPSTREAM_MODULE = (
    Path(__file__).parents[1] / "heretic/src/heretic/modifiers/abliteration.py"
)
PORT_MODULE = Path(__file__).parents[1] / "src/heretic_tpu/modifiers/abliteration.py"

GOOD_PROMPTS = [Prompt("You are a helpful assistant.", f"Good {i}") for i in range(24)]
BAD_PROMPTS = [Prompt("You are a helpful assistant.", f"Bad {i}") for i in range(20)]

SHAPES = {"attn.o_proj": (64, 48), "mlp.down_proj": (64, 160)}
LAYER_COUNT = 6


def weight_distributions(down_proj_max_weight: float) -> dict[str, WeightDistribution]:
    return {
        # Layers 0 and 1 are too far from the maximum.
        "attn.o_proj": WeightDistribution(
            max_weight=1.2,
            max_weight_position=3.5,
            min_weight=0.4,
            min_weight_distance=2.0,
        ),
        # Layer 0 is too far from the maximum, and the weight of layers 1 and 5 is
        # exactly 0. A maximum weight of 0 disables the component.
        "mlp.down_proj": WeightDistribution(
            max_weight=down_proj_max_weight,
            max_weight_position=3.0,
            min_weight=0.0,
            min_weight_distance=2.0,
        ),
    }


PARAMETERS = {
    "global": Parameters(
        direction_index=2.3,
        weight_distributions=weight_distributions(0.9),
    ),
    # A fractional part above 0.5 takes the other branch of torch.lerp.
    "global-upper": Parameters(
        direction_index=1.7,
        weight_distributions=weight_distributions(0.9),
    ),
    "per-layer": Parameters(
        direction_index=None,
        weight_distributions=weight_distributions(0.9),
    ),
    "mlp-disabled": Parameters(
        direction_index=None,
        weight_distributions=weight_distributions(0.0),
    ),
}


@pytest.fixture(autouse=True)
def prompts(monkeypatch: pytest.MonkeyPatch) -> None:
    """Serves the default good and bad prompt datasets from memory."""

    def load_prompts(settings: HereticSettings, specification: Any) -> list[Prompt]:
        if specification.dataset == "mlabonne/harmless_alpaca":
            return GOOD_PROMPTS
        assert specification.dataset == "mlabonne/harmful_behaviors"
        return BAD_PROMPTS

    monkeypatch.setattr("heretic_tpu.plugin.load_prompts", load_prompts)


def make_modifier(
    model: FakeModel,
    **settings: Any,
) -> tuple[Abliteration, Context]:
    """Initialises the modifier on a model and resets the model, as main.py does."""

    modifier = Abliteration(
        heretic_settings=model.settings,
        settings=Settings(**settings),
    )
    ctx = Context(settings=model.settings, model=model)  # ty:ignore[invalid-argument-type]
    modifier.init(ctx)
    modifier.reset_model(ctx)
    return modifier, ctx


def upstream_directions(
    good_means: np.ndarray,
    bad_means: np.ndarray,
    orthogonalize_direction: bool,
) -> np.ndarray:
    """The residual directions as upstream's init computes them, in PyTorch."""

    torch = pytest.importorskip("torch")
    F = torch.nn.functional

    good_means = torch.tensor(good_means)
    bad_means = torch.tensor(bad_means)
    residual_directions = F.normalize(bad_means - good_means, p=2, dim=1)
    if orthogonalize_direction:
        good_directions = F.normalize(good_means, p=2, dim=1)
        projection_vector = torch.sum(residual_directions * good_directions, dim=1)
        residual_directions = (
            residual_directions - projection_vector.unsqueeze(1) * good_directions
        )
        residual_directions = F.normalize(residual_directions, p=2, dim=1)
    return residual_directions.numpy()


def upstream_adapters(
    model: FakeModel,
    residual_directions: np.ndarray,
    parameters: Parameters,
    row_normalization: RowNormalization,
    A_reset: dict[str, np.ndarray],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Upstream's modify_model loop in PyTorch, on the model's weights, starting
    from the given reset A (and B = 0)."""

    torch = pytest.importorskip("torch")
    F = torch.nn.functional

    residual_directions = torch.tensor(residual_directions)
    if parameters.direction_index is None:
        residual_direction = None
    else:
        weight, index = math.modf(parameters.direction_index + 1)
        residual_direction = F.normalize(
            residual_directions[int(index)].lerp(
                residual_directions[int(index) + 1],
                weight,
            ),
            p=2,
            dim=0,
        )

    adapters = {}
    for component in model.get_abliterable_components():
        W_stack = torch.tensor(
            np.asarray(model.get_base_weights(component), np.float32)
        )
        layer_count, module_count, d_out, _ = W_stack.shape
        r = A_reset[component].shape[2]
        A = A_reset[component].copy()
        B = np.zeros((layer_count, module_count, d_out, r), np.float32)
        weight_distribution = parameters.weight_distributions[component]

        for layer_index in range(layer_count):
            distance = abs(layer_index - weight_distribution.max_weight_position)
            if distance > weight_distribution.min_weight_distance:
                continue
            weight = weight_distribution.max_weight + (
                distance / weight_distribution.min_weight_distance
            ) * (weight_distribution.min_weight - weight_distribution.max_weight)
            if weight == 0:
                continue
            if residual_direction is None:
                v = residual_directions[layer_index + 1]
            else:
                v = residual_direction

            for module_index in range(module_count):
                W = W_stack[layer_index, module_index]
                if row_normalization == RowNormalization.FULL:
                    W_org = W
                if row_normalization != RowNormalization.NONE:
                    W_row_norms = torch.linalg.vector_norm(W, dim=1, keepdim=True)
                    W = F.normalize(W, p=2, dim=1)
                lora_A = (v @ W).view(1, -1)
                lora_B = (-weight * v).view(-1, 1)
                if row_normalization == RowNormalization.PRE:
                    lora_B = W_row_norms * lora_B
                elif row_normalization == RowNormalization.FULL:
                    W = W + lora_B @ lora_A
                    W = F.normalize(W, p=2, dim=1)
                    W = W * W_row_norms
                    W = W - W_org
                    torch.manual_seed(model.settings.seed)
                    U, S, Vh = torch.svd_lowrank(W, q=2 * r + 4, niter=6)
                    U = U[:, :r]
                    S = S[:r]
                    Vh = Vh[:, :r].T
                    sqrt_S = torch.sqrt(S)
                    lora_B = U @ torch.diag(sqrt_S)
                    lora_A = torch.diag(sqrt_S) @ Vh
                A[layer_index, module_index] = lora_A.numpy()
                B[layer_index, module_index] = lora_B.numpy()

        adapters[component] = (A, B)
    return adapters


def relative_error(actual: np.ndarray, desired: np.ndarray) -> float:
    """The largest absolute error relative to the largest desired magnitude."""

    actual = np.asarray(actual, np.float64)
    desired = np.asarray(desired, np.float64)
    return float(np.max(np.abs(actual - desired)) / np.max(np.abs(desired)))


def class_members(path: Path, names: dict[str, list[str] | None]) -> dict[str, str]:
    """
    The normalised source (without comments or formatting) of top-level classes
    and functions in a module. `names` maps a class name to the methods to
    extract, or to None for the whole class or function.
    """

    tree = ast.parse(path.read_text())
    sources = {}
    for node in tree.body:
        if not isinstance(node, (ast.ClassDef, ast.FunctionDef)):
            continue
        if node.name not in names:
            continue
        methods = names[node.name]
        if methods is None:
            sources[node.name] = ast.unparse(node)
            continue
        for item in node.body:
            if isinstance(item, ast.FunctionDef) and item.name in methods:
                sources[f"{node.name}.{item.name}"] = ast.unparse(item)
    return sources


def test_unchanged_from_upstream() -> None:
    # Settings, parameters, their presentation, the search space and the modifier
    # names are upstream's.
    if not UPSTREAM_MODULE.is_file():
        pytest.skip("the upstream source is not checked out")

    names = {
        "WeightDistribution": None,
        "Parameters": None,
        "RowNormalization": None,
        "Settings": None,
        "Abliteration": [
            "reproducible",
            "modifier_name",
            "suggest_parameters",
            "reset_model",
        ],
    }
    upstream = class_members(UPSTREAM_MODULE, names)
    port = class_members(PORT_MODULE, names)
    assert len(upstream) == 8
    assert port == upstream


@pytest.mark.parametrize("orthogonalize_direction", [True, False])
@pytest.mark.parametrize("winsorization_quantile", [1.0, 0.9])
def test_directions_match_upstream(
    orthogonalize_direction: bool,
    winsorization_quantile: float,
) -> None:
    model = FakeModel.random(fake_settings(), layer_count=LAYER_COUNT, shapes=SHAPES)
    modifier, _ = make_modifier(
        model,
        orthogonalize_direction=orthogonalize_direction,
        winsorization_quantile=winsorization_quantile,
    )

    reference = upstream_directions(
        model.get_residuals_mean(GOOD_PROMPTS, winsorization_quantile),
        model.get_residuals_mean(BAD_PROMPTS, winsorization_quantile),
        orthogonalize_direction,
    )

    assert modifier.residual_directions.dtype == np.float32
    assert modifier.residual_directions.shape == (LAYER_COUNT + 1, 64)
    np.testing.assert_allclose(modifier.residual_directions, reference, atol=1e-6)
    assert model.lora_rank == 3


@pytest.mark.parametrize("row_normalization", list(RowNormalization))
@pytest.mark.parametrize("parameters", list(PARAMETERS))
@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float32])
def test_adapters_match_upstream(
    row_normalization: RowNormalization,
    parameters: str,
    dtype: Any,
) -> None:
    model = FakeModel.random(
        fake_settings(seed=7),
        layer_count=LAYER_COUNT,
        shapes=SHAPES,
        dtype=dtype,
    )
    modifier, ctx = make_modifier(model, row_normalization=row_normalization)
    A_reset = {
        component: np.asarray(model.get_lora(component)[0])
        for component in model.get_abliterable_components()
    }

    modifier.modify_model(ctx, PARAMETERS[parameters])
    reference = upstream_adapters(
        model,
        modifier.residual_directions,
        PARAMETERS[parameters],
        row_normalization,
        A_reset,
    )

    for component, (A_desired, B_desired) in reference.items():
        assert model.set_lora_calls[component] == 1
        A, B = (np.asarray(x) for x in model.get_lora(component))
        assert A.dtype == B.dtype == np.float32
        assert A.shape == A_desired.shape and B.shape == B_desired.shape

        for layer_index in range(LAYER_COUNT):
            if not B_desired[layer_index].any():
                # Skipped modules keep the reset adapters exactly.
                assert np.array_equal(A[layer_index], A_reset[component][layer_index])
                assert not B[layer_index].any()
            elif row_normalization == RowNormalization.FULL:
                # Each randomised SVD is up to about 1e-4 away from the exact
                # rank-r truncation here (both PyTorch's and the port's).
                delta = B[layer_index, 0] @ A[layer_index, 0]
                delta_desired = B_desired[layer_index, 0] @ A_desired[layer_index, 0]
                assert relative_error(delta, delta_desired) < 3e-4, (
                    component,
                    layer_index,
                )
            else:
                assert relative_error(A[layer_index], A_desired[layer_index]) < 1e-5
                assert relative_error(B[layer_index], B_desired[layer_index]) < 1e-5

    skipped = {
        component: [
            layer_index
            for layer_index in range(LAYER_COUNT)
            if not np.asarray(model.get_lora(component)[1][layer_index]).any()
        ]
        for component in model.get_abliterable_components()
    }
    assert skipped["attn.o_proj"] == [0, 1]
    if parameters == "mlp-disabled":
        assert skipped["mlp.down_proj"] == list(range(LAYER_COUNT))
    else:
        assert skipped["mlp.down_proj"] == [0, 1, 5]


def test_slow_reset_reapplies_adapters() -> None:
    model = FakeModel.random(fake_settings(), layer_count=LAYER_COUNT, shapes=SHAPES)
    modifier, ctx = make_modifier(model, row_normalization=RowNormalization.PRE)
    model.fast_reset = False
    modifier.reset_model(ctx)
    assert model.lora_rank == 1


def float64_svd_lowrank(
    A: np.ndarray,
    q: int,
    niter: int,
    key: jax.Array,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """linalg.svd_lowrank in float64, from the same random matrix (JAX's random
    numbers do not depend on the backend)."""

    transposed = A.shape[0] < A.shape[1]
    if transposed:
        A = A.T
    R = np.asarray(jax.random.normal(key, (A.shape[1], q), jnp.float32), np.float64)
    Q = np.linalg.qr(A @ R)[0]
    for _ in range(niter):
        Q = np.linalg.qr(A.T @ Q)[0]
        Q = np.linalg.qr(A @ Q)[0]
    U, S, Vh = np.linalg.svd(Q.T @ A, full_matrices=False)
    U, V = Q @ U, Vh.T
    if transposed:
        U, V = V, U
    return U, S, V


def float64_adapters(
    W: np.ndarray,
    v: np.ndarray,
    weight: float,
    row_normalization: RowNormalization,
    r: int,
    key: jax.Array,
) -> tuple[np.ndarray, np.ndarray]:
    """The adapters of a module, computed in float64."""

    W_org = W
    if row_normalization != RowNormalization.NONE:
        W_row_norms = np.linalg.norm(W, axis=1, keepdims=True)
        W = W / W_row_norms
    lora_A = (v @ W)[None, :]
    lora_B = (-weight * v)[:, None]
    if row_normalization == RowNormalization.PRE:
        lora_B = W_row_norms * lora_B
    elif row_normalization == RowNormalization.FULL:
        W = W + lora_B @ lora_A
        W = W / np.linalg.norm(W, axis=1, keepdims=True) * W_row_norms
        U, S, V = float64_svd_lowrank(W - W_org, 2 * r + 4, 6, key)
        lora_B = U[:, :r] * np.sqrt(S[:r])
        lora_A = np.sqrt(S[:r])[:, None] * V[:, :r].T
    return lora_A, lora_B


@pytest.mark.parametrize("row_normalization", list(RowNormalization))
def test_adapters_match_float64(
    device: jax.Device,
    row_normalization: RowNormalization,
) -> None:
    # Precision (2) of docs/DESIGN.md: discriminates only on the TPU. The modules
    # are wide, so svd_lowrank operates on the transpose.
    rng = np.random.default_rng(0)
    layer_count, module_count, d_out, d_in = 3, 2, 192, 256
    r = 1 if row_normalization != RowNormalization.FULL else 3
    W = rng.standard_normal((layer_count, module_count, d_out, d_in)) / np.sqrt(d_in)
    W = W.astype(jnp.bfloat16)
    directions = rng.standard_normal((layer_count, d_out))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    weights = np.array([0.8, 1.3, 0.5])
    apply = np.array([True, False, True])
    A_reset = rng.uniform(-1, 1, (layer_count, module_count, r, d_in))
    key = jax.random.key(0)

    A, B = get_adapters(
        jnp.asarray(W),
        jnp.asarray(directions, jnp.float32),
        jnp.asarray(weights, jnp.float32),
        jnp.asarray(apply),
        jnp.asarray(A_reset, jnp.float32),
        key,
        row_normalization=row_normalization,
    )
    A, B = np.asarray(A, np.float64), np.asarray(B, np.float64)

    for layer_index in range(layer_count):
        for module_index in range(module_count):
            module = (layer_index, module_index)
            if not apply[layer_index]:
                assert np.array_equal(A[module], A_reset[module].astype(np.float32))
                assert not B[module].any()
                continue
            A_desired, B_desired = float64_adapters(
                W[module].astype(np.float64),
                directions[layer_index].astype(np.float32).astype(np.float64),
                float(np.float32(weights[layer_index])),
                row_normalization,
                r,
                key,
            )
            if row_normalization == RowNormalization.FULL:
                # QR and SVD implementations choose the signs of the singular
                # vectors differently, so only the product is determined. (At the
                # default precision, the TPU's error is around 1e-3.)
                assert (
                    relative_error(B[module] @ A[module], B_desired @ A_desired) < 3e-5
                )
            else:
                assert relative_error(A[module], A_desired) < 1e-5
                assert relative_error(B[module], B_desired) < 1e-5


def adapter_arguments(
    layer_count: int,
    d_out: int,
    d_in: int,
    r: int,
) -> tuple[jax.ShapeDtypeStruct, ...]:
    """Abstract arguments of get_adapters for bfloat16 weights, with M = 1."""

    return (
        jax.ShapeDtypeStruct((layer_count, 1, d_out, d_in), jnp.bfloat16),
        jax.ShapeDtypeStruct((layer_count, d_out), jnp.float32),
        jax.ShapeDtypeStruct((layer_count,), jnp.float32),
        jax.ShapeDtypeStruct((layer_count,), jnp.bool_),
        jax.ShapeDtypeStruct((layer_count, 1, r, d_in), jnp.float32),
        jax.ShapeDtypeStruct((), jax.random.key(0).dtype),
    )


@pytest.mark.parametrize("row_normalization", list(RowNormalization))
def test_one_module_in_flight(row_normalization: RowNormalization) -> None:
    # Memory: the size of Qwen2.5-7B's mlp.down_proj. The temporaries hold at most
    # four float32 copies of one module, whatever the number of layers.
    d_out, d_in = 3584, 18944
    r = 1 if row_normalization != RowNormalization.FULL else 3
    module_bytes = d_out * d_in * 4

    temp_sizes = {}
    for layer_count in [4, 28]:
        compiled = get_adapters.lower(
            *adapter_arguments(layer_count, d_out, d_in, r),
            row_normalization=row_normalization,
        ).compile()
        temp_sizes[layer_count] = compiled.memory_analysis().temp_size_in_bytes

    assert temp_sizes[28] <= 4 * module_bytes
    # Only the stacked adapters grow with the number of layers (on the TPU, by a
    # few MB per layer, because their narrow dimensions are padded to tiles). A copy
    # of each layer's weights would add at least half a module per layer.
    assert temp_sizes[28] - temp_sizes[4] <= 0.5 * module_bytes


def equations(jaxpr: Any, path: tuple[str, ...] = ()) -> Iterator[tuple[Any, tuple]]:
    """Every equation of a jaxpr and its sub-jaxprs, with the primitives that
    enclose it."""

    for equation in jaxpr.eqns:
        yield equation, path
        for value in equation.params.values():
            for item in value if isinstance(value, (tuple, list)) else [value]:
                inner = getattr(item, "jaxpr", item)
                if hasattr(inner, "eqns"):
                    yield from equations(inner, path + (equation.primitive.name,))


@pytest.mark.parametrize("row_normalization", list(RowNormalization))
def test_skipped_modules_do_no_work(row_normalization: RowNormalization) -> None:
    d_out, d_in = 96, 160
    r = 1 if row_normalization != RowNormalization.FULL else 3
    jaxpr = jax.make_jaxpr(
        functools.partial(get_adapters, row_normalization=row_normalization)
    )(*adapter_arguments(5, d_out, d_in, r))

    conds = [
        path
        for equation, path in equations(jaxpr.jaxpr)
        if equation.primitive.name == "cond"
    ]
    # A single conditional, executed per module by the scan of lax.map.
    assert len(conds) == 1
    assert conds[0][-1] == "scan"

    for equation, _ in equations(jaxpr.jaxpr):
        if equation.primitive.name == "select_n":
            for operand in equation.invars:
                assert operand.aval.shape[-2:] != (d_out, d_in)


@pytest.mark.parametrize("row_normalization", list(RowNormalization))
def test_float32_dots_request_highest_precision(
    device: jax.Device,
    row_normalization: RowNormalization,
) -> None:
    # Precision (1) of docs/DESIGN.md. The default precision is forced back to
    # DEFAULT, so that a dot that does not request HIGHEST itself shows up.
    r = 1 if row_normalization != RowNormalization.FULL else 3
    with jax.default_matmul_precision("default"):
        text = get_adapters.lower(
            *adapter_arguments(3, 96, 160, r),
            row_normalization=row_normalization,
        ).as_text()
    precisions = float32_dot_precisions(text)
    assert precisions
    assert set(precisions) == {("HIGHEST", "HIGHEST")}


def test_one_compile_per_component_shape() -> None:
    # Shapes no other test uses, so that every compile is counted. Per-layer and
    # global directions, and any choice of ablated layers, share one program.
    model = FakeModel.random(
        fake_settings(),
        layer_count=LAYER_COUNT,
        shapes={"attn.o_proj": (40, 24), "mlp.down_proj": (40, 56)},
    )
    modifier, ctx = make_modifier(model, row_normalization=RowNormalization.FULL)

    cache_size = get_adapters._cache_size()
    for parameters in PARAMETERS.values():
        modifier.reset_model(ctx)
        modifier.modify_model(ctx, parameters)
    assert get_adapters._cache_size() - cache_size == 2
