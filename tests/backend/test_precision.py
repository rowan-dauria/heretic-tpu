# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

"""Matmul precision of `heretic_tpu.backend` and its float32 linear algebra.

Each test runs on the CPU and, marked `tpu`, on the TPU. Only the TPU run of the
numerical tests discriminates: on the CPU every float32 dot is IEEE float32
whatever its precision, while on the TPU a float32 matrix product at the default
precision is a single bfloat16 pass. The comparison of `lbfgs.step` with
`torch.optim.LBFGS` on the TPU is `test_matches_torch_in_float32` in
test_lbfgs.py. Every dot in lbfgs.py is a vector-vector product, which XLA:TPU
currently computes in float32 at any precision, so for lbfgs.py only the check of
the lowered program guards the precision.
"""

import functools
import re

import jax
import jax.numpy as jnp
import numpy as np

from heretic_tpu.backend import lbfgs, linalg

HIGHEST = jax.lax.Precision.HIGHEST


def float32_dot_precisions(text: str) -> list[tuple[str, str]]:
    """The precision of every dot_general with float32 operands in StableHLO text."""

    precisions = []
    for line in text.splitlines():
        if "stablehlo.dot_general" not in line:
            continue
        operands = re.search(r": \((tensor<[^>]*>), (tensor<[^>]*>)\)", line)
        assert operands, line
        if not all(re.search(r"[<x]f32>$", t) for t in operands.groups()):
            continue
        precision = re.search(r"precision = \[(\w+), (\w+)\]", line)
        precisions.append(precision.groups() if precision else ("DEFAULT", "DEFAULT"))
    return precisions


def least_squares(x: jax.Array, data: tuple[jax.Array, jax.Array]) -> jax.Array:
    A, b = data
    residual = jnp.matmul(A, x, precision=HIGHEST) - b
    return jnp.mean(residual * residual)


def ara_shaped(x: jax.Array, data: tuple[jax.Array, ...]) -> jax.Array:
    # The structure of ARA's objective: LoRA adapters of rank 2 on a 12×8 module.
    W_base, good_in, good_out, bad_in, bad_out, k = data
    A = x[:16].reshape(2, 8)
    B = x[16:].reshape(12, 2)
    W_eff = linalg.normalize(W_base + jnp.matmul(B, A, precision=HIGHEST), axis=1)
    new_good = jnp.matmul(good_in, W_eff.T, precision=HIGHEST)
    new_bad = jnp.matmul(bad_in, W_eff.T, precision=HIGHEST)
    return (
        jnp.mean((new_good - good_out) ** 2)
        + jnp.mean(linalg.knn_mean(new_bad, good_out, k, 15))
        - jnp.mean(linalg.knn_mean(new_bad, bad_out, k, 15))
    )


def entry_points() -> list[tuple[str, object, tuple]]:
    f32 = jnp.float32
    key = jax.random.key(0)
    a = jnp.ones((40, 16), f32)
    b = jnp.ones((30, 16), f32)
    lsq = (jnp.ones((30, 20), f32), jnp.ones(30, f32))
    ara = (
        jnp.ones((12, 8), f32),
        jnp.ones((32, 8), f32),
        jnp.ones((32, 12), f32),
        jnp.ones((32, 8), f32),
        jnp.ones((32, 12), f32),
        jnp.int32(5),
    )
    return [
        (
            "svd_lowrank tall",
            lambda A, key: linalg.svd_lowrank(A, 8, 6, key),
            (jnp.ones((64, 48), f32), key),
        ),
        (
            "svd_lowrank wide",
            lambda A, key: linalg.svd_lowrank(A, 8, 6, key),
            (jnp.ones((48, 64), f32), key),
        ),
        (
            "knn_mean",
            functools.partial(linalg.knn_mean, k_max=15),
            (a, b, jnp.int32(5)),
        ),
        (
            "knn_mean gradient",
            jax.grad(
                lambda a, b, k: jnp.sum(linalg.knn_mean(a, b, k, 15)), argnums=(0, 1)
            ),
            (a, b, jnp.int32(5)),
        ),
        (
            "normalize gradient",
            jax.grad(lambda x: jnp.sum(linalg.normalize(x) ** 3)),
            (a,),
        ),
        (
            "lbfgs.step",
            functools.partial(
                lbfgs.step, least_squares, lr=1.0, max_iter=20, history_size=10
            ),
            (jnp.zeros(20, f32), lbfgs.init(20, 10), lsq),
        ),
        (
            "lbfgs.step on an ARA-shaped objective",
            functools.partial(
                lbfgs.step, ara_shaped, lr=1.0, max_iter=20, history_size=10
            ),
            (jnp.zeros(40, f32), lbfgs.init(40, 10), ara),
        ),
    ]


def test_float32_dots_request_highest_precision(device: jax.Device) -> None:
    # Lowered with the default precision forced back to DEFAULT, so that any dot
    # that does not pass precision=HIGHEST itself shows up as [DEFAULT, DEFAULT].
    for name, function, args in entry_points():
        with jax.default_matmul_precision("default"):
            text = jax.jit(function).lower(*args).as_text()
        precisions = float32_dot_precisions(text)
        if name != "normalize gradient":
            assert precisions, f"{name}: no float32 dot found"
        for precision in precisions:
            assert precision == ("HIGHEST", "HIGHEST"), f"{name}: {precision}"


def test_float32_matmul_is_true_float32(device: jax.Device) -> None:
    # Relies only on the jax_default_matmul_precision set by heretic_tpu.backend.
    rng = np.random.default_rng(0)
    a = rng.standard_normal((512, 512)).astype(np.float32)
    b = rng.standard_normal((512, 512)).astype(np.float32)

    product = np.asarray(jnp.dot(jnp.asarray(a), jnp.asarray(b)), np.float64)
    reference = a.astype(np.float64) @ b.astype(np.float64)

    error = np.max(np.abs(product - reference)) / np.max(np.abs(reference))
    assert error < 1e-5


def test_svd_lowrank_matches_float64(device: jax.Device) -> None:
    rng = np.random.default_rng(0)
    U, _ = np.linalg.qr(rng.standard_normal((256, 4)))
    V, _ = np.linalg.qr(rng.standard_normal((192, 4)))
    A = (U * [1.0, 0.5, 0.25, 0.125]) @ V.T

    U, S, V = jax.jit(linalg.svd_lowrank, static_argnums=(1, 2))(
        jnp.asarray(A, jnp.float32), 12, 6, jax.random.key(0)
    )
    U, S, V = (np.asarray(x, np.float64) for x in (U, S, V))

    reconstruction_error = np.linalg.norm((U * S) @ V.T - A) / np.linalg.norm(A)
    assert reconstruction_error < 1e-5
    np.testing.assert_allclose(S[:4], np.linalg.svd(A, compute_uv=False)[:4], rtol=1e-5)


def test_knn_mean_matches_float64(device: jax.Device) -> None:
    rng = np.random.default_rng(0)
    a = rng.standard_normal((40, 64)).astype(np.float32)
    b = rng.standard_normal((30, 64)).astype(np.float32)
    cotangent = rng.standard_normal(40).astype(np.float32)
    k = 7

    mean, vjp = jax.vjp(
        lambda a, b: linalg.knn_mean(a, b, jnp.int32(k), 15),
        jnp.asarray(a),
        jnp.asarray(b),
    )
    gradient_a, gradient_b = vjp(jnp.asarray(cotangent))

    # Float64 reference from explicit differences: the mean distance to the k
    # nearest rows, whose gradient is the mean of the unit difference vectors.
    differences = a.astype(np.float64)[:, None, :] - b.astype(np.float64)[None, :, :]
    distances = np.linalg.norm(differences, axis=-1)
    nearest = np.argsort(distances, axis=1)[:, :k]
    rows = np.arange(40)[:, None]
    reference = distances[rows, nearest].mean(axis=1)
    weights = np.zeros_like(distances)
    weights[rows, nearest] = cotangent.astype(np.float64)[:, None] / k
    unit = differences / distances[..., None]
    reference_a = np.einsum("ij,ijd->id", weights, unit)
    reference_b = -np.einsum("ij,ijd->jd", weights, unit)

    np.testing.assert_allclose(mean, reference, rtol=1e-5)
    for gradient, reference in [(gradient_a, reference_a), (gradient_b, reference_b)]:
        error = np.linalg.norm(np.asarray(gradient, np.float64) - reference)
        assert error / np.linalg.norm(reference) < 1e-5
