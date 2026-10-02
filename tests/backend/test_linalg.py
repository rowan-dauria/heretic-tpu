# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

"""`heretic_tpu.backend.linalg` against the PyTorch functions upstream uses."""

import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from heretic_tpu.backend import linalg

torch = pytest.importorskip("torch")
F = torch.nn.functional


def relative_error(actual, desired) -> float:
    actual = np.asarray(actual, np.float64)
    desired = np.asarray(desired, np.float64)
    return float(np.linalg.norm(actual - desired) / np.linalg.norm(desired))


@pytest.mark.parametrize("axis", [1, 0, -1])
@pytest.mark.usefixtures("device")
def test_normalize_matches_torch(axis: int) -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((6, 5)).astype(np.float32)
    x[2] = 0
    cotangent = rng.standard_normal((6, 5)).astype(np.float32)

    x_torch = torch.tensor(x, requires_grad=True)
    y_torch = F.normalize(x_torch, p=2, dim=axis)
    (y_torch * torch.tensor(cotangent)).sum().backward()

    y, vjp = jax.vjp(lambda x: linalg.normalize(x, axis=axis), jnp.asarray(x))
    (gradient,) = vjp(jnp.asarray(cotangent))

    np.testing.assert_allclose(y, y_torch.detach().numpy(), rtol=1e-6, atol=1e-7)
    # Finite at the zero row, where the gradient is the cotangent / 1e-12.
    assert np.all(np.isfinite(gradient))
    np.testing.assert_allclose(gradient, x_torch.grad.numpy(), rtol=1e-5, atol=1e-6)


def low_rank(rng: np.random.Generator, m: int, n: int, rank: int) -> np.ndarray:
    # Well-separated singular values 1, 1/2, 1/4, …
    U, _ = np.linalg.qr(rng.standard_normal((m, rank)))
    V, _ = np.linalg.qr(rng.standard_normal((n, rank)))
    return (U * 0.5 ** np.arange(rank)) @ V.T


@pytest.mark.parametrize("shape", [(60, 40), (40, 60)])
@pytest.mark.usefixtures("device")
def test_svd_lowrank_matches_torch_shapes_and_exact_rank(
    shape: tuple[int, int],
) -> None:
    rng = np.random.default_rng(0)
    A = low_rank(rng, *shape, rank=3).astype(np.float32)

    U, S, V = jax.jit(linalg.svd_lowrank, static_argnums=(1, 2))(
        jnp.asarray(A), 8, 2, jax.random.key(0)
    )
    U_torch, S_torch, V_torch = torch.svd_lowrank(torch.tensor(A), q=8, niter=2)

    assert U.shape == U_torch.shape == (shape[0], 8)
    assert S.shape == S_torch.shape == (8,)
    assert V.shape == V_torch.shape == (shape[1], 8)
    assert U.dtype == S.dtype == V.dtype == jnp.float32

    # Orthonormal columns, and A recovered exactly because its rank is below q.
    np.testing.assert_allclose(U.T @ U, np.eye(8), atol=1e-5)
    np.testing.assert_allclose(V.T @ V, np.eye(8), atol=1e-5)
    assert relative_error((U * S) @ V.T, A) < 1e-5
    np.testing.assert_allclose(S[:3], S_torch[:3], rtol=1e-5)
    np.testing.assert_allclose(S[:3], [1, 0.5, 0.25], rtol=1e-5)
    assert np.all(S[3:] < 1e-5)


@pytest.mark.parametrize("rank", [1, 3])
@pytest.mark.parametrize("shape", [(96, 64), (64, 96)])
@pytest.mark.usefixtures("device")
def test_svd_lowrank_matches_torch_on_abliteration_deltas(
    shape: tuple[int, int],
    rank: int,
) -> None:
    # The delta of abliteration with row_normalization = "full": the rank-r
    # adapter of a row-renormalised rank-1 update, which is not exactly low-rank.
    rng = np.random.default_rng(rank)
    W = rng.standard_normal(shape).astype(np.float32)
    v = rng.standard_normal(shape[0]).astype(np.float32)
    v /= np.linalg.norm(v)
    W_row_norms = np.linalg.norm(W, axis=1, keepdims=True)
    W_normalized = W / W_row_norms
    A = (v @ W_normalized)[None, :]
    B = (-0.8 * v)[:, None]
    W_new = W_normalized + B @ A
    delta = W_new / np.linalg.norm(W_new, axis=1, keepdims=True) * W_row_norms - W

    def adapter_product(U, S, V):
        # Upstream's split of the truncated SVD into lora_B @ lora_A.
        sqrt_S = np.sqrt(np.asarray(S, np.float64)[:rank])
        lora_B = np.asarray(U, np.float64)[:, :rank] * sqrt_S
        lora_A = sqrt_S[:, None] * np.asarray(V, np.float64)[:, :rank].T
        return lora_B @ lora_A

    q = 2 * rank + 4
    ours = linalg.svd_lowrank(jnp.asarray(delta), q, 6, jax.random.key(0))
    torch.manual_seed(0)
    reference = torch.svd_lowrank(torch.tensor(delta), q=q, niter=6)

    # Same singular values and rank-r approximation as torch.svd_lowrank, which
    # is the optimal one (to rounding) for these spectra.
    np.testing.assert_allclose(ours[1][:rank], reference[1][:rank], rtol=1e-5)
    assert relative_error(adapter_product(*ours), adapter_product(*reference)) < 1e-4
    _, S_exact, _ = np.linalg.svd(delta.astype(np.float64))
    np.testing.assert_allclose(ours[1][:rank], S_exact[:rank], rtol=1e-5)


@pytest.mark.usefixtures("device")
def test_svd_lowrank_is_deterministic_in_its_key() -> None:
    A = jnp.asarray(low_rank(np.random.default_rng(0), 30, 20, 4), jnp.float32)
    first = linalg.svd_lowrank(A, 6, 2, jax.random.key(7))
    second = linalg.svd_lowrank(A, 6, 2, jax.random.key(7))
    for x, y in zip(first, second, strict=True):
        np.testing.assert_array_equal(x, y)


def knn_mean_torch(a: np.ndarray, b: np.ndarray, k: int):
    a_torch = torch.tensor(a, requires_grad=True)
    b_torch = torch.tensor(b, requires_grad=True)
    distances = torch.cdist(a_torch, b_torch)
    mean = distances.topk(k, dim=1, largest=False)[0].mean(1)
    return mean, a_torch, b_torch


@pytest.mark.parametrize("rows", [(40, 30), (12, 20)])
@pytest.mark.usefixtures("device")
def test_knn_mean_matches_torch_for_every_k(rows: tuple[int, int]) -> None:
    # More than 25 rows uses torch.cdist's matrix-multiply path, like the port;
    # fewer uses its direct path, which agrees to rounding.
    rng = np.random.default_rng(0)
    a = rng.standard_normal((rows[0], 16)).astype(np.float32)
    b = rng.standard_normal((rows[1], 16)).astype(np.float32)
    cotangent = rng.standard_normal(rows[0]).astype(np.float32)
    k_max = min(15, rows[1])

    @jax.jit
    def value_and_gradients(a, b, k):
        mean, vjp = jax.vjp(lambda a, b: linalg.knn_mean(a, b, k, k_max), a, b)
        return mean, *vjp(jnp.asarray(cotangent))

    for k in range(1, k_max + 1):
        mean, gradient_a, gradient_b = value_and_gradients(
            jnp.asarray(a), jnp.asarray(b), jnp.int32(k)
        )
        mean_torch, a_torch, b_torch = knn_mean_torch(a, b, k)
        (mean_torch * torch.tensor(cotangent)).sum().backward()

        np.testing.assert_allclose(mean, mean_torch.detach().numpy(), rtol=1e-5)
        assert relative_error(gradient_a, a_torch.grad) < 1e-5
        assert relative_error(gradient_b, b_torch.grad) < 1e-5

    # One compiled program serves every k.
    assert value_and_gradients._cache_size() == 1


@pytest.mark.usefixtures("device")
def test_knn_mean_has_finite_gradient_at_zero_distance() -> None:
    # Rows of a that coincide with rows of b, as ARA's bad outputs coincide
    # with themselves before any update.
    rng = np.random.default_rng(1)
    b = rng.standard_normal((30, 8)).astype(np.float32)
    a = np.concatenate([b[:10], rng.standard_normal((20, 8)).astype(np.float32)])

    mean, gradients = jax.value_and_grad(
        lambda a, b: jnp.sum(linalg.knn_mean(a, b, 3, 15)), argnums=(0, 1)
    )(jnp.asarray(a), jnp.asarray(b))
    mean_torch, a_torch, _ = knn_mean_torch(a, b, 3)
    mean_torch.sum().backward()

    assert np.all(np.isfinite(gradients[0])) and np.all(np.isfinite(gradients[1]))
    np.testing.assert_allclose(mean, mean_torch.sum().item(), rtol=1e-4)
    # Away from the coinciding rows, the gradients agree to rounding.
    np.testing.assert_allclose(
        gradients[0][10:], a_torch.grad[10:], rtol=1e-4, atol=1e-5
    )


def test_knn_mean_never_forms_pairwise_differences() -> None:
    a = jnp.zeros((40, 16), jnp.float32)
    b = jnp.zeros((30, 16), jnp.float32)
    jaxpr = jax.make_jaxpr(
        jax.grad(lambda a, b: jnp.sum(linalg.knn_mean(a, b, 5, 15)), argnums=(0, 1))
    )(a, b)

    # No array in the value and gradient computation is larger than the
    # [40, 30] distance matrix, so no [40, 30, 16] difference array is formed.
    shapes = re.findall(r"\[([\d,]+)\]", str(jaxpr))
    sizes = [np.prod([int(n) for n in shape.split(",")]) for shape in shapes]
    assert max(sizes) == 40 * 30
