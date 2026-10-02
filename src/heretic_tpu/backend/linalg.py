# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

"""Float32 linear algebra shared by the modifiers.

Every float32 dot passes `Precision.HIGHEST` explicitly, so that the results are
true float32 on TPU even when this module is used without importing the CLI
(see "Matmul precision" in docs/DESIGN.md).
"""

import jax
import jax.numpy as jnp
from jax import Array, lax


def _vector_norm(x: Array, axis: int) -> Array:
    # Euclidean norm with a zero (not NaN) gradient at the zero vector,
    # matching the subgradient PyTorch's vector_norm uses there.
    squared = jnp.sum(x * x, axis=axis, keepdims=True)
    positive = squared > 0
    return jnp.where(positive, jnp.sqrt(jnp.where(positive, squared, 1)), 0)


def normalize(x: Array, axis: int = -1) -> Array:
    """`x` divided by its Euclidean norm along `axis`, clamped below at 1e-12.

    The same as `F.normalize(x, p=2, dim=axis)`, in value and gradient.
    """

    return x / jnp.maximum(_vector_norm(x, axis), 1e-12)


def svd_lowrank(
    A: Array,
    q: int,
    niter: int,
    key: Array,
) -> tuple[Array, Array, Array]:
    """Randomised SVD of a matrix: a port of `torch.svd_lowrank` with `M = None`.

    Returns `(U, S, V)` with `A ≈ U diag(S) Vᵀ`, where `U` is `[m, q]`, `S` is `[q]`
    and `V` is `[n, q]`. The algorithm (Halko et al. 2009, algorithms 4.4 and 5.1)
    is PyTorch's, but the random draws differ.
    """

    m, n = A.shape[-2:]

    # PyTorch assumes that A is tall, and transposes it otherwise.
    transposed = m < n
    if transposed:
        A = jnp.swapaxes(A, -1, -2)

    def matmul(a: Array, b: Array) -> Array:
        return jnp.matmul(a, b, precision=lax.Precision.HIGHEST)

    def adjoint(a: Array) -> Array:
        return jnp.swapaxes(a, -1, -2)

    # get_approximate_basis: Q has q orthonormal columns with Q Qᵀ A ≈ A.
    R = jax.random.normal(key, (A.shape[-1], q), A.dtype)
    Q = jnp.linalg.qr(matmul(A, R)).Q

    def power_iteration(_: int, Q: Array) -> Array:
        Q = jnp.linalg.qr(matmul(adjoint(A), Q)).Q
        return jnp.linalg.qr(matmul(A, Q)).Q

    Q = lax.fori_loop(0, niter, power_iteration, Q)

    B = matmul(adjoint(Q), A)
    U, S, Vh = jnp.linalg.svd(B, full_matrices=False)
    V = adjoint(Vh)
    U = matmul(Q, U)

    if transposed:
        U, V = V, U

    return U, S, V


def knn_mean(a: Array, b: Array, k: Array | int, k_max: int) -> Array:
    """For each row of `a`, the mean Euclidean distance to its `k` nearest rows of `b`.

    Equals `torch.cdist(a, b).topk(k, largest=False)[0].mean(1)` in value and gradient.
    `k` may be traced; `k_max` is static and bounds it (`1 <= k <= k_max <= len(b)`),
    so that one compiled program serves every `k`.
    """

    # Distances from the Gram matrix, as torch.cdist computes them for more than
    # 25 rows, rather than from broadcast differences, which would materialise
    # an [len(a), len(b), d] array.
    squared_distances = (
        jnp.sum(a * a, axis=-1)[:, None]
        + jnp.sum(b * b, axis=-1)[None, :]
        - 2 * jnp.matmul(a, b.T, precision=lax.Precision.HIGHEST)
    )
    # Clamping keeps the gradient finite where rounding makes a distance vanish
    # (or go negative); it is zero there, like torch.cdist's subgradient.
    distances = jnp.sqrt(jnp.maximum(squared_distances, 1e-30))

    nearest_distances = -lax.top_k(-distances, k_max)[0]
    selected = jnp.arange(k_max) < k
    return jnp.sum(jnp.where(selected, nearest_distances, 0), axis=-1) / k
