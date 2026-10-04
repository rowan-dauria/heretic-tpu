# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

"""`heretic_tpu.backend.lbfgs` against `torch.optim.LBFGS` with strong-Wolfe line search.

Both optimisers run the same problems over several outer steps, with histories
small enough to wrap around. After every step, the loss returned, the iterate,
the evaluation and iteration counts and the whole carried state are compared.

In float64, the port and PyTorch agree to rounding on every problem and take
exactly the same branches. In float32 they agree tightly on Rosenbrock and least
squares. They cannot agree on the iterates of the ARA objective in float32,
because its L-BFGS trajectory is chaotic at that precision: the LoRA
parametrisation has flat directions (A → MA, B → BM⁻¹), the k-NN terms switch
neighbours discontinuously, and at B = 0 every "bad" output is its own nearest
neighbour at a distance that is pure rounding noise (in torch.cdist as in the
port), so the first gradient differs at the 1e-3 level between any two float32
implementations. Single-ulp differences (PyTorch evaluates some line-search
scalars as Python float64, and summation orders differ) are therefore amplified
to O(1) changes of A and B within one step. That test compares losses instead.
"""

import functools
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest

from heretic_tpu.backend import lbfgs, linalg

torch = pytest.importorskip("torch")
F = torch.nn.functional

HIGHEST = jax.lax.Precision.HIGHEST


@pytest.fixture
def cpu():
    """Runs on the CPU, also where the default backend is a TPU (no float64)."""

    with jax.default_device(jax.devices("cpu")[0]):
        yield


# Objectives, each in JAX (for the port) and PyTorch (for the reference).


def rosenbrock_jax(x: jax.Array, data: None) -> jax.Array:
    return jnp.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)


def rosenbrock_torch(x, data: None):
    return torch.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)


def least_squares_jax(x: jax.Array, data: tuple[jax.Array, jax.Array]) -> jax.Array:
    A, b = data
    residual = jnp.matmul(A, x, precision=HIGHEST) - b
    return jnp.mean(residual * residual)


def least_squares_torch(x, data):
    A, b = data
    residual = A @ x - b
    return torch.mean(residual * residual)


# The ARA objective as ara_optimise evaluates it, with the shapes of a LoRA
# adapter of rank R on a module with D_IN inputs and D_OUT outputs.
R, D_IN, D_OUT, N_ROWS, K, K_MAX = 4, 48, 40, 32, 5, 15


def ara_jax(x: jax.Array, data: tuple[jax.Array, ...]) -> jax.Array:
    W_base, W_row_norms, good_in, good_out, bad_in, bad_out, loss_weights, k = data
    A = x[: R * D_IN].reshape(R, D_IN)
    B = x[R * D_IN :].reshape(D_OUT, R)
    W_eff = W_base + jnp.matmul(B, A, precision=HIGHEST)
    W_eff = linalg.normalize(W_eff, axis=1) * W_row_norms
    new_good = jnp.matmul(good_in, W_eff.T, precision=HIGHEST)
    new_bad = jnp.matmul(bad_in, W_eff.T, precision=HIGHEST)
    preserve, steer, overcorrect = loss_weights
    return preserve * jnp.mean((new_good - good_out) ** 2) + steer * (
        jnp.mean(linalg.knn_mean(new_bad, good_out, k, K_MAX))
        + overcorrect * -jnp.mean(linalg.knn_mean(new_bad, bad_out, k, K_MAX))
    )


def ara_torch(x, data):
    # Upstream's objective and ara_loss, apart from the flat parameters and the
    # loss weights, which are tensors here.
    W_base, W_row_norms, good_in, good_out, bad_in, bad_out, loss_weights, k = data
    A = x[: R * D_IN].view(R, D_IN)
    B = x[R * D_IN :].view(D_OUT, R)

    def mean_distances_to_knn(a, b, k):
        distances = torch.cdist(a, b)
        nearest_distances, _ = distances.topk(k, dim=1, largest=False)
        return nearest_distances.mean(1)

    W_eff = W_base + (B @ A)
    W_eff = F.normalize(W_eff, p=2, dim=1) * W_row_norms
    new_good = good_in @ W_eff.T
    new_bad = bad_in @ W_eff.T
    preserve, steer, overcorrect = loss_weights
    return preserve * ((new_good - good_out) ** 2).mean() + steer * (
        mean_distances_to_knn(new_bad, good_out, k).mean()
        + overcorrect * -mean_distances_to_knn(new_bad, bad_out, k).mean()
    )


def bf16(x: np.ndarray) -> np.ndarray:
    return x.astype(ml_dtypes.bfloat16).astype(np.float32)


@dataclass
class Problem:
    fun_jax: Callable[[jax.Array, Any], jax.Array]
    fun_torch: Callable[[Any, Any], Any]
    x0: np.ndarray
    # Float arrays, cast to the dtype of the run, followed by `k` if it is set
    # (a traced int32 for the port, a Python int for PyTorch).
    data: tuple[np.ndarray, ...] | None
    k: int | None = None


def rosenbrock_problem() -> Problem:
    return Problem(
        rosenbrock_jax,
        rosenbrock_torch,
        np.linspace(-1.2, 1.0, 10),
        None,
    )


def least_squares_problem() -> Problem:
    rng = np.random.default_rng(0)
    return Problem(
        least_squares_jax,
        least_squares_torch,
        np.zeros(200),
        (rng.standard_normal((300, 200)), rng.standard_normal(300)),
    )


def ara_problem(seed: int = 0) -> Problem:
    # Random module I/O, rounded to bfloat16 like captured module I/O, and
    # adapters initialised as by apply_lora (A uniform, B zero).
    rng = np.random.default_rng(seed)
    W_base = bf16(rng.standard_normal((D_OUT, D_IN)) / np.sqrt(D_IN))
    good_in = bf16(rng.standard_normal((N_ROWS, D_IN)))
    bad_in = bf16(rng.standard_normal((N_ROWS, D_IN)) + 0.5)
    A = rng.uniform(-1 / np.sqrt(D_IN), 1 / np.sqrt(D_IN), (R, D_IN))
    return Problem(
        ara_jax,
        ara_torch,
        np.concatenate([A.ravel(), np.zeros(D_OUT * R)]),
        (
            W_base,
            np.linalg.norm(W_base, axis=1, keepdims=True),
            good_in,
            bf16(good_in @ W_base.T),
            bad_in,
            bf16(bad_in @ W_base.T),
            np.array([0.5, 0.3, 0.8]),
        ),
        k=K,
    )


def run_torch(
    problem: Problem,
    dtype: np.dtype,
    steps: int,
    **options: Any,
) -> list[dict[str, Any]]:
    """Runs torch.optim.LBFGS and records the result of every outer step."""

    x = torch.tensor(problem.x0.astype(dtype), requires_grad=True)
    data = None
    if problem.data is not None:
        data = tuple(torch.tensor(np.asarray(a, dtype)) for a in problem.data)
        if problem.k is not None:
            data += (problem.k,)
    optimizer = torch.optim.LBFGS([x], line_search_fn="strong_wolfe", **options)

    def closure():
        optimizer.zero_grad()
        loss = problem.fun_torch(x, data)
        loss.backward()
        return loss

    results = []
    for _ in range(steps):
        loss = optimizer.step(closure)
        state = optimizer.state[x]
        result = {
            "loss": loss.item(),
            "x": x.detach().numpy().copy(),
            "func_evals": state["func_evals"],
            "n_iter": state["n_iter"],
        }
        # PyTorch stores the rest of its state only once an iteration has run.
        if "d" in state:
            result |= {
                "d": state["d"].numpy().copy(),
                "t": float(state["t"]),
                "old_dirs": [y.numpy().copy() for y in state["old_dirs"]],
                "old_stps": [s.numpy().copy() for s in state["old_stps"]],
                "ro": [float(ro) for ro in state["ro"]],
                "H_diag": float(state["H_diag"]),
                "prev_flat_grad": state["prev_flat_grad"].numpy().copy(),
                "prev_loss": float(state["prev_loss"]),
            }
        results.append(result)
    return results


def run_port(
    problem: Problem,
    dtype: np.dtype,
    steps: int,
    *,
    lr: float,
    max_iter: int,
    history_size: int,
) -> list[dict[str, Any]]:
    """Runs lbfgs.step inside a jitted function, as ARA does."""

    x = jnp.asarray(problem.x0.astype(dtype))
    data = None
    if problem.data is not None:
        data = tuple(jnp.asarray(np.asarray(a, dtype)) for a in problem.data)
        if problem.k is not None:
            data += (jnp.int32(problem.k),)
    state = jax.tree.map(
        lambda leaf: leaf.astype(dtype) if leaf.dtype == jnp.float32 else leaf,
        lbfgs.init(x.size, history_size),
    )

    @jax.jit
    def optimise(x, state, data):
        return lbfgs.step(
            problem.fun_jax,
            x,
            state,
            data,
            lr=lr,
            max_iter=max_iter,
            history_size=history_size,
        )

    def history(rows: Any, row: int) -> np.ndarray:
        # A history row as the flat vector it stores. Its padding stays zero.
        flat = np.asarray(rows[row]).ravel()
        assert not flat[x.size :].any()
        return flat[: x.size]

    results = []
    for _ in range(steps):
        x, state, loss = optimise(x, state, data)
        # The history in PyTorch's order, oldest entry first.
        rows = [(int(state.head) + i) % history_size for i in range(int(state.num_old))]
        results.append(
            {
                "loss": float(loss),
                "x": np.asarray(x),
                "func_evals": int(state.func_evals),
                "n_iter": int(state.n_iter),
                "d": np.asarray(state.d),
                "t": float(state.t),
                "old_dirs": [history(state.old_dirs, row) for row in rows],
                "old_stps": [history(state.old_stps, row) for row in rows],
                "ro": [float(state.ro[row]) for row in rows],
                "H_diag": float(state.H_diag),
                "prev_flat_grad": np.asarray(state.prev_flat_grad),
                "prev_loss": float(state.prev_loss),
            }
        )
    assert optimise._cache_size() == 1
    return results


# The iterate and the losses, and the state derived from gradients (directions,
# gradients, curvature pairs and their scalars). The latter shrink near a minimum,
# where they are limited by cancellation, so they get their own tolerance.
TRAJECTORY = ["loss", "x", "prev_loss"]
STATE = ["d", "t", "H_diag", "prev_flat_grad", "old_dirs", "old_stps", "ro"]


def flatten(result: dict[str, Any], key: str) -> np.ndarray:
    value = result[key]
    if key in ["old_dirs", "old_stps", "ro"]:
        return np.concatenate([np.ravel(entry) for entry in value] + [np.zeros(0)])
    return np.ravel(np.asarray(value, np.float64))


def compare(
    port: list[dict[str, Any]],
    reference: list[dict[str, Any]],
    rtol: float,
    state_rtol: float,
) -> None:
    """Asserts identical counts after every step, and every field within
    tolerance relative to the largest magnitude that field reaches in the run."""

    for step, (actual, desired) in enumerate(zip(port, reference, strict=True)):
        for counter in ["func_evals", "n_iter"]:
            assert actual[counter] == desired[counter], (
                f"step {step}: {counter} {actual[counter]} != {desired[counter]}"
            )
        assert len(actual["old_dirs"]) == len(desired["old_dirs"]), f"step {step}"

    for keys, tolerance in [(TRAJECTORY, rtol), (STATE, state_rtol)]:
        for key in keys:
            scale = max(np.max(np.abs(flatten(r, key)), initial=0) for r in reference)
            for step, (actual, desired) in enumerate(zip(port, reference)):
                error = np.max(
                    np.abs(flatten(actual, key) - flatten(desired, key)), initial=0
                ) / max(scale, 1e-300)
                assert error <= tolerance, (
                    f"step {step}: {key}: relative error {error:.2e} > {tolerance:.0e}"
                )


@pytest.mark.parametrize(
    "problem, options, rtol, state_rtol",
    [
        # Up to 100 iterations with a history of 3, which wraps around.
        pytest.param(
            rosenbrock_problem,
            {"max_iter": 20, "history_size": 3},
            1e-7,
            1e-4,
            id="rosenbrock",
        ),
        pytest.param(
            rosenbrock_problem,
            {"max_iter": 6, "history_size": 4, "lr": 0.5},
            1e-7,
            1e-4,
            id="rosenbrock-lr",
        ),
        # Converges within the first steps, so the later steps end on the
        # lack-of-progress and loss-change tests.
        pytest.param(
            least_squares_problem,
            {"max_iter": 20, "history_size": 5},
            1e-7,
            1e-4,
            id="least-squares",
        ),
        # Upstream's ARA settings, and a history that wraps around. The ARA
        # trajectory amplifies rounding differences by orders of magnitude even
        # in float64 (see the module docstring).
        pytest.param(
            ara_problem,
            {"max_iter": 20, "history_size": 10},
            1e-7,
            1e-4,
            id="ara",
        ),
        pytest.param(
            ara_problem,
            {"max_iter": 20, "history_size": 3},
            1e-5,
            1e-3,
            id="ara-history",
        ),
    ],
)
@pytest.mark.usefixtures("cpu")
def test_matches_torch_in_float64(
    problem: Callable[[], Problem],
    options: dict[str, Any],
    rtol: float,
    state_rtol: float,
) -> None:
    options = {"lr": 1.0, **options}
    with jax.enable_x64(True):
        port = run_port(problem(), np.float64, 5, **options)
    reference = run_torch(problem(), np.float64, 5, **options)
    compare(port, reference, rtol, state_rtol)


@pytest.mark.parametrize(
    "problem",
    [
        pytest.param(rosenbrock_problem, id="rosenbrock"),
        pytest.param(least_squares_problem, id="least-squares"),
    ],
)
def test_matches_torch_in_float32(
    device: jax.Device,
    problem: Callable[[], Problem],
) -> None:
    # Five steps of five iterations with a history of 3. Float32 agreement is
    # limited by the amplification of rounding differences along the trajectory
    # (iterates within 1e-5 on the CPU), not by the algorithm.
    options = {"lr": 1.0, "max_iter": 5, "history_size": 3}
    port = run_port(problem(), np.float32, 5, **options)
    reference = run_torch(problem(), np.float32, 5, **options)
    compare(port, reference, 1e-4, 1e-2)


@pytest.mark.parametrize("seed", [0, 1])
def test_ara_matches_torch_loss_in_float32(device: jax.Device, seed: int) -> None:
    # The trajectories diverge (see the module docstring), so only the losses
    # reached after upstream's five steps are compared, relative to the
    # initial loss.
    options = {"lr": 1.0, "max_iter": 20, "history_size": 10}
    port = run_port(ara_problem(seed), np.float32, 6, **options)
    reference = run_torch(ara_problem(seed), np.float32, 6, **options)

    initial_loss = reference[0]["loss"]
    assert port[0]["loss"] == pytest.approx(initial_loss, rel=1e-5)
    final_loss = reference[-1]["loss"]
    assert final_loss < 0
    assert abs(port[-1]["loss"] - final_loss) <= 0.05 * abs(initial_loss)


@pytest.mark.usefixtures("cpu")
def test_edge_cases_match_torch() -> None:
    options = {"lr": 1.0, "history_size": 2}
    with jax.enable_x64(True):
        # max_eval = 1, so the line search runs with max_ls = 0 and returns
        # the bracket [0, t] after a single evaluation.
        for max_iter in [1, 2, 3]:
            port = run_port(
                rosenbrock_problem(), np.float64, 4, max_iter=max_iter, **options
            )
            reference = run_torch(
                rosenbrock_problem(), np.float64, 4, max_iter=max_iter, **options
            )
            compare(port, reference, 1e-10, 1e-10)

        # A history of one entry.
        port = run_port(
            rosenbrock_problem(), np.float64, 3, lr=1.0, max_iter=10, history_size=1
        )
        reference = run_torch(
            rosenbrock_problem(), np.float64, 3, lr=1.0, max_iter=10, history_size=1
        )
        compare(port, reference, 1e-10, 1e-10)

        # A gradient below the tolerance at the start: PyTorch returns at once,
        # counting the evaluation but no iteration.
        optimal = Problem(rosenbrock_jax, rosenbrock_torch, np.ones(4), None)
        port = run_port(optimal, np.float64, 2, max_iter=20, **options)
        reference = run_torch(optimal, np.float64, 2, max_iter=20, **options)
        assert [r["func_evals"] for r in port] == [1, 2]
        assert [r["n_iter"] for r in port] == [0, 0]
        assert np.array_equal(port[-1]["x"], np.ones(4))
        for actual, desired in zip(port, reference, strict=True):
            assert actual["func_evals"] == desired["func_evals"]
            assert actual["n_iter"] == desired["n_iter"]

        # A gradient above tolerance_grad, but with g·d above -tolerance_change:
        # the first iteration stops before the line search.
        def quadratic_jax(x, data):
            return 0.5 * jnp.sum(x * x)

        def quadratic_torch(x, data):
            return 0.5 * torch.sum(x * x)

        flat = Problem(quadratic_jax, quadratic_torch, np.full(3, 1e-5), None)
        port = run_port(flat, np.float64, 2, max_iter=20, **options)
        reference = run_torch(flat, np.float64, 2, max_iter=20, **options)
        compare(port, reference, 1e-12, 1e-12)
        assert port[0]["n_iter"] == 1
        assert port[0]["func_evals"] == 1


@pytest.mark.usefixtures("cpu")
def test_step_traces_without_host_synchronisation() -> None:
    problem = least_squares_problem()
    data = tuple(jnp.asarray(a, jnp.float32) for a in problem.data)
    x = jnp.zeros(200, jnp.float32)

    optimise = jax.jit(
        functools.partial(
            lbfgs.step, least_squares_jax, lr=1.0, max_iter=20, history_size=10
        ),
        donate_argnums=(1,),
    )
    jaxpr = str(jax.make_jaxpr(optimise)(x, lbfgs.init(200, 10), data))
    assert "while" in jaxpr
    assert "callback" not in jaxpr

    # New values for every traced argument reuse the executable, and the
    # donated state is replaced by a fresh one each time.
    state = lbfgs.init(200, 10)
    for scale in [1.0, 2.0]:
        x, state, _ = optimise(x, state, tuple(scale * a for a in data))
    assert optimise._cache_size() == 1


@pytest.mark.usefixtures("cpu")
def test_mismatched_state_raises() -> None:
    with pytest.raises(ValueError, match="history_size"):
        lbfgs.step(
            rosenbrock_jax,
            jnp.zeros(3),
            lbfgs.init(3, 4),
            None,
            lr=1.0,
            max_iter=5,
            history_size=5,
        )
    with pytest.raises(ValueError, match="initialised for 3 parameters"):
        lbfgs.step(
            rosenbrock_jax,
            jnp.zeros(4),
            lbfgs.init(3, 4),
            None,
            lr=1.0,
            max_iter=5,
            history_size=4,
        )


@pytest.mark.parametrize(
    "n_params, row_shape",
    [
        # ARA's down_proj and o_proj of Qwen3-4B at rank 50: whole tiles.
        (50 * (9728 + 2560), (4800, 128)),
        (50 * (4096 + 2560), (2600, 128)),
        (1024, (8, 128)),
        # Padded to whole 8 × 128 tiles.
        (1, (8, 128)),
        (1025, (16, 128)),
    ],
)
def test_history_rows_occupy_whole_tiles(
    n_params: int,
    row_shape: tuple[int, int],
) -> None:
    # Each history row is a block of whole (8, 128) TPU tiles, so that reading or
    # writing one row moves only that row's bytes (see lbfgs.history_row_shape).
    state = lbfgs.init(n_params, 10)
    assert state.old_dirs.shape == state.old_stps.shape == (10, *row_shape)
    assert lbfgs.history_row_shape(n_params) == row_shape
    assert state.d.shape == state.prev_flat_grad.shape == (n_params,)


@pytest.mark.parametrize("n_params", [1000, 2048])
def test_history_layout_does_not_change_results(
    device: jax.Device,
    monkeypatch: pytest.MonkeyPatch,
    n_params: int,
) -> None:
    # History rows are stored in whole tiles but read and written as flat vectors,
    # so the optimiser takes exactly the steps it takes with flat rows, padded
    # (1000 parameters) or not.
    rng = np.random.default_rng(0)
    data = (
        jnp.asarray(10 ** rng.uniform(0, 3, n_params), jnp.float32),
        jnp.asarray(rng.standard_normal(n_params), jnp.float32),
    )

    def quadratic(x: jax.Array, data: tuple[jax.Array, jax.Array]) -> jax.Array:
        curvature, target = data
        return 0.5 * jnp.sum(curvature * (x - target) ** 2)

    def run() -> tuple[np.ndarray, lbfgs.LBFGSState]:
        # A new function for each run, so that each traces the current layout.
        step = jax.jit(
            functools.partial(
                lbfgs.step, quadratic, lr=1.0, max_iter=20, history_size=4
            )
        )
        x = jnp.zeros(n_params, jnp.float32)
        state = lbfgs.init(n_params, 4)
        for _ in range(3):
            x, state, _ = step(x, state, data)
        return np.asarray(x), state

    x, state = run()
    with monkeypatch.context() as patch:
        patch.setattr(lbfgs, "history_row_shape", lambda n_params: (1, n_params))
        x_flat, state_flat = run()

    assert state.old_dirs.shape[1:] != state_flat.old_dirs.shape[1:]
    assert int(state.num_old) == 4
    assert np.array_equal(x, x_flat)
    for history, history_flat in [
        (state.old_dirs, state_flat.old_dirs),
        (state.old_stps, state_flat.old_stps),
    ]:
        rows = np.asarray(history).reshape(4, -1)
        assert np.array_equal(rows[:, :n_params], np.asarray(history_flat)[:, 0])
        assert not rows[:, n_params:].any()
