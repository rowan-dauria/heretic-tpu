# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

"""A traceable port of `torch.optim.LBFGS.step` with `line_search_fn="strong_wolfe"`.

The port mirrors PyTorch's algorithm branch by branch, including its constants,
tolerances and the state it carries across steps, but runs entirely in
`lax.while_loop` and `lax.cond`, so that a step traces inside an enclosing jitted
function with no host synchronisation. Arithmetic is in the dtype of the
parameters, float32 in practice (PyTorch keeps some scalars as Python floats, which
only changes rounding), and every dot passes `Precision.HIGHEST` (see "Matmul
precision" in docs/DESIGN.md).
"""

from collections.abc import Callable
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax import Array, lax

# Defaults of torch.optim.LBFGS, which upstream does not override.
TOLERANCE_GRAD = 1e-7
TOLERANCE_CHANGE = 1e-9

# Defaults of torch.optim.lbfgs._strong_wolfe.
C1 = 1e-4
C2 = 0.9

Objective = Callable[[Array, Any], Array]


class LBFGSState(NamedTuple):
    """The state `torch.optim.LBFGS` keeps across steps, as a pytree of arrays.

    The history (PyTorch's lists `old_dirs`, `old_stps` and `ro`) is a ring buffer
    of `history_size` rows: its `num_old` entries, oldest first, are stored in rows
    `head, head + 1, …` modulo `history_size`.
    """

    func_evals: Array  # int32 [], total number of function evaluations
    n_iter: Array  # int32 [], total number of iterations
    d: Array  # float32 [n], the last search direction
    t: Array  # float32 [], the last step length
    old_dirs: Array  # float32 [history_size, n], gradient differences y
    old_stps: Array  # float32 [history_size, n], steps s
    ro: Array  # float32 [history_size], 1 / (y·s)
    num_old: Array  # int32 [], number of history entries
    head: Array  # int32 [], row of the oldest history entry
    H_diag: Array  # float32 [], scale of the initial inverse Hessian approximation
    prev_flat_grad: Array  # float32 [n]
    prev_loss: Array  # float32 []


def init(n_params: int, history_size: int) -> LBFGSState:
    """The state of a freshly constructed optimiser over `n_params` parameters."""

    # Every leaf is a separate buffer, so that the state can be donated.
    return LBFGSState(
        func_evals=jnp.zeros((), jnp.int32),
        n_iter=jnp.zeros((), jnp.int32),
        d=jnp.zeros((n_params,), jnp.float32),
        t=jnp.zeros((), jnp.float32),
        old_dirs=jnp.zeros((history_size, n_params), jnp.float32),
        old_stps=jnp.zeros((history_size, n_params), jnp.float32),
        ro=jnp.zeros((history_size,), jnp.float32),
        num_old=jnp.zeros((), jnp.int32),
        head=jnp.zeros((), jnp.int32),
        H_diag=jnp.ones((), jnp.float32),
        prev_flat_grad=jnp.zeros((n_params,), jnp.float32),
        prev_loss=jnp.zeros((), jnp.float32),
    )


def _dot(a: Array, b: Array) -> Array:
    return jnp.dot(a, b, precision=lax.Precision.HIGHEST)


# Python's max(a, b) and min(a, b), which PyTorch applies to scalar tensors.
# Unlike jnp.maximum and jnp.minimum, they return the first argument unless the
# second compares greater (smaller), which decides how NaN propagates.
def _py_max(a: Array, b: Array) -> Array:
    return jnp.where(b > a, b, a)


def _py_min(a: Array, b: Array) -> Array:
    return jnp.where(b < a, b, a)


def _cubic_interpolate(
    x1: Array,
    f1: Array,
    g1: Array,
    x2: Array,
    f2: Array,
    g2: Array,
    bounds: tuple[Array, Array] | None = None,
) -> Array:
    # Minimiser of the cubic interpolating two points with function and
    # derivative values, clamped to the bounds (torch.optim.lbfgs._cubic_interpolate).
    if bounds is not None:
        xmin_bound, xmax_bound = bounds
    else:
        xmin_bound = jnp.where(x1 <= x2, x1, x2)
        xmax_bound = jnp.where(x1 <= x2, x2, x1)

    d1 = g1 + g2 - 3 * (f1 - f2) / (x1 - x2)
    d2_square = d1**2 - g1 * g2
    # The square root is only used where its argument is non-negative.
    d2 = jnp.sqrt(jnp.maximum(d2_square, 0))
    min_pos = jnp.where(
        x1 <= x2,
        x2 - (x2 - x1) * ((g2 + d2 - d1) / (g2 - g1 + 2 * d2)),
        x1 - (x1 - x2) * ((g1 + d2 - d1) / (g1 - g2 + 2 * d2)),
    )
    return jnp.where(
        d2_square >= 0,
        _py_min(_py_max(min_pos, xmin_bound), xmax_bound),
        (xmin_bound + xmax_bound) / 2.0,
    )


class _Bracketing(NamedTuple):
    t: Array
    f_new: Array
    g_new: Array
    gtd_new: Array
    t_prev: Array
    f_prev: Array
    g_prev: Array
    gtd_prev: Array
    ls_iter: Array
    ls_func_evals: Array


class _Zoom(NamedTuple):
    # The bracket, as arrays of two points (a single point is stored twice).
    bracket: Array
    bracket_f: Array
    bracket_g: Array
    bracket_gtd: Array
    low_pos: Array
    insuf_progress: Array
    done: Array
    ls_iter: Array
    ls_func_evals: Array


def _strong_wolfe(
    obj_func: Callable[[Array], tuple[Array, Array]],
    t: Array,
    d: Array,
    f: Array,
    g: Array,
    gtd: Array,
    max_ls: Array,
) -> tuple[Array, Array, Array, Array]:
    # Port of torch.optim.lbfgs._strong_wolfe. `obj_func(t)` evaluates the loss
    # and gradient at x + t * d. Returns the loss, gradient and step length of the
    # accepted point and the number of evaluations.
    d_norm = jnp.max(jnp.abs(d))

    # Evaluate the objective and gradient using the initial step.
    f_new, g_new = obj_func(t)

    # Bracket an interval containing a point satisfying the Wolfe criteria.
    # PyTorch checks the conditions at the top of its loop and breaks when one
    # holds, so the loop below stops when they hold, and the bracket is built
    # from the final carry afterwards.
    def conditions(c: _Bracketing) -> tuple[Array, Array, Array]:
        armijo_fails = (c.f_new > (f + C1 * c.t * gtd)) | (
            (c.ls_iter > 1) & (c.f_new >= c.f_prev)
        )
        curvature_holds = jnp.abs(c.gtd_new) <= -C2 * gtd
        derivative_non_negative = c.gtd_new >= 0
        return armijo_fails, curvature_holds, derivative_non_negative

    def bracketing_continues(c: _Bracketing) -> Array:
        return (c.ls_iter < max_ls) & ~jnp.any(jnp.stack(conditions(c)))

    def extrapolate(c: _Bracketing) -> _Bracketing:
        # Interpolate.
        min_step = c.t + 0.01 * (c.t - c.t_prev)
        max_step = c.t * 10
        t = _cubic_interpolate(
            c.t_prev,
            c.f_prev,
            c.gtd_prev,
            c.t,
            c.f_new,
            c.gtd_new,
            bounds=(min_step, max_step),
        )

        # Next step.
        f_new, g_new = obj_func(t)
        return _Bracketing(
            t=t,
            f_new=f_new,
            g_new=g_new,
            gtd_new=_dot(g_new, d),
            t_prev=c.t,
            f_prev=c.f_new,
            g_prev=c.g_new,
            gtd_prev=c.gtd_new,
            ls_iter=c.ls_iter + 1,
            ls_func_evals=c.ls_func_evals + 1,
        )

    c = lax.while_loop(
        bracketing_continues,
        extrapolate,
        _Bracketing(
            t=t,
            f_new=f_new,
            g_new=g_new,
            gtd_new=_dot(g_new, d),
            t_prev=jnp.zeros_like(t),
            f_prev=f,
            g_prev=g,
            gtd_prev=gtd,
            ls_iter=jnp.zeros((), jnp.int32),
            ls_func_evals=jnp.ones((), jnp.int32),
        ),
    )

    armijo_fails, curvature_holds, _ = conditions(c)
    # Reached the maximum number of iterations? PyTorch checks this after the
    # loop, so it takes precedence over the conditions.
    exhausted = c.ls_iter == max_ls
    # Otherwise the first condition that held in PyTorch's if/elif chain decides:
    # a failed Armijo test or a non-negative derivative give the bracket
    # [t_prev, t], and the curvature condition gives the single point [t].
    single_point = ~exhausted & ~armijo_fails & curvature_holds

    def pair(first: Array, second: Array) -> Array:
        return jnp.stack([first, second])

    bracket = jnp.where(
        exhausted,
        pair(jnp.zeros_like(c.t), c.t),
        jnp.where(single_point, pair(c.t, c.t), pair(c.t_prev, c.t)),
    )
    bracket_f = jnp.where(
        exhausted,
        pair(f, c.f_new),
        jnp.where(single_point, pair(c.f_new, c.f_new), pair(c.f_prev, c.f_new)),
    )
    bracket_g = jnp.where(
        exhausted,
        pair(g, c.g_new),
        jnp.where(single_point, pair(c.g_new, c.g_new), pair(c.g_prev, c.g_new)),
    )
    # PyTorch leaves bracket_gtd unset when the search is exhausted; the zoom
    # phase never reads it then.
    bracket_gtd = jnp.where(
        single_point,
        pair(c.gtd_new, c.gtd_new),
        pair(c.gtd_prev, c.gtd_new),
    )

    # Zoom phase: we now have a point satisfying the criteria, or a bracket
    # around it. We refine the bracket until we find the exact point satisfying
    # the criteria.
    def low_position(bracket_f: Array) -> Array:
        return jnp.where(bracket_f[0] <= bracket_f[1], 0, 1)

    def zoom_continues(z: _Zoom) -> Array:
        # PyTorch also breaks at the top of its loop when the line-search
        # bracket is too small.
        bracket_too_small = (
            jnp.abs(z.bracket[1] - z.bracket[0]) * d_norm < TOLERANCE_CHANGE
        )
        return ~z.done & (z.ls_iter < max_ls) & ~bracket_too_small

    def refine(z: _Zoom) -> _Zoom:
        low_pos = z.low_pos
        high_pos = 1 - low_pos

        # Compute the new trial value.
        t = _cubic_interpolate(
            z.bracket[0],
            z.bracket_f[0],
            z.bracket_gtd[0],
            z.bracket[1],
            z.bracket_f[1],
            z.bracket_gtd[1],
        )

        # Test that we are making sufficient progress: if `t` is close to the
        # boundary and we made insufficient progress in the last step, or `t` is
        # at one of the boundaries, move `t` to a position 0.1 * len(bracket)
        # away from the nearest boundary point.
        bracket_max = _py_max(z.bracket[0], z.bracket[1])
        bracket_min = _py_min(z.bracket[0], z.bracket[1])
        eps = 0.1 * (bracket_max - bracket_min)
        near_boundary = _py_min(bracket_max - t, t - bracket_min) < eps
        at_boundary = z.insuf_progress | (t >= bracket_max) | (t <= bracket_min)
        t = jnp.where(
            near_boundary & at_boundary,
            jnp.where(
                jnp.abs(t - bracket_max) < jnp.abs(t - bracket_min),
                bracket_max - eps,
                bracket_min + eps,
            ),
            t,
        )
        insuf_progress = near_boundary & ~at_boundary

        # Evaluate the new point.
        f_new, g_new = obj_func(t)
        gtd_new = _dot(g_new, d)

        # Armijo condition not satisfied or not lower than the lowest point:
        # the new point replaces the high end. Otherwise it becomes the new low
        # end, and if the derivative points away from the high end, the old low
        # end becomes the new high end first.
        replaces_high = (f_new > (f + C1 * t * gtd)) | (f_new >= z.bracket_f[low_pos])
        wolfe_holds = jnp.abs(gtd_new) <= -C2 * gtd
        low_becomes_high = (
            ~replaces_high
            & ~wolfe_holds
            & (gtd_new * (z.bracket[high_pos] - z.bracket[low_pos]) >= 0)
        )
        new_pos = jnp.where(replaces_high, high_pos, low_pos)

        def update(values: Array, new_value: Array) -> Array:
            values = jnp.where(
                low_becomes_high,
                values.at[high_pos].set(values[low_pos]),
                values,
            )
            return values.at[new_pos].set(new_value)

        bracket_f = update(z.bracket_f, f_new)
        return _Zoom(
            bracket=update(z.bracket, t),
            bracket_f=bracket_f,
            bracket_g=update(z.bracket_g, g_new),
            bracket_gtd=update(z.bracket_gtd, gtd_new),
            low_pos=jnp.where(replaces_high, low_position(bracket_f), low_pos),
            insuf_progress=insuf_progress,
            done=~replaces_high & wolfe_holds,
            ls_iter=z.ls_iter + 1,
            ls_func_evals=z.ls_func_evals + 1,
        )

    z = lax.while_loop(
        zoom_continues,
        refine,
        _Zoom(
            bracket=bracket,
            bracket_f=bracket_f,
            bracket_g=bracket_g,
            bracket_gtd=bracket_gtd,
            low_pos=low_position(bracket_f),
            insuf_progress=jnp.zeros((), bool),
            done=single_point,
            ls_iter=c.ls_iter,
            ls_func_evals=c.ls_func_evals,
        ),
    )

    return (
        z.bracket_f[z.low_pos],
        z.bracket_g[z.low_pos],
        z.bracket[z.low_pos],
        z.ls_func_evals,
    )


def _two_loop_recursion(flat_grad: Array, state: LBFGSState) -> Array:
    # The approximate inverse Hessian (L-BFGS) multiplied by the negative
    # gradient, with PyTorch's loop order and in-place update formulas.
    history_size = state.ro.shape[0]

    def row(i: Array) -> Array:
        return (state.head + i) % history_size

    def newest_to_oldest(j: Array, carry: tuple[Array, Array]) -> tuple[Array, Array]:
        q, al = carry
        i = state.num_old - 1 - j
        al_i = _dot(state.old_stps[row(i)], q) * state.ro[row(i)]
        return q - al_i * state.old_dirs[row(i)], al.at[i].set(al_i)

    q, al = lax.fori_loop(
        0,
        state.num_old,
        newest_to_oldest,
        (-flat_grad, jnp.zeros_like(state.ro)),
    )

    # Multiply by the initial Hessian.
    def oldest_to_newest(i: Array, r: Array) -> Array:
        be_i = _dot(state.old_dirs[row(i)], r) * state.ro[row(i)]
        return r + (al[i] - be_i) * state.old_stps[row(i)]

    return lax.fori_loop(0, state.num_old, oldest_to_newest, q * state.H_diag)


def _update_memory(flat_grad: Array, state: LBFGSState) -> LBFGSState:
    y = flat_grad - state.prev_flat_grad
    s = state.d * state.t
    ys = _dot(y, s)  # y*s

    def store(state: LBFGSState) -> LBFGSState:
        # When the history is full, the newest entry replaces the oldest one.
        history_size = state.ro.shape[0]
        full = state.num_old == history_size
        position = jnp.where(
            full, state.head, (state.head + state.num_old) % history_size
        )
        return state._replace(
            old_dirs=state.old_dirs.at[position].set(y),
            old_stps=state.old_stps.at[position].set(s),
            ro=state.ro.at[position].set(1.0 / ys),
            num_old=jnp.where(full, state.num_old, state.num_old + 1),
            head=jnp.where(full, (state.head + 1) % history_size, state.head),
            # Update the scale of the initial Hessian approximation.
            H_diag=ys / _dot(y, y),  # (y*y)
        )

    return lax.cond(ys > 1e-10, store, lambda state: state, state)


class _Iteration(NamedTuple):
    state: LBFGSState
    x: Array
    loss: Array
    flat_grad: Array
    n_iter: Array
    current_evals: Array
    done: Array


def step(
    fun: Objective,
    x: Array,
    state: LBFGSState,
    data: Any,
    *,
    lr: float,
    max_iter: int,
    history_size: int,
) -> tuple[Array, LBFGSState, Array]:
    """One `torch.optim.LBFGS.step` with `line_search_fn="strong_wolfe"`.

    `x` is the flat parameter vector (float32, like the floating-point leaves of
    `state`), and `fun(x, data)` returns the loss there. `fun` may close over
    static Python values but never over arrays, which go through the pytree
    `data`. `lr`, `max_iter` and `history_size` are Python values, and
    `max_eval = max_iter * 5 // 4` as in PyTorch. This function is meant to be
    traced inside an enclosing jitted function.

    Returns the new parameters, the new state, and the loss at the start of the
    step (what PyTorch's `step` returns).
    """

    if state.ro.shape != (history_size,):
        raise ValueError(
            f"The state holds a history of size {state.ro.shape[0]}, "
            f"but history_size is {history_size}."
        )

    max_eval = max_iter * 5 // 4
    value_and_grad = jax.value_and_grad(fun)
    dtype = x.dtype

    def evaluate(x: Array) -> tuple[Array, Array]:
        loss, flat_grad = value_and_grad(x, data)
        return loss.astype(dtype), flat_grad

    # Evaluate the initial f(x) and df/dx.
    orig_loss, flat_grad = evaluate(x)
    state = state._replace(func_evals=state.func_evals + 1)

    def iterate(it: _Iteration) -> _Iteration:
        n_iter = it.n_iter + 1
        state = it.state._replace(n_iter=it.state.n_iter + 1)
        first_iteration = state.n_iter == 1

        # Compute the gradient descent direction.
        def steepest_descent(state: LBFGSState) -> LBFGSState:
            return state._replace(
                d=-it.flat_grad,
                num_old=jnp.zeros((), jnp.int32),
                head=jnp.zeros((), jnp.int32),
                H_diag=jnp.ones_like(state.H_diag),
            )

        def lbfgs_direction(state: LBFGSState) -> LBFGSState:
            state = _update_memory(it.flat_grad, state)
            return state._replace(d=_two_loop_recursion(it.flat_grad, state))

        state = lax.cond(first_iteration, steepest_descent, lbfgs_direction, state)

        # Compute the step length: reset the initial guess for the step size.
        t = jnp.where(
            first_iteration,
            _py_min(jnp.ones((), dtype), 1.0 / jnp.sum(jnp.abs(it.flat_grad))) * lr,
            jnp.asarray(lr, dtype),
        )
        state = state._replace(prev_flat_grad=it.flat_grad, prev_loss=it.loss, t=t)

        # Directional derivative.
        gtd = _dot(it.flat_grad, state.d)  # g * d

        # The directional derivative is below tolerance.
        no_descent = gtd > -TOLERANCE_CHANGE

        def line_search(_: None) -> tuple[Array, Array, Array, Array, Array]:
            def obj_func(t: Array) -> tuple[Array, Array]:
                return evaluate(it.x + t * state.d)

            loss, flat_grad, t, ls_func_evals = _strong_wolfe(
                obj_func,
                state.t,
                state.d,
                it.loss,
                it.flat_grad,
                gtd,
                max_ls=max_eval - it.current_evals,
            )
            return it.x + t * state.d, loss, flat_grad, t, ls_func_evals

        def stop(_: None) -> tuple[Array, Array, Array, Array, Array]:
            return it.x, it.loss, it.flat_grad, state.t, jnp.zeros((), jnp.int32)

        x, loss, flat_grad, t, ls_func_evals = lax.cond(
            no_descent, stop, line_search, None
        )

        # Update the function evaluation counts.
        current_evals = it.current_evals + ls_func_evals
        state = state._replace(t=t, func_evals=state.func_evals + ls_func_evals)

        # Check conditions.
        done = (
            no_descent
            | (n_iter == max_iter)
            | (current_evals >= max_eval)
            # Optimal condition.
            | (jnp.max(jnp.abs(flat_grad)) <= TOLERANCE_GRAD)
            # Lack of progress.
            | (jnp.max(jnp.abs(state.d * t)) <= TOLERANCE_CHANGE)
            | (jnp.abs(loss - state.prev_loss) < TOLERANCE_CHANGE)
        )

        return _Iteration(
            state=state,
            x=x,
            loss=loss,
            flat_grad=flat_grad,
            n_iter=n_iter,
            current_evals=current_evals,
            done=done,
        )

    # If the initial point is already optimal, PyTorch returns at once and
    # leaves the rest of the state unchanged; starting the loop as done does
    # the same.
    it = lax.while_loop(
        lambda it: ~it.done & (it.n_iter < max_iter),
        iterate,
        _Iteration(
            state=state,
            x=x,
            loss=orig_loss,
            flat_grad=flat_grad,
            n_iter=jnp.zeros((), jnp.int32),
            current_evals=jnp.ones((), jnp.int32),
            done=jnp.max(jnp.abs(flat_grad)) <= TOLERANCE_GRAD,
        ),
    )

    return it.x, it.state, orig_loss
