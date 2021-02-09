########################################################################################################################
# Module: _utils.py
# Description: Some useful functions including TemporalGaussian potential evaluations and leapfrog integrator.
#
# Web: https://github.com/SamDuffield/mocat
########################################################################################################################

from typing import Any, Union, Callable, Tuple
from functools import partial
from warnings import warn

from decorator import decorator
from jax.lax import scan, while_loop, cond
from jax import jit, numpy as np, vmap, random
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from mocat.src.core import cdict


@partial(jit, static_argnums=(0,))
def leave_one_out_indices(n: int,
                          remove_index: int) -> np.ndarray:
    full_inds = np.arange(n + 1)
    leave_one_out_inds = np.where(np.arange(n) < remove_index, full_inds[:-1], full_inds[1:])
    return leave_one_out_inds


@jit
def _vectorised_gaussian_potential(x: np.ndarray,
                                   mean: Union[float, np.ndarray],
                                   sqrt_prec: np.ndarray) -> Union[float, np.ndarray]:
    diff = x - mean
    return 0.5 * np.sum(np.square(diff @ sqrt_prec), axis=-1)


@jit
def _mv_gaussian_potential(x: np.ndarray,
                           mean: Union[np.ndarray, float],
                           prec: np.ndarray) -> float:
    diff = x - mean
    return 0.5 * diff.T @ prec @ diff


@jit
def _mv_gaussian_potential_diag(x: np.ndarray,
                                mean: Union[np.ndarray, float],
                                prec_diag: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    diff = x - mean
    return 0.5 * np.sum(np.square(diff) * prec_diag, axis=-1)


def gaussian_potential(x: np.ndarray,
                       mean: Union[float, np.ndarray] = 0.,
                       prec: Union[float, np.ndarray] = None,
                       sqrt_prec: Union[float, np.ndarray] = None,
                       det_prec: float = None) -> Union[float, np.ndarray]:
    d = x.shape[-1]

    if prec is None and sqrt_prec is None:
        prec = 1.

    if isinstance(prec, float):
        prec = np.ones(d) * prec

    if isinstance(sqrt_prec, float):
        sqrt_prec = np.ones(d) * sqrt_prec

    if det_prec is None:
        if prec is not None and prec.ndim < 2:
            det_prec = np.prod(prec)
        elif sqrt_prec is not None and sqrt_prec.ndim < 2:
            det_prec = np.prod(sqrt_prec) ** 2

    if det_prec is None:
        # full precision matrix given but no det - computing without norm constant
        neg_log_z = 0
        warn('gaussian_potential queried with non-diagonal prec (or sqrt-prec) but no det_prec given'
             ' -> executing without normalising constant term')
    else:
        neg_log_z = (d * np.log(2 * np.pi) - np.log(det_prec)) / 2

    if x.ndim == 1 and sqrt_prec is None:
        # Single x value (not vectorised)
        if prec is None:
            out_val = _mv_gaussian_potential_diag(x, mean, 1.)
        elif prec.ndim < 2:
            out_val = _mv_gaussian_potential_diag(x, mean, prec)
        else:
            out_val = _mv_gaussian_potential(x, mean, prec)
    else:
        # Multiple x values (vectorised)
        if prec is not None and sqrt_prec is None:
            if prec.ndim < 2:
                sqrt_prec = np.sqrt(prec)
            else:
                sqrt_prec = np.linalg.cholesky(prec)
                warn('vectorised gaussian_potential queried with prec rather than sqrt_prec'
                     '-> executing using Cholesky decomp')

        if sqrt_prec is None:
            out_val = _mv_gaussian_potential_diag(x, mean, 1.)
        elif sqrt_prec.ndim < 2:
            out_val = _mv_gaussian_potential_diag(x, mean, sqrt_prec ** 2)
        else:
            out_val = _vectorised_gaussian_potential(x, mean, sqrt_prec)
    return out_val + neg_log_z


@partial(jit, static_argnums=(0,))
def leapfrog(potential_and_grad: Callable,
             state: cdict,
             stepsize: float,
             random_keys: np.ndarray) -> cdict:
    leapfrog_steps = len(random_keys)

    def leapfrog_step(init_state: cdict,
                      i: int):
        new_state = init_state.copy()

        p_half = init_state.momenta - stepsize / 2. * init_state.grad_potential
        new_state.value = init_state.value + stepsize * p_half

        new_state.potential, new_state.grad_potential = potential_and_grad(new_state.value, random_keys[i])

        new_state.momenta = p_half - stepsize / 2. * new_state.grad_potential

        next_sample_chain = new_state.copy()
        next_sample_chain.momenta = np.vstack([p_half, new_state.momenta])
        return new_state, next_sample_chain

    final_leapfrog, all_leapfrog = scan(leapfrog_step, state, np.arange(leapfrog_steps))

    all_leapfrog.momenta = np.concatenate(all_leapfrog.momenta)

    all_leapfrog = state[np.newaxis] + all_leapfrog

    return all_leapfrog


@decorator
def _metric_plot_decorate(plot_func: Callable,
                          *args, **kwargs) -> Union[plt.Axes, Tuple[Figure, plt.Axes]]:
    sample = args[0]
    ax = args[-1]

    if 'title' not in kwargs.keys() and hasattr(sample, 'name'):
        title = sample.name
    elif 'title' in kwargs.keys():
        title = kwargs['title']
        del kwargs['title']
    else:
        title = None

    if ax is None:
        fig, ax = plt.subplots()
        plot_func(*args[:-1], ax, **kwargs)

        if title is not None:
            ax.set_title(title)

        return fig, ax
    else:
        plot_func(*args, **kwargs)
        ax = args[-1]
        if title is not None:
            ax.set_title(title)
        return ax


@partial(jit, static_argnums=(0, 1, 3))
def _while_loop_stacked(cond_fun: Callable,
                        body_fun: Callable,
                        init_carry: Tuple[Any, Any],
                        max_iter: int) -> Tuple[Any, int]:
    # init_carry = (init_state, extra)
    # cond_func: (state, extra) -> bool
    # body_func: (state, extra) -> (state, extra)
    # while_loop_stacked -> stacked(state)

    def update_cond(previous_carry: Any,
                    final_iter: int,
                    live_iter: int) -> bool:
        return cond(live_iter <= (final_iter + 1),
                    lambda args: cond_fun(*args),
                    lambda _: False,
                    previous_carry)

    def update_kernel(previous_val_and_final_iter: Tuple[Any, int],
                      live_iter: int) -> Tuple[Tuple[Any, int], Any]:
        previous_val, final_iter = previous_val_and_final_iter

        update_bool = update_cond(previous_val, final_iter, live_iter)

        final_iter = np.where(update_bool, live_iter, final_iter)

        new_val = cond(update_bool,
                       lambda args: body_fun(*args),
                       lambda _: init_carry,
                       previous_val)

        return (new_val, final_iter), new_val[0]

    final_val_and_final_iter, stack = scan(update_kernel, (init_carry, 0), np.arange(1, max_iter + 1))
    _, final_final_iter = final_val_and_final_iter
    return stack, final_final_iter


def while_loop_stacked(cond_fun: Callable,
                       body_fun: Callable,
                       init_carry: Tuple[Any, Any],
                       max_iter: int = 1000) -> Any:
    full_stack, final_int = _while_loop_stacked(cond_fun, body_fun, init_carry, max_iter)
    return tuple(a[:final_int] for a in full_stack) if isinstance(full_stack, tuple) else full_stack[:final_int]


@partial(jit, static_argnums=(0,))
def bisect(fun: Callable,
           bounds: Union[list, np.ndarray],
           max_iter: int = 1000,
           tol: float = 1e-5) -> Tuple[np.ndarray, np.ndarray, int]:
    evals = np.array([fun(bounds[0]), fun(bounds[1])])
    increasing_bool = evals[1] > evals[0]

    def conv_check(int_state: Tuple[np.ndarray, np.ndarray, int]) -> bool:
        int_bounds, int_evals, iter_ind = int_state
        return ~np.any(np.array([np.min(np.abs(int_evals)) < tol,
                                 iter_ind >= max_iter,
                                 np.all(int_evals < 0),
                                 np.all(int_evals > 0)]))

    def body_func(int_state: Tuple[np.ndarray, np.ndarray, int]) -> Tuple[np.ndarray, np.ndarray, int]:
        int_bounds, int_evals, iter_ind = int_state

        new_pos = int_bounds[0] - int_evals[0] * (int_bounds[1] - int_bounds[0]) / (int_evals[1] - int_evals[0])
        new_eval = fun(new_pos)

        replace_upper = np.where(increasing_bool, new_eval > 0, new_eval < 0)

        out_bounds = np.where(replace_upper, np.array([int_bounds[0], new_pos]), np.array([new_pos, int_bounds[1]]))
        out_evals = np.where(replace_upper, np.array([int_evals[0], new_eval]), np.array([new_eval, int_evals[1]]))

        return out_bounds, out_evals, iter_ind + 1

    fin_bounds, fin_evals, fin_iter = while_loop(conv_check,
                                                 body_func,
                                                 (bounds, evals, 0))

    return fin_bounds, fin_evals, fin_iter


def extract_dimension(*args):
    for a in args:
        if isinstance(a, np.ndarray) and a.ndim > 0:
            return a.shape[-1]
    return None


def reset_covariance(obj: Any,
                     key: str,
                     value: Any):
    prec_key = key.replace('covariance', 'precision')
    if value.ndim < 2:
        sqrt_val = np.sqrt(value)
        setattr(obj, key + '_sqrt', sqrt_val)
        setattr(obj, prec_key + '_sqrt', 1 / sqrt_val)
        setattr(obj, prec_key + '_det', 1 / value)
    else:
        sqrt_mat = np.linalg.cholesky(value)
        setattr(obj, key + '_sqrt', sqrt_mat)
        setattr(obj, prec_key + '_sqrt', np.linalg.inv(sqrt_mat))
        setattr(obj, prec_key + '_det', 1 / np.linalg.det(value))


def bfgs_update(prev_sqrt: np.ndarray,
                prev_inv_sqrt: np.ndarray,
                val_diff: np.ndarray,
                grad_diff: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    sty = val_diff.T @ grad_diff

    prev_hess_val_diff = prev_sqrt @ prev_sqrt.T @ val_diff
    hess_val_ip = val_diff.T @ prev_hess_val_diff

    p = val_diff / sty
    # q = np.sqrt(sty / (prev_hess_val_diff.T @ grad_diff)) * prev_hess_val_diff - grad_diff
    q = np.sqrt(sty / hess_val_ip) * prev_hess_val_diff + grad_diff

    t = val_diff / hess_val_ip
    u = np.sqrt(hess_val_ip / sty) * grad_diff + prev_hess_val_diff

    dim = val_diff.size
    new_inv_sqrt = (np.eye(dim) - np.outer(p, q)) @ prev_inv_sqrt
    new_sqrt = (np.eye(dim) - np.outer(u, t)) @ prev_sqrt

    return new_sqrt, new_inv_sqrt


def _cond_bfgs_update(prev_sqrt: np.ndarray,
                      prev_inv_sqrt: np.ndarray,
                      val_diff: np.ndarray,
                      grad_diff: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    sty = val_diff.T @ grad_diff
    return cond(np.logical_or(sty <= 0, np.any(np.isnan(val_diff))),
                lambda _: (prev_sqrt, prev_inv_sqrt),
                lambda tup: bfgs_update(*tup),
                (prev_sqrt, prev_inv_sqrt, val_diff, grad_diff))


@jit
def bfgs(initial_sqrt: np.ndarray,
         initial_inv_sqrt: np.ndarray,
         vals: np.ndarray,
         grads: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    final_carry, _ = scan(
        lambda prev_sqrts, i: (_cond_bfgs_update(*prev_sqrts, vals[i] - vals[i - 1], grads[i] - grads[i - 1]),
                               None),
        (initial_sqrt, initial_inv_sqrt),
        np.arange(1, len(vals)))
    return final_carry


@jit
def l2_distance_matrix(vals: np.ndarray) -> np.ndarray:
    return vmap(lambda x: vmap(lambda y: np.sum(np.square(x - y)))(vals))(vals) ** 0.5


def _gc_close(z_over_rad):
    return -0.25 * z_over_rad ** 5 \
           + 0.5 * z_over_rad ** 4 \
           + 5 / 8 * z_over_rad ** 3 \
           - 5 / 3 * z_over_rad ** 2 \
           + 1


def _gc_mid(z_over_rad):
    return 1 / 12 * z_over_rad ** 5 \
           - 0.5 * z_over_rad ** 4 \
           + 5 / 8 * z_over_rad ** 3 \
           + 5 / 3 * z_over_rad ** 2 \
           - 5 * z_over_rad \
           + 4 \
           - 2 / 3 / z_over_rad


def _gc(z, radius):
    z_over_rad = np.abs(z) / radius
    out = np.zeros_like(z)
    out = np.where(z < radius, _gc_close(z_over_rad), out)
    out = np.where((z >= radius) * (z < 2 * radius), _gc_mid(z_over_rad), out)
    return out


def gc_matrix(dim: int,
              radius: float):
    return vmap(lambda i: _gc(np.min(np.array([np.abs(np.arange(dim) - i),
                                               np.abs(np.arange(dim) + dim - i),
                                               np.abs(np.arange(dim) - dim - i)]), axis=0), radius)
                )(np.arange(dim))