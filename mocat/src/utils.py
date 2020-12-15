########################################################################################################################
# Module: _utils.py
# Description: Some useful functions including TemporalGaussian potential evaluations and leapfrog integrator.
#
# Web: https://github.com/SamDuffield/mocat
########################################################################################################################

from typing import Any, Union, Callable, Tuple
from functools import partial

from decorator import decorator
import jax.numpy as np
from jax.lax import scan, while_loop, cond
from jax import jit, numpy as np
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
                                   mean: Union[np.ndarray, float],
                                   sqrt_prec: np.ndarray) -> Union[np.ndarray, float]:
    return 0.5 * np.sum(np.square((x - mean) @ sqrt_prec), axis=-1)


@jit
def _vectorised_gaussian_potential_diag(x: np.ndarray,
                                        mean: Union[np.ndarray, float],
                                        sqrt_prec_diag: np.ndarray) -> Union[np.ndarray, float]:
    return 0.5 * np.sum(np.square((x - mean) * sqrt_prec_diag), axis=-1)


@jit
def _vectorised_gaussian_potential_identcov(x: np.ndarray,
                                            mean: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
    return 0.5 * np.sum(np.square(x - mean), axis=-1)


@jit
def _mv_gaussian_potential(x: np.ndarray,
                           mean: Union[np.ndarray, float],
                           prec: np.ndarray) -> Union[np.ndarray, float]:
    diff = x - mean
    return 0.5 * diff.T @ prec @ diff


@jit
def _mv_gaussian_potential_diag(x: np.ndarray,
                                mean: Union[np.ndarray, float],
                                prec_diag: np.ndarray) -> Union[np.ndarray, float]:
    diff = x - mean
    return 0.5 * np.sum(np.square(diff) * prec_diag, axis=-1)


@jit
def _mv_gaussian_potential_identcov(x: np.ndarray,
                                    mean: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
    diff = x - mean
    return 0.5 * np.sum(np.square(diff), axis=-1)


def gaussian_potential(x: np.ndarray,
                       mean: Union[np.ndarray, float] = 0.,
                       prec: np.ndarray = None,
                       sqrt_prec: np.ndarray = None) -> np.ndarray:
    if x.ndim == 1 and sqrt_prec is None:
        if prec is None:
            return _mv_gaussian_potential_identcov(x, mean)
        elif prec.ndim == 1:
            return _mv_gaussian_potential_diag(x, mean, prec)
        else:
            return _mv_gaussian_potential(x, mean, prec)
    else:
        if prec is not None and sqrt_prec is None:
            sqrt_prec = np.linalg.cholesky(prec)

        if sqrt_prec is None:
            return _vectorised_gaussian_potential_identcov(x, mean)
        elif sqrt_prec.ndim < 2:
            return _vectorised_gaussian_potential_diag(x, mean, sqrt_prec)
        else:
            return _vectorised_gaussian_potential(x, mean, sqrt_prec)


@partial(jit, static_argnums=(1, 3))
def leapfrog(previous_state: cdict,
             grad_potential: Callable,
             stepsize: float,
             n_steps: int) -> cdict:
    def leapfrog_step(init_state: cdict,
                      _: Any):
        new_state = init_state.copy()

        p_half = init_state.momenta - stepsize / 2. * init_state.grad_potential
        new_state.value = init_state.value + stepsize * p_half

        new_state.grad_potential = grad_potential(new_state.value)

        new_state.momenta = p_half - stepsize / 2. * new_state.grad_potential

        next_sample_chain = new_state.copy()
        next_sample_chain.momenta = np.vstack([p_half, new_state.momenta])
        return new_state, next_sample_chain

    final_leapfrog, all_leapfrog = scan(leapfrog_step, previous_state, None, n_steps)
    all_leapfrog.momenta = np.concatenate(all_leapfrog.momenta)

    if hasattr(previous_state, '_all_leapfrog_value'):
        all_leapfrog._all_leapfrog_value = np.concatenate([previous_state.value[np.newaxis],
                                                           all_leapfrog.value])[np.newaxis]
    if hasattr(previous_state, '_all_leapfrog_momenta'):
        all_leapfrog._all_leapfrog_momenta = np.concatenate([previous_state.momenta[np.newaxis],
                                                             all_leapfrog.momenta])[np.newaxis]

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
    _, final_iter = final_val_and_final_iter
    return stack, final_iter


def while_loop_stacked(cond_fun: Callable,
                       body_fun: Callable,
                       init_carry: Tuple[Any, Any],
                       max_iter: int = 1000) -> Any:
    full_stack, final_int = _while_loop_stacked(cond_fun, body_fun, init_carry, max_iter)
    return tuple(a[:final_int] for a in full_stack) if isinstance(full_stack, tuple) else full_stack[:final_int]


@partial(jit, static_argnums=(0,))
def bisect(fun: Callable,
           bounds: np.ndarray,
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
    if value.ndim < 2:
        sqrt_val = np.sqrt(value)
        setattr(obj, key + '_sqrt', sqrt_val)
        setattr(obj, key.replace('covariance', 'precision') + '_sqrt', 1 / sqrt_val)
    else:
        sqrt_mat = np.linalg.cholesky(value)
        setattr(obj, key + '_sqrt', sqrt_mat)
        setattr(obj, key.replace('covariance', 'precision') + '_sqrt', np.linalg.inv(sqrt_mat))