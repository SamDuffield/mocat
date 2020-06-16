########################################################################################################################
# Module: utils.py
# Description: Some useful functions including Gaussian potential evaluations and leapfrog integrator.
#
# Web: https://github.com/SamDuffield/mocat
########################################################################################################################

from typing import Any, Union, Callable, Tuple
from functools import partial

from decorator import decorator
import jax.numpy as np
from jax.lax import scan
from jax import jit
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from mocat.src.core import CDict


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
def _mv_gaussian_potential_identcov(x: np.ndarray,
                                    mean: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
    diff = x - mean
    return 0.5 * np.sum(np.square(diff), axis=-1)


def gaussian_potential(x: np.ndarray,
                       mean: Union[np.ndarray, float] = 0.,
                       prec: np.ndarray = None,
                       sqrt_prec: np.ndarray = None) -> np.ndarray:
    if x.ndim == 1:
        if prec is None:
            return _mv_gaussian_potential_identcov(x, mean)
        else:
            return _mv_gaussian_potential(x, mean, prec)
    else:
        if prec is not None and sqrt_prec is None:
            sqrt_prec = np.linalg.cholesky(prec)

        if sqrt_prec is None:
            return _vectorised_gaussian_potential_identcov(x, mean)
        else:
            return _vectorised_gaussian_potential(x, mean, sqrt_prec)


@partial(jit, static_argnums=(1, 3))
def leapfrog(previous_state: CDict,
             grad_potential: Callable,
             stepsize: float,
             n_steps: int) -> CDict:
    def leapfrog_step(init_state: CDict,
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

