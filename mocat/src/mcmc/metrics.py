########################################################################################################################
# Module: metrics.py
# Description: Some common metrics to assess sample quality.
#
# Web: https://github.com/SamDuffield/mocat
########################################################################################################################

from typing import Union, Tuple

import jax.numpy as np
from jax import vmap
import numpy as onp         # For onp.fft.fft(x, n=n) and onp.unique(x, axis=0) (JAX doesn't support as of writing)
import matplotlib.pyplot as plt

from mocat.src.core import CDict
from mocat.src.utils import _metric_plot_decorate
from mocat.src.kernels import Kernel


def _is_cdict(input: Union[np.ndarray, CDict],
              value_out: bool = False) -> Union[bool, Tuple[bool, np.ndarray]]:
    if isinstance(input, CDict):
        return (True, input.value) if value_out else True
    elif isinstance(input, np.ndarray):
        return (False, input) if value_out else False
    else:
        raise ValueError("Input isn't ndarray or CDict")


def _acceptance_rate_values(array: np.ndarray) -> float:
    if array.ndim == 1:
        return (len(np.unique(array)) - 1) / (len(array) - 1)
    elif array.ndim == 2:
        return _acceptance_rate_values(array[:, 0])
    elif array.ndim == 3:
        return (len(onp.unique(array[:, :, 0], axis=0)) - 1) / (array.shape[0] - 1)
    else:
        raise TypeError("Can't compute acceptance rate of object")


def acceptance_rate(sample: Union[np.ndarray, CDict],
                    alpha: bool = True) -> float:

    _, vals = _is_cdict(sample, True)
    if hasattr(sample, 'alpha') and alpha:
        return float(sample.alpha.mean())
    else:
        return _acceptance_rate_values(vals)


def _next_pow_two(n: int) -> int:
    i = 1
    while i < n:
        i = i << 1
    return i


def _autocorrelation_1darray(x: Union[np.ndarray, CDict],
                             max_lag_eval: int) -> np.ndarray:
    if x.ndim != 1:
        raise ValueError("_autocorrelation_1darray must be applied to a 1 dimensional np.ndarray")
    max_lag_eval = min(max_lag_eval, len(x))
    n = _next_pow_two(max_lag_eval)
    f = onp.fft.fft(x - np.mean(x), n=2 * n)
    acf = np.fft.ifft(f * np.conjugate(f))[:max_lag_eval].real
    return acf / acf[0]


def autocorrelation(sample: Union[CDict, np.ndarray],
                    dim: int = None,
                    max_lag_eval: int = 1000,
                    ensemble_index: int = None) -> np.ndarray:

    _, vals = _is_cdict(sample, value_out=True)

    if vals.ndim == 3:
        if isinstance(sample, CDict) and hasattr(sample, 'potential'):
            if ensemble_index is None:
                if dim is None:
                    return _autocorrelation_1darray(np.mean(sample.potential, axis=1), max_lag_eval)
                else:
                    return _autocorrelation_1darray(np.mean(vals[:, :, dim], axis=1), max_lag_eval)
            else:
                return autocorrelation(vals[:, ensemble_index, :], dim, max_lag_eval)

        elif isinstance(vals, np.ndarray):
            if ensemble_index is None:
                if dim is None:
                    return _autocorrelation_1darray(np.mean(sample[:, :, 0], axis=1), max_lag_eval)
                else:
                    return _autocorrelation_1darray(np.mean(sample[:, :, dim], axis=1), max_lag_eval)
            else:
                return autocorrelation(sample[:, ensemble_index, :], dim, max_lag_eval)

    elif vals.ndim == 2:
        if isinstance(sample, CDict) and hasattr(sample, 'potential'):
            if dim is None:
                return _autocorrelation_1darray(sample.potential, max_lag_eval)
            else:
                return _autocorrelation_1darray(vals[:, dim], max_lag_eval)

        elif isinstance(vals, np.ndarray):
            if dim is None:
                return _autocorrelation_1darray(vals[:, 0], max_lag_eval)
            else:
                return _autocorrelation_1darray(vals[:, dim], max_lag_eval)

    elif vals.ndim == 1:
        if isinstance(sample, CDict) and hasattr(sample, 'potential'):
            if dim is None:
                return _autocorrelation_1darray(sample.potential, max_lag_eval)
            else:
                return _autocorrelation_1darray(vals, max_lag_eval)

        elif isinstance(vals, np.ndarray):
            return _autocorrelation_1darray(vals, max_lag_eval)

    else:
        raise ValueError("Input to autocorrelation must be numpy.ndarray or Sample")


def integrated_autocorrelation_time(sample: Union[np.ndarray, CDict],
                                    dim: int = None,
                                    max_lag_iat: int = None,
                                    max_lag_eval: int = 1000,
                                    ensemble_index: int = None) -> float:

    autocorr = autocorrelation(sample, dim, max_lag_eval, ensemble_index)
    iats_all = 1 + 2 * np.cumsum(autocorr[1:])

    # https://dfm.io/posts/autocorr/
    if max_lag_iat is None:
        c = 5
        m_minus_c_tau = np.arange(1, len(iats_all) + 1) - c * iats_all
        max_lag_iat = np.argmax(m_minus_c_tau > 0)

    return float(iats_all[max_lag_iat])


def ess(sample: Union[np.ndarray, CDict],
        dim: int = None,
        max_lag_iat: int = None,
        max_lag_eval: int = 1000,
        ensemble_index: int = None) -> float:

    _, vals = _is_cdict(sample, value_out=True)

    # https://dfm.io/posts/autocorr/
    iat = integrated_autocorrelation_time(sample, dim, max_lag_iat, max_lag_eval, ensemble_index)

    n_mult = np.prod(vals.shape[:-1]) if vals.ndim == 3 and ensemble_index is None else len(vals)

    return n_mult / iat


def ess_per_second(sample: CDict,
                   dim: int = None,
                   max_lag_iat: int = None,
                   max_lag_eval: int = 1000) -> float:

    if not hasattr(sample, 'time'):
        raise ValueError("Time not stored in CDict")
    else:
        return ess(sample, dim, max_lag_iat, max_lag_eval) / sample.time


def squared_jumping_distance(sample: Union[np.ndarray, CDict],
                             dim: int = 0) -> float:

    _, vals = _is_cdict(sample, value_out=True)
    sjd = vals[1:] - vals[:-1]
    sjd = sjd ** 2
    return float(sjd[..., dim].mean())


def ksd(sample: CDict,
        kernel: Kernel,
        weighted: bool = True) -> float:

    xs = sample.value
    grad_potentials = sample.grad_potential

    n = len(xs)

    if hasattr(sample, 'weights') is None and weighted:
        weights = sample.weights
    elif hasattr(sample, 'logit_weights') and weighted:
        weights = np.exp(sample.logit_weights)
        weights /= weights.sum()
    else:
        weights = np.ones(n) / n

    def k_0_inds(x_i, y_i):
        kern_diag_grad_xy_mat = np.sum(kernel.diag_grad_xy(xs[x_i], xs[y_i]))
        grad_kx_grad_py_mat = np.dot(kernel.grad_x(xs[x_i], xs[y_i]), grad_potentials[y_i])
        grad_ky_grad_px_mat = np.dot(grad_potentials[x_i], kernel.grad_y(xs[x_i], xs[y_i]))
        kern_grad_pxy = kernel(xs[x_i], xs[y_i]) * np.dot(grad_potentials[x_i], grad_potentials[y_i])

        return (kern_diag_grad_xy_mat + grad_kx_grad_py_mat + grad_ky_grad_px_mat + kern_grad_pxy) \
               * weights[x_i] * weights[y_i]

    return float(np.sqrt((vmap(k_0_inds)(np.arange(n), np.arange(n))).sum()))


@_metric_plot_decorate
def autocorrelation_plot(sample: Union[np.ndarray, CDict],
                         dim: int = None,
                         max_lag_plot: int = 100,
                         max_lag_eval: int = 1000,
                         ax: plt.Axes = None,
                         **kwargs):
    autocor = autocorrelation(sample, dim, max_lag_eval)
    ax.bar(range(1, 1 + len(autocor)), autocor, **kwargs)
    ax.set_xlim(0, max_lag_plot)


@_metric_plot_decorate
def trace_plot(sample: Union[np.ndarray, CDict],
               dim: int = None,
               last_n: int = None,
               ax: plt.Axes = None,
               **kwargs):
    _, vals = _is_cdict(sample, True)

    if last_n is None:
        start = 0
    else:
        start = vals.shape[0] - last_n

    if dim is None and isinstance(sample, CDict) and hasattr(sample, 'potential'):
        ax.plot(sample.potential[start:], **kwargs)
    else:
        if dim is None:
            dim = 0
        ax.plot(vals[start:, ..., dim], **kwargs)


@_metric_plot_decorate
def plot_2d_samples(sample: Union[np.ndarray, CDict],
                    dim1: int = 0,
                    dim2: int = 1,
                    s: float = 0.5,
                    ax: plt.Axes = None,
                    **kwargs):
    _, vals = _is_cdict(sample, True)

    if vals.shape[-1] == 1:
        raise TypeError('Samples are one dimensional - try mocat.hist_1d_samples')
    else:
        if vals.ndim == 3:
            vals = np.concatenate(vals)

        ax.scatter(vals[:, dim1], vals[:, dim2], s, **kwargs)


@_metric_plot_decorate
def hist_1d_samples(sample: Union[np.ndarray, CDict],
                    dim: int = 0,
                    bins: int = 50,
                    density: bool = True,
                    ax: plt.Axes = None,
                    **kwargs):
    _, vals = _is_cdict(sample, True)

    if vals.ndim == 3:
        vals = np.concatenate(vals)

    ax.hist(vals[:, dim], bins=bins, density=density, **kwargs)
