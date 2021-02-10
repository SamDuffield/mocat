########################################################################################################################
# Module: metrics.py
# Description: Some common metrics to assess run quality.
#
# Web: https://github.com/SamDuffield/mocat
########################################################################################################################

from typing import Union, Callable, Tuple
from warnings import warn

import jax.numpy as np
from decorator import decorator
from jax import vmap
from jax.scipy.special import logsumexp
import \
    numpy as onp  # For onp.fft.fft(vals, n_samps=n_samps) and onp.unique(vals, axis=0) (JAX doesn't support as of writing)
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from mocat.src.core import cdict
from mocat.src.kernels import Kernel


def extract_1d_vals(sample: Union[np.ndarray, cdict]) -> np.ndarray:
    vals = sample.potential if isinstance(sample, cdict) else sample
    if vals.ndim > 1:
        raise TypeError('vals must be 1 dimensional array or cdict with 1 dimensional potential')
    return vals


def _next_pow_two(n: int) -> int:
    i = 1
    while i < n:
        i = i << 1
    return i


def autocorrelation(sample: Union[np.ndarray, cdict],
                    max_lag_eval: int = 1000) -> np.ndarray:
    vals = extract_1d_vals(sample)
    max_lag_eval = min(max_lag_eval, len(vals))
    n = _next_pow_two(max_lag_eval)
    f = onp.fft.fft(vals - np.mean(vals), n=2 * n)
    acf = np.fft.ifft(f * np.conjugate(f))[:max_lag_eval].real
    return acf / acf[0]


def integrated_autocorrelation_time(sample: Union[np.ndarray, cdict],
                                    max_lag_iat: int = None,
                                    max_lag_eval: int = 1000) -> float:
    autocorr = autocorrelation(sample, max_lag_eval)
    iats_all = 1 + 2 * np.cumsum(autocorr[1:])

    # https://dfm.io/posts/autocorr/
    if max_lag_iat is None:
        c = 5
        m_minus_c_tau = np.arange(1, len(iats_all) + 1) - c * iats_all
        max_lag_iat = np.argmax(m_minus_c_tau > 0)

    return iats_all[max_lag_iat]


def ess_autocorrelation(sample: Union[np.ndarray, cdict],
                        max_lag_iat: int = None,
                        max_lag_eval: int = 1000) -> float:
    vals = extract_1d_vals(sample)
    iat = integrated_autocorrelation_time(vals, max_lag_iat, max_lag_eval)
    return len(vals) / iat


def log_ess_log_weights(log_weights: Union[np.ndarray, cdict]) -> float:
    if isinstance(log_weights, cdict) and hasattr(log_weights, 'log_weight'):
        log_weights = log_weights.log_weight
    else:
        raise TypeError('log_weights must be np.ndarray or cdict with log_weight attribute')
    log_ess = 2 * logsumexp(log_weights) - logsumexp(2 * log_weights)
    return np.exp(log_ess)


def ess_log_weights(log_weights: Union[np.ndarray, cdict]) -> float:
    return np.exp(log_ess_log_weights(log_weights))


def squared_jumping_distance(sample: Union[np.ndarray, cdict]) -> float:
    vals = sample.value if isinstance(sample, cdict) else sample
    sjd = vals[1:] - vals[:-1]
    sjd = sjd ** 2
    return sjd.mean(0)


def ksd(sample: Union[np.ndarray, cdict],
        kernel: Kernel,
        grad_potential: np.ndarray = None,
        log_weights: np.ndarray = None,
        **kernel_params) -> float:
    vals = sample.value if isinstance(sample, cdict) else sample

    if grad_potential is None and isinstance(sample, cdict) and hasattr(sample, 'grad_potential'):
        grad_potential = sample.grad_potential
    else:
        raise TypeError('grad_potential not found')

    n = len(vals)

    if log_weights is not None:
        weights = np.exp(log_weights)
        weights /= weights.sum()
    else:
        weights = np.ones(n) / n

    def k_0_inds(x_i, y_i):
        kern_diag_grad_xy_mat = np.sum(kernel.diag_grad_xy(vals[x_i], vals[y_i], **kernel_params))
        grad_kx_grad_py_mat = np.dot(kernel.grad_x(vals[x_i], vals[y_i], **kernel_params), grad_potential[y_i])
        grad_ky_grad_px_mat = np.dot(grad_potential[x_i], kernel.grad_y(vals[x_i], vals[y_i], **kernel_params))
        kern_grad_pxy = kernel(vals[x_i], vals[y_i], **kernel_params) * np.dot(grad_potential[x_i],
                                                                               grad_potential[y_i])

        return (kern_diag_grad_xy_mat + grad_kx_grad_py_mat + grad_ky_grad_px_mat + kern_grad_pxy) \
               * weights[x_i] * weights[y_i]

    return np.sqrt((vmap(lambda i: vmap(lambda j: k_0_inds(i, j))(np.arange(n)))(np.arange(n))).sum())


@decorator
def metric_plot(plot_func: Callable,
                *args,
                **kwargs) -> Union[plt.Axes, Tuple[Figure, plt.Axes]]:
    ax = args[-1]
    if ax is None:
        fig, ax = plt.subplots()
        plot_func(*args[:-1], ax, **kwargs)
        return fig, ax
    else:
        plot_func(*args, **kwargs)
        return ax


@metric_plot
def autocorrelation_plot(sample: Union[np.ndarray, cdict],
                         max_lag_plot: int = 100,
                         max_lag_eval: int = 1000,
                         ax: plt.Axes = None,
                         **kwargs):
    vals = extract_1d_vals(sample)

    autocor = autocorrelation(vals, max_lag_eval)
    ax.bar(range(1, 1 + len(autocor)), autocor, **kwargs)
    ax.set_xlim(0, max_lag_plot)


@metric_plot
def trace_plot(sample: Union[np.ndarray, cdict],
               last_n: int = None,
               ax: plt.Axes = None,
               **kwargs):
    vals = extract_1d_vals(sample)

    if last_n is None:
        start = 0
    else:
        start = vals.shape[0] - last_n
    ax.plot(vals[start:], **kwargs)


@metric_plot
def plot_2d_samples(sample: Union[np.ndarray, cdict],
                    dim1: int = 0,
                    dim2: int = 1,
                    s: float = 0.5,
                    ax: plt.Axes = None,
                    **kwargs):
    vals = sample.value if isinstance(sample, cdict) else sample

    if vals.ndim == 3:
        vals = np.concatenate(vals)
        warn('Concatenating 3 dimensional array to plot')
    if vals.ndim == 1 or vals.shape[-1] == 1:
        raise TypeError('Samples are one dimensional - try mocat.hist_1d_samples')
    ax.scatter(vals[:, dim1], vals[:, dim2], s, **kwargs)


@metric_plot
def hist_1d_samples(sample: Union[np.ndarray, cdict],
                    dim: int = 0,
                    bins: int = 50,
                    density: bool = True,
                    ax: plt.Axes = None,
                    **kwargs):
    vals = sample.value if isinstance(sample, cdict) else sample

    if vals.ndim == 1:
        vals = vals[..., np.newaxis]
    if vals.ndim == 3:
        vals = np.concatenate(vals)

    ax.hist(vals[:, dim], bins=bins, density=density, **kwargs)
