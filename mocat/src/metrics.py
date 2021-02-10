########################################################################################################################
# Module: metrics.py
# Description: Some common metrics to assess run quality.
#
# Web: https://github.com/SamDuffield/mocat
########################################################################################################################

from typing import Union, Callable, Tuple
from warnings import warn
from functools import partial

from decorator import decorator
from jax import vmap, random, numpy as jnp
from jax.scipy.special import logsumexp
import \
    numpy as ojnp  # For ojnp.fft.fft(vals, n_samps=n_samps) and ojnp.unique(vals, axis=0) (JAX doesn't support as of writing)
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from mocat.src.core import cdict
from mocat.src.kernels import Kernel


def extract_1d_vals(sample: Union[jnp.ndarray, cdict]) -> jnp.ndarray:
    vals = sample.potential if isinstance(sample, cdict) else sample
    if vals.ndim > 1:
        raise TypeError('vals must be 1 dimensional array or cdict with 1 dimensional potential')
    return vals


def _next_pow_two(n: int) -> int:
    i = 1
    while i < n:
        i = i << 1
    return i


def autocorrelation(sample: Union[jnp.ndarray, cdict],
                    max_lag_eval: int = 1000) -> jnp.ndarray:
    vals = extract_1d_vals(sample)
    max_lag_eval = min(max_lag_eval, len(vals))
    n = _next_pow_two(max_lag_eval)
    f = ojnp.fft.fft(vals - jnp.mean(vals), n=2 * n)
    acf = jnp.fft.ifft(f * jnp.conjugate(f))[:max_lag_eval].real
    return acf / acf[0]


def integrated_autocorrelation_time(sample: Union[jnp.ndarray, cdict],
                                    max_lag_iat: int = None,
                                    max_lag_eval: int = 1000) -> float:
    autocorr = autocorrelation(sample, max_lag_eval)
    iats_all = 1 + 2 * jnp.cumsum(autocorr[1:])

    # https://dfm.io/posts/autocorr/
    if max_lag_iat is None:
        c = 5
        m_minus_c_tau = jnp.arange(1, len(iats_all) + 1) - c * iats_all
        max_lag_iat = jnp.argmax(m_minus_c_tau > 0)

    return iats_all[max_lag_iat]


def ess_autocorrelation(sample: Union[jnp.ndarray, cdict],
                        max_lag_iat: int = None,
                        max_lag_eval: int = 1000) -> float:
    vals = extract_1d_vals(sample)
    iat = integrated_autocorrelation_time(vals, max_lag_iat, max_lag_eval)
    return len(vals) / iat


def log_ess_log_weight(log_weight: Union[jnp.ndarray, cdict]) -> float:
    if isinstance(log_weight, cdict) and hasattr(log_weight, 'log_weight'):
        log_weight = log_weight.log_weight
    if not isinstance(log_weight, jnp.ndarray):
        raise TypeError('log_weight must be jnp.ndarray or cdict with log_weight attribute')
    return 2 * logsumexp(log_weight) - logsumexp(2 * log_weight)


def ess_log_weight(log_weight: Union[jnp.ndarray, cdict]) -> float:
    return jnp.exp(log_ess_log_weight(log_weight))


def squared_jumping_distance(sample: Union[jnp.ndarray, cdict]) -> float:
    vals = sample.value if isinstance(sample, cdict) else sample
    sjd = vals[1:] - vals[:-1]
    sjd = sjd ** 2
    return sjd.mean(0)


def ksd(sample: Union[jnp.ndarray, cdict],
        kernel: Kernel,
        grad_potential: jnp.ndarray = None,
        log_weight: jnp.ndarray = None,
        batchsize: int = None,
        random_key: jnp.ndarray = None,
        **kernel_params) -> float:
    vals = sample.value if isinstance(sample, cdict) else sample
    n = len(vals)

    if grad_potential is None and isinstance(sample, cdict) and hasattr(sample, 'grad_potential'):
        grad_potential = sample.grad_potential
    else:
        raise TypeError('grad_potential not found')

    if batchsize is None:
        get_batch_inds = lambda _: jnp.arange(n)
    else:
        random_inds = random.choice(random_key, n, shape=(n, batchsize))
        get_batch_inds = lambda i: random_inds[i]

    if log_weight is not None:
        weights = jnp.exp(log_weight)
    else:
        weights = jnp.ones(n)

    sum_weights = weights.sum()

    def k_0_inds(x_i: int, y_i: int) -> float:
        kern_diag_grad_xy_mat = jnp.sum(kernel.diag_grad_xy(vals[x_i], vals[y_i], **kernel_params))
        grad_kx_grad_py_mat = jnp.dot(kernel.grad_x(vals[x_i], vals[y_i], **kernel_params), grad_potential[y_i])
        grad_ky_grad_px_mat = jnp.dot(grad_potential[x_i], kernel.grad_y(vals[x_i], vals[y_i], **kernel_params))
        kern_grad_pxy = kernel(vals[x_i], vals[y_i], **kernel_params) * jnp.dot(grad_potential[x_i],
                                                                                grad_potential[y_i])

        return (kern_diag_grad_xy_mat + grad_kx_grad_py_mat + grad_ky_grad_px_mat + kern_grad_pxy) \
               * weights[x_i] * weights[y_i]

    def v_k_0(i: int) -> jnp.ndarray:
        batch_inds = get_batch_inds(i)
        return vmap(partial(k_0_inds, i))(batch_inds) / (sum_weights * weights[batch_inds].sum())

    return jnp.sqrt((vmap(v_k_0)(jnp.arange(n))).sum())


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
def autocorrelation_plot(sample: Union[jnp.ndarray, cdict],
                         max_lag_plot: int = 100,
                         max_lag_eval: int = 1000,
                         ax: plt.Axes = None,
                         **kwargs):
    vals = extract_1d_vals(sample)

    autocor = autocorrelation(vals, max_lag_eval)
    ax.bar(range(1, 1 + len(autocor)), autocor, **kwargs)
    ax.set_xlim(0, max_lag_plot)


@metric_plot
def trace_plot(sample: Union[jnp.ndarray, cdict],
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
def plot_2d_samples(sample: Union[jnp.ndarray, cdict],
                    dim1: int = 0,
                    dim2: int = 1,
                    s: float = 0.5,
                    ax: plt.Axes = None,
                    **kwargs):
    vals = sample.value if isinstance(sample, cdict) else sample

    if vals.ndim == 3:
        vals = jnp.concatenate(vals)
        warn('Concatenating 3 dimensional array to plot')
    if vals.ndim == 1 or vals.shape[-1] == 1:
        raise TypeError('Samples are one dimensional - try mocat.hist_1d_samples')
    ax.scatter(vals[:, dim1], vals[:, dim2], s, **kwargs)


@metric_plot
def hist_1d_samples(sample: Union[jnp.ndarray, cdict],
                    dim: int = 0,
                    bins: int = 50,
                    density: bool = True,
                    ax: plt.Axes = None,
                    **kwargs):
    vals = sample.value if isinstance(sample, cdict) else sample

    if vals.ndim == 1:
        vals = vals[..., jnp.newaxis]
    if vals.ndim == 3:
        vals = jnp.concatenate(vals)

    ax.hist(vals[:, dim], bins=bins, density=density, **kwargs)
