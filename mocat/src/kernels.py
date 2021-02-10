########################################################################################################################
# Module: kernels.py
# Description: Kernel class definition plus some standard kernels.
#
# Web: https://github.com/SamDuffield/mocat
########################################################################################################################

from typing import Union
from functools import partial

from jax import jit, numpy as jnp

from mocat.src.core import cdict
from mocat.src.utils import l2_distance_matrix


class Kernel:
    parameters: cdict

    def __init__(self, **kwargs):
        if not hasattr(self, 'parameters'):
            self.parameters = cdict()
        self.parameters.__dict__.update(kwargs)

    def __repr__(self):
        return f"mocat.kernels.{self.__class__.__name__}"

    def _call(self,
              x: jnp.ndarray,
              y: jnp.ndarray,
              **kernel_params) -> Union[float, jnp.ndarray]:
        raise NotImplementedError(f'kernel.{self.__class__.__name__} call not implemented')

    def _grad_x(self,
                x: jnp.ndarray,
                y: jnp.ndarray,
                **kernel_params) -> Union[float, jnp.ndarray]:
        raise NotImplementedError(f'kernel.{self.__class__.__name__} grad_x not implemented')

    def _grad_y(self,
                x: jnp.ndarray,
                y: jnp.ndarray,
                **kernel_params) -> Union[float, jnp.ndarray]:
        raise NotImplementedError(f'kernel.{self.__class__.__name__} grad_x not implemented')

    def _diag_grad_xy(self,
                      x: jnp.ndarray,
                      y: jnp.ndarray,
                      **kernel_params) -> Union[float, jnp.ndarray]:
        raise NotImplementedError(f'kernel.{self.__class__.__name__} diag_grad_xy not implemented')

    def _flex_params(self, ijnput_params: dict) -> dict:
        kern_params = self.parameters.__dict__.copy()
        kern_params.update(ijnput_params)
        return kern_params

    def __call__(self,
                 x: jnp.ndarray,
                 y: jnp.ndarray,
                 **kwargs) -> Union[float, jnp.ndarray]:
        return self._call(x, y, **self._flex_params(kwargs))

    def grad_x(self,
               x: jnp.ndarray,
               y: jnp.ndarray,
               **kwargs) -> Union[float, jnp.ndarray]:
        return self._grad_x(x, y, **self._flex_params(kwargs))

    def grad_y(self,
               x: jnp.ndarray,
               y: jnp.ndarray,
               **kwargs) -> Union[float, jnp.ndarray]:
        return self._grad_y(x, y, **self._flex_params(kwargs))

    def diag_grad_xy(self,
                     x: jnp.ndarray,
                     y: jnp.ndarray,
                     **kwargs) -> Union[float, jnp.ndarray]:
        return self._diag_grad_xy(x, y, **self._flex_params(kwargs))


class Gaussian(Kernel):

    def __init__(self,
                 bandwidth: Union[float, jnp.ndarray] = 1.):
        super().__init__()
        self.parameters.bandwidth = bandwidth

    @partial(jit, static_argnums=(0,))
    def _call(self,
              x: jnp.ndarray,
              y: jnp.ndarray,
              bandwidth: Union[float, jnp.ndarray]) -> Union[float, jnp.ndarray]:
        diff = (x - y) / bandwidth
        return jnp.exp(-0.5 * jnp.sum(jnp.square(diff), axis=-1))

    @partial(jit, static_argnums=(0,))
    def _grad_x(self,
                x: jnp.ndarray,
                y: jnp.ndarray,
                bandwidth: Union[float, jnp.ndarray]) -> Union[float, jnp.ndarray]:
        return (y - x) * self._call(x, y, bandwidth=bandwidth) / bandwidth ** 2

    @partial(jit, static_argnums=(0,))
    def _grad_y(self,
                x: jnp.ndarray,
                y: jnp.ndarray,
                bandwidth: Union[float, jnp.ndarray]) -> Union[float, jnp.ndarray]:
        return (x - y) * self._call(x, y, bandwidth=bandwidth) / bandwidth ** 2

    @partial(jit, static_argnums=(0,))
    def _diag_grad_xy(self,
                      x: jnp.ndarray,
                      y: jnp.ndarray,
                      bandwidth: Union[float, jnp.ndarray]) -> Union[float, jnp.ndarray]:
        return (bandwidth ** 2 - (x - y) ** 2) * self._call(x, y, bandwidth=bandwidth) / bandwidth ** 4


class PreconditionedGaussian(Kernel):

    def __init__(self,
                 bandwidth: float = 1.,
                 precision: jnp.ndarray = None):
        super().__init__()
        self.parameters.bandwidth = bandwidth
        self.parameters.precision = precision

    @partial(jit, static_argnums=(0,))
    def _call(self,
              x: jnp.ndarray,
              y: jnp.ndarray,
              bandwidth: float,
              precision: jnp.ndarray) -> Union[float, jnp.ndarray]:
        diff = (x - y) / bandwidth
        return jnp.exp(-0.5 * jnp.sum(diff.T @ precision @ diff))

    @partial(jit, static_argnums=(0,))
    def _grad_x(self,
                x: jnp.ndarray,
                y: jnp.ndarray,
                bandwidth: float,
                precision: jnp.ndarray) -> Union[float, jnp.ndarray]:
        return precision @ (y - x) / bandwidth ** 2 * self._call(x, y, bandwidth=bandwidth, precision=precision)

    @partial(jit, static_argnums=(0,))
    def _grad_y(self,
                x: jnp.ndarray,
                y: jnp.ndarray,
                bandwidth: float,
                precision: jnp.ndarray) -> Union[float, jnp.ndarray]:
        return precision @ (x - y) / bandwidth ** 2 * self._call(x, y, bandwidth=bandwidth, precision=precision)

    @partial(jit, static_argnums=(0,))
    def _diag_grad_xy(self,
                      x: jnp.ndarray,
                      y: jnp.ndarray,
                      bandwidth: float,
                      precision: jnp.ndarray) -> Union[float, jnp.ndarray]:
        return precision / bandwidth ** 2 @ (1 - precision * ((x - y) / bandwidth) ** 2) \
               * self._call(x, y,
                            bandwidth=bandwidth,
                            precision=precision)


class IMQ(Kernel):

    def __init__(self,
                 c: Union[float, jnp.ndarray] = 1.,
                 beta: Union[float, jnp.ndarray] = -0.5,
                 bandwidth: Union[float, jnp.ndarray] = 1.):
        super().__init__()
        self.parameters.c = c
        self.parameters.beta = beta
        self.parameters.bandwidth = bandwidth

    @partial(jit, static_argnums=(0,))
    def _call(self,
              x: jnp.ndarray,
              y: jnp.ndarray,
              c: Union[float, jnp.ndarray],
              beta: Union[float, jnp.ndarray],
              bandwidth: Union[float, jnp.ndarray]) -> Union[float, jnp.ndarray]:
        diff = (x - y) / bandwidth
        return jnp.power(c + 0.5 * jnp.sum(jnp.square(diff), axis=-1), beta)

    @partial(jit, static_argnums=(0,))
    def _grad_x(self,
                x: jnp.ndarray,
                y: jnp.ndarray,
                c: Union[float, jnp.ndarray],
                beta: Union[float, jnp.ndarray],
                bandwidth: Union[float, jnp.ndarray]) -> Union[float, jnp.ndarray]:
        diff = (x - y) / bandwidth
        return diff * beta * jnp.power(c + 0.5 * jnp.sum(jnp.square(diff), axis=-1), beta - 1)

    @partial(jit, static_argnums=(0,))
    def _grad_y(self,
                x: jnp.ndarray,
                y: jnp.ndarray,
                c: Union[float, jnp.ndarray],
                beta: Union[float, jnp.ndarray],
                bandwidth: Union[float, jnp.ndarray]) -> Union[float, jnp.ndarray]:
        diff = (x - y) / bandwidth
        return - diff / bandwidth * beta \
               * jnp.power(c + 0.5 * jnp.sum(jnp.square(diff), axis=-1), beta - 1)

    @partial(jit, static_argnums=(0,))
    def _diag_grad_xy(self,
                      x: jnp.ndarray,
                      y: jnp.ndarray,
                      c: Union[float, jnp.ndarray],
                      beta: Union[float, jnp.ndarray],
                      bandwidth: Union[float, jnp.ndarray]) -> Union[float, jnp.ndarray]:
        diff = (x - y) / bandwidth
        base = c + 0.5 * jnp.sum(jnp.square(diff), axis=-1)
        return (- beta * jnp.power(base, beta - 1) + beta * (beta - 1) * diff ** 2 * jnp.power(base, beta - 2)) \
               / bandwidth ** 2


def median_bandwidth_update(vals: jnp.ndarray) -> float:
    # Note jax.numpy.median scales much worse than numpy.median,
    # thus this method is not currently recommended
    dist_mat = l2_distance_matrix(vals)
    return jnp.median(dist_mat) / jnp.sqrt(2 * jnp.log(dist_mat.shape[0]))


def mean_bandwidth_update(vals: jnp.ndarray) -> float:
    dist_mat = l2_distance_matrix(vals)
    return jnp.mean(dist_mat) / jnp.sqrt(2 * jnp.log(dist_mat.shape[0]))

