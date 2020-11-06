########################################################################################################################
# Module: kernels.py
# Description: Kernel class definition plus some standard kernels.
#
# Web: https://github.com/SamDuffield/mocat
########################################################################################################################

from typing import Union
from functools import partial

import jax.numpy as np
from jax import jit

from mocat.src.core import CDict


class Kernel:

    def __init__(self, **kwargs):
        if not hasattr(self, 'parameters'):
            self.parameters = CDict()
        self.parameters.__dict__.update(kwargs)

    def __repr__(self):
        return f"mocat.kernels.{self.__class__.__name__}"

    def _call(self,
              x: np.ndarray,
              y: np.ndarray,
              **kernel_params) -> Union[float, np.ndarray]:
        raise NotImplementedError(f'kernel.{self.__class__.__name__} call not implemented')

    def _grad_x(self,
                x: np.ndarray,
                y: np.ndarray,
                **kernel_params) -> Union[float, np.ndarray]:
        raise NotImplementedError(f'kernel.{self.__class__.__name__} grad_x not implemented')

    def _grad_y(self,
                x: np.ndarray,
                y: np.ndarray,
                **kernel_params) -> Union[float, np.ndarray]:
        raise NotImplementedError(f'kernel.{self.__class__.__name__} grad_x not implemented')

    def _diag_grad_xy(self,
                      x: np.ndarray,
                      y: np.ndarray,
                      **kernel_params) -> Union[float, np.ndarray]:
        raise NotImplementedError(f'kernel.{self.__class__.__name__} diag_grad_xy not implemented')

    def _flex_params(self, input_params: dict) -> dict:
        kern_params = self.parameters.__dict__.copy()
        kern_params.update(input_params)
        return kern_params

    def __call__(self,
                 x: np.ndarray,
                 y: np.ndarray,
                 **kwargs) -> Union[float, np.ndarray]:
        return self._call(x, y, **self._flex_params(kwargs))

    def grad_x(self,
               x: np.ndarray,
               y: np.ndarray,
               **kwargs) -> Union[float, np.ndarray]:
        return self._grad_x(x, y, **self._flex_params(kwargs))

    def grad_y(self,
               x: np.ndarray,
               y: np.ndarray,
               **kwargs) -> Union[float, np.ndarray]:
        return self._grad_y(x, y, **self._flex_params(kwargs))

    def diag_grad_xy(self,
                     x: np.ndarray,
                     y: np.ndarray,
                     **kwargs) -> Union[float, np.ndarray]:
        return self._diag_grad_xy(x, y, **self._flex_params(kwargs))


class Gaussian(Kernel):

    def __init__(self,
                 bandwidth: Union[float, np.ndarray] = 1.):
        super().__init__()
        self.parameters.bandwidth = bandwidth

    @partial(jit, static_argnums=(0,))
    def _call(self,
              x: np.ndarray,
              y: np.ndarray,
              bandwidth: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        diff = (x - y) / bandwidth
        return np.exp(-0.5 * np.sum(np.square(diff), axis=-1))

    @partial(jit, static_argnums=(0,))
    def _grad_x(self,
                x: np.ndarray,
                y: np.ndarray,
                bandwidth: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return (y - x) * self._call(x, y, bandwidth=bandwidth) / bandwidth ** 2

    @partial(jit, static_argnums=(0,))
    def _grad_y(self,
                x: np.ndarray,
                y: np.ndarray,
                bandwidth: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return (x - y) * self._call(x, y, bandwidth=bandwidth) / bandwidth ** 2

    @partial(jit, static_argnums=(0,))
    def _diag_grad_xy(self,
                      x: np.ndarray,
                      y: np.ndarray,
                      bandwidth: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return (bandwidth ** 2 - (x - y) ** 2) * self._call(x, y, bandwidth=bandwidth) / bandwidth ** 4


class PreconditionedGaussian(Kernel):

    def __init__(self,
                 bandwidth: float = 1.,
                 precision: np.ndarray = None):
        super().__init__()
        self.parameters.bandwidth = bandwidth
        self.parameters.precision = precision

    @partial(jit, static_argnums=(0,))
    def _call(self,
              x: np.ndarray,
              y: np.ndarray,
              bandwidth: float,
              precision: np.ndarray) -> Union[float, np.ndarray]:
        diff = (x - y) / bandwidth
        return np.exp(-0.5 * np.sum(diff.T @ precision @ diff))

    @partial(jit, static_argnums=(0,))
    def _grad_x(self,
                x: np.ndarray,
                y: np.ndarray,
                bandwidth: float,
                precision: np.ndarray) -> Union[float, np.ndarray]:
        return precision @ (y - x) / bandwidth ** 2 * self._call(x, y, bandwidth=bandwidth, precision=precision)

    @partial(jit, static_argnums=(0,))
    def _grad_y(self,
                x: np.ndarray,
                y: np.ndarray,
                bandwidth: float,
                precision: np.ndarray) -> Union[float, np.ndarray]:
        return precision @ (x - y) / bandwidth ** 2 * self._call(x, y, bandwidth=bandwidth, precision=precision)

    @partial(jit, static_argnums=(0,))
    def _diag_grad_xy(self,
                      x: np.ndarray,
                      y: np.ndarray,
                      bandwidth: float,
                      precision: np.ndarray) -> Union[float, np.ndarray]:
        return precision / bandwidth ** 2 @ (1 - precision * ((x - y) / bandwidth) ** 2) \
               * self._call(x, y,
                            bandwidth=bandwidth,
                            precision=precision)


class IMQ(Kernel):

    def __init__(self,
                 c: Union[float, np.ndarray] = 1.,
                 beta: Union[float, np.ndarray] = -0.5,
                 bandwidth: Union[float, np.ndarray] = 1.):
        super().__init__()
        self.parameters.c = c
        self.parameters.beta = beta
        self.parameters.bandwidth = bandwidth

    @partial(jit, static_argnums=(0,))
    def _call(self,
              x: np.ndarray,
              y: np.ndarray,
              c: Union[float, np.ndarray],
              beta: Union[float, np.ndarray],
              bandwidth: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        diff = (x - y) / bandwidth
        return np.power(c + 0.5 * np.sum(np.square(diff), axis=-1), beta)

    @partial(jit, static_argnums=(0,))
    def _grad_x(self,
                x: np.ndarray,
                y: np.ndarray,
                c: Union[float, np.ndarray],
                beta: Union[float, np.ndarray],
                bandwidth: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        diff = (x - y) / bandwidth
        return diff * beta * np.power(c + 0.5 * np.sum(np.square(diff), axis=-1), beta - 1)

    @partial(jit, static_argnums=(0,))
    def _grad_y(self,
                x: np.ndarray,
                y: np.ndarray,
                c: Union[float, np.ndarray],
                beta: Union[float, np.ndarray],
                bandwidth: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        diff = (x - y) / bandwidth
        return - diff / bandwidth * beta \
               * np.power(c + 0.5 * np.sum(np.square(diff), axis=-1), beta - 1)

    @partial(jit, static_argnums=(0,))
    def _diag_grad_xy(self,
                      x: np.ndarray,
                      y: np.ndarray,
                      c: Union[float, np.ndarray],
                      beta: Union[float, np.ndarray],
                      bandwidth: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        diff = (x - y) / bandwidth
        base = c + 0.5 * np.sum(np.square(diff), axis=-1)
        return (- beta * np.power(base, beta - 1) + beta * (beta - 1) * diff ** 2 * np.power(base, beta - 2)) \
               / bandwidth ** 2
