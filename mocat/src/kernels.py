########################################################################################################################
# Module: kernels.py
# Description: Kernel class definition plus some standard kernels.
#
# Web: https://github.com/SamDuffield/mocat
########################################################################################################################

import copy
from typing import Union
from functools import partial

import jax.numpy as np
from jax import jit

from mocat.src.core import CDict


class Kernel:

    def __init__(self, **kwargs):
        self.parameters = CDict()
        self.parameters.__dict__.update(kwargs)

    def __repr__(self):
        return f"mocat.kernels.{self.__class__.__name__}({self.__dict__.__repr__()})"

    def copy(self):
        return copy.deepcopy(self)

    def __call__(self,
                 x: np.ndarray,
                 y: np.ndarray) -> Union[float, np.ndarray]:
        raise NotImplementedError(f'kernel.{self.__class__.__name__} call not implemented')

    def grad_x(self,
               x: np.ndarray,
               y: np.ndarray) -> Union[float, np.ndarray]:
        raise NotImplementedError(f'kernel.{self.__class__.__name__} grad_x not implemented')

    def grad_y(self,
               x: np.ndarray,
               y: np.ndarray) -> Union[float, np.ndarray]:
        raise NotImplementedError(f'kernel.{self.__class__.__name__} grad_x not implemented')

    def diag_grad_xy(self,
                     x: np.ndarray,
                     y: np.ndarray) -> Union[float, np.ndarray]:
        raise NotImplementedError(f'kernel.{self.__class__.__name__} diag_grad_xy not implemented')


class Gaussian(Kernel):

    def __init__(self,
                 bandwidth: Union[float, np.ndarray] = 1.):
        super().__init__()
        self.parameters.bandwidth = bandwidth

    @partial(jit, static_argnums=(0,))
    def __call__(self,
                 x: np.ndarray,
                 y: np.ndarray) -> Union[float, np.ndarray]:
        diff = (x - y) / self.parameters.bandwidth
        return np.exp(-0.5 * np.sum(np.square(diff), axis=-1))

    @partial(jit, static_argnums=(0,))
    def grad_x(self,
               x: np.ndarray,
               y: np.ndarray) -> Union[float, np.ndarray]:
        return (y - x) * self(x, y) / self.parameters.bandwidth ** 2

    @partial(jit, static_argnums=(0,))
    def grad_y(self,
               x: np.ndarray,
               y: np.ndarray) -> Union[float, np.ndarray]:
        return (x - y) * self(x, y) / self.parameters.bandwidth ** 2

    @partial(jit, static_argnums=(0,))
    def diag_grad_xy(self,
                     x: np.ndarray,
                     y: np.ndarray) -> Union[float, np.ndarray]:
        return (self.parameters.bandwidth ** 2 - (x - y) ** 2) * self(x, y) / self.parameters.bandwidth ** 4


class IMQ(Kernel):

    def __init__(self,
                 c: float = 1.,
                 beta: float = -0.5,
                 bandwidth: Union[float, np.ndarray] = 1.):
        super().__init__()
        self.parameters.c = c
        self.parameters.beta = beta
        self.parameters.bandwidth = bandwidth

    @partial(jit, static_argnums=(0,))
    def __call__(self,
                 x: np.ndarray,
                 y: np.ndarray) -> Union[float, np.ndarray]:
        diff = (x - y) / self.parameters.bandwidth
        return np.power(self.parameters.c + 0.5 * np.sum(np.square(diff), axis=-1), self.parameters.beta)

    @partial(jit, static_argnums=(0,))
    def grad_x(self,
               x: np.ndarray,
               y: np.ndarray) -> Union[float, np.ndarray]:
        diff = (x - y) / self.parameters.bandwidth
        return diff * self.parameters.beta * np.power(self.parameters.c + 0.5 * np.sum(np.square(diff), axis=-1),
                                                      self.parameters.beta - 1)

    @partial(jit, static_argnums=(0,))
    def grad_y(self,
               x: np.ndarray,
               y: np.ndarray) -> Union[float, np.ndarray]:
        diff = (x - y) / self.parameters.bandwidth
        return - diff / self.parameters.bandwidth * self.parameters.beta\
               * np.power(self.parameters.c + 0.5 * np.sum(np.square(diff), axis=-1),
                          self.parameters.beta - 1)

    @partial(jit, static_argnums=(0,))
    def diag_grad_xy(self,
                     x: np.ndarray,
                     y: np.ndarray) -> Union[float, np.ndarray]:
        diff = (x - y) / self.parameters.bandwidth
        base = self.parameters.c + 0.5 * np.sum(np.square(diff), axis=-1)
        return (- self.parameters.beta * np.power(base, self.parameters.beta - 1)
                + self.parameters.beta * (self.parameters.beta - 1) * diff ** 2
                * np.power(base, self.parameters.beta - 2)) / self.parameters.bandwidth ** 2

