########################################################################################################################
# Module: scenarios/toy_scenarios.py
# Description: Some simple multivariate scenarios
#
# Web: https://github.com/SamDuffield/mocat
########################################################################################################################

from typing import Union

import jax.numpy as np
from jax import vmap, jit

from mocat.src.core import Scenario


class Gaussian(Scenario):
    name = "Gaussian"

    def __init__(self,
                 dim: int = 1,
                 mean: Union[float, np.ndarray] = None,
                 cov: Union[float, np.ndarray] = None):
        if mean is not None:
            self.dim = mean.shape[-1]
        elif cov is not None:
            self.dim = cov.shape[-1]
        else:
            self.dim = dim
        self.mean = np.zeros(dim) if mean is None else mean
        self.cov = np.eye(dim) if cov is None else cov
        super().__init__()

    def potential(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        x_diff = (x - self.mean) @ self.sqrt_prec
        return 0.5 * np.sum(np.square(x_diff), axis=-1)

    def __setattr__(self, key, value):
        self.__dict__[key] = value
        if key == 'cov':
            self.prec = np.eye(self.dim) if value is None else np.linalg.inv(value)
            self.sqrt_prec = np.eye(self.dim) if value is None else np.linalg.cholesky(self.prec)


class GaussianMixture(Scenario):
    name = "Gaussian Mixture"

    def __init__(self,
                 means: np.ndarray,
                 covs: np.ndarray = None,
                 weights: np.ndarray = None):
        self.means = means[:, np.newaxis] if means.ndim == 1 else means
        self.dim = self.means.shape[-1]
        if weights is None:
            # Default: equal mixture weights
            self.weights = np.ones(self.n_components) / self.n_components
        else:
            self.weights = weights
        self.covs = covs

        super().__init__()

    def __setattr__(self, key, value):
        self.__dict__[key] = value
        if key == 'covs':
            if value is None:
                # Default: identity covariances
                self.covs = np.array([np.eye(self.dim)] * self.n_components)
            elif value.shape == (self.dim, self.dim):
                self.covs = np.repeat(value[np.newaxis, :, :], self.n_components, axis=0)

            self.precs = vmap(np.linalg.inv)(self.covs)
            self.sqrt_precs = vmap(np.linalg.cholesky)(self.precs)
            self.det_precs = vmap(np.linalg.det)(self.precs)

            self.component_potential = jit(self.component_potential)
            self.component_potentials = jit(vmap(self.component_potential, (None, 0), 0))
            self.component_dens = jit(self.component_dens)
            self.dens = jit(self.dens)
            self.component_potential = jit(self.component_potential)

    @property
    def n_components(self) -> int:
        return len(self.means)

    def component_potential(self,
                            x: np.ndarray,
                            component_index: int) -> Union[float, np.ndarray]:
        return 0.5 * np.sum(np.square((x - self.means[component_index]) @
                                      self.sqrt_precs[component_index]), axis=-1) \
               - np.log(self.weights[component_index]
                        * self.det_precs[component_index]
                        / np.power(2 * np.pi, self.dim * 0.5))

    def component_dens(self,
                       x: np.ndarray,
                       component_index: int) -> Union[float, np.ndarray]:
        return np.exp(-self.component_potential(x, component_index))

    def dens(self, x: np.ndarray) -> Union[float, np.ndarray]:
        return np.sum(np.exp(-self.component_potentials(x, np.arange(self.n_components))), axis=0)

    def potential(self, x: np.ndarray) -> Union[float, np.ndarray]:
        return -np.log(self.dens(x))


class DoubleWell(Scenario):
    name = "Double Well"

    def __init__(self,
                 dim: int = 1):
        self.dim = dim
        super().__init__()

    def potential(self, x: np.ndarray) -> Union[float, np.ndarray]:
        return np.sum(np.power(x, 4), axis=-1) / 4. \
               - np.sum(np.power(x, 2), axis=-1) / 2.


class Rastrigin(Scenario):
    name = "Rastrigin"

    def __init__(self,
                 dim: int = 1,
                 a: float = 1.):
        self.dim = dim
        self.a = a
        super().__init__()

    def potential(self, x: np.ndarray) -> Union[float, np.ndarray]:
        return self.a*self.dim + np.sum(x**2 - self.a * np.cos(2 * np.pi * x), axis=-1)
