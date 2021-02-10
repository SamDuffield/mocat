########################################################################################################################
# Module: scenarios/toy_examples.py
# Description: Some simple multivariate scenarios
#
# Web: https://github.com/SamDuffield/mocat
########################################################################################################################

from typing import Union, Any

import jax.numpy as jnp
from jax import vmap, jit

from mocat.src.core import Scenario
from mocat.utils import reset_covariance, gaussian_potential


class Gaussian(Scenario):
    name = "Gaussian"

    covariance_sqrt: jnp.ndarray
    precision_sqrt: jnp.ndarray
    precision_det: float

    def __init__(self,
                 dim: int = 1,
                 mean: Union[float, jnp.ndarray] = None,
                 covariance: Union[float, jnp.ndarray] = None,
                 **kwargs):
        if mean is not None:
            self.dim = mean.shape[-1]
        elif covariance is not None:
            self.dim = covariance.shape[-1]
        else:
            self.dim = dim
        self.mean = jnp.zeros(dim) if mean is None else mean
        self.covariance = jnp.eye(dim) if covariance is None else covariance
        super().__init__(**kwargs)

    def likelihood_potential(self,
                             x: jnp.ndarray,
                             random_key: jnp.ndarray = None) -> Union[float, jnp.ndarray]:
        x_diff = (x - self.mean) @ self.precision_sqrt.T
        return 0.5 * jnp.sum(jnp.square(x_diff), axis=-1)

    def __setattr__(self,
                    key: str,
                    value: Any):
        self.__dict__[key] = value
        if key[-10:] == 'covariance':
            reset_covariance(self, key, value)


class GaussianMixture(Scenario):
    name = "Gaussian Mixture"

    def __init__(self,
                 means: jnp.ndarray,
                 covariances: jnp.ndarray = None,
                 weights: jnp.ndarray = None,
                 **kwargs):
        self.means = means[:, jnp.newaxis] if means.ndim == 1 else means
        self.dim = self.means.shape[-1]
        if weights is None:
            # Default: equal mixture weights
            self.weights = jnp.ones(self.n_components) / self.n_components
        else:
            self.weights = weights
        self.covariances = covariances

        super().__init__(**kwargs)

    def __setattr__(self, key, value):
        self.__dict__[key] = value
        if key == 'covariances':
            if value is None:
                # Default: identity covariances
                self.covariances = jnp.array([jnp.eye(self.dim)] * self.n_components)
            elif value.shape == (self.dim, self.dim):
                self.covariances = jnp.repeat(value[jnp.newaxis, :, :], self.n_components, axis=0)

            self.precisions = vmap(jnp.linalg.inv)(self.covariances)
            self.precision_sqrts = vmap(jnp.linalg.cholesky)(self.precisions)
            self.precision_dets = vmap(jnp.linalg.det)(self.precisions)

            self.component_potential = jit(self.component_potential)
            self.component_potentials = jit(vmap(self.component_potential, (None, 0), 0))
            self.component_dens = jit(self.component_dens)
            self.dens = jit(self.dens)
            self.component_potential = jit(self.component_potential)

    @property
    def n_components(self) -> int:
        return len(self.means)

    def component_potential(self,
                            x: jnp.ndarray,
                            component_index: int) -> Union[float, jnp.ndarray]:
        return 0.5 * jnp.sum(jnp.square((x - self.means[component_index]) @
                                      self.precision_sqrts[component_index].T), axis=-1) \
               - jnp.log(self.weights[component_index]
                        * self.precision_dets[component_index]
                        / jnp.power(2 * jnp.pi, self.dim * 0.5))

    def component_dens(self,
                       x: jnp.ndarray,
                       component_index: int) -> Union[float, jnp.ndarray]:
        return jnp.exp(-self.component_potential(x, component_index))

    def dens(self, x: jnp.ndarray) -> Union[float, jnp.ndarray]:
        return jnp.sum(jnp.exp(-self.component_potentials(x, jnp.arange(self.n_components))), axis=0)

    def likelihood_potential(self,
                             x: jnp.ndarray,
                             random_key: jnp.ndarray = None) -> Union[float, jnp.ndarray]:
        return -jnp.log(self.dens(x))


class DoubleWell(Scenario):
    name = "Double Well"

    def __init__(self,
                 dim: int = 1,
                 **kwargs):
        self.dim = dim
        super().__init__(**kwargs)

    def likelihood_potential(self,
                             x: jnp.ndarray,
                             random_key: jnp.ndarray = None) -> Union[float, jnp.ndarray]:
        return jnp.sum(jnp.power(x, 4), axis=-1) / 4. \
               - jnp.sum(jnp.power(x, 2), axis=-1) / 2.


class Rastrigin(Scenario):
    name = "Rastrigin"

    def __init__(self,
                 dim: int = 1,
                 a: float = 1.,
                 **kwargs):
        self.dim = dim
        self.a = a
        super().__init__(**kwargs)

    def likelihood_potential(self,
                             x: jnp.ndarray,
                             random_key: jnp.ndarray = None) -> Union[float, jnp.ndarray]:
        return self.a*self.dim + jnp.sum(x**2 - self.a * jnp.cos(2 * jnp.pi * x), axis=-1)
