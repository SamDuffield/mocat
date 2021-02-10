########################################################################################################################
# Module: ssm/scenarios/stochastic_volatility.py
# Description: Multivariate stochastic volatility model.
#
# Web: https://github.com/SamDuffield/mocat
########################################################################################################################

from typing import Union, Any

from jax import numpy as jnp, random
from mocat.src.utils import gaussian_potential, reset_covariance
from mocat.src.ssm.ssm import StateSpaceModel


class StochasticVolatility(StateSpaceModel):
    name = 'Stochastic Volatility'

    initial_covariance_sqrt = None
    initial_precision_sqrt = None
    transition_covariance_sqrt = None
    transition_precision_sqrt = None
    likelihood_covariance_diag_sqrt = None
    likelihood_precision_diag_sqrt = None

    def __init__(self,
                 dim: int,
                 initial_mean: jnp.ndarray = None,
                 initial_covariance: jnp.ndarray = None,
                 transition_matrix: jnp.ndarray = None,
                 transition_covariance: jnp.ndarray = None,
                 likelihood_covariance_diag: jnp.ndarray = None,
                 name: str = None):
        self.dim = dim
        self.dim_obs = dim
        if initial_mean is None:
            initial_mean = jnp.zeros(self.dim)
        self.initial_mean = initial_mean

        if initial_covariance is None:
            initial_covariance = jnp.eye(self.dim)
        self.initial_covariance = initial_covariance

        if transition_matrix is None:
            transition_matrix = jnp.eye(self.dim)
        self.transition_matrix = transition_matrix

        if transition_covariance is None:
            transition_covariance = jnp.eye(self.dim)
        self.transition_covariance = transition_covariance

        if likelihood_covariance_diag is None:
            likelihood_covariance_diag = jnp.eye(self.dim)
        self.likelihood_covariance_diag = likelihood_covariance_diag

        super().__init__(name=name)

    def __setattr__(self,
                    key: str,
                    value: Any):
        self.__dict__[key] = value
        if key[-10:] == 'covariance' or key[-15:] == 'covariance_diag':
            reset_covariance(self, key, value)

    def initial_potential(self,
                          x: jnp.ndarray,
                          t: Union[float, None]) -> Union[float, jnp.ndarray]:
        init_mean = self.initial_mean
        init_prec_sqrt = self.initial_precision_sqrt
        return gaussian_potential(x, init_mean, sqrt_prec=init_prec_sqrt)

    def initial_sample(self,
                       t: Union[float, None],
                       random_key: jnp.ndarray) -> Union[float, jnp.ndarray]:
        init_mean = self.initial_mean
        init_cov_sqrt = self.initial_covariance_sqrt
        return random.normal(random_key, shape=(self.dim,)) @ init_cov_sqrt.T + init_mean

    def transition_potential(self,
                             x_previous: jnp.ndarray,
                             t_previous: float,
                             x_new: jnp.ndarray,
                             t_new: float) -> Union[float, jnp.ndarray]:
        return gaussian_potential(x_new,
                                  x_previous @ self.transition_matrix.T,
                                  sqrt_prec=self.transition_precision_sqrt * jnp.sqrt(t_new - t_previous))

    def transition_sample(self,
                          x_previous: jnp.ndarray,
                          t_previous: float,
                          t_new: float,
                          random_key: jnp.ndarray) -> jnp.ndarray:
        return x_previous @ self.transition_matrix.T \
               + random.normal(random_key, shape=x_previous.shape) @ self.transition_covariance_sqrt.T \
               * jnp.sqrt(t_new - t_previous)

    def likelihood_potential(self,
                             x: jnp.ndarray,
                             y: jnp.ndarray,
                             t: float) -> Union[float, jnp.ndarray]:
        return gaussian_potential(y,
                                  0,
                                  sqrt_prec=self.likelihood_precision_diag_sqrt / jnp.exp(0.5*x))

    def likelihood_sample(self,
                          x: jnp.ndarray,
                          t: float,
                          random_key: jnp.ndarray) -> jnp.ndarray:
        return self.likelihood_covariance_diag_sqrt * jnp.exp(0.5*x) * random.normal(random_key, shape=x.shape)
