########################################################################################################################
# Module: abc/scenarios/gk.py
# Description: Linear Gaussian Bayesian inverse problem.
#
# Web: https://github.com/SamDuffield/mocat
########################################################################################################################

from typing import Union, Any

from jax import numpy as np, random

from mocat.src.abc.abc import ABCScenario
from mocat.src.utils import extract_dimension, reset_covariance, gaussian_potential


class LinearGaussian(ABCScenario):
    name = 'Linear Gaussian'

    prior_covariance_sqrt = None
    prior_precision_sqrt = None
    likelihood_covariance_sqrt = None
    likelihood_precision_sqrt = None

    def __init__(self,
                 prior_mean: Union[float, np.ndarray] = 0.,
                 prior_covariance: Union[float, np.ndarray] = 1.,
                 likelihood_matrix: Union[float, np.ndarray] = 1.,
                 likelihood_covariance: Union[float, np.ndarray] = 1.,
                 **kwargs):

        self.dim = extract_dimension(prior_mean, prior_covariance,
                                     likelihood_matrix, likelihood_covariance)
        if self.dim is None:
            raise AttributeError(f'Could not find dimension for {self.__class__.__name__}')

        if isinstance(prior_mean, float):
            prior_mean = np.ones(self.dim) * prior_mean
        self.prior_mean = prior_mean

        if isinstance(prior_covariance, float):
            prior_covariance = np.eye(self.dim) * prior_covariance
        self.prior_covariance = prior_covariance

        if isinstance(likelihood_matrix, float):
            likelihood_matrix = np.eye(self.dim) * likelihood_matrix
        self.dim_obs = likelihood_matrix.shape[0]
        self.likelihood_matrix = likelihood_matrix

        if isinstance(likelihood_covariance, float):
            likelihood_covariance = np.eye(self.dim_obs) * likelihood_covariance
        self.likelihood_covariance = likelihood_covariance

        super().__init__(**kwargs)

    def __setattr__(self,
                    key: str,
                    value: Any):
        self.__dict__[key] = value
        if key[-10:] == 'covariance':
            reset_covariance(self, key, value)

    def prior_sample(self,
                     random_key: np.ndarray) -> np.ndarray:
        return self.prior_mean + self.prior_covariance_sqrt @ random.normal(random_key, (self.dim,))

    def prior_potential(self,
                        x: np.ndarray) -> Union[float, np.ndarray]:
        return gaussian_potential(x, self.prior_mean, sqrt_prec=self.prior_precision_sqrt)

    def likelihood_sample(self,
                          x: np.ndarray,
                          random_key: np.ndarray) -> np.ndarray:
        return self.likelihood_matrix @ x + self.likelihood_covariance_sqrt @ random.normal(random_key, (self.dim_obs,))

    def likelihood_potential(self,
                             x: np.ndarray,
                             y: np.ndarray) -> Union[float, np.ndarray]:
        return gaussian_potential(y, x @ self.likelihood_matrix.T, sqrt_prec=self.likelihood_precision_sqrt)

    def potential(self, x: np.ndarray) -> Union[float, np.ndarray]:
        return self.prior_potential(x) + self.likelihood_potential(x, self.summary_statistic)

    def summarise_data(self,
                       data: np.ndarray) -> np.ndarray:
        return data

    def simulate(self,
                 x: np.ndarray,
                 random_key: np.ndarray) -> np.ndarray:
        return self.likelihood_sample(x, random_key)

