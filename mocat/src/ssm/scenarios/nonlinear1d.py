########################################################################################################################
# Module: ssm/scenarios/nonlinear1d.py
# Description: One-dimensional non-linear benchmark model.
#
# Web: https://github.com/SamDuffield/mocat
########################################################################################################################


from typing import Union

from jax import numpy as np, random
from mocat.src.ssm.ssm import StateSpaceModel
from mocat.src.utils import gaussian_potential


class NonLinear1DBenchmark(StateSpaceModel):
    name = '1D Non-linear Benchmark'
    dim = 1
    dim_obs = 1

    def __init__(self,
                 initial_sd: float = np.sqrt(2.),
                 transition_sd: float = np.sqrt(10.),
                 likelihood_sd: float = 1.,
                 name: str = None):
        self.initial_sd = initial_sd
        self.transition_sd = transition_sd
        self.likelihood_sd = likelihood_sd
        super().__init__(name=name)

    def initial_potential(self,
                          x: np.ndarray,
                          t: float) -> Union[float, np.ndarray]:
        return gaussian_potential(x / self.initial_sd)

    def initial_sample(self,
                       t: float,
                       random_key: np.ndarray) -> Union[float, np.ndarray]:
        return random.normal(random_key, (1,)) * self.initial_sd

    def _transition_mean(self,
                         x_previous: np.ndarray,
                         t_previous: float) -> np.ndarray:
        return 0.5 * x_previous\
               + 25 * x_previous / (1 + x_previous ** 2)\
               + 8 * np.cos(1.2 * t_previous)

    def transition_potential(self,
                             x_previous: np.ndarray,
                             t_previous: float,
                             x_new: np.ndarray,
                             t_new: float) -> Union[float, np.ndarray]:
        transition_mean = self._transition_mean(x_previous, t_previous)
        return gaussian_potential((x_new - transition_mean) / self.transition_sd)

    def transition_sample(self,
                          x_previous: np.ndarray,
                          t_previous: float,
                          t_new: float,
                          random_key: np.ndarray) -> np.ndarray:
        transition_mean = self._transition_mean(x_previous, t_previous)
        return transition_mean + random.normal(random_key, (1,)) * self.transition_sd

    def _likelihood_mean(self,
                         x: np.ndarray,
                         t: float) -> np.ndarray:
        return x ** 2 / 20

    def likelihood_potential(self,
                             x: np.ndarray,
                             y: np.ndarray,
                             t: float) -> Union[float, np.ndarray]:
        lik_mean = self._likelihood_mean(x, t)
        return gaussian_potential((y - lik_mean)/self.likelihood_sd)

    def likelihood_sample(self,
                          x: np.ndarray,
                          t: float,
                          random_key: np.ndarray) -> np.ndarray:
        lik_mean = self._likelihood_mean(x, t)
        return lik_mean + random.normal(random_key, (1,)) * self.likelihood_sd