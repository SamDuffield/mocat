########################################################################################################################
# Module: abc/scenarios/gk.py
# Description: SIR model.
#
# Web: https://github.com/SamDuffield/mocat
########################################################################################################################

from typing import Union, Tuple

from jax import numpy as np, random

from mocat.src.abc.abc import ABCScenario
from mocat.src.utils import _while_loop_stacked


exponential_sample_buffer = 1e-6


def sir_single_step(carry: Tuple[np.ndarray, float], extra: Tuple[np.ndarray, int, np.ndarray]):
    si, current_time = carry
    params, population_size, random_key = extra
    t_key, haz_key, random_key = random.split(random_key, 3)

    hazards = np.array([params[0] / population_size * si[0] * si[1],
                        params[1] * si[1]])
    sum_hazards = hazards.sum()
    t_increment = -np.log(exponential_sample_buffer + random.uniform(t_key)) / sum_hazards
    hazard_probs = hazards / sum_hazards

    possibilities = np.array([[si[0] - 1, si[1] + 1],
                              [si[0], si[1] - 1]])
    return (possibilities[random.choice(haz_key, 2, p=hazard_probs)], current_time + t_increment),\
           (params, population_size, random_key)


def sir_simulate(initial_si: np.ndarray,
                 params: np.ndarray,
                 population_size: int,
                 random_key: np.ndarray,
                 max_iter: int = 10000) -> Tuple[np.ndarray, np.ndarray]:
    (simulated_si, simulated_times), _ = _while_loop_stacked(lambda carry, extra: carry[0][1] > 0,
                                                             sir_single_step,
                                                             ((initial_si, 0.),
                                                              (params, population_size, random_key)),
                                                             max_iter)

    return simulated_times, simulated_si


class SIR(ABCScenario):
    name: str = 'Lotka-Volterra'
    dim: int = 2

    initial_si: np.ndarray = None
    prior_rates: np.ndarray = np.ones(dim)
    times: np.ndarray = None
    data: np.ndarray = None
    max_iter: int = 10000

    def prior_sample(self,
                     random_key: np.ndarray) -> np.ndarray:
        return -np.log(exponential_sample_buffer + random.uniform(random_key, (self.dim,))) / self.prior_rates

    def prior_potential(self,
                        x: np.ndarray) -> Union[float, np.ndarray]:
        return (self.prior_rates * x).sum()

    def likelihood_sample(self,
                          x: np.ndarray,
                          random_key: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return sir_simulate(self.initial_si, x, self.initial_si.sum(),
                            random_key, max_iter=self.max_iter)


class TransformedSIR(SIR):

    @staticmethod
    def constrain(x):
        return np.exp(x)

    @staticmethod
    def unconstrain(x):
        return np.log(x)

    def prior_sample(self,
                     random_key: np.ndarray) -> np.ndarray:
        return self.unconstrain(super().prior_sample(random_key))

    def prior_potential(self,
                        x: np.ndarray) -> Union[float, np.ndarray]:
        return (self.prior_rates * np.exp(x) - x).sum()

    def likelihood_sample(self,
                          x: np.ndarray,
                          random_key: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return super().likelihood_sample(self.constrain(x), random_key)

