########################################################################################################################
# Module: abc/scenarios/gk.py
# Description: Stochastic Lotka-Volterra (predator-prey).
#
# Web: https://github.com/SamDuffield/mocat
########################################################################################################################

from typing import Union, Tuple

from jax import numpy as np, random

from mocat.src.abc.abc import ABCScenario
from mocat.src.utils import _while_loop_stacked

exponential_sample_buffer = 1e-6


def lotka_volterra_single_step(carry: Tuple[np.ndarray, float], extra: Tuple[np.ndarray, np.ndarray]):
    prey_pred, current_time = carry
    params, random_key = extra
    t_key, haz_key, random_key = random.split(random_key, 3)

    hazards = np.array([params[0] * prey_pred[0],
                        params[1] * prey_pred[0] * prey_pred[1],
                        params[2] * prey_pred[1]])
    sum_hazards = hazards.sum()
    t_increment = -np.log(exponential_sample_buffer + random.uniform(t_key)) / sum_hazards
    hazard_probs = hazards / sum_hazards

    possibilities = np.array([[prey_pred[0] + 1, prey_pred[1]],
                              [prey_pred[0] - 1, prey_pred[1] + 1],
                              [prey_pred[0], prey_pred[1] - 1]])
    return (possibilities[random.choice(haz_key, 3, p=hazard_probs)], current_time + t_increment), (params, random_key)


def lotka_volterra_simulate(initial_prey_pred: np.ndarray,
                            times: np.ndarray,
                            params: np.ndarray,
                            random_key: np.ndarray,
                            max_iter: int = 10000) -> np.ndarray:
    max_time = np.max(times)

    (simulated_pred_prey, simulated_times), _ = _while_loop_stacked(lambda carry, extra: carry[1] < max_time,
                                                                    lotka_volterra_single_step,
                                                                    ((initial_prey_pred, times[0]),
                                                                     (params, random_key)),
                                                                    max_iter)

    simulated_times = np.where(simulated_times == 0, np.inf, simulated_times)
    return simulated_pred_prey[np.searchsorted(simulated_times, times[1:]) - 1]


class LotkaVolterra(ABCScenario):
    name: str = 'Lotka-Volterra'
    dim: int = 3

    initial_prey_pred: np.ndarray = None
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

    def simulate_data(self,
                      x: np.ndarray,
                      random_key: np.ndarray) -> np.ndarray:
        return lotka_volterra_simulate(self.initial_prey_pred, self.times, x, random_key, max_iter=self.max_iter)[:, 0]


class TransformedLotkaVolterra(LotkaVolterra):

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

    def simulate_data(self,
                      x: np.ndarray,
                      random_key: np.ndarray) -> np.ndarray:
        return super().simulate_data(self.constrain(x), random_key)
