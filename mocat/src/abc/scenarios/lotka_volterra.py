########################################################################################################################
# Module: abc/scenarios/gk.py
# Description: Stochastic Lotka-Volterra (predator-prey).
#
# Web: https://github.com/SamDuffield/mocat
########################################################################################################################

from typing import Union, Tuple

from jax import numpy as jnp, random

from mocat.src.abc.abc import ABCScenario
from mocat.src.utils import _while_loop_stacked

exponential_sample_buffer = 1e-6


def lotka_volterra_single_step(carry: Tuple[jnp.ndarray, float], extra: Tuple[jnp.ndarray, jnp.ndarray]):
    prey_pred, current_time = carry
    params, random_key = extra
    t_key, haz_key, random_key = random.split(random_key, 3)

    hazards = jnp.array([params[0] * prey_pred[0],
                        params[1] * prey_pred[0] * prey_pred[1],
                        params[2] * prey_pred[1]])
    sum_hazards = hazards.sum()
    t_increment = -jnp.log(exponential_sample_buffer + random.uniform(t_key)) / sum_hazards
    hazard_probs = hazards / sum_hazards

    possibilities = jnp.array([[prey_pred[0] + 1, prey_pred[1]],
                              [prey_pred[0] - 1, prey_pred[1] + 1],
                              [prey_pred[0], prey_pred[1] - 1]])
    return (possibilities[random.choice(haz_key, 3, p=hazard_probs)], current_time + t_increment), (params, random_key)


def lotka_volterra_simulate(initial_prey_pred: jnp.ndarray,
                            times: jnp.ndarray,
                            params: jnp.ndarray,
                            random_key: jnp.ndarray,
                            max_iter: int = 10000) -> jnp.ndarray:
    max_time = jnp.max(times)

    (simulated_pred_prey, simulated_times), _ = _while_loop_stacked(lambda carry, extra: carry[1] < max_time,
                                                                    lotka_volterra_single_step,
                                                                    ((initial_prey_pred, times[0]),
                                                                     (params, random_key)),
                                                                    max_iter)

    simulated_times = jnp.where(simulated_times == 0, jnp.inf, simulated_times)
    return simulated_pred_prey[jnp.searchsorted(simulated_times, times[1:]) - 1]


class LotkaVolterra(ABCScenario):
    name: str = 'Lotka-Volterra'
    dim: int = 3

    initial_prey_pred: jnp.ndarray = None
    prior_rates: jnp.ndarray = jnp.ones(dim)
    times: jnp.ndarray = None
    data: jnp.ndarray = None
    max_iter: int = 10000

    def prior_sample(self,
                     random_key: jnp.ndarray) -> jnp.ndarray:
        return -jnp.log(exponential_sample_buffer + random.uniform(random_key, (self.dim,))) / self.prior_rates

    def prior_potential(self,
                        x: jnp.ndarray) -> Union[float, jnp.ndarray]:
        return (self.prior_rates * x).sum()

    def likelihood_sample(self,
                      x: jnp.ndarray,
                      random_key: jnp.ndarray) -> jnp.ndarray:
        return lotka_volterra_simulate(self.initial_prey_pred, self.times, x, random_key, max_iter=self.max_iter)[:, 0]


class TransformedLotkaVolterra(LotkaVolterra):

    @staticmethod
    def constrain(x):
        return jnp.exp(x)

    @staticmethod
    def unconstrain(x):
        return jnp.log(x)

    def prior_sample(self,
                     random_key: jnp.ndarray) -> jnp.ndarray:
        return self.unconstrain(super().prior_sample(random_key))

    def prior_potential(self,
                        x: jnp.ndarray) -> Union[float, jnp.ndarray]:
        return (self.prior_rates * jnp.exp(x) - x).sum()

    def simulate_data(self,
                      x: jnp.ndarray,
                      random_key: jnp.ndarray) -> jnp.ndarray:
        return super().likelihood_sample(self.constrain(x), random_key)
