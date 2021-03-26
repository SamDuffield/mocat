########################################################################################################################
# Module: abc/scenarios/gk.py
# Description: SIR model.
#
# Web: https://github.com/SamDuffield/mocat
########################################################################################################################

from typing import Union, Tuple

from jax import numpy as jnp, random

from mocat.src.abc.abc import ABCScenario
from mocat.src.utils import _while_loop_stacked

exponential_sample_buffer = 1e-6


def sir_single_step(carry: Tuple[jnp.ndarray, float], extra: Tuple[jnp.ndarray, int, jnp.ndarray]):
    si, current_time = carry
    params, population_size, random_key = extra
    t_key, haz_key, random_key = random.split(random_key, 3)

    hazards = jnp.array([params[0] / population_size * si[0] * si[1],
                         params[1] * si[1]])
    sum_hazards = hazards.sum()
    t_increment = -jnp.log(exponential_sample_buffer + random.uniform(t_key)) / sum_hazards
    hazard_probs = hazards / sum_hazards

    possibilities = jnp.array([[si[0] - 1, si[1] + 1],
                               [si[0], si[1] - 1]])
    return (possibilities[random.choice(haz_key, 2, p=hazard_probs)], current_time + t_increment), \
           (params, population_size, random_key)


def sir_simulate(initial_si: jnp.ndarray,
                 params: jnp.ndarray,
                 population_size: int,
                 random_key: jnp.ndarray,
                 max_iter: int = 10000) -> Tuple[jnp.ndarray, jnp.ndarray]:
    (simulated_si, simulated_times), _, _ = _while_loop_stacked(lambda carry, extra: carry[0][1] > 0,
                                                                sir_single_step,
                                                                ((initial_si, 0.),
                                                                 (params, population_size, random_key)),
                                                                max_iter)

    return simulated_times, simulated_si


class SIR(ABCScenario):
    name: str = 'Lotka-Volterra'
    dim: int = 2

    initial_si: jnp.ndarray = None
    prior_rates: jnp.ndarray = jnp.ones(dim)
    times: jnp.ndarray = None
    data: jnp.ndarray = None
    max_iter: int = 10000

    def prior_sample(self,
                     random_key: jnp.ndarray) -> jnp.ndarray:
        return -jnp.log(exponential_sample_buffer + random.uniform(random_key, (self.dim,))) / self.prior_rates

    def prior_potential(self,
                        x: jnp.ndarray,
                        random_key: jnp.ndarray = None) -> Union[float, jnp.ndarray]:
        return (self.prior_rates * x).sum()

    def simulate_times_and_si(self,
                              x: jnp.ndarray,
                              random_key: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        return sir_simulate(self.initial_si, x, self.initial_si.sum(),
                            random_key, max_iter=self.max_iter)


class TransformedSIR(SIR):

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
                        x: jnp.ndarray,
                        random_key: jnp.ndarray = None) -> Union[float, jnp.ndarray]:
        return (self.prior_rates * jnp.exp(x) - x).sum()

    def simulate_times_and_si(self,
                              x: jnp.ndarray,
                              random_key: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        return super().simulate_times_and_si(self.constrain(x), random_key)
