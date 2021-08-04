########################################################################################################################
# Module: abc/scenarios/gk.py
# Description: Ricker model.
#
# Web: https://github.com/SamDuffield/mocat
########################################################################################################################

from typing import Tuple

from jax import numpy as jnp, random
from jax.lax import scan

from mocat.src.abc.abc import ABCScenario


def ricker_simulate(initial_n: int,
                    r: float,
                    noise_sd: float,
                    num_steps: int,
                    random_key: jnp.ndarray) -> jnp.ndarray:
    random_keys = random.split(random_key, num_steps)

    def body_fun(n: int, i: int) -> Tuple[int, int]:
        new_n = r * n * jnp.exp(-n + noise_sd * random.normal(random_keys[i]))
        return new_n, new_n

    _, ns = scan(body_fun, initial_n, jnp.arange(num_steps))
    return jnp.append(initial_n, ns)


class Ricker(ABCScenario):
    name: str = 'Ricker'
    dim: int = 3

    initial_n: int = 1
    num_steps: int = None
    observation_inds: jnp.ndarray = None
    data: jnp.ndarray = None

    def likelihood_sample(self,
                          x: jnp.ndarray,
                          random_key: jnp.ndarray) -> jnp.ndarray:
        dynamics_key, obs_key = random.split(random_key)
        ns = ricker_simulate(self.initial_n, x[0], x[1], self.num_steps, dynamics_key)
        observed_ns = ns[self.observation_inds]
        obs = random.poisson(obs_key, x[2] * observed_ns, shape=self.observation_inds.shape)
        return jnp.where(obs < 0, 0, obs)


