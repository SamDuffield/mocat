########################################################################################################################
# Module: abc/abc.py
# Description: Core objects for ABC, cases where likelihood can't be evaluated but can be simulated.
#
# Web: https://github.com/SamDuffield/mocat
########################################################################################################################
from typing import Union, Tuple

from jax import numpy as jnp, random
from mocat.src.core import cdict, Scenario, impl_checkable, is_implemented
from mocat.src.sample import Sampler


class ABCScenario(Scenario):
    data: jnp.ndarray = None

    def __init__(self,
                 name: str = None,
                 **kwargs):
        super().__init__(name, init_grad=False, **kwargs)

    @impl_checkable
    def potential(self,
                  x: jnp.ndarray,
                  random_key: jnp.ndarray) -> Union[float, jnp.ndarray]:
        raise TypeError(f'{self.name} abc_scenario potential is intractable')

    @impl_checkable
    def likelihood_potential(self,
                             x: jnp.ndarray,
                             random_key: jnp.ndarray) -> Union[float, jnp.ndarray]:
        raise TypeError(f'{self.name} abc_scenario likelihood_potential is intractable')

    def distance_function(self,
                          simulated_data: jnp.ndarray) -> Union[float, jnp.ndarray]:
        return jnp.sqrt(jnp.square(simulated_data - self.data).sum())


class ABCSampler(Sampler):

    def startup(self,
                abc_scenario: ABCScenario,
                n: int,
                random_key: jnp.ndarray = None,
                initial_state: cdict = None,
                initial_extra: cdict = None,
                startup_correction: bool = True,
                **kwargs) -> Tuple[cdict, cdict]:
        initial_state, initial_extra = super().startup(abc_scenario, n, random_key, initial_state, initial_extra,
                                                       **kwargs)

        if not hasattr(initial_state, 'prior_potential') and is_implemented(abc_scenario.prior_potential):
            initial_extra.random_key, subkey = random.split(initial_extra.random_key)
            initial_state.prior_potential = abc_scenario.prior_potential(initial_state.value, random_key)

        if not hasattr(initial_state, 'simulated_data'):
            initial_extra.random_key, subkey = random.split(initial_extra.random_key)
            initial_extra.simulated_data = abc_scenario.likelihood_sample(initial_state.value, subkey)

        if not hasattr(initial_state, 'distance'):
            initial_state.distance = abc_scenario.distance_function(initial_extra.simulated_data)
        return initial_state, initial_extra

