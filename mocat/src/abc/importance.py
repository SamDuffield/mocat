########################################################################################################################
# Module: abc/importance.py
# Description: Importance (or rejection) sampling for ABC, weight non-zero if simulated summary statistic hits ball.
#
# Web: https://github.com/SamDuffield/mocat
########################################################################################################################


from typing import Tuple

from jax import numpy as jnp, random
from mocat.src.abc.abc import ABCScenario, ABCSampler
from mocat.src.core import cdict, is_implemented


class ImportanceABC(ABCSampler):
    name = 'Importance ABC'

    def importance_proposal(self,
                            abc_scenario: ABCScenario,
                            random_key: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError(f'{self.__class__.__name__} importance_proposal not initiated')

    def importance_potential(self,
                             abc_scenario: ABCScenario,
                             x: jnp.ndarray) -> float:
        raise AttributeError(f'{self.__class__.__name__} importance_potential not initiated')

    def log_weight(self,
                   abc_scenario: ABCScenario,
                   state: cdict,
                   extra: cdict):
        return jnp.where(state.distance < extra.parameters.threshold,
                         -state.prior_potential + self.importance_potential(abc_scenario, state.value),
                         -jnp.inf)

    def startup(self,
                abc_scenario: ABCScenario,
                n: int,
                initial_state: cdict = None,
                initial_extra: cdict = None,
                **kwargs) -> Tuple[cdict, cdict]:
        if initial_state is None:
            if is_implemented(abc_scenario.prior_sample):
                initial_extra.random_key, sub_key = random.split(initial_extra.random_key)
                init_vals = abc_scenario.prior_sample(sub_key)
            else:
                init_vals = jnp.zeros(abc_scenario.dim)
            initial_state = cdict(value=init_vals)

        self.max_iter = n

        initial_state, initial_extra = super().startup(abc_scenario, n,
                                                       initial_state, initial_extra, **kwargs)

        initial_state.log_weight = self.log_weight(abc_scenario, initial_state, initial_extra)
        return initial_state, initial_extra

    def proposal(self,
                 abc_scenario: ABCScenario,
                 reject_state: cdict,
                 reject_extra: cdict) -> Tuple[cdict, cdict]:
        proposed_state = reject_state.copy()
        proposed_extra = reject_extra.copy()
        proposed_extra.random_key, subkey1, subkey2, subkey3 = random.split(reject_extra.random_key, 4)
        proposed_state.value = self.importance_proposal(abc_scenario, subkey1)
        proposed_state.prior_potential = abc_scenario.prior_potential(proposed_state.value, subkey2)
        proposed_state.simulated_data = abc_scenario.likelihood_sample(proposed_state.value, subkey3)
        proposed_state.distance = abc_scenario.distance_function(proposed_state.simulated_data)
        proposed_state.log_weight = self.log_weight(abc_scenario, proposed_state, proposed_extra)
        return proposed_state, proposed_extra

    def update(self,
               abc_scenario: ABCScenario,
               state: cdict,
               extra: cdict) -> Tuple[cdict, cdict]:
        proposed_state, proposed_extra = self.proposal(abc_scenario, state, extra)
        proposed_state.log_weight = self.log_weight(abc_scenario, proposed_state, proposed_extra)
        return proposed_state, proposed_extra


class VanillaABC(ImportanceABC):
    name = 'Vanilla ABC'

    def __init__(self,
                 threshold: float = jnp.inf,
                 acceptance_rate: float = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.parameters.threshold = threshold
        if acceptance_rate is not None:
            self.parameters.acceptance_rate = acceptance_rate
            self.clean_chain = self.clean_chain_ar

    def importance_proposal(self,
                            abc_scenario: ABCScenario,
                            random_key: jnp.ndarray) -> jnp.ndarray:
        return abc_scenario.prior_sample(random_key)

    def importance_potential(self,
                             abc_scenario: ABCScenario,
                             x: jnp.ndarray) -> float:
        return abc_scenario.prior_potential(x)

    def log_weight(self,
                   abc_scenario: ABCScenario,
                   state: cdict,
                   extra: cdict):
        return jnp.where(state.distance < extra.parameters.threshold,
                         0.,
                         -jnp.inf)

    def clean_chain_ar(self,
                       abc_scenario: ABCScenario,
                       chain_state: cdict):
        threshold = jnp.quantile(chain_state.distance, self.parameters.acceptance_rate)
        self.parameters.threshold = float(threshold)
        chain_state.log_weight = jnp.where(chain_state.distance < threshold,
                                           0.,
                                           -jnp.inf)
        return chain_state
