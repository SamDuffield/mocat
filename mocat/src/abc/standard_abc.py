########################################################################################################################
# Module: abc/standard_abc.py
# Description: Importance sampling and Markov ABC, accept if simulated summary_statistic hits ball.
#
# Web: https://github.com/SamDuffield/mocat
########################################################################################################################


from typing import Union, Tuple

from jax import numpy as np, random
from mocat.src.abc.abc import ABCSampler, ABCScenario
from mocat.src.core import cdict
from mocat.src.mcmc.corrections import Metropolis


class ImportanceABC(ABCSampler):
    name = 'Importance ABC'

    def importance_proposal(self,
                            abc_scenario: ABCScenario,
                            random_key: np.ndarray) -> np.ndarray:
        raise NotImplementedError(f'{self.__class__.__name__} importance_proposal not initiated')

    def importance_potential(self,
                             abc_scenario: ABCScenario,
                             x: np.ndarray) -> float:
        raise AttributeError(f'{self.__class__.__name__} importance_potential not initiated')

    def log_weight(self,
                   abc_scenario: ABCScenario,
                   state: cdict,
                   extra: cdict):
        return np.where(abc_scenario.distance_function(state.simulated_data)
                        < abc_scenario.threshold,
                        - abc_scenario.prior_potential(state.value)
                        + self.importance_potential(abc_scenario, state.value),
                        -np.inf)

    def acceptance_probability(self,
                               abc_scenario: ABCScenario,
                               reject_state: cdict, reject_extra: cdict,
                               proposed_state: cdict, proposed_extra: cdict) -> Union[float, np.ndarray]:
        return 1.

    def startup(self,
                abc_scenario: ABCScenario,
                initial_state: cdict = None,
                initial_extra: cdict = None,
                random_key: np.ndarray = None) -> Tuple[cdict, cdict]:
        initial_state, initial_extra = super().startup(abc_scenario, initial_state, initial_extra, random_key)
        initial_state.log_weight = self.log_weight(abc_scenario, initial_state, initial_extra)
        return initial_state, initial_extra

    def proposal(self,
                 abc_scenario: ABCScenario,
                 reject_state: cdict,
                 reject_extra: cdict) -> Tuple[cdict, cdict]:
        proposed_state = reject_state.copy()
        proposed_extra = reject_extra.copy()
        proposed_extra.random_key, subkey1, subkey2 = random.split(reject_extra.random_key, 3)
        proposed_state.value = self.importance_proposal(abc_scenario, subkey1)
        proposed_extra.simulated_summary = self.simulate_summary(abc_scenario, proposed_state.value, subkey2)
        proposed_state.distance = abc_scenario.distance_function(proposed_extra.simulated_summary)
        proposed_state.log_weight = self.log_weight(abc_scenario, proposed_state, proposed_extra)
        return proposed_state, proposed_extra


class VanillaABC(ImportanceABC):
    name = 'Vanilla ABC'

    def importance_proposal(self,
                            abc_scenario: ABCScenario,
                            random_key: np.ndarray) -> np.ndarray:
        return abc_scenario.prior_sample(random_key)

    def importance_potential(self,
                             abc_scenario: ABCScenario,
                             x: np.ndarray) -> float:
        return abc_scenario.prior_potential(x)

    def log_weight(self,
                   abc_scenario: ABCScenario,
                   state: cdict,
                   extra: cdict):
        return np.where(state.distance < abc_scenario.threshold,
                        0.,
                        -np.inf)


class RandomWalkABC(ABCSampler):
    name = 'Random Walk ABC'
    default_correction = Metropolis

    def __init__(self,
                 stepsize: float = None):
        super().__init__()
        self.parameters.stepsize = stepsize

    def acceptance_probability(self,
                               abc_scenario: ABCScenario,
                               reject_state: cdict, reject_extra: cdict,
                               proposed_state: cdict, proposed_extra: cdict) -> Union[float, np.ndarray]:
        return np.minimum(1., np.exp(-proposed_state.prior_potential
                                     + reject_state.prior_potential)
                          * proposed_state.distance < abc_scenario.threshold)

    def proposal(self,
                 abc_scenario: ABCScenario,
                 reject_state: cdict,
                 reject_extra: cdict) -> Tuple[cdict, cdict]:
        proposed_state = reject_state.copy()
        proposed_extra = reject_extra.copy()
        stepsize = reject_extra.parameters.stepsize
        subkey1, subkey2 = random.split(reject_extra.random_key)
        proposed_state.value = reject_state.value + np.sqrt(stepsize) * random.normal(subkey1, (abc_scenario.dim,))
        proposed_extra.simulated_summary = self.simulate_summary(abc_scenario, proposed_state.value, subkey2)
        proposed_state.distance = abc_scenario.distance_function(proposed_extra.simulated_summary)
        return proposed_state, proposed_extra
