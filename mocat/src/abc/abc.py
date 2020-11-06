########################################################################################################################
# Module: abc/abc.py
# Description: Core objects for ABC, cases where likelihood can't be evaluated but can be simulated.
#
# Web: https://github.com/SamDuffield/mocat
########################################################################################################################

from functools import partial
from typing import Union, Tuple

from jax import numpy as np, random, vmap, jit
from mocat.src.core import CDict, Scenario
from mocat.src.mcmc.sampler import MCMCSampler


class ABCScenario(Scenario):

    data: np.ndarray = None
    summary_statistic: np.ndarray = None
    threshold: float = None

    def __init__(self,
                 name: str = None,
                 **kwargs):
        if name is not None:
            self.name = name

        for key, value in kwargs.items():
            if hasattr(self, key):
                self.__dict__[key] = value

    def potential(self, x: np.ndarray) -> Union[float, np.ndarray]:
        raise TypeError(f'{self.name} abc_scenario potential is intractable')

    def dens(self, x: np.ndarray) -> Union[float, np.ndarray]:
        raise TypeError(f'{self.name} abc_scenario dens is intractable')

    def likelihood_sample(self,
                          x: np.ndarray,
                          random_key: np.ndarray) -> np.ndarray:
        raise NotImplementedError(f'{self.name} likelihood_sample not initiated')

    def simulate_data(self,
                      x: np.ndarray,
                      random_key: np.ndarray) -> np.ndarray:
        data_keys = random.split(random_key, self.data.shape[0])
        return vmap(self.likelihood_sample, (None, 0))(x, data_keys)

    def prior_potential(self,
                        x: np.ndarray) -> Union[float, np.ndarray]:
        raise AttributeError(f'{self.name} prior_potential not initiated')

    def prior_sample(self,
                     random_key: np.ndarray) -> Union[float, np.ndarray]:
        raise AttributeError(f'{self.name} prior_sample not initiated')

    def prior_conditional_potential(self,
                                    i: int,
                                    x_i: np.ndarray,
                                    x_minus_i: np.ndarray) -> Union[float, np.ndarray]:
        raise AttributeError(f'{self.name} prior_conditional_potential not initiated')

    def prior_conditional_sample(self,
                                 i: int,
                                 x_minus_i: np.ndarray,
                                 random_key: np.ndarray) -> Union[float, np.ndarray]:
        raise AttributeError(f'{self.name} prior_conditional_sample not initiated')

    def summarise_data(self,
                       data: np.ndarray):
        raise AttributeError(f'{self.name} summarise_data not initiated')

    def distance_function(self,
                          summary_statistic: np.ndarray) -> Union[float, np.ndarray]:
        raise AttributeError(f'{self.name} distance_function not initiated')


class ABCSampler(MCMCSampler):

    def __init__(self,
                 **kwargs):
        super().__init__(**kwargs)

    def startup(self,
                abc_scenario: ABCScenario,
                initial_state: CDict = None,
                initial_extra: CDict = None,
                random_key: np.ndarray = None) -> Tuple[CDict, CDict]:
        if abc_scenario.summary_statistic is None:
            abc_scenario.summary_statistic = abc_scenario.summarise_data(abc_scenario.data)

        if initial_state is None:
            x0 = np.zeros(abc_scenario.dim)
            random_key, subkey = random.split(random_key)
            initial_state = CDict(value=x0)

        if initial_extra is None:
            initial_extra = CDict(random_key=random_key,
                                  iter=0)
        if hasattr(self, 'parameters'):
            initial_extra.parameters = self.parameters.copy()

        random_key, subkey = random.split(random_key)
        initial_extra.simulated_data = abc_scenario.simulate_data(initial_state.value, subkey)
        initial_extra.simulated_summary_statistic = abc_scenario.summarise_data(initial_extra.simulated_data)
        initial_state.distance = abc_scenario.distance_function(initial_extra.simulated_summary_statistic)
        return initial_state, initial_extra

    def always(self,
               abc_scenario: ABCScenario,
               reject_state: CDict,
               reject_extra: CDict) -> Tuple[CDict, CDict]:
        return reject_state, reject_extra

    def proposal(self,
                 abc_scenario: ABCScenario,
                 reject_state: CDict,
                 reject_extra: CDict) -> Tuple[CDict, CDict]:
        raise NotImplementedError(f'{self.__class__.__name__} markov proposal not initiated')

    def proposal_potential(self,
                           abc_scenario: ABCScenario,
                           reject_state: CDict, reject_extra: CDict,
                           proposed_state: CDict, proposed_extra: CDict) -> Union[float, np.ndarray]:
        raise AttributeError(f'{self.__class__.__name__} proposal_potential not initiated')

    def acceptance_probability(self,
                               abc_scenario: ABCScenario,
                               reject_state: CDict, reject_extra: CDict,
                               proposed_state: CDict, proposed_extra: CDict) -> Union[float, np.ndarray]:
        raise AttributeError(f'{self.__class__.__name__} acceptance_probability not initiated')

