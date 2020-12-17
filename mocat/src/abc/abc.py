########################################################################################################################
# Module: abc/abc.py
# Description: Core objects for ABC, cases where likelihood can't be evaluated but can be simulated.
#
# Web: https://github.com/SamDuffield/mocat
########################################################################################################################

from typing import Union, Tuple

from jax import numpy as np, random, vmap, jit
from mocat.src.core import cdict, Scenario
from mocat.src.mcmc.sampler import MCMCSampler


class ABCScenario(Scenario):
    data: np.ndarray = None
    summary_statistic: np.ndarray = None

    def __init__(self,
                 name: str = None,
                 **kwargs):
        if name is not None:
            self.name = name

        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def potential(self, x: np.ndarray) -> Union[float, np.ndarray]:
        raise TypeError(f'{self.name} abc_scenario potential is intractable')

    def dens(self, x: np.ndarray) -> Union[float, np.ndarray]:
        raise TypeError(f'{self.name} abc_scenario dens is intractable')

    def likelihood_sample(self,
                          x: np.ndarray,
                          random_key: np.ndarray) -> np.ndarray:
        raise AttributeError(f'{self.name} likelihood_sample not initiated')

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

    def summarise_data(self,
                       data: np.ndarray) -> np.ndarray:
        return data

    def distance_function(self,
                          summarised_simulated_data: np.ndarray) -> Union[float, np.ndarray]:
        raise AttributeError(f'{self.name} distance_function not initiated')


class ABCSampler(MCMCSampler):
    tuning = None

    def __init__(self,
                 threshold: float = np.inf,
                 **kwargs):
        super().__init__(threshold=threshold, **kwargs)

    def startup(self,
                abc_scenario: ABCScenario,
                initial_state: cdict = None,
                initial_extra: cdict = None,
                random_key: np.ndarray = None) -> Tuple[cdict, cdict]:
        if abc_scenario.summary_statistic is None:
            abc_scenario.summary_statistic = abc_scenario.summarise_data(abc_scenario.data)

        if initial_state is None:
            x0 = np.zeros(abc_scenario.dim)
            random_key, subkey = random.split(random_key)
            initial_state = cdict(value=x0)

        if initial_extra is None:
            initial_extra = cdict(random_key=random_key,
                                  iter=0)
        if hasattr(self, 'parameters') and not hasattr(initial_extra, 'parameters'):
            initial_extra.parameters = self.parameters.copy()

        if not hasattr(initial_state, 'prior_potential'):
            initial_state.prior_potential = abc_scenario.prior_potential(initial_state.value)

        if not hasattr(initial_state, 'simulated_summary'):
            initial_extra.random_key, subkey = random.split(initial_extra.random_key)
            initial_extra.simulated_summary = self.simulate_summary(abc_scenario, initial_state.value, subkey)

        if not hasattr(initial_state, 'distance'):
            initial_state.distance = abc_scenario.distance_function(initial_extra.simulated_summary)
        return initial_state, initial_extra

    def always(self,
               abc_scenario: ABCScenario,
               reject_state: cdict,
               reject_extra: cdict) -> Tuple[cdict, cdict]:
        reject_extra.random_key, _ = random.split(reject_extra.random_key)
        return reject_state, reject_extra

    def proposal(self,
                 abc_scenario: ABCScenario,
                 reject_state: cdict,
                 reject_extra: cdict) -> Tuple[cdict, cdict]:
        raise NotImplementedError(f'{self.__class__.__name__} markov proposal not initiated')

    def proposal_potential(self,
                           abc_scenario: ABCScenario,
                           reject_state: cdict, reject_extra: cdict,
                           proposed_state: cdict, proposed_extra: cdict) -> Union[float, np.ndarray]:
        raise AttributeError(f'{self.__class__.__name__} proposal_potential not initiated')

    def acceptance_probability(self,
                               abc_scenario: ABCScenario,
                               reject_state: cdict, reject_extra: cdict,
                               proposed_state: cdict, proposed_extra: cdict) -> Union[float, np.ndarray]:
        raise AttributeError(f'{self.__class__.__name__} acceptance_probability not initiated')

    def simulate_summary(self,
                         abc_scenario: ABCScenario,
                         x: np.ndarray,
                         random_key: np.ndarray) -> np.ndarray:
        simulated_data = abc_scenario.simulate_data(x, random_key)
        return abc_scenario.summarise_data(simulated_data)


