########################################################################################################################
# Module: mcmc/sampler.py
# Description: Generic MCMC sampler class.
#
# Web: https://github.com/SamDuffield/mocat
########################################################################################################################

from typing import Tuple, Union

from jax import numpy as np

from mocat.src.core import Sampler, Scenario, CDict


class MCMCSampler(Sampler):
    default_correction = None

    def __init__(self,
                 **kwargs):

        if not hasattr(self, 'parameters'):
            self.parameters = CDict()

        if not hasattr(self, 'tuning'):
            self.tuning = CDict(parameter='stepsize',
                                target=None,
                                metric='alpha',
                                monotonicity='decreasing')

        # Initiate sampler class (set any additional parameters from init)
        super().__init__(**kwargs)

    def startup(self,
                scenario: Scenario,
                initial_state: CDict = None,
                initial_extra: CDict = None,
                random_key: np.ndarray = None) -> Tuple[CDict, CDict]:
        if initial_state is None:
            x0 = np.zeros(scenario.dim)
            initial_state = CDict(value=x0)

        if initial_extra is None:
            initial_extra = CDict(random_key=random_key,
                                  iter=0)
        if hasattr(self, 'parameters'):
            initial_extra.parameters = self.parameters.copy()

        return initial_state, initial_extra

    def always(self,
               scenario: Scenario,
               reject_state: CDict,
               reject_extra: CDict) -> Tuple[CDict, CDict]:
        return reject_state, reject_extra

    def proposal(self,
                 scenario: Scenario,
                 reject_state: CDict,
                 reject_extra: CDict) -> Tuple[CDict, CDict]:
        raise NotImplementedError(f'{self.__class__.__name__} markov proposal not initiated')

    def proposal_potential(self,
                           scenario: Scenario,
                           reject_state: CDict, reject_extra: CDict,
                           proposed_state: CDict, proposed_extra: CDict) -> Union[float, np.ndarray]:
        raise AttributeError(f'{self.__class__.__name__} proposal_potential not initiated')

    def acceptance_probability(self,
                               scenario: Scenario,
                               reject_state: CDict, reject_extra: CDict,
                               proposed_state: CDict, proposed_extra: CDict) -> Union[float, np.ndarray]:
        raise AttributeError(f'{self.__class__.__name__} acceptance_probability not initiated')


def mh_acceptance_probability(sampler: MCMCSampler,
                              scenario: Scenario,
                              reject_state: CDict, reject_extra: CDict,
                              proposed_state: CDict, proposed_extra: CDict) -> Union[float, np.ndarray]:
    pre_min_alpha = np.exp(- proposed_state.potential
                           + reject_state.potential
                           - sampler.proposal_potential(scenario,
                                                        proposed_state, proposed_extra,
                                                        reject_state, reject_extra)
                           + sampler.proposal_potential(scenario,
                                                        reject_state, reject_extra,
                                                        proposed_state, proposed_extra))

    return np.minimum(1., pre_min_alpha)
