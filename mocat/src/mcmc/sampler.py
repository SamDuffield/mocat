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

    update_initial_state = True
    update_initial_extra = True

    def __init__(self,
                 **kwargs):
        self.initial_state = None
        self.initial_extra = None

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
                random_key: np.ndarray):
        default_initial_state(self, scenario)
        default_initial_extra(self, random_key)

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


def default_initial_state(sampler: MCMCSampler,
                          scenario: Scenario):
    if sampler.initial_state is None \
            or not hasattr(sampler.initial_state, 'value') \
            or sampler.initial_state.value.shape[-1] != scenario.dim:
        x0 = np.zeros(scenario.dim)
        sampler.initial_state = CDict(value=x0)


def default_initial_extra(sampler: MCMCSampler,
                          random_key: np.ndarray):
    if sampler.initial_extra is None:
        sampler.initial_extra = CDict(random_key=random_key,
                                      iter=0)
    if hasattr(sampler, 'parameters'):
        sampler.initial_extra.parameters = sampler.parameters


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
