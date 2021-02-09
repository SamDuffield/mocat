########################################################################################################################
# Module: mcmc/sampler.py
# Description: Generic MCMC sampler class.
#
# Web: https://github.com/SamDuffield/mocat
########################################################################################################################

from typing import Tuple, Union, Type
from inspect import isclass

from jax import numpy as np, random

from mocat.src.core import Scenario, cdict, is_implemented
from mocat.src.sample import Sampler


class MCMCSampler(Sampler):
    correction: 'Correction' = None

    def __init__(self,
                 **kwargs):

        if not hasattr(self, 'tuning'):
            self.tuning = cdict(parameter='stepsize',
                                target=None,
                                metric='alpha',
                                monotonicity='decreasing')

        # Initiate sampler class (set any additional parameters from init)
        super().__init__(**kwargs)

    def startup(self,
                scenario: Scenario,
                n: int,
                random_key: np.ndarray = None,
                initial_state: cdict = None,
                initial_extra: cdict = None,
                startup_correction: bool = True,
                **kwargs) -> Tuple[cdict, cdict]:
        if initial_state is None:
            if is_implemented(scenario.prior_sample):
                random_key, sub_key = random.split(random_key)
                init_vals = scenario.prior_sample(sub_key)
            else:
                init_vals = np.zeros(scenario.dim)
            initial_state = cdict(value=init_vals)

        self.max_iter = n

        if 'correction' in kwargs.keys():
            self.correction = kwargs['correction']
            del kwargs['correction']

        self.correction = check_correction(self.correction)

        initial_state, initial_extra = super().startup(scenario, n, random_key, initial_state, initial_extra, **kwargs)

        if not hasattr(initial_extra, 'iter'):
            initial_extra.iter = 0

        if startup_correction:
            initial_state, initial_extra = self.correction.startup(scenario, self, n, random_key,
                                                                   initial_state, initial_extra, **kwargs)

        return initial_state, initial_extra

    def clean_chain(self,
                    scenario: Scenario,
                    chain_state: cdict) -> cdict:
        return self.correction.clean_chain(scenario, self, chain_state)

    def always(self,
               scenario: Scenario,
               reject_state: cdict,
               reject_extra: cdict) -> Tuple[cdict, cdict]:
        return reject_state, reject_extra

    def proposal(self,
                 scenario: Scenario,
                 reject_state: cdict,
                 reject_extra: cdict) -> Tuple[cdict, cdict]:
        raise NotImplementedError(f'{self.__class__.__name__} markov proposal not initiated')

    def proposal_potential(self,
                           scenario: Scenario,
                           reject_state: cdict, reject_extra: cdict,
                           proposed_state: cdict, proposed_extra: cdict) -> Union[float, np.ndarray]:
        raise AttributeError(f'{self.__class__.__name__} proposal_potential not initiated')

    def acceptance_probability(self,
                               scenario: Scenario,
                               reject_state: cdict, reject_extra: cdict,
                               proposed_state: cdict, proposed_extra: cdict) -> Union[float, np.ndarray]:
        raise AttributeError(f'{self.__class__.__name__} acceptance_probability not initiated')

    def update(self,
               scenario: Scenario,
               previous_state: cdict,
               previous_extra: cdict) -> Tuple[cdict, cdict]:

        previous_extra.iter += 1

        reject_state, reject_extra = self.always(scenario,
                                                 previous_state,
                                                 previous_extra)

        proposed_state, proposed_extra = self.proposal(scenario,
                                                       reject_state,
                                                       reject_extra)

        corrected_state, corrected_extra = self.correction(scenario, self,
                                                           reject_state, reject_extra,
                                                           proposed_state, proposed_extra)

        return corrected_state, corrected_extra


class Correction:

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __repr__(self):
        return f'mocat.Correction.{self.__class__.__name__}'

    def startup(self,
                scenario: Scenario,
                sampler: MCMCSampler,
                n: int,
                random_key: np.ndarray,
                initial_state: cdict,
                initial_extra: cdict,
                **kwargs) -> Tuple[cdict, cdict]:
        # Update kwargs
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return initial_state, initial_extra

    def correct(self,
                scenario: Scenario,
                sampler: MCMCSampler,
                reject_state: cdict,
                reject_extra: cdict,
                proposed_state: cdict,
                proposed_extra: cdict) -> Tuple[cdict, cdict]:
        raise NotImplementedError

    def __call__(self,
                 scenario: Scenario,
                 sampler: MCMCSampler,
                 reject_state: cdict,
                 reject_extra: cdict,
                 proposed_state: cdict,
                 proposed_extra: cdict) -> Tuple[cdict, cdict]:
        return self.correct(scenario, sampler,
                            reject_state, reject_extra,
                            proposed_state, proposed_extra)

    def clean_chain(self,
                    scenario: Scenario,
                    sampler: Sampler,
                    chain_state: cdict) -> cdict:
        return chain_state


class Uncorrected(Correction):

    def correct(self,
                scenario: Scenario,
                sampler: MCMCSampler,
                reject_state: cdict,
                reject_extra: cdict,
                proposed_state: cdict,
                proposed_extra: cdict) -> Tuple[cdict, cdict]:
        return proposed_state, proposed_extra


def check_correction(correction: Union[None, Correction, Type[Correction]]) -> Correction:
    if correction is None:
        correction = Uncorrected()
    elif isclass(correction) and issubclass(correction, Correction):
        correction = correction()
    elif not isinstance(correction, Correction):
        raise TypeError(f'Correction must be of type mocat.Correction')

    return correction
