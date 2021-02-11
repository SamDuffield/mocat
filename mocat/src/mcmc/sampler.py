########################################################################################################################
# Module: mcmc/sampler.py
# Description: Generic MCMC sampler class.
#
# Web: https://github.com/SamDuffield/mocat
########################################################################################################################

from typing import Tuple, Union, Type
from inspect import isclass

from jax import numpy as jnp, random

from mocat.src.core import Scenario, cdict, is_implemented
from mocat.src.sample import Sampler


class MCMCSampler(Sampler):
    correction: 'Correction' = None
    random_keys_per_iter: int = 2
    random_key_shape_per_iter = (1,)

    def __init__(self,
                 **kwargs):

        if not hasattr(self, 'tuning'):
            self.tuning = cdict(parameter='stepsize',
                                target=None,
                                metric='alpha',
                                monotonicity='decreasing')

        # Initiate sampler class (set any additional parameters from init)
        super().__init__(**kwargs)

    @property
    def random_key_shape_per_iter(self) -> Union[jnp.ndarray, tuple, int]:
        return self.random_keys_per_iter if self.correction is None\
            else self.random_keys_per_iter + self.correction.random_keys_per_iter

    def startup(self,
                scenario: Scenario,
                n: int,
                initial_state: cdict,
                initial_extra: cdict,
                startup_correction: bool = True,
                **kwargs) -> Tuple[cdict, cdict]:
        self.max_iter = n - 1
        initial_extra.iter = 0

        if initial_state is None:
            if is_implemented(scenario.prior_sample):
                init_vals = scenario.prior_sample(initial_extra.random_keys[0])
            else:
                init_vals = jnp.zeros(scenario.dim)
            initial_state = cdict(value=init_vals)

        if 'correction' in kwargs.keys():
            self.correction = kwargs['correction']
            del kwargs['correction']

        self.correction = check_correction(self.correction)

        initial_state, initial_extra = super().startup(scenario, n, initial_state, initial_extra, **kwargs)

        if startup_correction:
            initial_state, initial_extra = self.correction.startup(scenario, self, n,
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
                           proposed_state: cdict, proposed_extra: cdict) -> Union[float, jnp.ndarray]:
        raise AttributeError(f'{self.__class__.__name__} proposal_potential not initiated')

    def acceptance_probability(self,
                               scenario: Scenario,
                               reject_state: cdict, reject_extra: cdict,
                               proposed_state: cdict, proposed_extra: cdict) -> Union[float, jnp.ndarray]:
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
    random_keys_per_iter = 0

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __repr__(self):
        return f'mocat.Correction.{self.__class__.__name__}'

    def startup(self,
                scenario: Scenario,
                sampler: MCMCSampler,
                n: int,
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
