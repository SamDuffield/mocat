########################################################################################################################
# Module: transport/sampler.py
# Description: Generic transport sampler class - iteratively moves an ensemble of particles.
#
# Web: https://github.com/SamDuffield/mocat
########################################################################################################################

from typing import Tuple

from jax import random, vmap

from mocat.src.core import Scenario, cdict, is_implemented
from mocat.src.sample import Sampler


class TransportSampler(Sampler):

    def startup(self,
                scenario: Scenario,
                n: int,
                initial_state: cdict,
                initial_extra: cdict,
                **kwargs) -> Tuple[cdict, cdict]:
        if initial_state is None:
            initial_extra.random_key, sub_key = random.split(initial_extra.random_key)
            if is_implemented(scenario.prior_sample):
                init_vals = vmap(scenario.prior_sample)(random.split(sub_key, n))
            else:
                init_vals = random.normal(sub_key, shape=(n, scenario.dim))
            initial_state = cdict(value=init_vals)

        initial_state, initial_extra = super().startup(scenario, n, initial_state, initial_extra, **kwargs)

        return initial_state, initial_extra

    def update(self,
               scenario: Scenario,
               ensemble_state: cdict,
               extra: cdict) -> Tuple[cdict, cdict]:
        raise NotImplementedError(f'{self.name} update not initiated')

    def clean_chain(self,
                    scenario: Scenario,
                    chain_ensemble_state: cdict) -> cdict:
        return chain_ensemble_state

    def termination_criterion(self,
                              ensemble_state: cdict,
                              extra: cdict) -> bool:
        return extra.iter >= self.max_iter

