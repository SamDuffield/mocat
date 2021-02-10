########################################################################################################################
# Module: transport/sampler.py
# Description: Generic transport sampler class - iteratively moves an ensemble of particles.
#
# Web: https://github.com/SamDuffield/mocat
########################################################################################################################

from typing import Tuple

from jax import numpy as jnp, random, vmap

from mocat.src.core import Scenario, cdict, is_implemented
from mocat.src.sample import Sampler


class TransportSampler(Sampler):

    def startup(self,
                scenario: Scenario,
                n: int,
                random_key: jnp.ndarray = None,
                initial_state: cdict = None,
                initial_extra: cdict = None,
                **kwargs) -> Tuple[cdict, cdict]:
        if initial_state is None:
            random_key, sub_key = random.split(random_key)
            if is_implemented(scenario.prior_sample):
                init_vals = vmap(scenario.prior_sample)(random.split(sub_key, n))
            else:
                init_vals = random.normal(sub_key, shape=(n, scenario.dim))
            initial_state = cdict(value=init_vals)

        if initial_extra is None:
            initial_extra = cdict(random_key=random.split(random_key, n),
                                  iter=jnp.zeros(n, dtype='int32'))

        initial_state, initial_extra = super().startup(scenario, n, random_key, initial_state, initial_extra, **kwargs)

        return initial_state, initial_extra

    def update(self,
               scenario: Scenario,
               ensemble_state: cdict,
               ensemble_extra: cdict) -> Tuple[cdict, cdict]:
        raise NotImplementedError(f'{self.name} update not initiated')

    def clean_chain(self,
                    scenario: Scenario,
                    chain_ensemble_state: cdict) -> cdict:
        return chain_ensemble_state

    def termination_criterion(self,
                              ensemble_state: cdict,
                              ensemble_extra: cdict) -> bool:
        return ensemble_extra.iter[0] >= self.max_iter

