########################################################################################################################
# Module: abc/mcmc.py
# Description: Markovian ABC, accept if simulated data hits ball.
#
# Web: https://github.com/SamDuffield/mocat
########################################################################################################################
from typing import Union, Tuple

from jax import numpy as jnp, random
from mocat.src.abc.abc import ABCScenario, ABCSampler
from mocat.src.core import cdict
from mocat.src.mcmc.metropolis import Metropolis, RMMetropolis
from mocat.src.mcmc.sampler import MCMCSampler, Correction


class ABCMCMCSampler(ABCSampler, MCMCSampler):
    name = "ABC MCMC"
    correction: Correction = Metropolis
    tuning: cdict = None

    def __init__(self,
                 threshold: float = jnp.inf,
                 **kwargs):
        super().__init__(threshold=threshold, **kwargs)

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


class RandomWalkABC(ABCMCMCSampler):
    name = 'Random Walk ABC'
    correction = Metropolis

    def __init__(self,
                 threshold: float = None,
                 stepsize: float = None):
        super().__init__()
        self.parameters.threshold = threshold
        self.parameters.stepsize = stepsize

        self.tuning = cdict(parameter='threshold',
                            target=0.1,
                            metric='alpha',
                            monotonicity='increasing')

    def acceptance_probability(self,
                               abc_scenario: ABCScenario,
                               reject_state: cdict, reject_extra: cdict,
                               proposed_state: cdict, proposed_extra: cdict) -> Union[float, jnp.ndarray]:
        return jnp.minimum(1., jnp.exp(-proposed_state.prior_potential
                                     + reject_state.prior_potential)
                          * (proposed_state.distance < reject_extra.parameters.threshold))

    def proposal(self,
                 abc_scenario: ABCScenario,
                 reject_state: cdict,
                 reject_extra: cdict) -> Tuple[cdict, cdict]:
        proposed_state = reject_state.copy()
        proposed_extra = reject_extra.copy()
        stepsize = reject_extra.parameters.stepsize
        proposed_extra.random_key, subkey1, subkey2, subkey3 = random.split(reject_extra.random_key, 4)
        proposed_state.value = reject_state.value + jnp.sqrt(stepsize) * random.normal(subkey1, (abc_scenario.dim,))
        proposed_state.prior_potential = abc_scenario.prior_potential(proposed_state.value, subkey2)
        proposed_extra.simulated_data = abc_scenario.likelihood_sample(proposed_state.value, subkey3)
        proposed_state.distance = abc_scenario.distance_function(proposed_extra.simulated_data)
        return proposed_state, proposed_extra


class RMMetropolisDiagStepsize(RMMetropolis):

    def startup(self,
                abc_scenario: ABCScenario,
                sampler: MCMCSampler,
                n: int,
                random_key: jnp.ndarray,
                initial_state: cdict,
                initial_extra: cdict,
                **kwargs) -> Tuple[cdict, cdict]:
        initial_state, initial_extra = super().startup(abc_scenario, sampler, n, random_key,
                                                       initial_state, initial_extra, **kwargs)
        if sampler.parameters.stepsize is None:
            initial_extra.parameters.stepsize = 1.0
        if sampler.parameters.threshold is None:
            initial_extra.parameters.threshold = 50.
            initial_state.threshold = initial_extra.parameters.threshold
        initial_extra.parameters.stepsize = jnp.ones(abc_scenario.dim) * initial_extra.parameters.stepsize
        initial_state.stepsize = initial_extra.parameters.stepsize
        initial_extra.post_mean = initial_state.value
        initial_extra.diag_post_cov = initial_extra.parameters.stepsize * abc_scenario.dim / 2.38 ** 2
        return initial_state, initial_extra

    def adapt(self,
              state: cdict,
              extra: cdict) -> Tuple[cdict, cdict]:
        # Adapt threshold
        adapted_state, adapted_extra = super().adapt(state, extra)

        # Adapt stepsize to be 2.28^2 / d * diagonal covariance of samples
        rm_stepsize = self.rm_stepsize_scale * extra.iter ** -self.rm_stepsize_neg_exponent
        rm_stepsize = jnp.minimum(rm_stepsize, 1 - 1e-3)

        d = adapted_extra.diag_post_cov.size

        adapted_extra.diag_post_cov = (1 - rm_stepsize) * adapted_extra.diag_post_cov \
                                      + rm_stepsize * (adapted_state.value - adapted_extra.post_mean) ** 2

        adapted_extra.post_mean = (1 - rm_stepsize) * adapted_extra.post_mean + rm_stepsize * adapted_state.value

        adapted_extra.parameters.stepsize = adapted_extra.diag_post_cov / d * 2.38 ** 2
        adapted_state.stepsize = adapted_extra.parameters.stepsize
        return adapted_state, adapted_extra
