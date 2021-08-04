########################################################################################################################
# Module: mcmc/standard_mcmc.py
# Description: Some common MCMC proposals.
#
# Web: https://github.com/SamDuffield/mocat
########################################################################################################################

from typing import Tuple, Union

import jax.numpy as jnp
from jax import random

from mocat.src.core import cdict, Scenario
from mocat.src.mcmc.sampler import MCMCSampler
from mocat.src import utils
from mocat.src.mcmc.metropolis import Metropolis, mh_acceptance_probability


# Random-walk
# Identity pre-conditioner
class RandomWalk(MCMCSampler):
    name = 'Random Walk'
    correction = Metropolis

    def __init__(self,
                 stepsize: float = None):
        super().__init__()
        self.parameters.stepsize = stepsize
        self.tuning.target = 0.234

    def startup(self,
                scenario: Scenario,
                n: int,
                initial_state: cdict,
                initial_extra: cdict,
                **kwargs) -> Tuple[cdict, cdict]:
        initial_state, initial_extra = super().startup(scenario, n,
                                                       initial_state, initial_extra, **kwargs)
        initial_extra.random_key, scen_key = random.split(initial_extra.random_key)
        initial_state.potential = scenario.potential(initial_state.value, scen_key)
        return initial_state, initial_extra

    def proposal(self,
                 scenario: Scenario,
                 reject_state: cdict,
                 reject_extra: cdict) -> Tuple[cdict, cdict]:
        proposed_state = reject_state.copy()

        d = scenario.dim
        x = reject_state.value

        stepsize = reject_extra.parameters.stepsize

        reject_extra.random_key, subkey, scen_key = random.split(reject_extra.random_key, 3)

        proposed_state.value = x + jnp.sqrt(stepsize) * random.normal(subkey, (d,))
        proposed_state.potential = scenario.potential(proposed_state.value, scen_key)

        return proposed_state, reject_extra

    def acceptance_probability(self,
                               scenario: Scenario,
                               reject_state: cdict, reject_extra: cdict,
                               proposed_state: cdict, proposed_extra: cdict) -> Union[float, jnp.ndarray]:
        return jnp.minimum(1., jnp.exp(- proposed_state.potential + reject_state.potential))


# Underdamped Langevin
# Identity pre-conditioner/mass matrix
# Recover HMC with friction = jnp.inf
# Recover Overdamped with friction = jnp.inf and leapfrog_steps = 1
class Underdamped(MCMCSampler):
    name = 'Underdamped'
    correction = Metropolis

    def __init__(self,
                 stepsize: float = None,
                 leapfrog_steps: int = 1,
                 friction: float = jnp.inf):
        super().__init__()
        self.parameters.stepsize = stepsize
        self.parameters.leapfrog_steps = leapfrog_steps
        self.parameters.friction = friction
        self.tuning.target = 0.651

    def startup(self,
                scenario: Scenario,
                n: int,
                initial_state: cdict,
                initial_extra: cdict,
                **kwargs) -> Tuple[cdict, cdict]:
        initial_state, initial_extra = super().startup(scenario, n,
                                                       initial_state, initial_extra, **kwargs)
        initial_extra.random_key, prior_key, lik_key = random.split(initial_extra.random_key, 3)
        initial_state.prior_potential, initial_state.grad_prior_potential\
            = scenario.prior_potential_and_grad(initial_state.value, prior_key)
        initial_state.likelihood_potential, initial_state.grad_likelihood_potential\
            = scenario.likelihood_potential_and_grad(initial_state.value, prior_key)
        initial_state.potential\
            = initial_state.prior_potential + scenario.temperature * initial_state.likelihood_potential
        initial_state.grad_potential\
            = initial_state.grad_prior_potential + scenario.temperature * initial_state.grad_likelihood_potential
        if not hasattr(initial_state, 'momenta') or initial_state.momenta.shape[-1] != scenario.dim:
            initial_state.momenta = jnp.zeros(scenario.dim)
        return initial_state, initial_extra

    def always(self,
               scenario: Scenario,
               reject_state: cdict,
               reject_extra: cdict) -> Tuple[cdict, cdict]:
        d = scenario.dim

        stepsize = reject_extra.parameters.stepsize
        friction = reject_extra.parameters.friction

        reject_state.momenta = - reject_state.momenta

        # Update p - exactly according to solution of OU process
        # Accepted even if leapfrog step is rejected
        reject_extra.random_key, subkey = random.split(reject_extra.random_key)
        reject_state.momenta = reject_state.momenta * jnp.exp(- friction * stepsize) \
                               + jnp.sqrt(1 - jnp.exp(- 2 * friction * stepsize)) * random.normal(subkey, (d,))
        return reject_state, reject_extra

    def proposal(self,
                 scenario: Scenario,
                 reject_state: cdict,
                 reject_extra: cdict) -> Tuple[cdict, cdict]:
        random_keys = random.split(reject_extra.random_key, self.parameters.leapfrog_steps + 1)

        reject_extra.random_key = random_keys[0]

        all_leapfrog_state = utils.leapfrog(scenario.prior_potential_and_grad,
                                            scenario.likelihood_potential_and_grad,
                                            reject_state,
                                            reject_extra.parameters.stepsize,
                                            random_keys[1:],
                                            scenario.temperature)
        proposed_state = all_leapfrog_state[-1]

        proposed_state.momenta = - proposed_state.momenta

        return proposed_state, reject_extra

    def acceptance_probability(self,
                               scenario: Scenario,
                               reject_state: cdict, reject_extra: cdict,
                               proposed_state: cdict, proposed_extra: cdict) -> Union[float, jnp.ndarray]:
        pre_min_alpha = jnp.exp(- proposed_state.potential
                                + reject_state.potential
                                - utils.gaussian_potential(proposed_state.momenta)
                                + utils.gaussian_potential(reject_state.momenta))
        return jnp.minimum(1., pre_min_alpha)

