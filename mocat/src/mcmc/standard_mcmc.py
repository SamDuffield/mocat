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
    random_keys_per_iter = 2

    def __init__(self,
                 stepsize: float = None):
        super().__init__()
        self.parameters.stepsize = stepsize
        self.tuning.target = 0.234

    def startup(self,
                scenario: Scenario,
                n: int,
                initial_state: cdict = None,
                initial_extra: cdict = None,
                **kwargs) -> Tuple[cdict, cdict]:
        initial_state, initial_extra = super().startup(scenario, n,
                                                       initial_state, initial_extra, **kwargs)
        initial_state.potential = scenario.potential(initial_state.value, initial_extra.random_keys[1])
        return initial_state, initial_extra

    def proposal(self,
                 scenario: Scenario,
                 reject_state: cdict,
                 reject_extra: cdict) -> Tuple[cdict, cdict]:
        proposed_state = reject_state.copy()

        d = scenario.dim
        x = reject_state.value

        stepsize = reject_extra.parameters.stepsize

        proposed_state.value = x + jnp.sqrt(stepsize) * random.normal(reject_extra.random_keys[0], (d,))
        proposed_state.potential = scenario.potential(proposed_state.value, reject_extra.random_keys[1])

        return proposed_state, reject_extra

    def acceptance_probability(self,
                               scenario: Scenario,
                               reject_state: cdict, reject_extra: cdict,
                               proposed_state: cdict, proposed_extra: cdict) -> Union[float, jnp.ndarray]:
        return jnp.minimum(1., jnp.exp(- proposed_state.potential
                                       + reject_state.potential))


# Euler-Maruyama discretisation of Overdamped Langevin dynamics
# Identity pre-conditioner
class Overdamped(MCMCSampler):
    name = 'Overdamped'
    correction = Metropolis
    random_keys_per_iter = 2

    def __init__(self,
                 stepsize: float = None):
        super().__init__()
        self.parameters.stepsize = stepsize
        self.tuning.target = 0.574

    def startup(self,
                scenario: Scenario,
                n: int,
                initial_state: cdict = None,
                initial_extra: cdict = None,
                **kwargs) -> Tuple[cdict, cdict]:
        initial_state, initial_extra = super().startup(scenario, n,
                                                       initial_state, initial_extra, **kwargs)
        initial_state.potential, initial_state.grad_potential = scenario.potential_and_grad(initial_state.value,
                                                                                            initial_extra.random_keys[
                                                                                                1])
        return initial_state, initial_extra

    def proposal(self,
                 scenario: Scenario,
                 reject_state: cdict,
                 reject_extra: cdict) -> Tuple[cdict, cdict]:
        proposed_state = reject_state.copy()

        d = scenario.dim
        x = reject_state.value
        x_grad_potential = reject_state.grad_potential

        stepsize = reject_extra.parameters.stepsize

        proposed_state.value = x \
                               - stepsize * x_grad_potential \
                               + jnp.sqrt(2 * stepsize) * random.normal(reject_extra.random_keys[0], (d,))
        proposed_state.potential, proposed_state.grad_potential \
            = scenario.potential_and_grad(proposed_state.value,
                                          reject_extra.random_keys[1])

        return proposed_state, reject_extra

    def proposal_potential(self,
                           scenario: Scenario,
                           reject_state: cdict, reject_extra: cdict,
                           proposed_state: cdict, proposed_extra: cdict) -> Union[float, jnp.ndarray]:
        stepsize = reject_extra.parameters.stepsize

        return utils.gaussian_potential(proposed_state.value,
                                        reject_state.value - stepsize * reject_state.grad_potential,
                                        1. / (2 * stepsize))

    acceptance_probability = mh_acceptance_probability


# Hamiltonian Monte Carlo
# Identity pre-conditioner/mass matrix
class HMC(MCMCSampler):
    name = 'HMC'
    correction = Metropolis
    random_keys_per_iter = 2

    def __init__(self,
                 stepsize: float = None,
                 leapfrog_steps: int = None):
        super().__init__()
        self.parameters.stepsize = stepsize
        self.parameters.leapfrog_steps = leapfrog_steps
        self.tuning.target = 0.651

    @property
    def random_keys_per_iter(self):
        return self.parameters.leapfrog_steps + 1

    def startup(self,
                scenario: Scenario,
                n: int,
                initial_state: cdict = None,
                initial_extra: cdict = None,
                **kwargs) -> Tuple[cdict, cdict]:
        initial_state, initial_extra = super().startup(scenario, n,
                                                       initial_state, initial_extra, **kwargs)
        initial_state.potential, initial_state.grad_potential \
            = scenario.potential_and_grad(initial_state.value,
                                          initial_extra.random_keys[1])
        if not hasattr(initial_state, 'momenta') or initial_state.momenta.shape[-1] != scenario.dim:
            initial_state.momenta = jnp.zeros(scenario.dim)
        return initial_state, initial_extra

    def proposal(self,
                 scenario: Scenario,
                 reject_state: cdict,
                 reject_extra: cdict) -> Tuple[cdict, cdict]:
        d = scenario.dim

        reject_state.momenta = random.normal(reject_extra.random_keys[0], (d,))

        all_leapfrog_state = utils.leapfrog(scenario.potential_and_grad,
                                            reject_state,
                                            reject_extra.parameters.stepsize,
                                            reject_extra.random_keys[1:(self.parameters.leapfrog_steps + 1)])
        proposed_state = all_leapfrog_state[-1]

        # Technically we should reverse momenta now
        # but momenta target is symmetric and then immediately resampled at the next step anyway

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


# Underdamped Langevin
# Identity pre-conditioner/mass matrix
class Underdamped(MCMCSampler):
    name = 'Underdamped'
    correction = Metropolis
    random_keys_per_iter = 2

    def __init__(self,
                 stepsize: float = None,
                 leapfrog_steps: int = 1,
                 friction: float = 1.0):
        super().__init__()
        self.parameters.stepsize = stepsize
        self.parameters.leapfrog_steps = leapfrog_steps
        self.parameters.friction = friction
        self.tuning.target = 0.651

    @property
    def random_keys_per_iter(self):
        return self.parameters.leapfrog_steps + 1

    def startup(self,
                scenario: Scenario,
                n: int,
                initial_state: cdict = None,
                initial_extra: cdict = None,
                **kwargs) -> Tuple[cdict, cdict]:
        initial_state, initial_extra = super().startup(scenario, n,
                                                       initial_state, initial_extra, **kwargs)
        initial_state.potential, initial_state.grad_potential = scenario.potential_and_grad(initial_state.value,
                                                                                            initial_extra.random_keys[
                                                                                                1])
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

        reject_state.momenta = reject_state.momenta * -1

        # Update p - exactly according to solution of OU process
        # Accepted even if leapfrog step is rejected
        reject_state.momenta = reject_state.momenta * jnp.exp(- friction * stepsize) \
                               + jnp.sqrt(1 - jnp.exp(- 2 * friction * stepsize)) \
                               * random.normal(reject_extra.random_keys[0], (d,))
        return reject_state, reject_extra

    def proposal(self,
                 scenario: Scenario,
                 reject_state: cdict,
                 reject_extra: cdict) -> Tuple[cdict, cdict]:

        all_leapfrog_state = utils.leapfrog(scenario.potential_and_grad,
                                            reject_state,
                                            reject_extra.parameters.stepsize,
                                            reject_extra.random_keys[1:self.parameters.leapfrog_steps + 1])
        proposed_state = all_leapfrog_state[-1]

        proposed_state.momenta *= -1

        return proposed_state, reject_extra

    acceptance_probability = HMC.acceptance_probability
