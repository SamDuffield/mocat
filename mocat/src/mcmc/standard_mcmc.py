########################################################################################################################
# Module: mcmc/standard_mcmc.py
# Description: Some common MCMC proposals.
#
# Web: https://github.com/SamDuffield/mocat
########################################################################################################################

from typing import Tuple, Union
from functools import partial

import jax.numpy as np
from jax import random

from mocat.src.core import CDict, Scenario
from mocat.src.mcmc.sampler import MCMCSampler, mh_acceptance_probability
from mocat.src import utils
from mocat.src.mcmc.corrections import Metropolis


# Random-walk
# Identity pre-conditioner
class RandomWalk(MCMCSampler):
    name = 'Random Walk'
    default_correction = Metropolis()

    def __init__(self,
                 stepsize: float = None):
        super().__init__()
        self.parameters.stepsize = stepsize
        self.tuning.target = 0.234

    def proposal(self,
                        scenario: Scenario,
                        reject_state: CDict,
                        reject_extra: CDict) -> Tuple[CDict, CDict]:
        proposed_state = reject_state.copy()

        d = scenario.dim
        x = reject_state.value

        stepsize = reject_extra.parameters.stepsize

        reject_extra.random_key, subkey = random.split(reject_extra.random_key)

        proposed_state.value = x + np.sqrt(stepsize) * random.normal(subkey, (d,))

        return proposed_state, reject_extra

    def acceptance_probability(self,
                               scenario: Scenario,
                               reject_state: CDict, reject_extra: CDict,
                               proposed_state: CDict, proposed_extra: CDict) -> Union[float, np.ndarray]:
        return np.minimum(1., np.exp(- proposed_state.potential
                                     + reject_state.potential))


# Euler-Maruyama discretisation of Overdamped Langevin dynamics
# Identity pre-conditioner
class Overdamped(MCMCSampler):
    name = 'Overdamped'
    default_correction = Metropolis()

    def __init__(self,
                 stepsize: float = None):
        super().__init__()
        self.parameters.stepsize = stepsize
        self.tuning.target = 0.574

    def startup(self,
                scenario: Scenario,
                random_key: np.ndarray):
        super().startup(scenario, random_key)
        self.initial_state.grad_potential = scenario.grad_potential(self.initial_state.value)

    def proposal(self,
                        scenario: Scenario,
                        reject_state: CDict,
                        reject_extra: CDict) -> Tuple[CDict, CDict]:
        proposed_state = reject_state.copy()

        d = scenario.dim
        x = reject_state.value
        x_grad_potential = reject_state.grad_potential

        stepsize = reject_extra.parameters.stepsize

        reject_extra.random_key, subkey = random.split(reject_extra.random_key)

        proposed_state.value = x \
                               - stepsize * x_grad_potential \
                               + np.sqrt(2 * stepsize) * random.normal(subkey, (d,))
        proposed_state.grad_potential = scenario.grad_potential(proposed_state.value)

        return proposed_state, reject_extra

    def proposal_potential(self,
                           scenario: Scenario,
                           reject_state: CDict, reject_extra: CDict,
                           proposed_state: CDict, proposed_extra: CDict) -> Union[float, np.ndarray]:
        d = scenario.dim
        stepsize = reject_extra.parameters.stepsize

        return utils.gaussian_potential(proposed_state.value,
                                        reject_state.value - stepsize * reject_state.grad_potential,
                                        np.eye(d) / (2 * stepsize))

    acceptance_probability = mh_acceptance_probability


# Hamiltonian Monte Carlo
# Identity pre-conditioner/mass matrix
class HMC(MCMCSampler):
    name = 'HMC'
    default_correction = Metropolis()

    def __init__(self,
                 stepsize: float = None,
                 leapfrog_steps: int = None):
        super().__init__()
        self.parameters.stepsize = stepsize
        self.parameters.leapfrog_steps = leapfrog_steps
        self.tuning.target = 0.651

    def startup(self,
                scenario: Scenario,
                random_key: np.ndarray):
        super().startup(scenario, random_key)
        self.initial_state.grad_potential = scenario.grad_potential(self.initial_state.value)
        if not hasattr(self.initial_state, 'momenta') or self.initial_state.momenta.shape[-1] != scenario.dim:
            self.initial_state.momenta = np.zeros(scenario.dim)

    def proposal(self,
                        scenario: Scenario,
                        reject_state: CDict,
                        reject_extra: CDict) -> Tuple[CDict, CDict]:
        d = scenario.dim

        stepsize = reject_extra.parameters.stepsize

        reject_extra.random_key, subkey = random.split(reject_extra.random_key)

        reject_state.momenta = random.normal(subkey, (d,))

        proposed_state = utils.leapfrog(reject_state,
                                        scenario.grad_potential,
                                        stepsize,
                                        self.parameters.leapfrog_steps)[-1]

        # Techinically we should reverse momenta now
        # but momenta target is symmetric and then immediately resampled at the next step anyway

        return proposed_state, reject_extra

    def acceptance_probability(self,
                               scenario: Scenario,
                               reject_state: CDict, reject_extra: CDict,
                               proposed_state: CDict, proposed_extra: CDict) -> Union[float, np.ndarray]:
        pre_min_alpha = np.exp(- proposed_state.potential
                               + reject_state.potential
                               - utils.gaussian_potential(proposed_state.momenta)
                               + utils.gaussian_potential(reject_state.momenta))

        return np.minimum(1., pre_min_alpha)


# Underdamped Langevin
# Identity pre-conditioner/mass matrix
class Underdamped(MCMCSampler):
    name = 'Underdamped'
    default_correction = Metropolis()

    def __init__(self,
                 stepsize: float = None,
                 leapfrog_steps: int = 1,
                 friction: float = 1.0):
        super().__init__()
        self.parameters.stepsize = stepsize
        self.parameters.leapfrog_steps = leapfrog_steps
        self.parameters.friction = friction
        self.tuning.target = 0.651

    def startup(self,
                scenario: Scenario,
                random_key: np.ndarray):
        super().startup(scenario, random_key)
        self.initial_state.grad_potential = scenario.grad_potential(self.initial_state.value)
        if not hasattr(self.initial_state, 'momenta') or self.initial_state.momenta.shape[-1] != scenario.dim:
            self.initial_state.momenta = np.zeros(scenario.dim)

    def always(self,
               scenario: Scenario,
               reject_state: CDict,
               reject_extra: CDict) -> Tuple[CDict, CDict]:
        d = scenario.dim

        stepsize = reject_extra.parameters.stepsize
        friction = reject_extra.parameters.friction

        reject_state.momenta *= -1

        # Update p - exactly according to solution of OU process
        # Accepted even if leapfrog step is rejected
        reject_extra.random_key, subkey = random.split(reject_extra.random_key)
        reject_state.momenta = reject_state.momenta * np.exp(- friction * stepsize) \
                               + np.sqrt(1 - np.exp(- 2 * friction * stepsize)) * random.normal(subkey, (d,))
        return reject_state, reject_extra

    def proposal(self,
                        scenario: Scenario,
                        reject_state: CDict,
                        reject_extra: CDict) -> Tuple[CDict, CDict]:
        
        stepsize = reject_extra.parameters.stepsize
        
        proposed_state = utils.leapfrog(reject_state,
                                        scenario.grad_potential,
                                        stepsize,
                                        self.parameters.leapfrog_steps)[-1]
        proposed_state.momenta *= -1

        return proposed_state, reject_extra

    acceptance_probability = HMC.acceptance_probability


# Tamed Euler-Maruyama discretisation of Overdamped Langevin dynamics
# Identity pre-conditioner
class TamedOverdamped(MCMCSampler):
    name = 'Tamed Overdamped'
    default_correction = Metropolis()

    def __init__(self,
                 stepsize: float = None):
        super().__init__()
        self.parameters.stepsize = stepsize
        self.tuning.target = 0.574
        self.taming = self.taming_coord

    def startup(self,
                scenario: Scenario,
                random_key: np.ndarray):
        super().startup(scenario, random_key)
        self.initial_state.grad_potential = scenario.grad_potential(self.initial_state.value)

    def proposal(self,
                        scenario: Scenario,
                        reject_state: CDict,
                        reject_extra: CDict) -> Tuple[CDict, CDict]:
        proposed_state = reject_state.copy()

        d = scenario.dim
        x = reject_state.value
        x_grad_potential = reject_state.grad_potential

        stepsize = reject_extra.parameters.stepsize

        reject_extra.random_key, subkey = random.split(reject_extra.random_key)

        proposed_state.value = x \
                               - stepsize * x_grad_potential * self.taming(reject_state, reject_extra) \
                               + np.sqrt(2 * stepsize) * random.normal(subkey, (d,))
        proposed_state.grad_potential = scenario.grad_potential(proposed_state.value)

        return proposed_state, reject_extra

    @staticmethod
    def taming_global(state: CDict,
                      extra: CDict) -> Union[float, np.ndarray]:
        return 1. / (1 + extra.parameters.stepsize * np.linalg.norm(state.grad_potential))

    @staticmethod
    def taming_coord(state: CDict,
                     extra: CDict) -> Union[float, np.ndarray]:
        return 1. / (1 + extra.parameters.stepsize * np.abs(state.grad_potential))

    def proposal_potential(self,
                           scenario: Scenario,
                           reject_state: CDict, reject_extra: CDict,
                           proposed_state: CDict, proposed_extra: CDict) -> Union[float, np.ndarray]:
        d = scenario.dim
        stepsize = reject_extra.parameters.stepsize

        return utils.gaussian_potential(proposed_state.value,
                                        reject_state.value - stepsize * reject_state.grad_potential
                                        * self.taming(reject_state, reject_extra),
                                        np.eye(d) / (2 * stepsize))

    acceptance_probability = mh_acceptance_probability
