########################################################################################################################
# Module: mcmc/ensemble_mcmc.py
# Description: MCMC which uses an ensemble to adapt the proposal covariance.
#
# Web: https://github.com/SamDuffield/mocat
########################################################################################################################

from typing import Tuple, Union

import jax.numpy as np
from jax import random, vmap
from jax.lax import cond
from jax.ops import index_update

from mocat.src.mcmc.sampler import MCMCSampler
from mocat.src import utils
from mocat.src.core import Scenario, CDict
from mocat.src.mcmc.corrections import Metropolis


# Ensemble Random Walk with sample covariance preconditioner
class EnsembleRWMH(MCMCSampler):
    name = 'Ensemble Random Walk'
    default_correction = Metropolis()

    def __init__(self,
                 stepsize: float = None,
                 n_ensemble: int = None,
                 identity_scaling: float = 0.5,
                 samp_cov_scaling: float = 0.5):
        super().__init__()
        self.parameters.stepsize = stepsize
        self.parameters.n_ensemble = n_ensemble
        self.parameters.identity_scaling = identity_scaling
        self.parameters.samp_cov_scaling = samp_cov_scaling
        self.tuning.target = 0.234

    def startup(self,
                scenario: Scenario,
                initial_state: CDict = None,
                initial_extra: CDict = None,
                random_key: np.ndarray = None) -> Tuple[CDict, CDict]:
        _, initial_extra = super().startup(scenario, False, initial_extra, random_key)
        if initial_state is None:
            random_key, sub_key = random.split(random_key)
            x0 = random.normal(sub_key, shape=(self.parameters.n_ensemble, scenario.dim))
            initial_state = CDict(value=x0)
        return initial_state, initial_extra

    def proposal(self,
                 scenario: Scenario,
                 reject_state: CDict,
                 reject_extra: CDict) -> Tuple[CDict, CDict]:
        proposed_state = reject_state.copy()

        d = scenario.dim
        n_ensemble = self.parameters.n_ensemble
        ensemble = reject_state.value
        ensemble_index = reject_extra.iter % n_ensemble

        stepsize = reject_extra.parameters.stepsize
        identity_scaling = reject_extra.parameters.identity_scaling
        samp_cov_scaling = reject_extra.parameters.samp_cov_scaling

        leave_one_out_ensemble = np.where(np.expand_dims(np.arange(n_ensemble - 1), 1) < ensemble_index,
                                          ensemble[:-1],
                                          ensemble[1:])
        leave_one_out_cov = samp_cov_scaling * np.atleast_2d(np.cov(leave_one_out_ensemble.T))\
                            + identity_scaling * np.eye(d)
        leave_one_out_cov_sqrt = np.linalg.cholesky(leave_one_out_cov)

        reject_extra.random_key, subkey = random.split(reject_extra.random_key)

        precon_noise = leave_one_out_cov_sqrt @ random.normal(subkey, (d,))

        proposed_state.value = index_update(proposed_state.value, ensemble_index,
                                            reject_state.value[ensemble_index] + np.sqrt(stepsize) * precon_noise)

        return proposed_state, reject_extra

    def acceptance_probability(self,
                               scenario: Scenario,
                               reject_state: CDict, reject_extra: CDict,
                               proposed_state: CDict, proposed_extra: CDict) -> Union[float, np.ndarray]:
        ensemble_index = reject_extra.iter % self.parameters.n_ensemble
        return np.minimum(1., np.exp(- proposed_state.potential[ensemble_index]
                                     + reject_state.potential[ensemble_index]))


# Ensemble Overdamped Langevin with sample covariance preconditioner
class EnsembleOverdamped(MCMCSampler):
    name = 'Ensemble Overdamped'
    default_correction = Metropolis()

    def __init__(self,
                 stepsize: float = None,
                 n_ensemble: int = None,
                 identity_scaling: float = 0.5,
                 samp_cov_scaling: float = 0.5):
        super().__init__()
        self.parameters.stepsize = stepsize
        self.parameters.n_ensemble = n_ensemble
        self.parameters.identity_scaling = identity_scaling
        self.parameters.samp_cov_scaling = samp_cov_scaling
        self.tuning.target = 0.574

    def startup(self,
                scenario: Scenario,
                initial_state: CDict = None,
                initial_extra: CDict = None,
                random_key: np.ndarray = None) -> Tuple[CDict, CDict]:
        _, initial_extra = super().startup(scenario, False, initial_extra, random_key)
        if initial_state is None:
            random_key, sub_key = random.split(random_key)
            x0 = random.normal(sub_key, shape=(self.parameters.n_ensemble, scenario.dim))
            initial_state = CDict(value=x0)
        initial_state.grad_potential = vmap(scenario.grad_potential)(initial_state.value)
        initial_state, initial_extra = self.always(scenario, initial_state, initial_extra)
        return initial_state, initial_extra

    def always(self,
               scenario: Scenario,
               reject_state: CDict,
               reject_extra: CDict) -> Tuple[CDict, CDict]:
        d = scenario.dim
        n_ensemble = self.parameters.n_ensemble
        ensemble = reject_state.value
        ensemble_index = reject_extra.iter % n_ensemble

        identity_scaling = reject_extra.parameters.identity_scaling
        samp_cov_scaling = reject_extra.parameters.samp_cov_scaling

        leave_one_out_ensemble = np.where(np.expand_dims(np.arange(n_ensemble - 1), 1) < ensemble_index,
                                          ensemble[:-1],
                                          ensemble[1:])
        leave_one_out_cov = samp_cov_scaling * np.atleast_2d(
            np.cov(leave_one_out_ensemble.T)) + identity_scaling * np.eye(d)
        reject_extra.leave_one_out_cov = leave_one_out_cov
        reject_extra.leave_one_out_prec = np.linalg.inv(leave_one_out_cov)
        reject_extra.leave_one_out_cov_sqrt = np.linalg.cholesky(leave_one_out_cov)
        return reject_state, reject_extra

    def proposal(self,
                 scenario: Scenario,
                 reject_state: CDict,
                 reject_extra: CDict) -> Tuple[CDict, CDict]:
        proposed_state = reject_state.copy()

        d = scenario.dim
        n_ensemble = self.parameters.n_ensemble
        ensemble_index = reject_extra.iter % n_ensemble

        stepsize = reject_extra.parameters.stepsize

        leave_one_out_cov = reject_extra.leave_one_out_cov
        leave_one_out_cov_sqrt = reject_extra.leave_one_out_cov_sqrt

        reject_extra.random_key, subkey = random.split(reject_extra.random_key)

        precon_noise = leave_one_out_cov_sqrt @ random.normal(subkey, (d,))

        proposed_state.value = index_update(reject_state.value, ensemble_index,
                                            reject_state.value[ensemble_index]
                                            - stepsize * leave_one_out_cov @ reject_state.grad_potential[ensemble_index]
                                            + np.sqrt(2 * stepsize) * precon_noise)

        proposed_state.grad_potential = index_update(proposed_state.grad_potential, ensemble_index,
                                                     scenario.grad_potential(proposed_state.value[ensemble_index]))

        reject_extra.leave_one_out_cov = leave_one_out_cov

        return proposed_state, reject_extra

    def proposal_potential(self,
                           scenario: Scenario,
                           reject_state: CDict, reject_extra: CDict,
                           proposed_state: CDict, proposed_extra: CDict) -> Union[float, np.ndarray]:
        ensemble_index = reject_extra.iter % self.parameters.n_ensemble
        stepsize = reject_extra.parameters.stepsize
        leave_one_out_cov = reject_extra.leave_one_out_cov
        leave_one_out_prec = reject_extra.leave_one_out_prec

        return utils.gaussian_potential(proposed_state.value[ensemble_index],
                                        reject_state.value[ensemble_index]
                                        - stepsize * leave_one_out_cov @
                                        reject_state.grad_potential[ensemble_index],
                                        leave_one_out_prec / (2 * stepsize))

    def acceptance_probability(self,
                               scenario: Scenario,
                               reject_state: CDict, reject_extra: CDict,
                               proposed_state: CDict, proposed_extra: CDict) -> Union[float, np.ndarray]:
        ensemble_index = reject_extra.iter % self.parameters.n_ensemble

        pre_min_alpha = np.exp(- proposed_state.potential[ensemble_index]
                               + reject_state.potential[ensemble_index]
                               - self.proposal_potential(scenario,
                                                         proposed_state, proposed_extra,
                                                         reject_state, reject_extra)
                               + self.proposal_potential(scenario,
                                                         reject_state, reject_extra,
                                                         proposed_state, proposed_extra))

        return np.minimum(1., pre_min_alpha)
