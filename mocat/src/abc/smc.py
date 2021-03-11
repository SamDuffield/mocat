########################################################################################################################
# Module: abc/smc.py
# Description: SMC sampler for ABC problems.
#
# Web: https://github.com/SamDuffield/mocat
########################################################################################################################

from typing import Union, Tuple, Type
from inspect import isclass

from jax import numpy as jnp, random, vmap
from jax.lax import scan

from mocat.src.abc.abc import ABCScenario, ABCSampler
from mocat.src.abc.mcmc import ABCMCMCSampler, RandomWalkABC
from mocat.src.core import cdict, is_implemented
from mocat.src.transport.smc import SMCSampler
from mocat.src.mcmc.sampler import Correction, check_correction


class ABCSMCSampler(ABCSampler, SMCSampler):
    name = "ABC SMC"

    def __init__(self,
                 threshold_schedule: Union[None, jnp.ndarray] = None,
                 max_iter: int = int(1e4),
                 **kwargs):
        self.max_iter = max_iter
        self.threshold_schedule = threshold_schedule
        super().__init__(**kwargs)

    def __setattr__(self, key, value):
        super().__setattr__(key, value)
        if key == 'threshold_schedule':
            if value is None:
                self.next_threshold = self.next_threshold_adaptive
                self.max_iter = int(1e4)
            else:
                self.next_threshold = lambda state, extra: self.threshold_schedule[extra.iter]
                self.max_iter = len(value)

    def startup(self,
                abc_scenario: ABCScenario,
                n: int,
                initial_state: cdict,
                initial_extra: cdict,
                **kwargs) -> Tuple[cdict, cdict]:

        initial_state, initial_extra = SMCSampler.startup(self, abc_scenario, n,
                                                          initial_state, initial_extra, **kwargs)

        n = len(initial_state.value)
        if not hasattr(initial_state, 'prior_potential') and is_implemented(abc_scenario.prior_potential):
            random_keys = random.split(initial_extra.random_key, n + 1)
            initial_extra.random_key = random_keys[-1]
            initial_state.prior_potential = vmap(abc_scenario.prior_potential)(initial_state.value,
                                                                               random_keys[:n])

        if not hasattr(initial_state, 'simulated_data'):
            random_keys = random.split(initial_extra.random_key, n + 1)
            initial_extra.random_key = random_keys[-1]
            initial_state.simulated_data = vmap(abc_scenario.likelihood_sample)(initial_state.value,
                                                                                random_keys[:n])

        if not hasattr(initial_state, 'distance'):
            initial_state.distance = vmap(abc_scenario.distance_function)(initial_state.simulated_data)

        if not hasattr(initial_state, 'threshold'):
            if self.threshold_schedule is None:
                initial_state.threshold = jnp.zeros(n) + jnp.inf
            else:
                initial_state.threshold = jnp.zeros(n) + self.threshold_schedule[0]

        return initial_state, initial_extra

    def next_threshold_adaptive(self,
                                state: cdict,
                                extra: cdict) -> float:
        raise AttributeError(f'{self.name} next_threshold_adaptive not initiated')

    def clean_chain(self,
                    abc_scenario: ABCScenario,
                    chain_ensemble_state: cdict) -> cdict:
        chain_ensemble_state.threshold = chain_ensemble_state.threshold[:, 0]
        return chain_ensemble_state


def adapt_stepsize_scaled_diag_cov(ensemble_state: cdict,
                                   extra: cdict) -> Tuple[cdict, cdict]:
    n, d = ensemble_state.value.shape
    extra.parameters.stepsize = vmap(jnp.cov, (1,))(ensemble_state.value) / d * 2.38 ** 2
    return ensemble_state, extra


class MetropolisedABCSMCSampler(ABCSMCSampler):

    def __init__(self,
                 mcmc_sampler: Union[ABCMCMCSampler, Type[ABCMCMCSampler]] = None,
                 mcmc_correction: Union[Correction, Type[Correction], str] = 'sampler_default',
                 mcmc_steps: int = 1,
                 threshold_schedule: Union[None, jnp.ndarray] = None,
                 max_iter: int = int(1e4),
                 ess_threshold_retain: float = 0.9,
                 ess_threshold_resample: float = 0.5,
                 termination_alpha: float = 0.01,
                 **kwargs):
        super().__init__(max_iter=max_iter, threshold_schedule=threshold_schedule, **kwargs)
        if mcmc_sampler is None:
            mcmc_sampler = RandomWalkABC()
            self.adapt_mcmc_params = lambda ps, pe, ns, ne: adapt_stepsize_scaled_diag_cov(ns, ne)

        if isclass(mcmc_sampler):
            mcmc_sampler = mcmc_sampler()
        self.mcmc_sampler = mcmc_sampler
        if mcmc_correction != 'sampler_default':
            self.mcmc_sampler.correction = mcmc_correction
        self.parameters.mcmc_steps = mcmc_steps
        self.parameters.ess_threshold_retain = ess_threshold_retain
        self.parameters.ess_threshold_resample = ess_threshold_resample
        self.parameters.termination_alpha = termination_alpha

    def startup(self,
                abc_scenario: ABCScenario,
                n: int,
                initial_state: cdict,
                initial_extra: cdict,
                **kwargs) -> Tuple[cdict, cdict]:

        self.mcmc_sampler.correction = check_correction(self.mcmc_sampler.correction)

        initial_state, initial_extra = super().startup(abc_scenario, n, initial_state, initial_extra,
                                                       **kwargs)
        initial_extra.parameters.threshold = initial_state.threshold[0]

        initial_state, initial_extra = vmap(
            lambda state: self.mcmc_sampler.startup(abc_scenario,
                                                    initial_extra.parameters.mcmc_steps,
                                                    state,
                                                    initial_extra))(initial_state)
        initial_extra = initial_extra[0]

        initial_state, initial_extra = self.adapt(initial_state, initial_extra,
                                                  initial_state, initial_extra)
        return initial_state, initial_extra

    def resample_criterion(self,
                           ensemble_state: cdict,
                           extra: cdict) -> bool:
        return (ensemble_state.distance < extra.parameters.threshold).mean()\
               < extra.parameters.ess_threshold_resample

    def termination_criterion(self,
                              ensemble_state: cdict,
                              extra: cdict) -> bool:
        return jnp.logical_or(ensemble_state.alpha.mean() <= extra.parameters.termination_alpha,
                              extra.iter >= self.max_iter)

    def next_threshold_adaptive(self,
                                state: cdict,
                                extra: cdict) -> float:
        return jnp.quantile(state.distance, extra.parameters.ess_threshold_retain)

    def log_weight(self,
                   previous_ensemble_state: cdict,
                   previous_extra: cdict,
                   new_ensemble_state: cdict,
                   new_extra: cdict) -> jnp.ndarray:
        return jnp.where(new_ensemble_state.distance > new_extra.parameters.threshold, -jnp.inf, 0.)

    def clean_mcmc_chain(self,
                         chain_state: cdict,
                         chain_extra: cdict) -> Tuple[cdict, cdict]:
        clean_state = chain_state[-1]
        clean_extra = chain_extra[-1]
        clean_state.alpha = chain_state.alpha.mean()
        clean_extra.parameters = chain_extra.parameters[-1]
        return clean_state, clean_extra

    def forward_proposal(self,
                         abc_scenario: ABCScenario,
                         state: cdict,
                         extra: cdict,
                         random_key: jnp.ndarray) -> cdict:

        def mcmc_kernel(previous_carry: Tuple[cdict, cdict],
                        _: None) -> Tuple[Tuple[cdict, cdict], Tuple[cdict, cdict]]:
            new_carry = self.mcmc_sampler.update(abc_scenario, *previous_carry)
            return new_carry, new_carry

        extra.random_key = random_key

        start_state, start_extra = self.mcmc_sampler.startup(abc_scenario,
                                                             extra.parameters.mcmc_steps,
                                                             state,
                                                             extra)

        final_carry, chain = scan(mcmc_kernel,
                                  (start_state, start_extra),
                                  None,
                                  length=self.parameters.mcmc_steps)

        advanced_state, advanced_extra = self.clean_mcmc_chain(chain[0], chain[1])
        return advanced_state

    def adapt_mcmc_params(self,
                          previous_ensemble_state: cdict,
                          previous_extra: cdict,
                          new_ensemble_state: cdict,
                          new_extra: cdict) -> Tuple[cdict, cdict]:
        return new_ensemble_state, new_extra

    def adapt(self,
              previous_ensemble_state: cdict,
              previous_extra: cdict,
              new_ensemble_state: cdict,
              new_extra: cdict) -> Tuple[cdict, cdict]:
        n = new_ensemble_state.value.shape[0]
        new_extra.iter = previous_extra.iter
        next_threshold = self.next_threshold(new_ensemble_state, new_extra)
        new_ensemble_state.threshold = jnp.ones(n) * next_threshold
        new_extra.parameters.threshold = next_threshold
        new_ensemble_state.log_weight = self.log_weight(previous_ensemble_state, previous_extra,
                                                        new_ensemble_state, new_extra)
        new_ensemble_state, new_extra = self.adapt_mcmc_params(previous_ensemble_state, previous_extra,
                                                               new_ensemble_state, new_extra)

        return new_ensemble_state, new_extra

