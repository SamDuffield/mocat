########################################################################################################################
# Module: abc/smc.py
# Description: SMC sampler for ABC problems.
#
# Web: https://github.com/SamDuffield/mocat
########################################################################################################################

from typing import Union, Tuple, Type
from inspect import isclass

from jax import numpy as jnp, random, vmap
from jax.lax import cond, scan

from mocat.src.abc.abc import ABCScenario, ABCSampler
from mocat.src.abc.mcmc import ABCMCMCSampler, RandomWalkABC
from mocat.src.core import cdict, is_implemented
from mocat.src.transport.smc import SMCSampler
from mocat.src.transport.sampler import TransportSampler
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
                self.next_threshold = lambda state, extra: self.threshold_schedule[extra.iter[0]]
                self.max_iter = len(value)

    def startup(self,
                abc_scenario: ABCScenario,
                n: int,
                random_key: jnp.ndarray = None,
                initial_state: cdict = None,
                initial_extra: cdict = None,
                **kwargs) -> Tuple[cdict, cdict]:

        initial_state, initial_extra = TransportSampler.startup(self, abc_scenario, n, random_key,
                                                                initial_state, initial_extra, **kwargs)

        n = len(initial_state.value)
        if not hasattr(initial_state, 'prior_potential') and is_implemented(abc_scenario.prior_potential):
            double_random_keys = random.split(initial_extra.random_key[0], 2*n)
            initial_extra.random_key = double_random_keys[:n]
            initial_state.prior_potential = vmap(abc_scenario.prior_potential)(initial_state.value,
                                                                               double_random_keys[n:])

        if not hasattr(initial_state, 'log_weight'):
            initial_state.log_weight = jnp.zeros(n)

        if not hasattr(initial_state, 'simulated_data'):
            double_random_keys = random.split(initial_extra.random_key[0], 2*n)
            initial_extra.random_key = double_random_keys[:n]
            initial_extra.simulated_data = vmap(abc_scenario.likelihood_sample)(initial_state.value,
                                                                                double_random_keys[n:])

        if not hasattr(initial_state, 'distance'):
            initial_state.distance = vmap(abc_scenario.distance_function)(initial_extra.simulated_data)

        initial_extra.parameters = vmap(lambda _: initial_extra.parameters)(jnp.arange(n))

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

    def forward_proposal(self,
                         abc_scenario: ABCScenario,
                         previous_state: cdict,
                         previous_extra: cdict) -> Tuple[cdict, cdict]:
        raise AttributeError(f'{self.name} forward_proposal not initiated')

    def clean_chain(self,
                    abc_scenario: ABCScenario,
                    chain_ensemble_state: cdict) -> cdict:
        chain_ensemble_state.threshold = chain_ensemble_state.threshold[:, 0]
        return chain_ensemble_state


def adapt_stepsize_scaled_diag_cov(ensemble_state: cdict,
                                   ensemble_extra: cdict) -> Tuple[cdict, cdict]:
    n, d = ensemble_state.value.shape
    stepsize = vmap(jnp.cov, (1,))(ensemble_state.value) / d * 2.38 ** 2
    ensemble_extra.parameters.stepsize = jnp.repeat(stepsize[jnp.newaxis], n, axis=0)
    return ensemble_state, ensemble_extra


class MetropolisedABCSMCSampler(ABCSMCSampler):

    def __init__(self,
                 mcmc_sampler: Union[ABCMCMCSampler, Type[ABCMCMCSampler]] = None,
                 mcmc_correction: Union[Correction, Type[Correction], str] = 'sampler_default',
                 mcmc_steps: int = 20,
                 threshold_schedule: Union[None, jnp.ndarray] = None,
                 max_iter: int = int(1e4),
                 threshold_quantile_retain: float = 0.75,
                 **kwargs):
        super().__init__(max_iter=max_iter, threshold_schedule=threshold_schedule, **kwargs)
        if mcmc_sampler is None:
            mcmc_sampler = RandomWalkABC()
            self.adapt = adapt_stepsize_scaled_diag_cov

        if isclass(mcmc_sampler):
            mcmc_sampler = mcmc_sampler()
        self.mcmc_sampler = mcmc_sampler
        if mcmc_correction != 'sampler_default':
            self.mcmc_sampler.correction = mcmc_correction
        self.parameters.mcmc_steps = mcmc_steps
        self.parameters.threshold_quantile_retain = threshold_quantile_retain

    def startup(self,
                abc_scenario: ABCScenario,
                n: int,
                random_key: jnp.ndarray = None,
                initial_state: cdict = None,
                initial_extra: cdict = None,
                **kwargs) -> Tuple[cdict, cdict]:

        self.mcmc_sampler.correction = check_correction(self.mcmc_sampler.correction)

        initial_state, initial_extra = super().startup(abc_scenario, n, random_key, initial_state, initial_extra,
                                                       **kwargs)

        initial_state, initial_extra = vmap(
            lambda state, extra: self.mcmc_sampler.startup(abc_scenario,
                                                           n,
                                                           None,
                                                           state,
                                                           extra))(initial_state, initial_extra)

        initial_threshold = self.next_threshold(initial_state, initial_extra)
        initial_state.threshold = jnp.zeros(n) + initial_threshold
        initial_extra.parameters.threshold = initial_state.threshold
        if not hasattr(initial_state, 'log_weight'):
            initial_state.log_weight = jnp.zeros(n)
        initial_state.log_weight = jnp.where(initial_state.distance < initial_threshold,
                                            initial_state.log_weight, -jnp.inf)
        initial_state, initial_extra = self.adapt(initial_state, initial_extra)
        return initial_state, initial_extra

    def termination_criterion(self,
                              ensemble_state: cdict,
                              ensemble_extra: cdict) -> bool:
        return jnp.logical_or(ensemble_state.alpha.mean() <= 0.01,
                             ensemble_extra.iter[0] >= self.max_iter)

    def next_threshold_adaptive(self,
                                state: cdict,
                                extra: cdict) -> float:
        return jnp.quantile(state.distance, extra.parameters.threshold_quantile_retain[0])

    def log_weight(self,
                   previous_ensemble_state: cdict,
                   previous_ensemble_extra: cdict,
                   new_ensemble_state: cdict,
                   new_ensemble_extra: cdict) -> jnp.ndarray:
        return jnp.where(new_ensemble_state.distance < new_ensemble_extra.parameters.threshold, 0., -jnp.inf)

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
                         extra: cdict) -> Tuple[cdict, cdict]:

        def mcmc_kernel(previous_carry: Tuple[cdict, cdict],
                        _: None) -> Tuple[Tuple[cdict, cdict], Tuple[cdict, cdict]]:
            new_carry = self.mcmc_sampler.update(abc_scenario, *previous_carry)
            return new_carry, new_carry

        start_state, start_extra = self.mcmc_sampler.startup(abc_scenario,
                                                             extra.parameters.mcmc_steps,
                                                             None,
                                                             state,
                                                             extra)

        final_carry, chain = scan(mcmc_kernel,
                                  (start_state, start_extra),
                                  None,
                                  length=self.parameters.mcmc_steps)

        advanced_state, advanced_extra = self.clean_mcmc_chain(chain[0], chain[1])
        advanced_extra.iter = extra.iter

        return advanced_state, advanced_extra

    def update(self,
               abc_scenario: ABCScenario,
               ensemble_state: cdict,
               ensemble_extra: cdict) -> Tuple[cdict, cdict]:
        n = ensemble_state.value.shape[0]
        ensemble_extra.iter = ensemble_extra.iter + 1

        resample_bool = self.resample_criterion(ensemble_state, ensemble_extra)
        ensemble_state.log_weight = jnp.where(resample_bool, 0., ensemble_state.log_weight)

        resampled_ensemble_state, resampled_ensemble_extra \
            = cond(resample_bool,
                   lambda tup: self.resample(*tup),
                   lambda tup: tup,
                   (ensemble_state, ensemble_extra))

        advanced_state, advanced_extra = vmap(self.forward_proposal, in_axes=(None, 0, 0))(abc_scenario,
                                                                                           resampled_ensemble_state,
                                                                                           resampled_ensemble_extra)

        advanced_extra.iter = ensemble_extra.iter
        next_threshold = self.next_threshold(advanced_state, advanced_extra)
        advanced_state.threshold = jnp.ones(n) * next_threshold
        advanced_extra.parameters.threshold = advanced_state.threshold
        advanced_state.log_weight = self.log_weight(ensemble_state, ensemble_extra, advanced_state, advanced_extra)

        advanced_state, advanced_extra = self.adapt(advanced_state, advanced_extra)

        return advanced_state, advanced_extra

