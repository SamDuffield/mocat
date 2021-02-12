########################################################################################################################
# Module: transport/smc.py
# Description: SMC samplers for tempered sequence of distributions.
#
# Web: https://github.com/SamDuffield/mocat
########################################################################################################################

from typing import Tuple, Union, Type
from inspect import isclass

from jax import numpy as jnp, random, vmap
from jax.lax import scan, cond

from mocat.src.core import Scenario, cdict
from mocat.src.transport.sampler import TransportSampler
from mocat.src.mcmc.sampler import MCMCSampler, Correction, check_correction
from mocat.src.utils import bisect
from mocat.src.metrics import log_ess_log_weight


class SMCSampler(TransportSampler):
    name = 'SMC Sampler'

    def forward_proposal(self,
                         scenario: Scenario,
                         previous_state: cdict,
                         previous_extra: cdict) -> Tuple[cdict, cdict]:
        raise AttributeError(f'{self.name} forward_proposal not initiated')

    def adapt(self,
              ensemble_state: cdict,
              ensemble_extra: cdict) -> Tuple[cdict, cdict]:
        return ensemble_state, ensemble_extra

    def resample_criterion(self,
                           ensemble_state: cdict,
                           ensemble_extra: cdict) -> bool:
        return True

    def resample(self,
                 ensemble_state: cdict,
                 ensemble_extra: cdict) -> Tuple[cdict, cdict]:
        n = ensemble_state.value.shape[0]
        resampled_indices = random.categorical(ensemble_extra.random_key[0],
                                               ensemble_state.log_weight,
                                               shape=(n,))
        resampled_ensemble_state = ensemble_state[resampled_indices]
        resampled_ensemble_extra = ensemble_extra[resampled_indices]
        resampled_ensemble_state.log_weight = jnp.zeros(n)
        return resampled_ensemble_state, resampled_ensemble_extra

    def log_weight(self,
                   previous_ensemble_state: cdict,
                   previous_ensemble_extra: cdict,
                   new_ensemble_state: cdict,
                   new_ensemble_extra: cdict) -> jnp.ndarray:
        return previous_ensemble_state.log_weight + new_ensemble_state.log_weight

    def update(self,
               scenario: Scenario,
               ensemble_state: cdict,
               ensemble_extra: cdict) -> Tuple[cdict, cdict]:
        ensemble_extra.iter = ensemble_extra.iter + 1

        resample_bool = self.resample_criterion(ensemble_state, ensemble_extra)

        resampled_ensemble_state, resampled_ensemble_extra \
            = cond(resample_bool,
                   lambda tup: self.resample(*tup),
                   lambda tup: tup,
                   (ensemble_state, ensemble_extra))

        advanced_state, advanced_extra = vmap(self.forward_proposal, in_axes=(None, 0, 0))(scenario,
                                                                                           resampled_ensemble_state,
                                                                                           resampled_ensemble_extra)

        advanced_state.log_weight = self.log_weight(resampled_ensemble_state, resampled_ensemble_extra,
                                                    advanced_state, advanced_extra)

        advanced_state, advanced_extra = self.adapt(advanced_state, advanced_extra)

        return advanced_state, advanced_extra


class TemperedSMCSampler(SMCSampler):
    name = 'Tempered SMC Sampler'

    def __init__(self,
                 temperature_schedule: Union[None, jnp.ndarray] = None,
                 max_temperature: float = 1.,
                 max_iter: int = int(1e4),
                 **kwargs):
        self.max_iter = max_iter
        self.max_temperature = max_temperature
        self.temperature_schedule = temperature_schedule
        super().__init__(**kwargs)

    def __setattr__(self, key, value):
        super().__setattr__(key, value)
        if key == 'temperature_schedule':
            if value is None:
                self.next_temperature = self.next_temperature_adaptive
                self.max_iter = int(1e4)
            else:
                self.next_temperature = lambda state, extra: self.temperature_schedule[extra.iter[0]]
                self.max_temperature = value[-1]
                self.max_iter = len(value)

    def startup(self,
                scenario: Scenario,
                n: int,
                initial_state: cdict,
                initial_extra: cdict,
                **kwargs) -> Tuple[cdict, cdict]:

        initial_state, initial_extra = super().startup(scenario, n, initial_state, initial_extra, **kwargs)

        random_keys = random.split(initial_extra.random_key[0], 3*n)

        initial_extra.random_key = random_keys[:n]

        n = len(initial_state.value)
        initial_state.prior_potential = vmap(scenario.prior_potential)(initial_state.value, random_keys[n:(2*n)])
        initial_state.likelihood_potential = vmap(scenario.likelihood_potential)(initial_state.value,
                                                                                 random_keys[(2*n):])
        initial_state.potential = initial_state.prior_potential
        initial_state.temperature = jnp.zeros(n)
        initial_state.log_weight = jnp.zeros(n)

        scenario.temperature = 0.

        initial_extra.parameters = vmap(lambda _: initial_extra.parameters)(jnp.arange(n))

        return initial_state, initial_extra

    def next_temperature_adaptive(self,
                                  ensemble_state: cdict,
                                  ensemble_extra: cdict) -> float:
        raise AttributeError(f'{self.name} next_temperature_adaptive not initiated')

    def termination_criterion(self,
                              ensemble_state: cdict,
                              ensemble_extra: cdict) -> bool:
        return jnp.logical_or(ensemble_state.temperature[0] >= self.max_temperature,
                             ensemble_extra.iter[0] >= self.max_iter)

    def clean_chain(self,
                    scenario: Scenario,
                    chain_ensemble_state: cdict) -> cdict:
        chain_ensemble_state.temperature = chain_ensemble_state.temperature[:, 0]
        scenario.temperature = float(chain_ensemble_state.temperature[-1])
        return chain_ensemble_state

    def update(self,
               scenario: Scenario,
               ensemble_state: cdict,
               ensemble_extra: cdict) -> Tuple[cdict, cdict]:
        n = ensemble_state.value.shape[0]
        ensemble_extra.iter = ensemble_extra.iter + 1
        prev_temperature = ensemble_state.temperature[0]
        scenario.temperature = prev_temperature

        resample_bool = self.resample_criterion(ensemble_state, ensemble_extra)
        ensemble_state.log_weight = jnp.where(resample_bool, 0., ensemble_state.log_weight)

        resampled_ensemble_state, resampled_ensemble_extra \
            = cond(resample_bool,
                   lambda tup: self.resample(*tup),
                   lambda tup: tup,
                   (ensemble_state, ensemble_extra))

        advanced_state, advanced_extra = vmap(self.forward_proposal, in_axes=(None, 0, 0))(scenario,
                                                                                           resampled_ensemble_state,
                                                                                           resampled_ensemble_extra)

        advanced_extra.iter = ensemble_extra.iter
        next_temperature = self.next_temperature(advanced_state, advanced_extra)
        advanced_state.temperature = jnp.ones(n) * next_temperature
        advanced_state.log_weight = self.log_weight(ensemble_state, ensemble_extra, advanced_state, advanced_extra)

        advanced_state, advanced_extra = self.adapt(advanced_state, advanced_extra)

        return advanced_state, advanced_extra


class MetropolisedSMCSampler(TemperedSMCSampler):
    name = "Metropolised SMC Sampler"

    def __init__(self,
                 mcmc_sampler: Union[MCMCSampler, Type[MCMCSampler]],
                 mcmc_correction: Union[Correction, Type[Correction], str] = 'sampler_default',
                 mcmc_steps: int = 20,
                 max_iter: int = int(1e4),
                 temperature_schedule: Union[None, jnp.ndarray] = None,
                 max_temperature: float = 1.,
                 ess_threshold: float = 0.8,
                 bisection_tol: float = 1e-5,
                 max_bisection_iter: int = 1000,
                 **kwargs):
        if temperature_schedule is not None:
            if temperature_schedule[0] == 0.:
                temperature_schedule = temperature_schedule[1:]

        super().__init__(max_iter=max_iter, temperature_schedule=temperature_schedule, max_temperature=max_temperature,
                         **kwargs)
        if isclass(mcmc_sampler):
            mcmc_sampler = mcmc_sampler()
        self.mcmc_sampler = mcmc_sampler
        if mcmc_correction != 'sampler_default':
            self.mcmc_sampler.correction = mcmc_correction
        self.parameters.mcmc_steps = mcmc_steps

        self.parameters.ess_threshold = ess_threshold
        self.parameters.bisection_tol = bisection_tol
        self.parameters.max_bisection_iter = max_bisection_iter

    def __setattr__(self, key, value):
        if key == 'temperature_schedule':
            if value is not None and value[0] == 0.:
                value = value[1:]
        super().__setattr__(key, value)

    def startup(self,
                scenario: Scenario,
                n: int,
                initial_state: cdict,
                initial_extra: cdict,
                **kwargs) -> Tuple[cdict, cdict]:

        self.mcmc_sampler.correction = check_correction(self.mcmc_sampler.correction)

        initial_state, initial_extra = super().startup(scenario, n, initial_state, initial_extra, **kwargs)

        first_temp = self.next_temperature(initial_state, initial_extra)
        initial_state.temperature += first_temp
        initial_state.potential = initial_state.prior_potential + first_temp * initial_state.likelihood_potential
        initial_state.log_weight = - first_temp * initial_state.likelihood_potential

        initial_state, initial_extra = vmap(
            lambda state, extra: self.mcmc_sampler.startup(scenario,
                                                           n,
                                                           state,
                                                           extra))(initial_state, initial_extra)

        return initial_state, initial_extra

    @staticmethod
    def log_ess(current_temperature: float,
                new_temperature: float,
                likelihood_potential: jnp.ndarray) -> float:
        log_weight = - (new_temperature - current_temperature) * likelihood_potential
        return log_ess_log_weight(log_weight)

    def next_temperature_adaptive(self,
                                  ensemble_state: cdict,
                                  ensemble_extra: cdict) -> float:
        temperature_bounds = jnp.array([ensemble_state.temperature[0], self.max_temperature])
        likelihood_potential = ensemble_state.likelihood_potential
        log_n_samp_threshold = jnp.log(len(likelihood_potential) * self.parameters.ess_threshold)

        bisect_out_bounds, bisect_out_evals, bisect_out_iter \
            = bisect(lambda x: self.log_ess(temperature_bounds[0],
                                            x,
                                            likelihood_potential) - log_n_samp_threshold,
                     temperature_bounds,
                     max_iter=self.parameters.max_bisection_iter,
                     tol=self.parameters.bisection_tol)
        return bisect_out_bounds[jnp.argmin(jnp.abs(bisect_out_evals))]

    def clean_mcmc_chain(self,
                         chain_state: cdict,
                         chain_extra: cdict) -> Tuple[cdict, cdict]:
        clean_state = chain_state[-1]
        clean_extra = chain_extra[-1]
        clean_state.alpha = chain_state.alpha.mean()
        clean_extra.parameters = chain_extra.parameters[-1]
        return clean_state, clean_extra

    def forward_proposal(self,
                         scenario: Scenario,
                         state: cdict,
                         extra: cdict) -> Tuple[cdict, cdict]:

        def mcmc_kernel(previous_carry: Tuple[cdict, cdict],
                        _: None) -> Tuple[Tuple[cdict, cdict], Tuple[cdict, cdict]]:
            new_carry = self.mcmc_sampler.update(scenario, *previous_carry)
            return new_carry, new_carry

        start_state, start_extra = self.mcmc_sampler.startup(scenario,
                                                             state.value.shape[0],
                                                             state,
                                                             extra)

        final_carry, chain = scan(mcmc_kernel,
                                  (start_state, start_extra),
                                  None,
                                  length=self.parameters.mcmc_steps)

        advanced_state, advanced_extra = self.clean_mcmc_chain(chain[0], chain[1])

        advanced_extra.random_key, subkey = random.split(advanced_extra.random_key)

        advanced_state.prior_potential = scenario.prior_potential(advanced_state.value, subkey)
        advanced_state.likelihood_potential = (advanced_state.potential - advanced_state.prior_potential) \
                                              / scenario.temperature
        advanced_extra.iter = extra.iter
        return advanced_state, advanced_extra

    def log_weight(self,
                   previous_ensemble_state: cdict,
                   previous_ensemble_extra: cdict,
                   new_ensemble_state: cdict,
                   new_ensemble_extra: cdict) -> jnp.ndarray:
        return - (new_ensemble_state.temperature[0] - previous_ensemble_state.temperature[0]) \
               * new_ensemble_state.likelihood_potential
