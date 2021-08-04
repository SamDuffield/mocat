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
from jax.scipy.special import logsumexp

from mocat.src.core import Scenario, cdict
from mocat.src.transport.sampler import TransportSampler
from mocat.src.mcmc.sampler import MCMCSampler, Correction, check_correction
from mocat.src.mcmc.metropolis import Metropolis
from mocat.src.utils import bisect
from mocat.src.metrics import log_ess_log_weight


class SMCSampler(TransportSampler):
    name = 'SMC Sampler'

    def startup(self,
                scenario: Scenario,
                n: int,
                initial_state: cdict,
                initial_extra: cdict,
                **kwargs) -> Tuple[cdict, cdict]:
        initial_state, initial_extra = super().startup(scenario, n, initial_state, initial_extra, **kwargs)
        if not hasattr(initial_extra, 'resample_bool'):
            initial_extra.resample_bool = True
        if not hasattr(initial_state, 'log_weight'):
            initial_state.log_weight = jnp.zeros(n)
        if not hasattr(initial_state, 'ess'):
            initial_state.ess = jnp.zeros(n) + n
        return initial_state, initial_extra

    def forward_proposal(self,
                         scenario: Scenario,
                         previous_state: cdict,
                         previous_extra: cdict,
                         random_key: jnp.ndarray) -> cdict:
        raise AttributeError(f'{self.name} forward_proposal not initiated')

    def adapt(self,
              previous_ensemble_state: cdict,
              previous_extra: cdict,
              new_ensemble_state: cdict,
              new_extra: cdict) -> Tuple[cdict, cdict]:
        new_ensemble_state.log_weight = previous_ensemble_state.log_weight + new_ensemble_state.log_weight
        return new_ensemble_state, new_extra

    def resample_criterion(self,
                           ensemble_state: cdict,
                           extra: cdict) -> bool:
        return True

    def resample(self,
                 ensemble_state: cdict,
                 random_key: jnp.ndarray) -> cdict:
        n = ensemble_state.value.shape[0]
        resampled_indices = random.categorical(random_key,
                                               ensemble_state.log_weight,
                                               shape=(n,))
        resampled_ensemble_state = ensemble_state[resampled_indices]
        resampled_ensemble_state.log_weight = jnp.zeros(n)
        resampled_ensemble_state.ess = jnp.zeros(n) + n
        return resampled_ensemble_state

    def update(self,
               scenario: Scenario,
               ensemble_state: cdict,
               extra: cdict) -> Tuple[cdict, cdict]:
        extra.iter = extra.iter + 1
        n = ensemble_state.value.shape[0]

        extra.resample_bool = self.resample_criterion(ensemble_state, extra)

        random_keys_all = random.split(extra.random_key, n + 2)
        extra.random_key = random_keys_all[-1]

        resampled_ensemble_state \
            = cond(extra.resample_bool,
                   lambda state: self.resample(state, random_keys_all[-2]),
                   lambda state: state,
                   ensemble_state)

        advanced_state = vmap(self.forward_proposal,
                              in_axes=(None, 0, None, 0))(scenario,
                                                          resampled_ensemble_state,
                                                          extra,
                                                          random_keys_all[:n])
        advanced_state, advanced_extra = self.adapt(resampled_ensemble_state, extra,
                                                    advanced_state, extra)

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
        if key == 'temperature_schedule':
            if value is None:
                self.next_temperature = self.next_temperature_adaptive
                if hasattr(self, 'temperature_schedule') and self.temperature_schedule is not None \
                        and self.max_iter == len(self.temperature_schedule):
                    self.max_iter = int(1e4)
            else:
                self.next_temperature = lambda state, extra: self.temperature_schedule[extra.iter]
                self.max_temperature = value[-1]
                self.max_iter = len(value)
        super().__setattr__(key, value)

    def startup(self,
                scenario: Scenario,
                n: int,
                initial_state: cdict,
                initial_extra: cdict,
                initiate_potential: bool = True,
                **kwargs) -> Tuple[cdict, cdict]:
        if not hasattr(scenario, 'prior_sample'):
            raise TypeError(f'Likelihood tempering requires scenario {scenario.name} to have prior_sample implemented')

        initial_state, initial_extra = super().startup(scenario, n, initial_state, initial_extra, **kwargs)

        random_keys = random.split(initial_extra.random_key, 2 * n + 1)

        initial_extra.random_key = random_keys[-1]
        if initiate_potential:
            if hasattr(scenario, 'prior_potential_and_grad'):
                initial_state.prior_potential, initial_state.grad_prior_potential \
                    = vmap(scenario.prior_potential_and_grad)(initial_state.value, random_keys[:n])
                initial_state.likelihood_potential, initial_state.grad_likelihood_potential \
                    = vmap(scenario.likelihood_potential_and_grad)(initial_state.value, random_keys[n:(2 * n)])
                initial_state.potential = initial_state.prior_potential
                initial_state.grad_potential = initial_state.grad_prior_potential
            else:
                initial_state.prior_potential = vmap(scenario.prior_potential)(initial_state.value, random_keys[:n])
                initial_state.likelihood_potential = vmap(scenario.likelihood_potential)(initial_state.value,
                                                                                         random_keys[n:(2 * n)])
                initial_state.potential = initial_state.prior_potential

        initial_state.temperature = jnp.zeros(n)
        initial_state.log_weight = jnp.zeros(n)
        initial_state.ess = jnp.zeros(n) + n
        initial_state.log_norm_constant = logsumexp(initial_state.log_weight, b=1 / n) * jnp.ones(n)

        scenario.temperature = 0.

        return initial_state, initial_extra

    def next_temperature_adaptive(self,
                                  ensemble_state: cdict,
                                  extra: cdict) -> float:
        raise AttributeError(f'{self.name} next_temperature_adaptive not initiated')

    def termination_criterion(self,
                              ensemble_state: cdict,
                              extra: cdict) -> bool:
        return jnp.logical_or(jnp.logical_or(ensemble_state.temperature[0] >= self.max_temperature,
                                             extra.iter >= self.max_iter), jnp.isnan(ensemble_state.value).mean() > 0.1)

    def clean_chain(self,
                    scenario: Scenario,
                    chain_ensemble_state: cdict) -> cdict:
        chain_ensemble_state.temperature = chain_ensemble_state.temperature[:, 0]
        scenario.temperature = float(chain_ensemble_state.temperature[-1])
        chain_ensemble_state.ess = chain_ensemble_state.ess[:, 0]
        chain_ensemble_state.log_norm_constant = chain_ensemble_state.log_norm_constant[:, 0]
        return chain_ensemble_state

    def log_weight(self,
                   previous_ensemble_state: cdict,
                   previous_extra: cdict,
                   new_ensemble_state: cdict,
                   new_extra: cdict) -> Union[float, jnp.ndarray]:
        return 0.

    def adapt(self,
              previous_ensemble_state: cdict,
              previous_extra: cdict,
              new_ensemble_state: cdict,
              new_extra: cdict) -> Tuple[cdict, cdict]:
        n = new_ensemble_state.value.shape[0]
        next_temperature = self.next_temperature(new_ensemble_state, new_extra)
        new_ensemble_state.temperature = jnp.ones(n) * next_temperature
        new_ensemble_state.log_weight = previous_ensemble_state.log_weight \
                                        + self.log_weight(previous_ensemble_state, previous_extra,
                                                          new_ensemble_state, new_extra)
        new_ensemble_state.ess = jnp.ones(n) * jnp.exp(log_ess_log_weight(new_ensemble_state.log_weight))
        new_ensemble_state.potential = new_ensemble_state.prior_potential \
                                       + next_temperature * new_ensemble_state.likelihood_potential
        if hasattr(new_ensemble_state, 'grad_potential'):
            new_ensemble_state.grad_potential \
                = new_ensemble_state.grad_prior_potential \
                  + next_temperature * new_ensemble_state.grad_likelihood_potential

        new_ensemble_state.log_norm_constant \
            = previous_ensemble_state.log_norm_constant \
              + logsumexp(new_ensemble_state.log_weight) \
              - logsumexp(previous_ensemble_state.log_weight)

        return new_ensemble_state, new_extra

    def update(self,
               scenario: Scenario,
               ensemble_state: cdict,
               extra: cdict) -> Tuple[cdict, cdict]:
        scenario.temperature = ensemble_state.temperature[0]
        advanced_state, advanced_extra = super().update(scenario, ensemble_state, extra)
        return advanced_state, advanced_extra


class MetropolisedSMCSampler(TemperedSMCSampler):
    name = "Metropolised SMC Sampler"

    def __init__(self,
                 mcmc_sampler: Union[MCMCSampler, Type[MCMCSampler]],
                 mcmc_correction: Union[Correction, Type[Correction], str] = 'sampler_default',
                 mcmc_steps: int = 1,
                 max_iter: int = int(1e4),
                 temperature_schedule: Union[None, jnp.ndarray] = None,
                 max_temperature: float = 1.,
                 ess_threshold_retain: float = 0.9,
                 ess_threshold_resample: float = 0.5,
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

        self.parameters.ess_threshold_retain = ess_threshold_retain
        self.parameters.ess_threshold_resample = ess_threshold_resample
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
        scenario.temperature = first_temp
        initial_state.temperature += first_temp
        initial_state.potential = initial_state.prior_potential + first_temp * initial_state.likelihood_potential
        if hasattr(initial_state, 'grad_potential'):
            initial_state.grad_potential = initial_state.grad_prior_potential \
                                           + first_temp * initial_state.grad_likelihood_potential

        initial_state.log_weight = - first_temp * initial_state.likelihood_potential
        initial_state.ess = jnp.repeat(jnp.exp(log_ess_log_weight(initial_state.log_weight)), n)
        initial_state.log_norm_constant = logsumexp(initial_state.log_weight, b=1 / n) * jnp.ones(n)

        initial_state, initial_extra = vmap(
            lambda state: self.mcmc_sampler.startup(scenario,
                                                    n,
                                                    state,
                                                    initial_extra))(initial_state)
        initial_extra = initial_extra[0]
        return initial_state, initial_extra

    def resample_criterion(self,
                           ensemble_state: cdict,
                           extra: cdict) -> bool:
        return ensemble_state.ess[0] <= (extra.parameters.ess_threshold_resample * len(ensemble_state.value))

    @staticmethod
    def log_ess(previous_log_weight: jnp.ndarray,
                current_temperature: float,
                new_temperature: float,
                likelihood_potential: jnp.ndarray) -> float:
        log_weight = previous_log_weight - (new_temperature - current_temperature) * likelihood_potential
        return log_ess_log_weight(log_weight)

    def next_temperature_adaptive(self,
                                  ensemble_state: cdict,
                                  extra: cdict) -> float:
        temperature_bounds = jnp.array([ensemble_state.temperature[0], self.max_temperature])
        likelihood_potential = ensemble_state.likelihood_potential
        log_n_samp_threshold = jnp.log(ensemble_state.ess[0] * self.parameters.ess_threshold_retain)

        bisect_out_bounds, bisect_out_evals, bisect_out_iter \
            = bisect(lambda x: self.log_ess(ensemble_state.log_weight,
                                            temperature_bounds[0],
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
                         extra: cdict,
                         random_key: jnp.ndarray) -> cdict:

        def mcmc_kernel(previous_carry: Tuple[cdict, cdict],
                        _: None) -> Tuple[Tuple[cdict, cdict], Tuple[cdict, cdict]]:
            new_carry = self.mcmc_sampler.update(scenario, *previous_carry)
            return new_carry, new_carry

        extra.random_key = random_key

        start_state, start_extra = self.mcmc_sampler.startup(scenario,
                                                             extra.parameters.mcmc_steps,
                                                             state,
                                                             extra)

        final_carry, chain = scan(mcmc_kernel,
                                  (start_state, start_extra),
                                  None,
                                  length=self.parameters.mcmc_steps)

        advanced_state, advanced_extra = self.clean_mcmc_chain(chain[0], chain[1])

        advanced_state.prior_potential = scenario.prior_potential(advanced_state.value, advanced_extra.random_key)
        advanced_state.likelihood_potential = (advanced_state.potential - advanced_state.prior_potential) \
                                              / scenario.temperature
        return advanced_state

    def log_weight(self,
                   previous_ensemble_state: cdict,
                   previous_extra: cdict,
                   new_ensemble_state: cdict,
                   new_extra: cdict) -> jnp.ndarray:
        return - (new_ensemble_state.temperature[0] - previous_ensemble_state.temperature[0]) \
               * new_ensemble_state.likelihood_potential


class RMMetropolisedSMCSampler(MetropolisedSMCSampler):
    def __init__(self,
                 mcmc_sampler: Union[MCMCSampler, Type[MCMCSampler]],
                 mcmc_correction: Union[Correction, Type[Correction], str] = Metropolis,
                 mcmc_steps: int = 1,
                 max_iter: int = int(1e4),
                 temperature_schedule: Union[None, jnp.ndarray] = None,
                 max_temperature: float = 1.,
                 ess_threshold_retain: float = 0.9,
                 ess_threshold_resample: float = 0.5,
                 bisection_tol: float = 1e-5,
                 max_bisection_iter: int = 1000,
                 rm_stepsize: float = 1.,
                 **kwargs):
        super().__init__(mcmc_sampler=mcmc_sampler, mcmc_correction=mcmc_correction, mcmc_steps=mcmc_steps,
                         max_iter=max_iter, temperature_schedule=temperature_schedule, max_temperature=max_temperature,
                         ess_threshold_retain=ess_threshold_retain, ess_threshold_resample=ess_threshold_resample,
                         bisection_tol=bisection_tol, max_bisection_iter=max_bisection_iter, **kwargs)
        self.parameters.rm_stepsize = rm_stepsize

    def startup(self,
                scenario: Scenario,
                n: int,
                initial_state: cdict,
                initial_extra: cdict,
                **kwargs) -> Tuple[cdict, cdict]:
        initial_state, initial_extra = super().startup(scenario, n, initial_state, initial_extra, **kwargs)
        initial_state.stepsize = jnp.ones(n) * initial_extra.parameters.stepsize
        return initial_state, initial_extra

    def adapt(self,
              previous_ensemble_state: cdict,
              previous_extra: cdict,
              new_ensemble_state: cdict,
              new_extra: cdict) -> Tuple[cdict, cdict]:
        new_ensemble_state, new_extra = super().adapt(previous_ensemble_state, previous_extra,
                                                      new_ensemble_state, new_extra)
        log_stepsize = jnp.log(new_extra.parameters.stepsize)
        alpha_mean = jnp.average(new_ensemble_state.alpha,
                                 weights=jnp.exp(new_ensemble_state.log_weight - new_ensemble_state.log_weight.max()))
        new_log_stepsize = log_stepsize + new_extra.parameters.rm_stepsize \
                           * (alpha_mean - self.mcmc_sampler.tuning.target)
        new_extra.parameters.stepsize = jnp.exp(new_log_stepsize)

        new_ensemble_state.stepsize = jnp.ones(new_ensemble_state.value.shape[0]) * new_extra.parameters.stepsize
        return new_ensemble_state, new_extra

    def clean_chain(self,
                    scenario: Scenario,
                    chain_ensemble_state: cdict) -> cdict:
        chain_ensemble_state = super().clean_chain(scenario, chain_ensemble_state)
        chain_ensemble_state.stepsize = chain_ensemble_state.stepsize[:, 0]
        return chain_ensemble_state
