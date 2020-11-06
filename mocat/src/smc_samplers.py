########################################################################################################################
# Module: transport/smc.py
# Description: SMC samplers (for static models)
#
# Web: https://github.com/SamDuffield/mocat
########################################################################################################################

from time import time
from typing import Tuple, Type, Any, Callable, Union

from jax import numpy as np, jit, random, vmap, grad
from jax.lax import scan
from jax.scipy.special import logsumexp

from mocat.src.core import Scenario, CDict
from mocat.src.mcmc.sampler import MCMCSampler
from mocat.src.mcmc.corrections import Correction
from mocat.src.mcmc.run import startup_mcmc, check_correction
from mocat.src.utils import gaussian_potential, while_loop_stacked, bisect


class Scheduler:
    terminal_schedule_param = 1.

    scheduled_scenario: Scenario = None

    def initiate_adjusted_scenario(self,
                                   scenario: Scenario,
                                   initial_potential_func: Callable,
                                   initial_grad_potential_func: Callable = None):
        raise NotImplementedError

    @staticmethod
    def log_weight_function(previous_schedule_param: float,
                            new_schedule_param: float,
                            inital_potential_evals: np.ndarray,
                            target_potential_evals: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class Tempered(Scheduler):

    def initiate_adjusted_scenario(self,
                                   scenario: Scenario,
                                   initial_potential_func: Callable,
                                   initial_grad_potential_func: Callable = None):
        if initial_grad_potential_func is None:
            initial_grad_potential_func = grad(initial_potential_func)

        self.scheduled_scenario = Scenario(name=f'Tempered {scenario.name}')
        self.scheduled_scenario.dim = scenario.dim
        self.scheduled_scenario.schedule_param = 0.
        self.scheduled_scenario.potential = lambda x: (1 - self.scheduled_scenario.schedule_param) \
                                                      * initial_potential_func(x) \
                                                      + self.scheduled_scenario.schedule_param \
                                                      * scenario.potential(x)
        self.scheduled_scenario.grad_potential = lambda x: (1 - self.scheduled_scenario.schedule_param) \
                                                           * initial_grad_potential_func(x) \
                                                           + self.scheduled_scenario.schedule_param \
                                                           * scenario.grad_potential(x)

    @staticmethod
    def log_weight_function(previous_schedule_param: float,
                            new_schedule_param: float,
                            inital_potential_evals: np.ndarray,
                            target_potential_evals: np.ndarray) -> np.ndarray:
        return (new_schedule_param - previous_schedule_param) * (- target_potential_evals + inital_potential_evals)


class IQInterpolate(Scheduler):
    terminal_schedule_param = np.inf

    def initiate_adjusted_scenario(self,
                                   scenario: Scenario,
                                   initial_potential_func: Callable,
                                   initial_grad_potential_func: Callable = None):
        if initial_grad_potential_func is None:
            initial_grad_potential_func = grad(initial_potential_func)

        self.scheduled_scenario = Scenario(name=f'IQ Interpolated {scenario.name}')
        self.scheduled_scenario.dim = scenario.dim
        self.scheduled_scenario.schedule_param = 0.

        def scheduled_potential(x, schedule_param):
            init_pot = initial_potential_func(x)
            scen_pot = scenario.potential(x)
            diff_term = 2 * np.log(np.exp(0.5 * (init_pot - scen_pot)) / schedule_param + 1)
            diff_term = np.where(np.isinf(diff_term), 0, diff_term)
            return scen_pot + diff_term

        self.scheduled_scenario.potential = lambda x: scheduled_potential(x, self.scheduled_scenario.schedule_param)

        def scheduled_grad_potential(x, schedule_param):
            init_pot = initial_potential_func(x)
            scen_pot = scenario.potential(x)
            init_grad_pot = initial_grad_potential_func(x)
            scen_grad_pot = scenario.grad_potential(x)
            diff_term = (init_grad_pot - scen_grad_pot) / schedule_param * np.exp(0.5 * (init_pot - scen_pot)) \
                        / (np.exp(0.5 * (init_pot - scen_pot)) / schedule_param + 1)
            diff_term = np.where(np.isinf(diff_term), 0, diff_term)
            diff_term = np.where(np.isnan(diff_term), 0, diff_term)
            return scen_grad_pot + diff_term

        self.scheduled_scenario.grad_potential = lambda x: scheduled_grad_potential(x,
                                                                                    self.scheduled_scenario.schedule_param)

    @staticmethod
    def log_weight_function(previous_schedule_param: float,
                            new_schedule_param: float,
                            inital_potential_evals: np.ndarray,
                            target_potential_evals: np.ndarray) -> np.ndarray:
        a_maxs = np.max([inital_potential_evals, target_potential_evals], axis=0)
        shifted_inital_potential_evals = inital_potential_evals - a_maxs
        shifted_target_potential_evals = target_potential_evals - a_maxs

        return np.where(previous_schedule_param == 0.,
                        2 * np.log(np.exp(0.5 * shifted_inital_potential_evals)
                                   + np.exp(0.5 * shifted_target_potential_evals) * new_schedule_param)
                        - shifted_inital_potential_evals,
                        2 * (np.log(np.exp(0.5 * shifted_inital_potential_evals) / new_schedule_param
                                    + np.exp(0.5 * shifted_target_potential_evals))
                             - np.log(np.exp(0.5 * shifted_inital_potential_evals) / previous_schedule_param
                                      + np.exp(0.5 * shifted_target_potential_evals))))


def default_post_mcmc_update(previous_full_state: CDict,
                             previous_full_extra: CDict,
                             new_enlarged_state: CDict,
                             new_enlarged_extra: CDict) -> Tuple[CDict, CDict]:
    new_full_state = new_enlarged_state[:, -1]
    new_full_extra = new_enlarged_extra[:, -1]
    new_full_extra.parameters = new_full_extra.parameters[:, -1]
    return new_full_state, new_full_extra


def run_smc_sampler(scenario: Scenario,
                    n_samps: int,
                    random_key: np.ndarray,
                    mcmc_sampler: MCMCSampler,
                    mcmc_correction: Union[None, str, Correction, Type[Correction]] = 'sampler_default',
                    mcmc_steps: int = 20,
                    scheduler: Scheduler = None,
                    preschedule: np.ndarray = None,
                    initial_state: CDict = None,
                    initial_potential_func: Callable = None,
                    initial_grad_potential_func: Callable = None,
                    post_mcmc_update: Callable = default_post_mcmc_update,
                    max_iter: int = 1000,
                    ess_threshold: float = 0.8,
                    max_extend_multiplier: float = np.inf,
                    bisection_tol: float = 1e-5,
                    max_bisection_iter: int = 1000,
                    name: str = None) -> CDict:
    if initial_state is not None and initial_potential_func is None:
        raise ValueError('Found initial_state but not initial_potential_func')

    if initial_state is None and initial_potential_func is not None:
        raise ValueError('Found initial_potential_func but not initial_state')

    mcmc_sampler, mcmc_correction = check_correction(mcmc_sampler, mcmc_correction)

    if initial_state is None:
        random_key, sub_key = random.split(random_key)
        x0 = random.normal(sub_key, shape=(n_samps, scenario.dim))
        initial_state = CDict(value=x0)

        initial_potential_func = gaussian_potential
        initial_grad_potential_func = lambda x: x

    initial_extra = CDict(random_key=None,
                          iter=1)

    if scheduler is None:
        scheduler = Tempered()

    scheduler.initiate_adjusted_scenario(scenario, initial_potential_func, initial_grad_potential_func)

    initial_state, initial_extra = vmap(
        lambda state: startup_mcmc(scheduler.scheduled_scenario, mcmc_sampler, None, mcmc_correction,
                                   state, initial_extra))(initial_state)

    if None in initial_extra.parameters.__dict__.values():
        raise ValueError(f'None found in {mcmc_sampler.name} parameters (within SMC sampler): '
                         + str(mcmc_sampler.parameters))

    initial_extra.random_key = random.split(random_key, n_samps)

    start = time()

    if preschedule is not None:
        out_samps = _run_prescheduled_smc_sampler(scenario, mcmc_sampler, mcmc_correction, mcmc_steps,
                                                  scheduler, preschedule,
                                                  initial_state, initial_extra, initial_potential_func,
                                                  post_mcmc_update)
    else:
        out_samps = _run_adaptive_smc_sampler(scenario, mcmc_sampler, mcmc_correction, mcmc_steps,
                                              scheduler,
                                              initial_state, initial_extra, initial_potential_func,
                                              post_mcmc_update, max_iter,
                                              ess_threshold, max_extend_multiplier,
                                              bisection_tol, max_bisection_iter)

    out_samps.value.block_until_ready()
    end = time()
    out_samps.time = end - start

    if name is None:
        name = scenario.name + ": SMC Sampler - " + mcmc_sampler.name
    out_samps.name = name

    return out_samps


def _run_prescheduled_smc_sampler(scenario: Scenario,
                                  mcmc_sampler: MCMCSampler,
                                  mcmc_correction: Correction,
                                  mcmc_steps: int,
                                  scheduler: Scheduler,
                                  preschedule: np.ndarray,
                                  initial_state: CDict,
                                  initial_extra: CDict,
                                  initial_potential_func: Callable,
                                  post_mcmc_update: Callable) -> CDict:
    initial_state.weight_schedule = 0.

    initial_potential_vec = vmap(initial_potential_func)
    target_potential_vec = vmap(scenario.potential)

    smc_sampler_advance = init_smc_sampler_advance(scheduler, mcmc_sampler,
                                                   mcmc_correction,
                                                   mcmc_steps,
                                                   post_mcmc_update)

    def prescheduled_smc_kernel(previous_carry: Tuple[CDict, CDict],
                                iter_ind: int) -> Tuple[Tuple[CDict, CDict], CDict]:
        previous_state, previous_extra = previous_carry

        previous_schedule_param = preschedule[iter_ind - 1]
        new_schedule_param = preschedule[iter_ind]

        previous_initial_potential_evals = initial_potential_vec(previous_state.value)
        previous_target_potential_evals = target_potential_vec(previous_state.value)

        new_state, new_extra = smc_sampler_advance(previous_state,
                                                   previous_extra,
                                                   previous_schedule_param,
                                                   new_schedule_param,
                                                   previous_initial_potential_evals,
                                                   previous_target_potential_evals)
        return (new_state, new_extra), new_state

    final_carry, chain = scan(prescheduled_smc_kernel,
                              (initial_state, initial_extra),
                              np.arange(1, len(preschedule)))
    return chain


def _run_adaptive_smc_sampler(scenario: Scenario,
                              mcmc_sampler: MCMCSampler,
                              mcmc_correction: Correction,
                              mcmc_steps: int,
                              scheduler: Scheduler,
                              initial_state: CDict,
                              initial_extra: CDict,
                              initial_potential_func: Callable,
                              post_mcmc_update: Callable,
                              max_iter: int,
                              ess_threshold: float,
                              max_extend_multiplier: float,
                              bisection_tol: float,
                              max_bisection_iter: int) -> CDict:
    n_samps = initial_state.value.shape[-2]
    max_extend_multiplier = np.where(max_extend_multiplier == np.inf, 1e32, max_extend_multiplier)

    initial_state.weight_schedule = 0.

    initial_potential_vec = vmap(initial_potential_func)
    target_potential_vec = vmap(scenario.potential)

    log_n_samp_threshold = np.log(n_samps * ess_threshold)

    smc_sampler_advance = init_smc_sampler_advance(scheduler, mcmc_sampler,
                                                   mcmc_correction,
                                                   mcmc_steps,
                                                   post_mcmc_update)

    def log_ess(previous_schedule_param: float,
                new_schedule_param: float,
                previous_initial_potential_evals: np.ndarray,
                previous_target_potential_evals: np.ndarray) -> float:
        log_weights = scheduler.log_weight_function(previous_schedule_param,
                                                    new_schedule_param,
                                                    previous_initial_potential_evals,
                                                    previous_target_potential_evals)
        return 2 * logsumexp(log_weights) - logsumexp(2 * log_weights)

    @jit
    def adaptive_smc_kernel(previous_state: CDict,
                            previous_full_extra: Tuple[CDict, np.ndarray, np.ndarray]) \
            -> Tuple[CDict, Tuple[CDict, np.ndarray, np.ndarray]]:
        previous_extra, previous_initial_potential_evals, previous_target_potential_evals = previous_full_extra

        previous_schedule_param = previous_state.weight_schedule

        upper_bound = np.minimum(scheduler.terminal_schedule_param,
                                 previous_schedule_param * max_extend_multiplier)
        upper_bound = np.where(upper_bound == 0, np.minimum(10, scheduler.terminal_schedule_param), upper_bound)

        latest_bounds = np.array([previous_schedule_param, upper_bound])

        # Determine next schedule parameter
        bisect_out_bounds, bisect_out_evals, bisect_out_iter = bisect(lambda x: log_ess(previous_schedule_param,
                                                                                        x,
                                                                                        previous_initial_potential_evals,
                                                                                        previous_target_potential_evals)
                                                                                - log_n_samp_threshold,
                                                                      latest_bounds,
                                                                      max_iter=max_bisection_iter,
                                                                      tol=bisection_tol)
        new_schedule_param = bisect_out_bounds[np.argmin(np.abs(bisect_out_evals))]

        new_state, new_extra = smc_sampler_advance(previous_state,
                                                   previous_extra,
                                                   previous_schedule_param,
                                                   new_schedule_param,
                                                   previous_initial_potential_evals,
                                                   previous_target_potential_evals)

        new_initial_potential_evals = initial_potential_vec(previous_state.value)
        new_target_potential_evals = target_potential_vec(previous_state.value)

        return new_state, (new_extra, new_initial_potential_evals, new_target_potential_evals)

    initial_initial_potential_evals = initial_potential_vec(initial_state.value)
    initial_target_potential_evals = target_potential_vec(initial_state.value)

    initial_carry = (initial_state, (initial_extra, initial_initial_potential_evals, initial_target_potential_evals))

    chain = while_loop_stacked(lambda state, extra: state.weight_schedule < scheduler.terminal_schedule_param,
                               adaptive_smc_kernel,
                               initial_carry,
                               max_iter)
    return chain


def init_smc_sampler_advance(scheduler: Scheduler,
                             mcmc_sampler: MCMCSampler,
                             mcmc_correction: Correction,
                             mcmc_steps: int,
                             post_mcmc_update: Callable) -> Callable:
    # Move
    def mcmc_kernel(previous_carry: Tuple[CDict, CDict],
                    _: Any) -> Tuple[Tuple[CDict, CDict], Tuple[CDict, CDict]]:
        previous_state, previous_extra = previous_carry
        reject_state = previous_state.copy()
        reject_extra = previous_extra.copy()
        reject_extra.iter += 1

        reject_state, reject_extra = mcmc_sampler.always(scheduler.scheduled_scenario,
                                                         reject_state,
                                                         reject_extra)

        proposed_state, proposed_extra = mcmc_sampler.proposal(scheduler.scheduled_scenario,
                                                               reject_state,
                                                               reject_extra)

        corrected_state, corrected_extra = mcmc_correction(scheduler.scheduled_scenario, mcmc_sampler,
                                                           reject_state, reject_extra,
                                                           proposed_state, proposed_extra)

        return (corrected_state, corrected_extra), (corrected_state, corrected_extra)

    def mcmc_chain(start_state: CDict,
                   start_extra: CDict):
        start_state, start_extra = startup_mcmc(scheduler.scheduled_scenario,
                                                mcmc_sampler,
                                                None,
                                                mcmc_correction,
                                                start_state,
                                                start_extra)

        final_carry, chain = scan(mcmc_kernel,
                                  (start_state, start_extra),
                                  None,
                                  length=mcmc_steps)
        return chain

    mcmc_chains_vec = vmap(mcmc_chain)

    def smc_sampler_advance(previous_full_state: CDict,
                            previous_full_extra: CDict,
                            previous_schedule_param: float,
                            new_schedule_param: float,
                            previous_initial_potential_evals: np.ndarray,
                            previous_target_potential_evals: np.ndarray) -> Tuple[CDict, CDict]:
        if hasattr(previous_full_state, 'weight_schedule'):
            del previous_full_state.weight_schedule
        new_full_state = previous_full_state.copy()
        new_full_extra = previous_full_extra.copy()

        # Weight
        log_weights = scheduler.log_weight_function(previous_schedule_param,
                                                    new_schedule_param,
                                                    previous_initial_potential_evals,
                                                    previous_target_potential_evals)

        # Resample
        int_random_key = random.split(new_full_extra.random_key[0], 1)[0]
        sample_inds = random.categorical(int_random_key, log_weights, shape=(len(log_weights),))
        new_full_state = new_full_state[sample_inds]

        scheduler.scheduled_scenario.schedule_param = new_schedule_param

        new_full_state, new_full_extra = mcmc_chains_vec(new_full_state, new_full_extra)

        new_full_state, new_full_extra = post_mcmc_update(previous_full_state, previous_full_extra,
                                                          new_full_state, new_full_extra)

        new_full_state.weight_schedule = new_schedule_param
        return new_full_state, new_full_extra

    return smc_sampler_advance
