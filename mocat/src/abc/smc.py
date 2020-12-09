########################################################################################################################
# Module: abc/smc.py
# Description: SMC sampler for ABC problems.
#
# Web: https://github.com/SamDuffield/mocat
########################################################################################################################


from typing import Union, Tuple, Callable, Type
from time import time

from jax import numpy as np, random, vmap
from jax.lax import scan

from mocat.src.abc.abc import ABCSampler, ABCScenario
from mocat.src.abc.standard_abc import RandomWalkABC
from mocat.src.core import CDict
from mocat.src.utils import while_loop_stacked
from mocat.src.mcmc.corrections import Correction
from mocat.src.smc_samplers import default_post_mcmc_update
from mocat.src.mcmc.run import startup_mcmc, check_correction


def default_post_metropolis_abc_mcmc_update(previous_full_state: CDict,
                                            previous_full_extra: CDict,
                                            new_enlarged_state: CDict,
                                            new_enlarged_extra: CDict) -> Tuple[CDict, CDict]:
    new_full_state = new_enlarged_state[:, -1]
    new_full_extra = new_enlarged_extra[:, -1]
    new_full_extra.parameters = new_full_extra.parameters[:, -1]
    new_full_state.alpha = new_enlarged_state.alpha.mean(1)

    d = new_full_state.value.shape[-1]
    new_full_extra.parameters.stepsize = np.cov(new_full_state.value, rowvar=False) / d * 2.38 ** 2

    return new_full_state, new_full_extra


def run_abc_smc_sampler(abc_scenario: ABCScenario,
                        n_samps: int,
                        random_key: np.ndarray,
                        mcmc_sampler: ABCSampler = None,
                        mcmc_correction: Union[None, str, Correction, Type[Correction]] = 'sampler_default',
                        mcmc_steps: int = 20,
                        preschedule: np.ndarray = None,
                        initial_state: CDict = None,
                        post_mcmc_update: Callable = default_post_mcmc_update,
                        max_iter: int = 1000,
                        threshold_quantile_retain: float = 0.95,
                        continuation_func: Callable = None,
                        name: str = None) -> CDict:
    if initial_state is None:
        random_key, sub_key = random.split(random_key)
        x0 = vmap(abc_scenario.prior_sample)(random.split(sub_key, n_samps))
        initial_state = CDict(value=x0)

    if mcmc_sampler is None:
        mcmc_sampler = RandomWalkABC(stepsize=np.cov(initial_state.value, rowvar=False) / abc_scenario.dim * 2.38 ** 2)

    mcmc_sampler, mcmc_correction = check_correction(mcmc_sampler, mcmc_correction)

    initial_extra = CDict(random_key=None, iter=0)

    initial_state, initial_extra = vmap(
        lambda state: startup_mcmc(abc_scenario, mcmc_sampler, None, mcmc_correction,
                                   state, initial_extra))(initial_state)

    if None in initial_extra.parameters.__dict__.values():
        raise ValueError(f'None found in {mcmc_sampler.name} parameters (within ABC-SMC sampler): '
                         + str(mcmc_sampler.parameters))

    if post_mcmc_update is None:
        if hasattr(initial_state, 'alpha'):
            post_mcmc_update = default_post_metropolis_abc_mcmc_update
        else:
            post_mcmc_update = default_post_mcmc_update

    initial_extra.random_key = random.split(random_key, n_samps)

    start = time()

    if preschedule is not None:
        out_samps = _run_prescheduled_abc_smc_sampler(abc_scenario, mcmc_sampler, mcmc_correction, mcmc_steps,
                                                      preschedule, initial_state, initial_extra, post_mcmc_update)
    else:
        if continuation_func is None:
            continuation_func = lambda state, extra: state.alpha.mean() > 0.015

        out_samps = _run_adaptive_abc_smc_sampler(abc_scenario, mcmc_sampler, mcmc_correction, mcmc_steps,
                                                  initial_state, initial_extra,
                                                  post_mcmc_update, max_iter,
                                                  threshold_quantile_retain, continuation_func)

    out_samps.value.block_until_ready()
    end = time()
    out_samps.time = end - start

    if name is None:
        name = abc_scenario.name + ": ABC-SMC - " + mcmc_sampler.name
    out_samps.name = name
    return out_samps


def _run_prescheduled_abc_smc_sampler(abc_scenario: ABCScenario,
                                      mcmc_sampler: ABCSampler,
                                      mcmc_correction: Correction,
                                      mcmc_steps: int,
                                      preschedule: np.ndarray,
                                      initial_state: CDict,
                                      initial_extra: CDict,
                                      post_mcmc_update: Callable) -> CDict:
    initial_state.threshold_schedule = preschedule[0]

    abc_smc_sampler_advance = init_abc_smc_sampler_advance(abc_scenario,
                                                           mcmc_sampler,
                                                           mcmc_correction,
                                                           mcmc_steps,
                                                           post_mcmc_update)

    def prescheduled_smc_kernel(previous_carry: Tuple[CDict, CDict],
                                iter_ind: int) -> Tuple[Tuple[CDict, CDict], CDict]:
        previous_state, previous_extra = previous_carry

        new_state, new_extra = abc_smc_sampler_advance(abc_scenario,
                                                       previous_state,
                                                       previous_extra,
                                                       preschedule[iter_ind])
        return (new_state, new_extra), new_state

    final_carry, chain = scan(prescheduled_smc_kernel,
                              (initial_state, initial_extra),
                              np.arange(1, len(preschedule)))
    return chain


def _run_adaptive_abc_smc_sampler(abc_scenario: ABCScenario,
                                  mcmc_sampler: ABCSampler,
                                  mcmc_correction: Correction,
                                  mcmc_steps: int,
                                  initial_state: CDict,
                                  initial_extra: CDict,
                                  post_mcmc_update: Callable,
                                  max_iter: int,
                                  threshold_quantile_retain: float,
                                  continuation_func: Callable) -> CDict:
    initial_state.threshold_schedule = np.quantile(initial_state.distance, threshold_quantile_retain)

    abc_smc_sampler_advance = init_abc_smc_sampler_advance(abc_scenario,
                                                           mcmc_sampler,
                                                           mcmc_correction,
                                                           mcmc_steps,
                                                           post_mcmc_update)

    def adaptive_abc_smc_kernel(previous_state: CDict,
                                previous_extra: CDict) \
            -> Tuple[CDict, CDict]:
        # Determine next threshold
        new_threshold = np.quantile(previous_state.distance, threshold_quantile_retain)

        new_state, new_extra = abc_smc_sampler_advance(previous_state,
                                                       previous_extra,
                                                       new_threshold)

        return new_state, new_extra

    chain = while_loop_stacked(continuation_func,
                               adaptive_abc_smc_kernel,
                               (initial_state, initial_extra),
                               max_iter)
    return chain


def init_abc_smc_sampler_advance(abc_scenario: ABCScenario,
                                 mcmc_sampler: ABCSampler,
                                 mcmc_correction: Correction,
                                 mcmc_steps: int,
                                 post_mcmc_update: Callable) -> Callable:
    # Move
    def mcmc_kernel(previous_carry: Tuple[CDict, CDict],
                    iter_ind: int) -> Tuple[Tuple[CDict, CDict], Tuple[CDict, CDict]]:
        previous_state, previous_extra = previous_carry
        reject_state = previous_state.copy()
        reject_extra = previous_extra.copy()
        reject_extra.iter = iter_ind

        reject_state, reject_extra = mcmc_sampler.always(abc_scenario,
                                                         reject_state,
                                                         reject_extra)

        proposed_state, proposed_extra = mcmc_sampler.proposal(abc_scenario,
                                                               reject_state,
                                                               reject_extra)

        corrected_state, corrected_extra = mcmc_correction(abc_scenario, mcmc_sampler,
                                                           reject_state, reject_extra,
                                                           proposed_state, proposed_extra)

        return (corrected_state, corrected_extra), (corrected_state, corrected_extra)

    def mcmc_chain(start_state: CDict,
                   start_extra: CDict):
        final_carry, chain = scan(mcmc_kernel,
                                  (start_state, start_extra),
                                  np.arange(1, mcmc_steps + 1))
        return chain

    mcmc_chains_vec = vmap(mcmc_chain)

    def smc_sampler_advance(previous_full_state: CDict,
                            previous_full_extra: CDict,
                            new_threshold: float) -> Tuple[CDict, CDict]:
        if hasattr(previous_full_state, 'threshold'):
            del previous_full_state.threshold
        new_full_state = previous_full_state.copy()
        new_full_extra = previous_full_extra.copy()

        n = previous_full_state.value.shape[0]

        # Resample
        int_random_key, _ = random.split(new_full_extra.random_key[0])
        sample_inds = random.choice(int_random_key, a=n, p=previous_full_state.distance < new_threshold, shape=(n,))
        new_full_state = new_full_state[sample_inds]

        abc_scenario.new_threshold = new_threshold

        new_full_state, new_full_extra = mcmc_chains_vec(new_full_state, new_full_extra)

        new_full_state, new_full_extra = post_mcmc_update(previous_full_state, previous_full_extra,
                                                          new_full_state, new_full_extra)

        new_full_state.threshold = new_threshold
        return new_full_state, new_full_extra

    return smc_sampler_advance
