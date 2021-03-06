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
from mocat.src.core import cdict
from mocat.src.utils import while_loop_stacked
from mocat.src.mcmc.corrections import Correction, Metropolis
from mocat.src.smc_samplers import default_post_mcmc_update
from mocat.src.mcmc.run import startup_mcmc, check_correction


def default_post_metropolis_abc_mcmc_update(previous_full_state: cdict,
                                            previous_full_extra: cdict,
                                            new_enlarged_state: cdict,
                                            new_enlarged_extra: cdict) -> Tuple[cdict, cdict]:
    new_full_state = new_enlarged_state[:, -1]
    new_full_extra = new_enlarged_extra[:, -1]
    new_full_extra.parameters = new_full_extra.parameters[:, -1]
    new_full_state.alpha = new_enlarged_state.alpha.mean(1)

    d = new_full_state.value.shape[-1]
    new_full_extra.parameters.stepsize = vmap(np.cov, (1,))(new_full_state.value) / d * 2.38 ** 2
    # new_full_extra.parameters.stepsize = vmap(np.cov, (1,))(new_full_state.value)

    new_full_extra.parameters.threshold = new_full_extra.parameters.threshold[-1, -1]
    return new_full_state, new_full_extra


def run_abc_smc_sampler(abc_scenario: ABCScenario,
                        n_samps: int,
                        random_key: np.ndarray,
                        mcmc_sampler: ABCSampler = None,
                        mcmc_correction: Union[None, str, Correction, Type[Correction]] = 'sampler_default',
                        mcmc_steps: int = 10,
                        preschedule: np.ndarray = None,
                        initial_state: cdict = None,
                        initial_extra: cdict = None,
                        post_mcmc_update: Callable = default_post_mcmc_update,
                        max_iter: int = 100,
                        threshold_quantile_retain: float = 0.75,
                        continuation_func: Callable = None,
                        name: str = None) -> cdict:
    if initial_state is None:
        random_key, sub_key = random.split(random_key)
        x0 = vmap(abc_scenario.prior_sample)(random.split(sub_key, n_samps))
        initial_state = cdict(value=x0)

    if mcmc_sampler is None:
        mcmc_sampler = RandomWalkABC(stepsize=vmap(np.cov, (1,))(initial_state.value) / abc_scenario.dim * 2.38 ** 2,
                                     threshold=np.inf)
        # mcmc_sampler = RandomWalkABC(stepsize=vmap(np.cov, (1,))(initial_state.value), threshold=np.inf)

    mcmc_sampler, mcmc_correction = check_correction(mcmc_sampler, mcmc_correction)

    if initial_extra is None:
        initial_extra = cdict(iter=0)
    initial_extra.random_key = random.split(random_key, n_samps)

    init_extra_dict_map_inds = {key: 0 if isinstance(value, np.ndarray) and len(value) == n_samps else None
                                for key, value in initial_extra.__dict__.items()}
    init_extra_cdict_map_inds = cdict(**init_extra_dict_map_inds)

    initial_state, initial_extra = vmap(
        lambda state, extra: startup_mcmc(abc_scenario, mcmc_sampler, None, mcmc_correction,
                                          state, extra), (0, init_extra_cdict_map_inds))(initial_state,
                                                                                         initial_extra)
    initial_state.threshold_schedule = np.inf

    if None in initial_extra.parameters.__dict__.values():
        raise ValueError(f'None found in {mcmc_sampler.name} parameters (within ABC-SMC sampler): '
                         + str(mcmc_sampler.parameters))

    if post_mcmc_update is None:
        if isinstance(mcmc_correction, Metropolis):
            post_mcmc_update = default_post_metropolis_abc_mcmc_update
        else:
            post_mcmc_update = default_post_mcmc_update

    start = time()

    if preschedule is not None:
        out_samps = _run_prescheduled_abc_smc_sampler(abc_scenario, mcmc_sampler, mcmc_correction, mcmc_steps,
                                                      preschedule, initial_state, initial_extra, post_mcmc_update)
    else:
        if continuation_func is None:
            continuation_func = lambda state, extra: state.alpha.mean() > 0.01

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
                                      initial_state: cdict,
                                      initial_extra: cdict,
                                      post_mcmc_update: Callable) -> cdict:
    initial_state.threshold_schedule = preschedule[0]

    abc_smc_sampler_advance = init_abc_smc_sampler_advance(abc_scenario,
                                                           mcmc_sampler,
                                                           mcmc_correction,
                                                           mcmc_steps,
                                                           post_mcmc_update)

    def prescheduled_smc_kernel(previous_carry: Tuple[cdict, cdict],
                                iter_ind: int) -> Tuple[Tuple[cdict, cdict], cdict]:
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
                                  initial_state: cdict,
                                  initial_extra: cdict,
                                  post_mcmc_update: Callable,
                                  max_iter: int,
                                  threshold_quantile_retain: float,
                                  continuation_func: Callable) -> cdict:
    initial_state.threshold_schedule = initial_extra.parameters.threshold[0]

    abc_smc_sampler_advance = init_abc_smc_sampler_advance(abc_scenario,
                                                           mcmc_sampler,
                                                           mcmc_correction,
                                                           mcmc_steps,
                                                           post_mcmc_update)

    def adaptive_abc_smc_kernel(previous_state: cdict,
                                previous_extra: cdict) \
            -> Tuple[cdict, cdict]:
        # Determine next threshold
        new_threshold = np.quantile(previous_state.distance, threshold_quantile_retain)

        new_state, new_extra = abc_smc_sampler_advance(previous_state,
                                                       previous_extra,
                                                       new_threshold)

        return new_state, new_extra

    chain = while_loop_stacked(continuation_func,
                               adaptive_abc_smc_kernel,
                               (initial_state.copy(), initial_extra.copy()),
                               max_iter)
    return chain


def init_abc_smc_sampler_advance(abc_scenario: ABCScenario,
                                 mcmc_sampler: ABCSampler,
                                 mcmc_correction: Correction,
                                 mcmc_steps: int,
                                 post_mcmc_update: Callable) -> Callable:
    # Move
    def mcmc_kernel(previous_carry: Tuple[cdict, cdict],
                    iter_ind: int) -> Tuple[Tuple[cdict, cdict], Tuple[cdict, cdict]]:
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

    def mcmc_chain(start_state: cdict,
                   start_extra: cdict):
        final_carry, chain = scan(mcmc_kernel,
                                  (start_state, start_extra),
                                  np.arange(1, mcmc_steps + 1))
        return chain

    mcmc_chains_vec = vmap(mcmc_chain)

    def smc_sampler_advance(previous_full_state: cdict,
                            previous_full_extra: cdict,
                            new_threshold: float) -> Tuple[cdict, cdict]:
        if hasattr(previous_full_state, 'threshold_schedule'):
            del previous_full_state.threshold_schedule
        new_full_state = previous_full_state.copy()
        new_full_extra = previous_full_extra.copy()

        n = previous_full_state.value.shape[0]

        new_full_extra.parameters.threshold = new_threshold * np.ones(n)

        # Resample
        int_random_key, _ = random.split(new_full_extra.random_key[0])
        sample_inds = random.choice(int_random_key, a=n,
                                    p=previous_full_state.distance < new_threshold, shape=(n,))
        new_full_state = new_full_state[sample_inds]
        new_full_extra = new_full_extra[sample_inds]
        new_full_extra.random_key = previous_full_extra.random_key

        new_full_enlarged_state, new_full_enlarged_extra = mcmc_chains_vec(new_full_state, new_full_extra)

        new_full_state, new_full_extra = post_mcmc_update(previous_full_state, previous_full_extra,
                                                          new_full_enlarged_state, new_full_enlarged_extra)

        new_full_state.threshold_schedule = new_threshold
        return new_full_state, new_full_extra

    return smc_sampler_advance
