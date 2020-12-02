########################################################################################################################
# Module: backward/filters.py
# Description: Backward simulation for state-space models.
#
# Web: https://github.com/SamDuffield/mocat
########################################################################################################################

from typing import Tuple
from time import time
from functools import partial

from jax import numpy as np, random, vmap, jit
from jax.lax import while_loop, scan, cond, map

from mocat.src.core import CDict
from mocat.src.ssm.ssm import StateSpaceModel
from mocat.src.ssm.filters import ParticleFilter, run_particle_filter_for_marginals

bound_inflation = 1.01


def full_resample_single(ssm_scenario: StateSpaceModel,
                         x0_all: np.ndarray,
                         t: float,
                         x1_single: np.ndarray,
                         tplus1: float,
                         x0_log_weights: np.ndarray,
                         random_key: np.ndarray) -> np.ndarray:
    log_weights = x0_log_weights - vmap(ssm_scenario.transition_potential, (0, None, None, None))(x0_all, t,
                                                                                                  x1_single, tplus1)
    return x0_all[random.categorical(random_key, log_weights)]


def full_resampling(ssm_scenario: StateSpaceModel,
                    x0_all: np.ndarray,
                    t: float,
                    x1_all: np.ndarray,
                    tplus1: float,
                    x0_log_weights: np.ndarray,
                    random_key: np.ndarray) -> np.ndarray:
    return vmap(full_resample_single, (None, None, None, 0, None, None, 0)) \
        (ssm_scenario, x0_all, t, x1_all, tplus1, x0_log_weights, random.split(random_key, len(x1_all)))


def full_resample_single_cond(not_yet_accepted: bool,
                              x0_false: np.ndarray,
                              ssm_scenario: StateSpaceModel,
                              x0_all: np.ndarray,
                              t: float,
                              x1_single: np.ndarray,
                              tplus1: float,
                              x0_log_weights: np.ndarray,
                              random_key: np.ndarray) -> np.ndarray:
    return cond(not_yet_accepted,
                lambda _: full_resample_single(ssm_scenario, x0_all, t, x1_single, tplus1, x0_log_weights, random_key),
                lambda _: x0_false,
                None)


def rejection_proposal_single(ssm_scenario: StateSpaceModel,
                              x0_all: np.ndarray,
                              t: float,
                              x1_single: np.ndarray,
                              tplus1: float,
                              x0_log_weights: np.ndarray,
                              bound: float,
                              random_key: np.ndarray) \
        -> Tuple[np.ndarray, float, bool, np.ndarray]:
    random_key, choice_key, uniform_key = random.split(random_key, 3)
    x0_single = x0_all[random.categorical(choice_key, x0_log_weights)]
    conditional_dens = np.exp(-ssm_scenario.transition_potential(x0_single, t, x1_single, tplus1))
    return x0_single, conditional_dens, random.uniform(uniform_key) > conditional_dens / bound, random_key


def rejection_proposal_single_cond(not_yet_accepted: bool,
                                   x0_false: np.ndarray,
                                   ssm_scenario: StateSpaceModel,
                                   x0_all: np.ndarray,
                                   t: float,
                                   x1_single: np.ndarray,
                                   tplus1: float,
                                   x0_log_weights: np.ndarray,
                                   bound: float,
                                   random_key: np.ndarray) -> Tuple[np.ndarray, float, bool, np.ndarray]:
    return cond(not_yet_accepted,
                lambda _: rejection_proposal_single(ssm_scenario,
                                                    x0_all, t,
                                                    x1_single, tplus1,
                                                    x0_log_weights, bound, random_key),
                lambda _: (x0_false, 0., False, random_key),
                None)


def rejection_proposal_all(ssm_scenario: StateSpaceModel,
                           x0_all: np.ndarray,
                           t: float,
                           x1_all: np.ndarray,
                           tplus1: float,
                           x0_log_weights: np.ndarray,
                           not_yet_accepted_arr: np.ndarray,
                           x0_all_sampled: np.ndarray,
                           bound: float,
                           random_keys: np.ndarray,
                           rejection_iter: int) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, int]:
    n = len(x1_all)
    mapped_tup = map(lambda i: rejection_proposal_single_cond(not_yet_accepted_arr[i],
                                                              x0_all_sampled[i],
                                                              ssm_scenario,
                                                              x0_all,
                                                              t,
                                                              x1_all[i],
                                                              tplus1,
                                                              x0_log_weights,
                                                              bound,
                                                              random_keys[i]), np.arange(n))
    x0_all_sampled, dens_evals, not_yet_accepted_arr_new, random_keys = mapped_tup

    # Check if we need to start again
    max_dens = np.max(dens_evals)
    reset_bound = max_dens > bound
    bound = np.where(reset_bound, max_dens * bound_inflation, bound)
    not_yet_accepted_arr_new = np.where(reset_bound, np.ones(n, dtype='bool'), not_yet_accepted_arr_new)
    return not_yet_accepted_arr_new, x0_all_sampled, bound, random_keys, rejection_iter + 1


# @partial(jit, static_argnums=(0,))
# def rejection_proposal_single(ssm_scenario, x0_all, t, x1_single, tplus1, x0_log_weights, carry: tuple) \
#         -> Tuple[np.ndarray, float, bool, np.ndarray, int]:
#     _, int_bound, _, int_random_key, int_mr = carry
#     int_random_key, choice_key, uniform_key = random.split(int_random_key, 3)
#     x0_single = x0_all[random.categorical(choice_key, x0_log_weights)]
#     conditional_dens = np.exp(-ssm_scenario.transition_potential(x0_single, t, x1_single, tplus1))
#     return x0_single, np.maximum(conditional_dens, int_bound), random.uniform(
#         uniform_key) > conditional_dens / int_bound, int_random_key, int_mr + 1
#
#
#
# @partial(jit, static_argnums=(0,))
# def rejection_sample_single(ssm_scenario: StateSpaceModel,
#                             not_yet_accepted: bool,
#                             x0_false: np.ndarray,
#                             x0_all: np.ndarray,
#                             t: float,
#                             x1_single: np.ndarray,
#                             tplus1: float,
#                             x0_log_weights: np.ndarray,
#                             bound: float,
#                             maximum_rejections: int,
#                             random_key: np.ndarray) -> Tuple[np.ndarray, float, bool, np.ndarray, int]:
#
#     return while_loop(lambda tup: np.logical_and(tup[2], tup[4] < maximum_rejections),
#                       lambda tup: rejection_proposal_single(ssm_scenario, x0_all, t, x1_single, tplus1, x0_log_weights, tup),
#                       (x0_false, bound, not_yet_accepted, random_key, 1))


@partial(jit, static_argnums=(0, 7))
def rejection_resampling(ssm_scenario: StateSpaceModel,
                         x0_all: np.ndarray,
                         t: float,
                         x1_all: np.ndarray,
                         tplus1: float,
                         x0_log_weights: np.ndarray,
                         random_key: np.ndarray,
                         maximum_rejections: int,
                         init_bound_param: float) -> Tuple[np.ndarray, np.ndarray, float, int]:
    rejection_initial_keys = random.split(random_key, 3)
    n = len(x1_all)

    # Prerun to initiate bound
    x0_initial = x0_all[random.categorical(rejection_initial_keys[0], x0_log_weights, shape=(n,))]
    initial_cond_dens = np.exp(-vmap(ssm_scenario.transition_potential,
                                     (0, None, 0, None))(x0_initial, t, x1_all, tplus1))
    max_cond_dens = np.max(initial_cond_dens)
    initial_bound = np.where(max_cond_dens > init_bound_param, max_cond_dens * bound_inflation, init_bound_param)
    initial_not_yet_accepted_arr = random.uniform(rejection_initial_keys[1], (n,)) > initial_cond_dens / initial_bound

    out_tup = while_loop(lambda tup: np.logical_and(tup[0].sum() > 0, tup[-1] < maximum_rejections),
                         lambda tup: rejection_proposal_all(ssm_scenario, x0_all, t, x1_all, tplus1, x0_log_weights,
                                                            *tup),
                         (initial_not_yet_accepted_arr,
                          x0_initial,
                          initial_bound,
                          random.split(rejection_initial_keys[2], n),
                          1))
    not_yet_accepted_arr, final_particles, final_bound, random_keys, rej_attempted = out_tup

    final_particles = map(lambda i: full_resample_single_cond(not_yet_accepted_arr[i],
                                                              final_particles[i],
                                                              ssm_scenario,
                                                              x0_all,
                                                              t,
                                                              x1_all[i],
                                                              tplus1,
                                                              x0_log_weights,
                                                              random_keys[i]), np.arange(n))

    return final_particles


#
# def rejection_proposal_single(ssm_scenario: StateSpaceModel,
#                               x0_all: np.ndarray,
#                               t: float,
#                               x1_single: np.ndarray,
#                               tplus1: float,
#                               bound: float,
#                               cat_ind: int,
#                               unif_val: float) \
#         -> Tuple[np.ndarray, float, bool]:
#     x0_single = x0_all[cat_ind]
#     conditional_dens = np.exp(-ssm_scenario.transition_potential(x0_single, t, x1_single, tplus1))
#     return x0_single, conditional_dens, unif_val > conditional_dens / bound
#
#
# def rejection_proposal_single_cond(not_yet_accepted: bool,
#                                    x0_false: np.ndarray,
#                                    ssm_scenario: StateSpaceModel,
#                                    x0_all: np.ndarray,
#                                    t: float,
#                                    x1_single: np.ndarray,
#                                    tplus1: float,
#                                    bound: float,
#                                    cat_ind: int,
#                                    unif_val: float) -> Tuple[np.ndarray, float, bool]:
#     return cond(not_yet_accepted,
#                 lambda _: rejection_proposal_single(ssm_scenario,
#                                                     x0_all, t,
#                                                     x1_single, tplus1,
#                                                     bound, cat_ind, unif_val),
#                 lambda _: (x0_false, 0., False),
#                 None)
#
#
# def rejection_proposal_all(ssm_scenario: StateSpaceModel,
#                            x0_all: np.ndarray,
#                            t: float,
#                            x1_all: np.ndarray,
#                            tplus1: float,
#                            not_yet_accepted_arr: np.ndarray,
#                            x0_all_sampled: np.ndarray,
#                            bound: float,
#                            cat_inds: np.ndarray,
#                            unif_vals: np.ndarray) -> Tuple[Tuple[np.ndarray, np.ndarray, float], None]:
#     n = len(x1_all)
#     mapped_tup = map(lambda i: rejection_proposal_single_cond(not_yet_accepted_arr[i],
#                                                               x0_all_sampled[i],
#                                                               ssm_scenario,
#                                                               x0_all,
#                                                               t,
#                                                               x1_all[i],
#                                                               tplus1,
#                                                               bound,
#                                                               cat_inds[i],
#                                                               unif_vals[i]), np.arange(n))
#     x0_all_sampled, dens_evals, not_yet_accepted_arr_new = mapped_tup
#
#     # Check if we need to start again
#     max_dens = np.max(dens_evals)
#     reset_bound = max_dens > bound
#     bound = np.where(reset_bound, max_dens * bound_inflation, bound)
#     not_yet_accepted_arr_new = np.where(reset_bound, np.ones(n, dtype='bool'), not_yet_accepted_arr_new)
#     return not_yet_accepted_arr_new, x0_all_sampled, bound
#
#
# @partial(jit, static_argnums=(0, 7))
# def rejection_resampling(ssm_scenario: StateSpaceModel,
#                          x0_all: np.ndarray,
#                          t: float,
#                          x1_all: np.ndarray,
#                          tplus1: float,
#                          x0_log_weights: np.ndarray,
#                          random_key: np.ndarray,
#                          maximum_rejections: int,
#                          init_bound_param: float) -> Tuple[np.ndarray, np.ndarray, float]:
#     n = len(x1_all)
#
#     random_key, cat_key, unif_key, full_resamp_key = random.split(random_key, 4)
#
#     cat_inds = random.categorical(cat_key, logits=x0_log_weights, shape=(maximum_rejections, n))
#     unif_vals = random.uniform(unif_key, shape=(maximum_rejections, n))
#
#     # Prerun to initiate bound
#     x0_initial = x0_all[cat_inds[0]]
#     initial_cond_dens = np.exp(-vmap(ssm_scenario.transition_potential,
#                                      (0, None, 0, None))(x0_initial, t, x1_all, tplus1))
#     max_cond_dens = np.max(initial_cond_dens)
#     initial_bound = np.where(max_cond_dens > init_bound_param, max_cond_dens * bound_inflation, init_bound_param)
#     initial_not_yet_accepted_arr = unif_vals[0] > initial_cond_dens / initial_bound
#
#     # out_tup, _ = scan(lambda carry, rej_ind: (rejection_proposal_all(ssm_scenario, x0_all, t, x1_all, tplus1,
#     #                                                                 carry[0], carry[1],
#     #                                                                 carry[2], cat_inds[rej_ind], unif_vals[rej_ind]), None),
#     #                   (initial_not_yet_accepted_arr, x0_initial, initial_bound),
#     #                   np.arange(1, maximum_rejections))
#
#     out_tup = fori_loop(1, maximum_rejections,
#                         lambda rej_ind, carry: rejection_proposal_all(ssm_scenario, x0_all, t, x1_all, tplus1,
#                                                                       carry[0], carry[1],
#                                                                       carry[2], cat_inds[rej_ind], unif_vals[rej_ind]),
#                         (initial_not_yet_accepted_arr, x0_initial, initial_bound))
#
#     not_yet_accepted_arr, final_rej_particles, final_bound = out_tup
#
#     full_resamp_keys = random.split(full_resamp_key, n)
#
#     full_particles = map(lambda i: full_resample_single_cond(not_yet_accepted_arr[i],
#                                                              final_rej_particles[i],
#                                                              ssm_scenario,
#                                                              x0_all,
#                                                              t,
#                                                              x1_all[i],
#                                                              tplus1,
#                                                              x0_log_weights,
#                                                              full_resamp_keys[i]), np.arange(n))
#
#     return full_particles, not_yet_accepted_arr, final_bound


@partial(jit, static_argnums=(0, 3, 4))
def backward_simulation(ssm_scenario: StateSpaceModel,
                        marginal_particles: CDict,
                        random_key: np.ndarray,
                        n_samps: int,
                        maximum_rejections: int,
                        init_bound_param: float) -> CDict:
    marg_particles_vals = marginal_particles.value
    times = marginal_particles.t
    marginal_log_weights = marginal_particles.log_weights

    T, n_pf, d = marg_particles_vals.shape

    t_keys = random.split(random_key, T)
    final_particle_vals = marg_particles_vals[-1, random.categorical(t_keys[-1],
                                                                     marginal_log_weights[-1],
                                                                     shape=(n_samps,))]

    def back_sim_body(x_tplus1_all: np.ndarray, ind: int):
        x_t_all = rejection_resampling(ssm_scenario,
                                       marg_particles_vals[ind], times[ind],
                                       x_tplus1_all, times[ind + 1],
                                       marginal_log_weights[ind], t_keys[ind], maximum_rejections, init_bound_param)
        return x_t_all, x_t_all

    _, back_sim_particles = scan(back_sim_body,
                                 final_particle_vals,
                                 np.arange(T - 2, -1, -1), unroll=1)

    out_samps = marginal_particles.copy()
    out_samps.value = np.vstack([back_sim_particles[::-1], final_particle_vals[np.newaxis]])
    del out_samps.log_weights
    return out_samps


@partial(jit, static_argnums=(0, 3))
def backward_simulation_full(ssm_scenario: StateSpaceModel,
                             marginal_particles: CDict,
                             random_key: np.ndarray,
                             n_samps: int) -> CDict:
    marg_particles_vals = marginal_particles.value
    times = marginal_particles.t
    marginal_log_weights = marginal_particles.log_weights

    T, n_pf, d = marg_particles_vals.shape

    t_keys = random.split(random_key, T)
    final_particle_vals = marg_particles_vals[-1, random.categorical(t_keys[-1],
                                                                     marginal_log_weights[-1],
                                                                     shape=(n_samps,))]

    def back_sim_body(x_tplus1_all: np.ndarray, ind: int):
        x_t_all = full_resampling(ssm_scenario, marg_particles_vals[ind], times[ind],
                                  x_tplus1_all, times[ind + 1], marginal_log_weights[ind], t_keys[ind])
        return x_t_all, x_t_all

    _, back_sim_particles = scan(back_sim_body,
                                 final_particle_vals,
                                 np.arange(T - 2, -1, -1))

    out_samps = marginal_particles.copy()
    out_samps.value = np.vstack([back_sim_particles[::-1], final_particle_vals[np.newaxis]])
    del out_samps.log_weights
    return out_samps


def forward_filtering_backward_simulation(ssm_scenario: StateSpaceModel,
                                          particle_filter: ParticleFilter,
                                          y: np.ndarray,
                                          t: np.ndarray,
                                          n_samps: int,
                                          random_key: np.ndarray,
                                          n_pf: int = None,
                                          ess_threshold: float = 0.5,
                                          maximum_rejections: int = 0,
                                          transition_dens_bound_parameter: float = 0.) -> CDict:
    pf_key, bsi_key = random.split(random_key)

    if n_pf is None:
        n_pf = n_samps

    pf_start = time()
    pf_samps = run_particle_filter_for_marginals(ssm_scenario,
                                                 particle_filter,
                                                 y,
                                                 t,
                                                 pf_key,
                                                 n=n_pf,
                                                 ess_threshold=ess_threshold)
    pf_samps.value.block_until_ready()
    pf_end = time()
    if maximum_rejections > 0:
        bsi_samps = backward_simulation(ssm_scenario,
                                        pf_samps,
                                        bsi_key,
                                        n_samps,
                                        maximum_rejections,
                                        transition_dens_bound_parameter)
    else:
        bsi_samps = backward_simulation_full(ssm_scenario,
                                             pf_samps,
                                             bsi_key,
                                             n_samps)
    bsi_samps.value.block_until_ready()
    bsi_end = time()

    bsi_samps.time = bsi_end - pf_start
    bsi_samps.pf_time = pf_end - pf_start
    bsi_samps.bsi_time = bsi_end - pf_end

    return bsi_samps
