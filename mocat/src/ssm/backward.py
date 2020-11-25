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


# @partial(jit, static_argnums=(0,))
# def full_sample(ssm_scenario: StateSpaceModel,
#                 current_t_particle: np.ndarray,
#                 t_marginal_particles: np.ndarray,
#                 t: float,
#                 tplus1_particle: np.ndarray,
#                 tplus1: float,
#                 reject_bool: bool,
#                 t_marg_log_weights: np.ndarray,
#                 random_key: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
#     def full_sample_true() -> Tuple[np.ndarray, np.ndarray]:
#         random_key, cat_key = random.split(random_key)
#         adjusted_log_weights = t_marg_log_weights \
#                                - vmap(ssm_scenario.transition_potential, (0, None, None, None))(t_marginal_particles, t,
#                                                                                                 tplus1_particle, tplus1)
#         resampled_ind = random.categorical(cat_key, adjusted_log_weights)
#         return t_marginal_particles[resampled_ind], random_key
#
#     return cond(reject_bool,
#                 lambda tup: full_sample_true(),
#                 lambda tup: tup,
#                 (current_t_particle, random_key))


def rejection_proposal_true(ssm_scenario: StateSpaceModel,
                            t_marginal_particles: np.ndarray,
                            t: float,
                            tplus1_particle: np.ndarray,
                            tplus1: float,
                            t_marg_log_weights: np.ndarray,
                            bound_param: float,
                            random_key: np.ndarray) -> Tuple[np.ndarray, bool, float, np.ndarray]:
    random_key, cat_key, uniform_key = random.split(random_key, 3)
    resampled_ind = random.categorical(cat_key, t_marg_log_weights)
    t_particle = t_marginal_particles[resampled_ind]
    transition_dens = np.exp(-ssm_scenario.transition_potential(t_particle, t, tplus1_particle, tplus1))
    return t_particle, \
           random.uniform(uniform_key) > transition_dens / bound_param, \
           transition_dens, \
           random_key


def rejection_proposal(ssm_scenario: StateSpaceModel,
                       current_t_particle: np.ndarray,
                       t_marginal_particles: np.ndarray,
                       t: float,
                       tplus1_particle: np.ndarray,
                       tplus1: float,
                       reject_bool: bool,
                       t_marg_log_weights: np.ndarray,
                       bound_param: float,
                       random_key: np.ndarray) -> Tuple[np.ndarray, bool, float, np.ndarray]:
    return cond(reject_bool,
                lambda _: rejection_proposal_true(ssm_scenario, t_marginal_particles, t,
                                                  tplus1_particle, tplus1,
                                                  t_marg_log_weights, bound_param, random_key),
                lambda _: (current_t_particle, False, bound_param, random_key),
                None)

# def rejection_proposal(ssm_scenario: StateSpaceModel,
#                        current_t_particle: np.ndarray,
#                        t_marginal_particles: np.ndarray,
#                        t: float,
#                        tplus1_particle: np.ndarray,
#                        tplus1: float,
#                        reject_bool: bool,
#                        t_marg_log_weights: np.ndarray,
#                        bound_param: float,
#                        random_key: np.ndarray) -> Tuple[np.ndarray, bool, float, np.ndarray]:
#     return cond(False,
#                 lambda _: (current_t_particle, False, bound_param, random_key),
#                 lambda _: while_loop(lambda x: True, lambda _: _, (current_t_particle, False, bound_param, random_key)),
#                 None)


@partial(jit, static_argnums=(0,))
def rejection_proposal_map(ssm_scenario: StateSpaceModel,
                           current_t_particles: np.ndarray,
                           marginal_particles: np.ndarray,
                           t: float,
                           tplus1_particles: np.ndarray,
                           tplus1: float,
                           reject_arr: np.ndarray,
                           t_marg_log_weights: np.ndarray,
                           bound_param: float,
                           random_keys: np.ndarray) -> Tuple[np.ndarray, bool, float, np.ndarray]:
    return map(lambda i: rejection_proposal(ssm_scenario,
                                            current_t_particles[i],
                                            marginal_particles,
                                            t,
                                            tplus1_particles[i],
                                            tplus1,
                                            reject_arr[i],
                                            t_marg_log_weights,
                                            bound_param,
                                            random_keys[i]),
               np.arange(current_t_particles.shape[0]))


def full_sample_true(ssm_scenario: StateSpaceModel,
                     t_marginal_particles: np.ndarray,
                     t: float,
                     tplus1_particle: np.ndarray,
                     tplus1: float,
                     t_marg_log_weights: np.ndarray,
                     random_key: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    random_key, cat_key = random.split(random_key)
    adjusted_log_weights = t_marg_log_weights \
                           - vmap(ssm_scenario.transition_potential, (0, None, None, None))(t_marginal_particles, t,
                                                                                            tplus1_particle, tplus1)
    resampled_ind = random.categorical(cat_key, adjusted_log_weights)
    return t_marginal_particles[resampled_ind], random_key


def full_sample(ssm_scenario: StateSpaceModel,
                current_t_particle: np.ndarray,
                t_marginal_particles: np.ndarray,
                t: float,
                tplus1_particle: np.ndarray,
                tplus1: float,
                reject_bool: bool,
                t_marg_log_weights: np.ndarray,
                random_key: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return cond(reject_bool,
                lambda _: full_sample_true(ssm_scenario, t_marginal_particles, t,
                                           tplus1_particle, tplus1,
                                           t_marg_log_weights, random_key),
                lambda _: (current_t_particle, random_key),
                None)


@partial(jit, static_argnums=(0,))
def full_sample_map(ssm_scenario: StateSpaceModel,
                    current_t_particles: np.ndarray,
                    marginal_particles: np.ndarray,
                    t: float,
                    tplus1_particles: np.ndarray,
                    tplus1: float,
                    reject_arr: np.ndarray,
                    t_marg_log_weights: np.ndarray,
                    random_keys: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return map(lambda i: full_sample(ssm_scenario,
                                     current_t_particles[i],
                                     marginal_particles,
                                     t,
                                     tplus1_particles[i],
                                     tplus1,
                                     reject_arr[i],
                                     t_marg_log_weights,
                                     random_keys[i]),
               np.arange(current_t_particles.shape[0]))


def backward_simulation(ssm_scenario: StateSpaceModel,
                        marginal_particles: CDict,
                        maximum_rejections: int,
                        transition_dens_bound_parameter: float,
                        random_key: np.ndarray) -> CDict:
    marg_particles_vals = marginal_particles.value
    times = marginal_particles.t
    marginal_log_weights = marginal_particles.log_weights

    T, n, d = marg_particles_vals.shape

    t_keys = random.split(random_key, T)
    final_particle_vals = marg_particles_vals[-1, random.categorical(t_keys[-1], marginal_log_weights[-1], shape=(n,))]

    def back_sim_body(particles_tplus1: np.ndarray,
                      time_ind: int):
        def backward_sim_propose_and_ar(carry: Tuple[np.ndarray, np.ndarray, float, int, np.ndarray]) \
                -> Tuple[np.ndarray, np.ndarray, float, int, np.ndarray]:
            particles_t, reject_arr, bound_t, r_attemped, int_keys = carry

            particles_t, reject_arr, transition_densities, int_keys \
                = rejection_proposal_map(ssm_scenario,
                                         particles_t,
                                         marg_particles_vals[time_ind],
                                         times[time_ind],
                                         particles_tplus1,
                                         times[time_ind + 1],
                                         reject_arr,
                                         marginal_log_weights[time_ind],
                                         bound_t,
                                         int_keys)

            max_td = np.max(transition_densities)
            reset_bound = max_td > bound_t
            reject_arr = np.where(reset_bound, np.ones(n, dtype='bool'), reject_arr)
            bound_t = np.where(reset_bound, max_td * bound_inflation, bound_t)

            return particles_t, reject_arr, bound_t, r_attemped + 1, int_keys

        int_init_resample_key, uniforms_key, int_key = random.split(t_keys[time_ind], 3)

        init_resample_inds = random.categorical(int_init_resample_key,
                                                marginal_log_weights[time_ind],
                                                shape=(n,))

        init_particles = marg_particles_vals[time_ind, init_resample_inds]
        init_transition_potentials = vmap(ssm_scenario.transition_potential, (0, None, 0, None))(
            init_particles, times[time_ind],
            particles_tplus1, times[time_ind + 1])

        init_transition_densities = np.exp(-init_transition_potentials)

        init_dens_bound_t = np.maximum(transition_dens_bound_parameter,
                                       np.max(init_transition_densities) * bound_inflation)

        uniforms = random.uniform(uniforms_key, shape=(n,))
        init_rejections = uniforms > init_transition_densities / init_dens_bound_t

        init_rej_state = (init_particles,
                          init_rejections,
                          init_dens_bound_t,
                          1,
                          random.split(int_key, n))

        out_rej_state = while_loop(lambda tup: np.logical_and(tup[1].sum() > 0, tup[3] < maximum_rejections),
                                   backward_sim_propose_and_ar,
                                   init_rej_state)

        particles_t, reject_arr, bound_t, r_final, int_keys = out_rej_state

        particles_t, int_keys = full_sample_map(ssm_scenario,
                                                particles_t,
                                                marg_particles_vals[time_ind],
                                                times[time_ind],
                                                particles_tplus1,
                                                times[time_ind + 1],
                                                reject_arr,
                                                marginal_log_weights[time_ind],
                                                int_keys)

        return particles_t, particles_t

    zero_particles, back_sim_particles = scan(back_sim_body,
                                              final_particle_vals,
                                              np.arange(T - 2, -1, -1))

    out_samps = marginal_particles.copy()
    out_samps.value = np.vstack([back_sim_particles[::-1], final_particle_vals[np.newaxis]])
    del out_samps.log_weights
    return out_samps

#
# def backward_simulation(ssm_scenario: StateSpaceModel,
#                         marginal_particles: CDict,
#                         maximum_rejections: int,
#                         transition_dens_bound_parameter: float,
#                         random_key: np.ndarray) -> CDict:
#     marg_particles_vals = marginal_particles.value
#     times = marginal_particles.t
#     marginal_log_weights = marginal_particles.log_weights
#
#     T, n, d = marg_particles_vals.shape
#
#     t_keys = random.split(random_key, T)
#     final_particle_vals = marg_particles_vals[-1, random.categorical(t_keys[-1], marginal_log_weights[-1], shape=(n,))]
#
#     def back_sim_body(particles_tplus1: np.ndarray,
#                       time_ind: int):
#         def backward_sim_propose_and_ar(carry: Tuple[np.ndarray, np.ndarray, float, int, np.ndarray]) \
#                 -> Tuple[np.ndarray, np.ndarray, float, int, np.ndarray]:
#             particles_t, reject_arr, bound_t, r_attemped, int_keys = carry
#
#             new_particle_inds \
#                 = map(lambda i: cond(reject_arr[i],
#                                      lambda _: random.categorical(int_keys[i], marginal_log_weights[time_ind]),
#                                      lambda _: i,
#                                      None), np.arange(n))
#
#             new_transitions_pots \
#                 = map(lambda i: cond(reject_arr[i],
#                                      lambda _: ssm_scenario.transition_potential(
#                                          marg_particles_vals[time_ind, new_particle_inds[i]],
#                                      times[time_ind], particles_tplus1[i], times[time_ind + 1]),
#                                      lambda _: 0.,
#                                      None), np.arange(n))
#
#             transition_densities = np.where(reject_arr, np.exp(-new_transitions_pots), 0.)
#
#             max_td = np.max(transition_densities)
#
#             reject_arr \
#                 = map(lambda i: cond(reject_arr[i],
#                                      lambda _: random.uniform(int_keys[i]) > transition_densities[i] / bound_t,
#                                      lambda _: reject_arr[i],
#                                      None), np.arange(n))
#
#             reset_bound = max_td > bound_t
#             reject_arr = np.where(reset_bound, np.ones(n, dtype='bool'), reject_arr)
#             bound_t = np.where(reset_bound, max_td * bound_inflation, bound_t)
#
#             return particles_t[new_particle_inds], reject_arr, bound_t, r_attemped + 1, random.split(int_keys[0], n)
#
#         int_init_resample_key, uniforms_key, int_key = random.split(t_keys[time_ind], 3)
#
#         init_resample_inds = random.categorical(int_init_resample_key,
#                                                 marginal_log_weights[time_ind],
#                                                 shape=(n,))
#
#         init_particles = marg_particles_vals[time_ind, init_resample_inds]
#         init_transition_potentials = vmap(ssm_scenario.transition_potential, (0, None, 0, None))(
#             init_particles, times[time_ind],
#             particles_tplus1, times[time_ind + 1])
#
#         init_transition_densities = np.exp(-init_transition_potentials)
#
#         init_dens_bound_t = np.maximum(transition_dens_bound_parameter,
#                                        np.max(init_transition_densities) * bound_inflation)
#
#         uniforms = random.uniform(uniforms_key, shape=(n,))
#         init_rejections = uniforms > init_transition_densities / init_dens_bound_t
#
#         init_rej_state = (init_particles,
#                           init_rejections,
#                           init_dens_bound_t,
#                           1,
#                           random.split(int_key, n))
#
#         out_rej_state = while_loop(lambda tup: np.logical_and(tup[1].sum() > 0, tup[3] < maximum_rejections),
#                                    backward_sim_propose_and_ar,
#                                    init_rej_state)
#
#         particles_t, reject_arr, bound_t, r_final, int_keys = out_rej_state
#
#         particles_t, int_keys = full_sample_map(ssm_scenario,
#                                                 particles_t,
#                                                 marg_particles_vals[time_ind],
#                                                 times[time_ind],
#                                                 particles_tplus1,
#                                                 times[time_ind + 1],
#                                                 reject_arr,
#                                                 marginal_log_weights[time_ind],
#                                                 int_keys)
#
#         return particles_t, particles_t
#
#     zero_particles, back_sim_particles = scan(back_sim_body,
#                                               final_particle_vals,
#                                               np.arange(T - 2, -1, -1))
#
#     out_samps = marginal_particles.copy()
#     out_samps.value = np.vstack([back_sim_particles[::-1], final_particle_vals[np.newaxis]])
#     del out_samps.log_weights
#     return out_samps


def backward_simulation_full(ssm_scenario: StateSpaceModel,
                             marginal_particles: CDict,
                             random_key: np.ndarray) -> CDict:
    marg_particles_vals = marginal_particles.value
    times = marginal_particles.t
    marginal_log_weights = marginal_particles.log_weights

    T, n, d = marg_particles_vals.shape

    t_keys = random.split(random_key, T)
    final_particle_vals = marg_particles_vals[-1, random.categorical(t_keys[-1], marginal_log_weights[-1], shape=(n,))]

    def full_sample_true(t_marginal_particles: np.ndarray,
                         t: float,
                         tplus1_particle: np.ndarray,
                         tplus1: float,
                         t_marg_log_weights: np.ndarray,
                         random_key: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        random_key, cat_key = random.split(random_key)
        adjusted_log_weights = t_marg_log_weights \
                               - vmap(ssm_scenario.transition_potential, (0, None, None, None))(t_marginal_particles, t,
                                                                                                tplus1_particle, tplus1)
        resampled_ind = random.categorical(cat_key, adjusted_log_weights)
        return t_marginal_particles[resampled_ind], random_key

    def back_sim_body(particles_tplus1: np.ndarray,
                      time_ind: int):
        int_keys = random.split(t_keys[time_ind], n)
        particles_t, int_keys = vmap(full_sample_true,
                                     (None, None, 0, None, None, 0))(marg_particles_vals[time_ind],
                                                                     times[time_ind],
                                                                     particles_tplus1,
                                                                     times[time_ind + 1],
                                                                     marginal_log_weights[time_ind],
                                                                     int_keys)
        return particles_t, particles_t

    zero_particles, back_sim_particles = scan(back_sim_body,
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
                                          ess_threshold: float = 0.5,
                                          maximum_rejections: int = 5,
                                          transition_dens_bound_parameter: float = 0.) -> CDict:
    pf_key, bsi_key = random.split(random_key)

    pf_start = time()
    pf_samps = run_particle_filter_for_marginals(ssm_scenario,
                                                 particle_filter,
                                                 y,
                                                 t,
                                                 pf_key,
                                                 n=n_samps,
                                                 ess_threshold=ess_threshold)
    pf_samps.value.block_until_ready()
    pf_end = time()
    if maximum_rejections > 0:
        bsi_samps = backward_simulation(ssm_scenario,
                                        pf_samps,
                                        maximum_rejections,
                                        transition_dens_bound_parameter,
                                        bsi_key)
    else:
        bsi_samps = backward_simulation_full(ssm_scenario,
                                             pf_samps,
                                             bsi_key)
    bsi_samps.value.block_until_ready()
    bsi_end = time()

    bsi_samps.time = bsi_end - pf_start
    bsi_samps.pf_time = pf_end - pf_start
    bsi_samps.bsi_time = bsi_end - pf_end

    return bsi_samps
