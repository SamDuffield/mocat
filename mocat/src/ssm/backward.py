########################################################################################################################
# Module: backward/filters.py
# Description: Backward simulations for state-space models.
#
# Web: https://github.com/SamDuffield/mocat
########################################################################################################################

from typing import Tuple

from jax import numpy as np, random, vmap
from jax.lax import while_loop, scan
from jax.ops import index_update

from mocat.src.core import CDict
from mocat.src.ssm.ssm import StateSpaceModel


def full_single_backward_sim(ssm_scenario: StateSpaceModel,
                             particles_t: np.ndarray,
                             t: float,
                             particle_tplus1: np.ndarray,
                             tplus1: float,
                             t_log_weights: np.ndarray,
                             random_key) -> np.ndarray:
    transition_potentials = vmap(ssm_scenario.transition_potential, (0, None, None, None))(particles_t,
                                                                                           t,
                                                                                           particle_tplus1,
                                                                                           tplus1)
    log_weights = t_log_weights - transition_potentials
    return particles_t[random.categorical(random_key, log_weights, 1)]


def backward_simulation(ssm_scenario: StateSpaceModel,
                        marginal_particles: CDict,
                        maximum_rejections: int,
                        transition_dens_bound_parameter: float,
                        random_key: np.ndarray) -> CDict:
    particles_vals = marginal_particles.value
    times = marginal_particles.t
    marginal_log_weights = marginal_particles.log_weights

    T, n, d = particles_vals.shape

    t_keys = random.split(random_key, T)
    final_particle_vals = particles_vals[-1, random.categorical(t_keys[-1], marginal_log_weights[-1], shape=(n,))]

    vectorised_transition_pot = vmap(ssm_scenario.transition_potential, (0, None, 0, None))

    def back_sim_body(joint_particles_tplus1: np.ndarray,
                      time_ind: int):
        def backward_sim_propose_and_ar(carry: Tuple[np.ndarray, np.ndarray, float, int, np.ndarray]) \
                -> Tuple[np.ndarray, np.ndarray, float, int, np.ndarray]:
            particles_t, reject_arr, bound_t, r_attemped, int_key = carry

            n_left = reject_arr.sum()
            reject_particles_tplus1 = joint_particles_tplus1[reject_arr]

            int_key, int_cat_key, int_uniform_key = random.split(int_key, 3)

            resample_inds = random.categorical(int_cat_key, marginal_log_weights[time_ind], shape=(n_left,))
            proposed_part_vals = particles_vals[time_ind, resample_inds]
            transition_pots = vectorised_transition_pot(
                proposed_part_vals, times[time_ind],
                reject_particles_tplus1, times[time_ind + 1])
            transition_densities = np.exp(-transition_pots)

            int_uniforms = random.uniform(int_uniform_key, shape=(n_left,))
            reject_arr = index_update(reject_arr, reject_arr, int_uniforms < transition_densities / bound_t)
            particles_t = index_update(particles_t, reject_arr, particles_t[time_ind, resample_inds])

            max_td = np.max(transition_densities)
            reset_bound = max_td > bound_t
            reject_arr = np.where(reset_bound, np.zeros(n, dtype='bool'), reject_arr)
            bound_t = np.where(reset_bound, max_td, bound_t)

            return particles_t, reject_arr, bound_t, r_attemped + 1, int_key

        int_init_resample_key, uniforms_key, int_key = random.split(t_keys[time_ind], 3)

        init_resample_inds = random.categorical(int_init_resample_key,
                                                marginal_log_weights[time_ind],
                                                shape=(n,))

        init_particles = particles_vals[time_ind, init_resample_inds]
        init_transition_potentials = vectorised_transition_pot(
            init_particles, times[time_ind],
            joint_particles_tplus1, times[time_ind + 1])

        init_transition_densities = np.exp(-init_transition_potentials)

        init_dens_bound_t = np.maximum(transition_dens_bound_parameter, np.max(init_transition_densities))

        uniforms = random.uniform(uniforms_key, shape=(n,))
        init_rejections = uniforms > init_transition_densities / init_dens_bound_t

        init_rej_state = (init_particles,
                          init_rejections,
                          init_dens_bound_t,
                          1,
                          int_key)

        out_rej_state = while_loop(lambda tup: np.logical_and(tup[1].sum() < n, tup[2] < maximum_rejections),
                                   backward_sim_propose_and_ar,
                                   init_rej_state)

        particles_t, reject_arr, bound_t, r_final, int_key = out_rej_state

        n_full_samp = reject_arr.sum()
        full_samp_keys = random.split(int_key, n_full_samp)

        remaining_parts = vmap(full_single_backward_sim,
                               (None, None, None, 0, None, None, 0))(ssm_scenario,
                                                                     particles_vals[time_ind],
                                                                     times[time_ind],
                                                                     joint_particles_tplus1[reject_arr],
                                                                     times[time_ind + 1],
                                                                     marginal_log_weights[time_ind],
                                                                     full_samp_keys)
        particles_t = index_update(particles_t, reject_arr, remaining_parts)
        return particles_t

    zero_particles, back_sim_particles = scan(back_sim_body,
                                              final_particle_vals,
                                              np.arange(T, -1, -1))

    out_samps = marginal_particles.copy()
    out_samps.value = back_sim_particles
    del out_samps.log_weights
    return out_samps
