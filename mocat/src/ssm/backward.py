########################################################################################################################
# Module: backward/filters.py
# Description: Backward simulation for state-space models.
#
# Web: https://github.com/SamDuffield/mocat
########################################################################################################################

from typing import Tuple

from jax import numpy as np, random, vmap, jit
from jax.lax import while_loop, scan, cond

from mocat.src.core import CDict
from mocat.src.ssm.ssm import StateSpaceModel


bound_inflation = 1.01


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

    def rejection_proposal_true(t_marginal_particles: np.ndarray,
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
               random.uniform(uniform_key) < transition_dens / bound_param, \
               transition_dens, \
               random_key

    def rejection_proposal(current_t_particle: np.ndarray,
                           t_marginal_particles: np.ndarray,
                           t: float,
                           tplus1_particle: np.ndarray,
                           tplus1: float,
                           reject_bool: bool,
                           t_marg_log_weights: np.ndarray,
                           bound_param: float,
                           random_key: np.ndarray) -> Tuple[np.ndarray, bool, float, np.ndarray]:
        return cond(reject_bool,
                    lambda tup: rejection_proposal_true(*tup),
                    lambda _: (current_t_particle, False, bound_param, random_key),
                    (t_marginal_particles, t,
                     tplus1_particle, tplus1,
                     t_marg_log_weights, bound_param, random_key))

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

    def full_sample(current_t_particle: np.ndarray,
                    t_marginal_particles: np.ndarray,
                    t: float,
                    tplus1_particle: np.ndarray,
                    tplus1: float,
                    reject_bool: bool,
                    t_marg_log_weights: np.ndarray,
                    random_key: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return cond(reject_bool,
                    lambda tup: full_sample_true(*tup),
                    lambda _: (current_t_particle, random_key),
                    (t_marginal_particles, t,
                     tplus1_particle, tplus1,
                     t_marg_log_weights, random_key))

    vectorised_rejection_proposal = jit(vmap(rejection_proposal, (0, None, None, 0, None, 0, None, None, 0)))

    def back_sim_body(particles_tplus1: np.ndarray,
                      time_ind: int):
        def backward_sim_propose_and_ar(carry: Tuple[np.ndarray, np.ndarray, float, int, np.ndarray]) \
                -> Tuple[np.ndarray, np.ndarray, float, int, np.ndarray]:
            particles_t, reject_arr, bound_t, r_attemped, int_keys = carry

            particles_t, reject_arr, transition_densities, int_keys \
                = vectorised_rejection_proposal(particles_t,
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

        particles_t, int_keys = vmap(full_sample, (0, None, None, 0, None, 0, None, 0))(particles_t,
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
