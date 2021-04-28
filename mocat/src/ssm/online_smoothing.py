########################################################################################################################
# Module: ssm/online_smoothing.py
# Description: Online (fixed-lag) particle smoothing for state-space models.
#
# Web: https://github.com/SamDuffield/mocat
########################################################################################################################

from typing import Tuple
from time import time
from functools import partial

from jax import numpy as jnp, random, vmap, jit
from jax.lax import while_loop, scan, cond, map
from jax.ops import index_update

from mocat.src.core import cdict
from mocat.src.ssm.ssm import StateSpaceModel
from mocat.src.ssm.filtering import ParticleFilter, resample_particles, propagate_particle_filter
from mocat.src.ssm.backward import backward_simulation
from mocat.src.metrics import ess_log_weight


def full_stitch_single(ssm_scenario: StateSpaceModel,
                       x0_single: jnp.ndarray,
                       t: float,
                       x1_all: jnp.ndarray,
                       tplus1: float,
                       x1_log_weight: jnp.ndarray,
                       random_key: jnp.ndarray) -> jnp.ndarray:
    log_weight = x1_log_weight - vmap(ssm_scenario.transition_potential, (None, None, 0, None))(x0_single, t,
                                                                                                x1_all, tplus1)
    return random.categorical(random_key, log_weight)


def full_stitch(ssm_scenario: StateSpaceModel,
                x0_all: jnp.ndarray,
                t: float,
                x1_all: jnp.ndarray,
                tplus1: float,
                x1_log_weight: jnp.ndarray,
                random_key: jnp.ndarray) -> jnp.ndarray:
    return vmap(full_stitch_single, (None, 0, None, None, None, None, 0)) \
        (ssm_scenario, x0_all, t, x1_all, tplus1, x1_log_weight, random.split(random_key, len(x0_all)))


def full_stitch_single_cond(not_yet_accepted: bool,
                            x1_ind_false: jnp.ndarray,
                            ssm_scenario: StateSpaceModel,
                            x0_single: jnp.ndarray,
                            t: float,
                            x1_all: jnp.ndarray,
                            tplus1: float,
                            x1_log_weight: jnp.ndarray,
                            random_key: jnp.ndarray) -> jnp.ndarray:
    return cond(not_yet_accepted,
                lambda _: full_stitch_single(ssm_scenario, x0_single, t, x1_all, tplus1, x1_log_weight, random_key),
                lambda _: x1_ind_false,
                None)


def rejection_stitch_proposal_single(ssm_scenario: StateSpaceModel,
                                     x0_single: jnp.ndarray,
                                     t: float,
                                     x1_all: jnp.ndarray,
                                     tplus1: float,
                                     x1_log_weight: jnp.ndarray,
                                     bound: float,
                                     random_key: jnp.ndarray) \
        -> Tuple[jnp.ndarray, float, bool, jnp.ndarray]:
    random_key, choice_key, uniform_key = random.split(random_key, 3)
    x1_single_ind = random.categorical(choice_key, x1_log_weight)
    conditional_dens = jnp.exp(-ssm_scenario.transition_potential(x0_single, t, x1_all[x1_single_ind], tplus1))
    return x1_single_ind, conditional_dens, random.uniform(uniform_key) > conditional_dens / bound, random_key


def rejection_stitch_proposal_single_cond(not_yet_accepted: bool,
                                          x1_ind_false: jnp.ndarray,
                                          ssm_scenario: StateSpaceModel,
                                          x0_single: jnp.ndarray,
                                          t: float,
                                          x1_all: jnp.ndarray,
                                          tplus1: float,
                                          x1_log_weight: jnp.ndarray,
                                          bound: float,
                                          random_key: jnp.ndarray) -> Tuple[jnp.ndarray, float, bool, jnp.ndarray]:
    return cond(not_yet_accepted,
                lambda _: rejection_stitch_proposal_single(ssm_scenario,
                                                           x0_single, t,
                                                           x1_all, tplus1,
                                                           x1_log_weight, bound, random_key),
                lambda _: (x1_ind_false, 0., False, random_key),
                None)


def rejection_stitch_proposal_all(ssm_scenario: StateSpaceModel,
                                  x0_all: jnp.ndarray,
                                  t: float,
                                  x1_all: jnp.ndarray,
                                  tplus1: float,
                                  x1_log_weight: jnp.ndarray,
                                  bound_inflation: float,
                                  not_yet_accepted_arr: jnp.ndarray,
                                  x1_all_sampled_inds: jnp.ndarray,
                                  bound: float,
                                  random_keys: jnp.ndarray,
                                  rejection_iter: int,
                                  num_transition_evals: int) \
        -> Tuple[jnp.ndarray, jnp.ndarray, float, jnp.ndarray, int, int]:
    n = len(x1_all)
    mapped_tup = map(lambda i: rejection_stitch_proposal_single_cond(not_yet_accepted_arr[i],
                                                                     x1_all_sampled_inds[i],
                                                                     ssm_scenario,
                                                                     x0_all[i],
                                                                     t,
                                                                     x1_all,
                                                                     tplus1,
                                                                     x1_log_weight,
                                                                     bound,
                                                                     random_keys[i]), jnp.arange(n))
    x1_all_sampled_inds, dens_evals, not_yet_accepted_arr_new, random_keys = mapped_tup

    # Check if we need to start again
    max_dens = jnp.max(dens_evals)
    reset_bound = max_dens > bound
    bound = jnp.where(reset_bound, max_dens * bound_inflation, bound)
    not_yet_accepted_arr_new = jnp.where(reset_bound, jnp.ones(n, dtype='bool'), not_yet_accepted_arr_new)
    return not_yet_accepted_arr_new, x1_all_sampled_inds, bound, random_keys, rejection_iter + 1, \
           num_transition_evals + not_yet_accepted_arr.sum()


def rejection_stitching(ssm_scenario: StateSpaceModel,
                        x0_all: jnp.ndarray,
                        t: float,
                        x1_all: jnp.ndarray,
                        tplus1: float,
                        x1_log_weight: jnp.ndarray,
                        random_key: jnp.ndarray,
                        maximum_rejections: int,
                        init_bound_param: float,
                        bound_inflation: float) -> Tuple[jnp.ndarray, int]:
    rejection_initial_keys = random.split(random_key, 3)
    n = len(x1_all)

    # Prerun to initiate bound
    x1_initial_inds = random.categorical(rejection_initial_keys[0], x1_log_weight, shape=(n,))
    initial_cond_dens = jnp.exp(-vmap(ssm_scenario.transition_potential,
                                      (0, None, 0, None))(x0_all, t, x1_all[x1_initial_inds], tplus1))
    max_cond_dens = jnp.max(initial_cond_dens)
    initial_bound = jnp.where(max_cond_dens > init_bound_param, max_cond_dens * bound_inflation, init_bound_param)
    initial_not_yet_accepted_arr = random.uniform(rejection_initial_keys[1], (n,)) > initial_cond_dens / initial_bound

    out_tup = while_loop(lambda tup: jnp.logical_and(tup[0].sum() > 0, tup[-1] < maximum_rejections),
                         lambda tup: rejection_stitch_proposal_all(ssm_scenario, x0_all, t, x1_all, tplus1,
                                                                   x1_log_weight,
                                                                   bound_inflation, *tup),
                         (initial_not_yet_accepted_arr,
                          x1_initial_inds,
                          initial_bound,
                          random.split(rejection_initial_keys[2], n),
                          1,
                          n))
    not_yet_accepted_arr, x1_final_inds, final_bound, random_keys, rej_attempted, num_transition_evals = out_tup

    x1_final_inds = map(lambda i: full_stitch_single_cond(not_yet_accepted_arr[i],
                                                          x1_final_inds[i],
                                                          ssm_scenario,
                                                          x0_all[i],
                                                          t,
                                                          x1_all,
                                                          tplus1,
                                                          x1_log_weight,
                                                          random_keys[i]), jnp.arange(n))

    num_transition_evals = num_transition_evals + len(x1_all) * not_yet_accepted_arr.sum()

    return x1_final_inds, num_transition_evals


def fixed_lag_stitching(ssm_scenario: StateSpaceModel,
                        early_block: jnp.ndarray,
                        t: float,
                        recent_block: jnp.ndarray,
                        recent_block_log_weight: jnp.ndarray,
                        tplus1: float,
                        random_key: jnp.ndarray,
                        maximum_rejections: int,
                        init_bound_param: float,
                        bound_inflation: float) -> jnp.ndarray:
    x0_fixed_all = early_block[-1]

    x0_vary_all = recent_block[0]
    x1_vary_all = recent_block[1]

    non_interacting_log_weight = recent_block_log_weight \
                                 + vmap(ssm_scenario.transition_potential, (0, None, 0, None))(x0_vary_all, t,
                                                                                               x1_vary_all, tplus1)

    recent_stitched_inds, num_transition_evals = cond(maximum_rejections > 0,
                                                      lambda tup: rejection_stitching(ssm_scenario, *tup,
                                                                                      maximum_rejections=maximum_rejections,
                                                                                      init_bound_param=init_bound_param,
                                                                                      bound_inflation=bound_inflation),
                                                      lambda tup: (
                                                      full_stitch(ssm_scenario, *tup), len(x0_fixed_all) ** 2),
                                                      (x0_fixed_all, t, x1_vary_all, tplus1,
                                                       non_interacting_log_weight, random_key))

    return jnp.append(early_block, recent_block[1:, recent_stitched_inds], axis=0)


@partial(jit, static_argnums=(0, 1, 6, 7))
def propagate_particle_smoother_pf(ssm_scenario: StateSpaceModel,
                                   particle_filter: ParticleFilter,
                                   particles: cdict,
                                   y_new: jnp.ndarray,
                                   t_new: float,
                                   random_key: jnp.ndarray,
                                   lag: int,
                                   maximum_rejections: int,
                                   init_bound_param: float,
                                   bound_inflation: float) -> cdict:
    n = particles.value.shape[1]

    # Check particles are unweighted
    out_particles = cond(ess_log_weight(particles.log_weight[-1]) < (n - 1e-3),
                         lambda p: resample_particles(p, random_key, True),
                         lambda p: p.copy(),
                         particles)
    out_particles.log_weight = jnp.zeros(n)

    x_previous = out_particles.value[-1]
    t_previous = out_particles.t[-1]

    split_keys = random.split(random_key, len(x_previous))

    x_new, out_particles.log_weight = particle_filter.propose_and_intermediate_weight_vectorised(ssm_scenario,
                                                                                                 x_previous, t_previous,
                                                                                                 y_new, t_new,
                                                                                                 split_keys)

    out_particles.value = jnp.append(out_particles.value, x_new[jnp.newaxis], axis=0)
    out_particles.y = jnp.append(out_particles.y, y_new[jnp.newaxis])
    out_particles.t = jnp.append(out_particles.t, t_new)
    out_particles.ess = ess_log_weight(out_particles.log_weight)

    len_t = len(out_particles.t)
    stitch_ind_min_1 = len_t - lag - 1
    stitch_ind = len_t - lag

    # out_particles.value = cond(stitch_ind_min_1 >= 0,
    #                            lambda vals: fixed_lag_stitching(ssm_scenario,
    #                                                             vals[:(stitch_ind_min_1 + 1)],
    #                                                             out_particles.t[stitch_ind_min_1],
    #                                                             vals[stitch_ind_min_1:],
    #                                                             out_particles.log_weight,
    #                                                             out_particles.t[stitch_ind],
    #                                                             random_key,
    #                                                             maximum_rejections,
    #                                                             init_bound_param,
    #                                                             bound_inflation),
    #                            lambda vals: vals,
    #                            out_particles.value)

    if stitch_ind_min_1 >= 0:
        out_particles.value = fixed_lag_stitching(ssm_scenario,
                                                  out_particles.value[:(stitch_ind_min_1 + 1)],
                                                  out_particles.t[stitch_ind_min_1],
                                                  out_particles.value[stitch_ind_min_1:],
                                                  out_particles.log_weight,
                                                  out_particles.t[stitch_ind],
                                                  random_key,
                                                  maximum_rejections,
                                                  init_bound_param,
                                                  bound_inflation)

    out_particles.log_weight = jnp.where(stitch_ind_min_1 >= 0, jnp.zeros(n), out_particles.log_weight)
    return out_particles


@partial(jit, static_argnums=(0, 1, 6, 8))
def propagate_particle_smoother_bs(ssm_scenario: StateSpaceModel,
                                   particle_filter: ParticleFilter,
                                   particles: cdict,
                                   y_new: jnp.ndarray,
                                   t_new: float,
                                   random_key: jnp.ndarray,
                                   lag: int,
                                   ess_threshold: float,
                                   maximum_rejections: int,
                                   init_bound_param: float,
                                   bound_inflation: float) -> cdict:
    n = particles.value.shape[1]

    if not hasattr(particles, 'marginal_filter'):
        particles.marginal_filter = cdict(value=particles.value,
                                          log_weight=particles.log_weight,
                                          y=particles.y,
                                          t=particles.t,
                                          ess=particles.ess)

    split_keys = random.split(random_key, 4)

    out_particles = particles

    # Propagate marginal filter particles
    out_particles.marginal_filter = propagate_particle_filter(ssm_scenario, particle_filter, particles.marginal_filter,
                                                              y_new, t_new, split_keys[1], ess_threshold, False)
    out_particles.y = jnp.append(out_particles.y, y_new)
    out_particles.t = jnp.append(out_particles.t, t_new)
    out_particles.log_weight = jnp.zeros(n)
    out_particles.ess = out_particles.marginal_filter.ess[-1]

    len_t = len(out_particles.t)
    stitch_ind_min_1 = len_t - lag - 1
    stitch_ind = len_t - lag

    def back_sim_only(marginal_filter):
        return backward_simulation(ssm_scenario,
                                   marginal_filter,
                                   split_keys[2],
                                   n,
                                   maximum_rejections,
                                   init_bound_param,
                                   bound_inflation).value

    def back_sim_and_stitch(marginal_filter):
        backward_sim = backward_simulation(ssm_scenario,
                                           marginal_filter[stitch_ind_min_1:],
                                           split_keys[2],
                                           n,
                                           maximum_rejections,
                                           init_bound_param,
                                           bound_inflation)

        return fixed_lag_stitching(ssm_scenario,
                                   out_particles.value[:(stitch_ind_min_1 + 1)],
                                   out_particles.t[stitch_ind_min_1],
                                   backward_sim.value,
                                   jnp.zeros(n),
                                   out_particles.t[stitch_ind],
                                   random_key,
                                   maximum_rejections,
                                   init_bound_param,
                                   bound_inflation)

    if stitch_ind_min_1 >= 0:
        out_particles.value = back_sim_and_stitch(out_particles.marginal_filter)
    else:
        out_particles.value = back_sim_only(out_particles.marginal_filter)

    # out_particles.value = cond(stitch_ind_min_1 >= 0,
    #                            back_sim_and_stitch,
    #                            back_sim_only,
    #                            out_particles.marginal_filter)
    return out_particles


def propagate_particle_smoother(ssm_scenario: StateSpaceModel,
                                particle_filter: ParticleFilter,
                                particles: cdict,
                                y_new: jnp.ndarray,
                                t_new: float,
                                random_key: jnp.ndarray,
                                lag: int,
                                backward_sim: bool = True,
                                ess_threshold: float = 0.5,
                                maximum_rejections: int = 0,
                                init_bound_param: float = 0.,
                                bound_inflation: float = 1.01) -> cdict:
    if backward_sim:
        return propagate_particle_smoother_bs(ssm_scenario, particle_filter, particles, y_new, t_new, random_key, lag,
                                              ess_threshold, maximum_rejections, init_bound_param, bound_inflation)
    else:
        return propagate_particle_smoother_pf(ssm_scenario, particle_filter, particles, y_new, t_new, random_key, lag,
                                              maximum_rejections, init_bound_param, bound_inflation)
