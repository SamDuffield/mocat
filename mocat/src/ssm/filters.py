########################################################################################################################
# Module: ssm/filters.py
# Description: Particle filtering inc bootstrap filter.
#
# Web: https://github.com/SamDuffield/mocat
########################################################################################################################

from functools import partial
from typing import Union, Tuple

import jax.numpy as jnp
from jax import random, vmap, jit
from jax.lax import cond, scan
from jax.ops import index_update

from mocat.src.core import cdict
from mocat.src.ssm.ssm import StateSpaceModel
from mocat.src.metrics import ess_log_weight


class ParticleFilter:
    name = 'Particle Filter'

    def __init__(self,
                 name: str = None):
        if name is not None:
            self.name = name

    def __repr__(self):
        return f"mocat.ParticleFilter.{self.__class__.__name__}"

    def startup(self,
                ssm_scenario: StateSpaceModel):
        pass

    def initial_potential(self,
                          ssm_scenario: StateSpaceModel,
                          x: jnp.ndarray,
                          y: jnp.ndarray,
                          t: float) -> Union[float, jnp.ndarray]:
        return ssm_scenario.initial_potential(x, t)

    def initial_sample(self,
                       ssm_scenario: StateSpaceModel,
                       y: jnp.ndarray,
                       t: float,
                       random_key: jnp.ndarray) -> jnp.ndarray:
        return ssm_scenario.initial_sample(t, random_key)

    def initial_log_weight(self,
                           ssm_scenario: StateSpaceModel,
                           x: jnp.ndarray,
                           y: jnp.ndarray,
                           t: float) -> Union[float, jnp.ndarray]:
        return -ssm_scenario.likelihood_potential(x, y, t)

    @partial(jit, static_argnums=(0, 1))
    def initial_sample_vectorised(self,
                                  ssm_scenario: StateSpaceModel,
                                  y: jnp.ndarray,
                                  t: float,
                                  random_keys: jnp.ndarray) -> jnp.ndarray:
        init_vals = vmap(self.initial_sample, (None, None, None, 0))(ssm_scenario, y, t, random_keys)
        return init_vals

    @partial(jit, static_argnums=(0, 1))
    def initial_sample_and_weight_vectorised(self,
                                             ssm_scenario: StateSpaceModel,
                                             y: jnp.ndarray,
                                             t: float,
                                             random_keys: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        init_vals = vmap(self.initial_sample, (None, None, None, 0))(ssm_scenario, y, t, random_keys)

        if y is not None:
            init_log_weight = vmap(self.initial_log_weight, (None, 0, None, None))(ssm_scenario, init_vals, y, t)
        else:
            init_log_weight = jnp.zeros(len(init_vals))

        return init_vals, init_log_weight

    def proposal_potential(self,
                           ssm_scenario: StateSpaceModel,
                           x_previous: jnp.ndarray,
                           t_previous: float,
                           x_new: jnp.ndarray,
                           y_new: jnp.ndarray,
                           t_new: float) -> Union[float, jnp.ndarray]:
        raise AttributeError(f'{self.__class__.__name__} transition_potential not implemented')

    def proposal_sample(self,
                        ssm_scenario: StateSpaceModel,
                        x_previous: jnp.ndarray,
                        t_previous: float,
                        y_new: jnp.ndarray,
                        t_new: float,
                        random_key: jnp.ndarray) -> jnp.ndarray:
        raise AttributeError(f'{self.__class__.__name__} proposal_sample not implemented')

    def intermediate_log_weight(self,
                                ssm_scenario: StateSpaceModel,
                                x_previous: jnp.ndarray,
                                t_previous: float,
                                x_new: jnp.ndarray,
                                y_new: jnp.ndarray,
                                t_new: float) -> Union[float, jnp.ndarray]:
        return - ssm_scenario.transition_potential(x_previous, t_previous,
                                                   x_new, t_new) \
               - ssm_scenario.likelihood_potential(x_new, y_new, t_new) \
               + self.proposal_potential(ssm_scenario, x_previous, t_previous, x_new, y_new, t_new)

    @partial(jit, static_argnums=(0, 1))
    def proposal_sample_vectorised(self,
                                   ssm_scenario: StateSpaceModel,
                                   x_previous: jnp.ndarray,
                                   t_previous: float,
                                   y_new: jnp.ndarray,
                                   t_new: float,
                                   random_keys: jnp.ndarray) -> jnp.ndarray:
        x_new = vmap(self.proposal_sample, (None, 0, None, None, None, 0))(ssm_scenario,
                                                                           x_previous, t_previous,
                                                                           y_new, t_new,
                                                                           random_keys)
        return x_new

    @partial(jit, static_argnums=(0, 1))
    def propose_and_intermediate_weight_vectorised(self,
                                                   ssm_scenario: StateSpaceModel,
                                                   x_previous: jnp.ndarray,
                                                   t_previous: float,
                                                   y_new: jnp.ndarray,
                                                   t_new: float,
                                                   random_keys: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        x_new = vmap(self.proposal_sample, (None, 0, None, None, None, 0))(ssm_scenario,
                                                                           x_previous, t_previous,
                                                                           y_new, t_new,
                                                                           random_keys)
        log_weight_new = vmap(self.intermediate_log_weight, (None, 0, None, 0, None, None))(ssm_scenario,
                                                                                            x_previous, t_previous,
                                                                                            x_new, y_new, t_new)
        return x_new, log_weight_new


class BootstrapFilter(ParticleFilter):
    name = 'Bootstrap Filter'

    def proposal_potential(self,
                           ssm_scenario: StateSpaceModel,
                           x_previous: jnp.ndarray,
                           t_previous: float,
                           x_new: jnp.ndarray,
                           y_new: jnp.ndarray,
                           t_new: float) -> Union[float, jnp.ndarray]:
        return ssm_scenario.transition_potential(x_previous, t_previous, x_new, t_new)

    def proposal_sample(self,
                        ssm_scenario: StateSpaceModel,
                        x_previous: jnp.ndarray,
                        t_previous: float,
                        y_new: jnp.ndarray,
                        t_new: float,
                        random_key: jnp.ndarray) -> jnp.ndarray:
        return ssm_scenario.transition_sample(x_previous, t_previous, t_new, random_key)

    def intermediate_log_weight(self,
                                ssm_scenario: StateSpaceModel,
                                x_previous: jnp.ndarray,
                                t_previous: float,
                                x_new: jnp.ndarray,
                                y_new: jnp.ndarray,
                                t_new: float) -> Union[float, jnp.ndarray]:
        return -ssm_scenario.likelihood_potential(x_new, y_new, t_new)


def initiate_particles(ssm_scenario: StateSpaceModel,
                       particle_filter: ParticleFilter,
                       n: int,
                       random_key: jnp.ndarray,
                       y: jnp.ndarray = None,
                       t: float = None) -> cdict:
    particle_filter.startup(ssm_scenario)

    sub_keys = random.split(random_key, n)

    init_vals, init_log_weight = particle_filter.initial_sample_and_weight_vectorised(ssm_scenario, y, t, sub_keys)

    if init_vals.ndim == 1:
        init_vals = init_vals[..., jnp.newaxis]

    initial_sample = cdict(value=init_vals[jnp.newaxis],
                           log_weight=init_log_weight[jnp.newaxis],
                           t=jnp.atleast_1d(t) if t is not None else jnp.zeros(1),
                           y=y[jnp.newaxis] if y is not None else None,
                           ess=jnp.atleast_1d(ess_log_weight(init_log_weight)))
    return initial_sample


def _resample(x_w_r: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    x, log_weight, random_key = x_w_r
    n = x.shape[-2]
    random_key, sub_key = random.split(random_key)
    return x[..., random.categorical(sub_key, log_weight, shape=(n,)), :], jnp.zeros(n), random_key


def resample_particles(particles: cdict,
                       random_key: jnp.ndarray,
                       resample_full: bool = True) -> cdict:
    out_particles = particles.copy()
    if out_particles.log_weight.ndim == 1:
        out_particles.value, out_particles.log_weight, _ = _resample((particles.value,
                                                                      particles.log_weight,
                                                                      random_key))
    elif resample_full:
        out_particles.value, latest_log_weight, _ = _resample((particles.value,
                                                               particles.log_weight[-1],
                                                               random_key))
        out_particles.log_weight = index_update(out_particles.log_weight, -1, latest_log_weight)
    else:
        latest_value, latest_log_weight, _ = _resample((particles.value[-1],
                                                        particles.log_weight[-1],
                                                        random_key))
        out_particles.value = index_update(out_particles.value, -1, latest_value)
        out_particles.log_weight = index_update(out_particles.log_weight, -1, latest_log_weight)
    return out_particles


def propagate_particle_filter(ssm_scenario: StateSpaceModel,
                              particle_filter: ParticleFilter,
                              particles: cdict,
                              y_new: jnp.ndarray,
                              t_new: float,
                              random_key: jnp.ndarray,
                              ess_threshold: float = 0.5,
                              resample_full: bool = True) -> cdict:
    n = particles.value.shape[1]
    ess_previous = particles.ess[-1]
    out_particles = cond(ess_previous < ess_threshold * n,
                         lambda tup: resample_particles(*tup),
                         lambda tup: tup[0].copy(),
                         (particles, random_key, resample_full))

    x_previous = out_particles.value[-1]
    log_weight_previous = out_particles.log_weight[-1]
    t_previous = out_particles.t[-1]

    split_keys = random.split(random_key, len(x_previous))

    x_new, log_weight_new = particle_filter.propose_and_intermediate_weight_vectorised(ssm_scenario,
                                                                                       x_previous, t_previous,
                                                                                       y_new, t_new, split_keys)

    log_weight_new = log_weight_previous + log_weight_new

    out_particles.value = jnp.append(out_particles.value, x_new, axis=0)
    out_particles.log_weight = jnp.append(out_particles.log_weight, log_weight_new, axis=0)
    out_particles.y = jnp.append(out_particles.y, y_new)
    out_particles.t = jnp.append(out_particles.t, t_new)
    out_particles.ess = jnp.append(out_particles.ess, ess_log_weight(log_weight_new))
    return out_particles


def run_particle_filter_for_marginals(ssm_scenario: StateSpaceModel,
                                      particle_filter: ParticleFilter,
                                      y: jnp.ndarray,
                                      t: jnp.ndarray,
                                      random_key: jnp.ndarray,
                                      n: int = None,
                                      initial_sample: cdict = None,
                                      ess_threshold: float = 0.5) -> cdict:
    if y.ndim == 1:
        y = y[..., jnp.newaxis]

    if initial_sample is None:
        random_key, sub_key = random.split(random_key)
        init_y = y[0]
        y = y[1:]

        initial_sample = initiate_particles(ssm_scenario, particle_filter, n, sub_key, init_y, t[0])
        t = t[1:]

    if n is None:
        n = initial_sample.value.shape[1]

    num_propagate_steps = len(y)
    int_rand_keys = random.split(random_key, num_propagate_steps)

    def particle_filter_body(samps_previous: cdict,
                             iter_ind: int) -> Tuple[cdict, cdict]:
        x_previous = samps_previous.value
        log_weight_previous = samps_previous.log_weight
        int_rand_key = int_rand_keys[iter_ind]

        ess_previous = samps_previous.ess

        x_res, log_weight_res, int_rand_key = cond(ess_previous < (ess_threshold * n),
                                                   _resample,
                                                   lambda _: (x_previous, log_weight_previous, int_rand_key),
                                                   (x_previous, log_weight_previous, int_rand_key))

        split_keys = random.split(int_rand_key, len(x_previous))

        x_new, log_weight_new = particle_filter.propose_and_intermediate_weight_vectorised(ssm_scenario,
                                                                                           x_res,
                                                                                           samps_previous.t,
                                                                                           y[iter_ind],
                                                                                           t[iter_ind],
                                                                                           split_keys)

        log_weight_new = log_weight_res + log_weight_new

        samps_new = samps_previous.copy()
        samps_new.value = x_new
        samps_new.log_weight = log_weight_new
        samps_new.y = y[iter_ind]
        samps_new.t = t[iter_ind]
        samps_new.ess = ess_log_weight(log_weight_new)
        return samps_new, samps_new

    _, after_init_samps = scan(particle_filter_body,
                               initial_sample[-1],
                               jnp.arange(num_propagate_steps))

    out_samps = initial_sample.copy()
    out_samps.value = jnp.append(initial_sample.value, after_init_samps.value, axis=0)
    out_samps.log_weight = jnp.append(initial_sample.log_weight, after_init_samps.log_weight, axis=0)
    out_samps.y = jnp.append(initial_sample.y, after_init_samps.y, axis=0)
    out_samps.t = jnp.append(initial_sample.t, after_init_samps.t)
    out_samps.ess = jnp.append(initial_sample.ess, after_init_samps.ess)

    return out_samps
