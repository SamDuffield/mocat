########################################################################################################################
# Module: mcmc/qn_underdamped.py
# Description: Underdamped Langevin kernel with L-BFGS inverse Hessian pre-conditioner
#
# Web: https://github.com/SamDuffield/mocat
########################################################################################################################

from functools import partial
from typing import Tuple, Union, Callable

import jax.numpy as jnp
from jax import random, jit, vmap
from jax.lax import scan

from mocat.src.core import cdict, Scenario
from mocat.src.mcmc.sampler import MCMCSampler
from mocat.src.mcmc.metropolis import Metropolis
from mocat.src.utils import bfgs_sqrt_pqut, bfgs_sqrt_prod, bfgs_sqrt_transpose_prod
from mocat.src.transport.smc import RMMetropolisedSMCSampler


# Underdamped Langevin
# L-BFGS approximated inverse Hessian pre-conditioner/mass matrix
class QNUnderdamped(MCMCSampler):
    name = 'Quasi-Newton Underdamped'
    correction = Metropolis

    def __init__(self,
                 stepsize: float = None,
                 leapfrog_steps: int = 1,
                 friction: float = 1.0,
                 m: int = 30,
                 init_hessian_sqrt_diag: float = 1.,
                 pos_def_r: float = 0.5,
                 sort_bfgs: bool = False,
                 max_val: float = 1e10):
        super().__init__()
        self.parameters.stepsize = stepsize
        self.parameters.leapfrog_steps = leapfrog_steps
        self.parameters.friction = friction
        self.parameters.m = m
        self.parameters.init_hessian_sqrt_diag = init_hessian_sqrt_diag
        self.parameters.pos_def_r = pos_def_r
        self.parameters.max_val = max_val
        self.sort_bfgs = sort_bfgs
        self.tuning.target = 0.651

    def startup(self,
                scenario: Scenario,
                n: int,
                initial_state: cdict,
                initial_extra: cdict,
                **kwargs) -> Tuple[cdict, cdict]:
        initial_state, initial_extra = super().startup(scenario, n,
                                                       initial_state, initial_extra, **kwargs)
        initial_extra.random_key, prior_key, likelihood_key = random.split(initial_extra.random_key, 3)
        if not hasattr(initial_state, 'prior_potential') or not hasattr(initial_state, 'grad_prior_potential'):
            initial_state.prior_potential, initial_state.grad_prior_potential \
                = scenario.prior_potential_and_grad(initial_state.value, prior_key)
        if not hasattr(initial_state, 'likelihood_potential') or not hasattr(initial_state,
                                                                             'grad_likelihood_potential'):
            initial_state.likelihood_potential, initial_state.grad_likelihood_potential \
                = scenario.likelihood_potential_and_grad(initial_state.value, likelihood_key)
        initial_state.potential = initial_state.prior_potential \
                                  + scenario.temperature * initial_state.likelihood_potential
        initial_state.grad_potential = initial_state.grad_prior_potential \
                                       + scenario.temperature * initial_state.grad_likelihood_potential

        if not hasattr(initial_state, 'momenta') or initial_state.momenta.shape[-1] != scenario.dim:
            initial_state.momenta = jnp.zeros(scenario.dim)

        if self.sort_bfgs:
            self._bfgs_vals_grads = self._bfgs_vals_grads_sort
        else:
            self._bfgs_vals_grads = lambda v, p, g: (v, g)

        if not hasattr(initial_state, 'memory'):
            initial_state.memory \
                = cdict(value=jnp.zeros((initial_extra.parameters.m, scenario.dim)),
                        prior_potential=jnp.zeros(initial_extra.parameters.m) - jnp.inf,
                        grad_prior_potential=jnp.zeros((initial_extra.parameters.m, scenario.dim)),
                        likelihood_potential=jnp.zeros(initial_extra.parameters.m) - jnp.inf,
                        grad_likelihood_potential=jnp.zeros((initial_extra.parameters.m, scenario.dim)),
                        init_hessian_sqrt_diag=initial_extra.parameters.init_hessian_sqrt_diag)
            initial_state.memory \
                = self.update_memory(initial_state.memory, jnp.zeros(scenario.dim),
                                     -jnp.inf, jnp.zeros(scenario.dim),
                                     -jnp.inf, jnp.zeros(scenario.dim),
                                     scenario.temperature)

        return initial_state, initial_extra

    @staticmethod
    def _bfgs_vals_grads(value: jnp.ndarray,
                         potential: float,
                         grad_potential: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        raise NotImplementedError

    @staticmethod
    def _bfgs_vals_grads_sort(value: jnp.ndarray,
                              potential: jnp.ndarray,
                              grad_potential: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        inds = jnp.argsort(potential)
        return value[inds], grad_potential[inds]

    def update_memory(self,
                      memory: cdict,
                      value: jnp.ndarray,
                      prior_potential: float,
                      grad_prior_potential: jnp.ndarray,
                      likelihood_potential: float,
                      grad_likelihood_potential: jnp.ndarray,
                      temperature: float) -> cdict:
        memory.value = jnp.append(memory.value[1:], value[jnp.newaxis], axis=0)
        memory.prior_potential = jnp.append(memory.prior_potential[1:], prior_potential)
        memory.grad_prior_potential \
            = jnp.append(memory.grad_prior_potential[1:], grad_prior_potential[jnp.newaxis], axis=0)
        memory.likelihood_potential = jnp.append(memory.likelihood_potential[1:], likelihood_potential)
        memory.grad_likelihood_potential \
            = jnp.append(memory.grad_likelihood_potential[1:], grad_likelihood_potential[jnp.newaxis], axis=0)

        value, grad_potential \
            = self._bfgs_vals_grads(memory.value,
                                    memory.prior_potential + temperature * memory.likelihood_potential,
                                    memory.grad_prior_potential + temperature * memory.grad_likelihood_potential)

        update_bools = jnp.array(memory.prior_potential[:-1] > -jnp.inf)

        memory.ps, memory.qs, memory.us, memory.ts \
            = bfgs_sqrt_pqut(value, grad_potential,
                             init_hessian_sqrt_diag=memory.init_hessian_sqrt_diag, r=self.parameters.pos_def_r,
                             update_bools=update_bools)
        return memory

    def always(self,
               scenario: Scenario,
               reject_state: cdict,
               reject_extra: cdict) -> Tuple[cdict, cdict]:
        d = scenario.dim

        stepsize = reject_extra.parameters.stepsize
        friction = reject_extra.parameters.friction

        random_keys = random.split(reject_extra.random_key, 2)

        reject_extra.random_key = random_keys[-1]

        reject_state.memory = self.update_memory(reject_state.memory.copy(),
                                                 reject_state.value,
                                                 reject_state.prior_potential,
                                                 reject_state.grad_prior_potential,
                                                 reject_state.likelihood_potential,
                                                 reject_state.grad_likelihood_potential,
                                                 scenario.temperature)

        reject_state.momenta = - reject_state.momenta

        # Update p - exactly according to solution of OU process
        # Accepted even if leapfrog step is rejected
        precon_norm = bfgs_sqrt_prod(reject_state.memory.us, reject_state.memory.ts,
                                     random.normal(random_keys[-2], (d,)),
                                     reject_state.memory.init_hessian_sqrt_diag)
        reject_state.momenta = reject_state.momenta * jnp.exp(- friction * stepsize) \
                               + jnp.sqrt(1 - jnp.exp(- 2 * friction * stepsize)) * precon_norm

        return reject_state, reject_extra

    def proposal(self,
                 scenario: Scenario,
                 reject_state: cdict,
                 reject_extra: cdict) -> Tuple[cdict, cdict]:
        proposed_state = reject_state.copy()

        random_keys = random.split(reject_extra.random_key, self.parameters.leapfrog_steps + 1)
        reject_extra.random_key = random_keys[-1]

        all_leapfrog = bfgs_leapfrog(scenario.prior_potential_and_grad,
                                     scenario.likelihood_potential_and_grad,
                                     proposed_state,
                                     reject_extra.parameters.stepsize,
                                     random_keys[:self.parameters.leapfrog_steps],
                                     proposed_state.memory.ps,
                                     proposed_state.memory.qs,
                                     1 / proposed_state.memory.init_hessian_sqrt_diag,
                                     scenario.temperature)

        proposed_state.momenta = - all_leapfrog.momenta[-1]

        proposed_state.value = all_leapfrog.value[-1]
        proposed_state.prior_potential = all_leapfrog.prior_potential[-1]
        proposed_state.grad_prior_potential = all_leapfrog.grad_prior_potential[-1]
        proposed_state.likelihood_potential = all_leapfrog.likelihood_potential[-1]
        proposed_state.grad_likelihood_potential = all_leapfrog.grad_likelihood_potential[-1]
        proposed_state.potential \
            = proposed_state.prior_potential + scenario.temperature * proposed_state.likelihood_potential
        proposed_state.grad_potential \
            = proposed_state.grad_prior_potential + scenario.temperature * proposed_state.grad_likelihood_potential

        return proposed_state, reject_extra

    def acceptance_probability(self,
                               scenario: Scenario,
                               reject_state: cdict, reject_extra: cdict,
                               proposed_state: cdict, proposed_extra: cdict) -> Union[float, jnp.ndarray]:
        rej_sqrtprec_momenta_diff = bfgs_sqrt_transpose_prod(reject_state.memory.ps, reject_state.memory.qs,
                                                             reject_state.momenta,
                                                             1 / reject_state.memory.init_hessian_sqrt_diag)
        rej_momenta_pot = 0.5 * jnp.square(rej_sqrtprec_momenta_diff).sum()

        prop_sqrtprec_momenta_diff = bfgs_sqrt_transpose_prod(reject_state.memory.ps, reject_state.memory.qs,
                                                              proposed_state.momenta,
                                                              1 / reject_state.memory.init_hessian_sqrt_diag)
        prop_momenta_pot = 0.5 * jnp.square(prop_sqrtprec_momenta_diff).sum()

        pre_min_alpha = jnp.exp(- proposed_state.potential
                                + reject_state.potential
                                - prop_momenta_pot
                                + rej_momenta_pot)
        alpha = jnp.minimum(1., pre_min_alpha)
        alpha = jnp.where(
            jnp.any(jnp.abs(jnp.append(proposed_state.value, proposed_state.momenta))
                    > reject_extra.parameters.max_val), 0., alpha)
        return alpha


@partial(jit, static_argnums=(0, 1))
def _bfgs_leapfrog(prior_potential_and_grad: Callable,
                   likelihood_potential_and_grad: Callable,
                   state: cdict,
                   stepsize: float,
                   random_keys: jnp.ndarray,
                   bfgs_ps: jnp.ndarray,
                   bfgs_qs: jnp.ndarray,
                   bfgs_init_invhessian_sqrt_diag: Union[float, jnp.ndarray],
                   temperature: float) -> cdict:
    leapfrog_steps = len(random_keys)

    def leapfrog_step(init_state: cdict,
                      i: int):
        new_state = init_state.copy()

        p_half = init_state.momenta - stepsize / 2. * init_state.grad_potential
        minv_p_half = bfgs_sqrt_transpose_prod(bfgs_ps, bfgs_qs, p_half, bfgs_init_invhessian_sqrt_diag)
        minv_p_half = bfgs_sqrt_prod(bfgs_ps, bfgs_qs, minv_p_half, bfgs_init_invhessian_sqrt_diag)

        new_state.value = init_state.value + stepsize * minv_p_half

        new_state.prior_potential, new_state.grad_prior_potential \
            = prior_potential_and_grad(new_state.value, random_keys[i])

        new_state.likelihood_potential, new_state.grad_likelihood_potential \
            = likelihood_potential_and_grad(new_state.value, random_keys[i])

        new_state.potential = new_state.prior_potential + temperature * new_state.likelihood_potential
        new_state.grad_potential = new_state.grad_prior_potential + temperature * new_state.grad_likelihood_potential

        new_state.momenta = p_half - stepsize / 2. * new_state.grad_potential

        next_sample_chain = new_state.copy()
        next_sample_chain.momenta = jnp.vstack([p_half, new_state.momenta])
        return new_state, next_sample_chain

    final_leapfrog, all_leapfrog = scan(leapfrog_step, state, jnp.arange(leapfrog_steps))

    all_leapfrog.momenta = jnp.concatenate(all_leapfrog.momenta)

    all_leapfrog = state[jnp.newaxis] + all_leapfrog

    return all_leapfrog


def bfgs_leapfrog(prior_potential_and_grad: Callable,
                  likelihood_potential_and_grad: Callable,
                  state: cdict,
                  stepsize: float,
                  random_keys: jnp.ndarray,
                  bfgs_ps: jnp.ndarray,
                  bfgs_qs: jnp.ndarray,
                  bfgs_init_invhessian_sqrt_diag: Union[float, jnp.ndarray] = 1.,
                  temperature: float = 1.) -> cdict:
    return _bfgs_leapfrog(prior_potential_and_grad, likelihood_potential_and_grad,
                          state, stepsize, random_keys, bfgs_ps, bfgs_qs, bfgs_init_invhessian_sqrt_diag, temperature)
