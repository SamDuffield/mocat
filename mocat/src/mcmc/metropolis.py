########################################################################################################################
# Module: mcmc/metropolis.py
# Description: Accept/reject corrections for MCMC samplers.
#
# Web: https://github.com/SamDuffield/mocat
########################################################################################################################

from typing import Tuple, Union, Type
from inspect import isclass

from jax import random, numpy as jnp
from jax.lax import cond

from mocat.src.core import cdict, Scenario
from mocat.src.mcmc.sampler import MCMCSampler, Correction


def mh_acceptance_probability(sampler: MCMCSampler,
                              scenario: Scenario,
                              reject_state: cdict, reject_extra: cdict,
                              proposed_state: cdict, proposed_extra: cdict) -> Union[float, jnp.ndarray]:
    pre_min_alpha = jnp.exp(- proposed_state.potential
                            + reject_state.potential
                            - sampler.proposal_potential(scenario,
                                                         proposed_state, proposed_extra,
                                                         reject_state, reject_extra)
                            + sampler.proposal_potential(scenario,
                                                         reject_state, reject_extra,
                                                         proposed_state, proposed_extra))

    return jnp.minimum(1., pre_min_alpha)


class Metropolis(Correction):

    def startup(self,
                scenario: Scenario,
                sampler: MCMCSampler,
                n: int,
                initial_state: cdict,
                initial_extra: cdict,
                **kwargs) -> Tuple[cdict, cdict]:
        initial_state, initial_extra = super().startup(scenario, sampler, n,
                                                       initial_state, initial_extra, **kwargs)
        initial_state.alpha = 1.
        return initial_state, initial_extra

    def correct(self,
                scenario: Scenario,
                sampler: MCMCSampler,
                reject_state: cdict,
                reject_extra: cdict,
                proposed_state: cdict,
                proposed_extra: cdict) -> Tuple[cdict, cdict]:
        alpha = sampler.acceptance_probability(scenario,
                                               reject_state, reject_extra,
                                               proposed_state, proposed_extra)
        alpha = jnp.where(jnp.isnan(alpha), 0., alpha)

        random_key, subkey = random.split(proposed_extra.random_key)
        u = random.uniform(subkey)

        new_state, new_extra = cond(u < alpha,
                                    lambda _: (proposed_state, proposed_extra),
                                    lambda _: (reject_state, reject_extra),
                                    None)

        new_state.alpha = alpha
        new_extra.random_key = random_key
        return new_state, new_extra


class RMMetropolis(Correction):

    def __init__(self,
                 super_correction: Union[Correction, Type[Correction]] = Metropolis(),
                 adapt_cut_off: int = jnp.inf,
                 rm_stepsize_scale: float = 1.,
                 rm_stepsize_neg_exponent: float = 2 / 3,
                 log_update: bool = True):
        super().__init__()
        self.super_correction = super_correction() if isclass(super_correction) else super_correction
        self.super_correction.__init__()
        self.adapt_cut_off = adapt_cut_off
        self.rm_stepsize_scale = rm_stepsize_scale
        self.rm_stepsize_neg_exponent = rm_stepsize_neg_exponent
        self.log_update = log_update
        self._param_update = self.log_robbins_monro_update if self.log_update else self.robbins_monro_update
        self.tuning = None

    def startup(self,
                scenario: Scenario,
                sampler: MCMCSampler,
                n: int,
                initial_state: cdict,
                initial_extra: cdict,
                **kwargs) -> Tuple[cdict, cdict]:
        initial_state, initial_extra = super().startup(scenario, sampler, n,
                                                       initial_state, initial_extra, **kwargs)
        # Set tuning parameter (i.e. stepsize) to 2.38^2/d if not initiated and adaptive
        if hasattr(sampler.parameters, sampler.tuning.parameter) \
                and getattr(sampler.parameters, sampler.tuning.parameter) is None:
            setattr(initial_extra.parameters, sampler.tuning.parameter, 2.38 ** 2 / scenario.dim)

        setattr(initial_state, sampler.tuning.parameter, getattr(initial_extra.parameters,
                                                                 sampler.tuning.parameter))

        initial_state, initial_extra = self.super_correction.startup(scenario, sampler, n,
                                                                     initial_state, initial_extra, **kwargs)

        sampler.tuning.monotonicity = 1 if sampler.tuning.monotonicity in (1, 'increasing') else -1

        self.tuning = sampler.tuning

        self._param_update = self.log_robbins_monro_update if self.log_update else self.robbins_monro_update
        return initial_state, initial_extra

    def correct(self,
                scenario: Scenario,
                sampler: MCMCSampler,
                reject_state: cdict,
                reject_extra: cdict,
                proposed_state: cdict,
                proposed_extra: cdict) -> Tuple[cdict, cdict]:
        corrected_state, corrected_extra = self.super_correction.correct(scenario, sampler,
                                                                         reject_state, reject_extra,
                                                                         proposed_state, proposed_extra)

        adapted_state, adapted_extra = cond(corrected_extra.iter <= self.adapt_cut_off,
                                            lambda carry: self.adapt(*carry),
                                            lambda carry: carry,
                                            (corrected_state, corrected_extra))

        setattr(adapted_state, sampler.tuning.parameter, getattr(adapted_extra.parameters,
                                                                 sampler.tuning.parameter))
        return adapted_state, adapted_extra

    def adapt(self,
              state: cdict,
              extra: cdict) -> Tuple[cdict, cdict]:
        param = extra.parameters.__dict__[self.tuning.parameter]
        param = self._param_update(param, getattr(state, self.tuning.metric),
                                   self.tuning.target, self.tuning.monotonicity,
                                   self.rm_stepsize_scale
                                   * extra.iter ** -self.rm_stepsize_neg_exponent)
        extra.parameters.__dict__[self.tuning.parameter] = param
        return state, extra

    @staticmethod
    def robbins_monro_update(param, metric_value, target, monotonicity, rm_stepsize):
        return param - rm_stepsize * (metric_value - target) * monotonicity

    @staticmethod
    def log_robbins_monro_update(param, metric_value, target, monotonicity, rm_stepsize):
        log_param = jnp.log(param)
        new_log_param = RMMetropolis.robbins_monro_update(log_param,
                                                          metric_value,
                                                          target,
                                                          monotonicity,
                                                          rm_stepsize)
        return jnp.exp(new_log_param)
