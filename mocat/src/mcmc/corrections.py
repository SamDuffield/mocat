########################################################################################################################
# Module: mcmc/corrections.py
# Description: Correction mechanisms for MCMC samplers.
#
# Web: https://github.com/SamDuffield/mocat
########################################################################################################################

from typing import Tuple, Union, Type
from inspect import isclass

from jax import random, vmap, numpy as np
from jax.lax import cond
from jax.ops import index_update

from mocat.src.core import CDict, Scenario
from mocat.src.mcmc.sampler import MCMCSampler


class Correction:

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __repr__(self):
        return f'mocat.Correction.{self.__class__.__name__}'

    def startup(self,
                scenario: Scenario,
                sampler: MCMCSampler,
                initial_state: CDict,
                inital_extra: CDict) -> Tuple[CDict, CDict]:
        return initial_state, inital_extra

    def correct(self,
                scenario: Scenario,
                sampler: MCMCSampler,
                reject_state: CDict,
                reject_extra: CDict,
                proposed_state: CDict,
                proposed_extra: CDict) -> Tuple[CDict, CDict]:
        raise NotImplementedError

    def __call__(self,
                 scenario: Scenario,
                 sampler: MCMCSampler,
                 reject_state: CDict,
                 reject_extra: CDict,
                 proposed_state: CDict,
                 proposed_extra: CDict) -> Tuple[CDict, CDict]:
        return self.correct(scenario, sampler,
                            reject_state, reject_extra,
                            proposed_state, proposed_extra)


class Uncorrected(Correction):

    def correct(self,
                scenario: Scenario,
                sampler: MCMCSampler,
                reject_state: CDict,
                reject_extra: CDict,
                proposed_state: CDict,
                proposed_extra: CDict) -> Tuple[CDict, CDict]:
        return proposed_state, proposed_extra


def update_ensemble_potential(scenario: Scenario,
                              state: CDict,
                              extra: CDict) -> CDict:
    ensemble_index = extra.iter % extra.parameters.n_ensemble
    state.potential = index_update(state.potential, ensemble_index,
                                   scenario.potential(state.value[ensemble_index]))
    return state


def update_potential(scenario: Scenario,
                     state: CDict,
                     extra: CDict) -> CDict:
    state.potential = scenario.potential(state.value)
    return state


class Metropolis(Correction):

    def __init__(self):
        super().__init__()
        self._update_potential = None

    def startup(self,
                scenario: Scenario,
                sampler: MCMCSampler,
                initial_state: CDict,
                initial_extra: CDict) -> Tuple[CDict, CDict]:
        initial_state.alpha = 1.

        if initial_state.value.ndim == 2:
            initial_state.potential = vmap(scenario.potential)(initial_state.value)
            self._update_potential = update_ensemble_potential
        else:
            initial_state.potential = scenario.potential(initial_state.value)
            self._update_potential = update_potential
        return initial_state, initial_extra

    def correct(self,
                scenario: Scenario,
                sampler: MCMCSampler,
                reject_state: CDict,
                reject_extra: CDict,
                proposed_state: CDict,
                proposed_extra: CDict) -> Tuple[CDict, CDict]:
        proposed_state = self._update_potential(scenario, proposed_state, proposed_extra)

        alpha = sampler.acceptance_probability(scenario,
                                               reject_state, reject_extra,
                                               proposed_state, proposed_extra)
        alpha = np.where(np.isnan(alpha), 0., alpha)

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
                 adapt_cut_off: int = np.inf,
                 rm_stepsize_scale: float = 1.,
                 rm_stepsize_neg_exponent: float = 0.75,
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
                initial_state: CDict,
                initial_extra: CDict) -> Tuple[CDict, CDict]:
        # Set tuning parameter (i.e. stepsize) to 2.38^2/d if not initiated and adaptive
        if hasattr(sampler.parameters, sampler.tuning.parameter) \
                and getattr(sampler.parameters, sampler.tuning.parameter) is None:
            setattr(sampler.parameters, sampler.tuning.parameter, 2.38 ** 2 / scenario.dim)
            setattr(initial_extra.parameters, sampler.tuning.parameter, 2.38 ** 2 / scenario.dim)

        setattr(initial_state, sampler.tuning.parameter, getattr(initial_extra.parameters,
                                                                 sampler.tuning.parameter))

        initial_state, initial_extra = self.super_correction.startup(scenario, sampler, initial_state, initial_extra)

        sampler.tuning.monotonicity = 1 if sampler.tuning.monotonicity in (1, 'increasing') else -1

        self.tuning = sampler.tuning

        self._param_update = self.log_robbins_monro_update if self.log_update else self.robbins_monro_update
        return initial_state, initial_extra

    def correct(self,
                scenario: Scenario,
                sampler: MCMCSampler,
                reject_state: CDict,
                reject_extra: CDict,
                proposed_state: CDict,
                proposed_extra: CDict) -> Tuple[CDict, CDict]:
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
              state: CDict,
              extra: CDict) -> Tuple[CDict, CDict]:
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
        log_param = np.log(param)
        new_log_param = RMMetropolis.robbins_monro_update(log_param,
                                                          metric_value,
                                                          target,
                                                          monotonicity,
                                                          rm_stepsize)
        return np.exp(new_log_param)
