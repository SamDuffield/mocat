########################################################################################################################
# Module: tests/test_abc_linear_gaussian.py
# Description: Tests for ABC sampling (intractable likelihoods).
#
# Web: https://github.com/SamDuffield/mocat
########################################################################################################################

import unittest

import jax.numpy as jnp
from jax.random import PRNGKey
import numpy.testing as npt

from mocat.src.sample import run
from mocat.src.abc.scenarios.gk import GKTransformedUniformPrior, GKOnlyATransformedUniformPrior
from mocat.src.abc.importance import VanillaABC
from mocat.src.abc.mcmc import RandomWalkABC, RMMetropolisDiagStepsize
from mocat.src.abc.smc import MetropolisedABCSMCSampler


class GKThinOrder(GKTransformedUniformPrior):
    num_thin: int = 100
    n_unsummarised_data: int = 1000

    def summarise_data(self,
                       data: jnp.ndarray):
        order_stats = data.sort()
        thin_inds = jnp.linspace(0, len(data), self.num_thin, endpoint=False, dtype='int32')
        return order_stats[thin_inds]


class GKOnlyAThinOrder(GKOnlyATransformedUniformPrior):
    num_thin: int = 100
    n_unsummarised_data: int = 1000

    def summarise_data(self,
                       data: jnp.ndarray):
        order_stats = data.sort()
        thin_inds = jnp.linspace(0, len(data), self.num_thin, endpoint=False, dtype='int32')
        return order_stats[thin_inds]


class TestGK(unittest.TestCase):

    n = int(1e5)
    onlyA = False

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        if self.onlyA:
            self.scenario = GKOnlyAThinOrder()
            self.true_constrained_params = jnp.array([3.])
        else:
            self.scenario = GKThinOrder()
            self.true_constrained_params = jnp.array([3., 1., 2., 0.5])
        self.true_unconstrained_params = self.scenario.unconstrain(self.true_constrained_params)

        self.scenario.data = self.scenario.likelihood_sample(self.true_unconstrained_params,
                                                             random_key=PRNGKey(0))

    def _test_mean(self,
                   vals: jnp.ndarray,
                   precision: float = 3.0):
        samp_mean = vals.mean(axis=0)
        self.assertLess(jnp.abs(self.true_unconstrained_params - samp_mean).sum(), precision)


class TestVanillaGK(TestGK):

    def test_pre_threshold(self):
        threshold = 5.0
        sample = run(self.scenario, VanillaABC(threshold=threshold), n=self.n,
                     random_key=PRNGKey(0))
        self._test_mean(sample.value[sample.log_weight > -jnp.inf], 10.)

    def test_post_threshold(self):
        acceptance_rate = 0.1
        sampler = VanillaABC(acceptance_rate=acceptance_rate)
        sample = run(self.scenario, sampler, n=self.n,
                     random_key=PRNGKey(0))
        self._test_mean(sample.value[sample.log_weight > -jnp.inf], 10.)
        npt.assert_almost_equal(jnp.mean(sample.log_weight == 0), acceptance_rate, decimal=3)
        self.assertNotEqual(sampler.parameters.threshold, jnp.inf)


class TestVanillaGKOnlyA(TestGK):
    onlyA = True

    def test_pre_threshold(self):
        threshold = 1000.
        sample = run(self.scenario, VanillaABC(threshold=threshold), n=self.n,
                     random_key=PRNGKey(0))
        self._test_mean(sample.value[sample.log_weight > -jnp.inf], 10.)

    def test_post_threshold(self):
        acceptance_rate = 0.1
        sampler = VanillaABC(acceptance_rate=acceptance_rate)
        sample = run(self.scenario, sampler, n=self.n,
                     random_key=PRNGKey(0))
        self._test_mean(sample.value[sample.log_weight > -jnp.inf], 10.)
        npt.assert_almost_equal(jnp.mean(sample.log_weight == 0), acceptance_rate, decimal=3)
        self.assertNotEqual(sampler.parameters.threshold, jnp.inf)


class TestMCMCGK(TestGK):

    def test_adaptive_diag_stepsize(self):
        sampler = RandomWalkABC()
        sampler.tuning.target = 0.1
        sample = run(self.scenario, sampler, n=self.n,
                     random_key=PRNGKey(0), correction=RMMetropolisDiagStepsize(rm_stepsize_scale=0.01))
        self._test_mean(sample.value)
        npt.assert_almost_equal(sample.alpha.mean(), sampler.tuning.target, decimal=1)


class TestMCMCGKOnlyA(TestGK):

    onlyA = True

    def test_adaptive_diag_stepsize(self):
        sampler = RandomWalkABC()
        sampler.tuning.target = 0.1
        sample = run(self.scenario, sampler, n=self.n,
                     random_key=PRNGKey(0), correction=RMMetropolisDiagStepsize(rm_stepsize_scale=0.1))
        self._test_mean(sample.value)
        npt.assert_almost_equal(sample.alpha.mean(), sampler.tuning.target, decimal=1)


class TestMetSMCGK(TestGK):
    n = int(1e3)

    def test_threshold_preschedule(self):
        threshold_schedule = jnp.linspace(100., 10., 10)
        sampler = MetropolisedABCSMCSampler(threshold_schedule=threshold_schedule)
        sample = run(self.scenario, sampler, n=self.n,
                     random_key=PRNGKey(0))
        self._test_mean(sample.value[-1])

    def test_adaptive(self):
        retain_parameter = 0.8
        sampler = MetropolisedABCSMCSampler(ess_threshold_retain=retain_parameter, termination_alpha=0.1)
        sample = run(self.scenario, sampler, n=self.n,
                     random_key=PRNGKey(0))
        self._test_mean(sample.value[-1])


class TestMetSMCGKOnlyA(TestGK):
    n = int(1e3)
    onlyA = True

    def test_threshold_preschedule(self):
        threshold_schedule = jnp.linspace(100., 10., 10)
        sampler = MetropolisedABCSMCSampler(threshold_schedule=threshold_schedule)
        sample = run(self.scenario, sampler, n=self.n,
                     random_key=PRNGKey(0))
        self._test_mean(sample.value[-1])

    def test_adaptive(self):
        retain_parameter = 0.8
        sampler = MetropolisedABCSMCSampler(ess_threshold_retain=retain_parameter, termination_alpha=0.1)
        sample = run(self.scenario, sampler, n=self.n,
                     random_key=PRNGKey(0))
        self._test_mean(sample.value[-1])
