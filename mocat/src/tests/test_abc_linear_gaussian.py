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
from mocat.src.abc.scenarios.linear_gaussian import LinearGaussian
from mocat.src.abc.importance import VanillaABC
from mocat.src.abc.mcmc import RandomWalkABC, RMMetropolisDiagStepsize
from mocat.src.abc.smc import MetropolisedABCSMCSampler


class TestLinearGaussianABC(unittest.TestCase):
    prior_mean = jnp.zeros(3)
    prior_covariance = jnp.array([[1.0, -0.6, 0.5], [-0.6, 0.7, 0.1], [0.5, -0.6, 3.0]])
    data = jnp.array([1.4, -2.5])
    likelihood_matrix = jnp.array([[1.0, 0.5, 0.], [0.2, 0.3, 1.5]])
    likelihood_covariance = jnp.array([[0.02, -0.01], [-0.01, 0.05]])

    kalman_gain = prior_covariance @ likelihood_matrix.T \
                  @ jnp.linalg.inv(likelihood_matrix @ prior_covariance @ likelihood_matrix.T + likelihood_covariance)
    posterior_mean = prior_mean + kalman_gain @ (data - likelihood_matrix @ prior_mean)
    posterior_covariance = prior_covariance - kalman_gain @ likelihood_matrix @ prior_covariance

    n = int(1e5)

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.scenario = LinearGaussian(prior_mean=self.prior_mean,
                                       prior_covariance=self.prior_covariance,
                                       data=self.data,
                                       likelihood_matrix=self.likelihood_matrix,
                                       likelihood_covariance=self.likelihood_covariance)

    def _test_mean(self,
                   val: jnp.ndarray,
                   precision: float = 3.0):
        samp_mean = val.mean(axis=0)
        self.assertLess(jnp.abs(self.posterior_mean - samp_mean).sum(), precision)

    def _test_cov(self,
                   val: jnp.ndarray,
                   precision: float = 3.0):
        samp_cov = jnp.cov(val.T)
        self.assertLess(jnp.abs(self.posterior_covariance - samp_cov).sum(), precision)


class TestVanillaLinearGaussianABC(TestLinearGaussianABC):

    def test_pre_threshold(self):
        threshold = 5.0
        sample = run(self.scenario, VanillaABC(threshold=threshold), n=self.n,
                     random_key=PRNGKey(0))
        self._test_mean(sample.value[sample.log_weight > -jnp.inf], 10.)
        self._test_cov(sample.value[sample.log_weight > -jnp.inf], 10.)

    def test_post_threshold(self):
        acceptance_rate = 0.1
        sampler = VanillaABC(acceptance_rate=acceptance_rate)
        sample = run(self.scenario, sampler, n=self.n,
                     random_key=PRNGKey(0))
        self._test_mean(sample.value[sample.log_weight > -jnp.inf], 10.)
        self._test_cov(sample.value[sample.log_weight > -jnp.inf], 10.)
        npt.assert_almost_equal(jnp.mean(sample.log_weight == 0), acceptance_rate, decimal=3)
        self.assertNotEqual(sampler.parameters.threshold, jnp.inf)


class TestMCMCLinearGaussianABC(TestLinearGaussianABC):
    threshold = 1.0

    def test_fixed_stepsize(self):
        stepsize = 0.1
        sample = run(self.scenario, RandomWalkABC(stepsize=stepsize, threshold=self.threshold), n=self.n,
                     random_key=PRNGKey(0))
        self._test_mean(sample.value)
        self._test_cov(sample.value)

    def test_adaptive_diag_stepsize(self):
        sampler = RandomWalkABC()
        sampler.tuning.target = 0.1
        sample = run(self.scenario, sampler, n=self.n,
                     random_key=PRNGKey(0), correction=RMMetropolisDiagStepsize())
        self._test_mean(sample.value)
        self._test_cov(sample.value)
        npt.assert_almost_equal(sample.alpha.mean(), sampler.tuning.target, decimal=1)
        npt.assert_array_almost_equal(sample.stepsize[-1],
                                      jnp.diag(jnp.cov(sample.value, rowvar=False)) / self.scenario.dim * 2.38 ** 2,
                                      decimal=1)


class TestMetSMCLinearGaussianABC(TestLinearGaussianABC):
    n = int(1e3)

    def test_threshold_preschedule(self):
        threshold_schedule = jnp.linspace(10, 0.1, 10)
        sampler = MetropolisedABCSMCSampler(threshold_schedule=threshold_schedule)
        sample = run(self.scenario, sampler, n=self.n,
                     random_key=PRNGKey(0))
        self._test_mean(sample.value[-1])
        self._test_cov(sample.value[-1])

    def test_adaptive(self):
        retain_parameter = 0.8
        sampler = MetropolisedABCSMCSampler(ess_threshold_retain=retain_parameter)
        sample = run(self.scenario, sampler,
                     n=self.n,
                     random_key=PRNGKey(0))
        self._test_mean(sample.value[-1])
        self._test_cov(sample.value[-1])
