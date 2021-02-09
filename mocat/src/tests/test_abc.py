########################################################################################################################
# Module: tests/test_abc.py
# Description: Tests for ABC sampling (intractable likelihoods).
#
# Web: https://github.com/SamDuffield/mocat
########################################################################################################################

import unittest

import jax.numpy as np
from jax.random import PRNGKey
import numpy.testing as npt

from mocat.src.core import cdict
from mocat.src.sample import run
from mocat.src.abc.scenarios.linear_gaussian import LinearGaussian
from mocat.src.abc.importance import VanillaABC
from mocat.src.abc.mcmc import RandomWalkABC, RMMetropolisDiagStepsize
from mocat.src.abc.smc import MetropolisedABCSMCSampler


class TestLinearGaussianABC(unittest.TestCase):

    prior_mean = np.zeros(3)
    prior_covariance = np.array([[1.0, -0.6, 0.5], [-0.6, 0.7, 0.1], [0.5, -0.6, 3.0]])
    data = np.array([1.4, -2.5])
    likelihood_matrix = np.array([[1.0, 0.5, 0.], [0.2, 0.3, 1.5]])
    likelihood_covariance = np.array([[0.02, -0.01], [-0.01, 0.05]])

    kalman_gain = prior_covariance @ likelihood_matrix.T \
                  @ np.linalg.inv(likelihood_matrix @ prior_covariance @ likelihood_matrix.T + likelihood_covariance)
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

    @staticmethod
    def value_from_sample(sample: cdict):
        if sample.value.ndim == 3:
            val = sample.value[-1]
        else:
            val = sample.value
        return val

    def _test_mean(self,
                   sample: cdict,
                   precision: float = 3.0):
        val = self.value_from_sample(sample)
        samp_mean = val.mean(axis=0)
        self.assertLess(np.abs(self.posterior_mean - samp_mean).sum(), precision)

    def _test_cov(self,
                  sample: cdict,
                  precision: float = 3.0):
        val = self.value_from_sample(sample)
        samp_cov = np.cov(val.T)
        self.assertLess(np.abs(self.posterior_covariance - samp_cov).sum(), precision)


class TestVanillaLinearGaussianABC(TestLinearGaussianABC):

    def test_pre_threshold(self):
        threshold = 5.0
        sample = run(self.scenario, VanillaABC(threshold=threshold), n=self.n,
                     random_key=PRNGKey(0))
        self._test_mean(sample, 10.)
        self._test_cov(sample, 10.)

    def test_post_threshold(self):
        acceptance_rate = 0.1
        sampler = VanillaABC(acceptance_rate=acceptance_rate)
        sample = run(self.scenario, sampler, n=self.n,
                     random_key=PRNGKey(0))
        self._test_mean(sample, 10.)
        self._test_cov(sample, 10.)
        npt.assert_almost_equal(np.mean(sample.log_weight == 0), acceptance_rate, decimal=3)
        self.assertNotEqual(sampler.parameters.threshold, np.inf)


class TestMCMCLinearGaussianABC(TestLinearGaussianABC):

    threshold = 1.0

    def test_fixed_stepsize(self):
        stepsize = 0.1
        sample = run(self.scenario, RandomWalkABC(stepsize=stepsize, threshold=self.threshold), n=self.n,
                     random_key=PRNGKey(0))
        self._test_mean(sample)
        self._test_cov(sample)

    def test_adaptive_diag_stepsize(self):
        sampler = RandomWalkABC()
        sampler.tuning.target = 0.1
        sample = run(self.scenario, sampler, n=self.n,
                     random_key=PRNGKey(0), correction=RMMetropolisDiagStepsize())
        self._test_mean(sample)
        self._test_cov(sample)
        npt.assert_almost_equal(sample.alpha.mean(), sampler.tuning.target, decimal=1)
        npt.assert_array_almost_equal(sample.stepsize[-1],
                                      np.diag(np.cov(sample.value, rowvar=False)) / self.scenario.dim * 2.38 ** 2,
                                      decimal=1)


class TestMetSMCLinearGaussianABC(TestLinearGaussianABC):

    n = int(1e3)

    def test_threshold_preschedule(self):
        threshold_schedule = np.linspace(10, 0.1, 10)
        sampler = MetropolisedABCSMCSampler(threshold_schedule=threshold_schedule)
        sample = run(self.scenario, sampler, n=self.n,
                     random_key=PRNGKey(0))
        self._test_mean(sample)
        self._test_cov(sample)

    def test_adaptive(self):
        retain_parameter = 0.8
        sampler = MetropolisedABCSMCSampler(threshold_quantile_retain=retain_parameter)
        sample = run(self.scenario, sampler, n=self.n,
                     random_key=PRNGKey(0))
        self._test_mean(sample)
        self._test_cov(sample)


