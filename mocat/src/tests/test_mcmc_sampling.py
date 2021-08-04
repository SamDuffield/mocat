########################################################################################################################
# Module: tests/test_mcmc_sampling.py
# Description: Tests for MCMC sampling.
#
# Web: https://github.com/SamDuffield/mocat
########################################################################################################################

import unittest

import jax.numpy as jnp
from jax.random import PRNGKey
import numpy.testing as npt

from mocat.src.core import cdict
from mocat.src.scenarios.twodim import toy_examples
from mocat.src.metrics import ess_autocorrelation
from mocat.src.mcmc import standard_mcmc
from mocat.src.mcmc.metropolis import Metropolis, RMMetropolis
from mocat.src.sample import run


class TestMetropolisCorrelatedGaussian(unittest.TestCase):

    sampler = None

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.scenario_cov = jnp.array([[1., 0.9], [0.9, 2.]])
        self.scenario = toy_examples.Gaussian(covariance=self.scenario_cov)
        self.n = int(1e5)

        self.__test__ = self.sampler is not None
        if self.__test__:
            self.adapt_sample = run(self.scenario, self.sampler, self.n, PRNGKey(0), correction=RMMetropolis())
            self.sampler.parameters.stepsize = self.adapt_sample.stepsize[-1]
            self.warmstart_sample = run(self.scenario, self.sampler, self.n, PRNGKey(0), correction=Metropolis)

    def _test_mean(self,
                   sample: cdict):
        if sample.value.ndim == 3:
            val = jnp.concatenate(sample.value)
        else:
            val = sample.value
        samp_mean = val.mean(axis=0)
        npt.assert_array_almost_equal(samp_mean, jnp.zeros(2), decimal=1)

    def _test_cov(self,
                  sample: cdict):
        if sample.value.ndim == 3:
            val = jnp.concatenate(sample.value)
        else:
            val = sample.value
        samp_cov = jnp.cov(val.T)
        npt.assert_array_almost_equal(samp_cov, self.scenario_cov, decimal=1)

    def _test_ess_autocorrelation(self,
                  sample: cdict):
        samp_ess_pot = ess_autocorrelation(sample)
        samp_ess_0 = ess_autocorrelation(sample.value[:, 0])
        samp_ess_1 = ess_autocorrelation(sample.value[:, 1])

        self.assertFalse(samp_ess_pot == samp_ess_0)
        self.assertFalse(samp_ess_pot == samp_ess_1)
        self.assertFalse(samp_ess_0 == samp_ess_1)

        self.assertGreater(samp_ess_pot, self.n / 100)
        self.assertGreater(samp_ess_0, self.n / 100)
        self.assertGreater(samp_ess_1, self.n / 100)

    def _test_acceptance_rate(self,
                              sample: cdict):
        npt.assert_almost_equal(sample.alpha.mean(), self.sampler.tuning.target, decimal=1)


class TestRandomWalkCorrelatedGaussian(TestMetropolisCorrelatedGaussian):
    sampler = standard_mcmc.RandomWalk()

    def test_mean(self):
        self._test_mean(self.adapt_sample)
        self._test_mean(self.warmstart_sample)

    def test_cov(self):
        self._test_cov(self.adapt_sample)
        self._test_cov(self.warmstart_sample)

    def test_acceptance_rate(self):
        self._test_acceptance_rate(self.adapt_sample)
        self._test_acceptance_rate(self.warmstart_sample)

    def test_ess_autocorrelation(self):
        self._test_ess_autocorrelation(self.adapt_sample)
        self._test_ess_autocorrelation(self.warmstart_sample)


class TestUnderdampedCorrelatedGaussian(TestMetropolisCorrelatedGaussian):
    sampler = standard_mcmc.Underdamped(leapfrog_steps=3,
                                        friction=1.)

    def test_mean(self):
        self._test_mean(self.adapt_sample)
        self._test_mean(self.warmstart_sample)

    def test_cov(self):
        self._test_cov(self.adapt_sample)
        self._test_cov(self.warmstart_sample)

    def test_acceptance_rate(self):
        self._test_acceptance_rate(self.warmstart_sample)

    def test_ess_autocorrelation(self):
        self._test_ess_autocorrelation(self.adapt_sample)
        self._test_ess_autocorrelation(self.warmstart_sample)


if __name__ == '__main__':
    unittest.main()

