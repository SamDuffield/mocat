########################################################################################################################
# Module: tests/test_mcmc_sampling.py
# Description: Tests for MCMC sampling.
#
# Web: https://github.com/SamDuffield/mocat
########################################################################################################################

import unittest

import jax.numpy as np
from jax.random import PRNGKey
import numpy.testing as npt

from mocat.src.core import cdict
from mocat.src.scenarios import toy_scenarios
from mocat.src.mcmc.run import run_mcmc
from mocat.src.mcmc.metrics import ess, acceptance_rate
from mocat.src.mcmc import standard_mcmc, ensemble_mcmc
from mocat.src.mcmc.corrections import RMMetropolis


class TestMetropolisCorrelatedGaussian(unittest.TestCase):

    scenario_cov = np.array([[1., 0.9], [0.9, 2.]])
    scenario = toy_scenarios.Gaussian(cov=scenario_cov)
    n = int(1e5)

    sampler = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.__test__ = self.sampler is not None
        if self.__test__:
            self.adapt_sample = run_mcmc(self.scenario, self.sampler, self.n, PRNGKey(0), correction=RMMetropolis())
            self.warmstart_sample = run_mcmc(self.scenario, self.sampler, self.n, PRNGKey(0))

    def _test_mean(self,
                   sample: cdict):
        if sample.value.ndim == 3:
            val = np.concatenate(sample.value)
        else:
            val = sample.value
        samp_mean = val.mean(axis=0)
        npt.assert_array_almost_equal(samp_mean, np.zeros(2), decimal=1)

    def _test_cov(self,
                  sample: cdict):
        if sample.value.ndim == 3:
            val = np.concatenate(sample.value)
        else:
            val = sample.value
        samp_cov = np.cov(val.T)
        npt.assert_array_almost_equal(samp_cov, self.scenario_cov, decimal=1)

    def _test_ess(self,
                  sample: cdict):
        samp_ess_pot = ess(sample)
        samp_ess_0 = ess(sample, dim=0)
        samp_ess_1 = ess(sample, dim=1)

        self.assertFalse(samp_ess_pot == samp_ess_0)
        self.assertFalse(samp_ess_pot == samp_ess_1)
        self.assertFalse(samp_ess_0 == samp_ess_1)

        self.assertGreater(samp_ess_pot, self.n / 100)
        self.assertGreater(samp_ess_0, self.n / 100)
        self.assertGreater(samp_ess_1, self.n / 100)

    def _test_acceptance_rate(self,
                              sample: cdict):
        npt.assert_almost_equal(acceptance_rate(sample), self.sampler.tuning.target, decimal=1)


class TestRandomWalkCorrelatedGaussian(TestMetropolisCorrelatedGaussian):
    sampler = standard_mcmc.RandomWalk()

    def test_mean(self):
        self._test_mean(self.adapt_sample)
        self._test_mean(self.warmstart_sample)

    def test_cov(self):
        self._test_cov(self.adapt_sample)
        self._test_cov(self.warmstart_sample)

    def test_acceptance_rate(self):
        self._test_acceptance_rate(self.warmstart_sample)

    def test_ess(self):
        self._test_ess(self.adapt_sample)
        self._test_ess(self.warmstart_sample)


class TestOverdampedCorrelatedGaussian(TestMetropolisCorrelatedGaussian):
    sampler = standard_mcmc.Overdamped()

    def test_mean(self):
        self._test_mean(self.adapt_sample)
        self._test_mean(self.warmstart_sample)

    def test_cov(self):
        self._test_cov(self.adapt_sample)
        self._test_cov(self.warmstart_sample)

    def test_acceptance_rate(self):
        self._test_acceptance_rate(self.warmstart_sample)

    def test_ess(self):
        self._test_ess(self.adapt_sample)
        self._test_ess(self.warmstart_sample)


class TestHMCCorrelatedGaussian(TestMetropolisCorrelatedGaussian):
    sampler = standard_mcmc.HMC(leapfrog_steps=3)

    def test_mean(self):
        self._test_mean(self.adapt_sample)
        self._test_mean(self.warmstart_sample)

    def test_cov(self):
        self._test_cov(self.adapt_sample)
        self._test_cov(self.warmstart_sample)

    def test_acceptance_rate(self):
        self._test_acceptance_rate(self.warmstart_sample)

    def test_ess(self):
        self._test_ess(self.adapt_sample)
        self._test_ess(self.warmstart_sample)


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

    def test_ess(self):
        self._test_ess(self.adapt_sample)
        self._test_ess(self.warmstart_sample)


class TestEnsRandomWalkCorrelatedGaussian(TestMetropolisCorrelatedGaussian):
    sampler = ensemble_mcmc.EnsembleRWMH(n_ensemble=10)

    def test_mean(self):
        self._test_mean(self.adapt_sample)
        self._test_mean(self.warmstart_sample)

    def test_cov(self):
        self._test_cov(self.adapt_sample)
        self._test_cov(self.warmstart_sample)

    def test_acceptance_rate(self):
        self._test_acceptance_rate(self.warmstart_sample)

    def test_ess(self):
        self._test_ess(self.adapt_sample)
        self._test_ess(self.warmstart_sample)


class TestEnsOverdampedCorrelatedGaussian(TestMetropolisCorrelatedGaussian):
    sampler = ensemble_mcmc.EnsembleOverdamped(n_ensemble=10)

    def test_mean(self):
        self._test_mean(self.adapt_sample)
        self._test_mean(self.warmstart_sample)

    def test_cov(self):
        self._test_cov(self.adapt_sample)
        self._test_cov(self.warmstart_sample)

    def test_acceptance_rate(self):
        self._test_acceptance_rate(self.warmstart_sample)

    def test_ess(self):
        self._test_ess(self.adapt_sample)
        self._test_ess(self.warmstart_sample)


if __name__ == '__main__':
    unittest.main()

