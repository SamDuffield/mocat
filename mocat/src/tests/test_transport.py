########################################################################################################################
# Module: tests/test_transport.py
# Description: Tests for SVGD and SMC samplers.
#
# Web: https://github.com/SamDuffield/mocat
########################################################################################################################


import unittest

import jax.numpy as np
from jax.random import PRNGKey
import numpy.testing as npt

from mocat.src.core import CDict
from mocat.src.scenarios import toy_scenarios
from mocat.src.smc import run_smc_sampler
from mocat.src.svgd import run_svgd, median_kernel_param_update, mean_kernel_param_update
from mocat.src.mcmc.standard_mcmc import RandomWalk, Overdamped


class TestCorrelatedGaussian(unittest.TestCase):
    scenario_cov = np.array([[1., 0.9], [0.9, 2.]])
    scenario = toy_scenarios.Gaussian(cov=scenario_cov)
    n = int(1e4)

    def _test_mean(self,
                   sample: CDict):
        if sample.value.ndim == 3:
            val = np.concatenate(sample.value)
        else:
            val = sample.value
        samp_mean = val.mean(axis=0)
        npt.assert_array_almost_equal(samp_mean, np.zeros(2), decimal=1)

    def _test_cov(self,
                  sample: CDict):
        if sample.value.ndim == 3:
            val = np.concatenate(sample.value)
        else:
            val = sample.value
        samp_cov = np.cov(val.T)
        npt.assert_array_almost_equal(samp_cov, self.scenario_cov, decimal=1)


class TestSVGD(TestCorrelatedGaussian):
    n = int(1e2)
    n_iter = int(1e3)

    def test_fixed_kernel_params(self):
        sample = run_svgd(self.scenario, self.n, self.n_iter, stepsize=0.8)
        self._test_mean(sample)
        self._test_cov(sample)

    def test_median_update(self):
        sample = run_svgd(self.scenario, self.n, self.n_iter, stepsize=1.0,
                          kernel_param_update=median_kernel_param_update)
        self._test_mean(sample)
        self._test_cov(sample)

    def test_mean_update(self):
        sample = run_svgd(self.scenario, self.n, self.n_iter, stepsize=1.3,
                          kernel_param_update=mean_kernel_param_update)
        self._test_mean(sample)
        self._test_cov(sample)


class TestSMC(TestCorrelatedGaussian):

    def test_tempered_preschedule_RW(self):
        preschedule = np.arange(0., 1.1, 0.1)
        sample = run_smc_sampler(self.scenario, self.n, PRNGKey(0), RandomWalk(stepsize=1.0), preschedule=preschedule)
        self._test_mean(sample[-1])
        self._test_cov(sample[-1])
        npt.assert_array_equal(sample.weight_schedule, preschedule[1:])

    def test_tempered_preschedule_OD(self):
        preschedule = np.arange(0., 1.1, 0.1)
        sample = run_smc_sampler(self.scenario, self.n, PRNGKey(0), Overdamped(stepsize=1.0), preschedule=preschedule)
        self._test_mean(sample[-1])
        self._test_cov(sample[-1])
        npt.assert_array_equal(sample.weight_schedule, preschedule[1:])

    def test_tempered_adaptive_RW(self):
        sample = run_smc_sampler(self.scenario, self.n, PRNGKey(0), RandomWalk(stepsize=1.0))
        self._test_mean(sample[-1])
        self._test_cov(sample[-1])

    def test_tempered_adaptive_OD(self):
        sample = run_smc_sampler(self.scenario, self.n, PRNGKey(0), Overdamped(stepsize=1.0))
        self._test_mean(sample[-1])
        self._test_cov(sample[-1])


if __name__ == '__main__':
    unittest.main()
