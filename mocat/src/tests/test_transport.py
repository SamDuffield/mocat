########################################################################################################################
# Module: tests/test_transport.py
# Description: Tests for SVGD and SMC samplers.
#
# Web: https://github.com/SamDuffield/mocat
########################################################################################################################


import unittest
from typing import Tuple

from jax import numpy as jnp, random
import numpy.testing as npt

from mocat.src.core import cdict
from mocat.src.sample import run
from mocat.src.scenarios.twodim import toy_examples
from mocat.src.transport.smc import MetropolisedSMCSampler
from mocat.src.transport.svgd import SVGD
from mocat.src.kernels import median_bandwidth_update, mean_bandwidth_update
from mocat.src.mcmc.standard_mcmc import RandomWalk, Overdamped


class TestCorrelatedGaussian(unittest.TestCase):
    scenario_cov = jnp.array([[1., 0.9], [0.9, 2.]])
    scenario = toy_examples.Gaussian(covariance=scenario_cov)
    n = int(1e4)

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


class TestSVGD(TestCorrelatedGaussian):
    n = int(1e2)
    n_iter = int(1e3)

    def test_fixed_kernel_params(self):
        class SVGD_fixed(SVGD):
            def adapt(self,
                      ensemble_state: cdict,
                      ensemble_extra: cdict) -> Tuple[cdict, cdict]:
                return ensemble_state, ensemble_extra

        sample = run(self.scenario, SVGD_fixed(max_iter=self.n_iter, stepsize=0.8),
                     n=self.n,
                     random_key=random.PRNGKey(0))
        self._test_mean(sample[-1])
        self._test_cov(sample[-1])

    def test_median_update(self):
        class SVGD_median(SVGD):
            def adapt(self,
                      ensemble_state: cdict,
                      ensemble_extra: cdict) -> Tuple[cdict, cdict]:
                ensemble_extra.parameters.kernel_params.bandwidth = median_bandwidth_update(ensemble_state.value)
                ensemble_state.kernel_params = ensemble_extra.parameters.kernel_params
                return ensemble_state, ensemble_extra

        sample = run(self.scenario, SVGD_median(max_iter=self.n_iter, stepsize=1.0),
                     n=self.n,
                     random_key=random.PRNGKey(0))

        self._test_mean(sample[-1])
        self._test_cov(sample[-1])

    def test_mean_update(self):
        class SVGD_mean(SVGD):
            def adapt(self,
                      ensemble_state: cdict,
                      ensemble_extra: cdict) -> Tuple[cdict, cdict]:
                ensemble_extra.parameters.kernel_params.bandwidth = mean_bandwidth_update(ensemble_state.value)
                ensemble_state.kernel_params = ensemble_extra.parameters.kernel_params
                return ensemble_state, ensemble_extra

        sample = run(self.scenario, SVGD_mean(max_iter=self.n_iter, stepsize=1.0),
                     n=self.n,
                     random_key=random.PRNGKey(0))

        self._test_mean(sample)
        self._test_cov(sample)

    def test_callable_stepsize(self):
        sample = run(self.scenario, SVGD(max_iter=self.n_iter, stepsize=lambda i: i ** -0.5),
                     n=self.n,
                     random_key=random.PRNGKey(0))

        self._test_mean(sample)
        self._test_cov(sample)

    def test_ensemble_minibatch(self):
        sample = run(self.scenario, SVGD(max_iter=self.n_iter, stepsize=1.0, ensemble_batchsize=100),
                     n=1000,
                     random_key=random.PRNGKey(0))

        self._test_mean(sample)
        self._test_cov(sample)


class TestMetropolisedSMC(TestCorrelatedGaussian):
    preschedule = jnp.arange(0., 1.1, 0.1)

    def resample_final(self,
                       sample: cdict) -> cdict:
        unweighted_vals = sample.value[-1, random.categorical(random.PRNGKey(1),
                                                              logits=sample.log_weight[-1], shape=(self.n,))]
        unweighted_sample = cdict(value=unweighted_vals)
        return unweighted_sample

    def test_tempered_preschedule_RW(self):
        sample = run(self.scenario, MetropolisedSMCSampler(RandomWalk(stepsize=1.0)),
                     self.n,
                     random_key=random.PRNGKey(0),
                     temperature_schedule=self.preschedule)
        unweighted_sample = self.resample_final(sample)
        self._test_mean(unweighted_sample)
        self._test_cov(unweighted_sample)
        npt.assert_array_equal(sample.temperature, self.preschedule[1:])

    def test_tempered_preschedule_OD(self):
        sample = run(self.scenario, MetropolisedSMCSampler(Overdamped(stepsize=1.0)),
                     self.n,
                     random_key=random.PRNGKey(0),
                     temperature_schedule=self.preschedule)
        unweighted_sample = self.resample_final(sample)
        self._test_mean(unweighted_sample)
        self._test_cov(unweighted_sample)
        npt.assert_array_equal(sample.temperature, self.preschedule[1:])

    def test_tempered_adaptive_RW(self):
        sample = run(self.scenario, MetropolisedSMCSampler(RandomWalk(stepsize=1.0)),
                     self.n,
                     random_key=random.PRNGKey(0))
        unweighted_sample = self.resample_final(sample)
        self._test_mean(unweighted_sample)
        self._test_cov(unweighted_sample)

    def test_tempered_adaptive_OD(self):
        sample = run(self.scenario, MetropolisedSMCSampler(Overdamped(stepsize=1.0)),
                     self.n,
                     random_key=random.PRNGKey(0))
        unweighted_sample = self.resample_final(sample)
        self._test_mean(unweighted_sample)
        self._test_cov(unweighted_sample)


if __name__ == '__main__':
    unittest.main()
