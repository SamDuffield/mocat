########################################################################################################################
# Module: tests/test_transport.py
# Description: Tests for SVGD and SMC samplers.
#
# Web: https://github.com/SamDuffield/mocat
########################################################################################################################


import unittest
from typing import Tuple

from jax import numpy as jnp, random, vmap
import numpy.testing as npt

from mocat.src.core import cdict
from mocat.src.sample import run
from mocat.src.scenarios.twodim import toy_examples
from mocat.src.transport.smc import MetropolisedSMCSampler
from mocat.src.transport.svgd import SVGD
from mocat.src.kernels import median_bandwidth_update, mean_bandwidth_update
from mocat.src.mcmc.standard_mcmc import RandomWalk, Underdamped


class TestCorrelatedGaussian(unittest.TestCase):
    n: int
    scenario = toy_examples.Gaussian(covariance=jnp.array([[1., 0.9], [0.9, 2.]]))
    scenario.prior_sample = lambda rk: (random.normal(rk, shape=(2,)) * 7.)
    scenario.prior_potential = lambda x, rk: 0.5 * jnp.square(x / 7. ** 2).sum(-1)
    posterior_covariance = jnp.linalg.inv(jnp.linalg.inv(scenario.covariance) + 1/7**2 * jnp.eye(2))

    def _test_mean(self,
                   sample: cdict):
        if sample.value.ndim == 3:
            val = jnp.concatenate(sample.value)
        else:
            val = sample.value
        samp_mean = val.mean(axis=0)
        npt.assert_array_almost_equal(samp_mean, jnp.zeros(2), decimal=0)

    def _test_cov(self,
                  sample: cdict):
        if sample.value.ndim == 3:
            val = jnp.concatenate(sample.value)
        else:
            val = sample.value
        samp_cov = jnp.cov(val.T)
        npt.assert_array_almost_equal(samp_cov, self.posterior_covariance, decimal=1)

    def _test_log_norm_const(self,
                             sample):
        lik_prec = jnp.linalg.inv(self.scenario.covariance)
        tempered_covs = vmap(lambda t: jnp.linalg.inv(lik_prec * t + jnp.eye(2) / 7 ** 2))(sample.temperature)
        tempered_dets = vmap(jnp.linalg.det)(tempered_covs)

        npt.assert_array_almost_equal(sample.log_norm_constant, 0.5 * (jnp.log(tempered_dets) - 4 * jnp.log(7)), 0)

    def resample_final(self,
                       sample: cdict) -> cdict:
        unweighted_vals = sample.value[-1, random.categorical(random.PRNGKey(1),
                                                              logits=sample.log_weight[-1], shape=(self.n,))]
        unweighted_sample = cdict(value=unweighted_vals)
        return unweighted_sample


class TestSVGD(TestCorrelatedGaussian):
    n = int(1e2)
    n_iter = int(1e3)

    def test_fixed_kernel_params(self):
        sample = run(self.scenario, SVGD(max_iter=self.n_iter, stepsize=0.8),
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
        self._test_mean(sample[-1])
        self._test_cov(sample[-1])

    def test_callable_stepsize(self):
        sample = run(self.scenario, SVGD(max_iter=self.n_iter, stepsize=lambda i: 10 * i ** -0.5),
                     n=self.n,
                     random_key=random.PRNGKey(0))
        self._test_mean(sample[-1])
        self._test_cov(sample[-1])

    def test_ensemble_minibatch(self):
        sample = run(self.scenario, SVGD(max_iter=self.n_iter, stepsize=1.0, ensemble_batchsize=100),
                     n=self.n,
                     random_key=random.PRNGKey(0))
        self._test_mean(sample[-1])
        self._test_cov(sample[-1])


class TestMetropolisedSMC(TestCorrelatedGaussian):
    preschedule = jnp.arange(0., 1.1, 0.1)
    n = int(1e4)

    def test_tempered_preschedule_RW(self):
        sample = run(self.scenario, MetropolisedSMCSampler(RandomWalk(stepsize=1.0)),
                     self.n,
                     random_key=random.PRNGKey(0),
                     temperature_schedule=self.preschedule)
        unweighted_sample = self.resample_final(sample)
        self._test_mean(unweighted_sample)
        self._test_cov(unweighted_sample)
        self._test_log_norm_const(sample)
        npt.assert_array_equal(sample.temperature, self.preschedule[1:])

    def test_tempered_preschedule_UD(self):
        sample = run(self.scenario, MetropolisedSMCSampler(Underdamped(stepsize=1.0)),
                     self.n,
                     random_key=random.PRNGKey(0),
                     temperature_schedule=self.preschedule)
        unweighted_sample = self.resample_final(sample)
        self._test_mean(unweighted_sample)
        self._test_cov(unweighted_sample)
        self._test_log_norm_const(sample)
        npt.assert_array_equal(sample.temperature, self.preschedule[1:])

    def test_tempered_adaptive_RW(self):
        sample = run(self.scenario, MetropolisedSMCSampler(RandomWalk(stepsize=1.0)),
                     self.n,
                     random_key=random.PRNGKey(0))
        unweighted_sample = self.resample_final(sample)
        self._test_mean(unweighted_sample)
        self._test_cov(unweighted_sample)
        self._test_log_norm_const(sample)

    def test_tempered_adaptive_UD(self):
        sample = run(self.scenario, MetropolisedSMCSampler(Underdamped(stepsize=1.0, leapfrog_steps=10)),
                     self.n,
                     random_key=random.PRNGKey(0))
        unweighted_sample = self.resample_final(sample)
        self._test_mean(unweighted_sample)
        self._test_cov(unweighted_sample)
        self._test_log_norm_const(sample)


if __name__ == '__main__':
    unittest.main()
