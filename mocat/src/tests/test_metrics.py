########################################################################################################################
# Module: tests/test_metrics.py
# Description: Tests for metrics analysing run quality.
#
# Web: https://github.com/SamDuffield/mocat
########################################################################################################################

import unittest

import jax.numpy as jnp
from jax import random
import numpy.testing as npt

from mocat.src.core import cdict
from mocat.src import kernels, metrics


class TestAutocorrelation(unittest.TestCase):
    key = random.PRNGKey(0)
    ind_draws_arr = random.normal(key, (100,))
    ind_draws_cdict = cdict(value=ind_draws_arr)

    corr_draws_arr = 0.1 * jnp.cumsum(ind_draws_arr)
    corr_draws_cdict = cdict(value=corr_draws_arr)

    ind_draws_cdict_pot = cdict(value=corr_draws_arr,
                                potential=ind_draws_arr)
    corr_draws_cdict_pot = cdict(value=ind_draws_arr,
                                 potential=corr_draws_arr)

    def test_array(self):
        ind_autocorr = metrics.autocorrelation(self.ind_draws_arr)
        self.assertEqual(ind_autocorr[0], 1.)
        npt.assert_array_equal(jnp.abs(ind_autocorr[1:]) < 0.3, True)

        corr_autocorr = metrics.autocorrelation(self.corr_draws_arr)
        self.assertEqual(corr_autocorr[0], 1.)
        npt.assert_array_equal(jnp.abs(corr_autocorr[1]) > 0.5, True)
        npt.assert_array_equal(jnp.abs(corr_autocorr[50]) < 0.5, True)
        npt.assert_array_equal(jnp.abs(corr_autocorr[-1]) < 0.3, True)

    def test_cdict_pot(self):
        # Potential
        ind_autocorr = metrics.autocorrelation(self.ind_draws_cdict_pot)
        self.assertEqual(ind_autocorr[0], 1.)
        npt.assert_array_equal(jnp.abs(ind_autocorr[1:]) < 0.3, True)

        corr_autocorr = metrics.autocorrelation(self.corr_draws_cdict_pot)
        self.assertEqual(corr_autocorr[0], 1.)
        npt.assert_array_equal(jnp.abs(corr_autocorr[1]) > 0.5, True)
        npt.assert_array_equal(jnp.abs(corr_autocorr[50]) < 0.5, True)
        npt.assert_array_equal(jnp.abs(corr_autocorr[-1]) < 0.3, True)


class testIAT(unittest.TestCase):
    key = random.PRNGKey(0)
    ind_draws_arr = random.normal(key, (100,))
    ind_draws_cdict = cdict(value=ind_draws_arr)

    corr_draws_arr = 0.1 * jnp.cumsum(ind_draws_arr)

    def test_array(self):
        ind_iat = metrics.integrated_autocorrelation_time(self.ind_draws_arr)
        corr_iat = metrics.integrated_autocorrelation_time(self.corr_draws_arr)
        self.assertLess(ind_iat, 2.)
        self.assertGreater(corr_iat, 8.)


class testSJD(unittest.TestCase):
    zeros_arr = jnp.zeros(10)
    zeros_cdict = cdict(value=zeros_arr)
    seq_arr = jnp.arange(10)
    seq_cdict = cdict(value=seq_arr)

    def test_array(self):
        accept_rate = metrics.squared_jumping_distance(self.zeros_arr)
        self.assertEqual(accept_rate, 0.)

        accept_rate = metrics.squared_jumping_distance(self.seq_arr)
        self.assertEqual(accept_rate, 1.)

    def test_cdict(self):
        accept_rate = metrics.squared_jumping_distance(self.zeros_cdict)
        self.assertEqual(accept_rate, 0.)

        accept_rate = metrics.squared_jumping_distance(self.seq_cdict)
        self.assertEqual(accept_rate, 1.)


class testKSDStdGaussian(unittest.TestCase):
    key = random.PRNGKey(0)
    dim = 2

    n_small = 10
    ind_draws_arr_n_small = random.normal(key, (n_small, dim))
    sample_n_small = cdict(value=ind_draws_arr_n_small, grad_potential=ind_draws_arr_n_small)

    n_large = 1000
    ind_draws_arr_n_large = random.normal(key, (n_large, dim))
    sample_n_large = cdict(value=ind_draws_arr_n_large, grad_potential=ind_draws_arr_n_large)

    def testksd_gaussian_kernel(self):
        kernel = kernels.Gaussian(bandwidth=1.)
        ksd_n_small_a = metrics.ksd(self.sample_n_small, kernel)
        ksd_n_large_a = metrics.ksd(self.sample_n_large, kernel)
        ksd_n_large_a_minibatch = metrics.ksd(self.sample_n_small, kernel, ensemble_batchsize=100, random_key=self.key)

        self.assertLess(ksd_n_large_a, ksd_n_small_a)
        npt.assert_almost_equal(ksd_n_large_a, ksd_n_large_a_minibatch, 1)

        kernel.parameters.bandwidth = 10.
        ksd_n_small_b = metrics.ksd(self.sample_n_small, kernel)
        ksd_n_large_b = metrics.ksd(self.sample_n_large, kernel)

        self.assertLess(ksd_n_large_b, ksd_n_small_b)

    def testksd_IMQ_kernel(self):
        kernel = kernels.IMQ(bandwidth=1., c=1., beta=-0.5)
        ksd_n_small_a = metrics.ksd(self.sample_n_small, kernel)
        ksd_n_large_a = metrics.ksd(self.sample_n_large, kernel)
        ksd_n_large_a_minibatch = metrics.ksd(self.sample_n_small, kernel, ensemble_batchsize=100, random_key=self.key)

        self.assertLess(ksd_n_large_a, ksd_n_small_a)
        npt.assert_almost_equal(ksd_n_large_a, ksd_n_large_a_minibatch, 1)

        kernel.parameters.bandwidth = 10.
        kernel.parameters.c = 0.1
        kernel.parameters.beta = -0.1

        ksd_n_small_b = metrics.ksd(self.sample_n_small, kernel)
        ksd_n_large_b = metrics.ksd(self.sample_n_large, kernel)

        self.assertLess(ksd_n_large_b, ksd_n_small_b)


if __name__ == '__main__':
    unittest.main()
