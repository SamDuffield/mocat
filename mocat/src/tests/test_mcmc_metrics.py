########################################################################################################################
# Module: tests/test_mcmc_metrics.py
# Description: Tests for metrics analysing run quality.
#
# Web: https://github.com/SamDuffield/mocat
########################################################################################################################

import unittest

import jax.numpy as np
from jax import random
import numpy.testing as npt

from mocat.src.mcmc import metrics
from mocat.src.core import cdict
from mocat.src import kernels


class TestIscdict(unittest.TestCase):
    arr = np.arange(20).reshape(10, 2)
    cdict = cdict(value=arr)

    def test_array(self):
        out_bool = metrics._is_cdict(self.arr)
        self.assertFalse(out_bool)

        out_bool, out_arr = metrics._is_cdict(self.arr, True)
        self.assertFalse(out_bool)
        npt.assert_array_equal(self.arr, out_arr)

    def test_cdict(self):
        out_bool = metrics._is_cdict(self.cdict)
        self.assertTrue(out_bool)

        out_bool, out_arr = metrics._is_cdict(self.cdict, True)
        self.assertTrue(out_bool)
        npt.assert_array_equal(self.arr, out_arr)


class TestAcceptanceRate(unittest.TestCase):

    all_reject_arr = np.zeros((5, 2))
    all_accept_arr = np.arange(1, 21).reshape(10, 2)
    mixed_arr = np.concatenate([all_reject_arr, all_accept_arr])

    all_accept_cdict = cdict(value=all_accept_arr)
    all_reject_cdict = cdict(value=all_reject_arr)
    mixed_cdict = cdict(value=mixed_arr)

    all_accept_cdict.alpha = np.ones(len(all_accept_arr))
    all_reject_cdict.alpha = np.zeros(len(all_reject_arr))
    mixed_cdict.alpha = np.concatenate([all_accept_cdict.alpha, all_reject_cdict.alpha])

    def test_array(self):
        accept_rate = metrics.acceptance_rate(self.all_accept_arr)
        self.assertEqual(accept_rate, 1.)

        accept_rate = metrics.acceptance_rate(self.all_reject_arr)
        self.assertEqual(accept_rate, 0.)

        accept_rate = metrics.acceptance_rate(self.mixed_arr)
        self.assertEqual(accept_rate, 10/14)

    def test_cdict(self):
        accept_rate = metrics.acceptance_rate(self.all_accept_cdict, alpha=False)
        self.assertEqual(accept_rate, 1.)

        accept_rate = metrics.acceptance_rate(self.all_reject_cdict, alpha=False)
        self.assertEqual(accept_rate, 0.)

        accept_rate = metrics.acceptance_rate(self.mixed_cdict, alpha=False)
        self.assertEqual(accept_rate, 10/14)

    def test_cdict_alpha(self):
        accept_rate = metrics.acceptance_rate(self.all_accept_cdict)
        self.assertEqual(accept_rate, 1.)

        accept_rate = metrics.acceptance_rate(self.all_reject_cdict)
        self.assertEqual(accept_rate, 0.)

        accept_rate = metrics.acceptance_rate(self.mixed_cdict)
        self.assertAlmostEqual(accept_rate, 10/15)


class TestAcceptanceRateEnsemble(unittest.TestCase):

    all_reject_arr = np.zeros((5, 3, 2))
    all_accept_arr = np.arange(1, 25).reshape(4, 3, 2)
    mixed_arr = np.concatenate([all_reject_arr, all_accept_arr])

    def test_array(self):
        accept_rate = metrics.acceptance_rate(self.all_accept_arr)
        self.assertEqual(accept_rate, 1.)

        accept_rate = metrics.acceptance_rate(self.all_reject_arr)
        self.assertEqual(accept_rate, 0.)

        accept_rate = metrics.acceptance_rate(self.mixed_arr)
        self.assertEqual(accept_rate, 4/8)


class TestAutocorrelation(unittest.TestCase):

    key = random.PRNGKey(0)
    ind_draws_arr = random.normal(key, (100,))
    ind_draws_cdict = cdict(value=ind_draws_arr)

    corr_draws_arr = 0.1 * np.cumsum(ind_draws_arr)
    corr_draws_cdict = cdict(value=corr_draws_arr)

    ind_draws_cdict_pot = cdict(value=corr_draws_arr,
                                 potential=ind_draws_arr)
    corr_draws_cdict_pot = cdict(value=ind_draws_arr,
                                 potential=corr_draws_arr)

    def test_array(self):
        ind_autocorr = metrics.autocorrelation(self.ind_draws_arr)
        self.assertEqual(ind_autocorr[0], 1.)
        npt.assert_array_equal(np.abs(ind_autocorr[1:]) < 0.3, True)

        corr_autocorr = metrics.autocorrelation(self.corr_draws_arr)
        self.assertEqual(corr_autocorr[0], 1.)
        npt.assert_array_equal(np.abs(corr_autocorr[1]) > 0.5, True)
        npt.assert_array_equal(np.abs(corr_autocorr[50]) < 0.5, True)
        npt.assert_array_equal(np.abs(corr_autocorr[-1]) < 0.3, True)

    def test_cdict(self):
        ind_autocorr = metrics.autocorrelation(self.ind_draws_cdict)
        self.assertEqual(ind_autocorr[0], 1.)
        npt.assert_array_equal(np.abs(ind_autocorr[1:]) < 0.3, True)

        corr_autocorr = metrics.autocorrelation(self.corr_draws_cdict)
        self.assertEqual(corr_autocorr[0], 1.)
        npt.assert_array_equal(np.abs(corr_autocorr[1]) > 0.5, True)
        npt.assert_array_equal(np.abs(corr_autocorr[50]) < 0.5, True)
        npt.assert_array_equal(np.abs(corr_autocorr[-1]) < 0.3, True)

    def test_cdict_pot(self):
        # Potential
        ind_autocorr = metrics.autocorrelation(self.ind_draws_cdict_pot)
        self.assertEqual(ind_autocorr[0], 1.)
        npt.assert_array_equal(np.abs(ind_autocorr[1:]) < 0.3, True)

        corr_autocorr = metrics.autocorrelation(self.corr_draws_cdict_pot)
        self.assertEqual(corr_autocorr[0], 1.)
        npt.assert_array_equal(np.abs(corr_autocorr[1]) > 0.5, True)
        npt.assert_array_equal(np.abs(corr_autocorr[50]) < 0.5, True)
        npt.assert_array_equal(np.abs(corr_autocorr[-1]) < 0.3, True)

        # Value
        ind_autocorr = metrics.autocorrelation(self.corr_draws_cdict_pot, dim=0)
        self.assertEqual(ind_autocorr[0], 1.)
        npt.assert_array_equal(np.abs(ind_autocorr[1:]) < 0.3, True)

        corr_autocorr = metrics.autocorrelation(self.ind_draws_cdict_pot, dim=0)
        self.assertEqual(corr_autocorr[0], 1.)
        npt.assert_array_equal(np.abs(corr_autocorr[1]) > 0.5, True)
        npt.assert_array_equal(np.abs(corr_autocorr[50]) < 0.5, True)
        npt.assert_array_equal(np.abs(corr_autocorr[-1]) < 0.3, True)


class testIAT(unittest.TestCase):

    key = random.PRNGKey(0)
    ind_draws_arr = random.normal(key, (100,))
    ind_draws_cdict = cdict(value=ind_draws_arr)

    corr_draws_arr = 0.1 * np.cumsum(ind_draws_arr)

    def test_array(self):
        ind_iat = metrics.integrated_autocorrelation_time(self.ind_draws_arr)
        corr_iat = metrics.integrated_autocorrelation_time(self.corr_draws_arr)

        self.assertIsInstance(ind_iat, float)
        self.assertIsInstance(corr_iat, float)
        self.assertLess(ind_iat, 2.)
        self.assertGreater(corr_iat, 8.)


class testSJD(unittest.TestCase):
    zeros_arr = np.zeros(10)
    zeros_cdict = cdict(value=zeros_arr)
    seq_arr = np.arange(10)
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

        self.assertIsInstance(ksd_n_small_a, float)
        self.assertIsInstance(ksd_n_large_a, float)
        self.assertLess(ksd_n_large_a, ksd_n_small_a)

        kernel.parameters.bandwidth = 10.
        ksd_n_small_b = metrics.ksd(self.sample_n_small, kernel)
        ksd_n_large_b = metrics.ksd(self.sample_n_large, kernel)

        self.assertIsInstance(ksd_n_small_b, float)
        self.assertIsInstance(ksd_n_large_b, float)
        self.assertLess(ksd_n_large_b, ksd_n_small_b)

    def testksd_IMQ_kernel(self):
        kernel = kernels.IMQ(bandwidth=1., c=1., beta=-0.5)
        ksd_n_small_a = metrics.ksd(self.sample_n_small, kernel)
        ksd_n_large_a = metrics.ksd(self.sample_n_large, kernel)

        self.assertIsInstance(ksd_n_small_a, float)
        self.assertIsInstance(ksd_n_large_a, float)
        self.assertLess(ksd_n_large_a, ksd_n_small_a)

        kernel.parameters.bandwidth = 10.
        kernel.parameters.c = 0.1
        kernel.parameters.beta = -0.1

        ksd_n_small_b = metrics.ksd(self.sample_n_small, kernel)
        ksd_n_large_b = metrics.ksd(self.sample_n_large, kernel)

        self.assertIsInstance(ksd_n_small_b, float)
        self.assertIsInstance(ksd_n_large_b, float)
        self.assertLess(ksd_n_large_b, ksd_n_small_b)


if __name__ == '__main__':
    unittest.main()
