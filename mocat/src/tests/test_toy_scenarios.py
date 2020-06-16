########################################################################################################################
# Module: tests/test_toy_scenarios.py
# Description: Tests for univariate and multivariate toy scenarios
#
# Web: https://github.com/SamDuffield/mocat
########################################################################################################################


import unittest

import jax.numpy as np
import numpy.testing as npt

from mocat.src.scenarios import toy_scenarios


class Test1DGaussian(unittest.TestCase):

    scenario = toy_scenarios.Gaussian(1)

    def test_basic(self):
        npt.assert_equal(self.scenario.dim, 1)
        npt.assert_equal(self.scenario.mean.shape, (1,))
        npt.assert_equal(self.scenario.cov.shape, (1, 1))
        npt.assert_equal(self.scenario.prec.shape, (1, 1))
        npt.assert_equal(self.scenario.sqrt_prec.shape, (1, 1))

    def test_scalar_potential(self):
        npt.assert_array_equal(self.scenario.potential(0.), 0.)
        npt.assert_array_equal(self.scenario.potential(5.), 12.5)

    def test_scalar_grad_potential(self):
        npt.assert_array_equal(self.scenario.grad_potential(0.), 0.)
        npt.assert_array_equal(self.scenario.grad_potential(5.), 5.)

    def test_setcov(self):
        self.scenario.cov = np.array([[7.]])
        npt.assert_array_equal(self.scenario.prec, 1/7.)
        npt.assert_array_almost_equal(self.scenario.sqrt_prec, 0.37796447)


class TestNDGaussian(unittest.TestCase):

    scenario = toy_scenarios.Gaussian(10)

    def test_basic(self):
        npt.assert_equal(self.scenario.dim, 10)
        npt.assert_equal(self.scenario.mean.shape, (10,))
        npt.assert_equal(self.scenario.cov.shape, (10, 10))
        npt.assert_equal(self.scenario.prec.shape, (10, 10))
        npt.assert_equal(self.scenario.sqrt_prec.shape, (10, 10))

    def test_array_potential(self):
        npt.assert_array_equal(self.scenario.potential(np.zeros(10)), 0.)
        npt.assert_array_equal(self.scenario.potential(np.ones(10) * 3), 45.)

    def test_array_grad_potential(self):
        npt.assert_array_equal(self.scenario.grad_potential(np.zeros(10)), np.zeros(10))
        npt.assert_array_equal(self.scenario.grad_potential(np.ones(10) * 3.), np.ones(10) * 3.)


class Test1DGaussianMixture(unittest.TestCase):

    scenario = toy_scenarios.GaussianMixture(means=np.array([0, 1]))

    def test_basic(self):
        npt.assert_equal(self.scenario.dim, 1)
        npt.assert_equal(self.scenario.means.shape, (2, 1))
        npt.assert_equal(self.scenario.covs.shape, (2, 1, 1))
        npt.assert_equal(self.scenario.precs.shape, (2, 1, 1))
        npt.assert_equal(self.scenario.sqrt_precs.shape, (2, 1, 1))
        npt.assert_equal(self.scenario.det_precs.shape, (2,))

    def test_scalar_potential(self):
        npt.assert_array_equal(self.scenario.potential(0.), self.scenario.potential(1.))
        npt.assert_array_almost_equal(self.scenario.potential(5.), 9.601038)

    def test_scalar_grad_potential(self):
        npt.assert_array_equal(self.scenario.grad_potential(0.), -self.scenario.grad_potential(1.))
        npt.assert_array_almost_equal(self.scenario.grad_potential(5.), 4.010987)

    def test_setcovs(self):
        self.scenario.covs = np.repeat(np.array([[7.]])[np.newaxis, :, :], 2, axis=0)
        npt.assert_array_equal(self.scenario.precs, np.repeat(np.array([[1/7.]])[np.newaxis, :, :], 2, axis=0))
        npt.assert_array_almost_equal(self.scenario.sqrt_precs,
                                      np.repeat(np.array([[0.37796447]])[np.newaxis, :, :], 2, axis=0))


class TestNDGaussianMixture(unittest.TestCase):

    scenario = toy_scenarios.GaussianMixture(means=np.arange(15).reshape(5, 3))

    def test_basic(self):
        npt.assert_equal(self.scenario.dim, 3)
        npt.assert_equal(self.scenario.means.shape, (5, 3))
        npt.assert_equal(self.scenario.covs.shape, (5, 3, 3))
        npt.assert_equal(self.scenario.precs.shape, (5, 3, 3))
        npt.assert_equal(self.scenario.sqrt_precs.shape, (5, 3, 3))
        npt.assert_equal(self.scenario.det_precs.shape, (5,))

    def test_array_potential(self):
        npt.assert_array_almost_equal(self.scenario.potential(np.zeros(3)), 6.8662534)
        npt.assert_array_almost_equal(self.scenario.potential(np.ones(3) * 2), 6.8552055)

    def test_array_grad_potential(self):
        npt.assert_array_almost_equal(self.scenario.grad_potential(np.zeros(3)), np.array([0., -1, -2]))
        npt.assert_array_almost_equal(self.scenario.grad_potential(np.ones(3) * 2.), np.array([1.967039,
                                                                                               0.967039,
                                                                                               -0.032961]))


class Test1DDoubleWell(unittest.TestCase):

    scenario = toy_scenarios.DoubleWell(1)

    def test_basic(self):
        npt.assert_equal(self.scenario.dim, 1)

    def test_scalar_potential(self):
        npt.assert_array_equal(self.scenario.potential(-3.), self.scenario.potential(3.))
        npt.assert_array_equal(self.scenario.potential(0.), 0)
        npt.assert_array_almost_equal(self.scenario.potential(5.), 143.75)

    def test_scalar_grad_potential(self):
        npt.assert_array_equal(self.scenario.grad_potential(-3.), -self.scenario.grad_potential(3.))
        npt.assert_array_equal(self.scenario.grad_potential(0.), 0.)
        npt.assert_array_equal(self.scenario.grad_potential(1.), 0.)
        npt.assert_array_equal(self.scenario.grad_potential(-1.), 0.)
        npt.assert_array_almost_equal(self.scenario.grad_potential(5.), 120.)


class TestNDDoubleWell(unittest.TestCase):

    scenario = toy_scenarios.DoubleWell(6)

    def test_basic(self):
        npt.assert_equal(self.scenario.dim, 6)

    def test_array_potential(self):
        npt.assert_array_equal(self.scenario.potential(np.zeros(6)), 0.)
        npt.assert_array_equal(self.scenario.potential(np.ones(3) * 2), 6.)
        npt.assert_array_almost_equal(self.scenario.potential(np.arange(6)), 217.25)

    def test_array_grad_potential(self):
        npt.assert_array_equal(self.scenario.grad_potential(np.zeros(6)), np.zeros(6))
        npt.assert_array_equal(self.scenario.grad_potential(np.ones(6)), np.zeros(6))
        npt.assert_array_equal(self.scenario.grad_potential(-np.zeros(6)), np.zeros(6))
        npt.assert_array_almost_equal(self.scenario.grad_potential(np.arange(6, dtype='float32')),
                                      np.array([0., 0., 6., 24., 60., 120.]))


if __name__ == '__main__':
    unittest.main()
