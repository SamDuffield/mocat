########################################################################################################################
# Module: tests/test_toy_scenarios.py
# Description: Tests for bivariate toy scenarios
#
# Web: https://github.com/SamDuffield/mocat
########################################################################################################################


import unittest

import jax.numpy as np
import numpy.testing as npt

from mocat.src.scenarios.twodim import toy_examples


class TestGaussian(unittest.TestCase):
    scenario = toy_examples.Gaussian()

    def test_basic(self):
        npt.assert_equal(self.scenario.dim, 2)
        npt.assert_array_equal(self.scenario.mean, np.zeros(2))
        npt.assert_array_equal(self.scenario.covariance, np.eye(2))

    def test_vectorise(self):
        self.scenario._vectorise()
        npt.assert_array_equal(self.scenario.vec_potential(np.arange(6).reshape((3, 2))), np.array([0.5, 6.5, 20.5]))
        npt.assert_array_equal(self.scenario.vec_potential(np.arange(8).reshape((2, 2, 2))),
                               np.array([[0.5, 6.5],
                                         [20.5, 42.5]]))

    def test_auto_axes_lims(self):
        self.scenario.auto_axes_lims()
        npt.assert_almost_equal(-self.scenario.xlim[0], self.scenario.xlim[1], decimal=4)
        npt.assert_almost_equal(self.scenario.xlim[0], self.scenario.ylim[0], decimal=4)
        npt.assert_almost_equal(self.scenario.xlim[1], self.scenario.ylim[1], decimal=4)

        npt.assert_(self.scenario.xlim[1] > 1)
        npt.assert_(self.scenario.xlim[1] < 5)

        self.scenario.covariance = np.array([[10., 9.], [9., 50.]])
        self.scenario.auto_axes_lims()

        npt.assert_almost_equal(-self.scenario.xlim[0], self.scenario.xlim[1], decimal=4)
        npt.assert_almost_equal(-self.scenario.ylim[0], self.scenario.ylim[1], decimal=4)

        npt.assert_(self.scenario.xlim[1] > 5)
        npt.assert_(self.scenario.xlim[1] < 15)

        npt.assert_(self.scenario.ylim[1] > 10)
        npt.assert_(self.scenario.ylim[1] < 20)

        self.scenario.covariance = np.eye(2)


class TestBanana(unittest.TestCase):
    scenario = toy_examples.Banana()
    scenario.curviness = 0.03
    scenario.lengthiness = 100

    def test_basic(self):
        npt.assert_equal(self.scenario.dim, 2)

    def test_potential(self):
        npt.assert_array_equal(self.scenario.potential(np.zeros(2)), 4.5)
        npt.assert_array_equal(self.scenario.potential(np.arange(2)), 2.)

    def test_grad_potential(self):
        npt.assert_array_equal(self.scenario.grad_potential(np.zeros(2)), [0., -3.])
        npt.assert_array_equal(self.scenario.grad_potential(np.arange(2, dtype='float32')), [0., -2.])

    def test_vec_potential(self):
        self.scenario._vectorise()
        npt.assert_array_almost_equal(self.scenario.vec_potential(np.arange(6).reshape((3, 2))),
                                      np.array([2., 0.0272, 3.1552]))
        npt.assert_array_almost_equal(self.scenario.vec_potential(np.arange(8).reshape((2, 2, 2))),
                                      np.array([[2., 0.0272],
                                                [3.1552, 13.0831995]]))

    def test_auto_axes_lims(self):
        self.scenario.auto_axes_lims()
        npt.assert_almost_equal(-self.scenario.xlim[0], self.scenario.xlim[1], decimal=4)
        npt.assert_(abs(self.scenario.ylim[0]) > self.scenario.ylim[1])

        npt.assert_(self.scenario.xlim[1] > 15)
        npt.assert_(self.scenario.xlim[1] < 30)

        npt.assert_(self.scenario.ylim[0] < -5)
        npt.assert_(self.scenario.ylim[0] > -20)

        npt.assert_(self.scenario.ylim[1] > 1)
        npt.assert_(self.scenario.ylim[1] < 15)


class TestNealFunnel(unittest.TestCase):
    scenario = toy_examples.NealFunnel()

    def test_basic(self):
        npt.assert_equal(self.scenario.dim, 2)

    def test_potential(self):
        npt.assert_array_equal(self.scenario.potential(np.zeros(2)), 0.)
        npt.assert_array_almost_equal(self.scenario.potential(np.arange(2)), 0.166667)

    def test_grad_potential(self):
        npt.assert_array_almost_equal(self.scenario.grad_potential(np.zeros(2)), np.zeros(2))
        npt.assert_array_almost_equal(self.scenario.grad_potential(np.arange(2, dtype='float32')),
                                      np.array([0., 0.333333]))

    def test_vec_potential(self):
        self.scenario._vectorise()
        npt.assert_array_almost_equal(self.scenario.vec_potential(np.arange(6).reshape((3, 2))),
                                      np.array([0.166667, 1.946260, 4.823346]))
        npt.assert_array_almost_equal(self.scenario.vec_potential(np.arange(8).reshape((2, 2, 2))),
                                      np.array([[0.166666, 1.946260],
                                                [4.823346, 8.71022]]))

    def test_auto_axes_lims(self):
        self.scenario.auto_axes_lims()
        npt.assert_almost_equal(-self.scenario.xlim[0], self.scenario.xlim[1], decimal=4)
        npt.assert_almost_equal(-self.scenario.ylim[0], self.scenario.ylim[1], decimal=4)

        npt.assert_(self.scenario.xlim[1] > 1)
        npt.assert_(self.scenario.xlim[1] < 10)

        npt.assert_(self.scenario.ylim[1] > 1)
        npt.assert_(self.scenario.ylim[1] < 20)


class TestGaussianMixture(unittest.TestCase):
    scenario = toy_examples.GaussianMixture()

    def test_basic(self):
        npt.assert_equal(self.scenario.dim, 2)
        npt.assert_equal(self.scenario.means.ndim, 2)
        npt.assert_equal(self.scenario.means.shape[-1], 2)
        npt.assert_equal(self.scenario.covariances.ndim, 3)
        npt.assert_equal(self.scenario.covariances.shape[-1], 2)
        npt.assert_equal(self.scenario.precisions.ndim, 3)
        npt.assert_equal(self.scenario.precisions.shape[-1], 2)
        npt.assert_equal(self.scenario.precision_sqrts.ndim, 3)
        npt.assert_equal(self.scenario.precision_sqrts.shape[-1], 2)

    def test_vectorise(self):
        self.scenario._vectorise()
        npt.assert_array_almost_equal(self.scenario.vec_potential(np.arange(6).reshape((3, 2))),
                                      np.array([5.0128746, 3.7238297, 9.724172]))
        npt.assert_array_almost_equal(self.scenario.vec_potential(np.arange(8).reshape((2, 2, 2))),
                                      np.array([[5.0128746, 3.7238297],
                                                [9.724172, 23.72417]]))

    def test_auto_axes_lims(self):
        self.scenario.auto_axes_lims()
        npt.assert_almost_equal(-self.scenario.xlim[0], self.scenario.xlim[1], decimal=4)
        npt.assert_almost_equal(self.scenario.xlim[0], self.scenario.ylim[0], decimal=4)
        npt.assert_almost_equal(self.scenario.xlim[1], self.scenario.ylim[1], decimal=4)

        npt.assert_(self.scenario.xlim[1] > 1)
        npt.assert_(self.scenario.xlim[1] < 10)

        self.scenario.covariances = np.array([[10., 9.], [9., 30.]])
        self.test_basic()

        self.scenario.auto_axes_lims()

        npt.assert_almost_equal(-self.scenario.xlim[0], self.scenario.xlim[1], decimal=4)
        npt.assert_almost_equal(-self.scenario.ylim[0], self.scenario.ylim[1], decimal=4)

        npt.assert_(self.scenario.xlim[1] > 7)
        npt.assert_(self.scenario.xlim[1] < 15)

        npt.assert_(self.scenario.ylim[1] > 11)
        npt.assert_(self.scenario.ylim[1] < 16)

        self.scenario.covariances = np.eye(2)


class TestDoubleWell(unittest.TestCase):
    scenario = toy_examples.DoubleWell()

    def test_basic(self):
        npt.assert_equal(self.scenario.dim, 2)

    def test_vectorise(self):
        self.scenario._vectorise()
        npt.assert_array_almost_equal(self.scenario.vec_potential(np.arange(6).reshape((3, 2))),
                                      np.array([-0.25, 17.75, 199.75]))
        npt.assert_array_almost_equal(self.scenario.vec_potential(np.arange(8).reshape((2, 2, 2))),
                                      np.array([[-0.25, 17.75],
                                                [199.75, 881.75]]))

    def test_auto_axes_lims(self):
        self.scenario.auto_axes_lims()
        npt.assert_almost_equal(-self.scenario.xlim[0], self.scenario.xlim[1], decimal=4)
        npt.assert_almost_equal(self.scenario.xlim[0], self.scenario.ylim[0], decimal=4)
        npt.assert_almost_equal(self.scenario.xlim[1], self.scenario.ylim[1], decimal=4)

        npt.assert_(self.scenario.xlim[1] > 1)
        npt.assert_(self.scenario.xlim[1] < 5)


if __name__ == '__main__':
    unittest.main()
