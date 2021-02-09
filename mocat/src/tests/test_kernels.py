########################################################################################################################
# Module: tests/test_kernels.py
# Description: Tests for kernels
#
# Web: https://github.com/SamDuffield/mocat
########################################################################################################################

import unittest

import jax.numpy as np
import numpy.testing as npt

from mocat.src import kernels


class TestGaussianKernel(unittest.TestCase):
    kernel = kernels.Gaussian()

    def test_call(self):
        npt.assert_array_almost_equal(self.kernel(np.zeros(5), np.zeros(5)), 1.)
        npt.assert_array_almost_equal(self.kernel(np.zeros(5), np.ones(5)), 0.082085006)

    def test_grad_x(self):
        npt.assert_array_almost_equal(self.kernel.grad_x(np.zeros(5), np.zeros(5)), np.zeros(5))
        npt.assert_array_almost_equal(self.kernel.grad_x(np.zeros(5), np.ones(5)), np.ones(5) * 0.082085006)

    def test_grad_y(self):
        npt.assert_array_almost_equal(self.kernel.grad_y(np.zeros(5), np.zeros(5)), np.zeros(5))
        npt.assert_array_almost_equal(self.kernel.grad_y(np.zeros(5), np.ones(5)), np.ones(5) * -0.082085006)


class TestIMQKernel(unittest.TestCase):
    kernel = kernels.IMQ()

    def test_call(self):
        npt.assert_array_almost_equal(self.kernel(np.zeros(5), np.zeros(5)), 1.)
        npt.assert_array_almost_equal(self.kernel(np.zeros(5), np.ones(5)), 0.5345225)

    def test_grad_x(self):
        npt.assert_array_almost_equal(self.kernel.grad_x(np.zeros(5), np.zeros(5)), np.zeros(5))
        npt.assert_array_almost_equal(self.kernel.grad_x(np.zeros(5), np.ones(5)), np.ones(5) * 0.07636035)

    def test_grad_y(self):
        npt.assert_array_almost_equal(self.kernel.grad_y(np.zeros(5), np.zeros(5)), np.zeros(5))
        npt.assert_array_almost_equal(self.kernel.grad_y(np.zeros(5), np.ones(5)), np.ones(5) * -0.07636035)


if __name__ == '__main__':
    unittest.main()
