########################################################################################################################
# Module: tests/test_core.py
# Description: Tests for core and Sampler
#
# Web: https://github.com/SamDuffield/mocat
########################################################################################################################

import unittest

import jax.numpy as np
import mocat.src.sample
import numpy.testing as npt

from mocat.src import core
from mocat.src import sample


class Testcdict(unittest.TestCase):
    cdict = core.cdict(test_arr=np.ones((10, 3)),
                       test_float=3.)

    def test_init(self):
        npt.assert_(hasattr(self.cdict, 'test_arr'))
        npt.assert_array_equal(self.cdict.test_arr, np.ones((10, 3)))

        npt.assert_(hasattr(self.cdict, 'test_float'))
        npt.assert_equal(self.cdict.test_float, 3.)

    def test_copy(self):
        cdict2 = self.cdict.copy()
        npt.assert_(isinstance(cdict2, core.cdict))

        npt.assert_(isinstance(cdict2.test_arr, np.DeviceArray))
        npt.assert_array_equal(cdict2.test_arr, np.ones((10, 3)))

        npt.assert_(isinstance(cdict2.test_float, float))
        npt.assert_equal(cdict2.test_float, 3.)

        cdict2.test_arr = np.zeros(5)
        npt.assert_array_equal(self.cdict.test_arr, np.ones((10, 3)))

        cdict2.test_float = 9.
        npt.assert_equal(self.cdict.test_float, 3.)

    def test_getitem(self):
        cdict_0get = self.cdict[0]
        npt.assert_(isinstance(cdict_0get, core.cdict))

        npt.assert_(isinstance(cdict_0get.test_arr, np.DeviceArray))
        npt.assert_array_equal(cdict_0get.test_arr, np.ones(3))

        npt.assert_(isinstance(cdict_0get.test_float, float))
        npt.assert_equal(cdict_0get.test_float, 3.)

    def test_additem(self):
        cdict_other = core.cdict(test_arr=np.ones((2, 3)),
                                 test_float=7.,
                                 time=25.)

        self.cdict.time = 10.

        cdict_add = self.cdict + cdict_other

        npt.assert_(isinstance(cdict_add, core.cdict))

        npt.assert_(isinstance(cdict_add.test_arr, np.DeviceArray))
        npt.assert_array_equal(cdict_add.test_arr, np.ones((12, 3)))
        npt.assert_array_equal(cdict_add.time, 35.)

        npt.assert_(isinstance(cdict_add.test_float, float))
        npt.assert_equal(cdict_add.test_float, 3.)

        npt.assert_array_equal(self.cdict.test_arr, np.ones((10, 3)))
        npt.assert_equal(self.cdict.test_float, 3.)
        npt.assert_equal(self.cdict.time, 10.)
        del self.cdict.time


class TestSampler(unittest.TestCase):
    sampler = sample.Sampler(name='test', other=np.zeros(2))

    def test_init(self):
        npt.assert_equal(self.sampler.name, 'test')
        npt.assert_(hasattr(self.sampler, 'parameters'))
        npt.assert_array_equal(self.sampler.parameters.other, np.zeros(2))

    def test_copy(self):
        sampler2 = self.sampler.deepcopy()

        npt.assert_(isinstance(sampler2, sample.Sampler))

        sampler2.name = 'other'
        npt.assert_equal(self.sampler.name, 'test')

        sampler2.parameters.other = 10.
        npt.assert_array_equal(self.sampler.parameters.other, np.zeros(2))


if __name__ == '__main__':
    unittest.main()
