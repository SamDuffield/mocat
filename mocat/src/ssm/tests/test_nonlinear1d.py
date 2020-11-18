########################################################################################################################
# Module: ssm/tests/test_linear_gaussian.py
# Description: Tests for 1D non-linear benchmark model.
#
# Web: https://github.com/SamDuffield/mocat
########################################################################################################################


import unittest

import jax.numpy as np
from jax import random
import numpy.testing as npt
from mocat.src.ssm.filters import BootstrapFilter, run_particle_filter_for_marginals
from mocat.src.ssm.scenarios.nonlinear1d import NonLinear1DBenchmark


class Test1DBootstrap(unittest.TestCase):

    nlssm = NonLinear1DBenchmark()
    t = np.arange(20)
    sims = nlssm.simulate(t, random.PRNGKey(0))

    bootstrap_filter = BootstrapFilter()
    n = 1000

    def test_run_for_margs(self):
        pf_samps = run_particle_filter_for_marginals(self.nlssm,
                                                     self.bootstrap_filter,
                                                     self.sims.y,
                                                     self.t,
                                                     random.PRNGKey(0),
                                                     n=self.n)

        npt.assert_array_less(self.sims.x, np.max(pf_samps.value, axis=1))
        npt.assert_array_less(np.min(pf_samps.value, axis=1), self.sims.x)


if __name__ == '__main__':
    unittest.main()
