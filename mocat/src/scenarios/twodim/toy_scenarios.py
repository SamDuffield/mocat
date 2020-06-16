########################################################################################################################
# Module: scenarios/twodim/toy_scenarios.py
# Description: Initiates some toy 2D scenarios suitable for visualisation.
#
# Web: https://github.com/SamDuffield/mocat
########################################################################################################################

from typing import Union

import jax.numpy as np

from mocat.src.scenarios.twodim.vectorise import TwoDimScenario
from mocat.src.scenarios.toy_scenarios import Gaussian as NDGaussian
from mocat.src.scenarios.toy_scenarios import GaussianMixture as NDGaussianMixture
from mocat.src.scenarios.toy_scenarios import DoubleWell as NDDoubleWell
from mocat.src.scenarios.toy_scenarios import Rastrigin as NDRastrigin


class Gaussian(TwoDimScenario, NDGaussian):
    def __init__(self,
                 mean: np.ndarray = np.array([0., 0.]),
                 cov: np.ndarray = None):
        TwoDimScenario.__init__(self)
        NDGaussian.__init__(self, 2, mean, cov)


class Banana(TwoDimScenario):
    name = 'Banana'
    curviness = 0.03
    lengthiness = 100

    def potential(self, x: np.ndarray) -> Union[float, np.ndarray]:
        return 0.5 * (x[0] ** 2 / self.lengthiness + (x[1] + self.curviness * x[0] ** 2
                                                      - self.lengthiness * self.curviness) ** 2)


class NealFunnel(TwoDimScenario):
    name = "Neal's Funnel"

    def potential(self, x: np.ndarray) -> Union[float, np.ndarray]:
        return 0.5 * (x[1] ** 2 / 3 + x[0] ** 2 * np.exp(-x[1] / 2))


class GaussianMixture(TwoDimScenario, NDGaussianMixture):
    dim = 2

    def __init__(self,
                 means: np.ndarray = np.array([[-2, -2], [2, -2], [-2, 2], [2, 2]]),
                 **kwargs):
        TwoDimScenario.__init__(self)
        NDGaussianMixture.__init__(self, means, **kwargs)


class DoubleWell(TwoDimScenario, NDDoubleWell):
    def __init__(self):
        TwoDimScenario.__init__(self)
        NDDoubleWell.__init__(self, dim=2)


class Rastrigin(TwoDimScenario, NDRastrigin):
    def __init__(self,
                 a: float = 0.5):
        TwoDimScenario.__init__(self)
        NDRastrigin.__init__(self, dim=2, a=a)


