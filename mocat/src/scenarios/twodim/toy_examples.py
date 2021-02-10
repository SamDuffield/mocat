########################################################################################################################
# Module: scenarios/twodim/toy_examples.py
# Description: Initiates some toy 2D scenarios suitable for visualisation.
#
# Web: https://github.com/SamDuffield/mocat
########################################################################################################################

from typing import Union

import jax.numpy as jnp

from mocat.src.scenarios.twodim.vectorise import TwoDimToyScenario
from mocat.src.scenarios.toy_examples import Gaussian as NDGaussian
from mocat.src.scenarios.toy_examples import GaussianMixture as NDGaussianMixture
from mocat.src.scenarios.toy_examples import DoubleWell as NDDoubleWell
from mocat.src.scenarios.toy_examples import Rastrigin as NDRastrigin


class Gaussian(TwoDimToyScenario, NDGaussian):
    def __init__(self,
                 mean: jnp.ndarray = jnp.array([0., 0.]),
                 covariance: jnp.ndarray = None):
        TwoDimToyScenario.__init__(self)
        NDGaussian.__init__(self, 2, mean, covariance)


class Banana(TwoDimToyScenario):
    name = 'Banana'
    curviness = 0.03
    lengthiness = 100

    def likelihood_potential(self,
                             x: jnp.ndarray,
                             random_key: jnp.ndarray = None) -> Union[float, jnp.ndarray]:
        return 0.5 * (x[0] ** 2 / self.lengthiness + (x[1] + self.curviness * x[0] ** 2
                                                      - self.lengthiness * self.curviness) ** 2)


class NealFunnel(TwoDimToyScenario):
    name = "Neal's Funnel"

    def likelihood_potential(self,
                             x: jnp.ndarray,
                             random_key: jnp.ndarray = None) -> Union[float, jnp.ndarray]:
        return 0.5 * (x[1] ** 2 / 3 + x[0] ** 2 * jnp.exp(-x[1] / 2))


class GaussianMixture(TwoDimToyScenario, NDGaussianMixture):
    def __init__(self,
                 means: jnp.ndarray = jnp.array([[-2, -2], [2, -2], [-2, 2], [2, 2]]),
                 **kwargs):
        TwoDimToyScenario.__init__(self)
        NDGaussianMixture.__init__(self, means, **kwargs)


class DoubleWell(TwoDimToyScenario, NDDoubleWell):
    def __init__(self):
        TwoDimToyScenario.__init__(self)
        NDDoubleWell.__init__(self, dim=2)


class Rastrigin(TwoDimToyScenario, NDRastrigin):
    def __init__(self,
                 a: float = 1.):
        TwoDimToyScenario.__init__(self)
        NDRastrigin.__init__(self, dim=2, a=a)


