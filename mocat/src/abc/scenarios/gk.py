########################################################################################################################
# Module: abc/scenarios/gk.py
# Description: G and K likelihood.
#
# Web: https://github.com/SamDuffield/mocat
########################################################################################################################

from typing import Union

from jax import numpy as np, random
from jax.scipy.stats import norm

from mocat.src.abc.abc import ABCScenario


class GKTransformedUniformPrior(ABCScenario):
    name = 'GK'
    dim = 4

    c = 0.8

    prior_mins = 0
    prior_maxs = 10

    def likelihood_sample(self,
                          x: np.ndarray,
                          random_key: np.ndarray) -> np.ndarray:
        transformed_x = self.prior_mins + norm.cdf(x) * (self.prior_maxs - self.prior_mins)
        u = random.uniform(random_key)
        z = norm.ppf(u)
        expmingz = np.exp(-transformed_x[2] * z)
        return transformed_x[0] \
               + transformed_x[1] * (1 + self.c * (1 - expmingz) / (1 + expmingz)) \
               * z * (1 + z ** 2) ** transformed_x[3]

    def prior_potential(self,
                        x: np.ndarray) -> Union[float, np.ndarray]:
        return 0.5 * (x ** 2).sum()

    def prior_sample(self,
                     random_key: np.ndarray) -> Union[float, np.ndarray]:
        return random.normal(random_key, (self.dim,))
