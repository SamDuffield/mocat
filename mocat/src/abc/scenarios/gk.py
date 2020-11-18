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

buffer = 1e-5


class GKUniformPrior(ABCScenario):
    name = 'GK'
    dim = 4

    c = 0.8

    prior_mins = 0
    prior_maxs = 10

    def likelihood_sample(self,
                          x: np.ndarray,
                          random_key: np.ndarray) -> np.ndarray:
        u = random.uniform(random_key, minval=buffer, maxval=1 - buffer)
        z = norm.ppf(u)
        expmingz = np.exp(-x[2] * z)
        return x[0] \
               + x[1] * (1 + self.c * (1 - expmingz) / (1 + expmingz)) \
               * z * (1 + z ** 2) ** x[3]

    def prior_potential(self,
                        x: np.ndarray) -> Union[float, np.ndarray]:
        out = np.where(x > self.prior_mins, 1 / (self.prior_maxs - self.prior_mins), 0.)
        out = np.where(x < self.prior_maxs, out, 0.)
        return out

    def prior_sample(self,
                     random_key: np.ndarray) -> Union[float, np.ndarray]:
        return self.prior_mins + random.uniform(random_key, (self.dim,)) * (self.prior_maxs - self.prior_mins)


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
        u = random.uniform(random_key, minval=buffer, maxval=1 - buffer)
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


class GKOnlyAUniformPrior(ABCScenario):
    name = 'GK only A'
    dim = 1

    B = 1.
    g = 2.
    k = 0.5
    c = 0.8

    prior_mins = 0
    prior_maxs = 10

    def likelihood_sample(self,
                          x: np.ndarray,
                          random_key: np.ndarray) -> np.ndarray:
        u = random.uniform(random_key, minval=buffer, maxval=1 - buffer)
        z = norm.ppf(u)
        expmingz = np.exp(-self.g * z)
        return x[0] \
               + self.B * (1 + self.c * (1 - expmingz) / (1 + expmingz)) \
               * z * (1 + z ** 2) ** self.c

    def prior_potential(self,
                        x: np.ndarray) -> Union[float, np.ndarray]:
        out = np.where(x > self.prior_mins, 1 / (self.prior_maxs - self.prior_mins), 0.)
        out = np.where(x < self.prior_maxs, out, 0.)
        return out

    def prior_sample(self,
                     random_key: np.ndarray) -> Union[float, np.ndarray]:
        return self.prior_mins + random.uniform(random_key, (self.dim,)) * (self.prior_maxs - self.prior_mins)


class GKOnlyATransformedUniformPrior(ABCScenario):
    name = 'GK only A'
    dim = 1

    B = 1.
    g = 2.
    k = 0.5
    c = 0.8

    prior_mins = 0
    prior_maxs = 10

    def likelihood_sample(self,
                          x: np.ndarray,
                          random_key: np.ndarray) -> np.ndarray:
        transformed_x = self.prior_mins + norm.cdf(x) * (self.prior_maxs - self.prior_mins)
        u = random.uniform(random_key, minval=buffer, maxval=1 - buffer)
        z = norm.ppf(u)
        expmingz = np.exp(-self.g * z)
        return transformed_x[0] \
               + self.B * (1 + self.c * (1 - expmingz) / (1 + expmingz)) \
               * z * (1 + z ** 2) ** self.c

    def prior_potential(self,
                        x: np.ndarray) -> Union[float, np.ndarray]:
        return 0.5 * (x ** 2).sum()

    def prior_sample(self,
                     random_key: np.ndarray) -> Union[float, np.ndarray]:
        return random.normal(random_key, (self.dim,))
