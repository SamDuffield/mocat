########################################################################################################################
# Module: abc/scenarios/gk.py
# Description: G and K likelihood.
#
# Web: https://github.com/SamDuffield/mocat
########################################################################################################################

from abc import ABC as AbsBaseClass
from typing import Union

from jax import numpy as np, random, vmap
from jax.scipy.stats import norm

from mocat.src.abc.abc import ABCScenario

buffer = 1e-5


class _GK(ABCScenario):
    unsummarised_data: np.ndarray

    def single_likelihood_sample(self,
                                 x: np.ndarray,
                                 random_key: np.ndarray) -> np.ndarray:
        raise NotImplementedError(f'{self.name} single_likelihood_sample not implemented')

    def full_data_sample(self,
                         x: np.ndarray,
                         random_key: np.ndarray) -> np.ndarray:
        data_keys = random.split(random_key, self.unsummarised_data.shape[0])
        return vmap(self.single_likelihood_sample, (None, 0))(x, data_keys)

    def summarise_data(self,
                       data: np.ndarray) -> np.ndarray:
        raise NotImplementedError(f'{self.name} summarise_data not implemented')

    def likelihood_sample(self,
                          x: np.ndarray,
                          random_key: np.ndarray) -> np.ndarray:
        return self.summarise_data(self.full_data_sample(x, random_key))


class GKUniformPrior(_GK, AbsBaseClass):
    name = 'GK_old'
    dim = 4

    c = 0.8

    prior_mins = 0
    prior_maxs = 10

    def single_likelihood_sample(self,
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
        out = np.where(np.all(x > self.prior_mins), 1., np.inf)
        out = np.where(np.all(x < self.prior_maxs), out, np.inf)
        return out

    def prior_sample(self,
                     random_key: np.ndarray) -> Union[float, np.ndarray]:
        return self.prior_mins + random.uniform(random_key, (self.dim,)) * (self.prior_maxs - self.prior_mins)


class GKTransformedUniformPrior(_GK, AbsBaseClass):
    name = 'GK_old'
    dim = 4

    c = 0.8

    prior_mins = 0
    prior_maxs = 10

    def constrain(self,
                  unconstrained_x: np.ndarray):
        return self.prior_mins + norm.cdf(unconstrained_x) * (self.prior_maxs - self.prior_mins)

    def unconstrain(self,
                    constrained_x: np.ndarray):
        return norm.ppf((constrained_x - self.prior_mins) / (self.prior_maxs - self.prior_mins))

    def single_likelihood_sample(self,
                                 x: np.ndarray,
                                 random_key: np.ndarray) -> np.ndarray:
        transformed_x = self.constrain(x)
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


class GKOnlyAUniformPrior(_GK, AbsBaseClass):
    name = 'GK_old only A'
    dim = 1

    B = 1.
    g = 2.
    k = 0.5
    c = 0.8

    prior_mins = 0
    prior_maxs = 10

    def single_likelihood_sample(self,
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
        out = np.where(np.all(x > self.prior_mins), 1., np.inf)
        out = np.where(np.all(x < self.prior_maxs), out, np.inf)
        return out

    def prior_sample(self,
                     random_key: np.ndarray) -> Union[float, np.ndarray]:
        return self.prior_mins + random.uniform(random_key, (self.dim,)) * (self.prior_maxs - self.prior_mins)


class GKOnlyATransformedUniformPrior(_GK, AbsBaseClass):
    name = 'GK_old only A'
    dim = 1

    B = 1.
    g = 2.
    k = 0.5
    c = 0.8

    prior_mins = 0
    prior_maxs = 10

    def constrain(self,
                  unconstrained_x: np.ndarray):
        return self.prior_mins + norm.cdf(unconstrained_x) * (self.prior_maxs - self.prior_mins)

    def unconstrain(self,
                    constrained_x: np.ndarray):
        return norm.ppf((constrained_x - self.prior_mins) / (self.prior_maxs - self.prior_mins))

    def single_likelihood_sample(self,
                                 x: np.ndarray,
                                 random_key: np.ndarray) -> np.ndarray:
        transformed_x = self.constrain(x)
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
