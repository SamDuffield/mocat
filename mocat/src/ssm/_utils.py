########################################################################################################################
# Module: ssm/_utils.py
# Description: Useful functions for state-space models, filtering and smoothing - i.e. effective sample size.
#
# Web: https://github.com/SamDuffield/mocat
########################################################################################################################

from typing import Union

import jax.numpy as np
from jax import jit, vmap

from mocat.src.core import CDict


def ess(in_arg: Union[np.ndarray, CDict]) -> float:
    if isinstance(in_arg, CDict) and hasattr(in_arg, 'in_arg'):
        in_arg = in_arg.log_weights

    if in_arg.ndim > 1:
        in_arg = in_arg[-1]

    weights = np.exp(in_arg)
    return np.square(weights.sum()) / np.square(weights).sum()


@jit
def kalman_gain(cov_sqrt: np.ndarray,
                lik_mat: np.ndarray,
                lik_cov_sqrt: np.ndarray,
                inv_mat: np.ndarray = None) -> np.ndarray:
    if inv_mat is None:
        lik_mat_cov_sqrt = lik_mat @ cov_sqrt
        inv_mat = np.linalg.inv(lik_mat_cov_sqrt @ lik_mat_cov_sqrt.T + lik_cov_sqrt @ lik_cov_sqrt.T)
    return cov_sqrt @ cov_sqrt.T @ lik_mat.T @ inv_mat


def _gc_close(z_over_rad):
    return -0.25 * z_over_rad ** 5 \
           + 0.5 * z_over_rad ** 4 \
           + 5 / 8 * z_over_rad ** 3 \
           - 5 / 3 * z_over_rad ** 2 \
           + 1


def _gc_mid(z_over_rad):
    return 1 / 12 * z_over_rad ** 5 \
           - 0.5 * z_over_rad ** 4 \
           + 5 / 8 * z_over_rad ** 3 \
           + 5 / 3 * z_over_rad ** 2 \
           - 5 * z_over_rad \
           + 4 \
           - 2 / 3 / z_over_rad


def _gc(z, radius):
    z_over_rad = np.abs(z) / radius
    out = np.zeros_like(z)
    out = np.where(z < radius, _gc_close(z_over_rad), out)
    out = np.where((z >= radius) * (z < 2 * radius), _gc_mid(z_over_rad), out)
    return out


def gc_matrix(dim: int,
              radius: float):
    return vmap(lambda i: _gc(np.min(np.array([np.abs(np.arange(dim) - i),
                                               np.abs(np.arange(dim) + dim - i),
                                               np.abs(np.arange(dim) - dim - i)]), axis=0), radius)
                )(np.arange(dim))
