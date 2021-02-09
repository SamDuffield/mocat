########################################################################################################################
# Module: ssm/_utils.py
# Description: Useful functions for state-space models, filtering and smoothing - i.e. effective run size.
#
# Web: https://github.com/SamDuffield/mocat
########################################################################################################################

from typing import Union

import jax.numpy as np
from jax import jit

from mocat.src.core import cdict


def ess(in_arg: Union[np.ndarray, cdict]) -> float:
    if isinstance(in_arg, cdict) and hasattr(in_arg, 'in_arg'):
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


