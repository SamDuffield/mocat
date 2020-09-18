########################################################################################################################
# Module: transport/svgd.py
# Description: Stein Variational Gradient Descent
#
# Web: https://github.com/SamDuffield/mocat
########################################################################################################################

from time import time
from typing import Tuple, Callable
from functools import partial

from jax import numpy as np, random, vmap
from jax.lax import scan

from mocat.src.core import Scenario, CDict
from mocat.src.kernels import Kernel, Gaussian


def svgd_update(previous_state: CDict,
                kernel: Kernel) -> CDict:
    new_state = previous_state.copy()
    n = previous_state.value.shape[0]

    def phi_hat_func(x_i):
        return vmap(lambda x_j: -kernel._call(previous_state.value[x_j], previous_state.value[x_i],
                                              **previous_state.kernel_params)
                                * previous_state.grad_potential[x_j]
                                + kernel._grad_x(previous_state.value[x_j], previous_state.value[x_i],
                                                 **previous_state.kernel_params))(np.arange(n)).mean(axis=0)

    phi_hat = vmap(phi_hat_func)(np.arange(n))
    new_state.value = new_state.value + new_state.stepsize * phi_hat
    return new_state


def median_kernel_param_update(state: CDict) -> CDict:
    # Note jax.numpy.median scales much worse than numpy.median,
    # so this method is not currently recommended
    out_params = state.kernel_params
    dist_mat = vmap(lambda x: vmap(lambda y: np.sum(np.square(x-y)))(state.value))(state.value)**0.5
    out_params.bandwidth = np.median(dist_mat) ** 2 / np.log(dist_mat.shape[0])
    return out_params


def mean_kernel_param_update(state: CDict) -> CDict:
    out_params = state.kernel_params
    dist_mat = vmap(lambda x: vmap(lambda y: np.sum(np.square(x-y)))(state.value))(state.value)**0.5
    out_params.bandwidth = np.mean(dist_mat) ** 2 / np.log(dist_mat.shape[0])
    return out_params


def run_svgd(scenario: Scenario,
             n_samps: int,
             n_iter: int,
             stepsize: float = None,
             kernel: Kernel = None,
             initial_state: CDict = None,
             kernel_params: CDict = None,
             kernel_param_update: Callable = None,
             stepsize_update: Callable = None,
             return_int_samples: bool = False,
             name: str = None) -> CDict:
    if initial_state is None:
        initial_state = CDict(value=random.normal(random.PRNGKey(0), (n_samps, scenario.dim)))

    grad_vec = vmap(scenario.grad_potential)

    if not hasattr(initial_state, 'grad_potential'):
        initial_state.grad_potential = grad_vec(initial_state.value)

    if kernel is None:
        kernel = Gaussian()

    if kernel_params is None:
        kernel_params = kernel.parameters.copy()

    update_kern_params_bool = kernel_param_update is not None
    if not update_kern_params_bool:
        kernel_param_update = lambda sample: sample.kernel_params

    if not hasattr(initial_state, 'kernel_params'):
        initial_state.kernel_params = kernel_params

    initial_state.kernel_params = kernel_param_update(initial_state)

    update_stepsize_bool = stepsize_update is not None
    if not update_stepsize_bool:
        if stepsize is None:
            raise ValueError('No stepsize or stepsize_update found for SVGD')
        stepsize_update = lambda sample: sample.stepsize

    if not hasattr(initial_state, 'stepsize'):
        initial_state.stepsize = stepsize

    initial_state.stepsize = stepsize_update(initial_state)

    initial_state.iter = 0

    def svgd_kernel(previous_state: CDict,
                    iter_ind: int) -> CDict:

        new_state = svgd_update(previous_state, kernel)

        new_state.grad_potential = grad_vec(new_state.value)

        new_state.iter = iter_ind

        new_state.kernel_params = kernel_param_update(new_state)

        new_state.stepsize = stepsize_update(new_state)

        return new_state

    if return_int_samples:
        def svgd_kernel_ris(previous_state: CDict,
                            iter_ind: int) -> Tuple[CDict, CDict]:
            new_state = svgd_kernel(previous_state, iter_ind)
            return new_state, new_state
    else:
        def svgd_kernel_ris(previous_state: CDict,
                            iter_ind: int) -> Tuple[CDict, None]:
            new_state = svgd_kernel(previous_state, iter_ind)
            return new_state, None

    start = time()

    final_carry, chain = scan(svgd_kernel_ris,
                              initial_state,
                              np.arange(1, n_iter + 1))

    output = chain if return_int_samples else final_carry

    end = time()
    output.value.block_until_ready()
    output.time = end - start

    if return_int_samples:
        del output.iter

    if not update_kern_params_bool:
        del output.kernel_params

    if not update_stepsize_bool:
        del output.stepsize

    if name is None:
        name = scenario.name + ": SVGD"
    output.name = name

    output.run_params = CDict(name=output.name,
                              kernel=str(kernel),
                              kernel_param_update=update_kern_params_bool,
                              stepsize_update=update_stepsize_bool)
    return output
