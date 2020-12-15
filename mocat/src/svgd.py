########################################################################################################################
# Module: transport/svgd.py
# Description: Stein Variational Gradient Descent
#
# Web: https://github.com/SamDuffield/mocat
########################################################################################################################

from time import time
from typing import Tuple, Callable, Union

from jax import numpy as np, random, vmap
from jax.lax import scan
from jax.experimental.optimizers import adagrad, OptimizerState

from mocat.src.core import Scenario, cdict
from mocat.src.kernels import Kernel, Gaussian


def svgd_phi_hat(previous_state: cdict,
                 kernel: Kernel) -> cdict:
    n = previous_state.value.shape[0]

    def phi_hat_func(x_i):
        return vmap(lambda x_j: -kernel._call(previous_state.value[x_j], previous_state.value[x_i],
                                              **previous_state.kernel_params)
                                * previous_state.grad_potential[x_j]
                                + kernel._grad_x(previous_state.value[x_j], previous_state.value[x_i],
                                                 **previous_state.kernel_params))(np.arange(n)).mean(axis=0)

    phi_hat = vmap(phi_hat_func)(np.arange(n))
    return phi_hat


def median_bandwidth_update(state: cdict) -> cdict:
    # Note jax.numpy.median scales much worse than numpy.median,
    # thus this method is not currently recommended
    out_params = state.kernel_params
    dist_mat = vmap(lambda x: vmap(lambda y: np.sum(np.square(x - y)))(state.value))(state.value) ** 0.5
    out_params.bandwidth = np.median(dist_mat) / np.sqrt(2 * np.log(dist_mat.shape[0]))
    return out_params


def mean_bandwidth_update(state: cdict) -> cdict:
    out_params = state.kernel_params
    dist_mat = vmap(lambda x: vmap(lambda y: np.sum(np.square(x - y)))(state.value))(state.value) ** 0.5
    out_params.bandwidth = np.mean(dist_mat) / np.sqrt(2 * np.log(dist_mat.shape[0]))
    return out_params


def _run_svgd_all(scenario: Scenario,
                  n_iter: int,
                  stepsize: Union[float, Callable],
                  kernel: Kernel,
                  initial_state: cdict,
                  kernel_param_update: Callable,
                  optimiser: Callable,
                  **optim_params) -> cdict:
    opt_init, opt_update, get_params = optimiser(step_size=stepsize, **optim_params)

    grad_vec = vmap(scenario.grad_potential)
    initial_state.grad_potential = grad_vec(initial_state.value)

    initial_state.kernel_params = kernel_param_update(initial_state)

    initial_state.iter = 0

    def svgd_kernel_all(previous_carry: Tuple[cdict, OptimizerState],
                        iter_ind: int) -> Tuple[Tuple[cdict, OptimizerState], cdict]:
        previous_state, previous_opt_state = previous_carry

        phi_hat = svgd_phi_hat(previous_state, kernel)

        new_opt_state = opt_update(iter_ind, -phi_hat, previous_opt_state)

        new_state = previous_state.copy()
        new_state.value = get_params(new_opt_state)

        new_state.grad_potential = grad_vec(new_state.value)

        new_state.iter = iter_ind

        new_state.kernel_params = kernel_param_update(new_state)

        return (new_state, new_opt_state), new_state

    final_carry, chain = scan(svgd_kernel_all,
                              (initial_state, opt_init(initial_state.value)),
                              np.arange(1, n_iter + 1))
    return chain


def _run_svgd_final_only(scenario: Scenario,
                         n_iter: int,
                         stepsize: Union[float, Callable],
                         kernel: Kernel,
                         initial_state: cdict,
                         kernel_param_update: Callable,
                         optimiser: Callable,
                         **optim_params) -> cdict:
    opt_init, opt_update, get_params = optimiser(step_size=stepsize, **optim_params)

    grad_vec = vmap(scenario.grad_potential)
    initial_state.grad_potential = grad_vec(initial_state.value)

    initial_state.kernel_params = kernel_param_update(initial_state)

    initial_state.iter = 0

    def svgd_kernel_final_only(previous_carry: Tuple[cdict, OptimizerState],
                               iter_ind: int) -> Tuple[Tuple[cdict, OptimizerState], None]:
        previous_state, previous_opt_state = previous_carry

        phi_hat = svgd_phi_hat(previous_state, kernel)

        new_opt_state = opt_update(iter_ind, -phi_hat, previous_opt_state)

        new_state = previous_state.copy()
        new_state.value = get_params(new_opt_state)

        new_state.grad_potential = grad_vec(new_state.value)

        new_state.iter = iter_ind

        new_state.kernel_params = kernel_param_update(new_state)

        return (new_state, new_opt_state), None

    final_carry, chain = scan(svgd_kernel_final_only,
                              (initial_state, opt_init(initial_state.value)),
                              np.arange(1, n_iter + 1))
    return final_carry[0]


def run_svgd(scenario: Scenario,
             n_iter: int,
             stepsize: Union[float, Callable],
             kernel: Kernel = None,
             n_samps: int = None,
             initial_state: cdict = None,
             kernel_param_update: Callable = None,
             optimiser: Callable = adagrad,
             return_int_samples: bool = True,
             name: str = None,
             **optim_params) -> cdict:
    if n_samps is None and initial_state is None:
        raise ValueError('Either n_samps or initial_state required')

    if n_samps is None:
        n_samps = len(initial_state.value)

    if initial_state is None:
        initial_state = cdict(value=random.normal(random.PRNGKey(0), (n_samps, scenario.dim)))

    if kernel is None:
        kernel = Gaussian()

    update_kern_params_bool = kernel_param_update is not None
    if not update_kern_params_bool:
        kernel_param_update = lambda sample: sample.kernel_params

    if not hasattr(initial_state, 'kernel_params'):
        initial_state.kernel_params = kernel.parameters

    run_func = _run_svgd_all if return_int_samples else _run_svgd_final_only

    start = time()

    output = run_func(scenario,
                      n_iter,
                      stepsize,
                      kernel,
                      initial_state,
                      kernel_param_update,
                      optimiser,
                      **optim_params)

    end = time()
    output.value.block_until_ready()
    output.time = end - start

    if not update_kern_params_bool:
        del output.kernel_params

    if name is None:
        name = scenario.name + ": SVGD"
    output.name = name

    output.run_params = cdict(name=output.name,
                              kernel=str(kernel),
                              kernel_param_update=update_kern_params_bool,
                              stepsize=stepsize,
                              optimiser=str(optimiser),
                              optim_params=optim_params)
    return output
