########################################################################################################################
# Module: transport/svgd.py
# Description: Stein Variational Gradient Descent.
#
# Web: https://github.com/SamDuffield/mocat
########################################################################################################################

from typing import Tuple, Union, Callable

from jax import numpy as jnp, vmap, random
from jax.example_libraries.optimizers import adagrad

from mocat.src.core import Scenario, cdict, impl_checkable, is_implemented
from mocat.src.transport.sampler import TransportSampler
from mocat.src.kernels import Kernel, Gaussian, mean_bandwidth_update


def kernelised_grad_matrix(vals: jnp.ndarray,
                           grads: jnp.ndarray,
                           kernel: Kernel,
                           kernel_params: cdict,
                           batch_inds_mat: jnp.ndarray) -> jnp.ndarray:
    n = vals.shape[0]

    def phi_hat_func(x_i):
        return vmap(lambda x_j: -kernel._call(vals[x_j], vals[x_i], **kernel_params)
                                * grads[x_j]
                                + kernel._grad_x(vals[x_j], vals[x_i], **kernel_params)
                    )(batch_inds_mat[x_i]).mean(axis=0)

    phi_hat = vmap(phi_hat_func)(jnp.arange(n))
    return phi_hat


def adapt_bandwidth_mean(ensemble_state: cdict,
                         extra: cdict) -> Tuple[cdict, cdict]:
    extra.parameters.kernel_params.bandwidth = mean_bandwidth_update(ensemble_state.value)
    ensemble_state.kernel_params = extra.parameters.kernel_params
    return ensemble_state, extra


class SVGD(TransportSampler):
    name = 'SVGD'

    opt_init: Callable
    opt_update: Callable
    get_params: Callable
    get_batch_inds: Callable

    def __init__(self,
                 stepsize: Union[float, Callable],
                 max_iter: int = 1000,
                 kernel: Kernel = None,
                 kernel_params: cdict = None,
                 ensemble_batchsize: int = None,
                 optimiser: Callable = adagrad,
                 **optim_params):
        super().__init__(max_iter=max_iter)
        if kernel is None:
            kernel = Gaussian()
            if not is_implemented(self.adapt):
                self.adapt = adapt_bandwidth_mean

        if kernel_params is None:
            kernel_params = kernel.parameters

        self.parameters.stepsize = stepsize

        self.kernel = kernel
        self.parameters.kernel_params = kernel_params

        self.parameters.ensemble_batchsize = ensemble_batchsize

        self.optimiser = optimiser
        self.parameters.optim_params = optim_params

    def startup(self,
                scenario: Scenario,
                n: int,
                initial_state: cdict,
                initial_extra: cdict,
                **kwargs) -> Tuple[cdict, cdict]:
        initial_state, initial_extra = super().startup(scenario, n, initial_state, initial_extra, **kwargs)

        if self.parameters.ensemble_batchsize is None:
            self.parameters.ensemble_batchsize = n
            initial_extra.parameters.ensemble_batchsize = n

        if self.parameters.ensemble_batchsize == n:
            self.get_batch_inds = lambda _: jnp.repeat(jnp.arange(n)[None], n, axis=0)
        else:
            self.get_batch_inds = lambda rk: random.choice(rk, n, shape=(n, self.parameters.ensemble_batchsize,))

        del initial_extra.parameters.stepsize

        random_keys = random.split(initial_extra.random_key, n + 1)
        initial_extra.random_key = random_keys[-1]

        initial_state.potential, initial_state.grad_potential = vmap(scenario.potential_and_grad)(initial_state.value,
                                                                                                  random_keys[:n])

        initial_state, initial_extra = self.adapt(initial_state, initial_extra)

        self.opt_init, self.opt_update, self.get_params = self.optimiser(step_size=self.parameters.stepsize,
                                                                         **initial_extra.parameters.optim_params)
        initial_extra.opt_state = self.opt_init(initial_state.value)
        return initial_state, initial_extra

    @impl_checkable
    def adapt(self,
              ensemble_state: cdict,
              extra: cdict) -> Tuple[cdict, cdict]:
        return ensemble_state, extra

    def kernelised_grad_matrix(self,
                               vals: jnp.ndarray,
                               grads: jnp.ndarray,
                               kernel_params: cdict,
                               batch_inds_mat: jnp.ndarray) -> jnp.ndarray:
        return kernelised_grad_matrix(vals, grads, self.kernel, kernel_params, batch_inds_mat)

    def update(self,
               scenario: Scenario,
               ensemble_state: cdict,
               extra: cdict) -> Tuple[cdict, cdict]:
        n = ensemble_state.value.shape[0]
        extra.iter = extra.iter + 1

        random_keys = random.split(extra.random_key, n + 2)
        batch_inds = self.get_batch_inds(random_keys[-1])
        extra.random_key = random_keys[-2]

        phi_hat = self.kernelised_grad_matrix(ensemble_state.value,
                                              ensemble_state.grad_potential,
                                              extra.parameters.kernel_params,
                                              batch_inds)

        extra.opt_state = self.opt_update(extra.iter, -phi_hat, extra.opt_state)
        ensemble_state.value = self.get_params(extra.opt_state)

        ensemble_state.potential, ensemble_state.grad_potential \
            = vmap(scenario.potential_and_grad)(ensemble_state.value, random_keys[:n])

        ensemble_state, extra = self.adapt(ensemble_state, extra)

        return ensemble_state, extra
