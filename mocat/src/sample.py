########################################################################################################################
# Module: run.py
# Description: Sampler class and run function to generate samples
#
# Web: https://github.com/SamDuffield/mocat
########################################################################################################################

import copy
from typing import Tuple, Union, Type
from functools import partial
from time import time
from inspect import isclass

from jax import numpy as jnp, jit, random
from jax.lax import cond
from mocat.src.core import cdict, static_cdict, Scenario
from mocat.utils import while_loop_stacked


class Sampler:
    parameters: cdict
    name: str
    max_iter: int = 10000
    random_key_shape_per_iter: Union[jnp.ndarray, tuple, int]

    def __init__(self,
                 name: str = None,
                 **kwargs):

        if name is not None:
            self.name = name

        if not hasattr(self, 'parameters'):
            self.parameters = cdict()

        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                setattr(self.parameters, key, value)

    def __repr__(self):
        return f"mocat.Sampler.{self.__class__.__name__}"

    def deepcopy(self) -> 'Sampler':
        return copy.deepcopy(self)

    def startup(self,
                scenario: Scenario,
                n: int,
                initial_state: cdict,
                initial_extra: cdict,
                **kwargs) -> Tuple[cdict, cdict]:
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            if hasattr(self, 'parameters') and hasattr(self.parameters, key):
                setattr(self.parameters, key, value)

        if not hasattr(self, 'max_iter') \
                or not (isinstance(self.max_iter, int)
                        or (isinstance(self.max_iter, jnp.ndarray) and self.max_iter.dtype == 'int32')):
            raise AttributeError(self.__repr__() + ' max_iter must be int')

        if hasattr(self, 'parameters'):
            if not hasattr(initial_extra, 'parameters'):
                initial_extra.parameters = cdict()
            for key, value in self.parameters.__dict__.items():
                if not hasattr(initial_extra.parameters, key) or getattr(initial_extra.parameters, key) is None:
                    setattr(initial_extra.parameters, key, value)
        return initial_state, initial_extra

    def update(self,
               scenario: Scenario,
               state: cdict,
               extra: cdict) -> Tuple[cdict, cdict]:
        raise NotImplementedError(f'{self.name} update not initiated')

    def termination_criterion(self,
                              state: cdict,
                              extra: cdict) -> bool:
        return extra.iter >= self.max_iter

    def clean_chain(self,
                    scenario: Scenario,
                    chain_state: cdict):
        return chain_state

    def summary(self,
                scenario: Scenario,
                initial_state: cdict,
                initial_extra: cdict) -> cdict:
        summ = static_cdict()
        if hasattr(self, 'name'):
            summ.sampler = self.name

        if hasattr(scenario, 'name'):
            summ.scenario = scenario.name

        if hasattr(self, 'parameters'):
            summ.parameters = self.parameters

        if hasattr(self, 'tuning'):
            summ.tuning = self.tuning

        return summ


def run(scenario: Scenario,
        sampler: Union[Sampler, Type[Sampler]],
        n: int,
        random_key: Union[None, jnp.ndarray],
        initial_state: cdict = None,
        initial_extra: cdict = None,
        random_key_batchsize: int = 5000,
        **kwargs) -> Union[cdict, Tuple[cdict, jnp.ndarray]]:
    if isclass(sampler):
        sampler = sampler(**kwargs)

    sampler.n = n

    random_key_shape_per_iter = jnp.atleast_1d(sampler.random_key_shape_per_iter)
    num_random_keys_per_iter = random_key_shape_per_iter.prod()
    random_key_batchsize = jnp.maximum(num_random_keys_per_iter, random_key_batchsize)
    random_key_batchsize = jnp.minimum(random_key_batchsize, sampler.max_iter * num_random_keys_per_iter)
    random_key_batchsize = jnp.array(random_key_batchsize, dtype='int32')

    iters_per_batch = jnp.array(jnp.ceil(random_key_batchsize / num_random_keys_per_iter), dtype='int32')
    random_key_batchsize = iters_per_batch * num_random_keys_per_iter

    generate_new_rk_batch = lambda rk: random.split(rk,
                                                    random_key_batchsize
                                                    ).reshape(iters_per_batch, *random_key_shape_per_iter, 2)

    initial_random_keys = generate_new_rk_batch(random_key)

    if initial_extra is None:
        initial_extra = cdict()
    initial_extra.random_keys = initial_random_keys[0]

    initial_state, initial_extra = sampler.startup(scenario, n, initial_state, initial_extra, **kwargs)

    summary = sampler.summary(scenario, initial_state, initial_extra)

    def update_kernel(state: cdict,
                      extra_w_rks: Tuple[cdict, int, jnp.ndarray]) -> Tuple[cdict, Tuple[cdict, int, jnp.ndarray]]:
        extra, rk_iter, batch_random_keys = extra_w_rks
        rk_iter = (rk_iter + 1) % iters_per_batch

        batch_random_keys \
            = cond(rk_iter == 0,
                   generate_new_rk_batch,
                   lambda x: batch_random_keys,
                   batch_random_keys[-1, 0])

        extra.random_keys = batch_random_keys[rk_iter]
        state, extra = sampler.update(scenario, state, extra)
        return state, (extra, rk_iter, batch_random_keys)

    start = time()
    chain = while_loop_stacked(lambda state, extra_w_rks: ~sampler.termination_criterion(state, extra_w_rks[0]),
                               update_kernel,
                               (initial_state, (initial_extra, 0, initial_random_keys)),
                               sampler.max_iter)
    chain = initial_state[jnp.newaxis] + chain
    chain = sampler.clean_chain(scenario, chain)
    chain.value.block_until_ready()
    end = time()
    chain.time = end - start

    chain.summary = summary

    return chain
