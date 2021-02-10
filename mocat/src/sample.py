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

from jax import numpy as jnp, jit
from mocat.src.core import cdict, static_cdict, Scenario
from mocat.utils import while_loop_stacked


class Sampler:
    parameters: cdict
    name: str
    max_iter: int = 10000

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
                random_key: jnp.ndarray = None,
                initial_state: cdict = None,
                initial_extra: cdict = None,
                **kwargs) -> Tuple[cdict, cdict]:
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            if hasattr(self, 'parameters') and hasattr(self.parameters, key):
                setattr(self.parameters, key, value)

        if not hasattr(self, 'max_iter')\
            or not (isinstance(self.max_iter, int)\
                    or (isinstance(self.max_iter, jnp.ndarray) and self.max_iter.dtype == 'int32')):
            raise AttributeError(self.__repr__() + ' max_iter must be int')

        if initial_extra is None:
            initial_extra = cdict(random_key=random_key, iter=0)

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
        return extra.iter > self.max_iter

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
        **kwargs) -> Union[cdict, Tuple[cdict, jnp.ndarray]]:

    if isclass(sampler):
        sampler = sampler(**kwargs)

    sampler.n = n

    initial_state, initial_extra = sampler.startup(scenario, n, random_key, initial_state, initial_extra,
                                                   **kwargs)

    summary = sampler.summary(scenario, initial_state, initial_extra)

    transport_kernel = jit(partial(sampler.update, scenario))

    start = time()
    chain = while_loop_stacked(lambda state, extra: ~sampler.termination_criterion(state, extra),
                               transport_kernel,
                               (initial_state, initial_extra),
                               sampler.max_iter)
    chain = initial_state[jnp.newaxis] + chain
    chain = sampler.clean_chain(scenario, chain)
    chain.value.block_until_ready()
    end = time()
    chain.time = end - start

    chain.summary = summary

    return chain
