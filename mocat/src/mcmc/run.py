########################################################################################################################
# Module: mcmc/run.py
# Description: Run Markov chains!
#
# Web: https://github.com/SamDuffield/mocat
########################################################################################################################

from time import time
from typing import Union, Tuple, Type, Any
from inspect import isclass

from jax import numpy as np, jit
from jax.lax import scan

from mocat.src.core import Scenario, CDict
from mocat.src.mcmc.corrections import Correction, Uncorrected
from mocat.src.mcmc.sampler import MCMCSampler


def startup_mcmc(scenario: Scenario,
                 sampler: MCMCSampler,
                 random_key: Union[None, np.ndarray],
                 correction: Union[None, str, Correction, Type[Correction]],
                 **kwargs) -> Tuple[MCMCSampler, Correction]:
    # Setup correction
    if correction == 'sampler_default':
        correction = sampler.default_correction

    if correction is None:
        correction = Uncorrected()
    elif isclass(correction) and issubclass(correction, Correction):
        correction = correction(**kwargs)
    elif not isinstance(correction, Correction):
        raise TypeError(f'Correction must be of type mocat.Correction')

    # Update kwargs
    for key, value in kwargs:
        if hasattr(sampler, key):
            setattr(sampler, key, value)
        if hasattr(correction, key):
            setattr(correction, key, value)

    # Startup
    sampler.startup(scenario, random_key)
    correction.startup(scenario, sampler)

    if None in sampler.parameters.__dict__.values():
        raise ValueError(f'None found in {sampler.name}.parameters: \n{sampler.parameters}')

    # random_key = None -> use last key from previous run, otherwise set given random_key
    if random_key is not None:
        sampler.initial_extra.random_key = random_key
    sampler.initial_extra.iter = 0

    return sampler, correction


def mcmc_run_params(sampler, correction):
    return CDict(name=sampler.name,
                 parameters=sampler.parameters.copy(),
                 correction=correction.__class__.__name__,
                 tuning=sampler.tuning,
                 initial_state=sampler.initial_state.copy(),
                 initial_extra=sampler.initial_extra.copy())


def run_mcmc(scenario: Scenario,
             sampler: MCMCSampler,
             n: int,
             random_key: Union[None, np.ndarray],
             correction: Union[None, str, Correction, Type[Correction]] = 'sampler_default',
             name: str = None,
             return_random_key: bool = False,
             **kwargs) -> Union[CDict, Tuple[CDict, np.ndarray]]:
    sampler, correction = startup_mcmc(scenario, sampler, random_key, correction, **kwargs)

    run_params = mcmc_run_params(sampler, correction)

    @jit
    def markov_kernel(previous_carry: Tuple[CDict, CDict],
                      _: Any) -> Tuple[Tuple[CDict, CDict], CDict]:
        previous_state, previous_extra = previous_carry
        reject_state = previous_state.copy()
        reject_extra = previous_extra.copy()
        reject_extra.iter += 1

        reject_state, reject_extra = sampler.always(scenario,
                                                    reject_state,
                                                    reject_extra)

        proposed_state, proposed_extra = sampler.proposal(scenario,
                                                          reject_state,
                                                          reject_extra)

        corrected_state, corrected_extra = correction(scenario, sampler,
                                                      reject_state, reject_extra,
                                                      proposed_state, proposed_extra)

        return (corrected_state, corrected_extra), corrected_state

    start = time()

    final_carry, chain = scan(markov_kernel,
                              (sampler.initial_state, sampler.initial_extra),
                              None,
                              length=n)

    end = time()

    sampler.parameters = final_carry[1].parameters

    chain.run_params = run_params

    if sampler.update_initial_state:
        sampler.initial_state = final_carry[0]

    if sampler.update_initial_extra:
        sampler.initial_extra = final_carry[1]

    chain.time = end - start

    if name is None:
        name = scenario.name + ": " + sampler.name
    chain.name = name

    return (chain, final_carry[1].random_key) if return_random_key else chain
