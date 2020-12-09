########################################################################################################################
# Module: mcmc/run.py
# Description: Run Markov chains!
#
# Web: https://github.com/SamDuffield/mocat
########################################################################################################################

from time import time
from typing import Union, Tuple, Type, Any, TypeVar
from inspect import isclass

from jax import numpy as np, jit
from jax.lax import scan

from mocat.src.core import Scenario, CDict
from mocat.src.mcmc.corrections import Correction, Uncorrected
from mocat.src.mcmc.sampler import MCMCSampler

MCMCSamplerType = TypeVar('MCMCSamplerType', bound=MCMCSampler)


def check_correction(sampler: MCMCSamplerType,
                     correction: Union[None, str, Correction, Type[Correction]],
                     **kwargs) -> Tuple[MCMCSamplerType, Correction]:
    # Setup correction
    if correction == 'sampler_default' and hasattr(sampler, 'default_correction'):
        correction = sampler.default_correction

    if correction is None:
        correction = Uncorrected()
    elif isclass(correction) and issubclass(correction, Correction):
        correction = correction()
    elif not isinstance(correction, Correction):
        raise TypeError(f'Correction must be of type mocat.Correction')

    # Update kwargs
    for key, value in kwargs.items():
        if hasattr(sampler, key):
            setattr(sampler, key, value)
        elif hasattr(sampler, 'parameters') and hasattr(sampler.parameters, key):
            setattr(sampler.parameters, key, value)
        else:
            setattr(correction, key, value)

    return sampler, correction


def startup_mcmc(scenario: Scenario,
                 sampler: MCMCSampler,
                 random_key: Union[None, np.ndarray],
                 correction: Union[None, str, Correction, Type[Correction]],
                 initial_state: CDict = None,
                 initial_extra: CDict = None) -> Tuple[CDict, CDict]:
    # Startup
    initial_state, initial_extra = sampler.startup(scenario, initial_state, initial_extra, random_key)
    initial_state, initial_extra = correction.startup(scenario, sampler, initial_state, initial_extra)

    # random_keys = None -> use last key from previous run, otherwise set given random_keys
    if random_key is not None:
        initial_extra.random_key = random_key

    if not hasattr(initial_extra, 'iter'):
        initial_extra.iter = 0

    return initial_state, initial_extra


def mcmc_run_params(sampler: MCMCSampler,
                    correction: Correction,
                    initial_state: CDict,
                    initial_extra: CDict) -> CDict:
    return CDict(name=sampler.name,
                 parameters=sampler.parameters.copy(),
                 correction=correction.__class__.__name__,
                 tuning=sampler.tuning,
                 initial_state=initial_state.copy(),
                 initial_extra=initial_extra.copy())


def run_mcmc(scenario: Scenario,
             sampler: MCMCSampler,
             n: int,
             random_key: Union[None, np.ndarray],
             correction: Union[None, str, Correction, Type[Correction]] = 'sampler_default',
             initial_state: CDict = None,
             initial_extra: CDict = None,
             name: str = None,
             return_random_key: bool = False,
             **kwargs) -> Union[CDict, Tuple[CDict, np.ndarray]]:
    sampler, correction = check_correction(sampler, correction, **kwargs)
    initial_state, initial_extra = startup_mcmc(scenario, sampler, random_key, correction, initial_state, initial_extra)
    run_params = mcmc_run_params(sampler, correction, initial_state, initial_extra)

    if None in sampler.parameters.__dict__.values():
        raise ValueError(f'None found in {sampler.name}.parameters: \n{sampler.parameters}')

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
                              (initial_state, initial_extra),
                              None,
                              length=n)

    chain.value.block_until_ready()
    end = time()

    sampler.parameters = final_carry[1].parameters

    chain.run_params = run_params

    chain.time = end - start

    if name is None:
        name = scenario.name + ": " + sampler.name
    chain.name = name

    return (chain, final_carry[1].random_key) if return_random_key else chain
