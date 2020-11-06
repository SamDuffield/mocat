########################################################################################################################
# Module: abc/kalman.py
# Description: Kalman ABC, simulate data to get joint empirical mean and cov before conditioning on given data.
#
# Web: https://github.com/SamDuffield/mocat
########################################################################################################################


from typing import Union, Tuple

from jax import numpy as np, random, vmap
from mocat.src.abc.abc import ABCSampler, ABCScenario
from mocat.src.core import CDict


class KalmanPrior(ABCSampler):
    name = 'Kalman Prior'

    def __init__(self,
                 n_simulate: int,
                 additive_cov_inflation: np.ndarray):
        super().__init__()
        self.parameters.n_simulate = n_simulate
        self.parameters.additive_cov_inflation = additive_cov_inflation

    def acceptance_probability(self,
                               abc_scenario: ABCScenario,
                               reject_state: CDict, reject_extra: CDict,
                               proposed_state: CDict, proposed_extra: CDict) -> Union[float, np.ndarray]:
        return 1.

    def proposal(self,
                 abc_scenario: ABCScenario,
                 reject_state: CDict,
                 reject_extra: CDict) -> Tuple[CDict, CDict]:
        proposed_state = reject_state.copy()
        proposed_extra = reject_extra.copy()
        proposed_extra.random_key, subkey1, subkey2 = random.split(reject_extra.random_key, 3)
        prior_sample = abc_scenario.prior_sample(subkey1)

        sim_data_keys = random.split(subkey2, proposed_extra.parameters.n_simulate)
        simulated_data = vmap(abc_scenario.simulate_data, (None, 0))(prior_sample, sim_data_keys)
        simulated_summary_statistics = vmap(abc_scenario.summarise_data)(simulated_data)

        simulated_summary_mean = np.mean(simulated_summary_statistics)

        proposed_extra.simulated_data = abc_scenario.simulate_data(proposed_state.value, subkey2)
        proposed_state.distance = abc_scenario.distance_function(proposed_extra.simulated_data)
        return proposed_state, proposed_extra





