# mocat
Create, tune and compare Monte Carlo algorithms in JAX.

## Install
```
pip install mocat
```


## Define a target distribution
We always work with the target's potential (negative log density)
```python
from jax import numpy as np, random
import matplotlib.pyplot as plt
import mocat

class Rastrigin(mocat.Scenario):
    name = "Rastrigin"

    def __init__(self,
                 dim: int = 1,
                 a: float = 0.5):
        self.dim = dim
        self.a = a
        super().__init__()

    def potential(self, x: np.ndarray) -> float:
        return self.a*self.dim + np.sum(x**2 - self.a * np.cos(2 * np.pi * x), axis=-1)
```


## Compare samplers
Run MALA and HMC with a Robbins-Monro schedule to adapt the stepsize to a desired acceptance rate
```python
random_key = random.PRNGKey(0)

scenario_rastrigin = Rastrigin(5)

n = int(1e5)

mala = mocat.Overdamped()
mala_samps = mocat.run_mcmc(scenario_rastrigin, mala, n, random_key, correction=mocat.RMMetropolis())

hmc = mocat.HMC(leapfrog_steps=10)
hmc_samps = mocat.run_mcmc(scenario_rastrigin, hmc, n, random_key, correction=mocat.RMMetropolis())
```


Plot the first two dimensions along with trace plots and autocorrelation of the potential
```python
fig, axes = plt.subplots(3, 2)
mocat.plot_2d_samples(mala_samps, ax=axes[0,0])
mocat.plot_2d_samples(hmc_samps, ax=axes[0,1])

mocat.trace_plot(mala_samps, last_n=1000, ax=axes[1,0], title=None)
mocat.trace_plot(hmc_samps, last_n=1000, ax=axes[1,1], title=None)

mocat.autocorrelation_plot(mala_samps, ax=axes[2,0], title=None)
mocat.autocorrelation_plot(hmc_samps, ax=axes[2,1], title=None)
plt.tight_layout()
```
![comp-metrics](examples/images/MALA_HMC_Rastrigin.png?raw=true "MALA vs HMC - Rastrigin")

Plus functionality for effective sample size, acceptance rate, squared jumping distance, kernelised Stein discrepancies...


## Create your own MCMC sampler

```python
class Underdamped(mocat.MCMCSampler):
    name = 'Underdamped'
    default_correction = mocat.Metropolis()

    def __init__(self,
                 stepsize = None,
                 leapfrog_steps = 1,
                 friction = 1.0):
        super().__init__()
        self.parameters.stepsize = stepsize
        self.parameters.leapfrog_steps = leapfrog_steps
        self.parameters.friction = friction
        self.tuning.target = 0.651

    def startup(self, scenario, random_key):
        super().startup(scenario, random_key)
        self.initial_state.grad_potential = scenario.grad_potential(self.initial_state.value)
        if not hasattr(self.initial_state, 'momenta') or self.initial_state.momenta.shape[-1] != scenario.dim:
            self.initial_state.momenta = np.zeros(scenario.dim)

    def always(self, scenario, reject_state, reject_extra):
        d = scenario.dim

        stepsize = reject_extra.parameters.stepsize
        friction = reject_extra.parameters.friction

        reject_state.momenta = reject_state.momenta * -1

        reject_extra.random_key, subkey = random.split(reject_extra.random_key)
        reject_state.momenta = reject_state.momenta * np.exp(- friction * stepsize) \
                               + np.sqrt(1 - np.exp(- 2 * friction * stepsize)) * random.normal(subkey, (d,))
        return reject_state, reject_extra

    def proposal(self, scenario, reject_state, reject_extra):
        stepsize = reject_extra.parameters.stepsize
        
        proposed_state = mocat.utils.leapfrog(reject_state,
                                              scenario.grad_potential,
                                              stepsize,
                                              self.parameters.leapfrog_steps)[-1]
        proposed_state.momenta = proposed_state.momenta * -1

        return proposed_state, reject_extra

    def acceptance_probability(self, scenario, reject_state, reject_extra, proposed_state, proposed_extra):
        pre_min_alpha = np.exp(- proposed_state.potential
                               + reject_state.potential
                               - mocat.utils.gaussian_potential(proposed_state.momenta)
                               + mocat.utils.gaussian_potential(reject_state.momenta))

        return np.minimum(1., pre_min_alpha)
```



