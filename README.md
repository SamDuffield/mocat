# mocat
All things Monte Carlo, written in JAX.
- Markov chain Monte Carlo
- Transport samplers
    * Sequential Monte Carlo samplers (likelihood tempering)
    * Stein variational gradient descent
    
- Approximate Bayesian computation
    * Rejection/Importance ABC
    * MCMC ABC
    * SMC ABC
  
- State-space models
  * Particle filtering
  * Particle smoothing
  * Kalman filtering + smoothing

## Install
```
pip install mocat
```


## Define a target distribution
We always work with the target's potential (negative log density)
```python
from jax import numpy as jnp, random
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

    def potential(self,
                  x: jnp.ndarray,
                  random_key: jnp.ndarray) -> float:
        return self.a*self.dim + jnp.sum(x**2 - self.a * jnp.cos(2 * jnp.pi * x), axis=-1)
```


## Compare samplers
Run MALA and HMC with a Robbins-Monro schedule to adapt the stepsize to desired acceptance rate (defined in e.g. `mala.tuning`)
```python
random_key = random.PRNGKey(0)

scenario_rastrigin = Rastrigin(5)

n = int(1e5)

mala = mocat.Overdamped()
mala_samps = mocat.run(scenario_rastrigin, mala, n, random_key, correction=mocat.RMMetropolis())

hmc = mocat.HMC(leapfrog_steps=10)
hmc_samps = mocat.run(scenario_rastrigin, hmc, n, random_key, correction=mocat.RMMetropolis())
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

axes[0,0].set_title(scenario_rastrigin.name + ': ' + mala.name)
axes[0,1].set_title(scenario_rastrigin.name + ': ' + mala.name)
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

    def startup(self,
                scenario: Scenario,
                n: int,
                initial_state: cdict,
                initial_extra: cdict,
                **kwargs) -> Tuple[cdict, cdict]:
        initial_state, initial_extra = super().startup(scenario, n,
                                                       initial_state, initial_extra, **kwargs)
        initial_extra.random_key, scen_key = random.split(initial_extra.random_key)
        initial_state.potential, initial_state.grad_potential = scenario.potential_and_grad(initial_state.value,
                                                                                            scen_key)
        if not hasattr(initial_state, 'momenta') or initial_state.momenta.shape[-1] != scenario.dim:
            initial_state.momenta = jnp.zeros(scenario.dim)
        return initial_state, initial_extra

    def always(self, scenario, reject_state, reject_extra):
        d = scenario.dim

        stepsize = reject_extra.parameters.stepsize
        friction = reject_extra.parameters.friction

        reject_state.momenta = reject_state.momenta * -1

        reject_extra.random_key, subkey = random.split(reject_extra.random_key)
        reject_state.momenta = reject_state.momenta * jnp.exp(- friction * stepsize) \
                               + jnp.sqrt(1 - jnp.exp(- 2 * friction * stepsize)) * random.normal(subkey, (d,))
        return reject_state, reject_extra

    def proposal(self,
                 scenario: Scenario,
                 reject_state: cdict,
                 reject_extra: cdict) -> Tuple[cdict, cdict]:
        random_keys = random.split(reject_extra.random_key, self.parameters.leapfrog_steps + 1)
        reject_extra.random_key = random_keys[0]
        all_leapfrog_state = mocat.utils.leapfrog(scenario.potential_and_grad,
                                            reject_state,
                                            reject_extra.parameters.stepsize,
                                            random_keys[1:])
        proposed_state = all_leapfrog_state[-1]
        proposed_state.momenta *= -1
        return proposed_state, reject_extra

    def acceptance_probability(self, scenario, reject_state, reject_extra, proposed_state, proposed_extra):
        pre_min_alpha = jnp.exp(- proposed_state.potential
                               + reject_state.potential
                               - mocat.utils.gaussian_potential(proposed_state.momenta)
                               + mocat.utils.gaussian_potential(reject_state.momenta))
        return jnp.minimum(1., pre_min_alpha)
```



