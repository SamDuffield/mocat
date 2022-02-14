########################################################################################################################
# Module: core.py
# Description: mocat primitives Scenario and cdict.
#
# Web: https://github.com/SamDuffield/mocat
########################################################################################################################

import pickle
import copy
from typing import Union, Callable, Tuple
from pathlib import Path
from warnings import warn

import jax.numpy as jnp
from jax import grad, value_and_grad
from jax.tree_util import register_pytree_node_class
from jax.util import unzip2


@register_pytree_node_class
class cdict:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def copy(self) -> 'cdict':
        return cdict(**self.__dict__)

    def deepcopy(self) -> 'cdict':
        return copy.deepcopy(self)

    def __repr__(self):
        return f"mocat.cdict({self.__dict__.__repr__()})"

    def save(self,
             path: Union[str, Path],
             overwrite: bool = False):
        save_cdict(self, path, overwrite)

    def tree_flatten(self):
        return unzip2(self.__dict__.items())[::-1]

    @classmethod
    def tree_unflatten(cls, keys, xs):
        return cls(**dict(zip(keys, xs)))

    def __getitem__(self,
                    item: Union[str, int, slice, jnp.ndarray]) -> 'cdict':
        if isinstance(item, str):
            return self.__dict__[item]

        out_cdict = self.copy()
        for key, attr in out_cdict.__dict__.items():
            if (isinstance(attr, jnp.ndarray) and attr.ndim > 0) \
                    or (isinstance(attr, cdict) and not isinstance(attr, static_cdict)):
                out_cdict.__setattr__(key, attr[item])
        return out_cdict

    def __add__(self,
                other: 'cdict') -> 'cdict':
        out_cdict = self.copy()
        if other is None:
            return out_cdict
        for key, attr in out_cdict.__dict__.items():
            if hasattr(other, key):
                attr_atl = attr
                other_attr_atl = other.__dict__[key]
                if (isinstance(attr, jnp.ndarray) or isinstance(getattr(other, key), jnp.ndarray)):
                    out_cdict.__setattr__(key, jnp.append(jnp.atleast_1d(attr_atl),
                                                          jnp.atleast_1d(other_attr_atl), axis=0))
                elif ((isinstance(attr, cdict) and not isinstance(attr, static_cdict)) \
                      and isinstance(other_attr_atl, cdict) and not isinstance(other_attr_atl, static_cdict)) \
                        or key == 'time':
                    out_cdict.__setattr__(key, attr_atl + other_attr_atl)
        return out_cdict

    @property
    def is_empty(self):
        return self.__dict__ == {}

    def keys(self):
        return self.__dict__.keys()

    def __iter__(self):
        return self.__dict__.__iter__()


class static_cdict(cdict):
    pass


def save_cdict(in_cdict: cdict,
               path: Union[str, Path],
               overwrite: bool = False):
    path = Path(path)
    if path.suffix != '.cdict':
        path = path.with_suffix('.cdict')
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if overwrite:
            path.unlink()
        else:
            raise RuntimeError(f'File {path} already exists.')
    with open(path, 'wb') as file:
        pickle.dump(in_cdict, file)


def load_cdict(path: Union[str, Path]) -> cdict:
    path = Path(path)
    if not path.is_file():
        raise ValueError(f'Not a file: {path}')
    if path.suffix != '.cdict':
        raise ValueError(f'Not a .cdict file: {path}')
    with open(path, 'rb') as file:
        data = pickle.load(file)

    # Convert to DeviceArray (default loads as jnp.ndarray, although I expect JAX to update this at some point)
    for attr_key, attr in data.__dict__.items():
        if isinstance(attr, jnp.ndarray):
            data.__dict__[attr_key] = jnp.asarray(attr)

    return data


def impl_checkable(func: Callable) -> Callable:
    func.not_implemented = True
    return func


def is_implemented(func):
    return not hasattr(func, 'not_implemented')


def clean_1d(scen, attr):
    if hasattr(scen, attr) and is_implemented(getattr(scen, attr)):
        setattr(scen, '_' + attr, getattr(scen, attr))

        def cleaned_func(x: Union[jnp.ndarray, float],
                         random_key: jnp.ndarray) -> float:
            x = jnp.atleast_1d(x)
            x = jnp.reshape(x, (x.size, 1))
            return jnp.squeeze(getattr(scen, '_' + attr)(x, random_key))

        setattr(scen, attr, cleaned_func)


class Scenario:
    name: str
    dim: int
    temperature: float

    grad_potential: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
    potential_and_grad: Callable[[jnp.ndarray, jnp.ndarray], Tuple[float, jnp.ndarray]]
    grad_prior_potential: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
    prior_potential_and_grad: Callable[[jnp.ndarray, jnp.ndarray], Tuple[float, jnp.ndarray]]
    grad_likelihood_potential: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
    likelihood_potential_and_grad: Callable[[jnp.ndarray, jnp.ndarray], Tuple[float, jnp.ndarray]]

    grad_tempered_potential: Callable[[jnp.ndarray, float, jnp.ndarray], jnp.ndarray]
    tempered_potential_and_grad: Callable[[jnp.ndarray, float, jnp.ndarray], Tuple[float, jnp.ndarray]]

    def __init__(self,
                 name: str = None,
                 init_grad: bool = True,
                 **kwargs):
        if name is not None:
            self.name = name

        for key, value in kwargs.items():
            if hasattr(self, key):
                self.__dict__[key] = value

        if is_implemented(self.potential) or is_implemented(self.likelihood_potential):
            if self.dim == 1:
                clean_1d(self, 'potential')
                clean_1d(self, 'prior_potential')
                clean_1d(self, 'likelihood_potential')

            if not is_implemented(self.likelihood_potential) and is_implemented(self.potential):
                if not is_implemented(self.prior_potential):
                    self.prior_potential = lambda x, rk=None: jnp.array(0.)
                self.likelihood_potential = self.potential

            if not hasattr(self, 'temperature'):
                self.temperature = 1.

            if not is_implemented(self.prior_potential):
                warn(f'{self.name} prior_potential not initiated, assuming uniform')
                self.prior_potential = lambda x, random_key=None: jnp.array(0.)

            self.tempered_potential \
                = lambda x, temperature, random_key=None: self.prior_potential(x, random_key) \
                                                          + temperature * self.likelihood_potential(x, random_key)

            self.potential = lambda x, random_key=None: self.tempered_potential(x, self.temperature, random_key)

        if init_grad:
            self.init_grad()

    def __setattr__(self, key, value):
        self.__dict__[key] = value
        if key in ('potential', 'prior_potential', 'likelihood_potential') and hasattr(self, 'grad_potential'):
            self.init_grad()

    def init_grad(self):
        self.grad_prior_potential = grad(self.prior_potential)
        self.prior_potential_and_grad = value_and_grad(self.prior_potential)

        self.grad_likelihood_potential = grad(self.likelihood_potential)
        self.likelihood_potential_and_grad = value_and_grad(self.likelihood_potential)

        self.grad_tempered_potential \
            = lambda x, temperature, random_key=None: self.grad_prior_potential(x, random_key) \
                                                      + temperature * self.grad_likelihood_potential(x, random_key)

        def tempered_potential_and_grad(x: jnp.ndarray,
                                        temperature: float,
                                        random_key: jnp.ndarray) -> Tuple[float, jnp.ndarray]:
            prior_pot, prior_grad = self.prior_potential_and_grad(x, random_key)
            lik_pot, lik_grad = self.likelihood_potential_and_grad(x, random_key)
            return prior_pot + temperature * lik_pot, \
                   prior_grad + temperature * lik_grad

        self.tempered_potential_and_grad = tempered_potential_and_grad

        self.grad_potential = lambda x, random_key=None: self.grad_tempered_potential(x,
                                                                                      self.temperature,
                                                                                      random_key)
        self.potential_and_grad = lambda x, random_key=None: self.tempered_potential_and_grad(x,
                                                                                              self.temperature,
                                                                                              random_key)

    def __repr__(self):
        return f"mocat.Scenario.{self.__class__.__name__}({self.__dict__.__repr__()})"

    @impl_checkable
    def potential(self, x: jnp.ndarray,
                  random_key: jnp.ndarray = None) -> float:
        raise AttributeError(f'{self.name} potential not initiated')

    @impl_checkable
    def prior_potential(self,
                        x: jnp.ndarray,
                        random_key: jnp.ndarray = None) -> float:
        raise AttributeError(f'{self.name} prior_potential not initiated')

    @impl_checkable
    def prior_sample(self,
                     random_key: jnp.ndarray) -> Union[float, jnp.ndarray]:
        raise AttributeError(f'{self.name} prior_sample not initiated')

    @impl_checkable
    def likelihood_potential(self,
                             x: jnp.ndarray,
                             random_key: jnp.ndarray = None) -> float:
        raise AttributeError(f'{self.name} prior_potential not initiated')

    @impl_checkable
    def likelihood_sample(self,
                          x: jnp.ndarray,
                          random_key: jnp.ndarray) -> jnp.ndarray:
        raise AttributeError(f'{self.name} likelihood_sample not initiated')
