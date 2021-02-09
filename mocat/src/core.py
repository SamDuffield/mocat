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

import jax.numpy as np
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
                    item: Union[str, int, slice, np.ndarray]) -> 'cdict':
        if isinstance(item, str):
            return self.__dict__[item]

        out_cdict = self.copy()
        for key, attr in out_cdict.__dict__.items():
            if isinstance(attr, np.ndarray) or (isinstance(attr, cdict) and not isinstance(attr, static_cdict)):
                out_cdict.__setattr__(key, attr[item])
        return out_cdict

    def __add__(self,
                other: 'cdict') -> 'cdict':
        out_cdict = self.copy()
        if other is None:
            return out_cdict
        for key, attr in out_cdict.__dict__.items():
            if hasattr(other, key) and (isinstance(attr, np.ndarray) or isinstance(getattr(other, key), np.ndarray)):
                attr_atl = attr
                other_attr_atl = other.__dict__[key]
                out_cdict.__setattr__(key, np.append(np.atleast_1d(attr_atl),
                                                     np.atleast_1d(other_attr_atl), axis=0))
        if hasattr(self, 'time') and hasattr(other, 'time'):
            out_cdict.time = self.time + other.time
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

    # Convert to DeviceArray (default loads as np.ndarray, although I expect JAX to update this at some point)
    for attr_key, attr in data.__dict__.items():
        if isinstance(attr, np.ndarray):
            data.__dict__[attr_key] = np.asarray(attr)

    return data


def impl_checkable(func: Callable) -> Callable:
    func.not_implemented = True
    return func


def is_implemented(func):
    return not hasattr(func, 'not_implemented')


def clean_1d(scen, attr):
    if hasattr(scen, attr) and is_implemented(getattr(scen, attr)):
        setattr(scen, '_' + attr, getattr(scen, attr))

        def cleaned_func(x: Union[np.ndarray, float],
                         random_key: np.ndarray) -> float:
            x = np.atleast_1d(x)
            x = np.reshape(x, (x.size, 1))
            return np.squeeze(getattr(scen, '_' + attr)(x, random_key))

        setattr(scen, attr, cleaned_func)


class Scenario:
    name: str
    dim: int
    temperature: float

    grad_potential: Callable[[np.ndarray, np.ndarray], np.ndarray]
    potential_and_grad: Callable[[np.ndarray, np.ndarray], Tuple[float, np.ndarray]]
    grad_prior_potential: Callable[[np.ndarray, np.ndarray], np.ndarray]
    prior_potential_and_grad: Callable[[np.ndarray, np.ndarray], Tuple[float, np.ndarray]]
    grad_likelihood_potential: Callable[[np.ndarray, np.ndarray], np.ndarray]
    likelihood_potential_and_grad: Callable[[np.ndarray, np.ndarray], Tuple[float, np.ndarray]]

    grad_tempered_potential: Callable[[np.ndarray, float, np.ndarray], np.ndarray]
    tempered_potential_and_grad: Callable[[np.ndarray, float, np.ndarray], Tuple[float, np.ndarray]]

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

            if is_implemented(self.likelihood_potential):
                if not is_implemented(self.prior_potential):
                    warn(f'{self.name} prior_potential not initiated, assuming uniform')
                    self.prior_potential = lambda x, random_key=None: 0.

                if not hasattr(self, 'temperature'):
                    self.temperature = 1.

                self.tempered_potential \
                    = lambda x, temperature, random_key=None: self.prior_potential(x, random_key) \
                                                              + temperature * self.likelihood_potential(x, random_key)

                self.potential = lambda x, random_key=None: self.tempered_potential(x, self.temperature, random_key)

        if init_grad:
            self.init_grad()

    def init_grad(self):
        if is_implemented(self.likelihood_potential):
            self.grad_prior_potential = grad(self.prior_potential)
            self.prior_potential_and_grad = value_and_grad(self.prior_potential)

            self.grad_likelihood_potential = grad(self.likelihood_potential)
            self.likelihood_potential_and_grad = value_and_grad(self.likelihood_potential)

            self.grad_tempered_potential \
                = lambda x, temperature, random_key=None: self.grad_prior_potential(x, random_key) \
                                                          + temperature * self.grad_likelihood_potential(x, random_key)

            def tempered_potential_and_grad(x: np.ndarray,
                                            temperature: float,
                                            random_key: np.ndarray) -> Tuple[float, np.ndarray]:
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

        elif is_implemented(self.potential):
            self.grad_potential = grad(self.potential)
            self.potential_and_grad = value_and_grad(self.potential)

    def __repr__(self):
        return f"mocat.Scenario.{self.__class__.__name__}({self.__dict__.__repr__()})"

    @impl_checkable
    def potential(self, x: np.ndarray,
                  random_key: np.ndarray = None) -> float:
        raise AttributeError(f'{self.name} potential not initiated')

    @impl_checkable
    def prior_potential(self,
                        x: np.ndarray,
                        random_key: np.ndarray = None) -> float:
        raise AttributeError(f'{self.name} prior_potential not initiated')

    @impl_checkable
    def prior_sample(self,
                     random_key: np.ndarray) -> Union[float, np.ndarray]:
        raise AttributeError(f'{self.name} prior_sample not initiated')

    @impl_checkable
    def likelihood_potential(self,
                             x: np.ndarray,
                             random_key: np.ndarray = None) -> float:
        raise AttributeError(f'{self.name} prior_potential not initiated')

    @impl_checkable
    def likelihood_sample(self,
                          x: np.ndarray,
                          random_key: np.ndarray) -> np.ndarray:
        raise AttributeError(f'{self.name} likelihood_sample not initiated')
