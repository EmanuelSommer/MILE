"""Priors."""
from typing import Callable, NamedTuple

import jax
import jax.nn.initializers as jinit
import jax.numpy as jnp
import jax.scipy.stats as stats
from jax.flatten_util import ravel_pytree

from src.config.base import BaseStrEnum
from src.types import ParamTree


class PriorDist(BaseStrEnum):
    """Prior Distribution Names.

    Notes
    -----
    - This can either be a general distribution, where the user can pass the
        `parameters` from the `configuration` file like: `Normal`, `Laplace`, etc.

        Example::
        ```yaml
            name: Normal
            parameters:
                loc: 5.0
                scale: 2.0
        ```
        ```
        print(get_prior(**parameters))
        >>> Normal(loc=5.0, scale=2.0)
        ```

    - Or a `pre-defined` distribution, where the user chooses the `name`
        and the `parameters` are fully or partially defined.

        Example::
        ```yaml
            name: StandardNormal
            parameters:
                limit: 5.0
        ```
        ```
        print(get_prior(**parameters))
        >>> Normal(loc=0.0, scale=1.0, limit=5.0)
        ```
    - Complete logic for the initialization of the prior must be
        defined in the `.get_prior()` method.
    """

    NORMAL = 'Normal'
    StandardNormal = 'StandardNormal'
    LAPLACE = 'Laplace'

    def get_prior(self, **parameters):
        """Return a prior class."""
        return Prior.from_name(self, **parameters)


class Prior(NamedTuple):
    """Base Class for Priors."""

    f_init: Callable  # TODO: Callable[[KeyArray, ParamTree], jax.Array]
    log_prior: Callable[[ParamTree], jax.Array]
    name: str

    @classmethod
    def from_name(cls, name: PriorDist, **parameters):
        """Initialize the prior class instance."""
        if name == PriorDist.StandardNormal:
            return cls(
                name=PriorDist.StandardNormal,
                f_init=f_init_normal(),
                log_prior=log_prior_normal(),
            )
        elif name == PriorDist.NORMAL:
            return cls(
                name=PriorDist.NORMAL,
                f_init=f_init_normal(**parameters),
                log_prior=log_prior_normal(**parameters),
            )
        elif name == PriorDist.LAPLACE:
            return cls(
                name=PriorDist.LAPLACE,
                f_init=f_init_laplace(**parameters),
                log_prior=log_prior_laplace(**parameters),
            )
        else:
            raise NotImplementedError(
                f'Prior Distribution for {name} is not yet implemented.'
            )


def f_init_normal(loc: float = 0.0, scale: float = 1.0) -> Callable:
    """Initialize from Normal distribution."""
    if not loc == 0.0:
        print('Initialization: Ignoring location unequal to zero.')
    return jinit.normal(stddev=scale)


def log_prior_normal(loc: float = 0.0, scale: float = 1.0) -> Callable:
    """Evaluate Normal prior on all weights."""

    def log_prior(params: ParamTree) -> jax.Array:
        scores = stats.norm.logpdf(ravel_pytree(params)[0], loc=loc, scale=scale)
        return jnp.sum(scores)

    return log_prior


def f_init_laplace(loc: float = 0.0, scale: float = 1.0) -> Callable:
    """Initialize from Laplace distribution."""

    def f_init(key, shape, dtype=jnp.float32) -> jax.Array:
        w = jax.random.laplace(key, shape, dtype)
        return w * scale + loc

    return f_init


def log_prior_laplace(loc: float = 0.0, scale: float = 1.0) -> Callable:
    """Evaluate Laplace prior on all weights."""

    def log_prior(params: ParamTree) -> jax.Array:
        scores = stats.laplace.logpdf(ravel_pytree(params)[0], loc=loc, scale=scale)
        return jnp.sum(scores)

    return log_prior
