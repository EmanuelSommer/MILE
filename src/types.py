"""Type definitions for the mile module."""
import typing
from pathlib import Path
from typing import (
    NamedTuple,
    Protocol,
    TypeVar,
)

import jax
import jax.numpy as jnp
from blackjax.base import SamplingAlgorithm

ParamTree: typing.TypeAlias = dict[str, typing.Union[jax.Array, 'ParamTree']]
FileTree: typing.TypeAlias = dict[str, typing.Union[Path, 'FileTree']]
PRNGKey = jax.Array


# State in our context is a NamedTuple at least containing a position.
class State(NamedTuple):
    """Base `Sampler State` class for Probabilistic ML.

    Notes
    -----
    - The `Sampler State` is a NamedTuple that contains at least the `position`
        of the sampler.
    """

    position: ParamTree


# PosteriorFunction is used in full-batch sampling it only requires position.


class PosteriorFunction(Protocol):
    """Protocol for Posterior Function used in full-batch sampling.

    Signature:
    ---------
    `(position: ParamTree) -> jnp.ndarray`

    Notes
    -----
    - We define some protocols to bring some structure in developing new
        samplers and gradient estimators.
    """

    def __call__(self, position: ParamTree) -> jnp.ndarray:
        """Posterior Function for full-batch sampling."""
        ...


class GradEstimator(Protocol):
    """Protocol for Gradient Estimator function used in mini-batch sampling.

    Signature:
    ---------
    `(position: ParamTree, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray`

    Notes
    -----
    - We define some protocols to bring some structure in developing new
        samplers and gradient estimators.
    """

    def __call__(
        self, position: ParamTree, x: jnp.ndarray, y: jnp.ndarray
    ) -> jnp.ndarray:
        """Gradient Estimator function for mini-batch sampling."""
        ...


S = TypeVar('S', bound=State)


class Runner(Protocol):
    """Protocol to describe the Runner callables that are used in MCMC sampling.

    Signature:
    ---------
    `(rng_key: jnp.ndarray, state: S, batch: tuple[jnp.ndarray, jnp.ndarray], *args)
    -> S`

    Notes
    -----
        - it simply describes the signature of the callable (the runner) should have.
    """

    def __call__(
        self,
        rng_key: jnp.ndarray,
        state: S,
        batch: tuple[jnp.ndarray, jnp.ndarray],
        *args
    ) -> S:
        """Runner callable for MCMC sampling."""
        ...


Kernel = typing.Callable[..., SamplingAlgorithm]

# Warmup result in mini-batch sampling.
# Warmup Functions must return warmup state and tuned parameters as a dictionary
WarmupResult = tuple[State, dict[str, typing.Any]]
