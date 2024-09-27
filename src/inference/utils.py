"""Utility functions for the experiments."""
import jax
import jax.numpy as jnp

from src.types import ParamTree


def count_chains(samples: ParamTree) -> int:
    """Find number of chains in the samples.

    Raises:
        ValueError: If the number of chains is not consistent across layers.
    """
    n = set([x.shape[0] for x in jax.tree.leaves(samples)])
    if len(n) > 1:
        raise ValueError(f'Ambiguous chain dimension across layers. Found {n}')
    return n.pop()


def count_samples(samples: ParamTree) -> int:
    """Find number of samples in the samples.

    Raises:
        ValueError: If the number of samples is not consistent across layers.
    """
    n = set([x.shape[1] for x in jax.tree.leaves(samples)])
    if len(n) > 1:
        raise ValueError(f'Ambiguous sample dimension across layers. Found {n}')
    return n.pop()


def get_mem_size(x: ParamTree) -> int:
    """Get the memory size of the model."""
    return sum([x.nbytes for x in jax.tree_leaves(x)])


def count_params(params: ParamTree) -> int:
    """Count the number of parameters in the model."""
    return sum([x.size for x in jax.tree.leaves(params)])


def count_nan(params: ParamTree) -> ParamTree:
    """Count the number of NaNs in the parameter tree."""
    return jax.tree.map(lambda x: jnp.isnan(x).sum().item(), params)


def impute_nan(params: ParamTree, value: float = 0.0) -> ParamTree:
    """Impute NaNs in the parameter tree with a value."""
    return jax.tree.map(lambda x: jnp.where(jnp.isnan(x), value, x), params)
