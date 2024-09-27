"""Training utilities."""
import operator
import pickle
from functools import reduce
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import jax
import jax.numpy as jnp
import numpy as np
from flax.training.train_state import TrainState
from numpy.lib import format

from src.types import ParamTree
from src.utils import get_flattened_keys, set_by_path


def count_params(params: ParamTree) -> int:
    """Get the number of parameters in a model."""
    return sum(p.size for p in jax.tree.leaves(params))


def reconstruct_pytree(names: list[str], leaves: list[jnp.ndarray], delimiter='.'):
    """
    Reconstruct a pytree from a list of names and leaves.

    Args:
    names: list - list of names of the leaves
    leaves: list - list of leaves
    delimiter: str - delimiter to split the names

    Returns:
    dict - reconstructed pytree
    """
    pytree = {}
    for name, leaf in zip(names, leaves):
        levels = name.split(delimiter)
        pytree = set_by_path(pytree, levels, leaf)
    return pytree


def merge_trees(tree1: dict, tree2: dict):
    """
    Merge two pytrees.

    Args:
    tree1: dict - first pytree
    tree2: dict - second pytree

    Returns:
    dict - merged pytree
    """
    for k in tree2.keys():
        if k not in tree1:
            tree1[k] = tree2[k]
        else:
            if isinstance(tree1[k], dict):
                tree1[k] = merge_trees(tree1[k], tree2[k])
            else:
                tree1[k] = tree2[k]
    return tree1


def get_by_path(root: dict, items: list):
    """Access a nested object in root by item sequence."""
    return reduce(operator.getitem, items, root)


def save_params(dir: str | Path, params: ParamTree, idx: int | None = None):
    """
    Save model parameters to disk.

    Args:
        dir: str | Path - directory to save the parameters to
        params: dict - parameters to save
        idx: int | None - index to append to the file name (default: None)
    """
    if not isinstance(dir, Path):
        dir = Path(dir)
    if not dir.exists():
        dir.mkdir(parents=True)
    leaves, tree = jax.tree.flatten(params)
    if not (dir.parent / 'tree').exists():
        save_tree(dir.parent, tree)
    param_names = get_flattened_keys(params)
    name = f'params_{idx}.npz' if idx is not None else 'params.npz'
    np.savez_compressed(dir / name, **dict(zip(param_names, leaves)))


def save_tree(dir: str | Path, tree: ParamTree):
    """Save tree in .pkl format."""
    with open(dir / 'tree', 'wb') as f:
        pickle.dump(tree, f)


def load_tree(dir: str | Path) -> ParamTree:
    """Load tree in .pkl format."""
    with open(dir / 'tree', 'rb') as f:
        return pickle.load(f)


def load_params(params_path: str | Path, tree_path: str | Path) -> ParamTree:
    """Load model parameters from disk."""
    with open(tree_path, 'rb') as f:
        tree = pickle.load(f)
    with jnp.load(params_path) as param_zip:
        leaves = [jnp.array(param_zip[i]) for i in param_zip.files]
    return jax.tree.unflatten(tree, leaves)


def load_params_batch(
    params_path: list[str | Path], tree_path: str | Path
) -> ParamTree:
    """Load multiple model parameters and stack them for pmap compatibility."""
    params_path = [Path(p) if not isinstance(p, Path) else p for p in params_path]
    params_path = sorted(params_path, key=lambda x: int(x.stem.split('_')[-1]))
    if len(params_path) == 1:
        return load_params(params_path[0], tree_path)
    with open(tree_path, 'rb') as f:
        tree = pickle.load(f)
    params_list = []
    for param in params_path:
        with jnp.load(param) as param_zip:
            params_list.append([jnp.array(param_zip[i]) for i in param_zip.files])
    leaves = [
        jnp.stack([p[i] for p in params_list]) for i in range(len(params_list[0]))
    ]
    return jax.tree.unflatten(tree, leaves)


def load_samples_from_dir(dir: str | Path, tree_path: str | Path) -> ParamTree:
    """
    Load model parameters from disk, Specifically for probabilisticml framework.

    dir: str | Path -
        directory to load the parameters from either 'samples' or 'sampling_warmup'
        directory in selected experiment

    tree_path: str | Path - Path to the module tree
    """
    with open(tree_path, 'rb') as f:
        tree = pickle.load(f)
    if not isinstance(dir, Path):
        dir = Path(dir)
    chain_dirs = sorted(
        [d for d in dir.iterdir() if d.is_dir()],
        key=lambda x: int(x.stem.split('_')[-1]),
    )
    chain_stack = []

    for chain_dir in chain_dirs:
        samples = sorted(
            [p for p in chain_dir.iterdir() if p.suffix == '.npz'],
            key=lambda x: int(x.stem.split('_')[-1]),
        )
        params, param_names = _load_chain_samples(samples)
        chain_stack.append(params)
    leaves = [
        jnp.stack([chain[i] for chain in chain_stack]) for i in range(len(param_names))
    ]
    return jax.tree.unflatten(tree, leaves)


def _load_chain_samples(samples: list[Path]) -> tuple[list[jax.Array], list[str]]:
    if not samples:
        raise ValueError('No samples found in the directory')
    params = []
    names = []
    for sample in samples:
        with jnp.load(sample) as param_zip:
            names = param_zip.files
            params.append([jnp.array(param_zip[i]) for i in names])
    assert all([len(p) == len(names) for p in params]), 'Inconsistent number of samples'
    params = [jnp.stack([p[i] for p in params]) for i in range(len(names))]
    return params, names


def load_params_to_state(path: str | Path, tree_path: str | Path, state: TrainState):
    """Load model parameters from disk to state."""
    return state.replace(params=load_params(path, tree_path))


def add_params_to_zip(f: str | Path, params: ParamTree):
    """
    Add a tree to a numpy zip file.

    Args:
        f: str | Path - file path to the .npz file
        tree: dict - tree to add
        sep: str - separator for the keys (default: '.')
    """
    if not isinstance(f, Path):
        f = Path(f)
    if f.suffix != '.npz':
        f = f.with_suffix('.npz')
    if not f.parent.exists():
        f.parent.mkdir(parents=True)
    leaves, _ = jax.tree.flatten(params)
    param_names = get_flattened_keys(params)
    add_arrays_to_zip(f, leaves, param_names)


def add_arrays_to_zip(
    f: str | Path, arrs: list[jnp.ndarray] | np.ndarray, names: list[str] | str
):
    """
    Add arrays to a numpy zip file.

    Args:
        f: str | Path - file path to the .npz file
        arrs: list[jnp.ndarray] - arrays to add
        names: list[str] - names of the arrays to create in the zip file
    """
    if not isinstance(f, Path):
        f = Path(f)
    mode = 'w' if not f.exists() else 'a'
    if not isinstance(arrs, list):
        assert isinstance(names, str), 'both arrs and names must be lists'
        arrs = [arrs]
        names = [names]

    assert len(arrs) == len(names), 'arrs and names must have the same length'
    zpf = ZipFile(f, mode=mode, allowZip64=True, compression=ZIP_DEFLATED)
    try:
        for arr, name in zip(arrs, names):
            with zpf.open(name, 'w', force_zip64=True) as wf:
                format.write_array(wf, np.asanyarray(arr), allow_pickle=False)
    finally:
        zpf.close()


def remove_frozen_params(params: ParamTree, layers_to_freeze: list[int]):
    """
    Remove frozen parameters from the model ParamTree.

    Args:
        params (ParamTree): Model parameters.
        layers_to_freeze (list[int]): List of point delimited layer names to freeze.

    Returns:
        ParamTree: Model parameters with frozen layers removed
    """
    names = get_flattened_keys(params)
    if not all([k in names for k in layers_to_freeze]):
        raise ValueError('Invalid layer names to freeze. Available layers: ', names)
    non_frozen = [k for k in names if k not in layers_to_freeze]
    return reconstruct_pytree(
        names=non_frozen,
        leaves=[get_by_path(params, k.split('.')) for k in non_frozen],
    )
