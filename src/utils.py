"""General utility functions for the module sandbox."""
import inspect
import json
import logging
import operator
import time
from contextlib import contextmanager
from functools import reduce
from json.encoder import JSONEncoder

logger = logging.getLogger(__name__)


class CustomJSONEncoder(JSONEncoder):
    """Less restrictive JSON encoder, used only for pretty printing."""

    def default(self, obj):  # noqa: D102
        if inspect.isclass(obj):
            return obj.__name__
        if hasattr(obj, '__class__'):
            return obj.__class__.__name__
        return JSONEncoder.default(self, obj)


@contextmanager
def measure_time(name: str):
    """Masure execution time."""
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    logger.info(f'{name} took {end - start:.2f} seconds.')


def flatten_posterior_samples(posterior_samples: dict, prefix: str = ''):
    """Flatten the posterior samples."""
    flat_dict = {}
    for k, v in posterior_samples.items():
        if isinstance(v, dict):
            flat_dict.update(flatten_posterior_samples(v, prefix=f'{prefix}{k}.'))
        else:
            flat_dict[f'{prefix}{k}'] = v
    return flat_dict


def pretty_string_dict(d: dict, indent: int = 3):
    """Pretty print serializable dictionary."""
    return json.dumps(d, indent=indent, cls=CustomJSONEncoder)


def get_flattened_keys(d: dict, sep='.') -> list[str]:
    """Recursively get `sep` delimited path to the leaves of a tree.

    Parameters:
    -----------
    d: dict
        Parameter Tree to get the names of the leaves from.
    sep: str
        Separator for the tree path.

    Returns:
    --------
        list of names of the leaves in the tree.
    """
    keys = []
    for k, v in d.items():
        if isinstance(v, dict):
            keys.extend([f'{k}{sep}{kk}' for kk in get_flattened_keys(v)])
        else:
            keys.append(k)
    return keys


def get_by_path(tree: dict, path: list):
    """Access a nested object in root by item sequence."""
    return reduce(operator.getitem, path, tree)


def set_by_path(tree: dict, path: list, value):
    """Set a value in a pytree by path."""
    reduce(operator.getitem, path[:-1], tree)[path[-1]] = value
    return tree
