"""Implementation of Base DataLoader class."""
from abc import ABC, abstractmethod
from typing import Generator, Literal

import jax.numpy as jnp

from src.config.data import DataConfig


class BaseLoader(ABC):
    """Base class for all loaders."""

    def __init__(self, config: DataConfig):
        """Initialize the loader."""
        assert isinstance(config, DataConfig)
        self.config = config

    @property
    def dataset_name(self):
        return self.config.path.split('/')[-1].split('.')[0]

    def __str__(self):
        """Return informative string representation of the class."""
        return (
            f'{self.__class__.__name__}:\n'
            f' | Data: {self.dataset_name}\n'
            f' | Task: {self.config.task}'
        )

    @abstractmethod
    def iter(
        self, split: Literal['train', 'test', 'valid'], batch_size: int | None, **kwargs
    ) -> Generator[dict[str, jnp.ndarray], None, None]:
        """
        Return the next batch of data in dictionary format. e.g.

            {
                'feature': jnp.ndarray,
                'label': jnp.ndarray
            }
        containing batched (batch_size) features and labels.
        """
        pass

    @abstractmethod
    def shuffle(self, *args, **kwargs) -> None:
        """Shuffle the data for the next .iter() call."""
        pass

    @abstractmethod
    def __len__(self):
        """Return the length of the data."""
        pass
