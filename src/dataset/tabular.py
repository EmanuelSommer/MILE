"""DataLoader Implementations."""
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np

from src.config.data import (
    DataConfig,
    DatasetType,
    Task,
)
from src.dataset.base import BaseLoader


class TabularLoader(BaseLoader):
    """Tabular data loader."""

    def __init__(self, config: DataConfig, rng: jnp.ndarray, target_len: int = 1):
        """__init__ method for the TabularLoader class."""
        super().__init__(config)
        self.target_len = target_len
        assert self.config.data_type == DatasetType.TABULAR
        self._key = rng
        self.data = self.load_data(shuffle=True, normalize=config.normalize)
        if self.config.datapoint_limit:
            self.data = self.data[: self.config.datapoint_limit]
        self.data_train = self.data[: int(len(self.data) * self.config.train_split)]
        self.data_valid = self.data[
            int(len(self.data) * self.config.train_split) : int(
                len(self.data) * (self.config.train_split + self.config.valid_split)
            )
        ]
        self.data_test = self.data[
            int(len(self.data) * (self.config.train_split + self.config.valid_split)) :
        ]

    def __str__(self):
        """Return informative string representation of the class."""
        return (
            super().__str__() + '\n'
            f' | Features: {self.data.shape[-1] - self.target_len}\n'
            f' | Target Length: {self.target_len}\n'
            f' | Train: {len(self.data_train)}\n'
            f' | Valid: {len(self.data_valid)}\n'
            f' | Test: {len(self.data_test)}'
        )

    @property
    def key(self):
        """Return the next rng key for the dataloader."""
        self._key, key = jax.random.split(self._key)
        return key

    @property
    def train_x(self):
        """Return the training features."""
        return self.data_train[..., : -self.target_len].squeeze()

    @property
    def train_y(self):
        """Return the training labels."""
        if self.config.task == Task.CLASSIFICATION:
            return self.data_train[..., -self.target_len :].squeeze().astype(jnp.int32)
        return self.data_train[..., -self.target_len :].squeeze()

    @property
    def valid_x(self):
        """Return the validation features."""
        return self.data_valid[..., : -self.target_len].squeeze()

    @property
    def valid_y(self):
        """Return the validation labels."""
        if self.config.task == Task.CLASSIFICATION:
            return self.data_valid[..., -self.target_len :].squeeze().astype(jnp.int32)
        return self.data_valid[..., -self.target_len :].squeeze()

    @property
    def test_x(self):
        """Return the testing features."""
        return self.data_test[..., : -self.target_len].squeeze()

    @property
    def test_y(self):
        """Return the testing labels."""
        if self.config.task == Task.CLASSIFICATION:
            return self.data_test[..., -self.target_len :].squeeze().astype(jnp.int32)
        return self.data_test[..., -self.target_len :].squeeze()

    def iter(
        self,
        split: Literal['train', 'test', 'valid'],
        batch_size: int | None = None,
        n_devices: int = 1,
    ):
        """
        Return the next batch of data in dictionary format. e.g.

            {
                'feature': jnp.ndarray,
                'label': jnp.ndarray
            }
        containing batched (batch_size) features and labels.
        """
        assert split in ('train', 'test', 'valid')
        if split == 'train':
            return self._iter(self.data_train, batch_size, n_devices=n_devices)
        elif split == 'valid':
            return self._iter(self.data_valid, batch_size, n_devices=n_devices)
        else:
            return self._iter(self.data_test, batch_size, n_devices=n_devices)

    def load_data(self, shuffle: bool, normalize: bool = True):
        """
        Load the dataset from the specified file.

        Returns:
        - data_features (numpy.ndarray): An array containing the features of the
          dataset.
        - data_target (numpy.ndarray): An array containing the target
          variable(s) of the dataset.
        """
        if self.config.path.endswith('.npy'):
            data = np.load(self.config.path)
        elif self.config.path.endswith('.csv'):
            data = np.loadtxt(self.config.path, delimiter=',')
        elif self.config.path.endswith('.data'):
            data = np.genfromtxt(self.config.path, delimiter=' ')
        else:
            raise NotImplementedError(
                'Only .npy and .csv files are supported at this time.'
            )
        data = jnp.array(data)
        if normalize:
            if self.config.task == Task.CLASSIFICATION:
                # Dont normalize target
                data = jnp.concatenate(
                    [
                        (data[:, :-1] - data[:, :-1].mean(axis=0))
                        / data[:, :-1].std(axis=0),
                        data[:, -1:],
                    ],
                    axis=1,
                )
            else:
                data = (data - data.mean(axis=0)) / data.std(axis=0)
        if shuffle:
            data = self._shuffle(data)
        return data

    def _shuffle(self, data: jnp.ndarray):
        """Shuffle the data for the next dataloder iteration."""
        indices = jax.random.permutation(self.key, jnp.arange(len(data)))
        return data[indices]

    def shuffle(self, split: Literal['train', 'test', 'valid'] = 'train'):
        """Shuffle the data for the next dataloder iteration."""
        if split == 'train':
            self.data_train = self._shuffle(self.data_train)
        elif split == 'valid':
            self.data_valid = self._shuffle(self.data_valid)
        else:
            self.data_test = self._shuffle(self.data_test)

    def __len__(self):
        """Return the length of the data."""
        return len(self.data)

    def _iter(self, data: jnp.ndarray, batch_size: int | None, n_devices: int = 1):
        """Iterate over the data helper."""
        split_col = data.shape[-1] - self.target_len
        if not data.size:
            return
        if not batch_size:
            if n_devices > 1:  # Replicate data for multiple devices
                data = jnp.repeat(data[None, ...], n_devices, axis=0)
            if self.config.task == Task.CLASSIFICATION:
                yield {
                    'feature': data[..., :split_col],
                    'label': data[..., split_col:].astype(jnp.int32).squeeze(),
                }
            else:
                yield {
                    'feature': data[..., :split_col],
                    'label': data[..., split_col:].squeeze(),
                }

        else:
            n_batches = len(data) // batch_size
            data = data[: n_batches * batch_size]  # drop last
            splits = [
                jnp.array_split(
                    jax.random.permutation(self.key, jnp.arange(len(data))), n_batches
                )
                for _ in range(n_devices)
            ]
            for i in range(n_batches):
                batch = jnp.stack([data[split[i]] for split in splits])
                if n_devices == 1:
                    batch = batch.squeeze(0)
                if self.config.task == Task.CLASSIFICATION:
                    yield {
                        'feature': batch[..., :split_col],
                        'label': batch[..., split_col:].squeeze().astype(jnp.int32),
                    }
                else:
                    yield {
                        'feature': batch[..., :split_col],
                        'label': batch[..., split_col:].squeeze(),
                    }
