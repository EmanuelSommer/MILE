"""Image DataLoader Implementations."""
from typing import Literal

import jax
import jax.numpy as jnp

from src.config.data import (
    DataConfig,
    DatasetType,
    Source,
    Task,
)
from src.dataset.base import BaseLoader


class ImageLoader(BaseLoader):
    """Image data loader."""

    def __init__(self, config: DataConfig, rng: jnp.ndarray):
        """__init__ method for the TabularLoader class."""
        super().__init__(config)
        assert self.config.data_type == DatasetType.IMAGE
        self._key = rng
        self.data = self.load_data(shuffle=True, normalize=config.normalize)
        # data_train, data_valid, data_test are just to give correct length
        self.data_train = self.data['train_y']
        self.data_valid = self.data['valid_y']
        self.data_test = self.data['test_y']

    def __str__(self):
        """Return informative string representation of the class."""
        return (
            super().__str__() + '\n'
            f' | Train: {len(self.data["train_x"])}\n'
            f' | Valid: {len(self.data["valid_x"])}\n'
            f' | Test: {len(self.data["test_x"])}'
        )

    @property
    def key(self):
        """Return the next rng key for the dataloader."""
        self._key, key = jax.random.split(self._key)
        return key

    @property
    def train_x(self):
        """Return the training features."""
        return self.data['train_x']

    @property
    def train_y(self):
        """Return the training labels."""
        if self.config.task == Task.CLASSIFICATION:
            return self.data['train_y'].squeeze().astype(jnp.int32)
        return self.data['train_y'].squeeze()

    @property
    def valid_x(self):
        """Return the validation features."""
        return self.data['valid_x']

    @property
    def valid_y(self):
        """Return the validation labels."""
        if self.config.task == Task.CLASSIFICATION:
            return self.data['valid_y'].squeeze().astype(jnp.int32)
        return self.data['valid_y'].squeeze()

    @property
    def test_x(self):
        """Return the testing features."""
        return self.data['test_x']

    @property
    def test_y(self):
        """Return the testing labels."""
        if self.config.task == Task.CLASSIFICATION:
            return self.data['test_y'].squeeze().astype(jnp.int32)
        return self.data['test_y'].squeeze()

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
            return self._iter(
                self.data['train_x'],
                self.data['train_y'],
                batch_size,
                n_devices=n_devices,
            )
        elif split == 'valid':
            return self._iter(
                self.data['valid_x'],
                self.data['valid_y'],
                batch_size,
                n_devices=n_devices,
            )
        else:
            return self._iter(
                self.data['test_x'],
                self.data['test_y'],
                batch_size,
                n_devices=n_devices,
            )

    def load_data(self, shuffle: bool, normalize: bool = True):
        """
        Load the dataset from torchvision.

        Returns:
        - data_features (numpy.ndarray): An array containing the features of the
          dataset.
        - data_target (numpy.ndarray): An array containing the target
          variable(s) of the dataset.
        """
        if self.config.source == Source.TORCHVISION:
            # from torch.utils import data
            from torchvision import datasets, transforms

            dataset_name = self.config.path.split('/')[-1].split('.')[0]
            dataset_dir = '/'.join(self.config.path.split('/')[:-1])
            if dataset_name == 'mnist':
                train_dataset = datasets.MNIST(
                    root=dataset_dir,
                    train=True,
                    download=True,
                    transform=transforms.ToTensor(),
                )
                test_dataset = datasets.MNIST(
                    root=dataset_dir,
                    train=False,
                    download=True,
                    transform=transforms.ToTensor(),
                )
            elif dataset_name == 'fashion_mnist':
                train_dataset = datasets.FashionMNIST(
                    root=dataset_dir,
                    train=True,
                    download=True,
                    transform=transforms.ToTensor(),
                )
                test_dataset = datasets.FashionMNIST(
                    root=dataset_dir,
                    train=False,
                    download=True,
                    transform=transforms.ToTensor(),
                )
            elif dataset_name == 'cifar10':
                train_dataset = datasets.CIFAR10(
                    root=dataset_dir,
                    train=True,
                    download=True,
                    transform=transforms.ToTensor(),
                )
                test_dataset = datasets.CIFAR10(
                    root=dataset_dir,
                    train=False,
                    download=True,
                    transform=transforms.ToTensor(),
                )
            else:
                raise NotImplementedError(f'{dataset_name} not implemented yet.')
        elif self.config.source == Source.LOCAL:
            raise NotImplementedError('Local image data loading not implemented yet.')
        tr_split = self.config.train_split / (
            self.config.train_split + self.config.valid_split
        )
        tr_split_idx = int(len(train_dataset) * tr_split)
        data = {
            'train_x': jnp.array(train_dataset.data.numpy())[:tr_split_idx],
            'train_y': jnp.array(train_dataset.targets.numpy())[:tr_split_idx],
            'valid_x': jnp.array(train_dataset.data.numpy())[tr_split_idx:],
            'valid_y': jnp.array(train_dataset.targets.numpy())[tr_split_idx:],
            'test_x': jnp.array(test_dataset.data.numpy()),
            'test_y': jnp.array(test_dataset.targets.numpy()),
        }
        if self.config.datapoint_limit is not None:
            for key in data:
                data[key] = data[key][: self.config.datapoint_limit]

        if normalize:
            for key in data:
                if data[key].max() > 1:
                    data[key] = data[key] / 255.0

        if shuffle:
            data['train_x'], data['train_y'] = self._shuffle(
                data['train_x'], data['train_y']
            )
            data['valid_x'], data['valid_y'] = self._shuffle(
                data['valid_x'], data['valid_y']
            )
            data['test_x'], data['test_y'] = self._shuffle(
                data['test_x'], data['test_y']
            )

        # add channel dimension at position 1
        if len(data['train_x'].shape) == 3:
            data['train_x'] = data['train_x'][:, None, ...]
            data['valid_x'] = data['valid_x'][:, None, ...]
            data['test_x'] = data['test_x'][:, None, ...]
        return data

    def _shuffle(self, data_x: jnp.ndarray, data_y: jnp.ndarray):
        """Shuffle the data for the next dataloder iteration."""
        indices = jax.random.permutation(self.key, jnp.arange(len(data_x)))
        return data_x[indices], data_y[indices]

    def shuffle(self, split: Literal['train', 'test', 'valid'] = 'train'):
        """Shuffle the data for the next dataloder iteration."""
        if split == 'train':
            self.data['train_x'], self.data['train_y'] = self._shuffle(
                self.data['train_x'], self.data['train_y']
            )
        elif split == 'valid':
            self.data['valid_x'], self.data['valid_y'] = self._shuffle(
                self.data['valid_x'], self.data['valid_y']
            )
        else:
            self.data['test_x'], self.data['test_y'] = self._shuffle(
                self.data['test_x'], self.data['test_y']
            )

    def __len__(self):
        """Return the train length."""
        return len(self.data['train_x'])

    def _iter(
        self,
        data_x: jnp.ndarray,
        data_y: jnp.ndarray,
        batch_size: int | None,
        n_devices: int = 1,
    ):
        """Iterate over the data helper."""
        if not data_x.size:
            return
        if not batch_size:
            if n_devices > 1:  # Replicate data for multiple devices
                data_x = jnp.repeat(data_x[None, ...], n_devices, axis=0)
                data_y = jnp.repeat(data_y[None, ...], n_devices, axis=0)
            if self.config.task == Task.CLASSIFICATION:
                yield {
                    'feature': data_x,
                    'label': data_y.astype(jnp.int32).squeeze(),
                }
            else:
                yield {
                    'feature': data_x,
                    'label': data_y.squeeze(),
                }

        else:
            n_batches = len(data_x) // batch_size
            data_x = data_x[: n_batches * batch_size]  # drop last
            data_y = data_y[: n_batches * batch_size]
            splits = [
                jnp.array_split(
                    jax.random.permutation(self.key, jnp.arange(len(data_x))), n_batches
                )
                for _ in range(n_devices)
            ]
            for i in range(n_batches):
                batch_x = jnp.stack([data_x[split[i]] for split in splits])
                batch_y = jnp.stack([data_y[split[i]] for split in splits])
                if n_devices == 1:
                    batch_x = batch_x.squeeze(0)
                    batch_y = batch_y.squeeze(0)
                if self.config.task == Task.CLASSIFICATION:
                    yield {
                        'feature': batch_x,
                        'label': batch_y.squeeze().astype(jnp.int32),
                    }
                else:
                    yield {
                        'feature': batch_x,
                        'label': batch_y.squeeze(),
                    }
