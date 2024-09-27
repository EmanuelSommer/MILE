"""DataLoader Implementations."""
import logging
import re
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
)

import jax
import jax.numpy as jnp
from datasets import (
    Dataset,
    DatasetDict,
    concatenate_datasets,
    load_dataset,
)

from src.config.data import (
    DataConfig,
    DatasetType,
    Source,
    Task,
)
from src.dataset.base import BaseLoader

if TYPE_CHECKING:
    from src.dataset.utils import Tokenizer

logger = logging.getLogger(__name__)


class TextLoader(BaseLoader):
    """Text data loader."""

    def __init__(
        self,
        config: DataConfig,
        rng: jnp.ndarray,
        tokenizer: 'Tokenizer',
        context_len: int,
        omit_freq: int = 0,
        target_len: int = 1,
    ):
        """
        Initialize the text loader.

        Args:
            config: Data configuration.
            tokenizer: Text tokenizer.
            context_len: Context length used in padding.
            omit_freq: Omit rare characters in corpus with frequency less than this.
            seed: Random seed.
        """
        super().__init__(config)
        assert self.config.data_type == DatasetType.TEXT
        self._key = rng
        self.tokenizer = tokenizer
        self.context_len = context_len
        self.target_len = target_len
        self.data = self.load_data()
        _text_only = not (self.config.target_column or self.config.features)
        if omit_freq:
            self.data = omit_rare_chars(self.data, omit_freq)
        if self.tokenizer.needs_training:
            self.tokenizer.train(''.join(self.data['feature']))
        if _text_only:
            # Text-only dataset
            self.data = self.data.map(
                lambda x: self.tokenize(x), batched=True, batch_size=None
            )
        else:
            # Text and target dataset
            self.data = self.data.map(
                lambda x: self.tokenize(x, length=self.context_len),
                batched=True,
                batch_size=None,
            )
            if isinstance(self.data['label'][0], str):
                self.data = self.data.class_encode_column('label')
        self.data.set_format(type='jax')

        # Stack features and labels
        self.data = jnp.column_stack([self.data['feature'], self.data['label']])
        # Shuffle and limit Data
        self.data = self._shuffle(self.data)
        if self.config.datapoint_limit:
            self.data = self.data[: self.config.datapoint_limit]

        self.data_train = self.data[: int(len(self.data) * self.config.train_split)]
        logger.info('Class Balance Train: %s', jnp.bincount(self.data_train[:, -1]))
        self.data_valid = self.data[
            int(len(self.data) * self.config.train_split) : int(
                len(self.data) * (self.config.train_split + self.config.valid_split)
            )
        ]
        logger.info('Class Balance Valid: %s', jnp.bincount(self.data_valid[:, -1]))
        self.data_test = self.data[
            int(len(self.data) * (self.config.train_split + self.config.valid_split)) :
        ]
        logger.info('Class Balance Test: %s', jnp.bincount(self.data_test[:, -1]))

    def __str__(self):
        """Return informative string representation of the class."""
        return (
            super().__str__() + '\n'
            f' | Context Length: {self.context_len}\n'
            f' | Vocabulary Size: {self.tokenizer.vocab_size}\n'
            f' | Target Length: {self.target_len}\n'
            f' | Train: {len(self.data_train)}\n'
            f' | Valid: {len(self.data_valid)}\n'
            f' | Test: {len(self.data_test)}'
        )

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

    @property
    def key(self):
        """Return the next rng key for the dataloader."""
        self._key, key = jax.random.split(self._key)
        return key

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

    def tokenize(self, batch: dict[str,], length: int | None = None):
        """Transform for dataloader."""
        if not length:
            batch['feature'] = self.tokenizer.encode_batch(batch['feature'])
        else:
            batch['feature'] = self.tokenizer.encode_batch_with_padding(
                batch['feature'], length
            )
        return batch

    def __len__(self):
        """Return the length of the data."""
        return len(self.data)

    def load_data(self) -> Dataset:
        """Load text data."""
        if self.config.source == Source.LOCAL:
            if self.config.path.endswith('.txt'):
                data = Dataset.from_text(self.config.path, sample_by='document')
                assert not (
                    self.config.target_column or self.config.features
                ), 'Text dataset should not have target_column or features.'
            else:
                raise NotImplementedError
        elif self.config.source == Source.HUGGINGFACE:
            path = self.config.path.split('/')
            if len(path) > 1:
                data = load_dataset(path=path[-2], name='/'.join(path[-1:]))
            else:
                data = load_dataset(path=path[0])
            if self.config.target_column and self.config.target_column != 'label':
                data = data.rename_column(self.config.target_column, 'label')
            if self.config.features and self.config.features[0] != 'feature':
                data = data.rename_column(self.config.features[0], 'feature')
            data = data.select_columns(['feature', 'label'])
        else:
            raise NotImplementedError
        if isinstance(data, DatasetDict):
            data = concatenate_datasets(
                [data[key] for key in data.keys() if key != 'unsupervised']
            )
        return data

    def iter(
        self,
        split: Literal['train', 'test', 'valid'],
        batch_size: int = 0,
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

    def _iter(self, data: jnp.ndarray, batch_size: int, n_devices: int = 1):
        if not data.size:
            return
        split_col = data.shape[-1] - self.target_len
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


# Utility functions
def omit_rare_chars(data: Dataset, min_freq: int = 100):
    """Omit rare characters."""
    omit = get_rare_chars(''.join(data['feature']), min_freq)
    data = data.map(lambda x: remove_chars(x, omit), batched=True, batch_size=None)
    return data


def get_rare_chars(text: str, min_freq: int = 100):
    """Find rare characters in the text."""
    chars = sorted(set(text))
    return [c for c in chars if text.count(c) < min_freq]


def remove_chars(batch: dict[str, Any], omit: list[str] = []):
    """Remove characters from the dataset."""
    pattern = rf'({"|".join(re.escape(v) for v in omit)})'
    batch['feature'] = [re.sub(pattern, '', t) for t in batch['feature']]
    return batch
