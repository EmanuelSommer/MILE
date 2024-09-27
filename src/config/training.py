"""Training Configuration."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.config.base import BaseConfig, BaseStrEnum
from src.config.sampler import SamplerConfig
from src.config.warmstart import WarmStartConfig


class TokenizerName(BaseStrEnum):
    """Tokenizers for LLMs."""

    BPE = 'BPE'
    BERT = 'BERT'
    SingleChar = 'SingleChar'
    CustomBPE = 'CustomBPE'

    def get_tokenizer(self, **parameters):
        """Get tokenizer instance.

        Notes
        -----
        Implementation is still pending.
        """
        from src.dataset.utils import tokenizers

        return tokenizers[self.value](**parameters)


@dataclass(frozen=True)
class TokenizerConfig(BaseConfig):
    """Tokenizer Configuration Class."""

    tokenizer: TokenizerName = field(
        default=TokenizerName.CustomBPE,
        metadata={'description': 'Tokenizer to use for TEXT data.'},
    )
    parameters: dict[str, Any] = field(
        default_factory=dict,
        metadata={'description': 'Tokenizer parameters.'},
    )

    def get_tokenizer(self):
        """Get tokenizer instance."""
        return self.tokenizer.get_tokenizer(**self.parameters)


@dataclass(frozen=True)
class TrainingConfig(BaseConfig):
    """Training Configuration Class."""

    warmstart: WarmStartConfig = field(
        default_factory=WarmStartConfig,
        metadata={'description': 'Configuration for Frequentist pretraining as warmup'},
    )
    sampler: SamplerConfig = field(
        default_factory=SamplerConfig,
        metadata={'description': 'Sampler Configuration for Training.'},
    )
    tokenizer: TokenizerConfig = field(
        default_factory=TokenizerConfig,
        metadata={'description': 'Tokenizer Configuration (is not yet implemented!!)'},
    )

    @property
    def optimizer(self):
        return self.warmstart.optimizer

    @property
    def prior(self):
        return self.sampler.prior

    @property
    def warmstart_path(self):
        if self.warmstart.warmstart_exp_dir:
            return Path(self.warmstart.warmstart_exp_dir)

    @property
    def has_warmstart(self):
        return self.warmstart.include
