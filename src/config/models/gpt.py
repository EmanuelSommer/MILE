"""GPT model configuration."""
from dataclasses import dataclass, field

from src.config.models.base import FloatPrecision, ModelConfig


@dataclass(frozen=True)
class GPTConfig(ModelConfig):
    """GPT Model Configuration."""

    model: str = 'GPT'

    vocab_size: int = field(
        default=1000, metadata={'description': 'Vocabulary size for tokenization'}
    )
    context_len: int = field(
        default=8, metadata={'description': 'Maximum context length for predictions'}
    )
    emb_size: int = field(default=256, metadata={'description': 'Token Embedding size'})
    n_blocks: int = field(
        default=6, metadata={'description': 'Number of transformer blocks'}
    )
    n_heads: int = field(
        default=8, metadata={'description': 'Number of attention heads'}
    )
    qkv_dim: int = field(
        default=512, metadata={'description': 'Query, Key, Value dimension'}
    )
    bias: bool = field(
        default=False, metadata={'description': 'Whether to include bias terms'}
    )
    dropout: float = field(default=0.1, metadata={'description': 'Dropout rate'})

    dtype: FloatPrecision = field(
        default=FloatPrecision.FLOAT32, metadata={'description': 'FP precision'}
    )


@dataclass(frozen=True)
class AttentionClassifierConfig(GPTConfig):
    """Attention Classifier Configuration."""

    model: str = 'AttentionClassifier'
    n_classes: int = field(
        default=2, metadata={'description': 'Number of classes for classification'}
    )
    projection_dim: list[int] = field(
        hash=False,
        default_factory=lambda: [32],
        metadata={'description': 'Projection dimensions'},
    )


@dataclass(frozen=True)
class PretrainedAttentionClassifierConfig(AttentionClassifierConfig):
    """Pretrained Attention Classifier Configuration."""

    model: str = 'PretrainedAttentionClassifier'
    emb_path: str = field(
        default=None, metadata={'description': 'Path to pretrained embeddings'}
    )


@dataclass(frozen=True)
class EmbeddingClassifierConfig(AttentionClassifierConfig):
    """Embedding Classifier Configuration."""

    model: str = 'EmbeddingClassifier'
