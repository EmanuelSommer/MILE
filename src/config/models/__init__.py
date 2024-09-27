"""Configuration for deep learning models."""
from src.config.models.cnns import LeNetConfig, LeNettiConfig
from src.config.models.fcn import FCNConfig
from src.config.models.gpt import (
    GPTConfig, AttentionClassifierConfig, 
    EmbeddingClassifierConfig, PretrainedAttentionClassifierConfig,
)

__all__ = [
    'GPTConfig',
    'FCNConfig',
    'LeNetConfig',
    'LeNettiConfig',
    'AttentionClassifierConfig',
    'EmbeddingClassifierConfig',
    'PretrainedAttentionClassifierConfig',
]
