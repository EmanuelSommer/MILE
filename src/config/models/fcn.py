"""Fully connected NN model configuration."""
from dataclasses import dataclass, field

from src.config.models.base import Activation, ModelConfig


@dataclass(frozen=True)
class FCNConfig(ModelConfig):
    """FCN Model Configuration."""

    model: str = 'FCN'
    hidden_structure: list[int] = field(
        hash=False,
        default_factory=lambda: [10, 10],
        metadata={
            'description': 'Width (+ number) of hidden layers',
            'searchable': True,
        },
    )
    activation: Activation = field(
        default=Activation.RELU,
        metadata={
            'description': 'Activation func. for hidden layers',
            'searchable': True,
        },
    )

    use_bias: bool = field(
        default=True, metadata={'description': 'Whether to include bias terms'}
    )
