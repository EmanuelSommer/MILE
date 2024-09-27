"""Convolutional NN model configuration."""
from dataclasses import dataclass, field

from src.config.models.base import Activation, ModelConfig


@dataclass(frozen=True)
class LeNetConfig(ModelConfig):
    """FCN Model Configuration."""

    model: str = 'LeNet'
    activation: Activation = field(
        default=Activation.SIGMOID,
        metadata={'description': 'Activation func. for hidden layers'},
    )
    out_dim: int = field(
        default=10,
        metadata={'description': 'Output dimension of the model'},
    )
    use_bias: bool = field(
        default=True,
        metadata={'description': 'Whether to include bias terms'},
    )


@dataclass(frozen=True)
class LeNettiConfig(ModelConfig):
    """FCN Model Configuration."""

    model: str = 'LeNetti'
    activation: Activation = field(
        default=Activation.SIGMOID,
        metadata={'description': 'Activation func. for hidden layers'},
    )
    out_dim: int = field(
        default=10,
        metadata={'description': 'Output dimension of the model'},
    )
    use_bias: bool = field(
        default=True,
        metadata={'description': 'Whether to include bias terms'},
    )
