"""Basic ConvNN building blocks and models."""
from typing import Callable

import jax.numpy as jnp
from flax import linen as nn

from src.config.models.cnns import LeNetConfig, LeNettiConfig


class LeNet(nn.Module):
    """
    Implementation of LeNet.

    Args:
        config (LeNetConfig): The configuration for the model.
    """

    config: LeNetConfig

    def setup(self):
        """Initialize the fully connected neural network."""
        self.core = LeNetCore(
            activation=self.config.activation.flax_activation,
            out_dim=self.config.out_dim,
            use_bias=self.config.use_bias,
        )

    def __call__(self, x: jnp.ndarray):
        """Forward pass."""
        return self.core(x)


class LeNetCore(nn.Module):
    """Core Implementation of LeNet."""

    activation: Callable
    out_dim: int
    use_bias: bool

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        """
        Forward pass.

        Args:
            x (jnp.ndarray): The input data of
            shape (batch_size, channels, height, width).
        """
        x = x.transpose((0, 2, 3, 1))
        x = nn.Conv(
            features=6, kernel_size=(5, 5), strides=(1, 1), padding=2, name='conv1'
        )(x)
        x = self.activation(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2), padding='VALID')
        x = nn.Conv(
            features=16, kernel_size=(5, 5), strides=(1, 1), padding=0, name='conv2'
        )(x)
        x = self.activation(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2), padding='VALID')
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(features=120, use_bias=self.use_bias, name='fc1')(x)
        x = self.activation(x)
        x = nn.Dense(features=84, use_bias=self.use_bias, name='fc2')(x)
        x = self.activation(x)
        x = nn.Dense(features=self.out_dim, use_bias=self.use_bias, name='fc3')(x)
        return x


class LeNetti(nn.Module):
    """
    A super simple LeNet version.

    Args:
        config (LeNettiConfig): The configuration for the model.
    """

    config: LeNettiConfig

    def setup(self):
        """Initialize the fully connected neural network."""
        self.core = LeNettiCore(
            activation=self.config.activation.flax_activation,
            out_dim=self.config.out_dim,
            use_bias=self.config.use_bias,
        )

    def __call__(self, x: jnp.ndarray):
        """Forward pass."""
        return self.core(x)


class LeNettiCore(nn.Module):
    """Core Implementation of LeNetti."""

    activation: Callable
    out_dim: int
    use_bias: bool

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        """
        Forward pass.

        Args:
            x (jnp.ndarray): The input data of
            shape (batch_size, channels, height, width).
        """
        x = x.transpose((0, 2, 3, 1))
        x = nn.Conv(
            features=1, kernel_size=(3, 3), strides=(1, 1), padding=2, name='conv1'
        )(x)
        x = self.activation(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(features=8, use_bias=self.use_bias, name='fc1')(x)
        x = self.activation(x)
        x = nn.Dense(features=8, use_bias=self.use_bias, name='fc2')(x)
        x = self.activation(x)
        x = nn.Dense(features=8, use_bias=self.use_bias, name='fc3')(x)
        x = self.activation(x)
        x = nn.Dense(features=self.out_dim, use_bias=self.use_bias, name='fc4')(x)
        return x
