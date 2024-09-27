"""Basic NN building blocks."""
from typing import Callable

import jax.numpy as jnp
from flax import linen as nn


class FullyConnected(nn.Module):
    """
    Fully connected layers.

    Args:
        hidden_sizes (Sequence[int]): The hidden layer sizes.
        activation (Optional[Callable]): The activation function. Defaults to nn.relu.
        use_bias (bool): Whether to use bias. Defaults to True.
        last_layer_activation (Optional[Callable]): The activation function for the
          last layer. Defaults to None.
        blockid (Optional[str]): The block id. Defaults to None.
        dtype (Dtype): dtype used in computation. Defaults to jnp.float32.

    """

    hidden_sizes: tuple[int, ...]
    activation: Callable
    use_bias: bool = True
    last_layer_activation: Callable | None = None
    blockid: str | None = None
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        """Forward pass."""
        if self.blockid is None:
            blockid = ''
        else:
            blockid = f'{self.blockid}_'
        for i, hidden_size in enumerate(self.hidden_sizes):
            x = nn.Dense(
                features=hidden_size,
                dtype=self.dtype,
                use_bias=self.use_bias,
                name=f'{blockid}layer{i}',
            )(x)
            if i < len(self.hidden_sizes) - 1:
                x = self.activation(x)
            else:
                if self.last_layer_activation is not None:
                    x = self.last_layer_activation(x)
        return x


class MaskedMultiHeadSelfAttention(nn.Module):
    """Masked Multi-Head Attention Module."""

    n_heads: int
    qkv_dim: int
    bias: bool
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool = False):
        """Forward Pass."""
        _, _, C = x.shape
        mask = nn.make_causal_mask(x[:, :, 0], dtype=jnp.bool_)
        out = nn.MultiHeadDotProductAttention(
            num_heads=self.n_heads,
            qkv_features=self.qkv_dim,
            dtype=self.dtype,
            use_bias=self.bias,
            deterministic=deterministic,
            out_features=C,
        )(x, mask=mask)
        return out


class TokenEmbedding(nn.Module):
    """Embedding Layer for Tokens with optional Positional Encoding."""

    vocab_size: int
    emb_size: int
    dtype: jnp.dtype
    pos_size: int | None = None

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        """Forward Pass."""
        embed = nn.Embed(
            num_embeddings=self.vocab_size,
            features=self.emb_size,
            dtype=self.dtype,
            name='Embedding',
        )(x)
        if self.pos_size:
            pos = jnp.arange(x.shape[1]).reshape(1, -1)
            pos = nn.Embed(
                num_embeddings=self.pos_size,
                features=self.emb_size,
                dtype=self.dtype,
                name='PositionEmbedding',
            )(pos)
            embed += pos
        return embed


class PretrainedTokenEmbedding(nn.Module):
    """Embedding Layer for Tokens with Pretrained Weights."""

    vocab_size: int
    emb_size: int
    dtype: jnp.dtype
    pretrained_weights_path: str
    pos_size: int | None = None

    def setup(self):
        """Load the pretrained weights."""
        self.pretrained_weights = jnp.load(self.pretrained_weights_path)

        if self.pos_size:
            pos_emb_path = self.pretrained_weights_path.replace('emb', 'pos_emb')
            self.position_embedding = jnp.load(pos_emb_path)

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        """Forward Pass."""
        # Use the pretrained weights directly
        embed = jnp.take(self.pretrained_weights, x, axis=0)
        if self.pos_size:
            pos = jnp.arange(x.shape[1]).reshape(1, -1)
            pos = jnp.take(self.position_embedding, pos, axis=0)
            embed += pos
        return embed
