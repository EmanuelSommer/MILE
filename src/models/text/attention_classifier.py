"""Attention Classfier for Binary Text Classification."""
import flax.linen as nn
import jax
import jax.numpy as jnp

from src.config.models.gpt import (
    AttentionClassifierConfig,
    EmbeddingClassifierConfig,
    PretrainedAttentionClassifierConfig,
)
from src.flax_building_blocks.basic import (
    PretrainedTokenEmbedding,
    TokenEmbedding,
)


class AttentionClassifier(nn.Module):
    """Attention-based classifier."""

    config: AttentionClassifierConfig

    @nn.compact
    def __call__(self, x: jnp.ndarray, pad_id: int = 0, train: bool = True):
        """Forward pass."""
        dtype = self.config.dtype.flax_dtype
        _, T = x.shape
        assert T == self.config.context_len, 'Context length mismatch'
        # Attention Mask
        bool_mask = x != pad_id
        mask = jax.vmap(lambda x: jnp.outer(x, x))(bool_mask)[:, None, :, :]
        # Embedding
        x = TokenEmbedding(
            vocab_size=self.config.vocab_size,
            emb_size=self.config.emb_size,
            dtype=dtype,
            pos_size=self.config.context_len,
        )(x)
        # x = nn.LayerNorm(dtype=dtype)(x)
        # x = nn.Dropout(self.config.dropout, deterministic=not train)(x)
        out = nn.MultiHeadDotProductAttention(
            num_heads=self.config.n_heads,
            qkv_features=self.config.qkv_dim,
            out_features=self.config.emb_size,
            dtype=dtype,
            use_bias=self.config.bias,
            name='MDPA',
        )(x, mask=mask)
        # out = nn.LayerNorm(dtype=dtype)(out)

        # Average Pooling
        out = jnp.mean(out, axis=1)

        # Projection
        for proj_id, proj_dim in enumerate(self.config.projection_dim):
            out = nn.Dense(
                proj_dim,
                dtype=dtype,
                use_bias=self.config.bias,
                name=f'projection_{proj_id}',
            )(out)
            out = nn.gelu(out)
            # out = nn.Dropout(self.config.dropout, deterministic=not train)(out)

        # Classification
        logits = nn.Dense(
            self.config.n_classes,
            dtype=dtype,
            use_bias=self.config.bias,
            name='classifier',
        )(out)
        return logits
    

class PretrainedAttentionClassifier(nn.Module):
    """Attention-based classifier."""

    config: PretrainedAttentionClassifierConfig

    @nn.compact
    def __call__(self, x: jnp.ndarray, pad_id: int = 0, train: bool = True):
        """Forward pass."""
        dtype = self.config.dtype.flax_dtype
        _, T = x.shape
        assert T == self.config.context_len, 'Context length mismatch'
        # Attention Mask
        bool_mask = x != pad_id
        mask = jax.vmap(lambda x: jnp.outer(x, x))(bool_mask)[:, None, :, :]
        # Embedding
        x = PretrainedTokenEmbedding(
            vocab_size=self.config.vocab_size,
            emb_size=self.config.emb_size,
            dtype=dtype,
            pretrained_weights_path=self.config.emb_path,
            pos_size=self.config.context_len,
        )(x)
        # x = nn.LayerNorm(dtype=dtype)(x)
        # x = nn.Dropout(self.config.dropout, deterministic=not train)(x)
        out = nn.MultiHeadDotProductAttention(
            num_heads=self.config.n_heads,
            qkv_features=self.config.qkv_dim,
            out_features=self.config.emb_size,
            dtype=dtype,
            use_bias=self.config.bias,
            name='MDPA',
        )(x, mask=mask)
        # out = nn.LayerNorm(dtype=dtype)(out)

        # Average Pooling
        out = jnp.mean(out, axis=1)

        # Projection
        for proj_id, proj_dim in enumerate(self.config.projection_dim):
            out = nn.Dense(
                proj_dim,
                dtype=dtype,
                use_bias=self.config.bias,
                name=f'projection_{proj_id}',
            )(out)
            out = nn.gelu(out)
            # out = nn.Dropout(self.config.dropout, deterministic=not train)(out)

        out = nn.gelu(out)
        # out = nn.Dropout(self.config.dropout, deterministic=not train)(out)

        # Classification
        logits = nn.Dense(
            self.config.n_classes,
            dtype=dtype,
            use_bias=self.config.bias,
            name='classifier',
        )(out)
        return logits



class EmbeddingClassifier(nn.Module):
    """Attention-based classifier."""

    config: EmbeddingClassifierConfig

    @nn.compact
    def __call__(self, x: jnp.ndarray, attn_mask: jnp.ndarray):
        """Forward pass."""
        dtype = self.config.dtype.flax_dtype

        out = nn.MultiHeadDotProductAttention(
            num_heads=self.config.n_heads,
            qkv_features=self.config.qkv_dim,
            out_features=self.config.emb_size,
            dtype=dtype,
            use_bias=self.config.bias,
            name='MDPA',
        )(x, mask=attn_mask)
        # out = nn.LayerNorm(dtype=dtype)(out)

        # Average Pooling
        out = jnp.mean(out, axis=1)

        # Projection
        out = nn.Dense(
            2 * self.config.emb_size,
            dtype=dtype,
            use_bias=self.config.bias,
            name='projection',
        )(out)
        out = nn.gelu(out)
        # out = nn.Dropout(self.config.dropout, deterministic=not train)(out)

        # Classification
        logits = nn.Dense(
            self.config.n_classes,
            dtype=dtype,
            use_bias=self.config.bias,
            name='classifier',
        )(out)
        return logits
