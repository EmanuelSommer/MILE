"""Kernels used for training in the src."""
from typing import Callable

from blackjax import (
    hmc,
    mclmc,
    nuts,
)
from blackjax.base import SamplingAlgorithm

__all__ = ['nuts', 'hmc', 'mclmc']


KERNELS: dict[str, Callable[..., SamplingAlgorithm]] = {
    'nuts': nuts,
    'hmc': hmc,
    'mclmc': mclmc,
}

WARMUP_KERNELS: dict[str, Callable[..., SamplingAlgorithm]] = {}
