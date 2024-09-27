"""various metrics for evaluating the performance of a probabilistic model."""
import pickle
import warnings
from dataclasses import Field
from pathlib import Path
from typing import Callable, Sequence

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpyro.diagnostics as numpyro_diag
from flax.struct import PyTreeNode
from numpyro.distributions import Categorical, Normal

from src.config.data import Task


class Metrics(PyTreeNode):
    """Metrics base class."""

    step: jnp.ndarray

    def __len__(self):
        """Len of the metrics."""
        return self.shape[-1]

    @property
    def n_chains(self):
        """Number of chains."""
        if len(self.shape) < 2:
            return 1
        return self.shape[0]

    @property
    def shape(self):
        """Shape of the metrics."""
        return self.step.shape

    def __getitem__(self, index: int | str):
        """Get item by index or key."""
        if isinstance(index, str):
            return getattr(self, index)
        return self.replace(
            **{k: getattr(self, k)[index] for k, _ in self.__dict__.items()}
        )

    def pad(self, length: int):
        """Pad the metrics with nan."""
        if length > self.shape[-1]:
            return self.replace(
                **{
                    k: jnp.pad(
                        getattr(self, k),
                        pad_width=(
                            jnp.array((0, 0, 0, length - self.shape[-1])).reshape(2, 2)
                        ),
                        mode='constant',
                        constant_values=jnp.nan,
                    )
                    for k, _ in self.__dict__.items()
                }
            )
        return self

    @classmethod
    def empty(cls):
        """Create an empty metrics object."""
        em = jnp.empty(shape=(1, 0))
        return cls(**{k: em for k, _ in cls.__dataclass_fields__.items()})

    @classmethod
    def vstack(cls, metrics: Sequence['Metrics']):
        """Stack the metrics vertically."""
        if len(set([m.shape[-1] for m in metrics])) != 1:
            # pad with nan
            max_len = max([m.shape[-1] for m in metrics])
            for i, m in enumerate(metrics):
                if m.shape[-1] < max_len:
                    metrics[i] = m.pad(max_len)
        return cls._stack(metrics, jnp.vstack)

    @classmethod
    def cstack(cls, metrics: Sequence['Metrics']):
        """Merge metrics columnwise."""
        return cls._stack(metrics, jnp.column_stack)

    @classmethod
    def _stack(
        cls,
        metrics: Sequence['Metrics'],
        stack_fn: Callable[[Sequence[jax.Array]], jax.Array],
    ):
        """Stack the metrics."""
        if not metrics:
            return cls.empty()
        return cls(
            **{
                k: stack_fn([getattr(m, k) for m in metrics])
                for k, _ in metrics[0].__dict__.items()
            }
        )

    @property
    def is_empty(self):
        """Check if the metrics is empty."""
        return self.step.size == 0

    def savez(self, path: str):
        """Save the metrics to a npz file."""
        jnp.savez(path, **self.__dict__)

    def _save_plots(self, path: str):
        """Save the metrics plots."""
        if self.is_empty:
            return
        if not Path(path).exists():
            Path(path).mkdir(parents=True)
        for key in self.__dict__:
            if (
                isinstance(getattr(self, key), jnp.ndarray)
                and len(getattr(self, key).shape) == 2
                and key != 'step'
            ):
                for i in range(self.n_chains):
                    last_id = (~jnp.isnan(getattr(self, key)[i, :])).sum() - 1
                    if len(self) == 1:
                        plt.scatter(
                            1,
                            getattr(self, key)[i, :],
                            label=f'chain_{i}_{getattr(self, key)[i, last_id]:.2f}',
                        )
                    else:
                        plt.plot(
                            getattr(self, key)[i, :],
                            label=f'chain_{i}_{getattr(self, key)[i, last_id]:.2f}',
                        )
                plt.legend()
                plt.xlabel('step')
                plt.ylabel(key)
                plt.savefig(f'{path}/{key}.png')
                plt.close()


class MetricsStore(PyTreeNode):
    """Metrics store class."""

    train: Metrics
    valid: Metrics
    test: Metrics

    def save(self, path: str, save_plots: bool = False):
        """Save the metrics store to a file."""
        parent = Path(path).parent
        if not Path(parent).exists():
            Path(parent).mkdir(parents=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        if save_plots:
            self._save_plots(parent)

    @classmethod
    def vstack(cls, metrics: Sequence['MetricsStore']):
        """Stack the metrics store vertically."""
        return cls(
            **{
                k: v.vstack([m[k] for m in metrics])
                for k, v in metrics[0].__dict__.items()
                if isinstance(v, Metrics)
            }
        )

    def _save_plots(self, path: str):
        """Save the metrics store plots."""
        for key in self.__dict__:
            if isinstance(getattr(self, key), Metrics):
                getattr(self, key)._save_plots(f'{path}/{key}')

    @classmethod
    def load(cls, path: str) -> 'MetricsStore':
        """Load the metrics store from a file."""
        with open(path, 'rb') as f:
            return pickle.load(f)

    @classmethod
    def load_batch(cls, paths: Sequence[str]):
        """Load a batch of metrics stores."""
        metrics = [cls.load(p) for p in paths]
        cls_name = metrics[0].train.__class__

        return cls(
            train=cls_name.vstack([m.train for m in metrics]),
            valid=cls_name.vstack([m.valid for m in metrics]),
            test=cls_name.vstack([m.test for m in metrics]),
        )

    def __getitem__(self, key: str):
        """Get item by key."""
        return getattr(self, key)

    @classmethod
    def empty(cls):
        """Create an empty metrics store."""
        return cls(
            **{
                k: v.type.empty()
                for k, v in cls.__dataclass_fields__.items()
                if isinstance(v, Field) and v.type is Metrics
            }
        )


class ClassificationMetrics(Metrics):
    """Classification metrics."""

    cross_entropy: jnp.ndarray
    accuracy: jnp.ndarray


class RegressionMetrics(Metrics):
    """Regression metrics."""

    nlll: jnp.ndarray
    rmse: jnp.ndarray


def rank_normalize_array(samples: jnp.ndarray) -> jnp.ndarray:
    """
    Rank normalization of a JAX array.

    Parameters:
    ----------
    samples : jnp.ndarray
        Jax array of shape (n_chains, n_samples, ...).

    Returns:
    ----------
    jnp.ndarray:
        Rank normalized array.
    """
    n_samples = jnp.prod(jnp.array(samples.shape))
    # get the overall ranks
    ranks = jax.scipy.stats.rankdata(samples, axis=None).reshape(samples.shape)
    tmp = (ranks - 0.375) / (n_samples + 0.25)
    return jax.scipy.stats.norm.ppf(tmp)


def pointwise_lppd(lvals: jnp.ndarray, y: jnp.ndarray, task: Task) -> jnp.ndarray:
    """Calculate the pointwise log predictive probability density.

    Parameters:
    ----------
    lvals : jnp.ndarray
        Logits or mean and variance of the predictive distribution.
        Either shape (n_chains, n_samples, n_obs, 2) for regression or
        (n_chains, n_samples, n_obs, n_classes) for classification.
    y : jnp.ndarray
        target values of shape (n_obs).
    task : Task
        Type of learning target (classification or regression).

    Notes:
    ----------
    If len(lvals.shape) == 3 a chain dimension is added.
    If len(lvals.shape) == 2 a chain and sample dimension is added.

    Returns:
    ----------
    jnp.ndarray:
        Pointwise log predictive probability density.
    """
    if len(lvals.shape) >= 4:
        lvals.reshape(*lvals.shape[:2], -1)
    if len(lvals.shape) == 3:
        lvals = lvals.reshape(1, *lvals.shape)
    elif len(lvals.shape) == 2:
        lvals = lvals.reshape(1, 1, *lvals.shape)

    if task == Task.REGRESSION:
        lppd_pointwise = jnp.stack(
            [
                Normal(
                    loc=x[:, :, 0],
                    scale=jnp.exp(x[:, :, 1]).clip(min=1e-6, max=1e6),
                )
                .log_prob(y)
                .squeeze()
                for x in lvals
            ]
        )
    elif task == Task.CLASSIFICATION:
        lppd_pointwise = jnp.stack(
            [Categorical(logits=x).log_prob(y).squeeze() for x in lvals]
        )
    return lppd_pointwise


def lppd(lppd_pointwise: jnp.ndarray) -> jnp.ndarray:
    """Calculate the log predictive probability density.

    Parameters:
    ----------
    lppd_pointwise : jnp.ndarray
        Pointwise lppd of shape (n_chains, n_samples, n_obs).

    Returns:
    ----------
    jnp.ndarray:
        Log predictive probability density.
    """
    b = 1 / jnp.prod(jnp.array(lppd_pointwise.shape[:-1]))
    axis = tuple(range(len(lppd_pointwise.shape) - 1))
    return jax.scipy.special.logsumexp(lppd_pointwise, b=b, axis=axis).mean()


def GaussianNLLLoss(x: jnp.ndarray, mu: jnp.ndarray, sigma: jnp.ndarray):
    """Compute the Gaussian negative log-likelihood loss.

    Parameters:
    ----------
    x : jnp.array
        The target values of shape (n_obs).
    mu : jnp.array
        The predicted mean values of shape (..., n_obs).
    sigma : jnp.array
        The predicted standard deviation values of shape (..., n_obs).

    Returns:
    ----------
    jnp.array:
        The computed loss.
    """
    return jnp.log(
        (jnp.sqrt(2 * jnp.pi) * jnp.clip(sigma, 1e-5))
        + (x - mu) ** 2 / (2 * jnp.clip(sigma, 1e-5) ** 2)
    )


def SELoss(x: jnp.ndarray, mu: jnp.ndarray):
    """Compute the squared error loss.

    Parameters:
    ----------
    x : jnp.array
        The target values of shape (n_obs).
    mu : jnp.array
        The predicted mean values of shape (..., n_obs).

    Returns:
    ----------
    jnp.array:
        The computed loss.
    """
    return (x - mu) ** 2


def between_chain_var(x: jnp.ndarray) -> jnp.ndarray:
    """Calculate the between chain variance of an array.

    Parameters:
    ----------
    x : jnp.array
        Array with shape (n_chains, n_samples, ...).

    Returns:
    ----------
    jnp.array:
        Between chain variance.
    """
    return x.mean(axis=1).var(axis=0, ddof=1)


def within_chain_var(x: jnp.ndarray) -> jnp.ndarray:
    """Calculate the within chain variance of an array.

    Parameters:
    ----------
    x : jnp.array
        Array with shape (n_chains, n_samples, ...).

    Returns:
    ----------
    jnp.array:
        Within chain variance.
    """
    return x.var(axis=1, ddof=1).mean(axis=0)


def effective_sample_size(x: jnp.ndarray, rank_normalize: bool = True) -> jnp.ndarray:
    """Calculate the effective sample size of an array.

    Parameters:
    ----------
    x : jnp.array
        Array with shape (n_chains, n_samples, ...).
    rank_normalize : bool
        Whether to rank normalize the samples before calculating the ESS.

    Returns:
    ----------
    jnp.array:
        Effective sample size.
    """
    if rank_normalize:
        x = jnp.apply_along_axis(
            rank_normalize_array, 0, x.reshape(-1, *x.shape[2:])
        ).reshape(x.shape)
    return jnp.stack([numpyro_diag.effective_sample_size(x[None, ...]) for x in x])


def running_mean(x: jnp.ndarray, axis: int):
    """Compute running mean along a specified axis.

    Parameters:
    ----------
    x : jnp.array
        Input array.
    axis : int
        Axis along which the running mean is computed.

    Returns:
    ----------
    jnp.array:
        Running mean.
    """
    cumsum = jnp.cumsum(x, axis=axis)
    count = jnp.arange(1, x.shape[axis] + 1)[None, ..., None]
    return cumsum / count


def running_lppd(lppd_pointwise: jnp.ndarray):
    """
    Calculate the running log predictive probability density on n_samples axis.

    Parameters:
    ----------
    lppd_pointwise : jnp.array
        Pointwise lppd of shape (n_chains, n_samples, n_obs).

    Returns:
    ----------
    jnp.array:
        Running log predictive probability density of shape (n_chains, n_samples).
    """
    return (
        jnp.log(running_mean(jnp.exp(lppd_pointwise), axis=-2))
        .mean(axis=-1)
        .mean(axis=0)
    )


def gelman_split_r_hat(
    samples: jnp.ndarray,
    n_splits: int,
    rank_normalize: bool = True,
) -> jnp.ndarray:
    """Calculate the split Gelman-Rubin R-hat statistic for samples of MCMC chains.

    Parameters:
    ----------
    samples: jnp.ndarray
        MCMC chains as a JAX array with the shape (n_chains, n_samples, ...).
    n_splits: int
        Number of splits to split each chain into.
    rank_normalize: bool
        Whether to rank normalize the samples before calculating the R-hat.

    Returns:
    ----------
    rhat: jnp.ndarray
        R-hat statistic for each parameter (the other dimensions) as a JAX array.
    """
    n_chains = samples.shape[0]
    n_samples = samples.shape[1] / n_splits

    if (n_samples % 1) != 0:
        raise ValueError('Number of samples must be divisible by n_splits')

    if n_samples < 50:
        warnings.warn(
            message='Number of samples should be at least 50x the number of splits',
            category=UserWarning,
        )

    splits_total = n_chains * n_splits

    if rank_normalize:
        samples = jnp.apply_along_axis(
            rank_normalize_array, 0, samples.reshape(-1, *samples.shape[2:])
        ).reshape(samples.shape)

    splits = samples.reshape(splits_total, -1, *samples.shape[2:])
    wcv = within_chain_var(splits)
    bcv = between_chain_var(splits)
    numerator = ((n_samples - 1) / n_samples) * wcv + bcv
    rhat = jnp.sqrt(numerator / wcv)
    return rhat


def split_chain_r_hat(
    samples: jnp.ndarray,
    n_splits: int,
    rank_normalize: bool = True,
) -> jnp.ndarray:
    """Calculate the split chain R-hat statistic for samples of MCMC chains.

    Parameters
    ----------
    samples : jnp.ndarray
        MCMC chains as an array with the shape (n_chains, n_samples, ...).
    n_splits : int
        Number of splits to split each chain into.
    rank_normalize : bool
        Whether to rank normalize the samples before calculating the R-hat.

    Returns
    -------
    split_chain_rhat : jnp.ndarray
        Split R-hat statistic for each chain (n_chains, ...) as a JAX array.
    """
    return jnp.stack(
        [
            gelman_split_r_hat(chain[None, ...], n_splits, rank_normalize)
            for chain in samples
        ]
    )
