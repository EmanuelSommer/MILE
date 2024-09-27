"""Visual analysis of sampling results."""
import os
from pathlib import Path
from typing import Literal

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

import src.inference.metrics as sandbox_metrics
from src.config.data import Task
from src.inference.utils import count_chains, count_samples
from src.types import ParamTree
from src.utils import get_by_path, get_flattened_keys


def plot_param_movement(samples: ParamTree, random_n: int = 5):
    """Visualize randomly selected single parameter movement across chains.

    Parameters
    ----------
    `samples`: ParamTree
        Paramter Tree of samples to visualize trace plots `(n_chains, n_samples, ...)`.
    `random_n`: int
        Number of randomly selected parameters to visualize.
    """
    key = jax.random.PRNGKey(42)
    n_chains = count_chains(samples)
    n_samples = count_samples(samples)
    n_row = -(n_chains // -2)
    n_col = 2
    for name in get_flattened_keys(samples):
        fig, axs = plt.subplots(n_row, n_col, figsize=(5 * n_col, 3 * n_row))
        # Set axis label
        fig.supxlabel('Sampling Step')
        fig.supylabel('Parameter Value')
        fig.suptitle(f'Parameter movement for {name} across {n_chains} chains', size=10)
        for ax, i in zip(axs.flat, range(n_chains)):
            chain = jax.tree.map(lambda x: x[i], samples)
            params = get_by_path(chain, name.split('.')).reshape(n_samples, -1)
            idx = jax.random.permutation(key, jnp.arange(params.shape[1]))[:random_n]
            ax.set_title(f'Chain {i + 1}', size=8)
            for y in idx:
                ax.plot(params[:, y])


def plot_param_hist(samples: ParamTree, hist_limit: int | None = 10000):
    """Visualize histograms of parameters across chains for each architecture layer.

    Parameters
    ----------
    `samples` : ParamTree
        Paramter Tree of samples to visualize histograms.

    `hist_limit` : int
        Maximum number of observations to plot in the histogram, to cap memory usage.
    """
    key = jax.random.PRNGKey(42)
    names = get_flattened_keys(samples)
    n_row = -(len(names) // -2)
    n_col = 2
    fig, axs = plt.subplots(n_row, n_col, figsize=(5 * n_col, 4 * n_row))
    fig.suptitle('Parameter Histograms', size=12)
    fig.supxlabel('Parameter Value')
    fig.supylabel('Frequency')
    for name, ax in zip(names, axs.flat):
        params = get_by_path(samples, name.split('.')).flatten()
        if hist_limit and params.size > hist_limit:
            # Randomly sample from params if the size is too large
            idx = jnp.unique(jax.random.randint(key, (hist_limit,), 0, params.size))
            params = params[idx]
        ax.hist(params, bins=50)
        ax.set_title(name + f' ({params.size} observations)', size=10)


def plot_pca(
    samples: ParamTree,
    dim: Literal['2d', '3d'],
    max_chains: int | None = None,
    max_samples: int | None = None,
    annotate: bool = False,
):
    """
     Visualize the samples in principal component space using PCA.

     Parameters
     ----------
     `samples`: ParamTree
         Samples to visualize in PCA space.
     `dim`: Literal['2d', '3d']
         Dimension of the PCA plot.
     `max_chains`: int | None
         Maximum number of chains to plot.
    `max_samples`: int | None
         Maximum number of samples to plot.
     `annotate`: bool
         Annotate the samples with their corresponding sample index.
    """
    names = get_flattened_keys(samples)
    n_col = 2
    n_row = -(len(names) // -n_col)

    if max_chains:
        samples = jax.tree.map(lambda x: x[:max_chains], samples)
    if max_samples:
        # Evenly distribute the samples across the chains
        step = max(count_samples(samples) // max_samples, 1)
        samples = jax.tree.map(lambda x: x[:, ::step], samples)

    n_chains = count_chains(samples)
    n_samples = count_samples(samples)

    fig = plt.figure(figsize=(7.5 * n_col, 7.5 * n_row))

    pca2 = PCA(n_components=2)
    pca3 = PCA(n_components=3)

    for idx, name in enumerate(names):
        params = get_by_path(samples, name.split('.'))
        ax = fig.add_subplot(
            n_row, n_col, idx + 1, projection=dim if dim == '3d' else None
        )
        for i in range(n_chains):
            params_flat = params[i].reshape(n_samples, -1)
            if dim == '2d':
                ax.set_xlabel('PC_1')
                ax.set_ylabel('PC_2')
                pca_params = pca2.fit_transform(params_flat)
                ax.scatter(pca_params[..., 0], pca_params[..., 1])
                if annotate:
                    for j in range(n_samples):
                        ax.text(pca_params[j, 0], pca_params[j, 1], str(j), fontsize=12)
            elif dim == '3d':
                ax.set_xlabel('PC_1')
                ax.set_ylabel('PC_2')
                ax.set_zlabel('PC_3')
                if params_flat.shape[-1] < 3:
                    pass
                else:
                    pca_params = pca3.fit_transform(params_flat)
                    ax.scatter(
                        pca_params[..., 0], pca_params[..., 1], pca_params[..., 2]
                    )
                    if annotate:
                        for j in range(n_samples):
                            ax.text(
                                pca_params[j, 0],
                                pca_params[j, 1],
                                pca_params[j, 2],
                                str(j),
                                fontsize=7,
                            )
        ax.set_title(name)
        fig.tight_layout()


def plot_effective_sample_size(samples: ParamTree) -> dict[str, jnp.ndarray]:
    """Visualize the effective_sample_sizes for each layer in the architecture.

    Parameters
    ----------
    `samples`: ParamTree
        ParamTree of samples to compute and visualize effective sample sizes
            (n_chains, n_samples, ...).

    Returns
    -------
    dict[str, jnp.ndarray]
        Average Effective Sample Sizes for each layer in the architecture.
    """
    names = get_flattened_keys(samples)
    n_chains = count_chains(samples)
    n_samples = count_samples(samples)

    ess = {
        n: sandbox_metrics.effective_sample_size(v)
        for n, v in zip(names, jax.tree.leaves(samples))
    }
    _plot_boxplots(
        ess, f'Effective Sample Sizes across {n_chains} chains for {n_samples} samples'
    )
    return {n: jnp.mean(v) for n, v in ess.items()}


def plot_split_chain_r_hat(samples: ParamTree, n_splits: int) -> dict[str, jnp.ndarray]:
    """Visualize the split Gelman-Rubin R-hat statistic across chains for each layer.

    Parameters
    ----------
    `samples`: ParamTree
        ParamTree of samples to compute and visualize R-hat statistics.
    `n_splits`: int
        Number of splits of the chains.

    Returns
    -------
    dict[str, jnp.ndarray]
        Average Split Chain R-hat for each layer in the architecture.
    """
    names = get_flattened_keys(samples)
    n_chains = count_chains(samples)
    n_samples = count_samples(samples)
    rhat = {
        n: sandbox_metrics.split_chain_r_hat(v, n_splits)
        for n, v in zip(names, jax.tree.leaves(samples))
    }
    _plot_boxplots(
        rhat,
        f'{n_splits} Split Chain R-hat for {n_chains} chains each {n_samples} samples',
    )
    return {n: jnp.mean(v) for n, v in rhat.items()}


def plot_variances(
    samples: ParamTree,
) -> tuple[dict[str, jnp.ndarray], dict[str, jnp.ndarray]]:
    """Visualize the between-chain variance for each layer in the architecture.

    Parameters
    ----------
    `samples`: ParamTree of samples to compute and visualize variances.

    Returns
    -------
    tuple[dict[str, jnp.ndarray], dict[str, jnp.ndarray]]
        Average Between-chain and Within-chain variances for each layer.
    """
    names = get_flattened_keys(samples)
    n_chains = count_chains(samples)
    # n_samples = count_samples(samples)
    bcv = {
        n: sandbox_metrics.between_chain_var(v)
        for n, v in zip(names, jax.tree.leaves(samples))
    }
    wcv = {
        n: sandbox_metrics.within_chain_var(v)
        for n, v in zip(names, jax.tree.leaves(samples))
    }
    _plot_boxplots(bcv, f'Between-chain variance across {n_chains} chains')
    _plot_boxplots(wcv, f'Within-chain variance across {n_chains} chains')
    return (
        {n: jnp.mean(v) for n, v in bcv.items()},
        {n: jnp.mean(v) for n, v in wcv.items()},
    )


def _plot_boxplots(value_dict: dict[str, jnp.ndarray], title: str):
    """Vizualize boxplots for any mapping of dict[layer_name, param_stat]."""
    fig, ax = plt.subplots()
    fig.set_size_inches(w=len(value_dict) * 10 / 6, h=5)
    _ = ax.boxplot(
        [i.reshape(-1) for i in value_dict.values()],
        positions=[i for i in range(len(value_dict))],
        showfliers=False,
        showmeans=True,
    )
    if title:
        plt.title(title)
    plt.xlabel('Layer')
    plt.ylabel('Value')
    plt.grid(axis='y', linestyle='--', visible=True)
    _ = ax.set_xticklabels(value_dict.keys(), rotation=90)


def plot_warmstart_results(warmstart_dir: Path):
    """Plot the warmstart results.

    Parameters
    ----------
    `warmstart_dir`: Path
        Path to the warmstart directory containing the train, valid, and test results.

    Notes
    -----
    - This is framework-specific implementation and works only within the context of the
    `probabilisticml` framework.
    """
    partitions = ('train', 'valid', 'test')
    for partition in partitions:
        plots = os.listdir(warmstart_dir / partition)
        if plots:
            n_row = -(len(plots) // -2)
            n_col = 2
            fig = plt.figure(figsize=(10 * n_col, 6 * n_row))
            fig.suptitle(f'{partition.capitalize()} Set')
            for plot in plots:
                plt.subplot(n_row, n_col, plots.index(plot) + 1)
                img = plt.imread(warmstart_dir / partition / plot)
                plt.imshow(img)
                plt.axis(False)
                plt.tight_layout()


def plot_lppd(lvals: jnp.ndarray, labels: jnp.ndarray, task: Task):
    """Plot the log pointwise predictive density.

    Parameters
    ----------
    `lvals`: jnp.ndarray
        Logits from the model of shape:
        (n_chains, n_samples, n_obs) or (n_chains, n_samples, n_obs, n_classes).
    `labels`: jnp.ndarray
        Ground truth labels.
    `task`: Task
        Task type (Classification or Regression).
    """
    pointwise_lppd = sandbox_metrics.pointwise_lppd(lvals, labels, task=task)
    combined_lppd = sandbox_metrics.running_lppd(pointwise_lppd)
    plt.plot(
        combined_lppd, label='Combined', color='black', linestyle='--', linewidth=3
    )
    for i, p_lppd in enumerate(pointwise_lppd):
        lppd = sandbox_metrics.running_lppd(p_lppd)
        plt.plot(lppd, label=f'Chain {i + 1}')
    plt.xlabel('Number of samples')
    plt.ylabel('Log pointwise predictive density')
    plt.legend()
