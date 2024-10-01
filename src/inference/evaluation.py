"""Evaluation functions for DE and BDE models."""
from typing import Optional, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
from scipy.stats import mode

import src.inference.metrics as metrics
from src.config.data import Task
from src.inference.utils import count_chains, count_samples
from src.types import ParamTree


def predict_from_samples(
    model: nn.Module, samples: ParamTree, x: jnp.ndarray, **kwargs
) -> jnp.ndarray:
    """Forward pass predictions from samples.

    Parameters:
    ----------
    model: nn.Module
        Flax module.
    samples: ParamTree
        Samples of the model parameters of shape (n_samples, ...).
    x: jnp.ndarray
        Input features of shape (B, ...).
    kwargs: dict
        Additional arguments to pass to model forward pass.

    Returns:
    -------
    jnp.ndarray
        Predictions of shape (n_samples, B, ...).
    """
    # vapply = jax.vmap(partial(model.apply, **kwargs), in_axes=(0, None))
    # predictions = vapply({'params': samples}, x)
    # do it sequentially for now (memory issues)
    predictions = []
    n_samples = jax.tree_util.tree_leaves(samples)[0].shape[0]
    for i in range(n_samples):
        params = jax.tree_map(lambda x: x[i], samples)
        predictions.append(model.apply({'params': params}, x, **kwargs))
    predictions = jnp.stack(predictions)
    return predictions


def sample_from_predictions(
    predictions: jnp.ndarray, task: Task, rng_key: jnp.ndarray
) -> jnp.ndarray:
    """Sample predictions from n_sample forward pass predictions.

    Parameters:
    ----------
    predictions: jnp.ndarray
        Predictions of shape (n_chain, n_samples, B, ...).
    task: Task
        Task type (classification or regression).
    rng_key: jnp.ndarray
        Random key for sampling.

    Returns:
    -------
    jnp.ndarray
        Sampled predictions of shape (n_chain, n_samples, B, ...).
    """
    if task == Task.REGRESSION:
        loc = predictions[..., 0]
        scale = jnp.exp(predictions[..., 1]).clip(min=1e-6, max=1e6)
        predictions = jax.random.normal(rng_key, shape=jnp.shape(loc)) * scale + loc
    elif task == Task.CLASSIFICATION:
        predictions = jax.random.categorical(rng_key, logits=predictions)
    return predictions


def calibration_error(
    nominal_coverage: jnp.ndarray,
    observed_coverage: jnp.ndarray,
    weights: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """Calculate the calibration error."""
    if weights is None:
        return jnp.sqrt(jnp.mean(jnp.square(nominal_coverage - observed_coverage)))
    return jnp.sqrt(
        jnp.mean(weights * jnp.square(nominal_coverage - observed_coverage))
    )


def coverage_weighting(
    nominal_coverage: Union[jnp.ndarray, list], kappa: float = 1.0
) -> jnp.ndarray:
    """Calculate a coverage weighting."""
    if isinstance(nominal_coverage, list):
        nominal_coverage = jnp.array(nominal_coverage)
    return (nominal_coverage**kappa) / jnp.sum(nominal_coverage**kappa)


def get_quantiles(coverage: float) -> jnp.ndarray:
    """Get the quantiles for a given coverage level."""
    return jnp.array([0.5 - coverage / 2, 0.5 + coverage / 2])


def calculate_coverage(
    nominal_coverages: list,
    Y_test: jnp.ndarray,
    preds: jnp.ndarray,
) -> jnp.ndarray:
    """
    Calculate the coverage.

    Currently only supports full batch predictions.

    Parameters:
    ----------
    nominal_coverages: list
        List of nominal coverages.
    Y_test: jnp.ndarray
        True labels of shape (N,)
    preds: jnp.ndarray
        Predictions of shape (n_chains, n_samples, N)

    Returns:
    -------
    jnp.ndarray
        Coverage of shape (len(nominal_coverages),)
    """
    nominal_coverage_dict = {cov: get_quantiles(cov) for cov in nominal_coverages}
    quantiles = jnp.unique(jnp.concatenate([v for v in nominal_coverage_dict.values()]))
    quantile2index = {q.item(): i for i, q in enumerate(quantiles)}
    creds = jnp.quantile(preds.reshape(-1, preds.shape[-1]), q=quantiles, axis=0)
    coverage = []
    for nc in nominal_coverages:
        coverage.append(
            jnp.mean(
                (creds[quantile2index[nominal_coverage_dict[nc][0].item()]] <= Y_test)
                & (creds[quantile2index[nominal_coverage_dict[nc][1].item()]] >= Y_test)
            )
        )
    return jnp.array(coverage)


def predict_de(
    params: ParamTree,
    module: nn.Module,
    features: jnp.ndarray,
    labels: jnp.ndarray,
    batch_size: int | None,
    verbose: bool = True,
    **kwargs,
) -> jnp.ndarray:
    """Predict with Deep Ensemble members.

    Parameters:
    ----------
    params: ParamTree
        Deep Ensemble parameters of shape (n_members, ...).
    module: nn.Module
        Flax module.
    features: jnp.ndarray
        Input features of shape (B, ...).
    labels: jnp.ndarray
        Target labels of shape (B, ...).
    batch_size: int | None
        Batch size for prediction.
    verbose: bool
        Verbosity flag.
    kwargs: dict
        Additional arguments to pass to model forward pass.

    Returns:
    -------
    preds: jnp.ndarray
        Predictions of shape (n_members, B, ...).
    """
    n_chains = count_chains(params)
    if verbose:
        print(f'Predicting with {n_chains} chains')
    N_F, N_L = features.shape[0], labels.shape[0]
    assert N_F == N_L, 'Labels and Features must match on the first dimension'
    if batch_size:
        plan = jnp.array_split(jnp.arange(N_F), N_F // min(batch_size, N_F))
    else:
        plan = [jnp.arange(N_F)]
    for i, ids in enumerate(plan):
        if verbose:
            print(f'Predicting Batch {i + 1} of {len(plan)}')
        if i == 0:
            pred = predict_from_samples(
                model=module, samples=params, x=features[ids], **kwargs
            )
        else:
            pred = jnp.hstack(
                [
                    pred,
                    predict_from_samples(
                        model=module, samples=params, x=features[ids], **kwargs
                    ),
                ],
            )
    assert (
        n_chains == pred.shape[0]
    ), 'Number of chains must match the number of predictions on first dimension'
    if len(pred.shape) > 4:
        pred = pred.reshape(*pred.shape[:3], -1)
    return pred


def evaluate_de(
    params: ParamTree,
    module: nn.Module,
    features: jnp.ndarray,
    labels: jnp.ndarray,
    task: Task,
    batch_size: int | None,
    verbose: bool = True,
    metrics_dict: dict = {},
    n_samples: int = 0,
    rng_key: jnp.ndarray = jax.random.PRNGKey(42),
    nominal_coverages: Optional[list] = None,
    **kwargs,
) -> tuple[jnp.ndarray, dict]:
    """Evaluate the performance of Deep Ensemble models.

    Parameters:
    ----------
    params: ParamTree
        Deep Ensemble parameters of shape (n_members, ...).
    module: nn.Module
        Flax module.
    features: jnp.ndarray
        Input features of shape (B, ...).
    labels: jnp.ndarray
        Target labels of shape (B, ...).
    task: Task
        Task type (classification or regression).
    batch_size: int | None
        Batch size for prediction.
    verbose: bool
        Verbosity flag.
    metrics_dict: dict
        Dict to store metrics.
    n_samples: int
        Number of samples to draw from the posterior.
    rng_key: jnp.ndarray
        Random key for sampling from the posterior.
    nominal_coverages: list
        List of nominal coverages for calibration error.
    kwargs: dict
        Additional arguments to pass to model forward pass.

    Returns:
    -------
    preds: jnp.ndarray
        Predictions of shape (n_members, B, ...).
    metrics_dict: dict
        Dict with evaluation metrics.
    """
    if task == task.CLASSIFICATION:
        labels = labels.astype(jnp.int32)
    preds = predict_de(params, module, features, labels, batch_size, verbose, **kwargs)

    # Classification
    if task == Task.CLASSIFICATION:
        # Combined predictions
        lppd = metrics.lppd(
            lppd_pointwise=metrics.pointwise_lppd(preds, labels, task=task)
        )
        acc = jnp.mean(
            labels == mode(preds.argmax(axis=-1), axis=0).mode  # Mode over (Members)
        )
        print(
            f'Deep Ensemble Performance | LPPD: {lppd:.3f}, ACC: {acc:.4f}'  # noqa
        )  # noqa: E231
        metrics_dict['de_lppd'] = lppd
        metrics_dict['de_acc'] = acc
        print('_' * 50)

        # Per chain predictions
        for i, pred in enumerate(preds):
            lppd_point = metrics.pointwise_lppd(pred, labels, task=task)
            lppd = metrics.lppd(lppd_point)
            acc = jnp.mean(labels == pred.argmax(axis=-1))
            print(f'Chain {i} | LPPD: {lppd:.3f}, ACC: {acc:.4f}')  # noqa: E231

    # Regression
    elif task == Task.REGRESSION:
        if n_samples:
            vectorized_sample = jax.vmap(
                sample_from_predictions, in_axes=(None, None, 0), out_axes=1
            )
            point_preds = vectorized_sample(
                preds, task, jax.random.split(rng_key, n_samples)
            )
        # Combined predictions
        lppd = metrics.lppd(
            lppd_pointwise=metrics.pointwise_lppd(preds, labels, task=task)
        )
        rmse = jnp.sqrt(
            jnp.mean((labels - preds[..., 0].mean(axis=0)) ** 2)  # Mean over (Members)
        )
        print(
            f'Deep Ensemble Performance | LPPD: {lppd:.3f}, RMSE: {rmse:.4f}'  # noqa
        )  # noqa: E231
        metrics_dict['de_lppd'] = lppd
        metrics_dict['de_rmse'] = rmse
        print('_' * 50)

        # Per chain predictions
        for i, pred in enumerate(preds):
            lppd = metrics.lppd(
                lppd_pointwise=metrics.pointwise_lppd(pred, labels, task=task)
            )
            rmse = jnp.sqrt(jnp.mean((labels - pred[..., 0]) ** 2))
            print(f'Chain {i} | LPPD: {lppd:.3f}, RMSE: {rmse:.4f}')  # noqa: E231

        if nominal_coverages is not None:
            coverage = calculate_coverage(
                nominal_coverages,
                labels,
                point_preds,
            )
            cal_error = calibration_error(
                nominal_coverage=jnp.array(nominal_coverages),
                observed_coverage=coverage,
                weights=None,
            )
            print('_' * 50)
            print(f'Calibration Error: {cal_error:.4f}')
            metrics_dict['de_cal_error'] = cal_error
            for i, cov in enumerate(nominal_coverages):
                print(f'Coverage for {cov}: {coverage[i]:.4f}')
                metrics_dict[f'de_coverage_{cov}'] = coverage[i]
    return preds, metrics_dict


def predict_bde(
    params: ParamTree,
    module: nn.Module,
    features: jnp.ndarray,
    labels: jnp.ndarray,
    batch_size: int | None,
    verbose: bool = True,
    **kwargs,
) -> jnp.ndarray:
    """Predict with Bayesian Deep Ensemble members.

    Parameters:
    ----------
    params: ParamTree
        Bayesian Deep Ensemble samples of shape (n_chains, n_samples, ...).
    module: nn.Module
        Flax module.
    features: jnp.ndarray
        Input features of shape (B, ...).
    labels: jnp.ndarray
        Target labels of shape (B, ...).
    batch_size: int | None
        Batch size for prediction.
    verbose: bool
        Verbosity flag.
    kwargs: dict
        Additional arguments for prediction.

    Returns:
    -------
    jnp.ndarray
        Predictions of shape (n_chains, n_samples, B, ...).
    """
    n_chains, n_samples = count_chains(params), count_samples(params)
    if verbose:
        print(f'| Predicting with {n_chains} chains each with {n_samples} samples')
    N_F, N_L = features.shape[0], labels.shape[0]
    assert N_F == N_L, 'Labels and Features must match on the first dimension'
    if batch_size:
        plan = jnp.array_split(jnp.arange(N_F), N_F // min(batch_size, N_F))
    else:
        plan = [jnp.arange(N_F)]

    preds_over_chains = []
    for i in range(n_chains):
        param = jax.tree.map(lambda x: x[i], params)
        if verbose:
            print(f'> Starting Prediction for Chain {i + 1}')
        for j, ids in enumerate(plan):
            if verbose:
                print(f'| | Predicting Batch {j + 1} of {len(plan)}')
            if j == 0:
                preds = predict_from_samples(
                    model=module, samples=param, x=features[ids], **kwargs
                )
            else:
                preds = jnp.hstack(
                    [
                        preds,
                        predict_from_samples(
                            model=module, samples=param, x=features[ids], **kwargs
                        ),
                    ],
                )
        if len(preds.shape) > 4:
            preds = preds.reshape(*preds.shape[:3], -1)
        preds_over_chains.append(preds)
    C = len(preds_over_chains)
    S = preds_over_chains[0].shape[0]
    assert (
        C == n_chains and S == n_samples
    ), 'Number of chains and samples must match the predictions shape'
    return jnp.stack(preds_over_chains)


def evaluate_bde(
    params: ParamTree,
    module: nn.Module,
    features: jnp.ndarray,
    labels: jnp.ndarray,
    task: Task,
    batch_size: int | None,
    verbose: bool = True,
    metrics_dict: dict = {},
    key: jax.Array = jax.random.PRNGKey(42),
    nominal_coverages: Optional[list] = None,
    **kwargs,
) -> tuple[jnp.ndarray, dict]:
    """Evaluate the performance of a Bayesian Deep Ensemble model.

    Parameters:
    ----------
    params: ParamTree
        Bayesian Deep Ensemble samples of shape (n_chains, n_samples, ...).
    module: nn.Module
        Flax module.
    features: jnp.ndarray
        Input features of shape (B, ...).
    labels: jnp.ndarray
        Target labels of shape (B, ...).
    task: Task
        Task type (classification or regression).
    batch_size: int | None
        Batch size for prediction.
    verbose: bool
        Verbosity flag.
    metrics_dict: dict
        Dict to store metrics.
    key: jax.Array
        Random key for sampling from the posterior.
    kwargs: dict
        Additional arguments for prediction.

    Returns:
    -------
    jnp.ndarray
        Predictions of shape (n_chains, n_samples, B, ...).
    dict
        Dict with evaluation metrics
    """
    if task == Task.CLASSIFICATION:
        labels = labels.astype(jnp.int32)
    preds_over_chains = predict_bde(
        params, module, features, labels, batch_size, verbose, **kwargs
    )
    preds = sample_from_predictions(preds_over_chains, task=task, rng_key=key)

    # Classification
    if task == Task.CLASSIFICATION:
        if nominal_coverages is not None:
            print('Calculating coverage is not supported for classification yet.')
        # Combined predictions
        lppd = metrics.lppd(
            lppd_pointwise=metrics.pointwise_lppd(preds_over_chains, labels, task=task)
        )
        acc = jnp.mean(
            labels == mode(preds, axis=(0, 1)).mode
        )  # Mode over (Chain, Sam)
        print(
            f'Bayesian Deep Ensemble Performance | LPPD: {lppd:.3f}, ACC: {acc:.4f}'  # noqa
        )  # noqa
        # add lppd and acc to metrics dict
        metrics_dict['lppd'] = lppd
        metrics_dict['acc'] = acc
        print('_' * 50)

        # Per chain predictions
        for i, lv in enumerate(preds_over_chains):
            lppd = metrics.lppd(
                lppd_pointwise=metrics.pointwise_lppd(lv, labels, task=task)
            )
            acc = jnp.mean(labels == mode(preds[i], axis=0).mode)  # Mode over (Samples)
            print(f'Chain {i} | LPPD: {lppd:.3f}, ACC: {acc:.4f}')  # noqa: E231

    # Regression
    elif task == Task.REGRESSION:
        # Combined predictions
        nan_chains = jnp.isnan(preds).any(axis=(1, 2))
        if nan_chains.any():
            print(f'Warning: Chains {jnp.where(nan_chains)[0]} have NaN predictions')
            if nan_chains.all():
                nan_chains = jnp.repeat(True, len(nan_chains))
        lppd = metrics.lppd(
            lppd_pointwise=metrics.pointwise_lppd(
                preds_over_chains[~nan_chains], labels, task=task
            )
        )
        rmse = jnp.sqrt(
            # Mean over (Chain, Samp)
            jnp.mean((labels - preds[~nan_chains].mean(axis=[0, 1])) ** 2)
        )
        print(
            f'Bayesian Deep Ensemble Performance | LPPD: {lppd:.3f}, RMSE: {rmse:.4f}'  # noqa
        )
        metrics_dict['lppd'] = lppd
        metrics_dict['rmse'] = rmse
        print('_' * 50)

        # Per chain predictions
        for i, lvs in enumerate(preds_over_chains):
            lppd = metrics.lppd(
                lppd_pointwise=metrics.pointwise_lppd(lvs, labels, task=task)
            )
            rmse = jnp.sqrt(
                jnp.mean((labels - preds[i].mean(axis=0)) ** 2)  # Mean over Samples
            )
            print(f'Chain {i} | LPPD: {lppd:.3f}, RMSE: {rmse:.4f}')  # noqa: E231

        if nominal_coverages is not None:
            coverage = calculate_coverage(nominal_coverages, labels, preds[~nan_chains])
            cal_error = calibration_error(
                nominal_coverage=jnp.array(nominal_coverages),
                observed_coverage=coverage,
                weights=None,
            )
            print('_' * 50)
            print(f'Calibration Error: {cal_error:.4f}')
            metrics_dict['cal_error'] = cal_error
            for i, cov in enumerate(nominal_coverages):
                print(f'Coverage for {cov}: {coverage[i]:.4f}')
                metrics_dict[f'coverage_{cov}'] = coverage[i]
    return preds_over_chains, metrics_dict
