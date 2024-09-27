"""Convert frequentist Flax modules to Bayesian NNs."""
import logging

import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.scipy.stats as stats

import src.training.utils as train_utils
from src.config.data import Task
from src.training.priors import Prior
from src.types import ParamTree

logger = logging.getLogger(__name__)


class ProbabilisticModel:
    """Convert frequentist Flax modules to Bayesian Numpyro modules."""

    def __init__(
        self,
        module: nn.Module,
        params: ParamTree,
        prior: Prior,
        task: Task,
        n_batches: int = 1,
    ):
        """Initialize the ProbabilisticModel class for Bayesian training.

        Parameters:
        -----------
        module: nn.Module
            Flax module to use for Bayesian training
        params: ParamTree
            Parameters of the Flax module.
        prior: Prior
            Prior distribution for the parameters.
        task: Task
            Task type (classification or regression)
        n_batches: int
            Number of batches sampler will see in one epoch.
        """
        self.task = task
        self.module = module
        self.n_params = train_utils.count_params(params)
        self.n_batches = n_batches
        self.prior = prior
        logger.info(f'Initialized ProbModelBuilder for {self.task} task')

    def __str__(self):
        """Return informative string representation of the model."""
        return (
            f'{self.__class__.__name__}:\n'  # noqa
            f' | Task: {self.task}\n'
            f' | Params: {self.n_params}'
            f' | Batches: {self.n_batches}\n'
            f' | Prior: {self.prior.name}'
        )

    @property
    def minibatch(self):
        return self.n_batches > 1

    def log_prior(self, params: ParamTree) -> jax.Array:
        """Compute log prior for given parameters."""
        return self.prior.log_prior(params)

    def log_likelihood(
        self,
        params: ParamTree,
        x: jnp.ndarray,
        y: jnp.ndarray,
        **kwargs,
    ) -> jnp.ndarray:
        """Evaluate Log likelihood for given Parameter Tree and data.

        Parameters:
        -----------
        params: ParamTree
            Parameters of the model.
        x: jnp.ndarray
            Input data of shape (batch_size, ...).
        y: jnp.ndarray
            Target data of shape (batch_size, ...).
        kwargs: dict
            Additional keyword arguments to pass to the model forward pass.

        Raises:
        -------
        NotImplementedError: If computation for given `task` is not implemented.
        """
        lvals = self.module.apply({'params': params}, x, **kwargs)
        if self.task == Task.REGRESSION:
            return jnp.nansum(
                stats.norm.logpdf(
                    x=y,
                    loc=lvals[..., 0],
                    scale=jnp.exp(lvals[..., 1]).clip(min=1e-6, max=1e6),
                )
            )
        elif self.task == Task.CLASSIFICATION:
            # directly taken from https://num.pyro.ai/en/stable/_modules/
            # numpyro/distributions/discrete.html#CategoricalLogits
            batch_shape = jax.lax.broadcast_shapes(jnp.shape(y), jnp.shape(lvals)[:-1])
            y = jnp.expand_dims(y, -1)
            y = jnp.broadcast_to(y, batch_shape + (1,))
            log_pmf = lvals - jax.scipy.special.logsumexp(lvals, axis=-1, keepdims=True)
            log_pmf = jnp.broadcast_to(log_pmf, batch_shape + jnp.shape(log_pmf)[-1:])
            return jnp.nansum(jnp.take_along_axis(log_pmf, indices=y, axis=-1)[..., 0])
        else:
            raise NotImplementedError(
                f'Likelihood computation for {self.task} not implemented'
            )

    def log_unnormalized_posterior(
        self,
        position: ParamTree,
        x: jnp.ndarray,
        y: jnp.ndarray,
        **kwargs,
    ):
        """Log unnormalized posterior (potential) for given parameters and data.

        Parameters:
        -----------
        params: ParamTree
            Parameters of the model.
        x: jnp.ndarray
            Input data of shape (batch_size, ...).
        y: jnp.ndarray
            Target data of shape (batch_size, ...).
        kwargs: dict
            Additional keyword arguments to pass to the model forward pass.
        """
        return (
            self.log_prior(position)
            + self.log_likelihood(position, x, y, **kwargs) * self.n_batches
        )
