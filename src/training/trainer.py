"""Trainer module for training BDEs."""
import logging
import os
from functools import partial
from pathlib import Path

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

import src.inference.metrics as sandbox_metrics
import src.training.utils as train_utils
from src.config.core import Config
from src.config.data import DatasetType, Task
from src.dataset import (
    ImageLoader,
    TabularLoader,
    TextLoader,
)
from src.dataset.utils import CustomBPETokenizer
from src.inference.metrics import (
    ClassificationMetrics,
    MetricsStore,
    RegressionMetrics,
)
from src.inference.reporting import generate_html_report
from src.training.probabilistic import ProbabilisticModel
from src.training.sampling import inference_loop
from src.types import ParamTree
from src.utils import measure_time, pretty_string_dict

logger = logging.getLogger(__name__)

Loader = TabularLoader | TextLoader


class BDETrainer:
    """Trainer class for training BDEs.

    Attributes
    ----------
        config (Config): Configuration object for the trainer.

    Training
    --------
    Training is started by calling the `.train_bde()` method of the class instance.

    Steps
    -----
    1. Warmstart Phase: Train the Deep Ensemble Members: `.train_warmstart()`.
    2. Sampling Phase: Perform the sampling using MCMC: `.start_sampling()`.
    3. Reporting Phase: Generate the HTML report: `.generate_html_report()`.
    """

    def __init__(self, config: Config, **kwargs):
        """Initialize the trainer."""
        assert isinstance(config, Config)
        self.config = config
        self._key = config.jax_rng
        self.n_devices = jax.device_count()
        self.metrics_warmstart = MetricsStore.empty()
        self._completed = False

        # Setup directory
        logger.info('> Setting up directories...')
        _ = config.setup_dir()

        if self.n_chains % self.n_devices:
            raise ValueError(
                'n_chains must be divisible by the number of devices.'
                f'{self.n_chains} % {self.n_devices} != 0.'
            )
        self.train_plan = jnp.array_split(
            jnp.arange(config.n_chains), config.n_chains / self.n_devices
        )

        # Setup DataLoader
        logger.info('> Setting up DataLoader...')
        if config.data.data_type == DatasetType.TABULAR:
            self.loader = TabularLoader(
                config=config.data,
                rng=config.jax_rng,
                target_len=config.data.target_len,
            )
        elif config.data.data_type == DatasetType.TEXT:
            if not ('context_len' in config.model and 'vocab_size' in config.model):
                raise ValueError(
                    'Missing fields in the model config definition.'
                    'Add "context_len" and "vocab_size" fields '
                    f'to the model config definition. {config.model.__class__.__name__}'
                )
            tokenizer = CustomBPETokenizer(
                config.model['context_len'], max_vocab_size=config.model['vocab_size']
            )
            self.loader = TextLoader(
                config=config.data,
                rng=config.jax_rng,
                tokenizer=tokenizer,
                context_len=config.model['context_len'],
                omit_freq=100,
                target_len=self.config.data.target_len,
            )
        elif config.data.data_type == DatasetType.IMAGE:
            self.loader = ImageLoader(config=config.data, rng=config.jax_rng)

        # Setup Flax Model
        logger.info('> Setting up Flax Model...')
        self.module = config.get_flax_model()

        # Setup Probabilistic Model
        logger.info('> Setting up Probabilistic Model...')
        self.prob_model = ProbabilisticModel(
            module=self.module,
            params=self.init_module_params(n_device=1),
            prior=config.training.prior,
            n_batches=self.n_batches,
            task=config.data.task,
        )

        logger.info(f'> Trainer has been successfully initialized\n{self}')

    def __str__(self):
        """Return the string representation of the trainer."""
        return (
            f'{self.__class__.__name__}:\n'
            f' | Experiment: {self.config.experiment_name}\n'
            f' | Chains: {self.n_chains}\n'
            f' | Devices: {self.n_devices}\n'
            f' | Warmstart: {self.has_warmstart}\n'
            f' | Warmstart Checkpoint: {self.warmstart_exp}\n'
            f' | Module: {self.module.__class__.__name__}\n'
            f'{"-"*50}\n\t{self.loader}\n'
            f'{"-"*50}\n\t{self.prob_model}\n'
            f'{"-"*50}\n\tSampler Configuration:\n'
            f'{pretty_string_dict(self.config_sampler.to_dict())}\n'
        )

    def train_bde(self):
        """Train the BDE Model.

        Steps
        -----
        1. Warmstart Phase: Train the Deep Ensemble Members: `.train_warmstart()`.
        2. Sampling Phase: Perform the sampling using MCMC: `.start_sampling()`.
        3. Reporting Phase: Generate the HTML report: `.generate_html_report()`.
        """
        logger.info('> Starting BDE Training...')

        # Warmstart Phase
        if self.has_warmstart:
            logger.info('> Starting Deep Ensemble (Warmstart) Training...')
            self.train_warmstart()
        else:
            # TODO
            logger.warning(
                (
                    'Warmstart is not enabled!'
                    'Sampling will start from randomly initialized parameters.'
                )
            )
        # Sampling Phase
        # breakpoint() # comes in handy for pretraining of e.g. embeddings
        logger.info('> Starting Sampling...')
        self.start_sampling()
        self._completed = True
        logger.info('> BDE Training completed successfully.')

        # Reporting Phase
        logger.info('> Generating HTML Report...')
        self.generate_html_report()

    def generate_html_report(self):
        """Generate the HTML report."""
        if not self._completed:
            logger.warning(
                '\t| BDE Training was not completed. Report may not be accurate.'
            )
        with measure_time('time.report'):
            path = generate_html_report(self.config)
        logger.info(f'\t| Report has been successfully generated at {path}.')

    def get_single_input(self, batch_size: int = 1) -> jnp.ndarray:
        """Return a random input.

        Parameters
        ----------
            batch_size (int): Batch size.

        Returns
        -------
            (jnp.ndarray): Random input data of shape (batch_size, ...)
        """
        gen = self.loader.iter('train', batch_size=batch_size)
        try:
            return next(gen)['feature']
        finally:
            gen.close()

    def init_module_params(self, n_device: int) -> ParamTree:
        """Initialize the parameters for the module.

        Parameters
        ----------
            n_device (int): Number of devices.

        Returns
        -------
            (ParamTree): Initialized parameters, with leaf nodes of shape
                (N_DEVICE, ...) if n_device > 1, else (...)
        """
        if n_device > 1:  # replicate the input for multiple devices
            return jax.pmap(initialize_params, static_broadcasted_argnums=(2))(
                jax.random.split(self.key, n_device),
                self.get_single_input(batch_size=1)[None, ...].repeat(n_device, axis=0),
                self.module,
            )
        return initialize_params(
            rng=self.key,
            x=self.get_single_input(batch_size=1),
            module=self.module,
        )

    def init_training_state(self, n_device: int) -> TrainState:
        """Initialize the training state.

        Parameters
        ----------
            n_device (int): Number of devices.

        Returns
        -------
            (TrainState): Initialized training state.
        """
        if n_device > 1:  # replicate the input for multiple devices
            return jax.pmap(get_initial_state, static_broadcasted_argnums=(2, 3))(
                jax.random.split(self.key, n_device),
                self.get_single_input()[None, ...].repeat(n_device, axis=0),
                self.module,
                self.optimizer,
            )
        return get_initial_state(
            rng=self.key,
            x=self.get_single_input(),
            module=self.module,
            optimizer=self.optimizer,
        )

    @property
    def key(self):
        """Return the next rng key for the dataloader."""
        self._key, key = jax.random.split(self._key)
        return key

    @property
    def task(self):
        """Return the task."""
        return self.config.data.task

    @property
    def n_chains(self):
        """Return the number of chains."""
        return self.config.n_chains

    @property
    def config_warmstart(self):
        """Return the warmstart configuration."""
        return self.config.training.warmstart

    @property
    def config_sampler(self):
        """Return the sampler configuration."""
        return self.config.training.sampler

    @property
    def warmstart_exp(self):
        """Return the warmstart experiment name if available."""
        if self.config_warmstart.warmstart_exp_dir:
            return self.config_warmstart.warmstart_exp_dir.split('/')[-1]

    @property
    def has_warmstart(self):
        """Check if warmstart is enabled."""
        return self.config.training.has_warmstart

    @property
    def batch_size_w(self):
        """Return the batch size for warmstart."""
        return self.config_warmstart.batch_size or len(self.loader.data_train)

    @property
    def n_batches(self):
        """Return the number of batches."""
        return 1

    @property
    def optimizer(self):
        """Return the optimizer."""
        return self.config.training.optimizer

    @property
    def tree_path(self):
        """Return the tree path."""
        e_dir = self.exp_dir / 'tree'
        if not e_dir.exists():
            raise FileNotFoundError(f'Tree not found at {e_dir}')
        return e_dir

    @property
    def exp_dir(self):
        """Return the experiment directory."""
        return self.config.experiment_dir

    @property
    def _sampling_warmup_dir(self):
        if self.config_sampler.keep_warmup:
            return self.exp_dir / self.config_sampler._warmup_dir_name

    # NOTE: Don't modify the name: 'time.warmstart' is later used in
    # mile/inference/inference.ipynb for extracting the time from the logs.
    @measure_time('time.warmstart')
    def train_warmstart(self):
        """Start Warmstart phase of BDE (Deep Ensemble Training before sampling)."""
        cfg_warm = self.config_warmstart
        if not cfg_warm.warmstart_exp_dir:  # No checkpoints
            warm_path = self.exp_dir / cfg_warm._dir_name
            metrics_list: list[MetricsStore] = []

            for step in self.train_plan:  # Train Deep Ensemble Members
                state = self.init_training_state(n_device=len(step))

                # Train Deep Ensemble
                logger.info(f'\t| Starting Training Warmstart for chains {step}')
                state, metrics = self.train_de_member(state=state, n_parallel=len(step))
                logger.info(f'\t| Warmstart Training completed for chains {step}')
                metrics_list.append(metrics)

                # Save Checkpoints of Deep Ensemble Members
                for i, chain_n in enumerate(step):
                    if len(step) > 1:
                        train_utils.save_params(
                            warm_path,
                            jax.tree.map(lambda x: x[i], state.params),
                            chain_n,
                        )
                    else:
                        train_utils.save_params(warm_path, state.params, chain_n)
                    logger.info(f'\t| Deep Ensemble {chain_n} saved at {warm_path}')
                    self.loader.shuffle()  # Shuffle the data for the next chain

            # NOTE: save metrics once all chains are trained (might change in future)
            self.metrics_warmstart = MetricsStore.vstack(metrics_list)
            logger.info(f'\t| Saving Warmstart Metrics at {warm_path}')
            self.metrics_warmstart.save(
                path=warm_path / self.config_warmstart._metrics_fname, save_plots=True
            )

        else:  # With checkpoints
            warm_path = Path(cfg_warm.warmstart_exp_dir) / cfg_warm._dir_name
            # Check Avaliability of Deep Ensemble Checkpoints
            deep_ensembles = [
                i for i in os.listdir(warm_path) if i.startswith('params')
            ]
            if not deep_ensembles:
                raise ValueError(f'No chains found in the warmstart path {warm_path}')
            if len(deep_ensembles) != self.n_chains:
                if len(deep_ensembles) < self.n_chains:
                    raise ValueError(
                        (
                            'Not enough chains found in the warmstart path: '
                            f'{len(deep_ensembles)} < {self.n_chains}.'
                        )
                    )
                else:
                    logger.warning(
                        f'Using first {self.n_chains} from {len(deep_ensembles)} chains'
                    )
            # copy 'tree' file to the new directory as it is important in the inference
            tree = train_utils.load_tree(dir=warm_path.parent)
            train_utils.save_tree(dir=self.exp_dir, tree=tree)

    def train_de_member(
        self,
        state: TrainState,
        n_parallel: int,
    ) -> tuple[TrainState, MetricsStore]:
        """Train a single Deep Ensemble member.

        Parameters
        ----------
            state (TrainState): Initial Training State.
            n_parallel (int):
                Number of members in the Deep Ensemble to train in parallel
                across devices using jax.pmap (requires n_devices >= n_parallel).

        Returns
        -------
            (tuple[TrainState, MetricsStore]):
                Updated Training State and Metrics Store containing the metrics.
        """
        if self.task == Task.CLASSIFICATION:
            step_func = single_step_class
            pred_func = predict_class
        elif self.task == Task.REGRESSION:
            step_func = single_step_regr
            pred_func = predict_regr
        else:
            raise NotImplementedError(f'Task {self.task} not yet implemented.')

        if n_parallel > 1:
            _single_step = jax.pmap(step_func)
            _predict = jax.pmap(pred_func)
        else:
            _single_step = jax.jit(step_func)
            _predict = jax.jit(pred_func)

        valid_losses = jnp.array([]).reshape(n_parallel, 0)
        _stop_n = jnp.repeat(False, n_parallel)
        metrics_train, metrics_valid, metrics_test = [], [], []

        for epoch in range(self.config_warmstart.max_epochs):
            self.loader.shuffle()  # Shuffle the data inbetween epochs
            if jnp.all(_stop_n):
                break  # Early stopping condition

            # Train Single Epoch
            for i, batch in enumerate(
                self.loader.iter(
                    split='train', batch_size=self.batch_size_w, n_devices=n_parallel
                )
            ):
                state, metrics = _single_step(
                    state,
                    batch['feature'],
                    batch['label'],
                    _stop_n if n_parallel > 1 else _stop_n[0],
                )
                if isinstance(metrics, ClassificationMetrics):
                    logger.info(
                        f'Epoch {epoch} '
                        f'| Batch {i} | Training Loss: {metrics.cross_entropy} '
                        f'| Accuracy: {metrics.accuracy}'
                    )
                if isinstance(metrics, RegressionMetrics):
                    logger.info(
                        f'Epoch {epoch} | Batch {i} | Training Loss: {metrics.nlll} '
                        f'| RMSE: {metrics.rmse}'
                    )
                metrics_train.append(metrics)

            # Validate After each epoch
            # NOTE: In large datasets full-batch pass might be infeasible
            # but for now let's keep it simple
            for batch in self.loader.iter(
                'valid', n_devices=n_parallel, batch_size=None
            ):
                metrics = _predict(
                    state,
                    batch['feature'],
                    batch['label'],
                    jnp.repeat(False, n_parallel)
                    if n_parallel > 1
                    else jnp.bool(False),
                )
                metrics_valid.append(metrics)
                if isinstance(metrics, ClassificationMetrics):
                    logger.info(
                        f'Epoch {epoch} | Validation Loss: {metrics.cross_entropy} '
                        f'| Accuracy: {metrics.accuracy}'
                    )
                    valid_losses = jnp.append(
                        valid_losses,
                        metrics.cross_entropy[..., None]
                        if n_parallel > 1
                        else metrics.cross_entropy[..., None, None],
                        axis=-1,
                    )
                if isinstance(metrics, RegressionMetrics):
                    logger.info(
                        f'Epoch {epoch} | Validation Loss: {metrics.nlll} '
                        f'| RMSE: {metrics.rmse}'
                    )
                    valid_losses = jnp.append(
                        valid_losses,
                        metrics.nlll[..., None]
                        if n_parallel > 1
                        else metrics.nlll[..., None, None],
                        axis=-1,
                    )
                _stop_n = _stop_n + earlystop(
                    losses=valid_losses, patience=self.config_warmstart.patience
                )
                logger.info(f'Early stopping status at epoch {epoch}: {_stop_n}')

        # Test once after training is done.
        # NOTE: In large datasets full-batch pass might be infeasible.
        for batch in self.loader.iter('test', n_devices=n_parallel, batch_size=None):
            metrics = _predict(
                state,
                batch['feature'],
                batch['label'],
                jnp.repeat(False, n_parallel) if n_parallel > 1 else jnp.bool(False),
            )
            metrics_test.append(metrics)
            if isinstance(metrics, ClassificationMetrics):
                logger.info(
                    f'Epoch {epoch} | Test Loss: {metrics.cross_entropy} '
                    f'| Accuracy: {metrics.accuracy}'
                )
            if isinstance(metrics, RegressionMetrics):
                logger.info(
                    f'Epoch {epoch} | Test Loss: {metrics.nlll} '
                    f'| RMSE: {metrics.rmse}'
                )
        if isinstance(metrics, ClassificationMetrics):
            metrics_store = MetricsStore(
                train=ClassificationMetrics.cstack(metrics_train),
                valid=ClassificationMetrics.cstack(metrics_valid),
                test=ClassificationMetrics.cstack(metrics_test),
            )
        elif isinstance(metrics, RegressionMetrics):
            metrics_store = MetricsStore(
                train=RegressionMetrics.cstack(metrics_train),
                valid=RegressionMetrics.cstack(metrics_valid),
                test=RegressionMetrics.cstack(metrics_test),
            )
        else:
            raise ValueError('Metrics type not recognized.')

        return state, metrics_store

    # NOTE: Don't modify the name: 'time.sampling' is later used in
    # mile/inference/inference.ipynb for extracting the time from the logs.
    @measure_time('time.sampling')
    def start_sampling(self):
        """Start Sampling Phase of BDE (MCMC Sampling either Full- or Mini-Batch)."""
        if self.has_warmstart:
            if warm_exp := self.config_warmstart.warmstart_exp_dir:
                logger.info(
                    f'\t| Using Warmstart from experiment: {warm_exp.split("/")[-1]}'
                )
            warm_exp = (
                warm_exp or self.exp_dir.__str__()
            )  # We check whether we take warmstart from other exp or from current.

            warm_path = Path(warm_exp) / self.config_warmstart._dir_name

            chains = [
                warm_path / i for i in os.listdir(warm_path) if i.startswith('params')
            ]
        else:
            chains = []  # Warmstart disabled: we desire to start from random ParamTree.

        for step in self.train_plan:
            logger.info(f'\t| Starting Sampling for chains {step}')
            if chains:  # Warmstart enabled
                params = train_utils.load_params_batch(
                    params_path=[chains[i] for i in step], tree_path=self.tree_path
                )
            else:
                logger.warning(
                    '\t| No warmstart path found, starting sampling from random params.'
                )
                params = self.init_module_params(n_device=len(step))

            if not self.prob_model.minibatch:  # Full Batch Sampling
                log_post = partial(
                    self.prob_model.log_unnormalized_posterior,
                    x=self.loader.train_x,
                    y=self.loader.train_y,
                )
                inference_loop(
                    unnorm_log_posterior=log_post,
                    config=self.config_sampler,
                    rng_key=self.key,
                    init_params=params,
                    step_ids=step,
                    saving_path=self.exp_dir / self.config_sampler._dir_name,
                    saving_path_warmup=self._sampling_warmup_dir,
                )

            else:  # Mini-Batch Sampling
                raise NotImplementedError('Mini-Batch Sampling not yet implemented.')


def single_step_class(
    state: TrainState,
    x: jnp.ndarray,
    y: jnp.ndarray,
    early_stop: bool = False,
) -> tuple[TrainState, ClassificationMetrics]:
    """
    Perform a single training step for classification Task.

    Parameters
    ----------
        state (TrainState): Training State.
        x (jnp.ndarray): Input data of shape (B, ...).
        y (jnp.ndarray): Target data of shape (B,).
        early_stop (bool): Early stopping condition used in pipeline.

    Returns
    -------
        (tuple[TrainState, ClassificationMetrics]):
            Updated Training State and Classification Metrics.

    """

    def loss_fn(params: ParamTree):
        logits = state.apply_fn({'params': params}, x=x)
        loss = optax.softmax_cross_entropy(logits, jax.nn.one_hot(y, logits.shape[-1]))
        metrics = compute_metrics_class(logits, y, step=state.step)
        return loss.mean(), metrics

    def _single_step(state: TrainState):
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (_, metrics), grads = grad_fn(state.params)
        state = state.apply_gradients(grads=grads)
        return state, metrics

    def _fallback(state: TrainState):
        metrics = ClassificationMetrics(
            step=state.step, cross_entropy=jnp.nan, accuracy=jnp.nan
        )
        return state, metrics

    return jax.lax.cond(early_stop, _fallback, _single_step, state)


def single_step_regr(
    state: TrainState,
    x: jnp.ndarray,
    y: jnp.ndarray,
    early_stop: bool = False,
) -> tuple[TrainState, RegressionMetrics]:
    """
    Perform a single training step for regression Task.

    Parameters
    ----------
        state (TrainState): Training State.
        x (jnp.ndarray): Input data of shape (B, ...).
        y (jnp.ndarray): Target data of shape (B,).
        early_stop (bool): Early stopping condition used in pipeline.

    Returns
    -------
        (tuple[TrainState, RegressionMetrics]):
            Updated Training State and Regression Metrics.

    """

    def loss_fn(params: ParamTree):
        logits = state.apply_fn({'params': params}, x=x)
        loss = sandbox_metrics.GaussianNLLLoss(
            x=y,
            mu=logits[..., 0],
            sigma=jnp.exp(logits[..., 1]).clip(min=1e-6, max=1e6),
        )
        metrics = compute_metrics_regr(logits, y, step=state.step)
        return loss.mean(), metrics

    def _single_step(state: TrainState):
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (_, metrics), grads = grad_fn(state.params)
        state = state.apply_gradients(grads=grads)
        return state, metrics

    def _fallback(state: TrainState):
        metrics = RegressionMetrics(step=state.step, nlll=jnp.nan, rmse=jnp.nan)
        return state, metrics

    return jax.lax.cond(early_stop, _fallback, _single_step, state)


def predict_class(
    state: TrainState,
    x: jnp.ndarray,
    y: jnp.ndarray,
    early_stop: bool = False,
) -> ClassificationMetrics:
    """
    Predict the model for classification Task.

    Parameters
    ----------
        state (TrainState): Training State.
        x (jnp.ndarray): Input data of shape (B, ...).
        y (jnp.ndarray): Target data of shape (B,).
        early_stop (bool): Early stopping condition used in pipeline.

    Returns
    -------
        (ClassificationMetrics): Classification Metrics object containing the metrics.
    """

    def _pred(state: TrainState, x: jnp.ndarray, y: jnp.ndarray):
        logits = state.apply_fn({'params': state.params}, x=x)
        return compute_metrics_class(logits, y, step=state.step)

    def _fallback(*args, **kwargs):
        metrics = ClassificationMetrics(
            step=state.step, cross_entropy=jnp.nan, accuracy=jnp.nan
        )
        return metrics

    return jax.lax.cond(early_stop, _fallback, _pred, state, x, y)


def predict_regr(
    state: TrainState,
    x: jnp.ndarray,
    y: jnp.ndarray,
    early_stop: bool = False,
) -> RegressionMetrics:
    """Predict the model for regression Task.

    Parameters
    ----------
        state (TrainState): Training State.
        x (jnp.ndarray): Input data of shape (B, ...).
        y (jnp.ndarray): Target data of shape (B,).
        early_stop (bool): Early stopping condition used in pipeline.

    Returns
    -------
        (RegressionMetrics): Regression Metrics object containing the metrics.
    """

    def _pred(state: TrainState, x: jnp.ndarray, y: jnp.ndarray):
        logits = state.apply_fn({'params': state.params}, x=x)
        return compute_metrics_regr(logits, y, step=state.step)

    def _fallback(*args, **kwargs):
        metrics = RegressionMetrics(step=state.step, nlll=jnp.nan, rmse=jnp.nan)
        return metrics

    return jax.lax.cond(early_stop, _fallback, _pred, state, x, y)


def compute_metrics_class(
    logits: jnp.ndarray,
    y: jnp.ndarray,
    step: jnp.ndarray = jnp.nan,
) -> ClassificationMetrics:
    """Compute the metrics for classification Task.

    Parameters
    ----------
        logits (jnp.ndarray): Logits array.
        y (jnp.ndarray): Target array.
        step (jnp.ndarray): Step array, used for tracking steps in the pipeline.

    Returns
    -------
        (ClassificationMetrics): Metrics object containing recorded metrics.
    """
    y_one_hot = jax.nn.one_hot(y, num_classes=logits.shape[-1])
    cross_entroy = optax.softmax_cross_entropy(logits, y_one_hot)
    acc = jnp.mean(jnp.argmax(logits, axis=-1) == y)
    metrics = ClassificationMetrics(
        step=step, cross_entropy=cross_entroy.mean(), accuracy=acc
    )
    return metrics


def compute_metrics_regr(
    logits: jnp.ndarray,
    y: jnp.ndarray,
    step: jnp.ndarray = jnp.nan,
) -> RegressionMetrics:
    """Compute the metrics for regression Task.

    Parameters
    ----------
        logits (jnp.ndarray): Logits array.
        y (jnp.ndarray): Target array.
        step (jnp.ndarray): Step array, used for tracking steps in the pipeline.

    Returns
    -------
        (RegressionMetrics): Metrics object containing recorded metrics.
    """
    loss = sandbox_metrics.GaussianNLLLoss(
        x=y,
        mu=logits[..., 0],
        sigma=jnp.exp(logits[..., 1]).clip(min=1e-6, max=1e6),
    )
    se = sandbox_metrics.SELoss(y, logits[..., 0])
    metrics = RegressionMetrics(step=step, nlll=loss.mean(), rmse=jnp.sqrt(se.mean()))
    return metrics


def get_initial_state(
    rng: jnp.ndarray,
    x: jnp.ndarray,
    module: nn.Module,
    optimizer: optax.GradientTransformation,
) -> TrainState:
    """Get the initial training state.

    Parameters
    ----------
        rng (jnp.ndarray): rng key.
        x (jnp.ndarray): Input data for initializing the model.
        module (nn.Module): Flax Model.
        optimizer (optax.GradientTransformation): Optimizer.

    Returns
    -------
        (TrainState): Initial Training State.
    """
    params = module.init(rng, x=x)['params']
    return TrainState.create(apply_fn=module.apply, params=params, tx=optimizer)


def initialize_params(rng: jnp.ndarray, x: jnp.ndarray, module: nn.Module) -> ParamTree:
    """Get the initial parameters.

    Parameters
    ----------
        rng (jnp.ndarray): rng key.
        x (jnp.ndarray): Input data for initializing the model.
        model (nn.Module): Flax Model.

    Returns
    -------
        (ParamTree): Paramter Tree.
    """
    return module.init(rng, x=x)['params']


def earlystop(losses: jnp.ndarray, patience: int):
    """
    Check early stopping condition for losses shape (n_devices, N_steps).

    Parameters
    ----------
        losses (jnp.ndarray): Losses array.
        patience (int): Number of Patience steps for early stopping.

    Returns
    -------
        (jnp.ndarray): Boolean array of shape (n_devices,) where True indicates to stop.
    """
    if losses.shape[-1] < patience:
        return jnp.repeat(False, len(losses))
    reference_loss = losses[:, -(patience + 1)]
    reference_loss = jnp.expand_dims(reference_loss, axis=-1)
    recent_losses = losses[:, -(patience):]
    return jnp.all(recent_losses >= reference_loss, axis=1)
