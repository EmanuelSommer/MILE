"""Toolbox for Handling a (flax) BNN with Blackjax."""
import logging
import pickle
from functools import partial
from pathlib import Path

import jax
import jax.experimental
import jax.numpy as jnp
from blackjax.mcmc.mclmc import MCLMCInfo
from blackjax.mcmc.nuts import NUTSInfo

from src.config.sampler import Sampler, SamplerConfig
from src.training.callbacks import (
    progress_bar_scan,
    save_position,
)
from src.training.warmup import custom_mclmc_warmup, custom_window_adaptation
from src.types import (
    Kernel,
    ParamTree,
    PosteriorFunction,
    State,
    WarmupResult,
)

logger = logging.getLogger(__name__)

Info = NUTSInfo | MCLMCInfo


def inference_loop(
    unnorm_log_posterior: PosteriorFunction,
    config: SamplerConfig,
    rng_key: jnp.ndarray,
    init_params: ParamTree,
    step_ids: jnp.ndarray,
    saving_path: Path,
    saving_path_warmup: Path | None = None,
):
    """Blackjax inference loop for full-batch sampling.

    Parameters
    ----------
    unnorm_log_posterior : PosteriorFunction
        Unnormalized log posterior function check the `src.types` module.
    config : SamplerConfig
        Sampler configuration.
    rng_key : jnp.ndarray
        PRNG key.
    init_params : ParamTree
        Initial parameters to start the sampling from.
    step_ids : jnp.ndarray
        Step ids of the chain to be sampled.
    saving_path : Path
        Path to save the sampling samples.
    saving_path_warmup : Path, optional
        Path to save the warmup samples, by default None.
    Notes
    -----
    - Currently only supports nuts & mclmc kernel.
    """
    info = {}  # Put any information you might need later for analysis
    n_devices = len(step_ids)
    rng_key, warmup_key, sample_key = jax.random.split(rng_key, 3)
    assert config.warmup_steps > 0, 'Number of warmup steps must be greater than 0.'

    # Warmup
    logger.info('> Starting Warmup sampling...')
    if config.name == Sampler.NUTS:
        warmup_state, parameters = warmup_nuts(
            kernel=config.kernel,
            config=config,
            rng_key=warmup_key,
            init_params=init_params,
            step_ids=step_ids,
            unnorm_log_posterior=unnorm_log_posterior,
            n_devices=n_devices,
            saving_path=saving_path_warmup,
        )
    elif config.name == Sampler.MCLMC:
        warmup_state, parameters = warmup_mclmc(
            config=config,
            rng_key=warmup_key,
            init_params=init_params,
            unnorm_log_posterior=unnorm_log_posterior,
            n_devices=n_devices,
        )
        # Save the warmup parameters in a .txt file
        if not saving_path.exists():
            saving_path.mkdir(parents=True)
        with open((saving_path.parent / 'warmup_params.txt'), 'w') as f:
            if parameters['step_size'].shape == ():
                parameters['step_size'] = jnp.array([parameters['step_size']])
                parameters['L'] = jnp.array([parameters['L']])
            f.write(','.join([str(sts) for sts in parameters['step_size']]) + '\n')
            f.write(','.join([str(sts) for sts in parameters['L']]) + '\n')

    else:
        raise NotImplementedError(f'{config.name} does not have a warmup implemented.')
    logger.info('> Warmup sampling completed successfully.')

    warmup_state: State = jax.block_until_ready(warmup_state)  # lets block here aswell

    # Sampling with tuned parameters
    if config.n_thinning == 1:

        def _inference_loop(
            rng_key: jnp.ndarray,
            state: State,
            parameters: ParamTree,
            step_id: jnp.ndarray,
        ) -> Info:
            def one_step(state: State, xs: tuple[jnp.ndarray, jnp.ndarray]):
                idx, rng_key = xs
                state, info = sampler.step(rng_key, state)
                # dump y to disk
                jax.experimental.io_callback(
                    partial(save_position, base=saving_path),
                    result_shape_dtypes=state.position,
                    position=state.position,
                    idx=step_id,
                    n=idx,
                )
                return state, info

            one_step_ = jax.jit(
                progress_bar_scan(
                    n_steps=config.n_samples * n_devices, name=f'{config.name} Sampling'
                )(one_step)
            )

            sampler = config.kernel(unnorm_log_posterior, **parameters)
            keys = jax.random.split(rng_key, config.n_samples)
            _, infos = jax.lax.scan(
                f=one_step_, init=state, xs=(jnp.arange(config.n_samples), keys)
            )
            return infos

    elif config.n_thinning > 1:
        # only save every n_thinning samples and thus do not scan but loop
        def _inference_loop(
            rng_key: jnp.ndarray,
            state: State,
            parameters: ParamTree,
            step_id: jnp.ndarray,
        ) -> Info:
            def one_step(state: State, xs: tuple[jnp.ndarray, jnp.ndarray]):
                idx, rng_key = xs
                state, info = sampler.step(rng_key, state)

                def save_if_thinned():
                    jax.experimental.io_callback(
                        partial(save_position, base=saving_path),
                        result_shape_dtypes=state.position,
                        position=state.position,
                        idx=step_id,
                        n=idx,
                    )
                    return None

                jax.lax.cond(
                    idx % config.n_thinning == 0, save_if_thinned, lambda: None
                )
                return state, info

            one_step_ = jax.jit(
                progress_bar_scan(
                    n_steps=config.n_samples * n_devices, name=f'{config.name} Sampling'
                )(one_step)
            )

            sampler = config.kernel(unnorm_log_posterior, **parameters)
            keys = jax.random.split(rng_key, config.n_samples)
            _, infos = jax.lax.scan(
                f=one_step_, init=state, xs=(jnp.arange(config.n_samples), keys)
            )
            return infos

    if n_devices > 1:
        runner = jax.pmap(
            _inference_loop,
            in_axes=(0, 0, 0, 0),
        )
        _rng = jax.random.split(sample_key, n_devices)
    else:
        runner = _inference_loop
        _rng = sample_key

    # Run the sampling loop
    logger.info(f'> Starting {config.name} Sampling...')
    runner_info = runner(_rng, warmup_state, parameters, step_ids)

    # Explicitly wait for the computation to finish, doesnt matter if
    # we need runner_info or not.
    runner_info: Info = jax.block_until_ready(runner_info)

    logger.info(f'> {config.name} Sampling completed successfully.')

    if config.name == Sampler.NUTS:
        info.update(
            {
                'num_integration_steps': runner_info.num_integration_steps,
                'acceptance_rate': runner_info.acceptance_rate,
                'num_trajectory_expansions': runner_info.num_trajectory_expansions,
                'is_divergent': runner_info.is_divergent,
                'energy': runner_info.energy,
                'is_turning': runner_info.is_turning,
            }
        )

    # Save information to disk
    if not saving_path.exists():
        saving_path.mkdir(parents=True)
    with open(saving_path / 'info.pkl', 'wb') as f:
        pickle.dump(info, f)

# Implementation of warmup process for Full-Batch Sampling #
def warmup_nuts(
    kernel: Kernel,
    config: SamplerConfig,
    rng_key: jnp.ndarray,
    init_params: ParamTree,
    step_ids: jnp.ndarray,
    unnorm_log_posterior: PosteriorFunction,
    n_devices: int,
    saving_path: Path | None = None,
) -> WarmupResult:
    """Perform warmup for NUTS."""
    warmup_algo = custom_window_adaptation(
        algorithm=kernel,
        logdensity_fn=unnorm_log_posterior,
        progress_bar=True,
        saving_path=saving_path,
    )
    if n_devices > 1:
        runner = jax.pmap(
            warmup_algo.run,
            in_axes=(0, 0, 0, None, None),
            static_broadcasted_argnums=(3, 4),
        )
        _rng = jax.random.split(rng_key, n_devices)
    else:
        runner = warmup_algo.run
        _rng = rng_key

    warmup_state, parameters = runner(
        _rng,
        init_params,
        jnp.repeat(step_ids, config.warmup_steps).reshape(n_devices, -1).squeeze(),
        config.warmup_steps,
        n_devices,
    )
    return warmup_state, parameters


def warmup_mclmc(
    config: SamplerConfig,
    rng_key: jnp.ndarray,
    init_params: ParamTree,
    unnorm_log_posterior: PosteriorFunction,
    n_devices: int,
) -> WarmupResult:
    """Perform warmup for MCLMC."""
    warmup_algo = custom_mclmc_warmup(
        logdensity_fn=unnorm_log_posterior,
        diagonal_preconditioning=config.diagonal_preconditioning,
        desired_energy_var_start=config.desired_energy_var_start,
        desired_energy_var_end=config.desired_energy_var_end,
        trust_in_estimate=config.trust_in_estimate,
        num_effective_samples=config.num_effective_samples,
        step_size_init=config.step_size_init,
    )
    if n_devices > 1:
        runner = jax.pmap(
            warmup_algo.run,
            in_axes=(0, 0, None),
            static_broadcasted_argnums=(2),
        )
        _rng = jax.random.split(rng_key, n_devices)
    else:
        runner = warmup_algo.run
        _rng = rng_key

    warmup_state, parameters = runner(
        _rng,
        init_params,
        config.warmup_steps,
    )
    parameters = {'step_size': parameters.step_size, 'L': parameters.L}
    return warmup_state, parameters