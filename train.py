"""Train Script for BDE Experiments."""
import argparse
import logging
import os
import warnings
from pathlib import Path

logger = logging.getLogger(__name__)


def train_bde(config: 'Config', n_devices: int):
    """Train a BDE model for a given configuration."""
    from src.training.trainer import BDETrainer  # noqa

    n_devices = min(n_devices, config.n_chains)
    os.environ['XLA_FLAGS'] = f'--xla_force_host_platform_device_count={n_devices}'
    logger.info(f'> Running experiment: {config.experiment_name}')
    trainer = BDETrainer(config=config)
    trainer.train_bde()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='train.py',
        description='Train a BDE model for a given configuration.',
        epilog='Example usage: python3 train.py -c configs/config_regr.yaml',
    )
    parser.add_argument(
        '--config',
        '-c',
        type=str,
        help='Path to the configuration file or directory.',
        metavar='',
        required=True,
    )
    parser.add_argument(
        '--search_tree',
        '-s',
        type=str,
        help='Path to the search tree file.',
        metavar='',
        required=False,
    )
    parser.add_argument(
        '--devices',
        '-d',
        type=int,
        metavar='',
        default=1,
        help='Number of devices to use for training.',
    )
    parser.add_argument(
        '--device_limit',
        type=int,
        metavar='',
        default=1,
        help='Maximum number of devices to use.',
    )
    parser.add_argument(
        '--silent',
        action='store_true',
        help='Disable logging to console.',
    )
    parser.add_argument(
        '--outer_parallel',
        action='store_true',
        help='Run experiments in parallel.',
        default=False,
    )

    args = parser.parse_args()
    config_path = Path(args.config)
    n_devices = int(args.devices)

    if args.silent:
        logging.basicConfig(level=logging.WARNING)

    # Import here not to slow down the --help flag
    from src.config.core import Config

    if not config_path.exists():
        raise FileNotFoundError(
            f'Configuration file or directory not found: {config_path}'
        )
    else:
        if config_path.is_dir():  # Load all configs from directory
            if args.search_tree:
                warnings.warn(
                    'Ignoring search tree file when loading directory of configs.',
                    category=UserWarning,
                )
            configs = Config.from_dir(config_path)

        else:  # Single config file with potential search tree
            cfg = Config.from_file(config_path)

            if args.search_tree:
                # Remove duplicates
                configs = list(set(cfg.get_configs_grid_from_path(args.search_tree)))
                # Enumerate experiments to make sure they are unique
                for i, cfg in enumerate(configs):
                    cfg._modify_field(experiment_name=f'{cfg.experiment_name}_{i}')
            else:
                configs = [cfg]

    logger.info(f'Loaded {len(configs)} Experiment(s)')
    if args.outer_parallel:
        import multiprocessing

        os.environ[
            'XLA_FLAGS'
        ] = '--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1'  # noqa
        print(
            'Disabling stream logging in outer parallel mode, '
            'logs will only appear in log files.'
        )
        for config in configs:
            config._modify_field(logging=False)

        with multiprocessing.Pool(n_devices) as p:
            _ = p.starmap(train_bde, [(cfg, 1) for cfg in configs])

    else:
        if args.device_limit:
            os.environ[
                'XLA_FLAGS'
            ] = f'--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads={args.device_limit} --xla_gpu_strict_conv_algorithm_picker=false'  # noqa
        else:
            os.environ[
                'XLA_FLAGS'
            ] = '--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1 --xla_gpu_strict_conv_algorithm_picker=false'  # noqa
        for cfg in configs:
            train_bde(cfg, n_devices)
