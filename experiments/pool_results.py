"""Summarizes the results of the multiple experiments."""
import os
import pickle
import sys
from pathlib import Path

import pandas as pd

sys.path.append('../..')
from src.config.core import Config

DIR = '../../results/hyper_params_ablation' # Directory containing the results
DIR = Path(DIR)
FILTER = 'bike' # Filter the directories by this string


def main():
    """Summarizes the results of the multiple experiments."""
    results = []
    diagnostics = []
    all_dirs = os.listdir(DIR)
    filtered_dirs = [
        d for d in all_dirs if FILTER in d and 'air' not in d and 'pdf' not in d
    ]
    for path in filtered_dirs:
        config = Config.from_file(DIR / Path(path) / 'config.yaml')
        with open(DIR / Path(path) / 'metrics.pkl', 'rb') as f:
            metrics = pickle.load(f)
        tmp = {}
        tmp['total_time'] = metrics['total_time']
        tmp['de_time'] = metrics['warmstart_time']
        tmp['sampling_time'] = metrics['bde_time']

        tmp['lppd'] = metrics['lppd']
        tmp['de_lppd'] = metrics['de_lppd']
        if 'cal_error' in metrics:
            tmp['cal_error'] = metrics['cal_error']
        if 'de_cal_error' in metrics:
            tmp['de_cal_error'] = metrics['de_cal_error']
        if config.data.task == 'regr':
            tmp['rmse'] = metrics['rmse']
            tmp['de_rmse'] = metrics['de_rmse']
        elif config.data.task == 'class':
            tmp['acc'] = metrics['acc']
            tmp['de_acc'] = metrics['de_acc']

        tmp['experiment_name'] = config.experiment_name
        tmp['model_name'] = config.model.model
        if config.model.model == 'FCN':
            tmp['hidden_structure'] = '-'.join(
                [str(h) for h in config.model.hidden_structure]
            )
        tmp['data'] = config.data.path
        tmp['data'] = config.data.path.split('/')[-1].split('.')[0]
        tmp['datapoint_limit'] = config.data.datapoint_limit
        tmp['warmup_steps'] = config.training.sampler.warmup_steps
        tmp['n_samples'] = (
            config.training.sampler.n_samples / config.training.sampler.n_thinning
        )
        tmp['n_chains'] = config.training.sampler.n_chains
        tmp['n_thinning'] = config.training.sampler.n_thinning
        if config.training.sampler.name == 'nuts':
            tmp['sampler'] = 'NUTS'
            tmp['avg_acceptance_rate'] = metrics['acceptance_rate']
            tmp['avg_integration_steps'] = metrics['integration_steps']
        elif config.training.sampler.name == 'mclmc':
            tmp['sampler'] = 'MCLMC'
            tmp[
                'diagonal_preconditioning'
            ] = config.training.sampler.diagonal_preconditioning
            tmp['num_effective_samples'] = config.training.sampler.num_effective_samples
            tmp['trust_in_estimate'] = config.training.sampler.trust_in_estimate
            tmp[
                'desired_energy_var_start'
            ] = config.training.sampler.desired_energy_var_start
            tmp[
                'desired_energy_var_end'
            ] = config.training.sampler.desired_energy_var_end
            tmp['step_size_init'] = config.training.sampler.step_size_init

            tmp['step_size'] = metrics['step_size']
            tmp['step_size_std'] = metrics['step_size_std']
            tmp['L'] = metrics['L']
            tmp['L_std'] = metrics['L_std']

        tmp['rng'] = config.rng

        results.append(tmp)

        # if a diagnostics.csv file exists, read it and append to diagnostics
        if 'diagnostics.csv' in os.listdir(DIR / Path(path)):
            tmp_df = pd.read_csv(DIR / Path(path) / 'diagnostics.csv')
            # append identifier columns sampler, data, rng seed and lppd
            tmp_df['sampler'] = tmp['sampler']
            tmp_df['data'] = tmp['data']
            tmp_df['rng'] = tmp['rng']
            tmp_df['lppd'] = tmp['lppd']
            diagnostics.append(tmp_df)
    df = pd.DataFrame(results)
    filter_filler = '' if FILTER == '' else f'_{FILTER}'
    df.to_csv(DIR / f'aggr_results{filter_filler}.csv', index=False)
    if len(diagnostics) > 0:
        diagnostics_df = pd.concat(diagnostics)
        diagnostics_df.to_csv(DIR / f'aggr_diagnostics{filter_filler}.csv', index=False)


if __name__ == '__main__':
    main()
