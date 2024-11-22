"""Preprocess UCI datasets according to the PGPS repository."""
import os

import numpy as np
import pandas as pd

initial_preprocess = True
cleanup_class_labels = True
data_path = 'data'
datasets = [
    'sonar',
    'wine_red',
    'wine_white',
    'heart',
    'glass',
    'australian',
    'covertype',
]


class UCIDatasets:
    """UCI Preprocessing class for loading and processing UCI datasets.

    Follows the preprocessing steps from the PGPS repository:
    https://github.com/MingzhouFan97/PGPS/blob/main/experiments/UCI/pggf/datasets.py
    of the paper:
    Fan, M., Zhou, R., Tian, C., & Qian, X. Path-Guided Particle-based Sampling.
    In Forty-first International Conference on Machine Learning.

    There you can also get the raw data files for the UCI datasets.
    """

    def __init__(self, name, data_path='', device=None):
        """Initialize the UCI dataset class."""
        self.name = name
        self.data_path = data_path
        self.device = device
        self._load_and_process_dataset()

    def _load_and_process_dataset(self):
        """Load and process the UCI dataset."""
        data_file_map = {
            'sonar': 'sonar.csv',
            'wine_red': 'winequality-red.csv',
            'wine_white': 'winequality-white.csv',
            'heart': 'processed.cleveland.data',
            'glass': 'glass.data',
            'australian': 'australian.dat',
            'covertype': 'covtype.data',
        }

        delimiter_map = {
            'sonar': ',',
            'wine_red': ';',
            'wine_white': ';',
            'heart': ',',
            'glass': ',',
            'australian': ' ',
            'covertype': ',',
        }

        # Load the raw data
        file_path = os.path.join(self.data_path, data_file_map[self.name])
        data = pd.read_csv(
            file_path, header=0, delimiter=delimiter_map[self.name]
        ).values

        # Data-specific processing
        if self.name == 'heart':
            data[data == '?'] = 0  # Handle missing values for 'heart' dataset
        elif self.name == 'glass':
            data = data[:, 1:]  # Exclude ID column in 'glass' dataset
        elif self.name == 'german':
            data = data[:, :-2]  # Exclude last two columns in 'german' dataset
        elif self.name == 'covertype':
            data = data[
                np.random.permutation(np.arange(len(data)))[:8000]
            ]  # Subsample for 'covertype'

        data = data.astype(float)  # Convert all data to float type
        x = data[:, :-1]
        y = data[:, -1].astype(int)

        # Normalize the features
        x_means, x_stds = x.mean(axis=0), x.var(axis=0) ** 0.5
        x_stds[x_stds == 0] = 1.0  # Avoid division by zero for constant features
        x_normalized = (x - x_means) / x_stds

        processed_data = np.hstack((x_normalized, y.reshape(-1, 1)))

        # Save the processed data as <datasetname>.data
        save_path = os.path.join(self.data_path, f'{self.name}_proc.data')
        np.savetxt(save_path, processed_data, delimiter=' ')
        print(f'Processed data saved to {save_path}')


if initial_preprocess:
    # main run for all datasets
    for dataset in datasets:
        UCIDatasets(dataset, data_path)

if cleanup_class_labels:
    for dataset in datasets:
        data_path = 'data'
        data_file = f'{dataset}_proc.data'
        data = pd.read_csv(
            os.path.join(data_path, data_file), header=None, delimiter=' '
        ).values
        y = data[:, -1]
        unique_classes = np.unique(y)
        class_map = {unique_classes[i]: i for i in range(len(unique_classes))}
        y = np.array([class_map[y[i]] for i in range(len(y))])
        data[:, -1] = y
        # print unique classes and number of samples per class
        unique_classes, counts = np.unique(y, return_counts=True)
        print(f'Dataset: {dataset}')
        print(f'Unique classes: {unique_classes}')
        print(f'Number of samples per class: {counts}')
        save_path = os.path.join(data_path, data_file)
        np.savetxt(save_path, data, delimiter=' ')
        print(f'Processed data saved to {save_path}')
