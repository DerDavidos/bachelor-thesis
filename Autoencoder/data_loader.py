import os
import os.path
from pathlib import Path
from typing import Tuple

import numpy as np
from sklearn.model_selection import train_test_split

from configs import data_loader as config


def load_train_val_test_data(data_path: str) -> Tuple[np.array, np.array, np.array]:
    """ Loads and returns the train, validation and test data from the simulation specified in config.py

    Parameters:
        data_path (str): Path to data directory
    Returns:
        tuple (np.array): train, validation and test data
    Raises:
        FileNotFoundError: If path to data does not exist
    """

    if not os.path.exists(data_path):
        raise FileNotFoundError(f'Data does not exist at {data_path}')

    train_data, validation_data, test_data = [], [], []

    train_path = Path(f'{data_path}/train_data.npy')
    if train_path.is_file():
        with open(train_path, 'rb') as file:
            train_data = np.load(file, allow_pickle=True)

    validation_path = Path(f'{data_path}/validation_data.npy')
    if validation_path.is_file():
        with open(validation_path, 'rb') as file:
            validation_data = np.load(file, allow_pickle=True)

    test_path = Path(f'{data_path}/test_data.npy')
    if test_path.is_file():
        with open(test_path, 'rb') as file:
            test_data = np.load(file, allow_pickle=True)

    print(f'Train data size: {len(train_data)}, Validation data size: {len(validation_data)}, '
          f'Test data size: {len(test_data)}, Sequence length: {len(train_data[0])}')

    return train_data, validation_data, test_data


def save_train_val_test_data(simulation_type: str, n_cluster: str, simulation_number: str,
                             train_data: list, test_data: list, validation_data: list) -> None:
    """ Saves train, validation and test data into data folder

    Parameters:
        simulation_type (str): Type of simulation
        n_cluster (str): Number of cluster in simulation
        simulation_number (str): Number of simulation for the specific amount of cluster
        train_data (list): Train data set
        validation_data (list): Validation data set
        test_data (list): Test data set
    """

    path = f'data/{simulation_type}/n_cluster_{n_cluster}/simulation_{simulation_number}'

    Path(path).mkdir(parents=True, exist_ok=True)
    if len(train_data) != 0:
        with open(f'{path}/train_data.npy', 'wb') as file:
            np.save(file, train_data)
    if len(validation_data) != 0:
        with open(f'{path}/validation_data.npy', 'wb') as file:
            np.save(file, validation_data)
    if len(test_data) != 0:
        with open(f'{path}/test_data.npy', 'wb') as file:
            np.save(file, test_data)


def generate_all_train_val_test_data_sets(validation_percentage: float, test_percentage: float,
                                          override: bool = False) -> None:
    """ Generates train, validation and test data from files in spike folder

    Parameters:
        validation_percentage (float): Percentage of data set to be validation data
        test_percentage (float): Percentage of data set to be test data
        override (bool): Override the data for already created simulation
    """

    for spike_directory in os.listdir('spikes'):
        if spike_directory == 'README.md':
            continue
        for type_directory in os.listdir(f'spikes/{spike_directory}'):
            n_cluster = str(type_directory).split('_')[-1]
            directory = f'spikes/{spike_directory}/{type_directory}'
            for simulation_dir in os.listdir(directory):
                simulation_number = str(simulation_dir).split('_')[-1].split('.')[0]
                Path(f'data/{spike_directory}/{type_directory}/simulation_{simulation_number}').mkdir(parents=True,
                                                                                                      exist_ok=True)
                if override or \
                        len(os.listdir(f'data/{spike_directory}/{type_directory}/simulation_{simulation_number}')) == 0:

                    # Load aligned spikes
                    with open(f'{directory}/{simulation_dir}/spikes.npy', 'rb') as file:
                        aligned_spikes = np.load(file)

                    i = 0
                    while i < len(aligned_spikes):
                        if len(set(aligned_spikes[i])) == 1:
                            aligned_spikes = np.delete(aligned_spikes, i, 0)
                        else:
                            i += 1

                    print(f'{spike_directory}/{type_directory}/{simulation_dir}')
                    print(f'Data size: {len(aligned_spikes)}, Sequence length: {len(aligned_spikes[0])}')

                    validation_data = []
                    test_data = []
                    # Split data into train, validation
                    if validation_percentage != 0:
                        train_data, validation_data = train_test_split(
                            aligned_spikes, test_size=(validation_percentage + test_percentage), shuffle=False)
                        if test_percentage != 0:
                            validation_data, test_data = train_test_split(
                                validation_data,
                                test_size=(test_percentage / (validation_percentage + test_percentage)), shuffle=False)
                    else:
                        train_data = aligned_spikes

                    save_train_val_test_data(
                        simulation_type=spike_directory, n_cluster=n_cluster, simulation_number=simulation_number,
                        train_data=train_data, validation_data=validation_data, test_data=test_data)

                    print()


if __name__ == '__main__':
    generate_all_train_val_test_data_sets(validation_percentage=config.VALIDATION_PERCENTAGE,
                                          test_percentage=config.TEST_PERCENTAGE, override=config.OVERRIDE)
