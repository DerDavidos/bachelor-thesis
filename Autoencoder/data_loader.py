import os
from pathlib import Path
from typing import Tuple

import numpy as np
from sklearn.model_selection import train_test_split

import config

""""""""""""""""""""""""""""""""
VALIDATION_PERCENTAGE = 0.15
TEST_PERCENTAGE = 0.15
OVERRIDE = False
""""""""""""""""""""""""""""""""


def load_train_val_test_data() -> Tuple[np.array, np.array, np.array]:
    """ Loads and returs the train, validation and test data from the simulation specified in
    config.py

    Returns:
        tuple: train, validation and test data as numpy arrays
    """

    simulation_type = config.SIMULATION_TYPE
    simulation_number = config.SIMULATION_NUMBER

    path = f"data/{simulation_type}/simulation_{simulation_number}"

    with open(f"{path}/train_data.npy", 'rb') as file:
        train_data = np.load(file, allow_pickle=True)
    with open(f"{path}/validation_data.npy", 'rb') as file:
        validation_data = np.load(file, allow_pickle=True)
    with open(f"{path}/test_data.npy", 'rb') as file:
        test_data = np.load(file, allow_pickle=True)

    print(
        f"Train data size: {len(train_data)}, Test data size: {len(test_data)}, "
        f"Sequence length: {len(train_data[0])}")

    return train_data, validation_data, test_data


def save_train_val_test_data(simulation_type: str, simulation_number, train_data: list,
                             test_data: list, validation_data: list) -> None:
    """ Saves train, validation and test data into data folder

    Parameters:
        simulation_type: Type of simulation
        simulation_number: Number of simulation
        train_data: Train data set
        validation_data: Validation data set
        test_data: Test data set
    """
    path = f"data/{simulation_type}/simulation_{simulation_number}"

    Path(path).mkdir(parents=True, exist_ok=True)
    with open(f"{path}/train_data.npy", 'wb') as file:
        np.save(file, train_data)
    with open(f"{path}/validation_data.npy", 'wb') as file:
        np.save(file, validation_data)
    with open(f"{path}/test_data.npy", 'wb') as file:
        np.save(file, test_data)


def generate_all_train_val_test_data_sets(validation_percentage: float,
                                          test_percentage: float, override: bool = False) -> None:
    """ Generates train, validation and test data from files in spike folder

    Parameters:
        validation_percentage (float): Percentage of data set to be validation data
        test_percentage (float): Percentage of data set to be test data
        override (bool): Override the data for already created simulation
    """

    for directory in os.listdir("spikes"):
        if directory == "README.md":
            break

        for simulation_file in os.listdir(f"spikes/{directory}"):

            simulation_number = str(simulation_file).split("_")[1].split(".")[0]
            Path(f"data/{directory}/simulation_{simulation_number}").mkdir(parents=True,
                                                                           exist_ok=True)
            if override or len(
                    os.listdir(f"data/{directory}/simulation_{simulation_number}")) == 0:

                # Load aligned spikes
                with open(f"spikes/{directory}/{simulation_file}/spikes.npy", 'rb') as f:
                    aligned_spikes = np.load(f)

                i = 0
                while i < len(aligned_spikes):
                    if len(set(aligned_spikes[i])) == 1:
                        aligned_spikes = np.delete(aligned_spikes, i, 0)
                    else:
                        i += 1

                print(f"{directory}/{simulation_file}")
                print(
                    f"Data size: {len(aligned_spikes)}, Sequence length: {len(aligned_spikes[0])}")

                # Split data into train, validation
                train_data, validation_data = train_test_split(
                    aligned_spikes,
                    test_size=(validation_percentage + test_percentage),
                )
                validation_data, test_data = train_test_split(
                    validation_data,
                    test_size=(test_percentage / (validation_percentage + test_percentage)),
                )

                save_train_val_test_data(
                    simulation_type=directory,
                    simulation_number=simulation_number,
                    train_data=train_data, validation_data=validation_data,
                    test_data=test_data)

                print()


if __name__ == '__main__':
    generate_all_train_val_test_data_sets(validation_percentage=VALIDATION_PERCENTAGE,
                                          test_percentage=TEST_PERCENTAGE, override=OVERRIDE)
