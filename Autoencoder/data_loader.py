import os
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split


def load_train_val_test_data(simulation_number: int):
    with open(f"data/simulation_{simulation_number}/train_data.npy", 'rb') as file:
        train_data = np.load(file)
    with open(f"data/simulation_{simulation_number}/validation_data.npy", 'rb') as file:
        validation_data = np.load(file)
    with open(f"data/simulation_{simulation_number}/test_data.npy", 'rb') as file:
        test_data = np.load(file)

    print(
        f"Train data size: {len(train_data)}, Test data size: {len(test_data)}, "
        f"Sequence length: {len(train_data[0])}")

    return train_data, validation_data, test_data


def save_train_val_test_data(simulation_number, test_data, train_data, validation_data):
    Path(f"data/simulation_{simulation_number}").mkdir(parents=True, exist_ok=True)
    with open(f"data/simulation_{simulation_number}/train_data.npy", 'wb') as file:
        np.save(file, train_data)
    with open(f"data/simulation_{simulation_number}/validation_data.npy", 'wb') as file:
        np.save(file, validation_data)
    with open(f"data/simulation_{simulation_number}/test_data.npy", 'wb') as file:
        np.save(file, test_data)


def generate_all_train_val_test_data_sets():
    for simulation_file in os.listdir("spikes"):
        # Load aligned spikes
        with open(f"spikes/{simulation_file}", 'rb') as f:
            aligned_spikes = np.load(f)
        print(f"Data size: {len(aligned_spikes)}, Sequence length: {len(aligned_spikes[0])}")

        # Split data into train, validation
        train_data, validation_data = train_test_split(
            aligned_spikes,
            test_size=0.3,
        )
        validation_data, test_data = train_test_split(
            validation_data,
            test_size=0.5,
        )

        save_train_val_test_data(simulation_number=str(simulation_file).split("_")[1].split(".")[0],
                                 train_data=train_data, validation_data=validation_data,
                                 test_data=test_data)


if __name__ == '__main__':
    generate_all_train_val_test_data_sets()
