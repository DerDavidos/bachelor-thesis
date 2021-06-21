import pickle
from pathlib import Path

import torch

import autoencoder
import autoencoder_training
import data_loader
from configs import training as config


def train_model(train_with_clustering: bool, n_cluster_simulation: int, simulation_number: int,
                embedded_dim: int) -> None:
    """ Trains the Autoencoder PyTorch model from autoencoder_py.py with the setting in training.py

    Parameters:
        train_with_clustering (bool):
        embedded_dim (int):
        simulation_number (int):
        n_cluster_simulation (int):
    Raises:
        ValueError: If the batch size is too big for the tran/validation data
    """

    # Set path to save model to
    if train_with_clustering:
        model_path = f'models/{config.SIMULATION_TYPE}/n_cluster_{n_cluster_simulation}/' \
                     f'simulation_{simulation_number}_cluster_trained/sparse_{embedded_dim}'
    else:
        model_path = f'models/{config.SIMULATION_TYPE}/n_cluster_{n_cluster_simulation}/' \
                     f'simulation_{simulation_number}_not_cluster_trained/sparse_{embedded_dim}'
    print(model_path)

    # Get train and validation data
    Path(model_path).mkdir(parents=True, exist_ok=True)

    data_path = f'data/{config.SIMULATION_TYPE}/n_cluster_{n_cluster_simulation}/simulation_{simulation_number}'
    train_data, validation_data, _ = data_loader.load_train_val_test_data(data_path=data_path)

    # Transform train and validation data to tensors for training
    input_dim = len(train_data[0])
    train_data = train_data[:(len(train_data) - (len(train_data) % config.BATCH_SIZE))]
    if not train_data.any():
        raise ValueError('Batch size to big for train split.')
    train_data = [torch.tensor(s).unsqueeze(1).float() for s in train_data]
    train_data = torch.utils.data.DataLoader(train_data, batch_size=config.BATCH_SIZE, shuffle=True)

    validation_data = validation_data[:len(validation_data) - (len(validation_data) % config.BATCH_SIZE)]
    if not validation_data.any():
        raise ValueError('Batch size to big for validation split.')
    validation_data = [torch.tensor(s).unsqueeze(1).float() for s in validation_data]
    validation_data = torch.utils.data.DataLoader(validation_data, batch_size=config.BATCH_SIZE, shuffle=True)

    # Create Autoencoder model
    model = autoencoder.Autoencoder(input_dim=input_dim, embedded_dim=embedded_dim)

    # Train Autoencoder
    history = autoencoder_training.train_model(
        model=model, train_dataset=train_data, validation_dataset=validation_data, model_path=model_path,
        train_with_clustering=train_with_clustering, n_cluster=n_cluster_simulation,
    )

    # Save training history
    with open(f'{model_path}/train_history', 'wb') as his:
        pickle.dump(history, his, pickle.HIGHEST_PROTOCOL)

    print()
    print()


if __name__ == '__main__':
    # Train model with and without combined clustering loss
    for n_cluster in config.N_CLUSTER_SIMULATION:
        for sim_number in config.SIMULATION_NUMBER:
            for clustering in config.CLUSTERING:
                for embedded in config.EMBEDDED_DIMENSIONS:
                    train_model(train_with_clustering=clustering, n_cluster_simulation=n_cluster,
                                simulation_number=sim_number, embedded_dim=embedded)
