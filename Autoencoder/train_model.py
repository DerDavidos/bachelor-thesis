import pickle
from pathlib import Path

import torch

import autoencoder
import autoencoder_training
import config
import data_loader


def train_model(train_with_clustering: bool,
                n_cluster_simulation: int = config.N_CLUSTER_SIMULATION,
                simulation_number: int = config.SIMULATION_NUMBER,
                embedded_dim: int = config.EMBEDDED_DIMENSION) -> None:
    """ Trains the Autoencoder PyTorch model from autoencoder.py with the setting in config.py

    Parameters:
        train_with_clustering (bool):
        embedded_dim (int):
        simulation_number (int):
        n_cluster_simulation (int):
    """
    if train_with_clustering:
        model_path = f"models/{config.SIMULATION_TYPE}/n_cluster_{n_cluster_simulation}/" \
                     f"simulation_{simulation_number}_cluster_trained/" \
                     f"sparse_{embedded_dim}"
    else:
        model_path = f"models/{config.SIMULATION_TYPE}/n_cluster_{n_cluster_simulation}/" \
                     f"simulation_{simulation_number}_not_cluster_trained/" \
                     f"sparse_{embedded_dim}"

    # Get train and validation data
    Path(model_path).mkdir(parents=True, exist_ok=True)

    train_data, validation_data, _ = data_loader.load_train_val_test_data()

    # Transform train and validation data to tensors for training
    input_dim = len(train_data[0])
    train_data = train_data[:(len(train_data) - (len(train_data) % config.BATCH_SIZE))]
    if not train_data.any():
        raise ValueError("Batch size to big for train split.")
    train_data = [torch.tensor(s).unsqueeze(1).float() for s in train_data]
    train_data = torch.utils.data.DataLoader(train_data, batch_size=config.BATCH_SIZE, shuffle=True)

    validation_data = validation_data[
                      :len(validation_data) - (len(validation_data) % config.BATCH_SIZE)]
    if not validation_data.any():
        raise ValueError("Batch size to big for validation split.")
    validation_data = [torch.tensor(s).unsqueeze(1).float() for s in validation_data]
    validation_data = torch.utils.data.DataLoader(validation_data, batch_size=config.BATCH_SIZE,
                                                  shuffle=True)

    # Create Autoencoder model and print it's architecture
    model = autoencoder.Autoencoder(input_dim=input_dim, embedded_dim=embedded_dim)
    model = model
    print("\nModel architecture")
    print(model)

    # Train Autoencoder
    _, history = autoencoder_training.train_model(
        model,
        train_dataset=train_data,
        validation_dataset=validation_data,
        model_path=model_path,
        train_with_clustering=config.TRAINED_WITH_CLUSTERING,
        n_cluster=config.N_CLUSTER,
    )

    # Save training history
    with open(f"{model_path}/train_history", 'wb') as his:
        pickle.dump(history, his, pickle.HIGHEST_PROTOCOL)

    # Test the model with the test data
    # test_model.plot_training(simulation_number=simulation_number,
    #                         trained_with_clustering=train_with_clustering)


if __name__ == '__main__':
    # Train model with and without combined clustering loss
    for c in config.TRAIN_CLUSTERING:
        for e in config.TRAIN_EMBEDDED_DIMENSIONS:
            train_model(train_with_clustering=c, embedded_dim=e)
