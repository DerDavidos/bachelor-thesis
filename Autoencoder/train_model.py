import autoencoder
import numpy as np
import torch
from sklearn.model_selection import train_test_split
import pickle
import test_model
from pathlib import Path
import autoencoder_functions
import data_loader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

""""""""""""""""""""""""""""""""
TRAIN_WITH_CLUSTERING = False
N_CLUSTER = 5
BATCH_SIZE = 32
EPOCHS = 50
SIMULATIONS_TO_TRAIN = range(0, 5)
""""""""""""""""""""""""""""""""


def main(simulation: int = 0, batch_size: int = 32, epochs: int = 1,
         train_with_clustering: bool = False,
         n_cluster: int = None):
    # Load aligned spikes
    file = f"spikes/simulation_{simulation}.npy"
    with open(file, 'rb') as f:
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

    # Save train, validation and test data
    if train_with_clustering:
        directory = f"models/simulation_{simulation}_cluster_trained"
    else:
        directory = f"models/simulation_{simulation}"
    Path(directory).mkdir(parents=True, exist_ok=True)
    data_loader.save_train_val_test(directory, test_data, train_data, validation_data)

    # Transform train and validation data to tensors for training
    train_data = [torch.tensor(s).unsqueeze(1).float() for s in train_data]
    train_dataset = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_data = [torch.tensor(s).unsqueeze(1).float() for s in validation_data]
    val_dataset = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=True)

    # Create Autoencoder model and print it's architecture
    model = autoencoder.Autoencoder(input_dim=len(train_data[0]), embedded_dim=12)
    model = model.to(DEVICE)
    print("\nModel architecture")
    print(model)

    # Train Autoencoder
    model, history = autoencoder_functions.train_model(
        model,
        train_dataset=train_dataset,
        validation_dataset=val_dataset,
        n_epochs=epochs,
        model_path=f'{directory}/model.pth',
        batch_size=batch_size,
        train_with_clustering=train_with_clustering,
        n_cluster=n_cluster
    )

    # Save training history
    with open(f"{directory}/train_history", 'wb') as his:
        pickle.dump(history, his, pickle.HIGHEST_PROTOCOL)

    # Test the model with the test data
    test_dataset, _, _ = autoencoder_functions.create_dataset(test_data)
    test_model.plot_training(simulation_number=simulation,
                             trained_with_clustering=train_with_clustering)


if __name__ == '__main__':
    for i in SIMULATIONS_TO_TRAIN:
        main(simulation=i, batch_size=BATCH_SIZE, epochs=EPOCHS,
             train_with_clustering=TRAIN_WITH_CLUSTERING,
             n_cluster=N_CLUSTER)
