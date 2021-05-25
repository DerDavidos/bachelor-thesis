import pickle
from pathlib import Path

import torch

import autoencoder
import autoencoder_training
import config
import data_loader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

""""""""""""""""""""""""""""""""
TRAIN_WITH_CLUSTERING = [True, False]
BATCH_SIZE = 64
EPOCHS = 500
EARLY_STOPPING = 15
""""""""""""""""""""""""""""""""


def main(batch_size: int = 32, epochs: int = 1,
         train_with_clustering: bool = False,
         early_stopping: int = None):
    # if n_cluster is None and train_with_clustering:
    #     ground_truth = loadmat('../Matlab/1_SimDaten/ground_truth.mat')
    #     classes = np.array(ground_truth["spike_classes"][simulation_number])
    #     n_cluster = len(set(classes)) - 1
    #     print(f"Number of clusters: {n_cluster}")

    # Get train and validation data
    if train_with_clustering:
        directory = f"models/{config.SIMULATION_TYPE}/" \
                    f"simulation_{config.SIMULATION_NUMBER}_cluster_trained/" \
                    f"sparse_{config.EMBEDDED_DIMENSION}"
    else:
        directory = f"models/{config.SIMULATION_TYPE}/" \
                    f"simulation_{config.SIMULATION_NUMBER}/sparse_{config.EMBEDDED_DIMENSION}"
    Path(directory).mkdir(parents=True, exist_ok=True)

    train_data, validation_data, _ = data_loader.load_train_val_test_data()

    # Transform train and validation data to tensors for training
    input_dim = len(train_data[0])
    train_data = train_data[:(len(train_data) - (len(train_data) % batch_size))]
    if not train_data.any():
        raise ValueError("Batch size to big for train split.")
    train_data = [torch.tensor(s).unsqueeze(1).float() for s in train_data]
    train_data = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

    validation_data = validation_data[:len(validation_data) - (len(validation_data) % batch_size)]
    if not validation_data.any():
        raise ValueError("Batch size to big for validation split.")
    validation_data = [torch.tensor(s).unsqueeze(1).float() for s in validation_data]
    validation_data = torch.utils.data.DataLoader(validation_data, batch_size=batch_size,
                                                  shuffle=True)

    # Create Autoencoder model and print it's architecture
    model = autoencoder.Autoencoder(input_dim=input_dim, embedded_dim=config.EMBEDDED_DIMENSION)
    model = model.to(DEVICE)
    print("\nModel architecture")
    print(model)

    # Train Autoencoder
    _, history = autoencoder_training.train_model(
        model,
        train_dataset=train_data,
        validation_dataset=validation_data,
        n_epochs=epochs,
        model_path=f'{directory}/model.pth',
        batch_size=batch_size,
        train_with_clustering=train_with_clustering,
        n_cluster=config.N_CLUSTER,
        early_stopping=early_stopping,
    )

    # Save training history
    with open(f"{directory}/train_history", 'wb') as his:
        pickle.dump(history, his, pickle.HIGHEST_PROTOCOL)

    # Test the model with the test data
    # test_model.plot_training(simulation_number=simulation_number,
    #                         trained_with_clustering=train_with_clustering)


if __name__ == '__main__':
    for clustering in TRAIN_WITH_CLUSTERING:
        main(batch_size=BATCH_SIZE, epochs=EPOCHS,
             train_with_clustering=clustering, early_stopping=EARLY_STOPPING)
