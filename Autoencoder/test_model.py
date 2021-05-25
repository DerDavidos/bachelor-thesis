import pickle

import numpy as np
import torch

import autoencoder_functions
import config

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

""""""""""""""""""""""""""""""""
TRAINED_WITH_CLUSTERING = True
""""""""""""""""""""""""""""""""


def plot_training(trained_with_clustering: bool = False) -> None:
    """Plots the training and validation loss over the epochs and some spikes encoded and then
    decoded

    Parameters:
        trained_with_clustering (int): If the training of the model to plot was perforemed with
            clustering
    """
    if trained_with_clustering:
        directory = f"models/{config.SIMULATION_TYPE}/" \
                    f"simulation_{config.SIMULATION_NUMBER}_cluster_trained/" \
                    f"sparse_{config.EMBEDDED_DIMENSION}"
    else:
        directory = f"models/{config.SIMULATION_TYPE}/" \
                    f"simulation_{config.SIMULATION_NUMBER}/" \
                    f"sparse_{config.EMBEDDED_DIMENSION}"

    model = torch.load(f"{directory}/model.pth", map_location=torch.device(DEVICE))
    model = model.to(DEVICE)

    with open(f"{directory}/train_history", 'rb') as his:
        history = pickle.load(his)

    with open(f"data/{config.SIMULATION_TYPE}/simulation_{config.SIMULATION_NUMBER}/test_data.npy",
              'rb') as file:
        test_data = np.load(file)
    test_data, _, _ = autoencoder_functions.create_dataset(test_data)

    print()
    print("Model architecture")
    print(model)

    print()
    print("Example encoded spikes")
    print(autoencoder_functions.encode_data(model=model, data=test_data[0]))
    print()

    autoencoder_functions.plot_history(history=history)
    autoencoder_functions.test_reconstructions(model=model, test_dataset=test_data)


if __name__ == '__main__':
    plot_training(trained_with_clustering=TRAINED_WITH_CLUSTERING)
