import pickle

import numpy as np
import torch

import autoencoder_functions

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

""""""""""""""""""""""""""""""""
SIMULATION_NUMBER = 0
TRAINED_WITH_CLUSTERING = True
""""""""""""""""""""""""""""""""


def plot_training(simulation_number: int, trained_with_clustering: bool = False):
    if trained_with_clustering:
        directory = f"models/simulation_{simulation_number}_cluster_trained"
    else:
        directory = f"models/simulation_{simulation_number}"

    model = torch.load(f"{directory}/model.pth", map_location=torch.device(DEVICE))
    model = model.to(DEVICE)

    with open(f"{directory}/train_history", 'rb') as his:
        history = pickle.load(his)

    with open(f"data/simulation_{simulation_number}/test_data.npy", 'rb') as file:
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
    plot_training(simulation_number=SIMULATION_NUMBER,
                  trained_with_clustering=TRAINED_WITH_CLUSTERING)
