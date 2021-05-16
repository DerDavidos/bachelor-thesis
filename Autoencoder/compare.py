import numpy as np
import torch
from mat4py import loadmat

import clustering
import data_loader
import pca

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

""""""""""""""""""""""""""""""""
SIMULATION_NUMBER = 0
""""""""""""""""""""""""""""""""


def main(simulation_number: int):
    # Load train and test data
    train_data, _, test_data = data_loader.load_train_val_test_data(
        simulation_number=simulation_number)

    # Get Number of Classes for Simulation
    ground_truth = loadmat('../Matlab/1_SimDaten/ground_truth.mat')
    classes = np.array(ground_truth["spike_classes"][simulation_number])

    n_classes = len(set(classes)) - 1
    # n_classes = 1
    print(f"Number of clusters: {n_classes}")

    print("\nAutoencoder seperate training.")
    model = torch.load(f'models/simulation_{simulation_number}/model.pth',
                       map_location=torch.device(DEVICE))
    model = model.to(DEVICE)
    _, mse_per_cluster, _ = clustering.clustering(model=model, train_data=train_data,
                                                  test_data=test_data, n_cluster=n_classes)
    print(f"Cluster mean squarred: \033[31m{np.mean(mse_per_cluster)}\033[0m")

    print("\nAutoencoder trained with clustering.")
    model = torch.load(f'models/simulation_{simulation_number}_cluster_trained/model.pth',
                       map_location=torch.device(DEVICE))
    model = model.to(DEVICE)
    _, mse_per_cluster, _ = clustering.clustering(model=model, train_data=train_data,
                                                  test_data=test_data, n_cluster=n_classes)
    print(f"Cluster mean squarred: \033[31m{np.mean(mse_per_cluster)}\033[0m")

    print("\nPCA.")
    _, mse_per_cluster = pca.pca(n_components=model.embedded_dim, train_data=train_data,
                                 test_data=test_data, n_cluster=n_classes)
    print(f"Cluster mean squarred: \033[31m{np.mean(mse_per_cluster)}\033[0m")


if __name__ == '__main__':
    main(simulation_number=SIMULATION_NUMBER)
