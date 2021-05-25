import numpy as np
import torch

import clustering
import config
import data_loader
import pca

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    # Load train and test data
    train_data, _, test_data = data_loader.load_train_val_test_data()

    print(f"{config.SIMULATION_TYPE} simulation {config.SIMULATION_NUMBER}")
    print(f"Number of clusters: {config.N_CLUSTER}")
    print(f"Embedded dimension: {config.EMBEDDED_DIMENSION}")

    print("\nAutoencoder seperate training.")
    model = torch.load(f"models/{config.SIMULATION_TYPE}/"
                       f"simulation_{config.SIMULATION_NUMBER}/"
                       f"sparse_{config.EMBEDDED_DIMENSION}/model.pth",
                       map_location=torch.device(DEVICE))
    model = model.to(DEVICE)
    _, mse_per_cluster, _ = clustering.clustering(model=model, train_data=train_data,
                                                  test_data=test_data, n_cluster=config.N_CLUSTER)
    print(f"Cluster mean squarred: \033[31m{np.mean(mse_per_cluster)}\033[0m")

    print("\nAutoencoder trained with clustering.")
    model = torch.load(f"models/{config.SIMULATION_TYPE}/"
                       f"simulation_{config.SIMULATION_NUMBER}_cluster_trained/"
                       f"sparse_{config.EMBEDDED_DIMENSION}/model.pth",
                       map_location=torch.device(DEVICE))
    model = model.to(DEVICE)
    _, mse_per_cluster, _ = clustering.clustering(model=model, train_data=train_data,
                                                  test_data=test_data, n_cluster=config.N_CLUSTER)
    print(f"Cluster mean squarred: \033[31m{np.mean(mse_per_cluster)}\033[0m")

    print("\nPCA.")
    _, mse_per_cluster = pca.pca(n_components=config.EMBEDDED_DIMENSION, train_data=train_data,
                                 test_data=test_data, n_cluster=config.N_CLUSTER)
    print(f"Cluster mean squarred: \033[31m{np.mean(mse_per_cluster)}\033[0m")


if __name__ == '__main__':
    main()