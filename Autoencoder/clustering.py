from typing import Tuple

import numpy as np
import torch
from sklearn.cluster import KMeans

import autoencoder_functions
import config
import data_loader
import evaluate_functions
from autoencoder import Autoencoder


def autoencoder_clustering(model: Autoencoder, train_data: np.array, test_data: np.array,
                           n_cluster: int,
                           plot: bool = False) -> Tuple[KMeans, list, list]:
    """ Fits k-means on train data and evaluate on test data

    Parameters:
        model (Autoencoder): The model with to reduce the dimensions
        train_data (np.array): The data to fit the k-means on
        test_data (np.array): The data the clustering is evaluated by
        n_cluster (int): Number of cluster
        plot (bool): Plot train data cluster and all mean cluster together
    Returns:
        tuple: - KMeans: K-Means fitted on train data
               - list: Mean squarred error per cluster
    """

    # Init KMeans and fit it to the sparse representation of the training data
    kmeans = KMeans(
        n_clusters=n_cluster,
    )

    encoded_data = autoencoder_functions.encode_data(model=model, data=train_data,
                                                     batch_size=len(train_data))
    kmeans.fit(encoded_data)

    # Predict cluster of test data
    test_data_encoded = autoencoder_functions.encode_data(model=model, data=test_data,
                                                          batch_size=len(test_data))
    predictions = kmeans.predict(test_data_encoded)

    mse_per_cluster, kl_per_cluster = \
        evaluate_functions.evaluate_clustering(data=test_data, labels=kmeans.labels_,
                                               predictions=predictions, plot=plot)

    return kmeans, mse_per_cluster, kl_per_cluster


def main() -> None:
    """ Performs clustering using the trained autoencoder to reduce dimension on the test data set
    for the simulation defined in config.py """

    # Load train and test data
    train_data, _, test_data = data_loader.load_train_val_test_data()

    # Load model
    model = torch.load(f'{config.MODEL_PATH}/model.pth')

    print(f"Number of clusters: {config.N_CLUSTER}")

    kmeans, mse_per_cluster, kl_per_cluster = autoencoder_clustering(model=model,
                                                                     train_data=train_data,
                                                                     test_data=test_data,
                                                                     n_cluster=config.N_CLUSTER,
                                                                     plot=True)

    # Evaluate
    print(f"k-means inertia: {kmeans.inertia_}")

    # Evaluate
    print("\nAverage Euclidian distance from spikes to mean spikes in each cluster")
    for i, x in enumerate(mse_per_cluster):
        print(f"{i}: {x}")
    print(f"Average: \033[31m{np.mean(mse_per_cluster)}\033[0m")

    print("\nAverage KL_Divergence from spikes to other spikes in same cluster")
    for i, x in enumerate(kl_per_cluster):
        print(f"{i}: {x}")
    print(f"Average: \033[31m{np.mean(kl_per_cluster)}\033[0m")


if __name__ == '__main__':
    main()
