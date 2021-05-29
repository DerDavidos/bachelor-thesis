from typing import Tuple

import numpy as np
import torch
from sklearn.cluster import KMeans

import autoencoder_functions
import config
import data_loader
import evaluate_functions
from autoencoder import Autoencoder

""""""""""""""""""""""""""""""""
TRAINED_WITH_CLUSTERING = True
""""""""""""""""""""""""""""""""


def autoencoder_clustering(model: Autoencoder, train_data: np.ndarray, test_data: np.ndarray,
                           n_cluster: int,
                           plot: bool = False) -> Tuple[KMeans, list]:
    """ Fits k-means on train data and evaluate on test data

    Parameters:
        model (Autoencoder): The model with to reduce the dimensions
        train_data (np.ndarray): The data to fit the k-means on
        test_data (np.ndarray): The data the clustering is evaluated by
        n_cluster (int): Number of cluster
        plot (bool): Plot train data cluster and all mean cluster together
    Returns:
        tuple: - KMeans: K-Means fitted on train data
               - list: Mean squarred error per cluster
    """

    # Get min and max values for plot limit settings
    if plot:
        min_in_test_data = np.min(test_data)
        max_in_test_data = np.max(test_data)

    # Init KMeans and fit it to the sparse representation of the training data
    kmeans = KMeans(
        # init="random",
        n_clusters=n_cluster,
    )

    encoded_data = autoencoder_functions.encode_data(model=model, data=train_data,
                                                     batch_size=len(train_data))
    kmeans.fit(encoded_data)

    # Predict cluster of test data
    test_data_encoded = autoencoder_functions.encode_data(model=model, data=test_data,
                                                          batch_size=len(test_data))
    predictions = kmeans.predict(test_data_encoded)

    mse_per_cluster = evaluate_functions.evaluate_clustering(data=test_data, labels=kmeans.labels_,
                                                             predictions=predictions, plot=plot)

    return kmeans, mse_per_cluster


def main(trained_with_clustering: bool) -> None:
    """ Performs clustering using the trained autoencoder to reduce dimension on the test data set
    for the simulation defined in config.py

    Parameters:
        trained_with_clustering:
    """
    if trained_with_clustering:
        directory = f"models/{config.SIMULATION_TYPE}/" \
                    f"simulation_{config.SIMULATION_NUMBER}_cluster_trained/" \
                    f"sparse_{config.EMBEDDED_DIMENSION}"
    else:
        directory = f"models/{config.SIMULATION_TYPE}/" \
                    f"simulation_{config.SIMULATION_NUMBER}/" \
                    f"sparse_{config.EMBEDDED_DIMENSION}"

    # Load train and test data
    train_data, _, test_data = data_loader.load_train_val_test_data()

    # Load model
    model = torch.load(f'{directory}/model.pth')
    model = model

    print(f"Number of clusters: {config.N_CLUSTER}")

    kmeans, mse_per_cluster = autoencoder_clustering(model=model,
                                                     train_data=train_data,
                                                     test_data=test_data,
                                                     n_cluster=config.N_CLUSTER,
                                                     plot=True)

    # Evaluate
    print(f"k-means inertia: {kmeans.inertia_}")

    print("\nSquared mean distance from spikes in each cluster to cluster mean")
    for i, x in enumerate(mse_per_cluster):
        print(f"{i}: {x or 'Nan'}")
    print(f"Mean of clusters: \033[31m{np.mean(mse_per_cluster)}\033[0m")


if __name__ == '__main__':
    main(trained_with_clustering=TRAINED_WITH_CLUSTERING)
