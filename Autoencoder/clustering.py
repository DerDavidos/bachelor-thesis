from typing import Tuple

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

import autoencoder_functions
import config
import data_loader
from autoencoder import Autoencoder

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    cluster_center_decoded = autoencoder_functions.decode_data(model=model,
                                                               data=kmeans.cluster_centers_,
                                                               batch_size=len(
                                                                   kmeans.cluster_centers_))

    # Predict cluster of test data
    test_data_encoded = autoencoder_functions.encode_data(model=model, data=test_data,
                                                          batch_size=len(test_data))
    predictions = kmeans.predict(test_data_encoded)

    # Plot all spikes put in the same cluster
    all_mean = []
    mse_per_cluster = []

    for label in set(kmeans.labels_):
        cluster = []

        for i, spike in enumerate(test_data):
            if predictions[i] == label:
                cluster.append(spike)
                plt.plot(spike)

        if len(cluster) != 0:
            mean_cluster = np.mean(cluster, axis=0)
            distances_in_cluster = []
            for i, spike in enumerate(cluster):
                distances_in_cluster.append(np.mean(np.sqrt(np.abs(mean_cluster - spike))))
            mse_per_cluster.append(np.mean(distances_in_cluster))

            if plot:
                all_mean.append(mean_cluster)
                plt.plot(mean_cluster, color="red", linewidth=2)
        else:
            mse_per_cluster.append(0)

        if plot:
            plt.title(f"All spikes clustered into {label} (center of the cluster decoded in black)")
            plt.ylim(min_in_test_data, max_in_test_data)
            plt.plot(cluster_center_decoded[label], color="black", linewidth=2)
            plt.show()

    if plot:
        plt.title(f"All cluster means")
        plt.ylim(min_in_test_data, max_in_test_data)
        for x in all_mean:
            plt.plot(x)
        plt.show()

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
    model = torch.load(f'{directory}/model.pth',
                       map_location=torch.device(DEVICE))
    model = model.to(DEVICE)

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
