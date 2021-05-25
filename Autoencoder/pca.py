import math
from typing import Tuple

import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import config
import data_loader


def pca(n_components: int, train_data: np.ndarray, test_data: np.ndarray, n_cluster: int,
        plot: bool = False) -> Tuple[KMeans, list]:
    """ Fits PCA and k-means on train data and evaluate on test data

    Parameters:
        n_components (int): Number of PCA components
        train_data (np.ndarray): The data to fit the PCA and k-means on
        test_data (np.ndarray): The data the clustering is evaluated by
        n_cluster (int): Number of cluster
        plot (bool): Plot train data cluster and all mean cluster together
    Returns:
        tuple: - KMeans: K-means fitted on train data
               - list: Mean squarred error per cluster
    """

    # Init PCA and KMeans
    pca = PCA(n_components=n_components)
    kmeans = KMeans(
        # init="random",
        n_clusters=n_cluster,
    )

    # Fit PCA and KMeans to train data
    pca.fit(train_data)
    transformed_train_data = pca.transform(train_data)
    kmeans.fit(transformed_train_data)

    # Put test data in Clusters
    transformed_test_data = pca.transform(test_data)
    predictions = kmeans.predict(transformed_test_data)

    # Plot all spikes put in the same cluster
    min_in_test_data = np.min(test_data)
    max_in_test_data = np.max(test_data)
    mse_per_cluster = []
    all_mean_cluster = []
    for label in set(kmeans.labels_):
        cluster = []
        cluster_center = []
        for i, spike in enumerate(test_data):
            if predictions[i] == label:
                cluster.append(spike)
                cluster_center.append(math.sqrt(np.mean(
                    abs([label] - np.array(spike).reshape(-1)))))
                plt.plot(spike)

        if len(cluster) != 0:
            mean_cluster = np.mean(cluster, axis=0)
            distances_in_cluster = []
            for i, spike in enumerate(cluster):
                distances_in_cluster.append(np.sqrt(np.abs(mean_cluster - spike)))
            mse_per_cluster.append(np.mean(distances_in_cluster))
            if plot:
                plt.plot(mean_cluster, color="red", linewidth=2)
        else:
            mse_per_cluster.append(0)

        if plot:
            all_mean_cluster.append(mean_cluster)
            plt.title(f"All spikes clustered into {label} (center of the cluster decoded in black)")
            plt.plot(mean_cluster, color="yellow", linewidth=2)
            plt.ylim(min_in_test_data, max_in_test_data)
            plt.show()

    if plot:
        plt.title("All cluster means.")
        for x in all_mean_cluster:
            plt.plot(x)
        plt.ylim(min_in_test_data, max_in_test_data)
        plt.show()

    return kmeans, mse_per_cluster


def main():
    # Load train and test data
    train_data, _, test_data = data_loader.load_train_val_test_data()

    # if n_cluster is None:
    #     data = loadmat('../Matlab/1_SimDaten/ground_truth.mat')
    #     classes = np.array(data["spike_classes"][simulation_number])
    #     n_cluster = len(set(classes))
    #     print(f"Number of clusters: {n_cluster}")

    kmeans, mse_per_cluster = pca(train_data=train_data, test_data=test_data,
                                  n_cluster=config.N_CLUSTER,
                                  plot=True, n_components=config.EMBEDDED_DIMENSION)

    # print(kmeans.cluster_centers_)
    print(f"k-means inertia: {kmeans.inertia_}")

    # Evaluate
    print("\nMean distance from spikes to each other in each cluster")
    for i, x in enumerate(mse_per_cluster):
        print(f"{i}: {x}")
    print(f"Cluster mean: \033[31m{np.mean(mse_per_cluster)}\033[0m")


if __name__ == '__main__':
    main()
