from typing import Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import config
import data_loader
import evaluate_functions


def pca_clustering(n_components: int, train_data: np.array, test_data: np.array, n_cluster: int,
                   plot: bool = False) -> Tuple[KMeans, list, list]:
    """ Fits PCA and k-means on train data and evaluate on test data

    Parameters:
        n_components (int): Number of PCA components
        train_data (np.array): The data to fit the PCA and k-means on
        test_data (np.array): The data the clustering is evaluated by
        n_cluster (int): Number of cluster
        plot (bool): Plot train data cluster and all mean cluster together
    Returns:
        tuple: - KMeans: K-means fitted on train data
               - list: Mean squarred error per cluster
    """

    # Init PCA and KMeans
    pca = PCA(n_components=n_components)
    kmeans = KMeans(
        n_clusters=n_cluster,
        # algorithm="elkan"
    )

    # Fit PCA and KMeans to train data
    pca.fit(train_data)
    transformed_train_data = pca.transform(train_data)

    kmeans.fit(transformed_train_data)

    # Put test data in Clusters
    transformed_test_data = pca.transform(test_data)
    predictions = kmeans.predict(transformed_test_data)

    euclidian_per_cluster, kl_per_cluster = \
        evaluate_functions.evaluate_clustering(data=test_data,
                                               labels=kmeans.labels_,
                                               predictions=predictions,
                                               plot=plot)

    return kmeans, euclidian_per_cluster, kl_per_cluster


def main():
    """ Performs clustering using pca to reduce dimension on the test data set for the simulation
    defined in config.py """

    # Load train and test data
    train_data, _, test_data = data_loader.load_train_val_test_data()

    kmeans, euclidian_per_cluster, kl_per_cluster = \
        pca_clustering(n_components=config.EMBEDDED_DIMENSION,
                       train_data=train_data, test_data=test_data,
                       n_cluster=config.N_CLUSTER, plot=True)

    # print(kmeans.cluster_centers_)
    print(f"k-means inertia: {kmeans.inertia_}")

    # Evaluate
    print("\nAverage Euclidian distance from spikes to other spikes in same cluster")
    for i, x in enumerate(euclidian_per_cluster):
        print(f"{i}: {x}")
    print(f"Average: \033[31m{np.mean(euclidian_per_cluster)}\033[0m")

    print("\nAverage KL_Divergence from spikes to other spikes in same cluster")
    for i, x in enumerate(kl_per_cluster):
        print(f"{i}: {x}")
    print(f"Average: \033[31m{np.mean(kl_per_cluster)}\033[0m")


if __name__ == '__main__':
    main()
