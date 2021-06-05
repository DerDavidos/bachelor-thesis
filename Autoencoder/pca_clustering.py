import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import config
import data_loader
from evaluate import evaluate_functions


class PcaClusterer:

    def __init__(self, n_components: int, n_cluster: int, train_data: np.array):
        """ Fits PCA and k-means on train data

        Parameters:
            n_components (int): Number of PCA components
            train_data (np.array): The data to fit the PCA and k-means on
            n_cluster (int): Number of cluster
        """

        self.labels = [x for x in range(n_cluster)]

        self.pca = PCA(n_components=n_components)
        self.kmeans = KMeans(n_clusters=n_cluster)
        self.pca.fit(train_data)
        transformed_train_data = self.pca.transform(train_data)
        self.kmeans.fit(transformed_train_data)

    def predict(self, data: np.array):
        transformed_test_data = self.pca.transform(data)
        predictions = self.kmeans.predict(transformed_test_data)
        return predictions


def main(evaluate: bool) -> None:
    """ Performs clustering using pca to reduce dimension on the test data set for the simulation
    defined in config.py """

    # Load train and test data
    train_data, _, test_data = data_loader.load_train_val_test_data()

    pca_clusterer = PcaClusterer(n_components=config.EMBEDDED_DIMENSION, n_cluster=config.N_CLUSTER,
                                 train_data=train_data)

    predictions = pca_clusterer.predict(test_data)

    evaluate_functions.plot_cluster(data=test_data, labels=pca_clusterer.labels,
                                    predictions=predictions)

    if evaluate:
        euclidian_per_cluster, kl_per_cluster = \
            evaluate_functions.evaluate_clustering(data=test_data, labels=pca_clusterer.labels,
                                                   predictions=predictions)

        print("\nAverage Euclidian distance from spikes to other spikes in same cluster")
        for i, x in enumerate(euclidian_per_cluster):
            print(f"{i}: {x}")
        print(f"Average: \033[31m{np.mean(euclidian_per_cluster)}\033[0m")

        print("\nAverage KL_Divergence from spikes to other spikes in same cluster")
        for i, x in enumerate(kl_per_cluster):
            print(f"{i}: {x}")
        print(f"Average: \033[31m{np.mean(kl_per_cluster)}\033[0m")


if __name__ == '__main__':
    main(evaluate=config.EVALUATE)
