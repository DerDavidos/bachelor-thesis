import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import data_loader
import evaluate_functions
from configs import simulation as config


class PcaClusterer:
    """ Combined PCA to reduce dimension and k-means to cluster the data """

    def __init__(self, n_components: int, n_cluster: int, train_data: np.array) -> None:
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

    def predict(self, data: np.array) -> np.array:
        """ Predicts cluster labels of data with fitted pca and k-means

        Parameters:
            data (np.array): Data to predict cluster
        Returns:
            np.array: Array of predicted labels
        """

        transformed_test_data = self.pca.transform(data)
        predictions = self.kmeans.predict(transformed_test_data)
        return np.array(predictions)


def main(evaluate: bool) -> None:
    """ Performs clustering using pca to reduce dimension on the test data set for the simulation defined in config.py
    """

    # Load train and test data
    train_data, _, test_data = data_loader.load_train_val_test_data(config.DATA_PATH)

    pca_clusterer = PcaClusterer(n_components=config.EMBEDDED_DIMENSION, n_cluster=config.N_CLUSTER,
                                 train_data=train_data)

    predictions = pca_clusterer.predict(test_data)

    evaluate_functions.plot_cluster(data=test_data, labels=pca_clusterer.labels, predictions=predictions)

    if evaluate:
        euclidean_per_cluster, kl_per_cluster = \
            evaluate_functions.evaluate_clustering(data=test_data, labels=pca_clusterer.labels, predictions=predictions)

        print('\nAverage Euclidean distance from spikes to other spikes in same cluster')
        for i, x in enumerate(euclidean_per_cluster):
            print(f'{i}: {x}')
        print(f'Average: \033[31m{np.mean(euclidean_per_cluster)}\033[0m')

        print('\nAverage KL-Divergence from spikes to other spikes in same cluster')
        for i, x in enumerate(kl_per_cluster):
            print(f'{i}: {x}')
        print(f'Average: \033[31m{np.mean(kl_per_cluster)}\033[0m')


if __name__ == '__main__':
    main(evaluate=config.EVALUATE)
