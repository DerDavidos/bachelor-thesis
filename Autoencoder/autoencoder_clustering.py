import numpy as np
import torch
from sklearn.cluster import KMeans

import autoencoder_functions
import data_loader
import evaluate_functions
from autoencoder import Autoencoder
from configs import simulation as config


class AutoencoderClusterer:
    """ Combined Autoencoder model to reduce dimension and k-means to cluster the data """

    def __init__(self, model: Autoencoder, n_cluster: int, train_data: np.array) -> None:
        """ Initializes k-means with the train data reduced dimensionaly with the given Autoencoder model

        Parameters:
            model (Autoencoder): Autoencoder model to perform data encoding with
            train_data (np.array): The data to fit the PCA and k-means on
            n_cluster (int): Number of cluster
        """

        self.labels = [x for x in range(n_cluster)]

        self.model = model
        self.kmeans = KMeans(n_clusters=n_cluster)

        encoded_data = autoencoder_functions.encode_data(model=model, data=train_data, batch_size=len(train_data))
        self.kmeans.fit(encoded_data)

    def predict(self, data: np.array) -> np.array:
        """ Predicts cluser labels of data with Autoenocder model and k-means

        Parameters:
            data (np.array): Data to predict cluster
        Returns:
            np.array: Array of predcited labels
        """
        encoded_data = autoencoder_functions.encode_data(model=self.model, data=data, batch_size=len(data))
        predictions = self.kmeans.predict(encoded_data)
        return predictions


def main(evaluate: bool) -> None:
    """ Performs clustering using pca to reduce dimension on the test data set for the simulation defined in config.py
    """

    # Load train and test data
    train_data, _, test_data = data_loader.load_train_val_test_data(config.DATA_PATH)

    # Load model
    if config.TRAINED_WITH_CLUSTERING:
        model_path = f'models/{config.SIMULATION_TYPE}/n_cluster_{config.N_CLUSTER}/' \
                     f'simulation_{config.SIMULATION_NUMBER}_cluster_trained/sparse_{config.EMBEDDED_DIMENSION}'
    else:
        model_path = f'models/{config.SIMULATION_TYPE}/n_cluster_{config.N_CLUSTER}/' \
                     f'simulation_{config.SIMULATION_NUMBER}_not_cluster_trained/sparse_{config.EMBEDDED_DIMENSION}'
    model = torch.load(f'{model_path}/model.pth')

    pca_clusterer = AutoencoderClusterer(model=model, n_cluster=config.N_CLUSTER, train_data=train_data)

    predictions = pca_clusterer.predict(test_data)

    evaluate_functions.plot_cluster(data=test_data, labels=pca_clusterer.labels, predictions=predictions)

    if evaluate:
        euclidian_per_cluster, kl_per_cluster = \
            evaluate_functions.evaluate_clustering(data=test_data, labels=pca_clusterer.labels, predictions=predictions)

        print('\nAverage Euclidian distance from spikes to other spikes in same cluster')
        for i, x in enumerate(euclidian_per_cluster):
            print(f'{i}: {x}')
        print(f'Average: \033[31m{np.mean(euclidian_per_cluster)}\033[0m')

        print('\nAverage KL-Divergence from spikes to other spikes in same cluster')
        for i, x in enumerate(kl_per_cluster):
            print(f'{i}: {x}')
        print(f'Average: \033[31m{np.mean(kl_per_cluster)}\033[0m')


if __name__ == '__main__':
    main(evaluate=config.EVALUATE)
