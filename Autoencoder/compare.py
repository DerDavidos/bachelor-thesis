import numpy as np
import torch

import autoencoder_clustering
import data_loader
import evaluate_functions
import pca_clustering
from configs import simulation as config


def compare() -> None:
    """ Compares the clustering of spikes using the Autoencoder, the Autoencoder trained with
    clustering and pca to reduce the data dimension beforehand """

    # Load train and test data

    train_data, _, test_data = data_loader.load_train_val_test_data(config.DATA_PATH)

    print(f'{config.SIMULATION_TYPE} simulation {config.SIMULATION_NUMBER}')
    print(f'Number of clusters: {config.N_CLUSTER}')
    print(f'Embedded dimension: {config.EMBEDDED_DIMENSION}')

    # Evaluate Autoencoder trained without clustering
    print('\nAutoencoder separate training')
    model = torch.load(f'models/{config.SIMULATION_TYPE}/n_cluster_{config.N_CLUSTER}/'
                       f'simulation_{config.SIMULATION_NUMBER}_not_cluster_trained/'
                       f'sparse_{config.EMBEDDED_DIMENSION}/model.pth')
    clusterer = autoencoder_clustering.AutoencoderClusterer(model=model, n_cluster=config.N_CLUSTER,
                                                            train_data=train_data)
    predictions = clusterer.predict(test_data)
    euclidean_per_cluster, kl_per_cluster = \
        evaluate_functions.evaluate_clustering(data=test_data, labels=clusterer.labels, predictions=predictions)
    print(f'Euclidean: \033[31m{np.mean(euclidean_per_cluster)}\033[0m')
    print(f'KL-Divergence: \033[31m{np.mean(kl_per_cluster)}\033[0m')

    # Evaluate Autoencoder trained with clustering
    print('\nAutoencoder trained with clustering')
    model = torch.load(f'models/{config.SIMULATION_TYPE}/n_cluster_{config.N_CLUSTER}/'
                       f'simulation_{config.SIMULATION_NUMBER}_cluster_trained/'
                       f'sparse_{config.EMBEDDED_DIMENSION}/model.pth')
    clusterer = autoencoder_clustering.AutoencoderClusterer(model=model, n_cluster=config.N_CLUSTER,
                                                            train_data=train_data)
    predictions = clusterer.predict(test_data)
    euclidean_per_cluster, kl_per_cluster = \
        evaluate_functions.evaluate_clustering(data=test_data, labels=clusterer.labels, predictions=predictions)
    print(f'Euclidean: \033[31m{np.mean(euclidean_per_cluster)}\033[0m')
    print(f'KL-Divergence: \033[31m{np.mean(kl_per_cluster)}\033[0m')

    # Evaluate PCA Clustering
    print('\nPCA')
    pca_clusterer = pca_clustering.PcaClusterer(n_components=config.EMBEDDED_DIMENSION,
                                                n_cluster=config.N_CLUSTER, train_data=train_data)
    predictions = pca_clusterer.predict(test_data)
    euclidean_per_cluster, kl_per_cluster = \
        evaluate_functions.evaluate_clustering(data=test_data, labels=pca_clusterer.labels, predictions=predictions)
    print(f'Euclidean: \033[31m{np.mean(euclidean_per_cluster)}\033[0m')
    print(f'KL-Divergence: \033[31m{np.mean(kl_per_cluster)}\033[0m')


if __name__ == '__main__':
    compare()
