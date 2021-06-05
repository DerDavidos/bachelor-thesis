import numpy as np
import torch
from matplotlib import pyplot as plt

import autoencoder_clustering
import config
import data_loader
import evaluate_functions
import pca_clustering


def __evaluate_performance_per_sparsity(cluster, dimension):
    # Load train and test data
    train_data, _, test_data = data_loader.load_train_val_test_data()

    model = torch.load(f'models/{config.SIMULATION_TYPE}/n_cluster_{cluster}/'
                       f'simulation_{config.SIMULATION_NUMBER}_not_cluster_trained/sparse_{dimension}/model.pth')
    clusterer = autoencoder_clustering.AutoencoderClusterer(model=model, n_cluster=cluster, train_data=train_data)
    predictions = clusterer.predict(test_data)
    euclidian_per_cluster_0, kl_per_cluster_0 = \
        evaluate_functions.evaluate_clustering(data=test_data, labels=clusterer.labels, predictions=predictions)

    model = torch.load(f'models/{config.SIMULATION_TYPE}/n_cluster_{cluster}/'
                       f'simulation_{config.SIMULATION_NUMBER}_cluster_trained/sparse_{dimension}/model.pth')
    clusterer = autoencoder_clustering.AutoencoderClusterer(model=model, n_cluster=cluster, train_data=train_data)
    predictions = clusterer.predict(test_data)
    euclidian_per_cluster_1, kl_per_cluster_1 = \
        evaluate_functions.evaluate_clustering(data=test_data, labels=clusterer.labels, predictions=predictions)

    pca_clusterer = pca_clustering.PcaClusterer(n_components=dimension, n_cluster=cluster, train_data=train_data)
    predictions = pca_clusterer.predict(test_data)
    evaluate_functions.plot_cluster(data=test_data, labels=pca_clusterer.labels, predictions=predictions)
    euclidian_per_cluster_2, kl_per_cluster_2 = \
        evaluate_functions.evaluate_clustering(data=test_data, labels=pca_clusterer.labels, predictions=predictions)

    return [np.mean(euclidian_per_cluster_0), np.mean(euclidian_per_cluster_1), np.mean(euclidian_per_cluster_2)], \
           [np.mean(kl_per_cluster_0), np.mean(kl_per_cluster_1), np.mean(kl_per_cluster_2)]


def evaluate_performance_per_sparsity() -> None:
    euclidian_seperate_training = []
    euclidian_combined_training = []
    euclidian_pca = []

    kl_seperate_training = []
    kl_combined_training = []
    kl_pca = []

    for cluster in config.CLUSTER:
        euclidian, kl = __evaluate_performance_per_sparsity(cluster, dimension=8)

        euclidian_seperate_training.append(euclidian[0])
        euclidian_combined_training.append(euclidian[1])
        euclidian_pca.append(euclidian[2])

        kl_seperate_training.append(kl[0])
        kl_combined_training.append(kl[1])
        kl_pca.append(kl[2])

    plt.title(f'Euclidian distance')
    # plt.plot(euclidian_seperate_training, config.CLUSTER, label='Seperate Training', linewidth=2)
    plt.plot(euclidian_combined_training, config.CLUSTER, label='Combined Training', linewidth=2)
    plt.plot(euclidian_pca, config.CLUSTER, label='PCA', linewidth=2)
    plt.show()

    plt.title(f'KL-Divergenz')
    # plt.plot(kl_seperate_training, config.CLUSTER, label='Seperate Training', linewidth=2)
    plt.plot(kl_combined_training, config.CLUSTER, label='Combined Training', linewidth=2)
    plt.plot(kl_pca, config.CLUSTER, label='PCA', linewidth=2)
    plt.show()


if __name__ == '__main__':
    evaluate_performance_per_sparsity()
