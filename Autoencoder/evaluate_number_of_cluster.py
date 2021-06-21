import numpy as np
from matplotlib import pyplot as plt

import evaluate_functions
from configs import evaluate as config


def evaluate_performance_per_sparsity() -> None:
    euclidean_separate_training, euclidean_combined_training, euclidean_pca = [], [], []
    kl_separate_training, kl_combined_training, kl_pca = [], [], []

    for cluster in config.CLUSTER:

        euclidean_separate_training_intern, euclidean_combined_training_intern, euclidean_pca_intern = [], [], []
        kl_separate_training_intern, kl_combined_training_intern, kl_pca_intern = [], [], []

        for dimension in config.DIMENSIONS:
            euclidean, kl = evaluate_functions.evaluate_cluster_dimension(cluster=cluster, dimension=dimension)

            euclidean_separate_training_intern.append(euclidean[0])
            euclidean_combined_training_intern.append(euclidean[1])
            euclidean_pca_intern.append(euclidean[2])

            kl_separate_training_intern.append(kl[0])
            kl_combined_training_intern.append(kl[1])
            kl_pca_intern.append(kl[2])

        euclidean_separate_training.append(np.mean(euclidean_separate_training_intern))
        euclidean_combined_training.append(np.mean(euclidean_combined_training_intern))
        euclidean_pca.append(np.mean(euclidean_pca_intern))

        kl_separate_training.append(np.mean(kl_separate_training_intern))
        kl_combined_training.append(np.mean(kl_combined_training_intern))
        kl_pca.append(np.mean(kl_pca_intern))

    plt.title(f'Reduced Dimension size: {config.DIMENSIONS}'.replace('[', '').replace(']', ''))
    plt.scatter(config.CLUSTER, euclidean_pca, label='PCA', marker='s', linewidth=2, color='cyan')
    plt.scatter(config.CLUSTER, euclidean_separate_training, label='Autoencoder Separate Training', color='orange',
                marker='^', linewidth=1)
    plt.scatter(config.CLUSTER, euclidean_combined_training, label='Autoencoder Combined Training',
                marker='x', color='green', linewidth=2)

    plt.legend()
    plt.ylim([0, None])
    plt.ylabel('Euclidian Distance')
    plt.xlabel('Number of Cluster (different Spike types)')
    plt.savefig(
        f'images/per_cluster/euclidian_{config.TRAIN_SIMULATION_TYPE}{config.CLUSTER}{config.DIMENSIONS}.png'.replace(
            ' ',
            ''),
        bbox_inches='tight')
    plt.clf()

    plt.title(f'Reduced Dimension size: {config.DIMENSIONS}'.replace('[', '').replace(']', ''))
    plt.scatter(config.CLUSTER, kl_pca, label='PCA', marker='s', linewidth=2, color='cyan')
    plt.scatter(config.CLUSTER, kl_separate_training, label='Autoencoder Separate Training', color='orange',
                marker='^', linewidth=1)
    plt.scatter(config.CLUSTER, kl_combined_training, label='Autoencoder Combined Training', color='green',
                marker='x', linewidth=2)
    plt.legend()
    plt.ylim([0, None])
    plt.ylabel('KL-Divergence')
    plt.xlabel('Number of Cluster (different Spike types)')
    plt.savefig(
        f'images/per_cluster/kl_{config.TRAIN_SIMULATION_TYPE}{config.CLUSTER}{config.DIMENSIONS}.png'.replace(' ', ''),
        bbox_inches='tight')
    plt.clf()


if __name__ == '__main__':
    evaluate_performance_per_sparsity()
