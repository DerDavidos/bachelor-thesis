import numpy as np
from matplotlib import pyplot as plt

import evaluate_functions
from configs import evaluate as config


def evaluate_performance_per_sparsity() -> None:
    euclidean_separate_training, euclidean_combined_training, euclidean_pca = [], [], []
    kl_separate_training, kl_combined_training, kl_pca = [], [], []

    for dimension in config.DIMENSIONS:

        euclidean_separate_training_intern, euclidean_combined_training_intern, euclidean_pca_intern = [], [], []
        kl_separate_training_intern, kl_combined_training_intern, kl_pca_intern = [], [], []

        for cluster in config.CLUSTER:
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

    plt.title(f'Euclidean')
    plt.plot(config.DIMENSIONS, euclidean_separate_training, label='separate Training', marker='o', linewidth=2)
    plt.plot(config.DIMENSIONS, euclidean_combined_training, label='Combined Training', marker='o', linewidth=2)
    plt.plot(config.DIMENSIONS, euclidean_pca, label='PCA', marker='o', linewidth=2)
    plt.legend()
    plt.ylim([0, None])
    plt.show()

    print(kl_separate_training)
    print(kl_combined_training)
    print(kl_pca)

    plt.title(f'KL-Divergence')
    plt.plot(config.DIMENSIONS, kl_separate_training, label='separate Training', marker='o', linewidth=2)
    plt.plot(config.DIMENSIONS, kl_combined_training, label='Combined Training', marker='o', linewidth=2)
    plt.plot(config.DIMENSIONS, kl_pca, label='PCA', marker='o', linewidth=2)
    plt.legend()
    plt.ylim([0, None])
    plt.show()


if __name__ == '__main__':
    evaluate_performance_per_sparsity()
