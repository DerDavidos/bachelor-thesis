import numpy as np
from matplotlib import pyplot as plt

import evaluate_functions
from configs import evaluate as config


def evaluate_performance_per_sparsity() -> None:
    euclidian_seperate_training = []
    euclidian_combined_training = []
    euclidian_pca = []

    kl_seperate_training = []
    kl_combined_training = []
    kl_pca = []

    for dimension in config.DIMENSIONS:
        euclidian_seperate_training_intern = []
        euclidian_combined_training_intern = []
        euclidian_pca_intern = []

        kl_seperate_training_intern = []
        kl_combined_training_intern = []
        kl_pca_intern = []
        for cluster in config.CLUSTER:
            euclidian, kl = evaluate_functions.evaluate_cluster_dimension(cluster=cluster, dimension=dimension)

            euclidian_seperate_training_intern.append(euclidian[0])
            euclidian_combined_training_intern.append(euclidian[1])
            euclidian_pca_intern.append(euclidian[2])

            kl_seperate_training_intern.append(kl[0])
            kl_combined_training_intern.append(kl[1])
            kl_pca_intern.append(kl[2])

        euclidian_seperate_training.append(np.mean(euclidian_seperate_training_intern))
        euclidian_combined_training.append(np.mean(euclidian_combined_training_intern))
        euclidian_pca.append(np.mean(euclidian_pca_intern))

        kl_seperate_training.append(np.mean(kl_seperate_training_intern))
        kl_combined_training.append(np.mean(kl_combined_training_intern))
        kl_pca.append(np.mean(kl_pca_intern))

    plt.title(f'Embeddeed Dimensions Euclidian')
    plt.plot(config.DIMENSIONS, euclidian_seperate_training, label='Seperate Training', linewidth=2)
    plt.plot(config.DIMENSIONS, euclidian_combined_training, label='Combined Training', linewidth=2)
    plt.plot(config.DIMENSIONS, euclidian_pca, label='PCA', linewidth=2)
    plt.legend()
    plt.ylim([0, None])
    plt.show()

    print(kl_seperate_training)
    print(kl_combined_training)
    print(kl_pca)

    plt.title(f'Embeddeed Dimensions KL')
    plt.plot(config.DIMENSIONS, kl_seperate_training, label='Seperate Training', linewidth=2)
    plt.plot(config.DIMENSIONS, kl_combined_training, label='Combined Training', linewidth=2)
    plt.plot(config.DIMENSIONS, kl_pca, label='PCA', linewidth=2)
    plt.legend()
    plt.ylim([0, None])
    plt.show()


if __name__ == '__main__':
    evaluate_performance_per_sparsity()
