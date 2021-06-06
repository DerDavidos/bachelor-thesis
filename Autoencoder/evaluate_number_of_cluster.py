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

    for cluster in config.CLUSTER:
        euclidian, kl = evaluate_functions.evaluate_cluster_dimension(cluster=cluster, dimension=8)

        euclidian_seperate_training.append(euclidian[0])
        euclidian_combined_training.append(euclidian[1])
        euclidian_pca.append(euclidian[2])

        kl_seperate_training.append(kl[0])
        kl_combined_training.append(kl[1])
        kl_pca.append(kl[2])

    plt.title(f'Euclidian distance')
    plt.plot(config.CLUSTER, euclidian_seperate_training, label='Seperate Training', linewidth=2)
    plt.plot(config.CLUSTER, euclidian_combined_training, label='Combined Training', linewidth=2)
    plt.plot(config.CLUSTER, euclidian_pca, label='PCA', linewidth=2)
    plt.ylim([0, None])
    plt.legend()
    plt.show()

    plt.title(f'KL-Divergenz')
    plt.plot(config.CLUSTER, kl_seperate_training, label='Seperate Training', linewidth=2)
    plt.plot(config.CLUSTER, kl_combined_training, label='Combined Training', linewidth=2)
    plt.plot(config.CLUSTER, kl_pca, label='PCA', linewidth=2)
    plt.ylim([0, None])
    plt.legend()
    plt.show()


if __name__ == '__main__':
    evaluate_performance_per_sparsity()
