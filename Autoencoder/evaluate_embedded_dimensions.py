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

    print("Without Preprocessing")
    euclidean_no_pre, kl_no_pre = evaluate_functions.clustering_without_reduction()

    fig_path = f'images/per_dimension/type_{config.TRAIN_SIMULATION_NUMBER}{config.TRAIN_SIMULATION_TYPE}_' \
               f'{config.TEST_SIMULATION_NUMBER}{config.TEST_SIMULATION_TYPE}{config.CLUSTER}{config.DIMENSIONS}'

    plt.title(f'{config.CLUSTER} Spike Types'.replace('[', '').replace(']', ''))
    # plt.title(f'Simulation {config.TEST_SIMULATION_NUMBER}'.replace('[', '').replace(']', ''))
    plt.scatter(config.DIMENSIONS, euclidean_pca, label='PCA', marker='s', linewidth=2, color='cyan')
    plt.scatter(config.DIMENSIONS, euclidean_separate_training, label='Autoencoder Separate Training', color='orange',
                marker='^', linewidth=1)
    plt.scatter(config.DIMENSIONS, euclidean_combined_training, label='Autoencoder Combined Training',
                marker='x', color='green', linewidth=2)
    plt.axhline(y=euclidean_no_pre, label='No Feature Extraction', linestyle='dashed', color='red', linewidth=1)

    plt.legend()
    plt.xlim([config.DIMENSIONS[0] - 0.5, config.DIMENSIONS[-1] + 0.5])
    plt.ylim([0, None])
    plt.ylabel('Euclidian Distance')
    plt.xlabel('Reduced Dimension size')
    plt.savefig(f"{fig_path.replace('type', 'euclidean').replace(' ', '')}.png", bbox_inches='tight')

    with open(f"{fig_path.replace('type', 'euclidean').replace(' ', '')}.npy", 'wb') as file:
        np.save(file,
                np.array([euclidean_separate_training, euclidean_combined_training, euclidean_pca, euclidean_no_pre],
                         dtype=object))

    plt.clf()
    plt.title(f'{config.CLUSTER} Spike Types'.replace('[', '').replace(']', ''))
    # plt.title(f'Simulation {config.TEST_SIMULATION_NUMBER}'.replace('[', '').replace(']', ''))
    plt.scatter(config.DIMENSIONS, kl_pca, label='PCA', marker='s', linewidth=2, color='cyan')
    plt.scatter(config.DIMENSIONS, kl_separate_training, label='Autoencoder Separate Training', color='orange',
                marker='^', linewidth=1)
    plt.scatter(config.DIMENSIONS, kl_combined_training, label='Autoencoder Combined Training', color='green',
                marker='x', linewidth=2)
    plt.axhline(y=kl_no_pre, label='No Feature Extraction', linestyle='dashed', color='red', linewidth=1)

    plt.legend()
    plt.xlim([config.DIMENSIONS[0] - 0.5, config.DIMENSIONS[-1] + 0.5])
    plt.ylim([0, None])
    plt.ylabel('KL-Divergence')
    plt.xlabel('Reduced Dimension size')
    plt.savefig(f"{fig_path.replace('type', 'kl').replace(' ', '')}.png", bbox_inches='tight')

    with open(f"{fig_path.replace('type', 'kl').replace(' ', '')}.npy", 'wb') as file:
        np.save(file, np.array([kl_separate_training, kl_combined_training, kl_pca, kl_no_pre], dtype=object))

    print('Seperate', kl_separate_training)
    print('Combined', kl_combined_training)
    print('PCA', kl_pca)
    print('No reduction', kl_no_pre)


if __name__ == '__main__':
    evaluate_performance_per_sparsity()
