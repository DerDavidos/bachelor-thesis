import numpy as np

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

    titel = f'Reduced Dimension size: {config.DIMENSIONS}'

    fig_path = f'images/per_cluster/{config.TRAIN_SIMULATION_NUMBER}{config.TRAIN_SIMULATION_TYPE}_' \
               f'{config.TEST_SIMULATION_NUMBER}{config.TEST_SIMULATION_TYPE}'

    evaluate_functions.save_plot(titel=titel.replace('[', '').replace(']', ''),
                                 pca=euclidean_pca, separate_training=euclidean_separate_training,
                                 combined_training=euclidean_combined_training, no_pre=None,
                                 fig_path=f"{fig_path}/euclidean_{config.CLUSTER}{config.DIMENSIONS}.png".replace(' ',
                                                                                                                  ''),
                                 label=['Number of Cluster (different Spike types)', 'KL-Divergence'],
                                 x_values=config.CLUSTER)

    evaluate_functions.save_plot(titel=titel.replace('[', '').replace(']', ''),
                                 pca=kl_pca, separate_training=kl_separate_training,
                                 combined_training=kl_combined_training, no_pre=None,
                                 fig_path=f"{fig_path}/kl_{config.CLUSTER}{config.DIMENSIONS}.png".replace(' ',
                                                                                                           ''),
                                 label=['Number of Cluster (different Spike types)', 'KL-Divergence'],
                                 x_values=config.CLUSTER)


if __name__ == '__main__':
    evaluate_performance_per_sparsity()
