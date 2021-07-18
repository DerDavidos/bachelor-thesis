import numpy as np

import evaluate_functions
from configs import evaluate as config


def main():
    separate_training, combined_training, pca = [], [], []
    for dimension in config.DIMENSIONS:
        acc = evaluate_functions.evaluate_accuracy(cluster=config.CLUSTER[0], dimension=dimension)
        separate_training.append(np.mean(acc[0]))
        combined_training.append(np.mean(acc[1]))
        pca.append(np.mean(acc[2]))

    print("Without Preprocessing")
    no_pre = evaluate_functions.evaluate_accuracy_without_reduction(cluster=config.CLUSTER[0])

    titel = f'Simulation {config.TEST_SIMULATION_NUMBER}, {config.CLUSTER} different Spike Types'
    if config.TEST_SIMULATION_TYPE == 'own_generated':
        titel = f'{config.CLUSTER} different Spike Types'

    fig_path = f'images/per_dimension/{config.TRAIN_SIMULATION_NUMBER}{config.TRAIN_SIMULATION_TYPE}_' \
               f'{config.TEST_SIMULATION_NUMBER}{config.TEST_SIMULATION_TYPE}'

    evaluate_functions.save_plot(titel=titel.replace('[', '').replace(']', ''),
                                 pca=pca, separate_training=separate_training,
                                 combined_training=combined_training, no_pre=no_pre,
                                 fig_path=f"{fig_path}/accuracy_{config.CLUSTER}{config.DIMENSIONS}.png".replace(' ',
                                                                                                                 ''),
                                 label=['Accuracy', 'Reduced Dimension size'], x_values=config.DIMENSIONS)

    print(max(separate_training))
    print(max(combined_training))
    print(max(pca))

    print(separate_training)
    print(combined_training)
    print(pca)


if __name__ == '__main__':
    main()
