import os

import numpy as np
from matplotlib import pyplot as plt

from configs import simulation as config


def show_all_spikes() -> None:
    """ Plots all spikes from the simulation defined in config.py """

    path = f'spikes/{config.SIMULATION_TYPE}/n_cluster_{config.N_CLUSTER}/' \
           f'simulation_{config.SIMULATION_NUMBER}'

    with open(f'{path}/spikes.npy', 'rb') as file:
        aligned_spikes = np.load(file)

    labels = None
    if os.path.isfile(f'{path}/labels.npy'):
        with open(f'{path}/labels.npy', 'rb') as file:
            labels = np.load(file)

    print(aligned_spikes.shape)

    min_in_test_data = np.min(aligned_spikes)
    max_in_test_data = np.max(aligned_spikes)

    plt.title(f'{path}')
    plt.ylim(np.amin(aligned_spikes), np.amax(aligned_spikes))

    # for spike in aligned_spikes:
    #    plt.plot(spike)
    # plt.show()

    if labels is not None:
        mean_per_cluster = []
        for label in set(labels):
            all_spikes_in_label = []
            for i, spike in enumerate(aligned_spikes):
                if label == labels[i]:
                    all_spikes_in_label.append(spike)
                    plt.plot(spike)
            plt.plot(np.mean(all_spikes_in_label, axis=0), color='black')
            mean_per_cluster.append(np.mean(all_spikes_in_label, axis=0))
            plt.title(f'{label}, number: {len(all_spikes_in_label)}')
            plt.show()

        for i, spike in enumerate(mean_per_cluster):
            plt.plot(spike, label=f'f{i + 1}')
        plt.legend()
        plt.show()

    for i in range(0):
        plt.plot(aligned_spikes[i])
        plt.ylim(min_in_test_data, max_in_test_data)
        plt.title(f'Example spike {i}')
        plt.ylim(np.amin(aligned_spikes), np.amax(aligned_spikes))
        plt.show()

    print(f'Number of cluster: {config.N_CLUSTER}')


if __name__ == '__main__':
    show_all_spikes()
