import numpy as np
from matplotlib import pyplot as plt

from configs import config


def show_all_spikes() -> None:
    """ Plots all spikes from the simulation defined in config.py """

    with open(config.SPIKE_PATH, 'rb') as file:
        aligned_spikes = np.load(file)

    print(aligned_spikes.shape)

    min_in_test_data = np.min(aligned_spikes)
    max_in_test_data = np.max(aligned_spikes)

    plt.title(f'{config.SPIKE_PATH}')
    plt.ylim(np.amin(aligned_spikes), np.amax(aligned_spikes))

    for x in aligned_spikes:
        plt.plot(x)

    plt.show()

    for i in range(5):
        plt.plot(aligned_spikes[i])
        plt.ylim(min_in_test_data, max_in_test_data)
        plt.title(f'Example spike {i}')
        plt.ylim(np.amin(aligned_spikes), np.amax(aligned_spikes))
        plt.show()

    print(f'Number of cluster: {config.N_CLUSTER}')


if __name__ == '__main__':
    show_all_spikes()
