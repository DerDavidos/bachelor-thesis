import numpy as np
from matplotlib import pyplot as plt

import config


def show_all_spikes() -> None:
    """ Plots all spikes from the simulation defined in config.py """

    suf = ""
    if config.SIMULATION_TYPE == "own_generated":
        suf = "_" + str(config.N_CLUSTER)
    with open(f'spikes/{config.SIMULATION_TYPE}/simulation_{config.SIMULATION_NUMBER}/spikes.npy',
              'rb') as f:
        aligned_spikes = np.load(f)

    with open(f'spikes/{config.SIMULATION_TYPE}/simulation_{config.SIMULATION_NUMBER}/labels.npy',
              'rb') as f:
        label = np.load(f)

    print(aligned_spikes.shape)

    for x in aligned_spikes:
        plt.plot(x)

    plt.show()
    plt.title(f"All spikes in simulation {config.SIMULATION_NUMBER} from {config.SIMULATION_TYPE}")
    plt.ylim(np.amin(aligned_spikes), np.amax(aligned_spikes))

    print(len(label))
    print(len(aligned_spikes))

    min_in_test_data = np.min(aligned_spikes)
    max_in_test_data = np.max(aligned_spikes)

    for x in set(label):
        for i in range(len(label)):
            if label[i] == x:
                plt.plot(aligned_spikes[i])
        plt.ylim(min_in_test_data, max_in_test_data)
        plt.title(f"All spikes in {x}")
        plt.show()

    for i in range(3):
        plt.plot(aligned_spikes[i])
        plt.show()
        plt.ylim(min_in_test_data, max_in_test_data)
        plt.title(f"Example spike {i}")
        plt.ylim(np.amin(aligned_spikes), np.amax(aligned_spikes))

    print(f"Number of cluster: {config.N_CLUSTER}")


if __name__ == '__main__':
    show_all_spikes()
