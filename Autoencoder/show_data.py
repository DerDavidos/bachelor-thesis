import numpy as np
from matplotlib import pyplot as plt

import config


def show_all_spikes() -> None:
    """ Plots all spikes from the simulation defined in config.py """
    
    suf = ""
    if config.SIMULATION_TYPE == "own_generated":
        suf = "_" + str(config.N_CLUSTER)
    with open(f'spikes/{config.SIMULATION_TYPE}/simulation_{config.SIMULATION_NUMBER}{suf}.npy',
              'rb') as f:
        aligned_spikes = np.load(f)

    print(aligned_spikes.shape)

    for x in aligned_spikes:
        plt.plot(x)

    plt.show()
    plt.title(f"All spikes in simulation {config.SIMULATION_NUMBER} from {config.SIMULATION_TYPE}")
    plt.ylim(np.amin(aligned_spikes), np.amax(aligned_spikes))

    for i in range(10):
        plt.plot(aligned_spikes[i])
        plt.show()
        plt.title(f"Example spike {i}")
        plt.ylim(np.amin(aligned_spikes), np.amax(aligned_spikes))

    print(f"Number of cluster: {config.N_CLUSTER}")


if __name__ == '__main__':
    show_all_spikes()
