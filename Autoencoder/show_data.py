import numpy as np
from matplotlib import pyplot as plt

import config


def show_all_spikes():
    suf = ""
    if config.SIMULATION_TYPE == "own_generated":
        suf = "_" + config.N_CLUSTER
    with open(f'spikes/{config.SIMULATION_TYPE}/simulation_{config.SIMULATION_NUMBER}{suf}.npy',
              'rb') as f:
        aligned_spikes = np.load(f)

    print(aligned_spikes.shape)

    for x in aligned_spikes:
        plt.plot(x)

    plt.title(f"All spikes in simulation {config.SIMULATION_NUMBER} from {config.SIMULATION_TYPE}")
    plt.ylim(np.amin(aligned_spikes), np.amax(aligned_spikes))
    plt.show()

    print(f"Number of cluster: {config.N_CLUSTER}")


if __name__ == '__main__':
    show_all_spikes()
