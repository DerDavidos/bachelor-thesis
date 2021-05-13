import numpy as np
from mat4py import loadmat
from matplotlib import pyplot as plt

""""""""""""""""""""""""""""""""
SIMULATION_NUMBER = 0
""""""""""""""""""""""""""""""""


def show_data(simulation_number: int):
    with open(f'spikes/simulation_{simulation_number + 1}.npy', 'rb') as f:
        aligned_spikes = np.load(f)

    print(aligned_spikes.shape)
    max_len = len(aligned_spikes)

    data = loadmat('../Matlab/1_SimDaten/ground_truth.mat')

    classes = data["spike_classes"][simulation_number]

    cluster = {key: [] for key in classes}

    print(f"Number of Cluster: {len(cluster)}")

    for i in range(min(100000, max_len)):
        cluster[classes[i]].append(np.array(aligned_spikes[i]))

    for i in range(len(cluster)):
        plt.title(f"Cluster {i}")
        print(f"Cluster {i}: {len(cluster[i])} Spikes")
        for x in cluster[i]:
            plt.ylim(-60, 60)
            plt.plot(x)
        plt.show()


if __name__ == '__main__':
    show_data(simulation_number=SIMULATION_NUMBER)
