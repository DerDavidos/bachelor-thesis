import numpy as np
from matplotlib import pyplot as plt
from mat4py import loadmat


def main():
    with open('spikes/SimulationSpikes.npy', 'rb') as f:
        aligned_spikes = np.load(f)

    print(aligned_spikes.shape)
    max_len = len(aligned_spikes)

    data = loadmat('../Matlab/1_SimDaten/ground_truth.mat')

    classes = data["spike_classes"][0]

    cluster = {key: [] for key in classes}

    print(f"Number of Cluster: {len(cluster)}")

    for i in range(min(1000, max_len)):
        cluster[classes[i]].append(np.array(aligned_spikes[i]))

    for i in range(len(cluster)):
        plt.title(f"Cluster {i}")
        print(f"Cluster {i}: {len(cluster[i])} Spikes")
        for x in cluster[i]:
            plt.plot(x)
        plt.show()


if __name__ == '__main__':
    main()
