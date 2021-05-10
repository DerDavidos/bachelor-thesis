import numpy as np
from matplotlib import pyplot as plt
from mat4py import loadmat


def main():
    simulation = 0

    with open(f'spikes/simulation_{simulation + 1}.npy', 'rb') as f:
        aligned_spikes = np.load(f)

    print(aligned_spikes.shape)
    max_len = len(aligned_spikes)

    data = loadmat('../Matlab/1_SimDaten/ground_truth.mat')

    classes = data["spike_classes"][simulation]

    cluster = {key: [] for key in classes}

    print(f"Number of Cluster: {len(cluster)}")

    for i in range(min(100000, max_len)):
        cluster[classes[i]].append(np.array(aligned_spikes[i]))

    all_min = min(cluster)
    all_max = max(cluster)

    for i in range(len(cluster)):
        plt.title(f"Cluster {i}")
        print(f"Cluster {i}: {len(cluster[i])} Spikes")
        for x in cluster[i]:
            #plt.ylim(all_min, all_max)
            plt.plot(x)
        plt.show()


if __name__ == '__main__':
    main()
