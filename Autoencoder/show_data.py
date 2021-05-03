import numpy as np
from matplotlib import pyplot as plt


def main():
    with open('data/SimulationSpikes.npy', 'rb') as f:
        aligned_spikes = np.load(f)

    print(aligned_spikes[0])

    print(aligned_spikes.shape)
    for x in aligned_spikes:
        plt.plot(x)

    plt.show()


if __name__ == '__main__':
    main()
