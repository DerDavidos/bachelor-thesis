from sklearn.decomposition import PCA
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from mat4py import loadmat


def main():
    simulation = 0

    file = f"spikes/simulation_{simulation + 1}.npy"
    with open(file, 'rb') as f:
        aligned_spikes = np.load(f)

    aligned_spikes = aligned_spikes[:int(len(aligned_spikes) * 1)]

    print(f"Data size: {len(aligned_spikes)}, Sequence length: {len(aligned_spikes[0])}")

    data = loadmat('../Matlab/1_SimDaten/ground_truth.mat')
    classes = np.array(data["spike_classes"][simulation])

    n_classes = len(set(classes))
    print(f"Number of clusters: {n_classes}")

    train_data, val_data = train_test_split(
        aligned_spikes,
        test_size=0.15,
    )

    pca = PCA(.95)

    pca.fit(train_data)

    transformed = pca.transform(val_data)

    kmeans = KMeans(
        init="random",
        n_clusters=n_classes,
    )

    kmeans.fit(transformed)

    # print(kmeans.cluster_centers_)

    print(f"k-means inertia: {kmeans.inertia_}")

    for i, x in enumerate(kmeans.cluster_centers_):
        plt.title(f"Center of Cluster {i}")
        plt.ylim(-50, 50)
        plt.plot(x)
        plt.show()

    print("First 20 cluster labeled")
    print(kmeans.labels_[:20])

    print("Compared to real labels (numbers do not line up)")
    print(classes[:20])


if __name__ == '__main__':
    main()
