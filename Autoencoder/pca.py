from sklearn.decomposition import PCA
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from mat4py import loadmat


def main():
    simulation = 0

    file = f"spikes/simulation_{simulation}.npy"
    with open(file, 'rb') as f:
        aligned_spikes = np.load(f)

    aligned_spikes = aligned_spikes[:int(len(aligned_spikes) * 1)]

    print(f"Data size: {len(aligned_spikes)}, Sequence length: {len(aligned_spikes[0])}")

    # TODO: Ground truth
    # data = loadmat('../Matlab/1_SimDaten/ground_truth.mat')
    # classes = np.array(data["spike_classes"][simulation])

    n_classes = 4
    # n_classes = len(set(classes))

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

    min_in_cluster_centers = kmeans.cluster_centers_.min()
    max_in_cluster_centers = kmeans.cluster_centers_.max()
    for i, x in enumerate(kmeans.cluster_centers_):
        plt.title(f"Center of Cluster {i}")
        plt.ylim(min_in_cluster_centers, max_in_cluster_centers)
        plt.plot(x)
        plt.show()

    min_in_aligned_spikes = val_data.min()
    max_in_aligned_spikes = val_data.max()
    for c in set(kmeans.labels_):
        for i, spike in enumerate(val_data):
            if kmeans.labels_[i] == c:
                plt.plot(spike)
        plt.title(f"All spikes clustered in {c}")
        plt.ylim(min_in_aligned_spikes, max_in_aligned_spikes)
        plt.show()

    # print("First 20 cluster labeled")
    # print(kmeans.labels_[:20])
    # print("Compared to real labels (numbers do not line up)")
    # print(classes[:20])


if __name__ == '__main__':
    main()
