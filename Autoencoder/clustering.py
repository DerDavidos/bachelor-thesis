# import matplotlib.pyplot as plt
# from kneed import KneeLocator
# from sklearn.datasets import make_blobs
# from sklearn.metrics import silhouette_score
# from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import autoencoder
import torch
import numpy as np
from mat4py import loadmat
from matplotlib import pyplot as plt
import time

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    simulation = 0

    time_out_after_plot = 0.3

    model = torch.load(f'models/simulation_{simulation}/model.pth',
                       map_location=torch.device(DEVICE))
    model = model.to(DEVICE)

    # TODO: Ground truth
    # data = loadmat('../Matlab/1_SimDaten/ground_truth.mat')
    # classes = np.array(data["spike_classes"][simulation])

    n_classes = 8
    # n_classes = len(set(classes))

    print(f"Number of clusters: {n_classes}")

    file = f"spikes/simulation_{simulation}.npy"
    with open(file, 'rb') as f:
        aligned_spikes = np.load(f)

    aligned_spikes = aligned_spikes[:int(len(aligned_spikes) * 1)]
    encoded_data = autoencoder.encode_data(model, aligned_spikes, batch_size=len(aligned_spikes))

    kmeans = KMeans(
        # init="random",
        n_clusters=n_classes,
    )

    kmeans.fit(encoded_data)

    print(f"k-means inertia: {kmeans.inertia_}")

    # min_in_cluster_centers = kmeans.cluster_centers_.min()
    # max_in_cluster_centers = kmeans.cluster_centers_.max()
    # for i, x in enumerate(kmeans.cluster_centers_):
    #     plt.title(f"Center of Cluster {i}")
    #     plt.ylim(min_in_cluster_centers, max_in_cluster_centers)
    #     plt.plot(x)
    #     plt.show()
    #     time.sleep(time_out_after_plot)

    # print(kmeans.cluster_centers_)
    cluster_center_decoded = autoencoder.decode_data(model, kmeans.cluster_centers_,
                                                     batch_size=len(kmeans.cluster_centers_))

    min_in_aligned_spikes = aligned_spikes.min()
    max_in_aligned_spikes = aligned_spikes.max()

    for x in cluster_center_decoded:
        plt.plot(x)
    plt.legend(range(len(cluster_center_decoded)))
    plt.title(f"Center of Cluster decoded")
    plt.ylim(min_in_aligned_spikes, max_in_aligned_spikes)
    plt.show()

    for c in set(kmeans.labels_):
        for i, spike in enumerate(aligned_spikes):
            if kmeans.labels_[i] == c:
                plt.plot(spike)
        plt.title(f"All spikes clustered into {c} (center of the cluster decoded in black)")
        plt.ylim(min_in_aligned_spikes, max_in_aligned_spikes)
        plt.plot(cluster_center_decoded[c], color="black", linewidth=3)
        plt.show()
        time.sleep(time_out_after_plot)

    # print("First 20 cluster labeled")
    # print(kmeans.labels_[:20])
    # print("Compared to real labels (numbers do not line up)")
    # print(classes[:20])

    plt.show()


if __name__ == '__main__':
    main()
