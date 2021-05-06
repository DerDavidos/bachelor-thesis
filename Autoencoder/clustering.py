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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    simulation = 0

    model = torch.load(f'models/model_simulation_{simulation}.pth',
                       map_location=torch.device(DEVICE))
    model = model.to(DEVICE)

    data = loadmat('../Matlab/1_SimDaten/ground_truth.mat')
    classes = np.array(data["spike_classes"][simulation])

    n_classes = len(set(classes))
    print(f"Number of clusters: {n_classes}")

    file = f"spikes/simulation_{simulation + 1}.npy"
    with open(file, 'rb') as f:
        aligned_spikes = np.load(f)

    aligned_spikes = aligned_spikes[:int(len(aligned_spikes) * 0.01)]
    encoded_data = autoencoder.encode_data(model, aligned_spikes, batch_size=len(aligned_spikes))

    kmeans = KMeans(
        init="random",
        n_clusters=n_classes,
    )

    kmeans.fit(encoded_data)

    print(f"k-means inertia: {kmeans.inertia_}")

    for i, x in enumerate(kmeans.cluster_centers_):
        plt.title(f"Center of Cluster {i}")
        plt.ylim(-1, 1)
        plt.plot(x)
        plt.show()

    # print(kmeans.cluster_centers_)
    cluster_center_decoded = autoencoder.decode_data(model, kmeans.cluster_centers_,
                                                     batch_size=len(kmeans.cluster_centers_))

    for i, x in enumerate(cluster_center_decoded):
        plt.title(f"Center of Cluster {i} decoded")
        plt.ylim(-50, 50)
        plt.plot(x)
        plt.show()

    print("First 20 cluster labeled")
    print(kmeans.labels_[:20])

    print("Compared to real labels (numbers do not line up)")
    print(classes[:20])


if __name__ == '__main__':
    main()
