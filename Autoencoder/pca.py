from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
import data_loader
import numpy as np

""""""""""""""""""""""""""""""""
SIMULATION_NUMBER = 0
""""""""""""""""""""""""""""""""


def main(simulation_number: int):
    # Load train and test data
    train_data, _, test_data = data_loader.load_train_val_test_data(
        simulation_number=simulation_number)

    # TODO: Ground truth
    # data = loadmat('../Matlab/1_SimDaten/ground_truth.mat')
    # classes = np.array(data["spike_classes"][simulation])
    # n_classes = len(set(classes))

    n_classes = 8
    print(f"Number of clusters: {n_classes}")

    # Init PCA and KMeans
    pca = PCA(n_components=12)
    kmeans = KMeans(
        # init="random",
        n_clusters=n_classes,
    )

    # Fit PCA and KMeans to train data
    pca.fit(train_data)
    transformed_train_data = pca.transform(train_data)
    kmeans.fit(transformed_train_data)

    # print(kmeans.cluster_centers_)
    print(f"k-means inertia: {kmeans.inertia_}")

    # Put test data in Clusters
    transformed_test_data = pca.transform(test_data)
    predictions = kmeans.predict(transformed_test_data)

    # Evaluate and Plot all spikes put in the same cluster
    min_in_test_data = test_data.min()
    max_in_test_data = test_data.max()
    all_cluster = []
    for label in set(kmeans.labels_):
        cluster = []
        for i, spike in enumerate(test_data):
            if predictions[i] == label:
                cluster.append(spike)
                plt.plot(spike)
        mean_cluster = np.mean(cluster, axis=0)
        distances_in_cluster = []
        for i, spike in enumerate(cluster):
            distances_in_cluster.append(abs(mean_cluster - spike))
        all_cluster.append(np.mean(distances_in_cluster))

        plt.title(f"All spikes clustered into {label} (center of the cluster decoded in black)")
        plt.plot(mean_cluster, color="red", linewidth=2)
        plt.ylim(min_in_test_data, max_in_test_data)
        plt.show()

    print("\nMean distance from spikes to each other in each cluster")
    for i, x in enumerate(all_cluster):
        print(f"{i}: {x}")
    print(f"Cluster mean: \033[31m{np.mean(all_cluster)}\033[0m")

    # print("First 20 cluster labeled")
    # print(kmeans.labels_[:20])
    # print("Compared to real labels (numbers do not line up)")
    # print(classes[:20])


if __name__ == '__main__':
    main(simulation_number=SIMULATION_NUMBER)
