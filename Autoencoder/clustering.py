from sklearn.cluster import KMeans
import torch
from matplotlib import pyplot as plt
import data_loader
import autoencoder_functions
import numpy as np
import math

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

""""""""""""""""""""""""""""""""
SIMULATION_NUMBER = 0
TRAINED_WITH_CLUSTERING = True
""""""""""""""""""""""""""""""""


def main(simulation_number: int, trained_with_clustering: bool = False):
    if trained_with_clustering:
        directory = f"models/simulation_{simulation_number}_cluster_trained"
    else:
        directory = f"models/simulation_{simulation_number}"

    # Load train and test data
    train_data, _, test_data = data_loader.load_train_val_test_data(
        simulation_number=simulation_number)

    # Load model
    model = torch.load(f'{directory}/model.pth',
                       map_location=torch.device(DEVICE))
    model = model.to(DEVICE)

    # TODO: Ground truth
    # Get Number of Classes for Simulation
    # ground_truth = loadmat('../Matlab/1_SimDaten/ground_truth.mat')
    # classes = np.array(ground_truth["spike_classes"][simulation])
    n_classes = 9
    # n_classes = len(set(classes))
    print(f"Number of clusters: {n_classes}")

    # Init KMeans and fit it to the sparse representation of the training data
    kmeans = KMeans(
        # init="random",
        n_clusters=n_classes,
    )
    encoded_data = autoencoder_functions.encode_data(model, train_data, batch_size=len(train_data))
    kmeans.fit(encoded_data)

    cluster_center_decoded = autoencoder_functions.decode_data(model, kmeans.cluster_centers_,
                                                               batch_size=len(
                                                                   kmeans.cluster_centers_))

    # Plot the decoded center of each Class to see what kind of spikes it represents
    min_in_test_data = test_data.min()
    max_in_test_data = test_data.max()
    for x in cluster_center_decoded:
        plt.plot(x)
    plt.legend(range(len(cluster_center_decoded)))
    plt.title(f"Center of Cluster decoded")
    plt.ylim(min_in_test_data, max_in_test_data)
    plt.show()

    # Predict cluster of test data
    test_data_encoded = autoencoder_functions.encode_data(model, test_data,
                                                          batch_size=len(test_data))
    predictions = kmeans.predict(test_data_encoded)

    # Evaluate and Plot all spikes put in the same cluster
    print(f"k-means inertia: {kmeans.inertia_}")

    all_cluster = []
    all_cluster_center = []
    for label in set(kmeans.labels_):
        cluster = []
        cluster_center = []
        for i, spike in enumerate(test_data):
            if predictions[i] == label:
                cluster.append(spike)
                cluster_center.append(math.sqrt(np.mean(
                    abs([label] - np.array(spike).reshape(-1)))))
                plt.plot(spike)
        all_cluster_center.append(np.mean(cluster_center))
        mean_cluster = np.mean(cluster, axis=0)
        distances_in_cluster = []
        for i, spike in enumerate(cluster):
            distances_in_cluster.append(abs(mean_cluster - spike))
        all_cluster.append(np.mean(distances_in_cluster))

        plt.title(f"All spikes clustered into {label} (center of the cluster decoded in black)")
        plt.ylim(min_in_test_data, max_in_test_data)
        plt.plot(cluster_center_decoded[label], color="black", linewidth=2)
        plt.plot(mean_cluster, color="red", linewidth=2)
        plt.show()

    print("\nMean distance from spikes to cluster center")
    for i, x in enumerate(all_cluster_center):
        print(f"{i}: {x}")
    print(f"Cluster mean: {np.mean(all_cluster_center)}")

    print("\nMean distance from spikes to each other in each cluster")
    for i, x in enumerate(all_cluster):
        print(f"{i}: {x or 'Nan'}")
    print(f"Cluster mean: \033[31m{np.mean(all_cluster)}\033[0m")

    # print("First 20 cluster labeled")
    # print(kmeans.labels_[:20])
    # print("Compared to real labels (numbers do not line up)")
    # print(classes[:20])


if __name__ == '__main__':
    main(simulation_number=SIMULATION_NUMBER, trained_with_clustering=TRAINED_WITH_CLUSTERING)
