from sklearn.cluster import KMeans
import torch
from matplotlib import pyplot as plt
import data_loader
import autoencoder_functions

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

""""""""""""""""""""""""""""""""
SIMULATION_NUMBER = 0
""""""""""""""""""""""""""""""""


def main(simulation_number):
    # Load train and test data
    train_data, test_data = data_loader.load_train_test_data(simulation_number=simulation_number)

    # Load model
    model = torch.load(f'models/simulation_{simulation_number}/model.pth',
                       map_location=torch.device(DEVICE))
    model = model.to(DEVICE)

    # TODO: Ground truth
    # Get Number of Classes for Simulation
    # ground_truth = loadmat('../Matlab/1_SimDaten/ground_truth.mat')
    # classes = np.array(ground_truth["spike_classes"][simulation])
    n_classes = 8
    # n_classes = len(set(classes))
    print(f"Number of clusters: {n_classes}")

    # Init KMeans and fit it to the sparse representation of the training data
    kmeans = KMeans(
        # init="random",
        n_clusters=n_classes,
    )
    encoded_data = autoencoder_functions.encode_data(model, train_data, batch_size=len(train_data))
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

    # Plot the decoded center of each Class to see what kind of spikes it represents
    cluster_center_decoded = autoencoder_functions.decode_data(model, kmeans.cluster_centers_,
                                                               batch_size=len(
                                                                   kmeans.cluster_centers_))
    min_in_test_data = test_data.min()
    max_in_test_data = test_data.max()
    for x in cluster_center_decoded:
        plt.plot(x)
    plt.legend(range(len(cluster_center_decoded)))
    plt.title(f"Center of Cluster decoded")
    plt.ylim(min_in_test_data, max_in_test_data)
    plt.show()

    # Plot all spikes put in the same cluster
    test_data_encoded = autoencoder_functions.encode_data(model, test_data,
                                                          batch_size=len(test_data))
    predictions = kmeans.predict(test_data_encoded)
    for label in set(kmeans.labels_):
        for i, spike in enumerate(test_data):
            if predictions[i] == label:
                plt.plot(spike)
        plt.title(f"All spikes clustered into {label} (center of the cluster decoded in black)")
        plt.ylim(min_in_test_data, max_in_test_data)
        plt.plot(cluster_center_decoded[label], color="black", linewidth=3)
        plt.show()

    # print("First 20 cluster labeled")
    # print(kmeans.labels_[:20])
    # print("Compared to real labels (numbers do not line up)")
    # print(classes[:20])


if __name__ == '__main__':
    main(simulation_number=SIMULATION_NUMBER)
