from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
import data_loader

""""""""""""""""""""""""""""""""
SIMULATION_NUMBER = 0
""""""""""""""""""""""""""""""""


def main(simulation_number):
    # Load train and test data
    train_data, test_data = data_loader.load_train_test_data(simulation_number=simulation_number)

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

    # Plot all spikes put in the same cluster
    min_in_test_data = test_data.min()
    max_in_test_data = test_data.max()
    for label in set(kmeans.labels_):
        for i, spike in enumerate(test_data):
            if predictions[i] == label:
                plt.plot(spike)
        plt.title(f"All spikes clustered into {label} (center of the cluster decoded in black)")
        plt.ylim(min_in_test_data, max_in_test_data)
        plt.show()

    # print("First 20 cluster labeled")
    # print(kmeans.labels_[:20])
    # print("Compared to real labels (numbers do not line up)")
    # print(classes[:20])


if __name__ == '__main__':
    main(simulation_number=SIMULATION_NUMBER)
