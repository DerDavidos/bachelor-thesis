import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.stats import entropy
from sklearn.cluster import KMeans

import autoencoder_clustering
import data_loader
import pca_clustering
from configs import evaluate as config


def evaluate_clustering(data: np.array, labels: list, predictions: list) -> [np.array]:
    """ Evaluate the clustering

    Parameters:
        data (np.array): Spikes that have been predicted
        labels (list): All different labels
        predictions (list): Predicted label number of according spikes
    Returns:
         [np.array]: Mean distance of spikes from the mean of the cluster
    """

    kl_addition = np.min(data) * -1 + 0.00001

    euclidean_per_cluster = []
    kl_per_cluster = []

    for label in set(labels):
        cluster = []

        for i, spike in enumerate(data):
            if predictions[i] == label:
                cluster.append(spike)

        if len(cluster) != 0:
            euclidean_in_cluster = []
            kl_in_cluster = []
            for i1, spike1 in enumerate(cluster):

                spike1 = spike1 + kl_addition
                for i2, spike2 in enumerate(cluster):
                    if i1 != i2:
                        spike2 = spike2 + kl_addition
                        euclidean_in_cluster.append(np.linalg.norm(spike1 - spike2))

                        kl_in_cluster.append(entropy(spike1, spike2))

            euclidean_per_cluster.append(np.mean(euclidean_in_cluster))

            kl_per_cluster.append(np.mean(kl_in_cluster) * 100)

        else:
            euclidean_per_cluster.append(0)

    return euclidean_per_cluster, kl_per_cluster


def plot_cluster(data: np.array, labels: list, predictions: list) -> None:
    """ Plots Cluster

    Parameters:
        data (np.array): Spikes that have been predicted
        labels (list): All different labels
        predictions (list): Predicted label number of according spikes
    """

    min_in_test_data = np.min(data)
    max_in_test_data = np.max(data)

    all_mean = []

    for label in labels:
        cluster = []

        for i, spike in enumerate(data):
            if predictions[i] == label:
                cluster.append(spike)
                plt.plot(spike)

        if len(cluster) != 0:
            mean_cluster = np.mean(cluster, axis=0)
        else:
            mean_cluster = 0

        all_mean.append(mean_cluster)
        plt.title(f'All spikes clustered into {label} (cluster mean in yellow)')
        plt.plot(mean_cluster, color='yellow', linewidth=2)
        plt.ylim(min_in_test_data, max_in_test_data)
        plt.show()

    plt.title(f'All cluster means')
    plt.ylim(min_in_test_data, max_in_test_data)
    for x in all_mean:
        plt.plot(x)
    plt.show()


def evaluate_cluster_dimension(cluster: int, dimension: int) -> [list, list]:
    """ Calculates Euclidean and KL-Divergence for separate training, combined training and pca

    Parameters:
        cluster (int): Number of cluster
        dimension (int): Embedded dimension of data
    Returns:
        [list, list]: The three euclidean distances and three KL-Divergences as floats
    """

    print(f'Evaluating: Cluster: {cluster}, Dimension: {dimension}')

    # Load train and test data
    data_path = f'data/{config.TRAIN_SIMULATION_TYPE}/n_cluster_{cluster}/simulation_{config.TRAIN_SIMULATION_NUMBER}'
    train_data, _, test_data = data_loader.load_train_val_test_data(data_path)
    if config.TEST_SIMULATION_TYPE != config.TRAIN_SIMULATION_TYPE:
        data_path = f'data/{config.TEST_SIMULATION_TYPE}/n_cluster_{cluster}/simulation_{config.TEST_SIMULATION_NUMBER}'
        fit_data, _, test_data = data_loader.load_train_val_test_data(data_path)

    # Autoencoder with seperate training
    model = torch.load(f'models/{config.TRAIN_SIMULATION_TYPE}/n_cluster_{cluster}/'
                       f'simulation_{config.TRAIN_SIMULATION_NUMBER}_not_cluster_trained/sparse_{dimension}/model.pth')
    clusterer = autoencoder_clustering.AutoencoderClusterer(model=model, n_cluster=cluster, train_data=train_data)
    if config.TEST_SIMULATION_TYPE != config.TRAIN_SIMULATION_TYPE:
        clusterer.fit_kmeans(fit_data)
    predictions = clusterer.predict(test_data)
    euclidean_per_cluster_0, kl_per_cluster_0 = \
        evaluate_clustering(data=test_data, labels=clusterer.labels, predictions=predictions)

    # Autoencoder with combined training
    model = torch.load(f'models/{config.TRAIN_SIMULATION_TYPE}/n_cluster_{cluster}/'
                       f'simulation_{config.TRAIN_SIMULATION_NUMBER}_cluster_trained/sparse_{dimension}/model.pth')
    clusterer = autoencoder_clustering.AutoencoderClusterer(model=model, n_cluster=cluster, train_data=train_data)
    if config.TEST_SIMULATION_TYPE != config.TRAIN_SIMULATION_TYPE:
        clusterer.fit_kmeans(fit_data)
    predictions = clusterer.predict(test_data)
    euclidean_per_cluster_1, kl_per_cluster_1 = \
        evaluate_clustering(data=test_data, labels=clusterer.labels, predictions=predictions)

    # PCA
    pca_clusterer = pca_clustering.PcaClusterer(n_components=dimension, n_cluster=cluster, train_data=train_data)
    if config.TEST_SIMULATION_TYPE != config.TRAIN_SIMULATION_TYPE:
        pca_clusterer.fit_kmeans(fit_data)
    predictions = pca_clusterer.predict(test_data)
    euclidean_per_cluster_2, kl_per_cluster_2 = \
        evaluate_clustering(data=test_data, labels=pca_clusterer.labels, predictions=predictions)

    return [np.mean(euclidean_per_cluster_0), np.mean(euclidean_per_cluster_1), np.mean(euclidean_per_cluster_2)], \
           [np.mean(kl_per_cluster_0), np.mean(kl_per_cluster_1), np.mean(kl_per_cluster_2)]


def clustering_without_reduction() -> [float, float]:
    """ Performs k-means clustering wihtout reducing data before

    Returns:
        [float, float]: Euclidan Distance, KL-Divergence
    """
    euclidean = []
    kl = []

    for cluster in config.CLUSTER:
        print(f'Evaluating: Cluster: {cluster}')

        # Load train and test data
        if config.TEST_SIMULATION_TYPE == config.TRAIN_SIMULATION_TYPE:
            data_path = f'data/{config.TRAIN_SIMULATION_TYPE}/n_cluster_{cluster}/simulation_{config.TRAIN_SIMULATION_NUMBER}'
            train_data, _, test_data = data_loader.load_train_val_test_data(data_path)
        else:
            data_path = f'data/{config.TEST_SIMULATION_TYPE}/n_cluster_{cluster}/simulation_{config.TEST_SIMULATION_NUMBER}'
            train_data, _, test_data = data_loader.load_train_val_test_data(data_path)

        labels = [x for x in range(cluster)]

        kmeans = KMeans(n_clusters=cluster)

        kmeans.fit(train_data)

        predictions = kmeans.predict(test_data)

        euclidean_per_cluster, kl_per_cluster = evaluate_clustering(data=test_data, labels=labels,
                                                                    predictions=predictions)

        euclidean.append(np.mean(euclidean_per_cluster))
        kl.append(np.mean(kl_per_cluster))

    return np.mean(euclidean), np.mean(kl)


def __evaluate_accuracy(predictions, labels):
    pass


def evaluate_accuracy(cluster, dimension):
    print(f'Evaluating: Cluster: {cluster}, Dimension: {dimension}')

    # Load train and test data
    data_path = f'data/{config.TRAIN_SIMULATION_TYPE}/n_cluster_{cluster}/simulation_{config.TRAIN_SIMULATION_NUMBER}'
    train_data, val_data, test_data = data_loader.load_train_val_test_data(data_path)

    label_path = f'spikes/{config.TRAIN_SIMULATION_TYPE}/n_cluster_{cluster}/simulation_{config.TRAIN_SIMULATION_NUMBER}'
    with open(label_path, 'rb') as file:
        labels = np.load(file, allow_pickle=True)[len(train_data) + len(val_data)]

    if config.TEST_SIMULATION_TYPE != config.TRAIN_SIMULATION_TYPE:
        data_path = f'data/{config.TEST_SIMULATION_TYPE}/n_cluster_{cluster}/simulation_{config.TEST_SIMULATION_NUMBER}'
        fit_data, val_data, test_data = data_loader.load_train_val_test_data(data_path)

        label_path = f'spikes/{config.TEST_SIMULATION_TYPE}/n_cluster_{cluster}/simulation_{config.TEST_SIMULATION_NUMBER}'
        with open(label_path, 'rb') as file:
            labels = np.load(file, allow_pickle=True)[len(fit_data) + len(val_data)]

    # Autoencoder with seperate training
    model = torch.load(f'models/{config.TRAIN_SIMULATION_TYPE}/n_cluster_{cluster}/'
                       f'simulation_{config.TRAIN_SIMULATION_NUMBER}_not_cluster_trained/sparse_{dimension}/model.pth')
    clusterer = autoencoder_clustering.AutoencoderClusterer(model=model, n_cluster=cluster, train_data=train_data)
    if config.TEST_SIMULATION_TYPE != config.TRAIN_SIMULATION_TYPE:
        clusterer.fit_kmeans(fit_data)
    predictions = clusterer.predict(test_data)
    accuracy_seperate = __evaluate_accuracy(predictions=predictions, labels=labels)

    # Autoencoder with combined training
    model = torch.load(f'models/{config.TRAIN_SIMULATION_TYPE}/n_cluster_{cluster}/'
                       f'simulation_{config.TRAIN_SIMULATION_NUMBER}_cluster_trained/sparse_{dimension}/model.pth')
    clusterer = autoencoder_clustering.AutoencoderClusterer(model=model, n_cluster=cluster, train_data=train_data)
    if config.TEST_SIMULATION_TYPE != config.TRAIN_SIMULATION_TYPE:
        clusterer.fit_kmeans(fit_data)
    predictions = clusterer.predict(test_data)
    accuracy_combined = __evaluate_accuracy(predictions=predictions, labels=labels)

    # PCA
    pca_clusterer = pca_clustering.PcaClusterer(n_components=dimension, n_cluster=cluster, train_data=train_data)
    if config.TEST_SIMULATION_TYPE != config.TRAIN_SIMULATION_TYPE:
        pca_clusterer.fit_kmeans(fit_data)
    predictions = pca_clusterer.predict(test_data)
    accuracy_pca = __evaluate_accuracy(predictions=predictions, labels=labels)

    return accuracy_seperate, accuracy_combined, accuracy_pca
