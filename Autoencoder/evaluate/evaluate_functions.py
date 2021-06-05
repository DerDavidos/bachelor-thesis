import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import entropy


def evaluate_clustering(data: np.array, labels: list, predictions: list) -> \
        [np.array]:
    """ Evaluate the clustering

    Parameters:
        data (np.array): Spikes that have been predicted
        labels (list): All differnt labels
        predictions (list): Predicted label number of according spikes
    Returns:
         [np.array]: Mean distance of spikes from the mean of the cluster
    """

    kl_addition = np.min(data) * -1 + 0.000001

    euclidian_per_cluster = []
    kl_per_cluster = []

    for label in set(labels):
        cluster = []

        for i, spike in enumerate(data):
            if predictions[i] == label:
                cluster.append(spike)
                plt.plot(spike)

        if len(cluster) != 0:
            euclidian_in_cluster = []
            kl_in_cluster = []
            for i1, spike1 in enumerate(cluster):

                spike1 = spike1 + kl_addition
                for i2, spike2 in enumerate(cluster):
                    if i1 != i2:
                        spike2 = spike2 + kl_addition
                        euclidian_in_cluster.append(np.linalg.norm(spike1 - spike2))

                        kl_in_cluster.append(entropy(spike1, spike2))

            euclidian_per_cluster.append(np.mean(euclidian_in_cluster))

            kl_per_cluster.append(np.mean(kl_in_cluster) * 100)

        else:
            euclidian_per_cluster.append(0)

    return euclidian_per_cluster, kl_per_cluster


def plot_cluster(data: np.array, labels: list, predictions: list) -> None:
    """ Plots Cluster

    Parameters:
        data (np.array): Spikes that have been predicted
        labels (list): All differnt labels
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
        plt.title(f"All spikes clustered into {label} (cluster mean in yellow)")
        plt.plot(mean_cluster, color="yellow", linewidth=2)
        plt.ylim(min_in_test_data, max_in_test_data)
        plt.show()

    plt.title(f"All cluster means")
    plt.ylim(min_in_test_data, max_in_test_data)
    for x in all_mean:
        plt.plot(x)
    plt.show()
