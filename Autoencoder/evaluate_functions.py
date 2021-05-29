import numpy as np
from matplotlib import pyplot as plt


def evaluate_clustering(data: np.ndarray, labels: list, predictions: list, plot: bool = False) -> \
        [np.ndarray]:
    """ Evaluate the clustering

    Parameters:
        data (np.ndarray): Spikes that have been predicted
        labels (list): All differnt labels
        predictions (list): Predicted label number of according spikes
        plot (bool): If to plot cluster
    Returns:
         [np.ndarray]: Mean distance of spikes from the mean of the cluster
    """
    # Plot all spikes put in the same cluster
    min_in_test_data = np.min(data)
    max_in_test_data = np.max(data)

    all_mean = []
    mse_per_cluster = []

    for label in set(labels):
        cluster = []

        for i, spike in enumerate(data):
            if predictions[i] == label:
                cluster.append(spike)
                plt.plot(spike)

        if len(cluster) != 0:
            mean_cluster = np.mean(cluster, axis=0)
            distances_in_cluster = []
            for i, spike in enumerate(cluster):
                distances_in_cluster.append(np.mean(np.sqrt(np.abs(mean_cluster - spike))))
            mse_per_cluster.append(np.mean(distances_in_cluster))

            if plot:
                all_mean.append(mean_cluster)
        else:
            mean_cluster = 0
            mse_per_cluster.append(0)

        if plot:
            plt.title(f"All spikes clustered into {label} (cluster mean in yellow)")
            plt.plot(mean_cluster, color="yellow", linewidth=2)
            plt.ylim(min_in_test_data, max_in_test_data)
            plt.show()

    if plot:
        plt.title(f"All cluster means")
        plt.ylim(min_in_test_data, max_in_test_data)
        for x in all_mean:
            plt.plot(x)
        plt.show()

    return mse_per_cluster
