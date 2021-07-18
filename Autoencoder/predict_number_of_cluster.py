import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import evaluate_functions
from configs import evaluate as config


def determinate_number_of_cluster(data) -> int:
    """ Predicts number of different spikes (cluster) in spike data

    Returns:
        int: Number of predicted cluster
    """

    wcss = []
    max_cluster = 8
    for i in range(2, max_cluster + 1):
        pca = PCA(n_components=8)
        kmeans = KMeans(n_clusters=i)

        pca.fit(data)
        transformed_train_data = pca.transform(data)
        kmeans.fit(transformed_train_data)
        predictions = kmeans.predict(transformed_train_data)

        _, kl_per_cluster = \
            evaluate_functions.evaluate_clustering(data=data, labels=list(set(kmeans.labels_)), predictions=predictions)

        wcss.append(np.mean(kl_per_cluster))
    predicted_n_cluster = 2
    print(wcss)

    while predicted_n_cluster < max_cluster - 2 and \
            (wcss[predicted_n_cluster]) < wcss[predicted_n_cluster + 1] * 1.5:
        predicted_n_cluster += 1
    print(predicted_n_cluster)
    return predicted_n_cluster


def plot_predicted_and_real_number_of_cluster() -> None:
    """ For each number of cluster the number of cluster is predicted and plot together with the real number """

    plot_cluster = []
    max_data_size_to_check = 500

    for cluster in config.CLUSTER:
        with open(
                f'data/{config.TRAIN_SIMULATION_TYPE}/n_cluster_{cluster}/simulation_{config.TRAIN_SIMULATION_NUMBER}/'
                f'test_data.npy', 'rb') as file:
            data = np.load(file, allow_pickle=True)
            if len(data) > max_data_size_to_check:
                data = data[:max_data_size_to_check]
        plot_cluster.append(determinate_number_of_cluster(data=data))

    # plt.scatter(config.CLUSTER, config.CLUSTER)
    plt.scatter(config.CLUSTER, plot_cluster)
    plt.ylim([1.5, config.CLUSTER[-1] + 0.5])
    plt.xlim([1.5, config.CLUSTER[-1] + 0.5])
    plt.plot([0, 10], [0, 10])
    plt.ylabel('Predicted Number of Cluster')
    plt.xlabel('Real Number of CLuster')
    plt.show()


if __name__ == '__main__':
    plot_predicted_and_real_number_of_cluster()
