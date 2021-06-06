import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import evaluate_functions
from configs import evaluate as config


def determinate_number_of_cluster(data) -> int:
    """ Predictes number of different spikes (cluster) in spike data

    Returns:
        int: Number of predicted cluster
    """

    wcss = []
    max_cluster = 6
    for i in range(2, max_cluster + 1):
        pca = PCA(n_components=8)
        kmeans = KMeans(n_clusters=i)

        pca.fit(data)
        transformed_train_data = pca.transform(data)
        kmeans.fit(transformed_train_data)
        predictions = kmeans.predict(transformed_train_data)

        euclidian_per_cluster, kl_per_cluster = \
            evaluate_functions.evaluate_clustering(data=data, labels=list(set(kmeans.labels_)), predictions=predictions)

        wcss.append(np.mean(kl_per_cluster))
    predicted_number_of_cluster = 2
    print(wcss)
    while predicted_number_of_cluster + 2 < max_cluster \
            and wcss[predicted_number_of_cluster] < wcss[predicted_number_of_cluster + 1] * 1.25:
        predicted_number_of_cluster += 1

    return predicted_number_of_cluster


def plot_predicted_and_real_number_of_cluster() -> None:
    """ For each number of cluster the number of cluster is predicted and plot together with the real number """

    for cluster in config.CLUSTER:
        with open(f'../data/{config.SIMULATION_TYPE}/n_cluster_{cluster}/simulation_{config.SIMULATION_NUMBER}/'
                  f'test_data.npy', 'rb') as file:
            data = np.load(file, allow_pickle=True)
        print(cluster, determinate_number_of_cluster(data=data))


if __name__ == '__main__':
    plot_predicted_and_real_number_of_cluster()
