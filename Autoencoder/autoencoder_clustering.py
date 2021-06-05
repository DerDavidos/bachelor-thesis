import numpy as np
import torch
from sklearn.cluster import KMeans

import config
import data_loader
from autoencoder import autoencoder_functions
from autoencoder.autoencoder import Autoencoder
from evaluate import evaluate_functions


class AutoencoderClusterer:

    def __init__(self, model: Autoencoder, n_cluster: int, train_data: np.array):
        """

        Parameters:
            train_data (np.array): The data to fit the PCA and k-means on
            n_cluster (int): Number of cluster
        """

        self.labels = [x for x in range(n_cluster)]

        self.model = model
        self.kmeans = KMeans(n_clusters=n_cluster)

        encoded_data = autoencoder_functions.encode_data(model=model, data=train_data,
                                                         batch_size=len(train_data))
        self.kmeans.fit(encoded_data)

    def predict(self, data: np.array):
        encoded_data = autoencoder_functions.encode_data(model=self.model, data=data,
                                                         batch_size=len(data))
        predictions = self.kmeans.predict(encoded_data)
        return predictions


def main(evaluate: bool) -> None:
    """ Performs clustering using pca to reduce dimension on the test data set for the simulation
    defined in config.py """

    # Load train and test data
    train_data, _, test_data = data_loader.load_train_val_test_data()

    # Load model
    model = torch.load(f'{config.MODEL_PATH}/model.pth')

    pca_clusterer = AutoencoderClusterer(model=model,
                                         n_cluster=config.N_CLUSTER,
                                         train_data=train_data)

    predictions = pca_clusterer.predict(test_data)

    evaluate_functions.plot_cluster(data=test_data, labels=pca_clusterer.labels,
                                    predictions=predictions)

    if evaluate:
        euclidian_per_cluster, kl_per_cluster = \
            evaluate_functions.evaluate_clustering(data=test_data, labels=pca_clusterer.labels,
                                                   predictions=predictions)

        print("\nAverage Euclidian distance from spikes to other spikes in same cluster")
        for i, x in enumerate(euclidian_per_cluster):
            print(f"{i}: {x}")
        print(f"Average: \033[31m{np.mean(euclidian_per_cluster)}\033[0m")

        print("\nAverage KL_Divergence from spikes to other spikes in same cluster")
        for i, x in enumerate(kl_per_cluster):
            print(f"{i}: {x}")
        print(f"Average: \033[31m{np.mean(kl_per_cluster)}\033[0m")


if __name__ == '__main__':
    main(evaluate=config.EVALUATE)
