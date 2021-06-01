import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans

import autoencoder_functions
import config
from autoencoder import Autoencoder


class __ClusteringLoss:
    """ Custom PyTorch clustering loss """

    def __init__(self, n_cluster):
        self.n_cluster = n_cluster
        self.kmeans = KMeans(
            n_clusters=n_cluster,
        )

    def criterion(self, model: Autoencoder, seq_pred: list, seq_true: list, train_dataset: list):
        """ Calculates reconstruction and clustering loss

        Parameters:
            model (Autoencoder): Number of samples to propagated through the model at once
            seq_pred (list): The reconstructed time series
            seq_true (list): The original time series
            train_dataset (list):
        """

        # Get reconstruction loss as mean squared error
        reconstruction_loss = nn.MSELoss()(seq_pred, seq_true)

        # Create spare representation and fit k-means on it
        cluster_loss = []
        for i, data in enumerate(train_dataset):
            encode_data = autoencoder_functions.encode_data(model, np.array(data),
                                                            batch_size=len(data))
            self.kmeans.fit(encode_data)

            cluster_loss.append(self.kmeans.inertia_)
        cluster_loss = np.mean(cluster_loss)

        loss = reconstruction_loss + cluster_loss

        # print(float(reconstruction_loss), cluster_loss, float(loss))

        return loss


def __train_model(model: Autoencoder, optimizer: torch.optim, criterion: nn, train_dataset: list,
                  validation_dataset: list, update_percent: int,
                  train_with_clustering: bool = False, ):
    model = model.train()
    train_losses = []
    for i, seq_true in enumerate(train_dataset):
        if i % update_percent == 0:
            print("\r", int(i / len(train_dataset) * 100), "%", sep="", end="")
        optimizer.zero_grad()

        seq_true = seq_true
        seq_pred = model(seq_true)

        if train_with_clustering:
            model.eval()
            loss = criterion.criterion(model, seq_pred, seq_true, train_dataset)
            model.train()
        else:
            loss = criterion(seq_pred, seq_true)

        loss.backward()

        optimizer.step()

        train_losses.append(loss.item())

    print("\r100%", sep="")

    validation_losses = []
    model = model.eval()
    with torch.no_grad():
        for seq_true in validation_dataset:
            seq_true = seq_true
            seq_pred = model(seq_true)

            if train_with_clustering:
                loss = criterion.criterion(model, seq_pred, seq_true, validation_dataset)
            else:
                loss = criterion(seq_pred, seq_true)

            validation_losses.append(loss.item())

    train_loss = np.mean(train_losses) / config.BATCH_SIZE
    validation_loss = np.mean(validation_losses) / config.BATCH_SIZE

    return train_loss, validation_loss


def train_model(model: Autoencoder, train_dataset: list, validation_dataset: list,
                model_path: str = None, train_with_clustering: bool = False,
                n_cluster: int = None):
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)

    if not train_with_clustering:
        criterion = nn.MSELoss()
    else:
        criterion = __ClusteringLoss(n_cluster)

    history = dict(train=[], val=[])
    best_loss = 10000.0
    update_percent = max(1, int(len(train_dataset) / 100))

    best_epoch = 0

    for epoch in range(1, config.EPOCHS + 1):
        print()
        print("Epoch", epoch)

        train_loss, validation_loss = __train_model(model=model, optimizer=optimizer,
                                                    criterion=criterion,
                                                    train_dataset=train_dataset,
                                                    validation_dataset=validation_dataset,
                                                    update_percent=update_percent,
                                                    train_with_clustering=train_with_clustering)

        if validation_loss < best_loss * (1 - config.EARLY_STOPPING_MIN_IMPROVEMENT):
            best_loss = validation_loss

            best_epoch = epoch

            if model_path is not None:
                torch.save(model, f"{model_path}/model.pth")
                print(f"Saved model to '{model_path}'.")

        print(f'train loss {train_loss}, validation loss {validation_loss}')

        history["train"].append(train_loss)
        history["val"].append(validation_loss)

        if config.EARLY_STOPPING is not None and epoch - best_epoch == config.EARLY_STOPPING:
            print("Early stopping.")
            break

    return model.eval(), history
