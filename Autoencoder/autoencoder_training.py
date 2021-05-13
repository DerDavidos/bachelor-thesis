from autoencoder import Autoencoder
import torch
import torch.nn as nn
import numpy as np
import copy
from sklearn.cluster import KMeans
import math
import autoencoder_functions

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def __train_model(model: Autoencoder, optimizer: torch.optim, criterion: nn, train_dataset: list,
                  validation_dataset: list, update_percent: int, batch_size: int = 1,
                  train_with_clustering: bool = False, ):
    model = model.train()
    train_losses = []
    for i, seq_true in enumerate(train_dataset):
        if i % update_percent == 0:
            print("\r", int(i / len(train_dataset) * 100), "%", sep="", end="")
        optimizer.zero_grad()

        seq_true = seq_true.to(DEVICE)
        seq_pred = model(seq_true)

        if train_with_clustering:
            loss = criterion.criterion(model, seq_pred, seq_true)
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
            seq_true = seq_true.to(DEVICE)
            seq_pred = model(seq_true)

            if train_with_clustering:
                loss = criterion.criterion(model, seq_pred, seq_true)
            else:
                loss = criterion(seq_pred, seq_true)

            validation_losses.append(loss.item())

    train_loss = np.mean(train_losses) / batch_size
    validation_loss = np.mean(validation_losses) / batch_size

    return train_loss, validation_loss


class __ClusteringLoss:

    def __init__(self, n_cluster):
        self.kmeans = KMeans(
            # init="random",
            n_clusters=n_cluster,
        )

    def criterion(self, model, seq_pred, seq_true):
        # Get reconstruction loss as mean squared error
        reconstruction_loss = nn.MSELoss().to(DEVICE)(seq_pred, seq_true)

        # Create spare representation and fit k-means on it
        encode_seq_pred = autoencoder_functions.encode_data(model, seq_pred,
                                                            batch_size=len(seq_pred))

        self.kmeans.fit(encode_seq_pred)

        cluster_center_decoded = autoencoder_functions.decode_data(
            model,
            self.kmeans.cluster_centers_,
            batch_size=len(
                self.kmeans.cluster_centers_)
        )

        cluster_loss = []
        for label in set(self.kmeans.labels_):
            cluster = []
            for i, spike in enumerate(seq_true):
                if self.kmeans.labels_[i] == label:
                    cluster.append(
                        math.sqrt(np.mean(
                            abs(cluster_center_decoded[label] - np.array(spike).reshape(-1)))))
            if len(cluster):
                cluster_loss.append(np.mean(cluster))
        cluster_loss = np.mean(cluster_loss)
        loss = reconstruction_loss + (cluster_loss * 10)

        return loss


def train_model(model: Autoencoder, train_dataset: list, validation_dataset: list, n_epochs: int,
                model_path: str = None, batch_size: int = 1, train_with_clustering: bool = False,
                n_cluster: int = None, early_stopping: int = None):
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)

    if not train_with_clustering:
        criterion = nn.MSELoss().to(DEVICE)
    else:
        criterion = __ClusteringLoss(n_cluster)

    history = dict(train=[], val=[])
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 10000.0
    update_percent = max(1, int(len(train_dataset) / 100))

    best_epoch = 0

    for epoch in range(1, n_epochs + 1):
        print()
        print("Epoch", epoch)

        train_loss, validation_loss = __train_model(model=model, optimizer=optimizer,
                                                    criterion=criterion,
                                                    train_dataset=train_dataset,
                                                    validation_dataset=validation_dataset,
                                                    update_percent=update_percent,
                                                    batch_size=batch_size,
                                                    train_with_clustering=train_with_clustering)

        if validation_loss < best_loss:
            best_loss = validation_loss
            best_model_wts = copy.deepcopy(model.state_dict())

            best_epoch = epoch

            if model_path is not None:
                torch.save(model, model_path)
                print(f"Saved model to '{model_path}'.")

        print(f'train loss {train_loss}, validation loss {validation_loss}')

        history["train"].append(train_loss)
        history["val"].append(validation_loss)

        if early_stopping is not None and epoch - best_epoch == early_stopping:
            print("Early stopping.")
            break

    model.load_state_dict(best_model_wts)
    return model.eval(), history
