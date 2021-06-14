from typing import Union

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import entropy
from sklearn.cluster import KMeans

import autoencoder_functions
from autoencoder import Autoencoder
from configs import training as config


class ClusteringLoss:
    """ Custom PyTorch clustering loss """

    def __init__(self, n_cluster):
        self.n_cluster = n_cluster
        self.kmeans = KMeans(n_clusters=n_cluster)

    def criterion(self, model: Autoencoder, seq_pred: list, seq_true: list) -> float:
        """ Calculates reconstruction and clustering loss

        Parameters:
            model (Autoencoder): Number of samples to propagated through the model at once
            seq_pred (list): The reconstructed time series
            seq_true (list): The original time series
        Returns:
            float: Loss
        """

        # Get reconstruction loss as mean squared error
        reconstruction_loss = nn.MSELoss()(seq_pred, seq_true)

        # Create spare representation and fit k-means on it
        data = np.array(seq_true)
        encode_data = autoencoder_functions.encode_data(model, data, batch_size=len(data))
        self.kmeans.fit(encode_data)

        # Get KL-Divergence
        kl_addition = np.min(data) * -1 + 0.00001
        kl_per_cluster = []

        for label in set(self.kmeans.labels_):
            cluster = []

            for j, spike in enumerate(data):
                if self.kmeans.labels_[j] == label:
                    cluster.append(spike)

            if len(cluster) > 1:
                kl_in_cluster = []
                for i1, spike1 in enumerate(cluster):
                    spike1 = spike1 + kl_addition
                    for i2, spike2 in enumerate(cluster):
                        if i1 != i2:
                            spike2 = spike2 + kl_addition
                            kl_in_cluster.append(entropy(spike1, spike2))
                kl_per_cluster.append(np.mean(kl_in_cluster) * 100)
            else:
                kl_per_cluster.append(0)

        cluster_loss = np.mean(kl_per_cluster) * 50

        # Combine losses
        loss = reconstruction_loss + cluster_loss

        return loss


def train_epoch(model: Autoencoder, optimizer: torch.optim, criterion: Union[nn.MSELoss, ClusteringLoss],
                train_dataset: list, validation_dataset: list, update_percentage: int,
                train_with_clustering: bool = False) -> [float, float]:
    """ Trains Autoencoder on one epoch
    
    Parameters:
        model (Autoencoder): The model to train
        optimizer (optim): The optimizer
        criterion (Union[nn.MSELoss, ClusteringLoss]): Calculate the loss with
        train_dataset (list): Data to train the epoch on
        validation_dataset (list): Data to validate the epoch on
        update_percentage (int): For which value for each multiply the percentage displayed is updated
        train_with_clustering (bool): If the Autoencoder is trained with loss including clustering loss
    Returns:
        [float, float]: Train and validation loss
    """

    # Trains epoch
    model = model.train()
    train_losses = []
    for i, seq_true in enumerate(train_dataset):
        if i % update_percentage == 0:
            print('\r', int(i / len(train_dataset) * 100), '%', sep='', end='')
        optimizer.zero_grad()

        seq_true = seq_true
        seq_pred = model(seq_true)

        if train_with_clustering:
            loss = criterion.criterion(model, seq_pred, seq_true)
        else:
            loss = criterion(seq_pred, seq_true)

        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
    train_loss = np.mean(train_losses) / config.BATCH_SIZE

    print('\r100%', sep='')

    # Validates epoch
    validation_losses = []
    model = model.eval()
    with torch.no_grad():
        for seq_true in validation_dataset:
            seq_true = seq_true
            seq_pred = model(seq_true)

            if train_with_clustering:
                loss = criterion.criterion(model, seq_pred, seq_true)
            else:
                loss = criterion(seq_pred, seq_true)

            validation_losses.append(loss.item())
    validation_loss = np.mean(validation_losses) / config.BATCH_SIZE

    return train_loss, validation_loss


def train_model(model: Autoencoder, train_dataset: list, validation_dataset: list, model_path: str = None,
                train_with_clustering: bool = False, n_cluster: int = None) -> [dict]:
    """ Trains Autoencoder model

    Parameters:
        model: Model to train
        train_dataset: Data to train on
        validation_dataset: Data to validate on
        model_path: Path to save model to
        train_with_clustering: If the loss includes clustering loss
        n_cluster: Number of Cluster
    Returns:
        dict: Trainings history
    """

    # Init variables
    history = dict(train=[], val=[])
    best_loss = 10000.0
    update_percentage = max(1, int(len(train_dataset) / 100))
    best_epoch = 0

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)

    # Use different criterion to get loss value depending it should include the clustering loss
    if not train_with_clustering:
        criterion = nn.MSELoss()
    else:
        criterion = ClusteringLoss(n_cluster)

    # Train fore each epoch
    for epoch in range(1, config.EPOCHS + 1):
        print()
        print(f'Epoch {epoch}')

        train_loss, validation_loss = train_epoch(model=model, optimizer=optimizer,
                                                  criterion=criterion,
                                                  train_dataset=train_dataset,
                                                  validation_dataset=validation_dataset,
                                                  update_percentage=update_percentage,
                                                  train_with_clustering=train_with_clustering)

        # Checks if validation did improve enough to be new best
        if validation_loss < best_loss * (1 - config.EARLY_STOPPING_MIN_IMPROVEMENT):
            best_loss = validation_loss
            best_epoch = epoch

            if model_path is not None:
                torch.save(model, f'{model_path}/model.pth')
                print(f'Saved model to "{model_path}".')

        print(f'train loss {train_loss}, validation loss {validation_loss}')

        history['train'].append(train_loss)
        history['val'].append(validation_loss)

        # Checks if Early Stopping should be performed
        if config.EARLY_STOPPING is not None and epoch - best_epoch == config.EARLY_STOPPING:
            print('Early stopping.')
            break

    return history
