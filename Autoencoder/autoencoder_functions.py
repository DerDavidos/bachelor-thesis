from autoencoder import Autoencoder
import torch
import torch.nn as nn
import numpy as np
import copy
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

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
        encode_seq_pred = encode_data(model, seq_pred, batch_size=len(seq_pred))
        self.kmeans.fit(encode_seq_pred)

        # TODO: Get clustering loss
        raise NotImplementedError
        # loss = torch.mean(torch.tensor(self.kmeans.inertia_) + mse)

        return reconstruction_loss


def train_model(model: Autoencoder, train_dataset: list, validation_dataset: list, n_epochs: int,
                model_path: str = None, batch_size: int = 1, train_with_clustering: bool = False,
                n_cluster: int = None):
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)

    if not train_with_clustering:
        criterion = nn.MSELoss().to(DEVICE)
    else:
        criterion = __ClusteringLoss(n_cluster)

    history = dict(train=[], val=[])
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 10000.0
    update_percent = max(1, int(len(train_dataset) / 100))

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

            if model_path is not None:
                torch.save(model, model_path)
                print(f"Saved model to '{model_path}'.")

        print(f'Epoch {epoch}: train loss {train_loss} validation loss {validation_loss}')

        history["train"].append(train_loss)
        history["val"].append(validation_loss)

    model.load_state_dict(best_model_wts)
    return model.eval(), history


def create_dataset(sequences: list):
    dataset = [torch.tensor(s).unsqueeze(1).float() for s in sequences]
    _, seq_len, n_features = torch.stack(dataset).shape
    return dataset, seq_len, n_features


def predict(model: Autoencoder, dataset: list):
    predictions, losses = [], []
    criterion = nn.L1Loss(reduction='sum').to(DEVICE)
    with torch.no_grad():
        model = model.eval()
        for seq_true in dataset:
            if len(seq_true.shape) == 2:
                seq_true = seq_true.reshape(1, -1, 1)
            seq_true = seq_true.to(DEVICE)
            seq_pred = model(seq_true)

            loss = criterion(seq_pred, seq_true)

            predictions.append(seq_pred.cpu().numpy().flatten())
            losses.append(loss.item())
    return predictions, losses


def test_reconstructions(model: Autoencoder, test_dataset: list, max_graphs: int = 15):
    predictions, pred_losses = predict(model, test_dataset)

    print("Number of test spikes:", len(test_dataset))
    print("Average prediction loss:", sum(pred_losses) / len(pred_losses))

    for i in range(min(len(test_dataset), max_graphs)):
        plt.plot(test_dataset[i])
        plt.plot(predictions[i])
        plt.title("Test spike " + str(i))
        plt.show()


def encode_data(model: Autoencoder, data: list, batch_size: int = 1):
    if isinstance(data, torch.Tensor):
        data = data.detach().numpy()
    data = np.array(data)
    data = torch.tensor(data).float()
    if len(data.shape) == 1:
        data = data.reshape(-1, 1)
    if len(data.shape) == 2:
        data = data.reshape(batch_size, -1, 1)

    data = model.encode_data(data)
    data = data.reshape(batch_size, -1).detach().numpy()

    return data


def decode_data(model: Autoencoder, data: list, batch_size: int = 1):
    data = np.array(data)
    data = torch.tensor(data).float()
    if len(data.shape) == 1:
        data = data.reshape(-1, 1)
    if len(data.shape) == 2:
        data = data.reshape(batch_size, -1, 1)

    data = model.decode_data(data)
    data = data.reshape(batch_size, -1).detach().numpy()

    return data


def plot_history(history: dict):
    ax = plt.figure().gca()
    ax.plot(history['train'])
    ax.plot(history['val'])
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'])
    plt.title('Loss over training epochs')
    plt.show()
