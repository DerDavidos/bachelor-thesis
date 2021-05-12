from autoencoder import Autoencoder
import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
