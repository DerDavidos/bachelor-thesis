import numpy as np
import torch
from matplotlib import pyplot as plt

from autoencoder import Autoencoder


def predict(model: Autoencoder, dataset: np.array) -> np.array:
    predictions = []
    with torch.no_grad():
        model = model.eval()
        for seq_true in dataset:
            if len(seq_true.shape) == 2:
                seq_true = seq_true.reshape(1, -1, 1)
            seq_true = torch.tensor(seq_true).float()
            seq_pred = model(seq_true)

            predictions.append(seq_pred.cpu().numpy().flatten())

    return predictions


def test_reconstructions(model: Autoencoder, test_dataset: np.array, max_graphs: int = 5):
    predictions = predict(model, test_dataset)

    print(f'Number of test spikes: {len(test_dataset)}')

    for i in range(min(len(test_dataset), max_graphs)):
        plt.plot(test_dataset[i])
        plt.plot(predictions[i])
        plt.title('Test spike ' + str(i))
        plt.show()


def encode_data(model: Autoencoder, data: np.array, batch_size: int) -> np.array:
    """ Uses the Encoder component of the Autoencoder to encode data

    Parameters:
        model: The Autoencoder model to encode the data with
        data: The data to encode
        batch_size: Batch size of the given data
    Returns:
         np.array: The encoded data
    """

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


def decode_data(model: Autoencoder, data: np.array, batch_size: int) -> np.array:
    """ Uses the Decoder component of the Autoencoder to decode data

    Parameters:
        model: The Autoencoder model to decode the data with
        data: The data to decode
        batch_size: Batch size of the given data
    Returns:
         np.array: The decoded data
    """

    data = torch.tensor(data).float()
    if len(data.shape) == 1:
        data = data.reshape(-1, 1)
    if len(data.shape) == 2:
        data = data.reshape(batch_size, -1, 1)

    data = model.decode_data(data)
    data = data.reshape(batch_size, -1).detach().numpy()

    return data


def plot_history(history: dict) -> None:
    """ Plots the train and validation accuracy over the training epochs of a model

    Parameters:
        history (dict): The trainings history
    """

    ax = plt.figure().gca()
    ax.plot(history['train'])
    ax.plot(history['val'])
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'])
    plt.title('Loss over training epochs')
    plt.show()
