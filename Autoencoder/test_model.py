import pickle

import torch

import autoencoder_functions
import config
import data_loader


def plot_training() -> None:
    """Plots the training and validation loss over the epochs and some spikes encoded and then
    decoded """

    model = torch.load(f"{config.MODEL_PATH}/model.pth")

    with open(f"{config.MODEL_PATH}/train_history", 'rb') as his:
        history = pickle.load(his)

    _, _, test_data = data_loader.load_train_val_test_data()

    print()
    print("Model architecture")
    print(model)

    print()
    print("Example encoded spikes")
    print(autoencoder_functions.encode_data(model=model, data=test_data[0]))
    print()

    autoencoder_functions.plot_history(history=history)
    autoencoder_functions.test_reconstructions(model=model, test_dataset=test_data)


if __name__ == '__main__':
    plot_training()
