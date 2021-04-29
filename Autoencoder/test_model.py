from autoencoder import Autoencoder, Encoder, Decoder
import autoencoder
import numpy as np
import torch
import pickle
import autoencoder

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def plot_training(model=None, history=None, test_data=None, test_data_file="data/test_data"):
    if model is None:
        model = torch.load('models/model.pth', map_location=torch.device(DEVICE))
        model = model.to(DEVICE)

    if history is None:
        with open("data/history", 'rb') as his:
            history = pickle.load(his)

    if test_data is None:
        with open(test_data_file, 'rb') as dat:
            if ".npy" in test_data_file:
                test_data = np.load(dat)
                test_data = test_data[:max(len(test_data), 1000)]
            else:
                test_data = pickle.load(dat)
        test_data, _, _ = autoencoder.create_dataset(test_data)

    print()
    print("Model architecture")
    print(model)

    print()
    print("Example encoded data")
    print(model.encoder(test_data[0]))
    print()

    autoencoder.plot_histroy(history)
    autoencoder.test_reconstructions(model, test_data)


if __name__ == '__main__':
    # Can use 'Messungen' for testing when available
    # plot_training(test_data_file="data/Messungen.npy")
    plot_training(test_data_file="data/GeneratorSpikes.npy")
