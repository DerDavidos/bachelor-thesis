import numpy as np
import torch
import pickle
import autoencoder

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def plot_training(model: autoencoder.Autoencoder = None, history: dict = None, test_data: list = None,
                  test_data_file: str = "data/training/test_data"):
    if model is None:
        model = torch.load('models/model.pth', map_location=torch.device(DEVICE))
        model = model.to(DEVICE)

    if history is None:
        with open("data/training/history", 'rb') as his:
            history = pickle.load(his)

    if test_data is None:
        with open(test_data_file, 'rb') as dat:
            if ".npy" in test_data_file:
                test_data = np.load(dat)
                test_data = test_data[:min(len(test_data), 1000)]
            else:
                test_data = pickle.load(dat)
        test_data, _, _ = autoencoder.create_dataset(test_data)

    print()
    print("Model architecture")
    print(model)

    print()
    print("Example encoded data")
    print(autoencoder.encode_data(model=model, data=test_data[0]))
    print()

    autoencoder.plot_history(history=history)
    autoencoder.test_reconstructions(model=model, test_dataset=test_data)


if __name__ == '__main__':
    # Can use 'Messungen' for testing when available
    # plot_training(test_data_file="data/Messungen.npy")
    plot_training()
