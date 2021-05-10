import numpy as np
import torch
import pickle
import autoencoder

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def plot_training(simulation_number: int):
    dictionary = f"models/simulation_{simulation_number}/"

    model = torch.load(f"{dictionary}model.pth", map_location=torch.device(DEVICE))
    model = model.to(DEVICE)

    with open(f"{dictionary}history", 'rb') as his:
        history = pickle.load(his)

    with open(f"{dictionary}test_data", 'rb') as dat:
        test_data = pickle.load(dat)
    test_data, _, _ = autoencoder.create_dataset(test_data)

    print()
    print("Model architecture")
    print(model)

    print()
    print("Example encoded spikes")
    print(autoencoder.encode_data(model=model, data=test_data[0]))
    print()

    autoencoder.plot_history(history=history)
    autoencoder.test_reconstructions(model=model, test_dataset=test_data)


if __name__ == '__main__':
    plot_training(simulation_number=0)
