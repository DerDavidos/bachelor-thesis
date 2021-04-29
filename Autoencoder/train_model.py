from autoencoder import Autoencoder, Encoder, Decoder
import autoencoder
import numpy as np
import torch
from sklearn.model_selection import train_test_split
import sys
import pickle
import test_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    # file = 'data/SimulationSpikes.npy'
    file = 'data/GeneratorSpikes.npy'

    with open(file, 'rb') as f:
        aligned_spikes = np.load(f)

    aligned_spikes = aligned_spikes[:int(len(aligned_spikes) * 1)]

    print(f"Data size: {len(aligned_spikes)}, Sequence length: {len(aligned_spikes[0])}")

    train_data, val_data = train_test_split(
        aligned_spikes,
        test_size=0.15,
        # random_state=RANDOM_SEED
    )

    val_data, test_data = train_test_split(
        val_data,
        test_size=0.33,
        # random_state=RANDOM_SEED
    )

    with open("data/test_data", 'wb') as dat:
        pickle.dump(test_data, dat, pickle.HIGHEST_PROTOCOL)

    train_dataset, seq_len, n_features = autoencoder.create_dataset(train_data)
    val_dataset, _, _ = autoencoder.create_dataset(val_data)
    test_dataset, _, _ = autoencoder.create_dataset(test_data)

    # print("nf", n_features)
    model = autoencoder.Autoencoder(seq_len=seq_len, n_features=n_features, embedding_dim=16)
    model = model.to(DEVICE)

    print()
    print("Model architecture")
    print(model)

    model, history = autoencoder.train_model(
        model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        n_epochs=3
    )

    model_path = 'models/model.pth'
    torch.save(model, model_path)
    print(f"Saved model to {model_path}.")

    with open("data/history", 'wb') as his:
        pickle.dump(history, his, pickle.HIGHEST_PROTOCOL)

    print()
    print("Example encoded data")
    print(model.encoder(train_dataset[0]))
    print()

    test_model.plot_training(model, history, test_dataset)


if __name__ == '__main__':
    main()
