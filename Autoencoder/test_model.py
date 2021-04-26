from autoencoder import Autoencoder, Encoder, Decoder
import autoencoder
import numpy as np
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    file = 'data/SimulationSpikes.npy'
    # file = 'data/GeneratorSpikes.npy'

    with open(file, 'rb') as f:
        aligned_spikes = np.load(f)

    aligned_spikes = aligned_spikes[:min(int(len(aligned_spikes) * 0.1), 50)]

    print(f"Data size: {len(aligned_spikes)}, Sequence length: {len(aligned_spikes[0])}")

    test_dataset, seq_len, n_features = autoencoder.create_dataset(aligned_spikes)

    model = torch.load('models/colab_model.pth', map_location=torch.device(DEVICE))
    model = model.to(DEVICE)

    print()
    print("Model architecture")
    print(model)

    print()
    print("Example encoded data")
    print(model.encoder(test_dataset[0]))
    print()

    autoencoder.test_reconstructions(model, test_dataset, max_graphs=20)


if __name__ == '__main__':
    main()
