import sys

sys.path.append('../../../Autoencoder')
from autoencoder import Autoencoder, Encoder, Decoder, create_dataset
# import autoencoder
import numpy as np
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def perform_feature_extraction(spikes):
    spikes = np.array(spikes)
    spikes = spikes.reshape(-1, 64)

    dataset, seq_len, n_features = create_dataset(spikes)

    # model = torch.load('colab_model.pth', map_location=torch.device(DEVICE))
    # model = model.to(DEVICE)

    # dataset = model(dataset)

    return spikes
