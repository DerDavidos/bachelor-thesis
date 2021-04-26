import sys

sys.path.append('../../../Autoencoder')
from autoencoder import Autoencoder, Encoder, Decoder
import autoencoder
import numpy as np
import torch

import sys

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def perform_feature_extraction(spikes):
    # print(spikes)
    spikes = np.array(spikes)
    # print(spikes.shape)
    spikes = spikes.reshape(-1, 64)
    # print(spikes.shape)
    dataset, seq_len, n_features = autoencoder.create_dataset(spikes)

    model = torch.load('../../../Autoencoder/models/colab_model.pth', map_location=torch.device(DEVICE))
    model = model.to(DEVICE)
    dataset = model(dataset)
    # print(dataset)
    return dataset

# print(perform_feature_extraction([[1] * 64]))
