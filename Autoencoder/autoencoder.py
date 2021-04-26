import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import copy
from matplotlib import pyplot as plt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Encoder(nn.Module):

    def __init__(self, seq_len, n_features, embedding_dim):
        super(Encoder, self).__init__()

        self.cnn = nn.Conv1d(64, 64, kernel_size=1)
        self.LeakyReLU = nn.LeakyReLU()

        self.pool = nn.MaxPool1d(2)

        # First LSTM
        self.lstm1 = nn.LSTM(
            input_size=1,
            hidden_size=32,
            num_layers=1,
            batch_first=True,
            # bidirectional=True,
        )

        # Second LSTM
        self.lstm2 = nn.LSTM(
            input_size=1,
            hidden_size=16,
            num_layers=1,
            batch_first=True,
            # bidirectional=True,
        )

    def forward(self, x):
        x = x.reshape((1, -1, 1))

        # Convolution + ReLu
        x = self.cnn(x)
        x = x.reshape((1, 64, 1))
        x = self.LeakyReLU(x)

        # Max Pooling
        x = x.reshape((1, 1, 64))
        x = self.pool(x)
        x = x.reshape((1, 32, 1))

        # Fist LSTM
        _, (x, _) = self.lstm1(x)
        x = x.reshape((1, 32, 1))

        # Second LSTM
        _, (x, _) = self.lstm2(x)
        x = x.reshape((1, -1, 1))

        return x


class Decoder(nn.Module):

    def __init__(self, seq_len, input_dim, n_features):
        super(Decoder, self).__init__()

        # First LSTM
        self.lstm1 = nn.LSTM(
            input_size=1,
            hidden_size=32,
            num_layers=1,
            batch_first=True,
            # bidirectional=True,
        )

        # Second LSTM
        self.lstm2 = nn.LSTM(
            input_size=1,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            # bidirectional=True,
        )

        self.de_cnn = nn.ConvTranspose1d(64, 64, 1)
        # self.output_layer = nn.Linear(64, 64)

    def forward(self, x):
        # Fist LSTM
        _, (x, _) = self.lstm1(x)
        x = x.reshape((1, 32, 1))

        # Second LSTM
        _, (x, _) = self.lstm2(x)
        x = x.reshape((1, 64, 1))

        # De-Convolution
        x = self.de_cnn(x)
        x = x.reshape((-1, 1))

        return x


class Autoencoder(nn.Module):

    def __init__(self, seq_len, n_features, embedding_dim):
        super(Autoencoder, self).__init__()

        # seq_len = seq_len * 2
        # n_features = n_features * 2

        self.encoder = Encoder(seq_len=seq_len, n_features=n_features, embedding_dim=embedding_dim).to(DEVICE)
        self.decoder = Decoder(seq_len=seq_len, input_dim=embedding_dim, n_features=n_features).to(DEVICE)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x


def train_model(model, train_dataset, val_dataset, n_epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.L1Loss(reduction='sum').to(DEVICE)
    history = dict(train=[], val=[])

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 10000.0
    len_dataset = len(train_dataset)
    update_percent = max(1, int(len_dataset / 200))

    for epoch in range(1, n_epochs + 1):
        print()
        print("Epoch", epoch)
        model = model.train()
        train_losses = []
        for i, seq_true in enumerate(train_dataset, start=0):
            if i % update_percent == 0:
                print("\r", round(i / len_dataset * 100, 2), "%", sep="", end="")
            optimizer.zero_grad()

            seq_true = seq_true.to(DEVICE)
            seq_pred = model(seq_true)

            loss = criterion(seq_pred, seq_true)

            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        print("\r100%", sep="", end="")

        val_losses = []
        model = model.eval()
        with torch.no_grad():
            for seq_true in val_dataset:
                seq_true = seq_true.to(DEVICE)
                seq_pred = model(seq_true)

                loss = criterion(seq_pred, seq_true)
                val_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)

        history['train'].append(train_loss)
        history['val'].append(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
        print()
        print(f'Epoch {epoch}: train loss {train_loss} val loss {val_loss}')

    model.load_state_dict(best_model_wts)
    return model.eval(), history


def create_dataset(sequences):
    # sequences = df.astype(np.float32).to_numpy().tolist()
    dataset = [torch.tensor(s).unsqueeze(1).float() for s in sequences]
    n_seq, seq_len, n_features = torch.stack(dataset).shape
    return dataset, seq_len, n_features


def predict(model, dataset):
    predictions, losses = [], []
    criterion = nn.L1Loss(reduction='sum').to(DEVICE)
    with torch.no_grad():
        model = model.eval()
        for seq_true in dataset:
            seq_true = seq_true.to(DEVICE)
            seq_pred = model(seq_true)

            loss = criterion(seq_pred, seq_true)

            predictions.append(seq_pred.cpu().numpy().flatten())
            losses.append(loss.item())
    return predictions, losses


def test_reconstructions(model, test_dataset, max_graphs=15):
    predictions, pred_losses = predict(model, test_dataset)

    print("Number of test spikes:", len(test_dataset))
    print("Average prediction loss:", sum(pred_losses) / len(pred_losses))

    for i in range(min(len(test_dataset), max_graphs)):
        plt.plot(test_dataset[i])
        plt.plot(predictions[i])
        plt.title("Test spike " + str(i))
        plt.show()


def encode_data(model, data):
    data = data.shape(1, 64, 1)
    model.encoder(data)
    data = data.shape(64)

    return data
