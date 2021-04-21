import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import copy

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Encoder(nn.Module):

    def __init__(self, seq_len, n_features, embedding_dim):
        super(Encoder, self).__init__()

        self.seq_len, self.n_features = seq_len, n_features
        self.embedding_dim, self.hidden_dim = embedding_dim, 2 * embedding_dim

        # 1D CNN + Leaky ReLU + MaxPool
        self.cnn = nn.Conv1d(self.seq_len, self.seq_len, 1)
        self.LeakyReLU = nn.LeakyReLU()
        self.max_pool = nn.MaxPool1d(1)

        # First LSTM
        self.rnn1 = nn.LSTM(
            input_size=n_features,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True,
            # bidirectional=True,
        )
        self.relu1 = nn.ReLU()
        self.dropout1 = torch.nn.Dropout(0.3)

        # Second LSTM
        self.rnn2 = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=embedding_dim,
            num_layers=1,
            batch_first=True,
            # bidirectional=True,
        )
        self.relu2 = nn.ReLU()
        self.dropout2 = torch.nn.Dropout(0.3)

    def forward(self, x):
        x = x.reshape((1, self.seq_len, self.n_features))

        # 1D CNN + Leaky RELU + MaxPool
        x = self.cnn(x)
        x = self.LeakyReLU(x)
        x = self.max_pool(x)

        # Fist LSTM
        x, (_, _) = self.rnn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        # Second LSTM
        _, (hidden_n, _) = self.rnn2(x)
        hidden_n = self.relu2(hidden_n)
        hidden_n = self.dropout2(hidden_n)

        return hidden_n.reshape((self.n_features, self.embedding_dim))


class Decoder(nn.Module):

    def __init__(self, seq_len, input_dim, n_features):
        super(Decoder, self).__init__()

        self.seq_len, self.input_dim = seq_len, input_dim
        self.hidden_dim, self.n_features = 2 * input_dim, n_features

        # First LSTM
        self.rnn1 = nn.LSTM(
            input_size=input_dim,
            hidden_size=input_dim,
            num_layers=1,
            batch_first=True,
            # bidirectional=True,
        )
        self.relu1 = nn.ReLU()
        self.dropout1 = torch.nn.Dropout(0.3)

        # Second LSTM
        self.rnn2 = nn.LSTM(
            input_size=input_dim,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True,
            # bidirectional=True,
        )
        self.relu2 = nn.ReLU()
        self.dropout2 = torch.nn.Dropout(0.3)

        # Deconvolution
        self.de_cnn = nn.ConvTranspose1d(self.seq_len, self.seq_len, 1)

        self.output_layer = nn.Linear(self.hidden_dim, n_features)

    def forward(self, x):
        x = x.repeat(self.seq_len, self.n_features)
        x = x.reshape((self.n_features, self.seq_len, self.input_dim))

        # First LSTM
        x, (hidden_n, cell_n) = self.rnn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        # Second LSTM
        x, (hidden_n, cell_n) = self.rnn2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        # Deconvolution
        x = self.de_cnn(x)

        x = x.reshape((self.seq_len, self.hidden_dim))
        return self.output_layer(x)


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
