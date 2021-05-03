import torch
import torch.nn as nn
import numpy as np
import copy
from matplotlib import pyplot as plt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()

        self.convolution_1d = nn.Conv1d(1, 5, kernel_size=5, padding=2, padding_mode="circular")
        self.leaky_re_lu = nn.LeakyReLU()
        self.max_pooling = nn.MaxPool1d(2)

        # First LSTM
        self.lstm_1 = nn.LSTM(
            input_size=32,
            hidden_size=32,
            num_layers=1,
            batch_first=True,
            # bidirectional=True,
        )

        # Second LSTM
        self.lstm_2 = nn.LSTM(
            input_size=32,
            hidden_size=16,
            num_layers=1,
            batch_first=True,
            # bidirectional=True,
        )

    def forward(self, x, batch_size):
        x = x.reshape((batch_size, 1, -1))

        # Convolution + ReLu
        x = self.convolution_1d(x)
        x = self.leaky_re_lu(x)

        # Max Pooling
        x = self.max_pooling(x)

        # Fist LSTM
        _, (x, _) = self.lstm_1(x)

        x = x.reshape((batch_size, 1, -1))

        # Second LSTM
        _, (x, _) = self.lstm_2(x)
        x = x.reshape(batch_size, 16, 1)

        return x


class Decoder(nn.Module):

    def __init__(self):
        super(Decoder, self).__init__()

        self.up_sample = nn.Upsample(64)

        self.transpose_convolution_1d = nn.ConvTranspose1d(1, 5, kernel_size=5, padding=2)
        self.transpose_convolution_1d_2 = nn.ConvTranspose1d(5, 1, kernel_size=1)

    def forward(self, x, batch_size):
        x = x.reshape(batch_size, 1, -1)

        # Up sampling
        x = self.up_sample(x)

        # De-Convolution
        x = self.transpose_convolution_1d(x)
        x = self.transpose_convolution_1d_2(x)
        x = x.reshape((batch_size, -1, 1))

        return x


class Autoencoder(nn.Module):

    def __init__(self):
        super(Autoencoder, self).__init__()

        # seq_len = seq_len * 2
        # n_features = n_features * 2

        self.__encoder = Encoder().to(DEVICE)
        self.__decoder = Decoder().to(DEVICE)

    def forward(self, x):

        if len(x.shape) == 2:
            x = x.reshape(1, -1, 1)
        batch_size = x.shape[0]
        x = self.__encoder(x, batch_size)
        x = self.__decoder(x, batch_size)

        return x

    def encode_data(self, x):
        if len(x.shape) == 2:
            x = x.reshape(1, -1, 1)
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1, 1)
        x = self.__encoder(x, batch_size)

        return x


def __train_model(model: Autoencoder, optimizer: torch.optim, criterion: nn, train_dataset: list,
                  validation_dataset: list, update_percent: int, batch_size: int = 1):
    model = model.train()
    train_losses = []
    for i, seq_true in enumerate(train_dataset):
        if i % update_percent == 0:
            print("\r", int(i / len(train_dataset) * 100), "%", sep="", end="")
        optimizer.zero_grad()

        seq_true = seq_true.to(DEVICE)
        seq_pred = model(seq_true)

        loss = criterion(seq_pred, seq_true)

        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

    print("\r100%", sep="")

    validation_losses = []
    model = model.eval()
    with torch.no_grad():
        for seq_true in validation_dataset:
            seq_true = seq_true.to(DEVICE)
            seq_pred = model(seq_true)

            loss = criterion(seq_pred, seq_true)
            validation_losses.append(loss.item())

    train_loss = np.mean(train_losses) / batch_size
    validation_loss = np.mean(validation_losses) / batch_size

    return train_loss, validation_loss


def train_model(model: Autoencoder, train_dataset: list, validation_dataset: list, n_epochs: int,
                model_path: str = None, batch_size: int = 1):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.L1Loss(reduction='sum').to(DEVICE)
    history = dict(train=[], val=[])

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 10000.0
    update_percent = max(1, int(len(train_dataset) / 100))

    for epoch in range(1, n_epochs + 1):
        print()
        print("Epoch", epoch)

        train_loss, validation_loss = __train_model(model=model, optimizer=optimizer,
                                                    criterion=criterion,
                                                    train_dataset=train_dataset,
                                                    validation_dataset=validation_dataset,
                                                    update_percent=update_percent,
                                                    batch_size=batch_size)

        if validation_loss < best_loss:
            best_loss = validation_loss
            best_model_wts = copy.deepcopy(model.state_dict())

            if model_path is not None:
                torch.save(model, model_path)
                print(f"Saved model to '{model_path}'.")

        print(f'Epoch {epoch}: train loss {train_loss} validation loss {validation_loss}')

    model.load_state_dict(best_model_wts)
    return model.eval(), history


def create_dataset(sequences: list):
    # sequences = df.astype(np.float32).to_numpy().tolist()
    dataset = [torch.tensor(s).unsqueeze(1).float() for s in sequences]
    _, seq_len, n_features = torch.stack(dataset).shape
    return dataset, seq_len, n_features


def predict(model: Autoencoder, dataset: list):
    predictions, losses = [], []
    criterion = nn.L1Loss(reduction='sum').to(DEVICE)
    with torch.no_grad():
        model = model.eval()
        for seq_true in dataset:
            if len(seq_true.shape) == 2:
                seq_true = seq_true.reshape(1, -1, 1)
            seq_true = seq_true.to(DEVICE)
            seq_pred = model(seq_true)

            loss = criterion(seq_pred, seq_true)

            predictions.append(seq_pred.cpu().numpy().flatten())
            losses.append(loss.item())
    return predictions, losses


def test_reconstructions(model: Autoencoder, test_dataset: list, max_graphs: int = 15):
    predictions, pred_losses = predict(model, test_dataset)

    print("Number of test spikes:", len(test_dataset))
    print("Average prediction loss:", sum(pred_losses) / len(pred_losses))

    for i in range(min(len(test_dataset), max_graphs)):
        plt.plot(test_dataset[i])
        plt.plot(predictions[i])
        plt.title("Test spike " + str(i))
        plt.show()


def encode_data(model: Autoencoder, data: torch.Tensor):
    data = model.encode_data(data)
    data = data.reshape(-1).detach().numpy()

    return data


def plot_history(history: dict):
    ax = plt.figure().gca()
    ax.plot(history['train'])
    ax.plot(history['val'])
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'])
    plt.title('Loss over training epochs')
    plt.show()
