import torch
import torch.nn as nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Encoder(nn.Module):

    def __init__(self, input_dim, embedded_dim):
        super(Encoder, self).__init__()

        hidden_dim = int(input_dim / 2)

        self.convolution_1d = nn.Conv1d(1, 5, kernel_size=11, padding=5, padding_mode="replicate")
        self.leaky_re_lu = nn.LeakyReLU()
        self.max_pooling = nn.MaxPool1d(2)

        # First LSTM
        self.lstm_1 = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            # bidirectional=True,
        )

        # Second LSTM
        self.lstm_2 = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=embedded_dim,
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

        x = x.reshape(batch_size, -1, 1)
        return x


class Decoder(nn.Module):

    def __init__(self, output_dim):
        super(Decoder, self).__init__()

        self.up_sample = nn.Upsample(output_dim)

        self.transpose_convolution_1d = nn.ConvTranspose1d(1, 5, kernel_size=11, padding=5)
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

    def __init__(self, input_dim, embedded_dim):
        super(Autoencoder, self).__init__()

        self.embedded_dim = embedded_dim

        self.__encoder = Encoder(input_dim=input_dim, embedded_dim=self.embedded_dim).to(DEVICE)
        self.__decoder = Decoder(output_dim=input_dim).to(DEVICE)

    def forward(self, x):
        if len(x.shape) != 3:
            raise SyntaxError("Wrong input dimension")

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

    def decode_data(self, x):
        if len(x.shape) == 2:
            x = x.reshape(1, -1, 1)
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1, 1)
        x = self.__decoder(x, batch_size)

        return x
