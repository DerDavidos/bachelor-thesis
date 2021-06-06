import torch
import torch.nn as nn


class Encoder(nn.Module):
    """ Encoder component of Autoencoder """

    def __init__(self, input_dim, embedded_dim):
        super(Encoder, self).__init__()

        hidden_dim = int(input_dim / 2)

        self.convolution_1d_1 = nn.Conv1d(1, 5, kernel_size=9, padding=4, padding_mode="replicate")
        self.leaky_re_lu_1 = nn.LeakyReLU()

        self.convolution_1d_2 = nn.Conv1d(5, 5, kernel_size=5, padding=2, padding_mode="replicate")
        self.leaky_re_lu_2 = nn.LeakyReLU()

        self.max_pooling = nn.MaxPool1d(2)

        # First LSTM
        self.lstm_1 = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=int(hidden_dim / 2),
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        # Second LSTM
        self.lstm_2 = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=int(embedded_dim / 2),
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, x: torch.Tensor, batch_size: int):
        """ Propages the data in Tensor x through the Encoder

        Parameters:
            x (torch.Tensor): Data
            batch_size (int): Size of the data batch
        Returns:
            torch.Tensor: Data after all Autoencoder layers
        """

        x = x.reshape((batch_size, 1, -1))

        # Convolution + ReLu
        x = self.convolution_1d_1(x)
        x = self.leaky_re_lu_1(x)

        x = self.convolution_1d_2(x)
        x = self.leaky_re_lu_2(x)

        # Max Pooling
        x = self.max_pooling(x)

        # Fist LSTM
        _, (x, _) = self.lstm_1(x)
        x = x.swapaxes(0, 1)
        x = x.reshape((batch_size, 1, -1))

        # Second LSTM
        _, (x, _) = self.lstm_2(x)
        x = x.swapaxes(0, 1)
        x = x.reshape(batch_size, -1, 1)

        return x


class Decoder(nn.Module):
    """ Decoder component of Autoencoder """

    def __init__(self, output_dim):
        super(Decoder, self).__init__()

        self.up_sample = nn.Upsample(output_dim)

        self.transpose_convolution_1d_2 = nn.ConvTranspose1d(1, 8, kernel_size=8, padding=3)

        self.transpose_convolution_1d_1 = nn.ConvTranspose1d(8, 5, kernel_size=16, padding=8)

        self.transpose_convolution_1d_resize = nn.ConvTranspose1d(5, 1, kernel_size=1)

    def forward(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
        """ Propages the data in Tensor x through the Decoder

        Parameters:
            x (torch.Tensor): Data
            batch_size (int): Size of the data batch
        Returns:
            torch.Tensor: Data after all Decoder layers
        """

        x = x.reshape(batch_size, 1, -1)

        # Up sampling
        x = self.up_sample(x)

        # De-Convolution\
        x = self.transpose_convolution_1d_2(x)

        x = self.transpose_convolution_1d_1(x)

        x = self.transpose_convolution_1d_resize(x)
        x = x.reshape((batch_size, -1, 1))

        return x


class Autoencoder(nn.Module):
    """ Pytorch Autoencoder model """

    def __init__(self, input_dim, embedded_dim):
        super(Autoencoder, self).__init__()

        self.embedded_dim = embedded_dim

        self.__encoder = Encoder(input_dim=input_dim, embedded_dim=self.embedded_dim)
        self.__decoder = Decoder(output_dim=input_dim)

    def forward(self, x: torch.Tensor):
        """ Propages the data in Tensor x through the Autoencoder

        Parameters:
            x (torch.Tensor): Data
        Returns:
            torch.Tensor: Data after all Autoencoder layers
        """

        if len(x.shape) != 3:
            raise SyntaxError('Wrong input dimension')

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
