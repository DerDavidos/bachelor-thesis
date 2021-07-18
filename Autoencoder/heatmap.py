from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt

import autoencoder_clustering
import autoencoder_functions
import data_loader
from configs import simulation as config

BATCH_SIZE = 8

activation = {}


def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()

    return hook


class CombinedLoss:
    """ Custom PyTorch clustering loss """

    def __init__(self):
        self.label_loss = nn.MultiLabelSoftMarginLoss()

    def criterion(self, classification_vector, localization_vector, labels, epoch):
        """ Calculates classification and localization loss combined """

        cls_loss = self.label_loss(classification_vector, labels)
        loc_loss = self.label_loss(localization_vector, labels)

        a = 0.1
        if epoch > 60:
            a = 0.9

        loss = (1 - a) * cls_loss + a * loc_loss

        return loss


class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.softmax = nn.Softmax(dim=1)

        # Shared layers
        self.up_sample = nn.Upsample(48)
        self.transpose_convolution_1d_2 = nn.ConvTranspose1d(1, 8, kernel_size=5, padding=2)
        self.transpose_convolution_1d_1 = nn.ConvTranspose1d(8, 4, kernel_size=9, padding=4)
        self.transpose_convolution_1d_resize = nn.ConvTranspose1d(4, 1, kernel_size=1)

        # Classification layers
        self.lin = nn.Linear(48, config.N_CLUSTER)

        # Localization layers
        self.maps = nn.Conv1d(1, config.N_CLUSTER, kernel_size=1)
        self.avgPool = nn.AvgPool1d(48)
        self.relu = nn.ReLU()

    def forward(self, x, batch_size):
        x = x.reshape((batch_size, 1, -1))

        # Shared layers
        x = self.up_sample(x)
        x = self.transpose_convolution_1d_2(x)
        x = self.transpose_convolution_1d_1(x)
        x = self.transpose_convolution_1d_resize(x)
        x = x.reshape(batch_size, -1)

        # Classification layers
        cls = self.lin(x)
        cls = self.softmax(cls)

        # Localization layers
        x = x.reshape(batch_size, 1, -1)
        loc = self.maps(x)
        loc = self.relu(loc)
        loc = self.avgPool(loc)
        loc = self.softmax(loc)
        loc = loc.reshape(batch_size, -1)

        return cls, loc


def main():
    model = torch.load(f'models/{config.SIMULATION_TYPE}/n_cluster_{config.N_CLUSTER}/'
                       f'simulation_{config.SIMULATION_NUMBER}_cluster_trained/'
                       f'sparse_{config.EMBEDDED_DIMENSION}/model.pth')

    net = Net()
    # MultiLabelMarginLoss
    combined_loss = CombinedLoss()

    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    data_path = f'data/{config.SIMULATION_TYPE}/n_cluster_{config.N_CLUSTER}/simulation_{config.SIMULATION_NUMBER}'
    train_data, _, test_data = data_loader.load_train_val_test_data(data_path)

    test_data_encoded = autoencoder_functions.encode_data(model=model, data=test_data, batch_size=len(test_data))

    train_data = train_data[:(len(train_data) - (len(train_data) % BATCH_SIZE))]
    if not train_data.any():
        raise ValueError('Batch size to big for train split.')

    plot_data = test_data_encoded

    clusterer = autoencoder_clustering.AutoencoderClusterer(model=model, n_cluster=config.N_CLUSTER,
                                                            train_data=train_data)
    predictions = clusterer.predict(train_data)
    print(set(predictions))

    train_data_encoded = autoencoder_functions.encode_data(model=model, data=train_data, batch_size=len(train_data))

    train_data_encoded = [torch.tensor(s).unsqueeze(1).float() for s in train_data_encoded]
    train_data_encoded = torch.utils.data.DataLoader(train_data_encoded, batch_size=BATCH_SIZE, shuffle=False)

    # predictions = torch.utils.data.DataLoader(predictions, batch_size=BATCH_SIZE, shuffle=False)

    for epoch in range(500):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(train_data_encoded):
            # get the inputs; data is a list of [inputs, labels]
            inputs = data
            labels = [[0] * config.N_CLUSTER for _ in range(BATCH_SIZE)]

            for j, pred in enumerate(predictions[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]):
                labels[j][pred] = 1
            labels = torch.tensor(labels)

            optimizer.zero_grad()
            cls, loc = net(inputs, batch_size=BATCH_SIZE)

            loss = combined_loss.criterion(cls, loc, labels, epoch)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 100 == 99:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.6f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')

    test_predictions = clusterer.predict(test_data)

    test_data_encoded = test_data_encoded[:(len(test_data_encoded) - (len(test_data_encoded) % BATCH_SIZE))]
    if not test_data.any():
        raise ValueError('Batch size to big for train split.')
    test_data_encoded = [torch.tensor(s).unsqueeze(1).float() for s in test_data_encoded]
    test_data_encoded = torch.utils.data.DataLoader(test_data_encoded, batch_size=BATCH_SIZE, shuffle=False)

    cls_correct, loc_correct, total = 0, 0, 0

    with torch.no_grad():
        for i, data in enumerate(test_data_encoded):
            images, labels = data, torch.tensor(test_predictions[i * BATCH_SIZE:(i + 1) * BATCH_SIZE])
            # calculate outputs by running images through the network
            cls, loc = net(images, batch_size=BATCH_SIZE)
            # the class with the highest energy is what we choose as prediction
            _, cls_predicted = torch.max(cls.data, 1)
            _, loc_predicted = torch.max(loc.data, 1)

            total += labels.size(0)
            cls_correct += (cls_predicted == labels).sum().item()
            loc_correct += (loc_predicted == labels).sum().item()

    print(cls_correct, loc_correct)
    print(total)
    print('Cls Accuracy: %d %%' % (100 * cls_correct / total))
    print('Loc Accuracy: %d %%' % (100 * loc_correct / total))

    net.relu.register_forward_hook(get_activation('maps'))

    global_fig_path = f'heatmaps/{config.SIMULATION_TYPE}/n_cluster_{config.N_CLUSTER}/' \
                      f'simulation_{config.SIMULATION_NUMBER}_cluster_trained/sparse_{config.EMBEDDED_DIMENSION}'

    for i in range(0, min(100, len(test_data))):
        encoded_input = torch.tensor(plot_data[i]).float()
        encoded_input.unsqueeze_(0)

        cls, loc = net(encoded_input, batch_size=1)
        _, loc_predicted = torch.max(loc.data, 1)
        loc_predicted = int(loc_predicted)

        if loc_predicted == test_predictions[i]:
            print(loc_predicted)
            fig_path = f'{global_fig_path}/data_{i}'
            Path(fig_path).mkdir(parents=True, exist_ok=True)

            kernel_activation = activation['maps'].squeeze()[test_predictions[i]]

            plt.title('Spike data')
            plt.plot(test_data[i])
            plt.savefig(f'{fig_path}/spike', bbox_inches='tight')
            plt.clf()

            plt.title(f'Activation for kernel {loc_predicted}')
            plt.plot(kernel_activation)
            plt.savefig(f'{fig_path}/activation', bbox_inches='tight')
            plt.clf()

        # else:
        #    print('Localization prediction and label not equal!')


if __name__ == '__main__':
    main()
