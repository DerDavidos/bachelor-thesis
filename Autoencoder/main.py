from autoencoder import Autoencoder, Encoder, Decoder
import autoencoder
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import sys

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def plot_histroy(history):
    ax = plt.figure().gca()
    ax.plot(history['train'])
    ax.plot(history['val'])
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'])
    plt.title('Loss over training epochs')
    plt.show()


def test_reconstructions(model, test_dataset, max_graphs=15):
    predictions, pred_losses = autoencoder.predict(model, test_dataset)

    print("Number of test spikes:", len(test_dataset))
    print("Average prediction loss:", sum(pred_losses) / len(pred_losses))

    for i in range(min(len(test_dataset), max_graphs)):
        plt.plot(test_dataset[i])
        plt.plot(predictions[i])
        plt.title("Test spike " + str(i))
        plt.show()


if __name__ == '__main__':

    load_model = False

    if len(sys.argv) > 1:
        if sys.argv[1] == "True":
            load_model = True

    file = 'data/SimulationSpikes.npy'
    # file = 'data/GeneratorSpikes.npy'

    with open(file, 'rb') as f:
        alignSpikes = np.load(f)

    alignSpikes = alignSpikes[:int(len(alignSpikes) * 1)]

    print("Data size:", len(alignSpikes), ",Sequence length:", len(alignSpikes[0]))

    train_data, val_data = train_test_split(
        alignSpikes,
        test_size=0.15,
        # random_state=RANDOM_SEED
    )

    val_data, test_data = train_test_split(
        val_data,
        test_size=0.33,
        # random_state=RANDOM_SEED
    )

    train_dataset, seq_len, n_features = autoencoder.create_dataset(train_data)
    val_dataset, _, _ = autoencoder.create_dataset(val_data)
    test_dataset, _, _ = autoencoder.create_dataset(test_data)

    if load_model:
        model = torch.load('models/colab_model.pth', map_location=torch.device(DEVICE))
        model = model.to(DEVICE)

        print()
        print("Model architecture")
        print(model)

    else:
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
            n_epochs=7
        )

        plot_histroy(history)

        model_path = 'models/model.pth'
        torch.save(model, model_path)

    print()
    print("Example encoded data")
    print(model.encoder(train_dataset[0]))
    print()

    test_reconstructions(model, test_dataset)
