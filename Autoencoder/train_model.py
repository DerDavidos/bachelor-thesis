import autoencoder
import numpy as np
import torch
from sklearn.model_selection import train_test_split
import pickle
import test_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    simulation = 0

    batch_size = 64

    file = f"spikes/simulation_{simulation + 1}.npy"
    with open(file, 'rb') as f:
        aligned_spikes = np.load(f)

    aligned_spikes = aligned_spikes[:int(len(aligned_spikes) * 1)]

    print(f"Data size: {len(aligned_spikes)}, Sequence length: {len(aligned_spikes[0])}")

    train_data, val_data = train_test_split(
        aligned_spikes,
        test_size=0.15,
        # random_state=RANDOM_SEED
    )

    val_data, test_data = train_test_split(
        val_data,
        test_size=0.33,
        # random_state=RANDOM_SEED
    )

    with open("training/test_data", 'wb') as dat:
        pickle.dump(test_data, dat, pickle.HIGHEST_PROTOCOL)

    train_data = [torch.tensor(s).unsqueeze(1).float() for s in train_data]
    train_dataset = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

    val_data = [torch.tensor(s).unsqueeze(1).float() for s in val_data]
    val_dataset = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=True)

    model = autoencoder.Autoencoder(input_dim=len(train_data[0]), embedded_dim=12)
    model = model.to(DEVICE)

    print()
    print("Model architecture")
    print(model)

    model, history = autoencoder.train_model(
        model,
        train_dataset=train_dataset,
        validation_dataset=val_dataset,
        n_epochs=25,
        model_path=f'models/model_simulation_{simulation}.pth',
        batch_size=batch_size,
    )

    with open("training/history", 'wb') as his:
        pickle.dump(history, his, pickle.HIGHEST_PROTOCOL)

    print(history)

    test_dataset, _, _ = autoencoder.create_dataset(test_data)

    test_model.plot_training(model, history, test_dataset)


if __name__ == '__main__':
    main()
