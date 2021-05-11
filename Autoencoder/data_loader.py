import pickle
from pathlib import Path


def load_train_test_data(simulation_number: int):
    file = f"models/simulation_{simulation_number}/data/train_data"
    with open(file, 'rb') as dat:
        train_data = pickle.load(dat)
    file = f"models/simulation_{simulation_number}/data/test_data"
    with open(file, 'rb') as dat:
        test_data = pickle.load(dat)

    print(
        f"Train data size: {len(train_data)}, Test data size: {len(test_data)}, "
        f"Sequence length: {len(train_data[0])}")

    return train_data, test_data


def save_train_val_test(directory, test_data, train_data, validation_data):
    Path(f"{directory}/data").mkdir(parents=True, exist_ok=True)
    with open(f"{directory}/data/train_data", 'wb') as dat:
        pickle.dump(train_data, dat, pickle.HIGHEST_PROTOCOL)
    with open(f"{directory}/data/validation_data", 'wb') as dat:
        pickle.dump(validation_data, dat, pickle.HIGHEST_PROTOCOL)
    with open(f"{directory}/data/test_data", 'wb') as dat:
        pickle.dump(test_data, dat, pickle.HIGHEST_PROTOCOL)
