import os

import numpy as np
from mat4py import loadmat
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

EVALUATE = False

""""""""""""" Simulation Settings """""""""""""""""""
# SIMULATION_TYPE: pedreira, martinez, own_generated
SIMULATION_TYPE = "own_generated"
N_CLUSTER_SIMULATION = 4
SIMULATION_NUMBER = 1
EMBEDDED_DIMENSION = 8
TRAINED_WITH_CLUSTERING = True
""""""""""""" Simulation Settings """""""""""""""""""

""""""""""""" Data Loader Settings """""""""""""""""""
VALIDATION_PERCENTAGE = 0.15
TEST_PERCENTAGE = 0.15
OVERRIDE = False
""""""""""""" Data Loader Settings """""""""""""""""""

SPIKE_PATH = f'spikes/{SIMULATION_TYPE}/n_cluster_{N_CLUSTER_SIMULATION}/' \
             f'simulation_{SIMULATION_NUMBER}.npy'

DATA_PATH = f'data/{SIMULATION_TYPE}/n_cluster_{N_CLUSTER_SIMULATION}/' \
            f'simulation_{SIMULATION_NUMBER}'

if TRAINED_WITH_CLUSTERING:
    MODEL_PATH = f"models/{SIMULATION_TYPE}/n_cluster_{N_CLUSTER_SIMULATION}/" \
                 f"simulation_{SIMULATION_NUMBER}_cluster_trained/" \
                 f"sparse_{EMBEDDED_DIMENSION}"
else:
    MODEL_PATH = f"models/{SIMULATION_TYPE}/n_cluster_{N_CLUSTER_SIMULATION}/" \
                 f"simulation_{SIMULATION_NUMBER}_not_cluster_trained/" \
                 f"sparse_{EMBEDDED_DIMENSION}"


def determinate_number_of_cluster() -> int:
    with open(f"{DATA_PATH}/train_data.npy", 'rb') as file:
        train_data = np.load(file, allow_pickle=True)

    pca = PCA(n_components=8)
    wcss = []
    K = range(1, 11)
    for i in K:
        model = KMeans(n_clusters=i)
        train_data_sparse = pca.fit_transform(train_data)
        model.fit(train_data_sparse)
        wcss.append(model.inertia_)
    predicted_number_of_cluster = 1
    while wcss[predicted_number_of_cluster - 1] > wcss[predicted_number_of_cluster] * 1.3:
        predicted_number_of_cluster += 1
    print(f"Config: Predicted number of Cluster: {predicted_number_of_cluster}, "
          f"True number: {N_CLUSTER_SIMULATION}")

    return predicted_number_of_cluster


if SIMULATION_TYPE == "martinez":
    N_CLUSTER = 3

if SIMULATION_TYPE == "pedreira":
    data = loadmat('../Matlab/1_SimDaten/ground_truth.mat')
    classes = np.array(data["spike_classes"][SIMULATION_NUMBER])
    N_CLUSTER = len(set(classes))
    # N_CLUSTER = 6

if SIMULATION_TYPE == "own_generated":
    N_CLUSTER = N_CLUSTER_SIMULATION

    if os.path.exists(DATA_PATH):
        determinate_number_of_cluster()
