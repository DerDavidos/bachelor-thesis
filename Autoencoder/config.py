import os

import numpy as np
from mat4py import loadmat
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

""""""""""""" Simulation Settings """""""""""""""""""
# SIMULATION_TYPE: pedreira, martinez, own_generated
SIMULATION_TYPE = "own_generated"
N_CLUSTER_SIMULATION = 3
SIMULATION_NUMBER = 0
EMBEDDED_DIMENSION = 12
TRAINED_WITH_CLUSTERING = True
""""""""""""" Simulation Settings """""""""""""""""""

""""""""""""" Trainings Settings """""""""""""""""""
BATCH_SIZE = 256
EPOCHS = 500
EARLY_STOPPING = 10
EARLY_STOPPING_MIN_IMPROVEMENT = 0.005
TRAIN_EMBEDDED_DIMENSIONS = [8, 12]
TRAIN_N_CLUSTER_SIMULATION = range(3, 3 + 1)
TRAIN_CLUSTERING = [True, False]  # With True and False trains with and without clustering in loss
""""""""""""" Trainings Settings """""""""""""""""""

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
    else:
        print("Spike data does not exist.")
