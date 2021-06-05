import os

import numpy as np
from mat4py import loadmat

""""""""""""" Trainings Settings """""""""""""""""""
TRAIN_BATCH_SIZE = 512
TRAIN_EPOCHS = 500
TRAIN_EARLY_STOPPING = 10
TRAIN_EARLY_STOPPING_MIN_IMPROVEMENT = 0.005

# SIMULATION_TYPE: pedreira, martinez, own_generated
TRAIN_SIMULATION_TYPE = 'own_generated'
TRAIN_SIMULATION_NUMBER = [0]
TRAIN_EMBEDDED_DIMENSIONS = [8, 12, 16]
TRAIN_N_CLUSTER_SIMULATION = range(5, 5 + 1)
TRAIN_CLUSTERING = [True, False]  # With True and False trains with and without clustering in loss
""""""""""""" Trainings Settings """""""""""""""""""

""""""""""""" Test Model Settings """""""""""""""""""
# SIMULATION_TYPE: pedreira, martinez, own_generated
TEST_SIMULATION_TYPE = 'own_generated'
TEST_N_CLUSTER_SIMULATION = 4
TEST_SIMULATION_NUMBER = 0
TEST_EMBEDDED_DIMENSION = 8
TEST_TRAINED_WITH_CLUSTERING = True

if TEST_TRAINED_WITH_CLUSTERING:
    TEST_MODEL_PATH = f'models/{TEST_SIMULATION_TYPE}/n_cluster_{TEST_N_CLUSTER_SIMULATION}/' \
                      f'simulation_{TEST_SIMULATION_NUMBER}_cluster_trained/' \
                      f'sparse_{TEST_EMBEDDED_DIMENSION}'
else:
    TEST_MODEL_PATH = f'models/{TEST_SIMULATION_TYPE}/n_cluster_{TEST_N_CLUSTER_SIMULATION}/' \
                      f'simulation_{TEST_SIMULATION_NUMBER}_not_cluster_trained/' \
                      f'sparse_{TEST_EMBEDDED_DIMENSION}'

TEST_DATA_PATH = f'../data/{TEST_SIMULATION_TYPE}/n_cluster_{TEST_N_CLUSTER_SIMULATION}/' \
                 f'simulation_{TEST_SIMULATION_NUMBER}'
""""""""""""" Test Model Settings """""""""""""""""""

""""""""""""" Simulation Settings """""""""""""""""""
# SIMULATION_TYPE: pedreira, martinez, own_generated
SIMULATION_TYPE = 'own_generated'
N_CLUSTER_SIMULATION = 5
SIMULATION_NUMBER = 0
EMBEDDED_DIMENSION = 12
TRAINED_WITH_CLUSTERING = True
EVALUATE = False
""""""""""""" Simulation Settings """""""""""""""""""

""""""""""""" Data Loader Settings """""""""""""""""""
VALIDATION_PERCENTAGE = 0.15
TEST_PERCENTAGE = 0.15
OVERRIDE = False
""""""""""""" Data Loader Settings """""""""""""""""""

SPIKE_PATH = f'spikes/{SIMULATION_TYPE}/n_cluster_{N_CLUSTER_SIMULATION}/simulation_{SIMULATION_NUMBER}.npy'

DATA_PATH = f'data/{SIMULATION_TYPE}/n_cluster_{N_CLUSTER_SIMULATION}/simulation_{SIMULATION_NUMBER}'

if TRAINED_WITH_CLUSTERING:
    MODEL_PATH = f'models/{SIMULATION_TYPE}/n_cluster_{N_CLUSTER_SIMULATION}/' \
                 f'simulation_{SIMULATION_NUMBER}_cluster_trained/sparse_{EMBEDDED_DIMENSION}'
else:
    MODEL_PATH = f'models/{SIMULATION_TYPE}/n_cluster_{N_CLUSTER_SIMULATION}/' \
                 f'simulation_{SIMULATION_NUMBER}_not_cluster_trained/sparse_{EMBEDDED_DIMENSION}'

if SIMULATION_TYPE == 'martinez':
    N_CLUSTER = 3

if SIMULATION_TYPE == 'pedreira':
    data = loadmat('../Matlab/1_SimDaten/ground_truth.mat')
    classes = np.array(data['spike_classes'][SIMULATION_NUMBER])
    N_CLUSTER = len(set(classes))
    # N_CLUSTER = 6

if SIMULATION_TYPE == 'own_generated':
    N_CLUSTER = N_CLUSTER_SIMULATION

    if os.path.exists(DATA_PATH):
        pass
        # determinate_number_of_cluster()
