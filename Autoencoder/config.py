import numpy as np
from mat4py import loadmat

""""""""""""""""""""""""""""""""
# SIMULATION_TYPE: pedreira, martinez, own_generated
SIMULATION_TYPE = "own_generated"
SIMULATION_NUMBER = 1
EMBEDDED_DIMENSION = 8
""""""""""""""""""""""""""""""""

if SIMULATION_TYPE == "martinez":
    N_CLUSTER = 3

if SIMULATION_TYPE == "pedreira":
    data = loadmat('../Matlab/1_SimDaten/ground_truth.mat')
    classes = np.array(data["spike_classes"][SIMULATION_NUMBER])
    N_CLUSTER = len(set(classes))
    # N_CLUSTER = 6

if SIMULATION_TYPE == "own_generated":
    with open(f'spikes/{SIMULATION_TYPE}/simulation_{SIMULATION_NUMBER}/labels.npy',
              'rb') as f:
        label = np.load(f, allow_pickle=True)
    N_CLUSTER = len(set(label))
