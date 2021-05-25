import os

import numpy as np
from mat4py import loadmat

""""""""""""""""""""""""""""""""
# pedreira, martinez, own_generated
SIMULATION_TYPE = "own_generated"
SIMULATION_NUMBER = 0
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
    for sim in os.listdir("spikes/own_generated"):
        if int(sim.split("_")[1]) == SIMULATION_NUMBER:
            N_CLUSTER = int(sim.split("_")[2].split(".")[0])
