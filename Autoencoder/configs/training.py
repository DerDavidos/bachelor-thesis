""""""""""""" Trainings Settings """""""""""""""""""
BATCH_SIZE = 128
EPOCHS = 500
EARLY_STOPPING = 10
EARLY_STOPPING_MIN_IMPROVEMENT = 0.01

# SIMULATION_TYPE: martinez, own_generated
SIMULATION_TYPE = 'own_generated'
SIMULATION_NUMBER = [0]
EMBEDDED_DIMENSIONS = [x for x in range(2, 24 + 1, 2)]
# EMBEDDED_DIMENSIONS = [12]
N_CLUSTER_SIMULATION = range(5, 5 + 1)
CLUSTERING = [True, False]  # With True and False trains with and without clustering in loss
""""""""""""" Trainings Settings """""""""""""""""""
