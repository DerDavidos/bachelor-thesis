""""""""""""" Trainings Settings """""""""""""""""""
BATCH_SIZE = 512
EPOCHS = 500
EARLY_STOPPING = 10
EARLY_STOPPING_MIN_IMPROVEMENT = 0.01

# SIMULATION_TYPE: pedreira, martinez, own_generated
SIMULATION_TYPE = 'own_generated'
SIMULATION_NUMBER = [0]
EMBEDDED_DIMENSIONS = [8, 12, 16]
N_CLUSTER_SIMULATION = range(2, 5 + 1)
CLUSTERING = [True, False]  # With True and False trains with and without clustering in loss
""""""""""""" Trainings Settings """""""""""""""""""
