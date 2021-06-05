""""""""""""" Trainings Settings """""""""""""""""""
BATCH_SIZE = 512
EPOCHS = 500
EARLY_STOPPING = 10
EARLY_STOPPING_MIN_IMPROVEMENT = 0.005

# SIMULATION_TYPE: pedreira, martinez, own_generated
SIMULATION_TYPE = "own_generated"
TRAIN_SIMULATION_NUMBER = [0]
TRAIN_EMBEDDED_DIMENSIONS = [8, 12]
TRAIN_N_CLUSTER_SIMULATION = range(5, 5 + 1)
TRAIN_CLUSTERING = [True, False]  # With True and False trains with and without clustering in loss
""""""""""""" Trainings Settings """""""""""""""""""

""""""""""""" Test Model Settings """""""""""""""""""
# SIMULATION_TYPE: pedreira, martinez, own_generated
TEST_SIMULATION_TYPE = "own_generated"
TEST_N_CLUSTER_SIMULATION = 4
TEST_SIMULATION_NUMBER = 1
TEST_EMBEDDED_DIMENSION = 8
TEST_TRAINED_WITH_CLUSTERING = True

if TEST_TRAINED_WITH_CLUSTERING:
    TEST_MODEL_PATH = f"../models/{TEST_SIMULATION_TYPE}/n_cluster_{TEST_N_CLUSTER_SIMULATION}/" \
                      f"simulation_{TEST_SIMULATION_NUMBER}_cluster_trained/" \
                      f"sparse_{TEST_EMBEDDED_DIMENSION}"
else:
    TEST_MODEL_PATH = f"../models/{TEST_SIMULATION_TYPE}/n_cluster_{TEST_N_CLUSTER_SIMULATION}/" \
                      f"simulation_{TEST_SIMULATION_NUMBER}_not_cluster_trained/" \
                      f"sparse_{TEST_EMBEDDED_DIMENSION}"
""""""""""""" Test Model Settings """""""""""""""""""
