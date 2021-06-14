""""""""""""" Test Model Settings """""""""""""""""""
# SIMULATION_TYPE: pedreira, martinez, own_generated
SIMULATION_TYPE = 'own_generated'
N_CLUSTER_SIMULATION = 3
SIMULATION_NUMBER = 0
EMBEDDED_DIMENSION = 8
TRAINED_WITH_CLUSTERING = True
""""""""""""" Test Model Settings """""""""""""""""""

if TRAINED_WITH_CLUSTERING:
    MODEL_PATH = f'models/{SIMULATION_TYPE}/n_cluster_{N_CLUSTER_SIMULATION}/' \
                 f'simulation_{SIMULATION_NUMBER}_cluster_trained/' \
                 f'sparse_{EMBEDDED_DIMENSION}'
else:
    MODEL_PATH = f'models/{SIMULATION_TYPE}/n_cluster_{N_CLUSTER_SIMULATION}/' \
                 f'simulation_{SIMULATION_NUMBER}_not_cluster_trained/sparse_{EMBEDDED_DIMENSION}'

DATA_PATH = f'data/{SIMULATION_TYPE}/n_cluster_{N_CLUSTER_SIMULATION}/simulation_{SIMULATION_NUMBER}'
