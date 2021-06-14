import sys
import matlab.engine
import numpy as np
import pathlib
from itertools import chain
from pathlib import Path
import os

""""""""""""""""""""""""""""""""
# martinez, own_generated
SIMULATION_TYPES = [
    'martinez',
    # 'own_generated'
]
""""""""""""""""""""""""""""""""


def get_data_from_matlab(simulation_type: str):
    """ Starts Matlab engine and uses Matlab code to get aligined spikes and save it as numpy arrays

    Parameters:
        simulation_type: Which simulation type to load spikes from
    """
    print(simulation_type)

    if simulation_type == 'martinez':
        for simulation in range(5):
            eng = matlab.engine.start_matlab()

            # adding one for Matlab counting
            spikes, _ = eng.sendDataToPython(3, simulation + 1, nargout=2)

            spikes = np.array(spikes)

            path = f'{pathlib.Path(__file__).parent.absolute()}/../Autoencoder/spikes/martinez/n_cluster_3'

            Path(path).mkdir(parents=True, exist_ok=True)

            np.save(f'{path}/simulation_{simulation}', spikes)

            print(spikes.shape)
            print(f'Saved {simulation}')
            eng.close()

    if simulation_type == 'own_generated':

        eng = matlab.engine.start_matlab()
        spikes, labels = eng.sendDataToPython(1, 0, nargout=2)
        eng.close()

        labels = np.array(labels).reshape(-1)
        new_labels = []
        for i in range(1, len(labels), 2):
            if int(labels[i]) != 0:
                new_labels.append(int(labels[i] - 1))
        labels = len(set(new_labels))

        path = f'{pathlib.Path(__file__).parent.absolute()}/../Autoencoder/spikes/own_generated/n_cluster_{labels}'

        Path(path).mkdir(parents=True, exist_ok=True)

        sims = [-1]
        for x in os.listdir(f'../Autoencoder/spikes/own_generated/n_cluster_{labels}'):
            sims.append(int(x.split('_')[1].split('.')[0]))
        sim_number = max(sims) + 1

        np.save(f'{path}/simulation_{sim_number}', np.array(spikes))

        print(f'Saved, {path}/simulation_{sim_number}')


if __name__ == '__main__':
    for sim in SIMULATION_TYPES:
        get_data_from_matlab(simulation_type=sim)
