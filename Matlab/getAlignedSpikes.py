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
    # 'martinez',
    'own_generated'
]
""""""""""""""""""""""""""""""""


def get_data_from_matlab(simulation_type: str):
    """ Starts Matlab engine and uses Matlab code to get aligined spikes and save it as numpy arrays

    Parameters:
        simulation_type: Which simulation type to load spikes from
    """
    print(simulation_type)

    if simulation_type == 'martinez':
        path = f'{pathlib.Path(__file__).parent.absolute()}/../Autoencoder/spikes/martinez/n_cluster_3/'
        # Simulation 2 throws an error
        for simulation in [0, 1, 3, 4]:
            eng = matlab.engine.start_matlab()

            # adding one for Matlab counting
            spikes, _ = eng.sendDataToPython(3, simulation + 1, nargout=2)

            spikes = np.array(spikes)

            Path(f'{path}/simulation_{simulation}').mkdir(parents=True, exist_ok=True)

            np.save(f'{path}/simulation_{simulation}/spikes', spikes)

            print(spikes.shape)
            print(f'Saved {simulation}')
            eng.close()

    if simulation_type == 'own_generated':

        eng = matlab.engine.start_matlab()
        spikes, labels = eng.sendDataToPython(1, 0, nargout=2)
        eng.close()

        labels = labels[1]
        if len(labels) != len(spikes):
            raise Error(f'Spike and labels lenght are not equal, len(spikes): {len(spikes)}, len(labels): {len(labels)}')
        print(f'Number of Cluster: {len(set(labels))}')

        path = f'{pathlib.Path(__file__).parent.absolute()}/../Autoencoder/spikes/own_generated/n_cluster_{len(set(labels))}'
        Path(path).mkdir(parents=True, exist_ok=True)

        sims = [-1]
        for x in os.listdir(path):
            sims.append(int(x.split('_')[1].split('.')[0]))
        sim_number = max(sims) + 1

        path += f'/simulation_{sim_number}'
        Path(path).mkdir(parents=True, exist_ok=True)

        np.save(f'{path}/spikes', np.array(spikes))
        np.save(f'{path}/labels', np.array(labels))

        print(f'Saved: {path}')


if __name__ == '__main__':
    for sim in SIMULATION_TYPES:
        get_data_from_matlab(simulation_type=sim)
