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
    #'martinez',
    #'pedreira'
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
            spikes, labels = eng.sendDataToPython(3, simulation + 1, nargout=2)

            spikes = np.array(spikes)

            labels = labels[1]
            if len(labels) != len(spikes):
                raise Error(
                    f'Spike and labels lenght are not equal, len(spikes): {len(spikes)}, len(labels): {len(labels)}')
            print(f'Number of Cluster: {len(set(labels))}')

            Path(f'{path}/simulation_{simulation}').mkdir(parents=True, exist_ok=True)

            np.save(f'{path}/simulation_{simulation}/spikes', spikes)
            np.save(f'{path}/simulation_{simulation}/labels', np.array(labels))

            print(spikes.shape)
            print(f'Saved {simulation}')
            eng.close()

    if simulation_type == 'pedreira':

        path = f'{pathlib.Path(__file__).parent.absolute()}/../Autoencoder/spikes/pedreira'

        for simulation in range(0, 11):
            eng = matlab.engine.start_matlab()

            # adding one for Matlab counting
            spikes, labels = eng.sendDataToPython(2, simulation + 1, nargout=2)

            spikes = np.array(spikes)

            labels = labels[1]
            if len(labels) != len(spikes):
                raise Error(
                    f'Spike and labels lenght are not equal, len(spikes): {len(spikes)}, len(labels): {len(labels)}')

            new_spikes = []
            new_labels = []

            for i in range(len(spikes)):
                if labels[i] != 0:
                    new_spikes.append(spikes[i])
                    new_labels.append(labels[i])
            spikes = np.array(new_spikes)
            labels = np.array(new_labels)

            print(f'Number of Cluster: {len(set(labels))}')

            cluster_path = f'{path}/n_cluster_{len(set(labels))}'

            Path(f'{cluster_path}/simulation_{simulation}').mkdir(parents=True, exist_ok=True)

            np.save(f'{cluster_path}/simulation_{simulation}/spikes', spikes)
            np.save(f'{cluster_path}/simulation_{simulation}/labels', np.array(labels))

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
