import sys
import matlab.engine
import numpy as np
import pathlib
from itertools import chain
from pathlib import Path
import os

""""""""""""""""""""""""""""""""
# pedreira, martinez, own_generated
SIMULATION_TYPES = [
    # "pedreira",
    # "martinez",
    "own_generated"
]
""""""""""""""""""""""""""""""""


def get_data_from_matlab(simulation_type: str):
    """ Starts Matlab engine and uses Matlab code to get aligined spikes and save it as numpy arrays

    Parameters:
        simulation_type: Which simulation type to load spikes from
    """
    print(simulation_type)
    if simulation_type == "pedreira":
        for simulation in chain(range(0, 9), range(89, 95)):
            eng = matlab.engine.start_matlab()
            # adding one for Matlab counting
            spikes, _ = eng.sendDataToPython(2, simulation + 1, nargout=2)

            spikes = np.array(spikes)

            path = f"{pathlib.Path(__file__).parent.absolute()}/../Autoencoder/spikes/pedreira/" \
                   f"simulation_{simulation}"

            Path(path).mkdir(parents=True, exist_ok=True)

            np.save(f"{path}/spikes", spikes)

            print(spikes.shape)
            print(f"Saved {simulation}")
            eng.close()

    if simulation_type == "martinez":
        for simulation in range(5):
            eng = matlab.engine.start_matlab()

            # adding one for Matlab counting
            spikes, _ = eng.sendDataToPython(3, simulation + 1, nargout=2)

            spikes = np.array(spikes)

            path = f"{pathlib.Path(__file__).parent.absolute()}/../Autoencoder/spikes/pedreira/" \
                   f"simulation_{simulation}"

            Path(path).mkdir(parents=True, exist_ok=True)

            np.save(f"{path}/spikes", spikes)

            print(spikes.shape)
            print(f"Saved {simulation}")
            eng.close()

    if simulation_type == "own_generated":

        sims = [-1]
        for x in os.listdir("../Autoencoder/spikes/own_generated"):
            sims.append(int(x.split("_")[1]))
        sim_number = max(sims) + 1

        eng = matlab.engine.start_matlab()
        spikes, labels = eng.sendDataToPython(1, 0, nargout=2)
        eng.close()

        path = f"{pathlib.Path(__file__).parent.absolute()}/../Autoencoder/spikes/own_generated/" \
               f"simulation_{sim_number}"

        Path(path).mkdir(parents=True, exist_ok=True)

        labels = np.array(labels).reshape(-1)
        for i in range(len(labels)):
            labels[i] -= 1

        print(len(labels), len(spikes))

        np.save(f"{path}/spikes", np.array(spikes))
        np.save(f"{path}/labels", labels)

        print(f"Saved")


if __name__ == '__main__':
    for sim in SIMULATION_TYPES:
        get_data_from_matlab(simulation_type=sim)
