import sys
import matlab.engine
import numpy as np
import pathlib
from itertools import chain

""""""""""""""""""""""""""""""""
# pedreira, martinez, own_generated
SIMULATION_TYPES = [
    #"pedreira",
    #"martinez",
    "own_generated"
]
""""""""""""""""""""""""""""""""

def get_data_from_matlab(simulation_type: str):
    print(simulation_type)
    if simulation_type == "pedreira":
        for simulation in chain(range(0, 9), range(89, 95)):
            eng = matlab.engine.start_matlab()
            # adding one for Matlab counting
            spikes = eng.sendDataToPython(2, simulation + 1, nargout=1)

            spikes = np.array(spikes)

            np.save(
                f"{pathlib.Path(__file__).parent.absolute()}/../Autoencoder"
                f"/spikes/pedreira/simulation_{simulation}",
                spikes)

            print(spikes.shape)
            print(f"Saved {simulation}")
            eng.close()

    if simulation_type == "martinez":
        for simulation in range(5):
            eng = matlab.engine.start_matlab()
            # adding one for Matlab counting
            spikes = eng.sendDataToPython(3, simulation + 1, nargout=1)

            spikes = np.array(spikes)

            np.save(
                f"{pathlib.Path(__file__).parent.absolute()}/../Autoencoder"
                f"/spikes/pedreira/simulation_{simulation}",
                spikes)

            print(spikes.shape)
            print(f"Saved {simulation}")
            eng.close()

    if simulation_type == "own_generated":
        eng = matlab.engine.start_matlab()
        # adding one for Matlab counting
        spikes = eng.sendDataToPython(1, 0, nargout=1)

        np.save(
            f"{pathlib.Path(__file__).parent.absolute()}/../Autoencoder"
            f"/spikes/own_generated/new_generated",
            np.array(spikes))

        print(len(spikes))

        print(f"Saved")
        eng.close()


if __name__ == '__main__':
    for sim in SIMULATION_TYPES:
        get_data_from_matlab(simulation_type=sim)
