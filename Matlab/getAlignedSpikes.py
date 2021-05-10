import sys
import matlab.engine
import numpy as np
import pathlib


def main():
    for simulation in range(5):
        eng = matlab.engine.start_matlab()
        # adding one for Matlab counting
        spikes = eng.sendDataToPython(simulation + 1, nargout=1)

        np.save(
            f"{pathlib.Path(__file__).parent.absolute()}/../Autoencoder/spikes/simulation_{simulation}",
            np.array(spikes))
        print(f"Saved {simulation}")
        eng.close()


if __name__ == '__main__':
    main()
