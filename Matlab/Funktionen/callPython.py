import numpy as np
import os
import pathlib


def SaveFeatures(alignSpikes, path):
    np.save(str(pathlib.Path(__file__).parent.absolute()) + "/../../Autoencoder/spikes/" +
            path.split("/")[-1].replace(".mat", ""), np.array(alignSpikes))
    print("Saveed")
