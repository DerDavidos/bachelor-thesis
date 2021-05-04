import numpy as np
import os
import pathlib

def feature_extraction(alignSpikes, noFrame, coef_pos, corrcoeff, FESpikes, FESpikes_neg, cnt, cnt2, TMSpike):
    return FESpikes, FESpikes_neg, TMSpike, coef_pos, corrcoeff, cnt, cnt2


def testwr():
    with open("/Users/david/thesis/test.txt", 'w') as w:
        w.write("Test")


def SaveFeatures(alignSpikes, path):
    np.save(str(pathlib.Path(__file__).parent.absolute()) + "/../../Autoencoder/spikes/" + path.split("/")[-1], np.array(alignSpikes))
