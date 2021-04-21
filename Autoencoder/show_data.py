import numpy as np
import DerDavidosHelper.custom_print as cp
import DerDavidosHelper.array_helper as ah

from matplotlib import pyplot as plt

with open('data/SimulationSpikes.npy', 'rb') as f:
    alignSpikes = np.load(f)

ah.pa(alignSpikes)

print(alignSpikes.shape)
for x in alignSpikes:
    plt.plot(x)

plt.show()

