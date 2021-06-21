import numpy as np
from matplotlib import pyplot as plt


def main():
    values = [
        'U_SA * exp(-t/tau)',
        '-U_SA * exp(-t/tau)',
        'U_SA * n * gaussmf(t, [0.6 * tau t(end)/2])',
        'U_SA * (n * gaussmf(t, [tau/2 t(end)/2]) - (1-n)*sin(2*pi*f*t))',
        'U_SA * (U0 + exp(-(t-t(A(1)))/tau). * heaviside(t-t(A(1))) - (1-n) * sin(2*pi*f*t))',
    ]

    spikes = []
    for x in [1, 2, 3]:
        with open(f'spike_{x}.npy', 'rb') as file:
            spikes.append(np.mean(np.load(file), axis=0))

    plt.ylim(np.amin(spikes) * 1.05, np.amax(spikes) * 1.05)

    for i, spike in enumerate(spikes):
        plt.plot(spike, label=values[i])

    plt.legend()
    plt.ylabel('U_{AP} / µV')
    plt.xlabel('Postion in time')
    plt.title('Average spike from each spike type')
    plt.show()


if __name__ == '__main__':
    main()
