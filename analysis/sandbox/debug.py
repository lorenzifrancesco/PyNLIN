"""
Just a debug script for functionalities
"""
import matplotlib.pyplot as plt
import numpy as np


def main():
    """Plot stored debug integrals if present."""
    integrals = np.load("results/debug_integrals.npy", allow_pickle=True)

    plt.figure(figsize=(10, 6))

    for i, waveform in enumerate(integrals):
        plt.plot(waveform, label=f'Waveform {i+1}')

    plt.xlabel('Sample Index')
    plt.ylabel('Intensity')
    plt.title('Waveforms from Integrals')
    plt.legend(loc='best')
    plt.grid(True)

    plt.show()


if __name__ == "__main__":
    main()
