import matplotlib.pyplot as plt
import numpy as np

import pynlin
import pynlin.constellations
import pynlin.nlin
import pynlin.pulses
import pynlin.utils
import pynlin.wdm
from pynlin.utils import dBm2watt

plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['font.weight'] = '500'
plt.rcParams['font.size'] = '26'


def arity_coefficient():
    """Plot modulation-order dependent kurtosis factors for QAM/PSK constellations."""
    m_QAM = [4, 16, 64, 256, 1024]
    m_PSK = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    arity_list = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    fig_arity = fig_arity = plt.figure(figsize=(10, 5))
    var_QAM = []
    var_PSK = []
    for m in m_QAM:
        average_power = dBm2watt(0)
        qam = pynlin.constellations.QAM(m)
        qam_symbols = qam.symbols()
        qam_symbols = qam_symbols / np.sqrt(np.mean(np.abs(qam_symbols)**2)) # normalize the symbol average energy to 1
        var_QAM.append(np.mean(np.abs(qam_symbols)**4) /
                       np.mean(np.abs(qam_symbols)**2) ** 2 - 1)

    for m in m_PSK:
        average_power = dBm2watt(0)
        qam = pynlin.constellations.PSK(m)
        qam_symbols = qam.symbols()

        qam_symbols = qam_symbols / np.sqrt(np.mean(np.abs(qam_symbols)**2))
        var_PSK.append(np.mean(np.abs(qam_symbols)**4) /
                       np.mean(np.abs(qam_symbols)**2) ** 2 - 1)
    print("QAM variance factors:", var_QAM)
    print("QAM values for mu_0 ", np.array(var_QAM) + 1)
    var_QAM = np.array(var_QAM)
    var_PSK = np.array(var_PSK)
    plt.semilogx(m_QAM, var_QAM, color='black', base=2,
                 linestyle="none", marker="x", markersize=13, label="QAM")
    plt.semilogx(m_PSK, var_PSK, color='black', base=2,
                 linestyle="none", marker="o", markersize=10, label="PSK")
    plt.annotate("{:1.3f}".format(
        var_QAM[1]), (16, var_QAM[1] - 0.05))
    plt.annotate("{:1.3f}".format(
        var_QAM[2]), (64, var_QAM[2] - 0.05))
    plt.annotate("{:1.3f}".format(var_QAM[3]),
                 (256 - 70, var_QAM[3] - 0.05))
    plt.annotate("{:1.3f}".format(var_QAM[4]),
                 (1024 - 390, var_QAM[4] - 0.05))
    plt.grid()
    plt.xlabel("Modulation order")
    plt.xticks(ticks=arity_list, labels=arity_list)
    plt.ylabel(r"$\mu$ normalized to 16-QAM")
    # plt.yticks(ticks=constellation_variance, labels=["1.0", "1.190", "1.235"])
    plt.subplots_adjust(wspace=0.0, hspace=0, right=9.8 / 10, bottom=-1)
    plt.legend(loc="center right")
    plt.minorticks_on()

    fig_arity.tight_layout()
    fig_arity.savefig("media/modulation_order_noise.pdf")
    print("Arity coefficient plot saved in media/modulation_order_noise.pdf")


def constellation_statistics():
    """Compute and plot symbol-energy variance scaling versus QAM order."""
    average_energy = 1
    power_dBm_list = np.linspace(-20, 0, 3)
    arity_list = [16, 64, 256]

    constellation_variance = []

    for ar_idx, M in enumerate(arity_list):
        qam = pynlin.constellations.QAM(M)

        qam_symbols = qam.symbols()
        cardinality = len(qam_symbols)

        # assign specific average optical energy
        qam_symbols = qam_symbols / \
            np.sqrt(np.mean(np.abs(qam_symbols)**2)) * \
            np.sqrt(average_energy)

        constellation_variance.append(
            np.mean(np.abs(qam_symbols)**4) - np.mean(np.abs(qam_symbols)**2) ** 2)
        print(f"QAM-{M:>5}: {constellation_variance[ar_idx]:10.5f}")

    fig_arity = plt.figure(figsize=(8, 9))

    # normalized to 16-QAM variance
    plt.loglog(arity_list, constellation_variance,
               marker='x', markersize=10, color='black')
    plt.minorticks_off()
    plt.grid()
    plt.xlabel("Modulation order")
    plt.xticks(ticks=arity_list, labels=arity_list)
    plt.ylabel(r"variance scale factor")
    plt.yticks(ticks=constellation_variance, labels=["1.0", "1.190", "1.235"])
    plt.subplots_adjust(wspace=0.0, hspace=0, right=9.8 / 10, bottom=-1)

    fig_arity.tight_layout()
    fig_arity.savefig("media/order_noise.pdf")

if __name__ == "__main__":
    arity_coefficient()
    constellation_statistics()
    # plt.show()
