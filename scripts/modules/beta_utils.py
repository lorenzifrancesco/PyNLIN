import numpy as np
import logging
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import pynlin.wdm
from pynlin.utils import nu2lambda
from scripts.modules.load_fiber_values import load_group_delay, load_dummy_group_delay
from numpy import polyval
from pynlin.fiber import MMFiber
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import ScalarFormatter
import scripts.modules.cfg as cfg

rc('text', usetex=True)

def beta2rms(beta2a, beta2b):
    return -np.sqrt(beta2a**2 + beta2b**2)


def beta2rms_complementary(beta2rms, beta2a):
    tmp = 2 * beta2rms**2 - beta2a**2
    assert (tmp > 0)
    return -np.sqrt(tmp)


def beta2avg(beta2a, beta2b):
    return (beta2a + beta2b) / 2


def beta2avg_complementary(beta2avg, beta2a):
    return beta2avg - beta2a


def fig3_fig4(cf_file="./input/mmf.toml"):
    formatter = ScalarFormatter()
    formatter.set_scientific(True)
    formatter.set_powerlimits([0, 0])

    cf = cfg.load_toml_to_struct("./input/mmf.toml")
    oi_fit = np.load('results/oi_fit.npy')
    oi_avg = np.load('results/oi_avg.npy')
    use_avg_oi = False

    print(
        f"Loading a ITU-T standardized WDM grid \n [spacing: {cf.channel_spacing * 1e-9:.3e}GHz, center: {cf.center_frequency * 1e-12:.3e}THz] \n")
    # beta1_params = load_dummy_group_delay()
    beta1_params = load_group_delay()
    # beta1_params = load_dummy_group_delay()
    # print(beta1_params.shape)
    dpi = 300
    grid = False
    wdm = pynlin.wdm.WDM(
        spacing=cf.channel_spacing,
        num_channels=cf.n_channels,
        center_frequency=cf.center_frequency
    )
    freqs = wdm.frequency_grid()
    modes = [0, 1, 2, 3]
    mode_names = ['LP01', 'LP11', 'LP21', 'LP02']

    fiber = MMFiber(
        effective_area=80e-12,
        overlap_integrals=oi_fit,
        group_delay=beta1_params,
        length=100e3
    )

    beta1 = np.zeros((len(modes), len(freqs)))
    for i in modes:
        beta1[i, :] = fiber.group_delay.evaluate_beta1(i, freqs)
    beta2 = np.zeros((len(modes), len(freqs)))
    for i in modes:
        beta2[i, :] = fiber.group_delay.evaluate_beta2(i, freqs)
    beta1 = np.array(beta1)
    beta2 = np.array(beta2)
    # print(beta2[1, :])

#   plt.clf()
#   sns.heatmap(beta1, cmap="coolwarm", square=False,
#               xticklabels=freqs, yticklabels=modes)
#   plt.xlabel('Frequency (Hz)')
#   plt.ylabel('Modes')
#   plt.title('Beta1 Heatmap')
#   plt.savefig("media/dispersion/disp.png", dpi=dpi)

    # for each channel, we compute the total number of collisions that
    # needs to be computed for evaluating the total noise on that channel.
    T = 1 / cf.baud_rate
    L = cf.fiber_length

    collisions = np.zeros((len(modes), len(freqs)))
    for i in range(len(modes)):
        for j in range(len(freqs)):
            collisions[i, j] = np.floor(np.abs(np.sum(beta1 - beta1[i, j])) * L / T)

    collisions_single = np.zeros((1, len(freqs)))
    for j in range(len(freqs)):
        collisions_single[0, j] = np.floor(
            np.abs(np.sum(beta1[0:] - beta1[0, j])) * L / T)

    nlin = np.zeros((len(modes), len(freqs)))
    # only flat up to this point

    ############################
    # GAUSSIAN NOISE
    colors = ["blue", "orange", "green", "red"]

    def pair_noise(dgd):
        return np.where(
            dgd > 3e-15,  # this needs to be set carefully
            L / (T * dgd),
            1.772 * (L / (T * np.sqrt(2 * np.pi)))**2
        )
    #
    for i in range(len(modes)):
        for j in range(len(freqs)):
            nlin[i, j] = np.sum(pair_noise(np.abs(beta1 - beta1[i, j]))
                                [(beta1 - beta1[i, j]) != 0])
    #
    plt.clf()
    plt.figure(figsize=(3.6, 3.2))
    for i in range(4):
        plt.semilogy(freqs * 1e-12,
                     nlin[i, :] * 1e-30,
                     label=mode_names[i],
                     lw=1.2,
                     color=colors[i])
    #

    def pair_noise(dgd):
        return np.where(
            dgd > 1e-20,
            L / (T * dgd),
            1.772 * (L / (T * np.sqrt(2 * np.pi)))**2
        )
    #
    for i in range(len(modes)):
        for j in range(len(freqs)):
            nlin[i, j] = np.sum(pair_noise(np.abs(beta1 - beta1[i, j]))
                                [(beta1 - beta1[i, j]) != 0])
    #
    for i in range(4):
        plt.semilogy(freqs * 1e-12,
                     nlin[i, :] * 1e-30,
                     #  label=mode_names[i],
                     lw=1.2,
                     ls=":",
                     color=colors[i])
    plt.xlabel(r'$f \; [\mathrm{THz}]$')
    plt.ylabel(r'$\mathrm{NLIN} \; [\mathrm{km}^2/\mathrm{ps}^{2}]$')
    plt.legend(labelspacing=0.1)
    plt.grid(grid)
    plt.tight_layout()
    plt.savefig(f"media/nlin.pdf", dpi=dpi)
    ############################

    nlin_no_cross = np.zeros((len(modes), len(freqs)))
    for i in range(len(modes)):
        for j in range(len(freqs)):
            nlin_no_cross[i, j] = np.sum(
                L / (np.abs(beta1[i, :] - beta1[i, j])[(beta1[i, :] - beta1[i, j]) != 0] * T))

    # plt.clf(
    # sns.heatmap(collisions, cmap="magma", square=False, xticklabels=freqs, yticklabels=modes)
    # plt.xlabel('Frequency (Hz)')
    # plt.ylabel('Modes')
    # plt.title('Total number of collision due to system')
    # plt.savefig("media/dispersion/disp.png")
    # plt.show()

    plt.clf()
    for i in range(4):
        plt.plot(freqs * 1e-12, collisions[i, :], label=mode_names[i])
    plt.xlabel('Frequency (THz)')
    plt.ylabel(r'$m_{\mathrm{max}}$')
    plt.legend(labelspacing=0.1)
    # plt.grid(grid)
    plt.savefig(f"media/dispersion/collisions.png", dpi=dpi)
    # plt.show()

    plt.clf()
    plt.plot(freqs * 1e-12, collisions_single[0, :], label=mode_names[i])
    plt.xlabel('Frequency (THz)')
    plt.ylabel(r'$m_{\mathrm{max}}$')
    plt.legend(labelspacing=0.1)
    plt.grid(grid)
    plt.savefig(f"media/dispersion/collisions_single.png", dpi=dpi)
    # plt.show()

    plt.clf()
    for i in range(len(modes)):
        plt.semilogy(freqs * 1e-12, nlin_no_cross[i, :]
                     * 1e-30, label=mode_names[i], marker='x')
    plt.xlabel('Frequency (THz)')
    plt.ylabel('NLIN coeff')
    plt.legend(labelspacing=0.1)
    plt.grid(grid)
    plt.tight_layout()
    plt.savefig(f"media/dispersion/nlin_no_cross.png", dpi=dpi)
    # plt.show()

    plt.clf()
    plt.figure(figsize=(3.6, 3.4))
    for i in range(4):
        plt.plot(freqs * 1e-12, beta1[i, :] * 1e9, label=mode_names[i], lw=2)
    minn = np.min(beta1)
    maxx = np.max(beta1)
    #
    # WDM band edges
    plt.axvline(190.9, color="grey", ls="-.", lw=1.5)
    plt.axvline(200.9, color="grey", ls="-.", lw=1.5)
    #
    # plt.xticks([185, 193, 196, 206])
    freq_boundaries = [189, 192.7, 197, 206]
    for i, label in enumerate(['L', 'C', 'S']):
        plt.text(freq_boundaries[i] + 1, 4.8932, label, ha='center', va='bottom')
        plt.axvline(freq_boundaries[i], color="pink", lw=1)
    #
    plt.xlabel(r'$f \; [\mathrm{THz}]$')
    plt.ylabel(r'$\beta_1 [\mathrm{ns}/\mathrm{m}]$')
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0), useOffset=4.89)

    plt.legend(labelspacing=0.1)
    # plt.grid(grid)
    plt.tight_layout()
    plt.savefig(f"media/dispersion/beta1.pdf", dpi=dpi)

    # plt.clf()
    # plt.figure(figsize=(4.6, 4))

    # for i in range(4):
    #     plt.plot(freqs * 1e-12, beta2[i, :] * 1e27, label=mode_names[i])
    # plt.xlabel(r'$f \; [\mathrm{THz}]$')
    # plt.ylabel(r'$\beta_2$ [ps$^2$/km]')
    # plt.legend(labelspacing=0.1)
    # # plt.grid(grid)
    # plt.tight_layout()
    # plt.savefig(f"media/dispersion/beta2.png", dpi=300)


def get_fig4(cf_file = "./input/mmf.toml"):
    cf = cfg.load_toml_to_struct(cf_file)
    wdm = pynlin.wdm.WDM(
        spacing=cf.channel_spacing,
        num_channels=cf.n_channels,
        center_frequency=cf.center_frequency
    )
    oi_fit = np.load('results/oi_fit.npy')
    beta1_params = load_group_delay()
    fiber = MMFiber(
        effective_area=80e-12,
        overlap_integrals=oi_fit,
        group_delay=beta1_params,
        length=100e3
    )

    freqs = wdm.frequency_grid()
    modes = [0, 1, 2, 3]
    beta1 = np.zeros((len(modes), len(freqs)))
    for i in modes:
        beta1[i, :] = fiber.group_delay.evaluate_beta1(i, freqs)
    beta2 = np.zeros((len(modes), len(freqs)))
    for i in modes:
        beta2[i, :] = fiber.group_delay.evaluate_beta2(i, freqs)
    beta1 = np.array(beta1)
    beta1_differences = np.abs(
        beta1[:, :, np.newaxis, np.newaxis] - beta1[np.newaxis, np.newaxis, :, :])
    beta1_differences = beta1_differences[beta1_differences != 0]

    x_norm = cf.fiber_length * cf.baud_rate
    mask = (beta1_differences < 200 * 1e-1)
    hist, edges = np.histogram(np.log(beta1_differences[mask] * 1e12), bins=50)
    hist = hist / 2.0
    print(edges)
    print(hist)
    plt.clf()
    plt.figure(figsize=(3.6, 2.4))
    plt.bar(np.power(10, edges[:-1]),
            hist,
            width=np.diff(np.power(10, edges)) / 1.5,
            zorder=3,
            edgecolor='blue',
            facecolor='none')

    x_start = 0.2 / x_norm * 1e12
    x_end = 3.0 / x_norm * 1e12
    x_start = 1e-4
    x_end  = 2e-2
    plt.axvline(x_start, color='red', lw=1, ls='--')
    plt.axvline(x_end, color='red', lw=1, ls='--')
    plt.axvspan(x_start, x_end, color='red', alpha=0.3)
    # count how many channel pairs are between x_start and x_end
    count = np.sum((beta1_differences * 1e12 > x_start) & (beta1_differences * 1e12 < x_end))
    total = np.sum(mask)
    print(f"Number of channel pairs with DGD between {x_start:.2f} and {x_end:.2f} ps/m: {count} over {total} -> {count/total:.2%}")
    plt.xlabel(r'$\Delta\beta_1$ [ps/m]')
    # plt.xlabel(r'$L_W/L$')
    plt.ylabel('channel pair count')
    plt.grid(axis='y', zorder=1)
    plt.tight_layout()
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig(f"media/4-statistics.pdf", dpi=300)
    print("Saved figure 4 to media/4-statistics.pdf")
    # mask = (beta1_differences < 0.1 * 1e-12)
    # print("Average DGD: ", np.mean(beta1_differences * 1e12))
    # hist, edges = np.histogram(beta1_differences[mask]*1e12, bins=20)
    # hist = hist / 2.0
    # plt.clf()
    # plt.figure(figsize=(4, 3.5))
    # plt.bar(edges[:-1], hist, width=np.diff(edges), zorder=3)
    # plt.xlabel('DGD (ps/m)')
    # plt.ylabel('channel pair count')
    # plt.grid(axis='y', zorder=0)
    # plt.tight_layout()
    # plt.savefig(f"media/dispersion/DGD_histogram_zoom.png", dpi=dpi)

    # fig = plt.figure(figsize=(3.6, 2.5))  # Overall figure size
    # # gs = GridSpec(nrows=2, ncols=1, height_ratios=[2, 1], hspace=0.1)  # The height_ratios adjust the relative sizes
    # gs = GridSpec(nrows=1, ncols=1, height_ratios=[1], hspace=0.1)  # The height_ratios adjust the relative sizes
    # hist, edges = np.histogram(beta1_differences[mask]*1e12, bins=200)
    # hist = hist / 2.0
    # # Create subplots
    # ax1 = fig.add_subplot(gs[0])  # Top subplot (smaller)
    # # ax2 = fig.add_subplot(gs[1])  # Bottom subplot (larger)
    # # ax3 = fig.add_subplot(gs[2])  # Bottom subplot (larger)
    # # Plot histogram on the top subplot
    # ax1.bar(edges[:-1], hist, width=np.diff(edges), zorder=3)
    # print("WARN: we are handling edges in a strange way!")
    # edges = edges *1e-12
    # ax1.set_ylabel('channel pair count')
    # ax1.grid(axis='y', zorder=0)
    # ax1.set_xticklabels([])
    # ax1.yaxis.set_major_formatter(formatter)
    # # ax2.plot(edges[:-1]*1e12, edges[:-1]*L/T, color='blue')
    # # ax2.set_ylabel(r'$m_{\mathrm{max}}$')
    # # ax2.semilogy(edges[:-1]*1e12, L/T / edges[:-1] * 1e-30, color='red')
    # # ax2.set_ylabel(r'$N \; [\mathrm{km}^2/\mathrm{ps}^2$]')
    # ax1.set_xlabel(r'$\Delta{\beta_1} [\mathrm{ps}/\mathrm{m}]$')
    # # ax2.legend(loc='upper right')
    # # plt.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
    # # plt.tight_layout()
    # plt.subplots_adjust(left=0.2, right=0.95, top=0.93, bottom=0.15, hspace=0.3)
    # plt.savefig(f"media/4-statistics.pdf", dpi=dpi)

    # plt.clf()
    # plt.figure(figsize=(4.6, 4))
    # for i in range(4):
    #     plt.plot(freqs * 1e12, (beta1[i, :] - beta1[1, :])
    #              * 1e12, label=mode_names[i])
    # plt.xlabel(r'$f \; [\mathrm{THz}]$')
    # plt.ylabel(r'$\Delta\beta_1 \; [ps/m]$')
    # plt.legend(labelspacing=0.1)
    # plt.grid(grid)
    # plt.tight_layout()
    # plt.savefig(f"media/dispersion/DMGD_LP01.png", dpi=dpi)


if __name__ == "__main__":
    get_fig4()
