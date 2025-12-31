import numpy as np
import logging
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import pynlin.wdm
from pynlin.utils import nu2lambda
from analysis.fiber_analysis.load_fiber_values import load_group_delay, load_dummy_group_delay
from numpy import polyval
from pynlin.fiber import MMFiber
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import ScalarFormatter
import analysis.utils.cfg as cfg
from matplotlib.ticker import FuncFormatter
from loguru import logger as lg
from analysis.log_init import init_logging
init_logging()


from matplotlib.colors import ListedColormap
from matplotlib import cm
base = plt.get_cmap("jet", 256)
newcolors = base(np.linspace(0, 1, 256))
newcolors[0, :] = np.array([1, 1, 1, 1])  # set lowest color to white
whited_cm = ListedColormap(newcolors)


rc('text', usetex=True)


def beta2rms(beta2a, beta2b):
    """RMS combination of two beta2 values (negative for normal dispersion)."""
    return -np.sqrt((beta2a**2 + beta2b**2)/2)


def beta2rms_complementary(beta2rms, beta2a):
    """Return beta2b such that rms(beta2a, beta2b) equals the target rms value."""
    tmp = 2 * beta2rms**2 - beta2a**2
    assert (tmp > 0)
    return -np.sqrt(tmp)


def beta2avg(beta2a, beta2b):
    """Arithmetic mean of two beta2 values."""
    return (beta2a + beta2b) / 2


def beta2avg_complementary(beta2avg, beta2a):
    """Return beta2b such that average(beta2a, beta2b) equals the target average."""
    return beta2avg - beta2a


def fig3_fig4(cf_file="./input/mmf.toml"):
    """Recreate dispersion/NLIN plots used in figures 3 and 4."""
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


def plot_channel_dgd_distribution(cf_file = "./input/mmf.toml"):
    """Plot distribution of channel walk-offs (L/LW) across all mode pairs."""
    cf = cfg.load_toml_to_struct(cf_file)
    wdm = pynlin.wdm.WDM(
        spacing=cf.channel_spacing,
        num_channels=cf.n_channels,
        center_frequency=cf.center_frequency
    )
    oi_fit = np.load('results/oi_fit.npy')
    beta1_params = load_group_delay()
    fiber = MMFiber(
        effective_area=cf.effective_area,
        overlap_integrals=oi_fit,
        group_delay=beta1_params,
        length=cf.fiber_length
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

    ##
    
    
    
    
    assert    cf.baud_rate == 33e9, "Adjust DGD window lines for different baud rates"
    assert cf.fiber_length == 70e3, "Adjust DGD window lines for different fiber lengths"
   
    x_norm = cf.fiber_length * cf.baud_rate
    mask = (beta1_differences < 200 * 1e-1)
    lg.debug(f"Min and max DGD (all pairs):  {np.min(beta1_differences)*1e12:.2e} ps/m,     {np.max(beta1_differences)*1e12:.2e} ps/m")
    lg.debug(f"Average DGD (all pairs):  {np.mean(beta1_differences)*1e12:.2f} ps/m")
    lg.debug(f"Average L/LW :  {np.mean(beta1_differences* x_norm):.2f}")
    total_pairs = np.sum(mask) # unique pairs only
    hist, edges = np.histogram(np.log10(beta1_differences[mask]*x_norm), bins=50)
    hist = hist / 2.0

    plt.clf()
    fig, ax = plt.subplots(figsize=(3.6, 3))
    ax.bar(np.power(10, edges[:-1]),
        hist,
        width=np.diff(np.power(10, edges)) / 1.5,
        zorder=3,
        edgecolor='blue',
        facecolor='none')

    x_start = 2.89 # ideal value no raman no dispersion at 20% of the relative precision 0.6
    x_end   = 12 # ideal value no raman no dispersion at 20% of the relative precision 20.0
    # count in-window and total
    count = np.sum((beta1_differences * x_norm > x_start) & (beta1_differences * x_norm < x_end))
    total = np.sum(mask)
    total_pairs = total / 2.0  # match the /2 applied to hist
    print(f"Number of channel pairs with DGD between {x_start:.2e} and {x_end:.2e} ps/m: "
        f"{count/2:.0f} over {total_pairs:.0f} -> {(count/2)/total_pairs:.2%}")

    # labels/scales on left axis
    # L/LW = \Delta\beta_1 * L * Rb
    ax.set_xlabel(r'$L/L_W$')
    ax.set_ylabel('channel pair count')
    ax.grid(axis='y', zorder=1)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.axvline(x_start, color='red', lw=1, ls='--', zorder=4)
    ax.axvline(x_end, color='red', lw=1, ls='--')
    ax.axvspan(x_start, x_end, color='red', alpha=0.3)
    ### ANDAMENTO
    #     # --- log-log least squares fit on histogram (ignore zero-count bins) ---
    # # bin centers in log space (geometric mean of edges)
    # log_edges = np.log10(edges)
    # log_centers = 0.5 * (log_edges[:-1] + log_edges[1:])
    # centers = 10**log_centers

    # # remove zero/negative bins to avoid -inf in logs
    # valid = hist > 0
    # x_fit = log_centers[valid]           # log10(x)
    # y_fit = np.log10(hist[valid])        # log10(y)

    # if x_fit.size >= 2:
    #     # linear LS in log space: y = a*x + b
    #     a, b = np.polyfit(x_fit, y_fit, 1)

    #     # R^2 in log space
    #     y_pred = a * x_fit + b
    #     ss_res = np.sum((y_fit - y_pred)**2)
    #     ss_tot = np.sum((y_fit - np.mean(y_fit))**2)
    #     r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    #     # smooth x for plotting the fitted power law on the same axes
    #     x_plot = np.logspace(log_edges[0], log_edges[-1], 400)
    #     y_plot = 10**b * x_plot**a

    #     ax.plot(x_plot, y_plot, lw=1.5, label=f"LS fit: y ≈ C x^{a:.2f}")
    #     ax.legend(frameon=False)
    #     lg.info(f"Histogram log-log LS fit: slope a={a:.6f}, intercept b={b:.6f} (base-10 logs), R^2={r2:.4f}")
    # else:
    #     lg.warning("Not enough nonzero histogram bins to perform log-log fit.")

    
    # # --- RIGHT Y AXIS (percent of total unique pairs) ---
    # ax_right = ax.twinx()
    # ax_right.set_yscale('log')           # keep scales aligned
    # ax_right.set_ylim(ax.get_ylim())     # match limits to left axis

    # def to_percent(y, pos):
    #     if total_pairs == 0:
    #         return "0%"
    #     return f"{(y / total_pairs) * 100:.0f}%"

    # ax_right.yaxis.set_major_formatter(FuncFormatter(to_percent))
    # ax_right.set_ylabel('of channel pairs [%]')

    fig.tight_layout()
    fig.savefig("media/dgd-statistics.pdf", dpi=300)
    plt.close(fig)
    print("Saved figure 4 to media/dgd-statistics.pdf")


def plot_channel_gvd_distribution(cf_file = "./input/mmf.toml"):
    """Plot distribution of channel GVD across all modes/channels."""
    from analysis.nlin.collision import get_systems_dispersions
    cf = cfg.load_toml_to_struct(cf_file)
    wdm = pynlin.wdm.WDM(
        spacing=cf.channel_spacing,
        num_channels=cf.n_channels,
        center_frequency=cf.center_frequency
    )
    oi_fit = np.load('results/oi_fit.npy')
    beta1_params = load_group_delay()
    fiber = MMFiber(
        effective_area=cf.effective_area,
        overlap_integrals=oi_fit,
        group_delay=beta1_params,
        length= cf.fiber_length
    )

    freqs = wdm.frequency_grid()
    X, Y = get_systems_dispersions()
    max_gvd = np.max(np.abs(X))
    lg.info(f"max val X: {np.max(X):.2e}, max val Y: {np.max(Y):.2e}")
    
    
    
    plt.clf()
    # --- Scatter plot ---
    plt.figure(figsize=(3.2, 2.4))
    plt.scatter(X, Y, s=1, alpha=0.15, color='black')
    ax = plt.gca()
    ax.set_aspect('equal')
    # ax.set_xlim(0, 2)
    # ax.set_ylim(0, 2)
    ax.set_xlabel(r'$L/L_{DA}$')
    ax.set_ylabel(r'$L/L_{DB}$')
    ax.set_xticks(np.linspace(0, 2, 5))
    ax.set_yticks(np.linspace(0, 2, 5))
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("media/gvdab_scatter.pdf", dpi=300)
    print("Saved scatter plot → media/gvdab_scatter.pdf")
    plt.clf()

    # --- 2D histogram (heatmap) ---
    # --- 2D histogram (heatmap) : upper triangle only ---
    mask = Y >= X  # use Y <= X for the lower triangle

    fig, ax = plt.subplots(figsize=(2, 2))
    H, xedges, yedges, im = plt.hist2d(
        X, Y,
        bins=15, range=[[0, max_gvd], [0, max_gvd]], cmap=whited_cm
    )
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, aspect=20)
    cbar.set_label('pair count')
    cbar.set_ticks([0,10000])
    cbar.set_ticklabels(['0', r'$10^4$'])
    # plt.colorbar(im, label='pair count')

    ax = plt.gca()
    ax.set_aspect('equal')
    ax.plot([0, max_gvd], [0, max_gvd], color='gray', lw=0.5, ls='--', alpha=0.6)  # diagonal guide
    ax.set_xlabel(r'$L/L_{DA}$')
    ax.set_ylabel(r'$L/L_{DB}$')
    ax.set_xticks(np.linspace(0, 2, 2))
    ax.set_yticks(np.linspace(0, 2, 2))

    plt.tight_layout()
    plt.savefig("media/gvdab_hist2d_upper.pdf", dpi=300, bbox_inches ="tight", pad_inches=0.02 )  # or keep your original name
    print("Saved 2D histogram (upper triangle) → media/gvdab_hist2d_upper.pdf")
    plt.clf()

    
    
    T = 1 / cf.baud_rate
    beta2A = X * T**2 / cf.fiber_length
    beta2B = Y * T**2 / cf.fiber_length

    # keep only positive values
    mask = (beta2A > 0) & (beta2B > 0)
    beta2A = beta2A[mask]
    beta2B = beta2B[mask]

    # rotate to u,v coordinates (in ps²/km)
    u = (beta2A + beta2B) * 1e24
    v = (beta2B - beta2A) * 1e24

    plt.figure(figsize=(3.2, 2.4))
    plt.hist2d(u, v, bins=30,
            range=[[0, np.percentile(u, 99)], [0, np.percentile(v, 99)]],
            cmap='nipy_spectral')
    plt.colorbar(label='pair count')
    ax = plt.gca()
    ax.set_aspect('auto')
    ax.axhline(0, color='red', lw=0.8, ls='--', alpha=0.6)
    ax.set_xlabel(r'$\beta_{2A} + \beta_{2B} \; [\mathrm{ps^2/km}]$')
    ax.set_ylabel(r'$\beta_{2B} - \beta_{2A} \; [\mathrm{ps^2/km}]$')
    plt.tight_layout()
    plt.savefig("media/gvdab_uv_beta2.pdf", dpi=300)
    plt.clf()


    
    fig = plt.figure(figsize=(3.6, 2.4))
    fig.tight_layout()
    fig.savefig("media/dgd-statistics.pdf", dpi=300)
    print("Saved figure 4 to media/dgd-statistics.pdf")


if __name__ == "__main__":
    plot_channel_dgd_distribution()
    # plot_channel_gvd_distribution()
