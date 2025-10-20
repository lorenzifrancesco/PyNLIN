from scripts.modules.nlin_estimator import collision_coeffs_system, total_nlin
from pynlin.utils import watt2dBm, dBm2watt
import scripts.modules.cfg as cfg
from matplotlib.ticker import ScalarFormatter
import pynlin.wdm
from matplotlib import rc
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger as lg
from scripts.modules.log_init import init_logging
init_logging()

# import plotly.graph_objects as go
# import seaborn as sns


# # SMF
# def get_nlin_prefactor_smf(cf):

#     print("Using gamma = ", gamma)
#     P_in = dBm2watt(cf.launch_power)
#     constellation_factor = 0.32 * 1.19 # 64-QAM (<|b|^4>/<|b|^2>^2 - 1)
#     nlin_prefactor = (P_in)**3 * gamma**2 * constellation_factor / (cf.baud_rate**2)
#     print("nlin prefactor", nlin_prefactor)
#     return nlin_prefactor

def plot_case_study_fits():
    pass


def plot_case_study_noise(
        use_dBm_scale=False,
        also_plot_smf=False,
        also_plot_noninteracting=True,
        name="xxx"):
    formatter = ScalarFormatter()
    formatter.set_scientific(True)
    formatter.set_powerlimits([0, 0])
    # rc('text', usetex=True)
    dpi = 300
    grid = False

    smf_file = "./input/smf.toml"
    mmf_file = "./input/mmf.toml"
    cf_smf = cfg.load_toml_to_struct(smf_file)
    cf_mmf = cfg.load_toml_to_struct(mmf_file)
    print(
        f"Loading a ITU-T standardized WDM grid \n [spacing: {cf_smf.channel_spacing * 1e-9:.3e}GHz, center: {cf_smf.center_frequency * 1e-12:.3e}THz] \n")
    assert (cf_smf.fiber_length == cf_mmf.fiber_length)
    assert (cf_smf.baud_rate == cf_mmf.baud_rate)
    assert (cf_smf.n_channels == cf_mmf.n_channels)
    if (cf_smf.n_channels == cf_mmf.n_channels * cf_mmf.n_modes):
        print("We are comparing the same total rate")
    else:
        print(
            f"Warning: the number of channels is different: {cf_smf.n_channels} vs {cf_mmf.n_channels * cf_mmf.n_modes}")
    assert (cf_smf.center_frequency == cf_mmf.center_frequency)
    wdm = pynlin.wdm.WDM(
        spacing=cf_smf.channel_spacing,
        num_channels=cf_smf.n_channels,
        center_frequency=cf_smf.center_frequency
    )
    freqs_smf = wdm.frequency_grid()
    wdm = pynlin.wdm.WDM(
        spacing=cf_mmf.channel_spacing,
        num_channels=cf_mmf.n_channels,
        center_frequency=cf_mmf.center_frequency
    )
    freqs_mmf = wdm.frequency_grid()

    ccfs = collision_coeffs_system(cf_mmf,
                                   ipulse=1,
                                   recompute=False)
    lg.debug(
        f"A few collisions (should be of order 1e-1, 1e-2): {ccfs[0, 0, :, :5]}")
    nlin_mmf = total_nlin(cf_mmf,
                          ccfs,
                          use_kappa=True,
                          use_x_mode=True,
                          )
    nlin_mmf_noninteracting = total_nlin(cf_mmf,
                                         ccfs,
                                         use_kappa=True,
                                         use_x_mode=False,
                                         )
    nlin_smf = total_nlin(cf_smf,
                          ccfs,
                          use_kappa=True,
                          use_x_mode=False,
                          )
    lg.debug(
        f"A few NLIN coeffs MMF (should be of order ... W): {nlin_mmf[0, :5]} W, {watt2dBm(nlin_mmf[0, :5])} dBm")
    lg.debug(
        f"A few NLIN coeffs MMF noninteracting (should be larger): {nlin_mmf_noninteracting[0, :5]} W, {watt2dBm(nlin_mmf_noninteracting[0, :5])} dBm")
    lg.debug(
        f"A few NLIN coeffs SMF (should be of order ... W): {nlin_smf[0, :5]} W, {watt2dBm(nlin_smf[0, :5])} dBm")
    colors = ["blue", "orange", "green", "red", "gray"]
    linestyles = ["-", "-", "-", "-", "--"]
    labels = ["LP01", "LP1", "LP11", "LP11", "SMF(LP01)"]

    if use_dBm_scale:
        ylabel = r'$P_\mathrm{NLIN} \; [\mathrm{dBm}]$'
        plot_function = plt.plot
        def y_function_mmf(x): return watt2dBm(x)
        def y_function_smf(x): return watt2dBm(x)
    else:
        ylabel = r'$\sum\limits_{B\neq A}\mathcal{N}_{AB} \quad [\mathrm{km}^2/\mathrm{ps}^{2}]$'
        plot_function = plt.semilogy
        def y_function_mmf(x): return x * 1e-30
        def y_function_smf(x): return x * 1e-30

    plt.clf()
    lw = 1.4
    plt.figure(figsize=(3.6, 3.2))
    for i in range(cf_mmf.n_modes):
        plot_function(freqs_mmf * 1e-12,
                      y_function_mmf(nlin_mmf[i, :]),
                      lw=lw,
                      color=colors[i],
                      ls=linestyles[i],
                      label=labels[i])
        if also_plot_noninteracting:
            plot_function(freqs_mmf * 1e-12,
                          y_function_mmf(nlin_mmf_noninteracting[i, :]),
                          lw=lw,
                          color=colors[i],
                          ls='-.')
    if also_plot_smf:
        plot_function(freqs_smf * 1e-12,
                      y_function_smf(nlin_smf[0, :]),
                      lw=lw,
                      color=colors[-1],
                      ls=linestyles[-1],
                      label=labels[-1])
    #
    plt.xlabel(r'$f \; [\mathrm{THz}]$')
    plt.ylabel(ylabel)
    # plt.legend(labelspacing=0.1)
    plt.grid(grid)
    plt.tight_layout()
    # plt.ylim([2e-2, 1e0])
    plt.savefig(f"media/nlin"+name+".pdf", dpi=dpi)
    print("The figure is saved as media/nlin"+name+".pdf")

    functions = [np.mean, np.median, np.max, np.min]
    function_names = ["mean  ", "median", "max   ", "min   "]
    for foo, name in zip(functions, function_names):
        avg_nlin_mmf = foo(nlin_mmf)
        avg_nlin_smf = foo(nlin_smf)
        print(
            name + f" NLIN coeff per channel: MMF -> {avg_nlin_mmf:4.3e} | SMF -> {avg_nlin_smf:4.3e}")
        # apply QAM 16 and -10 dBm input power
        print(
            name + f" NLIN power per channel: MMF -> {watt2dBm(avg_nlin_mmf):4.1f} dBm | SMF -> {watt2dBm(avg_nlin_smf):4.1f} dBm")
        print("-" * 20)


def plot_case_study_noise_histogram(use_kappa=False,
                                    use_smf=False,
                                    use_fB=False,
                                    use_dBm_scale=False):
    formatter = ScalarFormatter()
    formatter.set_scientific(True)
    formatter.set_powerlimits([0, 0])
    rc('text', usetex=True)
    dpi = 300
    grid = False

    smf_file = "./input/smf.toml"
    mmf_file = "./input/mmf.toml"
    cf_smf = cfg.load_toml_to_struct(smf_file)
    cf_mmf = cfg.load_toml_to_struct(mmf_file)
    print(
        f"Loading a ITU-T standardized WDM grid \n [spacing: {cf_smf.channel_spacing * 1e-9:.3e}GHz, center: {cf_smf.center_frequency * 1e-12:.3e}THz] \n")
    assert (cf_smf.fiber_length == cf_mmf.fiber_length)
    assert (cf_smf.baud_rate == cf_mmf.baud_rate)
    # assert (cf_smf.n_channels == cf_mmf.n_channels * cf_mmf.n_modes)
    assert (cf_smf.center_frequency == cf_mmf.center_frequency)
    wdm = pynlin.wdm.WDM(
        spacing=cf_smf.channel_spacing,
        num_channels=cf_smf.n_channels,
        center_frequency=cf_smf.center_frequency
    )
    freqs_smf = wdm.frequency_grid()
    wdm = pynlin.wdm.WDM(
        spacing=cf_mmf.channel_spacing,
        num_channels=cf_mmf.n_channels,
        center_frequency=cf_mmf.center_frequency
    )
    freqs_mmf = wdm.frequency_grid()

    nlin_mmf = collision_coeffs_system(cf_mmf,
                                       use_kappa=use_kappa,
                                       use_fB=use_fB,
                                       use_dBm_scale=use_dBm_scale,)
    nlin_smf = collision_coeffs_system(cf_smf,
                                       use_kappa=False,
                                       use_fB=True,
                                       use_dBm_scale=use_dBm_scale,)
    # for each channel, we compute the total number of collisions that
    # needs to be computed for evaluating the total noise on that channel.
    T = 1 / cf_mmf.baud_rate
    L = cf_mmf.fiber_length

    colors = ["blue", "orange", "green", "red", "gray"]
    linestyles = ["-", "-", "-", "-", "--"]
    labels = ["LP01", "LP1", "LP11", "LP11", "SMF(LP01)"]

    if use_dBm_scale:
        ylabel = r'$\mathrm{n.\;  of\; channels}$'
        plot_function = plt.hist
        def y_function_mmf(x): return watt2dBm(x)
        def y_function_smf(x): return watt2dBm(x)
    else:
        ylabel = r'$\mathrm{NLIN} \; [\mathrm{km}^2/\mathrm{ps}^{2}]$'
        plot_function = plt.hist
        def y_function_mmf(x): return x * 1e-30
        def y_function_smf(x): return x * 1e-30

    plt.clf()
    plt.figure(figsize=(3.6, 2.8))
    y_data = y_function_mmf(nlin_mmf)

    y_extremes = [np.min(y_data), -40]
    n_bins = 25
    bin_width = (y_extremes[1] - y_extremes[0]) / n_bins
    bins = np.arange(y_extremes[0], y_extremes[1] + bin_width, bin_width)
    alpha = 0.1
    alpha2 = 0.99
    lw = 5
    for i in range(cf_mmf.n_modes):
        plot_function(y_data[i, :],
                      bins=bins,
                      histtype='stepfilled',
                      alpha=alpha,
                      lw=lw,
                      color=colors[i],)
        plot_function(y_data[i, :],
                      bins=bins,
                      histtype='step',
                      lw=lw,
                      alpha=alpha2,
                      color=colors[i],)
    if use_smf:
        plot_function(y_function_smf(nlin_smf[0, :]),
                      bins=bins,
                      histtype='stepfilled',
                      alpha=alpha,
                      lw=lw,
                      color=colors[-1],)
        plot_function(y_function_smf(nlin_smf[0, :]),
                      bins=bins,
                      histtype='step',
                      lw=lw,
                      alpha=alpha2,
                      linestyle='--',
                      color=colors[-1],)
    #
    plt.xlabel(r'$P_\mathrm{NLIN} \; [\mathrm{dBm}]$')
    plt.ylabel(ylabel)
    plt.yscale('log')
    # plt.legend(labelspacing=0.1)
    plt.grid(grid)
    plt.tight_layout()
    # plt.ylim([2e-2, 1e0])
    plt.savefig(f"media/6-noise.pdf", dpi=dpi)
    print("The figure is saved as media/6-noise.pdf")


if __name__ == "__main__":
    plot_case_study_noise(
        use_dBm_scale=True,
        also_plot_smf=True,
        also_plot_noninteracting=True,  # FIXME check
        name="realistic")

    # for realistic in [True, False]:
    #     if realistic:
    #         name = "realistic"
    #     else:
    #         name = "idealized"

    exit()
    plot_case_study_noise_histogram(use_kappa=True,
                                    use_smf=True,
                                    use_fB=True,
                                    use_dBm_scale=True)