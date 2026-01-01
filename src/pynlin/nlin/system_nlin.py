import matplotlib.pyplot as plt
import numpy as np
from loguru import logger as lg
from matplotlib import rc
from matplotlib.ticker import ScalarFormatter

import pynlin.utils.cfg as cfg
import pynlin.wdm
from pynlin.log_init import init_logging
from pynlin.nlin.nlin_estimator import collision_coeffs_system, total_nlin
from pynlin.utils import dBm2watt, watt2dBm

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
    """Placeholder for NLIN fitting visualization (not yet implemented)."""
    pass


def plot_case_study_noise(
        use_dBm_scale=False,
        also_plot_smf=False,
        also_plot_noninteracting=True,
        name="xxx"):
    """Plot NLIN per channel for MMF (and optionally SMF) case studies."""
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

    ccfs_mmf = collision_coeffs_system(cf_mmf,
                                       ipulse=1,
                                       recompute=False)
    ccfs_smf = collision_coeffs_system(cf_smf,
                                       ipulse=1,
                                       recompute=False)
    lg.debug(
        f"A few collisions (should be of order 1e-1, 1e-2): {ccfs_mmf[0, 0, :, :5]}")
    nlin_mmf = total_nlin(cf_mmf,
                          ccfs_mmf,
                          use_kappa=True,
                          use_x_mode=True,
                          )
    nlin_mmf_noninteracting = total_nlin(cf_mmf,
                                         ccfs_mmf,
                                         use_kappa=True,
                                         use_x_mode=False,
                                         )
    nlin_smf = total_nlin(cf_smf,
                          ccfs_smf,
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


def plot_case_study_noise_histogram(
        use_dBm_scale=True,
        also_plot_smf=False,
        also_plot_noninteracting=False,
        name="",
        n_bins=25,
        dBm_range=(-53, -40),
        coeff_range=None):
    """Visualize NLIN distributions across channels with optional cross-mode variants."""
    formatter = ScalarFormatter()
    formatter.set_scientific(True)
    formatter.set_powerlimits([0, 0])
    # rc('text', usetex=True)   # keep consistent with the first function (off)
    dpi = 300
    grid = False

    # --- Load configs & sanity checks (as in the first function) ---
    smf_file = "./input/smf.toml"
    mmf_file = "./input/mmf.toml"
    cf_smf = cfg.load_toml_to_struct(smf_file)
    cf_mmf = cfg.load_toml_to_struct(mmf_file)
    print(
        f"Loading a ITU-T standardized WDM grid \n "
        f"[spacing: {cf_smf.channel_spacing * 1e-9:.3e}GHz, center: {cf_smf.center_frequency * 1e-12:.3e}THz] \n"
    )
    assert (cf_smf.fiber_length == cf_mmf.fiber_length)
    assert (cf_smf.baud_rate == cf_mmf.baud_rate)
    if (cf_smf.n_channels == cf_mmf.n_channels * cf_mmf.n_modes):
        print("We are comparing the same total rate")
    else:
        print(
            f"Warning: the number of channels is different: {cf_smf.n_channels} "
            f"vs {cf_mmf.n_channels * cf_mmf.n_modes}"
        )
    assert (cf_smf.center_frequency == cf_mmf.center_frequency)

    # Build WDM grids (not strictly needed for the histogram, but kept for parity)
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

    # --- Compute collisions & NLIN exactly like the first function ---
    ccfs_mmf = collision_coeffs_system(cf_mmf, ipulse=1, recompute=False)
    ccfs_smf = collision_coeffs_system(cf_smf, ipulse=1, recompute=False)

    # Interacting (with cross-mode terms)
    nlin_mmf = total_nlin(
        cf_mmf, ccfs_mmf,
        use_kappa=True,
        use_x_mode=True,
    )
    # Non-interacting (no cross-mode terms)
    nlin_mmf_non = total_nlin(
        cf_mmf, ccfs_mmf,
        use_kappa=True,
        use_x_mode=False,
    )
    # SMF baseline (single-mode, no cross-mode)
    nlin_smf = total_nlin(
        cf_smf, ccfs_smf,
        use_kappa=True,
        use_x_mode=False,
    )

    lg.debug(
        f"A few NLIN coeffs MMF: {nlin_mmf[0, :5]} W, {watt2dBm(nlin_mmf[0, :5])} dBm")
    lg.debug(
        f"A few NLIN coeffs MMF noninteracting: {nlin_mmf_non[0, :5]} W, {watt2dBm(nlin_mmf_non[0, :5])} dBm")
    lg.debug(
        f"A few NLIN coeffs SMF: {nlin_smf[0, :5]} W, {watt2dBm(nlin_smf[0, :5])} dBm")

    # --- Plot styling aligned with the first function ---
    colors = ["blue", "orange", "green", "red", "gray"]
    linestyles = ["-", "-", "-", "-", "--"]
    labels = ["LP01", "LP1", "LP11", "LP11", "SMF(LP01)"]

    # --- X scaling & label ---
    if use_dBm_scale:
        xlabel = r'$P_\mathrm{NLIN} \; [\mathrm{dBm}]$'
        def y_transform(x): return watt2dBm(x)
        # Histogram domain selection
        x_min, x_max = dBm_range
    else:
        xlabel = r'$\sum\limits_{B\neq A}\mathcal{N}_{AB} \; [\mathrm{km}^2/\mathrm{ps}^{2}]$'
        def y_transform(x): return x * 1e-30
        # Determine coefficient range if not provided
        all_coeff = np.concatenate([
            y_transform(nlin_mmf).ravel(),
            y_transform(nlin_mmf_non).ravel(),
            y_transform(nlin_smf).ravel()
        ])
        if coeff_range is None:
            x_min, x_max = float(np.min(all_coeff)), float(np.max(all_coeff))
        else:
            x_min, x_max = coeff_range

    # Build bins
    # Use linspace to guarantee exactly n_bins and stable edges for overlays
    bins = np.linspace(x_min, x_max, n_bins + 1)

    # --- Plot (histograms per mode) ---
    plt.clf()
    plt.figure(figsize=(3.6, 2.8))

    alpha_fill = 0.12
    alpha_edge = 0.95
    lw = 2.5

    # Interacting MMF per-mode
    data_int = y_transform(nlin_mmf)  # shape: [n_modes, n_channels]
    for i in range(cf_mmf.n_modes):
        plt.hist(
            data_int[i, :], bins=bins,
            histtype='stepfilled', alpha=alpha_fill,
            lw=lw, color=colors[i],
            label=labels[i] if i == 0 else None
        )
        plt.hist(
            data_int[i, :], bins=bins,
            histtype='step', alpha=alpha_edge,
            lw=lw, color=colors[i]
        )

    # Optional: overlay non-interacting MMF
    if also_plot_noninteracting:
        data_non = y_transform(nlin_mmf_non)
        for i in range(cf_mmf.n_modes):
            plt.hist(
                data_non[i, :], bins=bins,
                histtype='step', alpha=0.9, lw=1.8,
                linestyle='-.', color=colors[i]
            )

    # Optional: overlay SMF (LP01)
    if also_plot_smf:
        smf_data = y_transform(nlin_smf[0, :])
        plt.hist(
            smf_data, bins=bins,
            histtype='stepfilled', alpha=alpha_fill,
            lw=lw, color=colors[-1]
        )
        plt.hist(
            smf_data, bins=bins,
            histtype='step', alpha=alpha_edge,
            lw=lw, color=colors[-1], linestyle='--',
            label=labels[-1]
        )

    # Axes & save
    plt.xlabel(xlabel)
    plt.ylabel(r'$\mathrm{n.\;of\;channels}$')
    plt.yscale('log')
    # plt.legend(labelspacing=0.1)  # keep off like the first, or enable if you prefer
    plt.grid(grid)
    plt.tight_layout()
    outpath = f"media/nlin_hist{name}.pdf"
    plt.savefig(outpath, dpi=dpi)
    print(f"The figure is saved as {outpath}")


if __name__ == "__main__":
    plot_case_study_noise(
        use_dBm_scale=True,
        also_plot_smf=True,
        also_plot_noninteracting=True,  # FIXME check
        name="realistic")
    exit()
    plot_case_study_noise_histogram(use_dBm_scale=True,
                                    also_plot_noninteracting=True,
                                    also_plot_smf=True)
    exit()
