import matplotlib.pyplot as plt
import numpy as np
from loguru import logger as lg
from matplotlib.ticker import ScalarFormatter
from pathlib import Path

import pynlin.wdm
from pynlin.log_init import init_logging
from pynlin.nlin.nlin_estimator import collision_coeffs_system, total_nlin
from pynlin.raman.solvers import MMFRamanAmplifier, SMWidebandRamanAmplifier
from pynlin.system import System
from pynlin.utils import dBm2watt, nu2lambda, watt2dBm
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


def compute_raman_profiles(system: System,
                           save_path: str = "results/uwb_profiles.npz",
                           integration_steps: int = 300):
    """Compute Raman/ISRS power evolution using the NumPy amplifier and persist results."""
    wdm = system.wdm
    fiber = system.fiber
    pumps = system.pump_specs or []
    num_pumps = len(pumps)
    n_modes = getattr(fiber, "n_modes", 1)

    freqs = wdm.frequency_grid()
    wavelengths = nu2lambda(freqs)

    def _initial_signal_powers():
        launch_dbm = system.launch_power if system.launch_power is not None else -5.0
        powers_dbm = np.full(len(wavelengths), launch_dbm)
        if hasattr(wdm, "band_specs") and hasattr(wdm, "_band_slices"):
            for name, slc in wdm._band_slices.items():
                spec = wdm.band_specs.get(name)
                if spec and spec.launch_power_dbm is not None:
                    powers_dbm[slc] = spec.launch_power_dbm
        return dBm2watt(powers_dbm)

    signal_power_w = np.tile(_initial_signal_powers()[:, None], (1, n_modes))

    # pump arrays
    if num_pumps:
        pump_wavelengths = np.array([p.wavelength for p in pumps])
        pump_power_dbm = np.array([p.power_dbm for p in pumps])
        pump_power_w = np.tile(dBm2watt(pump_power_dbm)[:, None], (1, n_modes))
        pump_dirs = np.array([p.direction for p in pumps])
    else:
        pump_wavelengths = np.array([])
        pump_power_w = np.zeros((0, n_modes))
        pump_dirs = np.array([])

    # z-grid
    z = np.linspace(0, fiber.length, integration_steps)

    if n_modes == 1:
        # Single-mode Raman amplifier with per-band launch powers
        amp = SMWidebandRamanAmplifier(fiber)
        pump_solution, signal_solution, _ = amp.solve_from_system(
            system,
            z=z,
            solver="ivp",
            odeint_kwargs={"rtol": 1e-2, "atol": 1e-4, "mxstep": 500},
            ase=False,
        )
        pump_solution = pump_solution[:, :, None] if num_pumps else np.zeros((len(z), 0, 1))
        signal_solution = signal_solution[:, :, None]
    else:
        if num_pumps:
            amp = MMFRamanAmplifier()
            direction = np.concatenate(
                [np.repeat(pump_dirs, n_modes), np.ones(len(wavelengths) * n_modes)]
            )
            pump_solution, signal_solution, _ = amp.solve(
                signal_power=signal_power_w,
                signal_wavelength=wavelengths,
                pump_power=pump_power_w,
                pump_wavelength=pump_wavelengths,
                z=z,
                fiber=fiber,
                counterpumping=np.any(pump_dirs < 0),
                ase=False,
                direction=direction,
            )
        else:
            # simple loss-only propagation
            losses = fiber.losses if hasattr(fiber, "losses") else np.array([0, 0, 0])
            pump_solution = np.zeros((len(z), 0, n_modes))
            signal_solution = np.empty((len(z), len(wavelengths), n_modes))
            for i, wl in enumerate(wavelengths):
                alpha_db_per_m = np.polyval(losses, wl)
                alpha_linear = alpha_db_per_m * np.log(10) / 10  # dB/m -> Np/m
                signal_solution[:, i, :] = signal_power_w[i] * np.exp(-alpha_linear * z)[:, None]

    # Band-level power summary (start/end)
    band_summaries = []
    if hasattr(wdm, "band_specs") and hasattr(wdm, "_band_slices"):
        for name, slc in wdm._band_slices.items():
            start_power = watt2dBm(np.nanmean(signal_solution[0, slc, :]))
            end_power = watt2dBm(np.nanmean(signal_solution[-1, slc, :]))
            band_summaries.append((name, start_power, end_power))
            lg.info(f"Band {name}: avg start power {start_power:.2f} dBm, end power {end_power:.2f} dBm")
    else:
        start_power = watt2dBm(np.nanmean(signal_solution[0]))
        end_power = watt2dBm(np.nanmean(signal_solution[-1]))
        band_summaries.append(("full", start_power, end_power))
        lg.info(f"Full band: avg start power {start_power:.2f} dBm, end power {end_power:.2f} dBm")

    if np.isnan(signal_solution).any() or np.isnan(pump_solution).any():
        lg.warning("NaNs detected in Raman profiles; check solver stability and inputs.")

    np.savez(
        save_path,
        z=z,
        wavelengths=wavelengths,
        signal_solution=signal_solution,
        pump_solution=pump_solution,
        pump_wavelengths=pump_wavelengths,
        pump_dirs=pump_dirs,
        band_summaries=band_summaries,
    )
    lg.info(f"Saved Raman/ISRS power profiles to {save_path}")
    return z, wavelengths, signal_solution, pump_solution


def plot_case_study_noise(
        use_dBm_scale=False,
        also_plot_noninteracting=True,
        name="xxx"):
    """Plot NLIN per channel for MMF (and optionally SMF) case studies."""
    formatter = ScalarFormatter()
    formatter.set_scientific(True)
    formatter.set_powerlimits([0, 0])
    # rc('text', usetex=True)
    dpi = 300
    grid = False

    cfg_path = Path("./input/uwb_struct.toml")
    cf = System.from_toml(cfg_path)
    print(
        f"Loading a WDM grid \n [spacing: {cf.channel_spacing * 1e-9:.3e}GHz, center: {cf.center_frequency * 1e-12:.3e}THz] \n")
    freqs = cf.wdm.frequency_grid()

    ccfs = collision_coeffs_system(cf,
                                   ipulse=1,
                                   recompute=False)
    lg.debug(
        f"A few collisions (should be of order 1e-1, 1e-2): {ccfs[0, 0, :, :5]}")
    nlin_mmf = total_nlin(cf,
                          ccfs,
                          use_kappa=True,
                          use_x_mode=True,
                          )
    nlin_mmf_noninteracting = total_nlin(cf,
                                         ccfs,
                                         use_kappa=True,
                                         use_x_mode=False,
                                         )
    lg.debug(
        f"A few NLIN coeffs MMF (should be of order ... W): {nlin_mmf[0, :5]} W, {watt2dBm(nlin_mmf[0, :5])} dBm")
    lg.debug(
        f"A few NLIN coeffs MMF noninteracting (should be larger): {nlin_mmf_noninteracting[0, :5]} W, {watt2dBm(nlin_mmf_noninteracting[0, :5])} dBm")
    colors = ["blue", "orange", "green", "red", "gray"]
    linestyles = ["-", "-", "-", "-", "--"]
    labels = ["LP01", "LP1", "LP11", "LP11", "SMF(LP01)"]

    if use_dBm_scale:
        ylabel = r'$P_\mathrm{NLIN} \; [\mathrm{dBm}]$'
        plot_function = plt.plot
        def y_function_mmf(x): return watt2dBm(x)
    else:
        ylabel = r'$\sum\limits_{B\neq A}\mathcal{N}_{AB} \quad [\mathrm{km}^2/\mathrm{ps}^{2}]$'
        plot_function = plt.semilogy
        def y_function_mmf(x): return x * 1e-30

    plt.clf()
    lw = 1.4
    plt.figure(figsize=(3.6, 3.2))
    for i in range(cf.n_modes):
        plot_function(freqs * 1e-12,
                      y_function_mmf(nlin_mmf[i, :]),
                      lw=lw,
                      color=colors[i],
                      ls=linestyles[i],
                      label=labels[i])
        if also_plot_noninteracting:
            plot_function(freqs * 1e-12,
                          y_function_mmf(nlin_mmf_noninteracting[i, :]),
                          lw=lw,
                          color=colors[i],
                          ls='-.')
    #
    plt.xlabel(r'$f \; [\mathrm{THz}]$')
    plt.ylabel(ylabel)
    # plt.legend(labelspacing=0.1)
    plt.grid(grid)
    plt.tight_layout()
    # plt.ylim([2e-2, 1e0])
    plt.savefig(f"media/nlin"+name+".pdf", dpi=dpi)
    lg.info("The figure is saved as media/nlin"+name+".pdf")

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

    cfg_path = Path("./input/uwb_struct.toml")
    cf = System.from_toml(cfg_path)
    print(
        f"Loading a WDM grid \n "
        f"[spacing: {cf.channel_spacing * 1e-9:.3e}GHz, center: {cf.center_frequency * 1e-12:.3e}THz] \n"
    )
    freqs = cf.wdm.frequency_grid()

    # --- Compute collisions & NLIN exactly like the first function ---
    ccfs_mmf = collision_coeffs_system(cf, ipulse=1, recompute=False)

    # Interacting (with cross-mode terms)
    nlin_mmf = total_nlin(
        cf, ccfs_mmf,
        use_kappa=True,
        use_x_mode=True,
    )
    # Non-interacting (no cross-mode terms)
    nlin_mmf_non = total_nlin(
        cf, ccfs_mmf,
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
    for i in range(cf.n_modes):
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
        for i in range(cf.n_modes):
            plt.hist(
                data_non[i, :], bins=bins,
                histtype='step', alpha=0.9, lw=1.8,
                linestyle='-.', color=colors[i]
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
    # First, compute and save Raman/ISRS power profiles using provided pumps
    cfg_path = Path("./input/uwb_struct.toml")
    system = System.from_toml(cfg_path)
    compute_raman_profiles(system, save_path="results/uwb_profiles.npz")

    plot_case_study_noise(
        use_dBm_scale=True,
        also_plot_noninteracting=True,
        name="uwb_uwb_struct")
    plot_case_study_noise_histogram(use_dBm_scale=True,
                                    also_plot_noninteracting=True,
                                    name="uwb_uwb_struct")
    exit()
