import os
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger as lg
from matplotlib.ticker import ScalarFormatter
from pathlib import Path

import pynlin.wdm
from pynlin.log_init import init_logging
from pynlin.nlin.nlin_estimator_uwb import (
    collision_coeffs_system_uwb,
    total_nlin_uwb,
)
from pynlin.raman.power_profiles import simulate_power_profiles
from pynlin.raman.plot_optimization import plot_profiles
from pynlin.system import System
from pynlin.utils import watt2dBm

os.environ.setdefault("LOGURU_LEVEL", "DEBUG")
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
                           save_path: str = "results/uwb_power_profiles.npy",
                           integration_steps: int = 300,
                           recompute: bool = False):
    """Compute Raman profiles using the standard Jiang pipeline in power_profiles.py."""
    save_path = Path(save_path)
    if save_path.exists() and not recompute:
        lg.info(f"Raman profiles already exist at {save_path}; skipping recompute.")
        return
    cfg_path = system.source or Path("./input/uwb_struct.toml")
    simulate_power_profiles(
        cfg_path=cfg_path,
        output_path=save_path,
        z_points=integration_steps,
    )
    lg.info(f"Saved Raman/ISRS power profiles to {save_path}")
    return


def plot_power_profiles(system: System, profile_path: Path | str) -> None:
    """Load a saved profile file and plot it with plot_profiles."""
    p_path = Path(profile_path)
    if not p_path.exists():
        lg.warning(f"Profile file {p_path} not found; skipping plot.")
        return
    lg.info(f"Loading profile data for plotting from {p_path}")
    data = np.load(p_path, allow_pickle=True).item()

    sig_wl = data.get("signal_wavelengths")
    sig_sol = data.get("signal_sol")
    if sig_wl is None or sig_sol is None:
        lg.warning("Profile file missing signal_wavelengths or signal_sol; skipping plot.")
        return

    signal_solution = np.array(sig_sol, dtype=float)
    if signal_solution.ndim == 2:
        signal_solution = signal_solution[:, :, None]

    pump_solution = data.get("pump_sol")
    pump_wavelengths = data.get("pump_wavelengths")
    pump_powers = data.get("pump_powers")

    if pump_solution is None:
        pump_solution = np.zeros((signal_solution.shape[0], 0, system.n_modes))
    else:
        pump_solution = np.array(pump_solution, dtype=float)
        if pump_solution.ndim == 2:
            pump_solution = pump_solution[:, :, None]

    if pump_wavelengths is None:
        pump_wavelengths = np.array([])
    if pump_powers is None:
        pump_powers = np.zeros((len(pump_wavelengths), 1))
    else:
        pump_powers = np.array(pump_powers, dtype=float)
        if pump_powers.ndim == 1:
            pump_powers = pump_powers[:, None]

    plot_profiles(
        signal_wavelengths=sig_wl,
        signal_solution=signal_solution,
        ase_solution=None,
        pump_wavelengths=pump_wavelengths,
        pump_solution=pump_solution,
        pump_powers=pump_powers,
        cf=system,
        wallpaper_mode=False,
        use_active_naming=False,
    )
    lg.info("Saved profile plot via plot_profiles.")


def _load_profile_launch_powers(profile_path: Path | str, expected_channels: int) -> np.ndarray | None:
    """Load per-channel launch powers (W) from a saved Raman profile file."""
    p_path = Path(profile_path)
    if not p_path.exists():
        lg.warning(f"Profile file {p_path} not found; using uniform launch power.")
        return None
    try:
        lg.info(f"Loading profile data from {p_path}")
        payload = np.load(p_path, allow_pickle=True)
        if isinstance(payload, np.lib.npyio.NpzFile):
            data = payload
        else:
            data = payload.item()
    except Exception as exc:
        lg.warning(f"Failed loading profile {p_path}: {exc}")
        return None

    sig_sol = data.get("signal_sol")
    z_grid = data.get("z")
    if sig_sol is None or z_grid is None:
        lg.warning("Profile file missing signal_sol or z; using uniform launch power.")
        return None

    sig_power = np.array(sig_sol, dtype=float)
    if sig_power.ndim == 3 and sig_power.shape[-1] == 1:
        sig_power = sig_power[..., 0]
    if sig_power.ndim == 3:
        sig_power = np.sum(sig_power, axis=-1)
    if sig_power.ndim != 2 or sig_power.shape[0] != z_grid.size:
        lg.warning(f"Unexpected signal_sol shape {sig_power.shape}; using uniform launch power.")
        return None

    span = float(z_grid[-1] - z_grid[0])
    launch_override = np.trapezoid(sig_power, z_grid, axis=0) / span
    if launch_override.size != expected_channels:
        lg.warning(f"Profile channels ({launch_override.size}) != expected ({expected_channels}); ignoring override.")
        return None
    lg.info("Using per-channel launch powers from Raman profile for NLIN.")
    return launch_override


def _nlin_cache_path(profile_path: Path | str | None,
                     use_kappa: bool,
                     use_x_mode: bool) -> Path:
    """Return a cache path for total NLIN arrays."""
    tag = Path(profile_path).stem if profile_path is not None else "default"
    return Path("results") / f"total_nlin_{tag}_k{int(use_kappa)}_x{int(use_x_mode)}.npy"


def plot_case_study_noise(
        use_dBm_scale=False,
        also_plot_noninteracting=True,
        name="xxx",
        profile_path: Path | str = Path("results/uwb_power_profiles.npy"),
        use_profile: bool = True):
    """Plot NLIN per channel for MMF (and optionally SMF) case studies."""
    formatter = ScalarFormatter()
    formatter.set_scientific(True)
    formatter.set_powerlimits([0, 0])
    # rc('text', usetex=True)
    dpi = 300
    grid = False

    cfg_path = Path("./input/uwb_struct.toml")
    syst = System.from_toml(cfg_path)
    freqs = syst.wdm.frequency_grid()

    launch_override = _load_profile_launch_powers(profile_path, syst.n_channels) if use_profile else None

    ccfs = collision_coeffs_system_uwb(syst, # FIXME this is the thing sucking the most time
                                       ipulse=1,
                                       recompute=True,
                                       profile_path=profile_path)
    nlin_uwb = total_nlin_uwb(
        syst,
        ccfs,
        use_kappa=True,
        use_x_mode=True,
        launch_powers_w=launch_override,
        cache_path=_nlin_cache_path(profile_path, use_kappa=True, use_x_mode=True),
    )
    nlin_uwb_noninteracting = total_nlin_uwb(
        syst,
        ccfs,
        use_kappa=True,
        use_x_mode=False,
        launch_powers_w=launch_override,
        cache_path=_nlin_cache_path(profile_path, use_kappa=True, use_x_mode=False),
    )
    lg.debug(
        f"A few NLIN coeffs MMF (should be of order ... W): {nlin_uwb[0, :5]} W, {watt2dBm(nlin_uwb[0, :5])} dBm")
    lg.debug(
        f"A few NLIN coeffs MMF noninteracting (should be larger): {nlin_uwb_noninteracting[0, :5]} W, {watt2dBm(nlin_uwb_noninteracting[0, :5])} dBm")
    colors = ["blue", "orange", "green", "red", "gray"]
    linestyles = ["-", "-", "-", "-", "--"]
    labels = ["LP01", "LP1", "LP11", "LP11", "SMF(LP01)"]

    if use_dBm_scale:
        ylabel = r'$P_\mathrm{NLIN} \; [\mathrm{dBm}]$'
        plot_function = plt.plot
        def y_function_uwb(x): return watt2dBm(x)
    else:
        ylabel = r'$\sum\limits_{B\neq A}\mathcal{N}_{AB} \quad [\mathrm{km}^2/\mathrm{ps}^{2}]$'
        plot_function = plt.semilogy
        def y_function_uwb(x): return x * 1e-30

    plt.clf()
    lw = 1.4
    plt.figure(figsize=(3.6, 3.2))
    for i in range(syst.n_modes):
        plot_function(freqs * 1e-12,
                      y_function_uwb(nlin_uwb[i, :]),
                      lw=lw,
                      color=colors[i],
                      ls=linestyles[i],
                      label=labels[i])
        if also_plot_noninteracting:
            plot_function(freqs * 1e-12,
                          y_function_uwb(nlin_uwb_noninteracting[i, :]),
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
        avg_nlin_mmf = foo(nlin_uwb)
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
        coeff_range=None,
        profile_path: Path | str | None = None):
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
    ccfs_uwb = collision_coeffs_system_uwb(
        cf,
        ipulse=1,
        recompute=False,
        profile_path=profile_path,
    )

    # Interacting (with cross-mode terms)
    nlin_uwb = total_nlin_uwb(
        cf, ccfs_uwb,
        use_kappa=True,
        use_x_mode=True,
        cache_path=_nlin_cache_path(profile_path, use_kappa=True, use_x_mode=True),
    )
    # Non-interacting (no cross-mode terms)
    nlin_uwb_non = total_nlin_uwb(
        cf, ccfs_uwb,
        use_kappa=True,
        use_x_mode=False,
        cache_path=_nlin_cache_path(profile_path, use_kappa=True, use_x_mode=False),
    )

    lg.debug(
        f"A few NLIN coeffs MMF: {nlin_uwb[0, :5]} W, {watt2dBm(nlin_uwb[0, :5])} dBm")
    lg.debug(
        f"A few NLIN coeffs MMF noninteracting: {nlin_uwb_non[0, :5]} W, {watt2dBm(nlin_uwb_non[0, :5])} dBm")

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
            y_transform(nlin_uwb).ravel(),
            y_transform(nlin_uwb_non).ravel(),
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
    data_int = y_transform(nlin_uwb)  # shape: [n_modes, n_channels]
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
        data_non = y_transform(nlin_uwb_non)
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
    print(system.summary())
    # exit()
    try:
        system.report_max_l_normalizations()
    except Exception as exc:
        lg.warning(f"Could not report L/LD or L/LW: {exc}")
    # compute_raman_profiles(system, save_path="results/uwb_power_profiles.npy") # FIXME this is not saving a menaingful file
    plot_power_profiles(system, "results/uwb_power_profiles.npy")
    profile_path = Path("results/uwb_power_profiles.npy")
    # exit()
    plot_case_study_noise(
        use_dBm_scale=True,
        also_plot_noninteracting=True,
        name="uwb_struct",
        profile_path=profile_path,
        use_profile=True)
    plot_case_study_noise_histogram(use_dBm_scale=True,
                                    also_plot_noninteracting=True,
                                    name="uwb_struct",
                                    profile_path=profile_path)
    exit()
