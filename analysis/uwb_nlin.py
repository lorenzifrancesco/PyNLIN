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
from pynlin.utils import dBm2watt, watt2dBm
from scipy.constants import c

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
                           recompute: bool = False,
                           jiang_cfg=None,
                           max_power_w: float | None = None):
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
        jiang_cfg=jiang_cfg,
    )
    if max_power_w is not None:
        payload = np.load(save_path, allow_pickle=True).item()
        sig = payload.get("signal_sol")
        if sig is None:
            raise ValueError(f"Raman profile {save_path} missing signal_sol.")
        sig = np.asarray(sig, dtype=float)
        if not np.all(np.isfinite(sig)):
            raise ValueError(f"Raman profile {save_path} contains non-finite values.")
        max_val = float(np.nanmax(sig))
        if max_val > float(max_power_w):
            raise ValueError(
                f"Raman profile {save_path} max power {max_val:.3e} W exceeds {max_power_w:.3e} W."
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
    """Load per-channel launch powers (W) from a saved Raman profile file (z≈0)."""
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

    # Use launch powers at fiber input (closest to z=0), not span-averaged power.
    idx0 = int(np.argmin(np.abs(z_grid)))
    launch_override = sig_power[idx0]
    if launch_override.size != expected_channels:
        lg.warning(f"Profile channels ({launch_override.size}) != expected ({expected_channels}); ignoring override.")
        return None
    lg.info("Using per-channel launch powers from Raman profile for NLIN (z≈0).")
    return launch_override


def _load_profile_output_powers(profile_path: Path | str, expected_channels: int) -> np.ndarray | None:
    """Load per-channel output powers (W) from a saved Raman profile file."""
    p_path = Path(profile_path)
    if not p_path.exists():
        lg.warning(f"Profile file {p_path} not found; skipping output power plot.")
        return None
    try:
        lg.info(f"Loading profile data for output power from {p_path}")
        payload = np.load(p_path, allow_pickle=True)
        if isinstance(payload, np.lib.npyio.NpzFile):
            data = payload
        else:
            data = payload.item()
    except Exception as exc:
        lg.warning(f"Failed loading profile {p_path}: {exc}")
        return None

    sig_sol = data.get("signal_sol")
    if sig_sol is None:
        lg.warning("Profile file missing signal_sol; skipping output power plot.")
        return None

    sig_power = np.array(sig_sol, dtype=float)
    if sig_power.ndim == 3 and sig_power.shape[-1] == 1:
        sig_power = sig_power[..., 0]
    if sig_power.ndim == 3:
        sig_power = np.sum(sig_power, axis=-1)
    if sig_power.ndim != 2:
        lg.warning(f"Unexpected signal_sol shape {sig_power.shape}; skipping output power plot.")
        return None

    output_power = sig_power[-1]
    if output_power.size != expected_channels:
        lg.warning(
            f"Profile channels ({output_power.size}) != expected ({expected_channels}); skipping output power plot."
        )
        return None
    return output_power


def _load_ase_band_lines(profile_path: Path | str, system: System):
    """Return ASE band averages as (name, f_min, f_max, avg_w) tuples."""
    ase_path = Path("results/uwb_ase_power_profiles.npy")
    if not ase_path.exists():
        raise FileNotFoundError(f"ASE profile not found at {ase_path}")
    payload = np.load(ase_path, allow_pickle=True)
    if isinstance(payload, np.lib.npyio.NpzFile):
        data = payload
    elif isinstance(payload, np.ndarray) and payload.shape == ():
        data = payload.item()
    elif isinstance(payload, dict):
        data = payload
    else:
        raise TypeError(f"Unexpected ASE payload type {type(payload)}")

    def _describe_value(value):
        if isinstance(value, np.ndarray):
            array_str = np.array2string(value, threshold=np.inf)
            return f"ndarray shape={value.shape} dtype={value.dtype} values={array_str}"
        return repr(value)

    lg.info(f"Using ASE profile data from {ase_path}")
    # if hasattr(data, "keys"):
    #     for key in sorted(data.keys()):
    #         try:
    #             value = data.get(key)
    #         except Exception:
    #             value = None
    # lg.info(f"[ASE profile] {key}: {_describe_value(value)}")

    ase = data.get("ase_sol_fixed") if hasattr(data, "get") else None
    if ase is None and hasattr(data, "get"):
        ase = data.get("ase_sol")
    if ase is None and hasattr(data, "get"):
        ase = data.get("ase_solution")
    if ase is None:
        lg.warning("ASE profile missing ASE solution; skipping ASE overlay.")
        return []

    ase = np.asarray(ase, dtype=float)
    if ase.ndim == 3 and ase.shape[-1] == 1:
        ase = ase[..., 0]
    if ase.ndim == 3:
        ase = np.sum(ase, axis=-1)
    if ase.ndim != 2:
        lg.warning(f"Unexpected ASE shape {ase.shape}; skipping ASE overlay.")
        return []

    ase_wavelengths = None
    if hasattr(data, "get"):
        for key in ("ase_signal_wavelengths", "ase_wavelengths", "signal_wavelengths", "signal_wavelength"):
            if key in data:
                ase_wavelengths = np.asarray(data.get(key), dtype=float)
                break
    if ase_wavelengths is not None and ase_wavelengths.size != ase.shape[1]:
        ase_wavelengths = None

    ase_decimation = 1
    if hasattr(data, "get"):
        ase_decimation = int(data.get("ase_decimation", 1))
    scale = ase_decimation if ase_decimation > 0 else 1
    if scale == 1 and system.n_channels and ase.shape[1] and system.n_channels % ase.shape[1] == 0:
        inferred = system.n_channels // ase.shape[1]
        if inferred > 1:
            scale = inferred
            lg.info(f"Inferred ASE decimation scale {scale} from channel counts.")
    decimated_indices = data.get("ase_signal_indices") if hasattr(data, "get") else None
    if decimated_indices is None:
        decimated_indices = np.arange(0, system.n_channels, ase_decimation, dtype=int)
    else:
        decimated_indices = np.asarray(decimated_indices, dtype=int)

    if ase.shape[1] != decimated_indices.size:
        if ase.shape[1] == system.n_channels:
            decimated_indices = np.arange(0, ase.shape[1], ase_decimation, dtype=int)
            ase = ase[:, decimated_indices]
            if ase_wavelengths is not None and ase_wavelengths.size == system.n_channels:
                ase_wavelengths = ase_wavelengths[decimated_indices]
        elif ase_wavelengths is None:
            lg.warning(
                f"ASE channels ({ase.shape[1]}) do not match decimation "
                f"indices ({decimated_indices.size}); skipping ASE overlay."
            )
            return []

    ase_out = ase[-1]
    freqs = system.wdm.frequency_grid()
    lines = []
    if isinstance(system.wdm, pynlin.wdm.IrregularWDM):
        for name, slc in system.wdm._band_slices.items():
            f_band = freqs[slc]
            f_min, f_max = float(np.min(f_band)), float(np.max(f_band))
            if ase_wavelengths is not None:
                ase_freqs = c / ase_wavelengths
                mask = (ase_freqs >= f_min) & (ase_freqs <= f_max)
            else:
                mask = (decimated_indices >= slc.start) & (decimated_indices < slc.stop)
            if not np.any(mask):
                continue
            avg_w = float(np.mean(ase_out[mask]) / scale)
            lines.append((name, f_min, f_max, avg_w))
    else:
        avg_w = float(np.mean(ase_out) / scale)
        lines.append(("all", float(np.min(freqs)), float(np.max(freqs)), avg_w))
    return lines


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
        use_profile: bool = True,
        combined_height: float = 4.5):
    """Plot output power, combined noise, and SNR per channel in one figure."""
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
    output_power = _load_profile_output_powers(profile_path, syst.n_channels) if use_profile else None

    ccfs = collision_coeffs_system_uwb(
        syst,
        ipulse=1,
        recompute=False,
        profile_path=profile_path,
    )
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

    lw = 1.4
    scatter_size = 8
    scatter_lw = 0.05

    ase_lines = _load_ase_band_lines(profile_path, syst) if use_profile else []
    ase_per_channel = np.zeros_like(freqs, dtype=float)
    assigned = np.zeros_like(freqs, dtype=bool)
    if ase_lines:
        for _, f_min, f_max, avg_w in ase_lines:
            mask = (freqs >= f_min) & (freqs <= f_max)
            ase_per_channel[mask] = avg_w
            assigned[mask] = True
        if not np.all(assigned):
            lg.warning("Some channels missing ASE band assignment; assuming 0 ASE there.")
    else:
        lg.warning("ASE data unavailable; using NLIN-only noise/SNR.")

    if output_power is not None:
        signal_power = output_power
    elif launch_override is not None:
        signal_power = launch_override
    else:
        launch_dbm = syst.launch_power if syst.launch_power is not None else -5.0
        signal_power = np.full_like(freqs, dBm2watt(launch_dbm), dtype=float)

    denom_floor = 1e-18
    fig, axes = plt.subplots(
        3,
        1,
        sharex=True,
        figsize=(3.6, float(combined_height)),
        gridspec_kw={"hspace": 0.06},
    )

    def _nudge_offset(ax):
        offset = ax.yaxis.get_offset_text()
        offset.set_x(-0.2)
        offset.set_y(2)
        offset.set_horizontalalignment("left")
        offset.set_verticalalignment("top")

    # Output power subplot
    ax_power = axes[0]
    if output_power is None:
        ax_power.text(
            0.02,
            0.5,
            "No output power data",
            transform=ax_power.transAxes,
            fontsize=8,
            va="center",
        )
    else:
        output_dbm = watt2dBm(np.maximum(output_power, 1e-18))
        ax_power.scatter(
            freqs * 1e-12,
            output_dbm,
            s=scatter_size,
            facecolors="none",
            edgecolors="black",
            alpha=0.8,
            linewidths=scatter_lw,
        )
    ax_power.set_ylabel(r'$P_\mathrm{out}\;[\mathrm{dBm}]$')
    ax_power.grid(grid)
    _nudge_offset(ax_power)

    # Noise subplot (NLIN + ASE)
    ax_noise = axes[1]
    for i in range(syst.n_modes):
        noise = np.maximum(nlin_uwb[i, :] + ase_per_channel, denom_floor)
        noise_dbm = watt2dBm(noise)
        ax_noise.scatter(
            freqs * 1e-12,
            noise_dbm,
            s=scatter_size,
            facecolors="none",
            edgecolors=colors[i],
            alpha=0.8,
            linewidths=scatter_lw,
            label="NLIN" if i == 0 else None,
        )
        if also_plot_noninteracting:
            noise_non = np.maximum(nlin_uwb_noninteracting[i, :] + ase_per_channel, denom_floor)
            noise_non_dbm = watt2dBm(noise_non)
            ax_noise.scatter(
                freqs * 1e-12,
                noise_non_dbm,
                s=scatter_size,
                facecolors="none",
                edgecolors=colors[i],
                alpha=0.5,
                marker="o",
                linewidths=scatter_lw,
            )
    if ase_lines:
        label_used = False
        for band, f_min, f_max, avg_w in ase_lines:
            avg_dbm = float(watt2dBm(max(avg_w, denom_floor)))
            label = "ASE avg (band)" if not label_used else None
            ax_noise.hlines(
                avg_dbm,
                f_min * 1e-12,
                f_max * 1e-12,
                colors="red",
                linestyles="--",
                linewidth=1.0,
                label=label,
            )
            label_used = True
    ax_noise.set_ylabel(r'$P_\mathrm{noise}\;[\mathrm{dBm}]$')
    ax_noise.set_ylim(bottom=-90)
    ax_noise.grid(grid)
    ax_noise.legend(loc="best", fontsize=8, frameon=False)
    _nudge_offset(ax_noise)

    # SNR subplot
    ax_snr = axes[2]
    for i in range(syst.n_modes):
        denom = np.maximum(nlin_uwb[i, :] + ase_per_channel, denom_floor)
        snr_db = 10 * np.log10(np.maximum(signal_power, denom_floor) / denom)
        ax_snr.scatter(
            freqs * 1e-12,
            snr_db,
            s=scatter_size,
            facecolors="none",
            edgecolors="green",
            alpha=0.8,
            linewidths=scatter_lw,
            label=labels[i],
        )
    ax_snr.set_xlabel(r'$\mathnormal{f} \; [\mathrm{THz}]$')
    ax_snr.set_ylabel(r'$\mathrm{SNR}\;[\mathrm{dB}]$')
    ax_snr.grid(grid)
    _nudge_offset(ax_snr)

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.06, left=0.18)
    out_path = f"media/power_noise_snr{name}.pdf"
    fig.savefig(out_path, dpi=dpi)
    lg.info(f"The figure is saved as {out_path}")

    # --- Individual plots (power / noise / SNR) ---
    fig_power, ax_power = plt.subplots(figsize=(3.6, 3.2))
    if output_power is None:
        ax_power.text(
            0.02,
            0.5,
            "No output power data",
            transform=ax_power.transAxes,
            fontsize=8,
            va="center",
        )
    else:
        output_dbm = watt2dBm(np.maximum(output_power, 1e-18))
        ax_power.scatter(
            freqs * 1e-12,
            output_dbm,
            s=scatter_size,
            facecolors="none",
            edgecolors="black",
            alpha=0.8,
            linewidths=scatter_lw,
        )
    ax_power.set_xlabel(r'$f \; [\mathrm{THz}]$')
    ax_power.set_ylabel(r'$P_\mathrm{out}\;[\mathrm{dBm}]$')
    ax_power.grid(grid)
    fig_power.tight_layout()
    out_path = f"media/output_power{name}.pdf"
    fig_power.savefig(out_path, dpi=dpi)
    lg.info(f"The figure is saved as {out_path}")

    fig_noise, ax_noise = plt.subplots(figsize=(3.6, 3.2))
    for i in range(syst.n_modes):
        noise = np.maximum(nlin_uwb[i, :] + ase_per_channel, denom_floor)
        noise_dbm = watt2dBm(noise)
        ax_noise.scatter(
            freqs * 1e-12,
            noise_dbm,
            s=scatter_size,
            facecolors="none",
            edgecolors=colors[i],
            alpha=0.8,
            linewidths=scatter_lw,
            label="NLIN" if i == 0 else None,
        )
        if also_plot_noninteracting:
            noise_non = np.maximum(nlin_uwb_noninteracting[i, :] + ase_per_channel, denom_floor)
            noise_non_dbm = watt2dBm(noise_non)
            ax_noise.scatter(
                freqs * 1e-12,
                noise_non_dbm,
                s=scatter_size,
                facecolors="none",
                edgecolors=colors[i],
                alpha=0.5,
                marker="o",
                linewidths=scatter_lw,
            )
    if ase_lines:
        label_used = False
        for band, f_min, f_max, avg_w in ase_lines:
            avg_dbm = float(watt2dBm(max(avg_w, denom_floor)))
            label = "ASE avg (band)" if not label_used else None
            ax_noise.hlines(
                avg_dbm,
                f_min * 1e-12,
                f_max * 1e-12,
                colors="red",
                linestyles="--",
                linewidth=1.0,
                label=label,
            )
            label_used = True
    ax_noise.set_xlabel(r'$f \; [\mathrm{THz}]$')
    ax_noise.set_ylabel(r'$P_\mathrm{noise}\;[\mathrm{dBm}]$')
    ax_noise.set_ylim(bottom=-90)
    ax_noise.grid(grid)
    ax_noise.legend(loc="best", fontsize=8, frameon=False)
    fig_noise.tight_layout()
    out_path = f"media/noise{name}.pdf"
    fig_noise.savefig(out_path, dpi=dpi)
    lg.info(f"The figure is saved as {out_path}")

    fig_snr, ax_snr = plt.subplots(figsize=(3.6, 3.2))
    for i in range(syst.n_modes):
        denom = np.maximum(nlin_uwb[i, :] + ase_per_channel, denom_floor)
        snr_db = 10 * np.log10(np.maximum(signal_power, denom_floor) / denom)
        ax_snr.scatter(
            freqs * 1e-12,
            snr_db,
            s=scatter_size,
            facecolors="none",
            edgecolors="green",
            alpha=0.8,
            linewidths=scatter_lw,
            label=labels[i],
        )
    ax_snr.set_xlabel(r'$f \; [\mathrm{THz}]$')
    ax_snr.set_ylabel(r'$\mathrm{SNR}\;[\mathrm{dB}]$')
    ax_snr.grid(grid)
    fig_snr.tight_layout()
    out_path = f"media/snr{name}.pdf"
    fig_snr.savefig(out_path, dpi=dpi)
    lg.info(f"The figure is saved as {out_path}")

    functions = [np.mean, np.median, np.max, np.min]
    function_names = ["mean  ", "median", "max   ", "min   "]
    for foo, name in zip(functions, function_names):
        avg_nlin_uwb = foo(nlin_uwb)
        print(
            name + f" NLIN coeff per channel: UWB -> {avg_nlin_uwb:4.3e} ") #| SMF -> {avg_nlin_smf:4.3e}")
        # apply QAM 16 and -10 dBm input power
        print(
            name + f" NLIN power per channel: UWB -> {watt2dBm(avg_nlin_uwb):4.1f} dBm ")#| SMF -> {watt2dBm(avg_nlin_smf):4.1f} dBm")
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
    if os.getenv("POGGIOLINI_WORKFLOW") == "1":
        from analysis.poggiolini_nlin import run_poggiolini_workflow
        run_poggiolini_workflow()
        raise SystemExit(0)
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
