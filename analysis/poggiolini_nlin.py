import os
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger as lg
from scipy.constants import c

from pynlin.nlin.nlin_estimator_uwb import collision_coeffs_system_uwb, total_nlin_uwb
import pynlin.nlin.nlin_estimator_uwb as nlin_uwb
from pynlin.nlin.pcfm_gn import PcfmConfig, compute_gn_numeric, compute_gn_direct, compute_pcfm_nlin
from pynlin.system import System
from pynlin.utils import dBm2watt
from pynlin.constellations import QAM

from pynlin.raman.solvers_jiang import JiangIterativeConfig
from pynlin.nlin.pcfm_gn import load_signal_profiles, normalize_spp, fit_spp_polynomials
from analysis.uwb_nlin import (
    _load_profile_launch_powers,
    _load_profile_output_powers,
    _nlin_cache_path,
    compute_raman_profiles,
    plot_power_profiles,
)

PROFILE_MAX_W = 10.0


def _qam_mu0(order: int) -> float:
    """Return mu0 = <|b|^4>/<|b|^2>^2 for uniform QAM."""
    qam = QAM(order)
    syms = qam.symbols()
    p2 = np.mean(np.abs(syms) ** 2)
    if p2 <= 0:
        raise ValueError("Invalid QAM average power for mu0.")
    return float(np.mean(np.abs(syms) ** 4) / (p2 ** 2))


def _td_prefactor_coeffs(mode_a: int, mode_b: int, n_modes: int) -> tuple[float, float]:
    """Return (a, b) so prefactor = a * mu0 + b."""
    if n_modes == 1:
        mode_a = mode_b = 0
    if mode_a == mode_b:
        a = 2.0 * nlin_uwb.SPATIAL_MODES[mode_a] + 3.0
        b = -4.0
    else:
        a = 2.0 * nlin_uwb.SPATIAL_MODES[mode_b]
        b = -2.0 * nlin_uwb.SPATIAL_MODES[mode_b]
    return a, b


def _td_modulation_components(system: System,
                              collision_coeffs: np.ndarray,
                              launch_powers_w: np.ndarray | None,
                              use_kappa: bool = True,
                              use_x_mode: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (constant_prefactor, sum_a, sum_b) for TD modulation scaling."""
    L = float(system.fiber_length)
    br = float(system.pulse.baud_rate)
    y_norm = 1.0 / (L * br) ** 2
    collision_coeffs_si = collision_coeffs / y_norm

    n_modes, n_freqs, _, _ = collision_coeffs_si.shape
    if launch_powers_w is None:
        power_dbm = system.launch_power if system.launch_power is not None else -5.0
        P_in_arr = np.full((n_modes, n_freqs), dBm2watt(power_dbm))
    else:
        P_raw = np.asarray(launch_powers_w, dtype=float)
        if P_raw.ndim == 1:
            if P_raw.size != n_freqs:
                raise ValueError(f"launch_powers_w length {P_raw.size} != n_freqs {n_freqs}")
            P_in_arr = np.broadcast_to(P_raw[None, :], (n_modes, n_freqs))
        elif P_raw.shape == (n_modes, n_freqs):
            P_in_arr = P_raw
        else:
            raise ValueError(f"launch_powers_w shape {P_raw.shape} incompatible with (n_modes,n_freqs)=({n_modes},{n_freqs})")

    freqs = system.wdm.frequency_grid()
    n2 = 2.6e-20
    aeff = nlin_uwb._effective_area_array(system, freqs)
    gamma = n2 * (2.0 * np.pi * freqs) / (aeff * c)
    gamma = gamma[None, :]
    constant_prefactor = (P_in_arr ** 3) * (gamma ** 2) / (br ** 2)

    kappa2 = nlin_uwb.get_kappa2_matrix_uwb(system, use_kappa, use_x_mode)

    sum_a = np.zeros((n_modes, n_freqs), dtype=float)
    sum_b = np.zeros_like(sum_a)
    for mA in range(n_modes):
        for nuA in range(n_freqs):
            for mB in range(n_modes):
                a, b = _td_prefactor_coeffs(mA, mB, n_modes)
                weight = kappa2[mA, mB]
                coeff_sum = float(np.sum(collision_coeffs_si[mA, nuA, mB, :]))
                sum_a[mA, nuA] += weight * coeff_sum * a
                sum_b[mA, nuA] += weight * coeff_sum * b
    return constant_prefactor, sum_a, sum_b


def _profile_needs_recompute(profile_path: Path | str, max_power_w: float = PROFILE_MAX_W) -> bool:
    path = Path(profile_path)
    if not path.exists():
        return True
    try:
        data = np.load(path, allow_pickle=True)
        if isinstance(data, np.lib.npyio.NpzFile):
            sig = data.get("signal_sol")
        else:
            data = data.item() if isinstance(data, np.ndarray) and data.shape == () else data
            sig = data.get("signal_sol") if isinstance(data, dict) else None
    except Exception as exc:
        lg.warning(f"Failed to read profile for validation: {exc}")
        return True
    if sig is None:
        lg.warning("Profile missing signal_sol; recompute required.")
        return True
    sig = np.asarray(sig, dtype=float)
    if not np.all(np.isfinite(sig)):
        lg.warning("Profile contains non-finite values; recompute required.")
        return True
    max_val = float(np.nanmax(sig))
    if max_val > max_power_w:
        lg.warning(f"Profile max power {max_val:.2e} W exceeds {max_power_w:.2e} W; recompute required.")
        return True
    return False


def _load_launch_powers_csv(path: Path | str, freqs_hz: np.ndarray) -> np.ndarray | None:
    """Load per-channel launch powers from a CSV with frequency_THz,launch_power_dbm."""
    csv_path = Path(path)
    if not csv_path.exists():
        return None
    try:
        data = np.genfromtxt(csv_path, delimiter=",", names=True)
    except Exception as exc:
        lg.warning(f"Failed reading launch power CSV {csv_path}: {exc}")
        return None
    if "frequency_THz" not in data.dtype.names or "launch_power_dbm" not in data.dtype.names:
        lg.warning(f"CSV {csv_path} missing required headers frequency_THz, launch_power_dbm.")
        return None
    freqs_thz = np.asarray(data["frequency_THz"], dtype=float)
    powers_dbm = np.asarray(data["launch_power_dbm"], dtype=float)
    if freqs_thz.size != powers_dbm.size:
        lg.warning(f"CSV {csv_path} has mismatched frequency/power lengths.")
        return None
    freqs_csv = freqs_thz * 1e12
    order = np.argsort(freqs_csv)
    freqs_csv = freqs_csv[order]
    powers_dbm = powers_dbm[order]
    powers_interp_dbm = np.interp(freqs_hz, freqs_csv, powers_dbm)
    return dBm2watt(powers_interp_dbm)


def _resolve_launch_powers(system: System,
                           profile_path: Path | str | None,
                           launch_csv_path: Path | str | None,
                           use_profile: bool = True) -> np.ndarray:
    """Resolve per-channel launch powers (W) with priority: profile > CSV > TOML.

    When use_profile=True, raise if profile launch powers do not match the
    current launch setting (CSV if provided, otherwise TOML).
    """
    freqs = system.wdm.frequency_grid()

    def _launch_from_toml() -> np.ndarray:
        if hasattr(system.wdm, "band_specs") and system.wdm.band_specs:
            launch = np.zeros_like(freqs, dtype=float)
            for name, slc in system.wdm._band_slices.items():
                spec = system.wdm.band_specs.get(name)
                power_dbm = spec.launch_power_dbm if spec else None
                if power_dbm is None:
                    power_dbm = system.launch_power if system.launch_power is not None else -5.0
                launch[slc] = dBm2watt(power_dbm)
            return launch
        power_dbm = system.launch_power if system.launch_power is not None else -5.0
        return np.full_like(freqs, dBm2watt(power_dbm), dtype=float)

    def _validate_launch(label: str, launch: np.ndarray) -> np.ndarray:
        if (not np.all(np.isfinite(launch)) or np.any(launch <= 0) or np.max(launch) > PROFILE_MAX_W):
            raise ValueError(f"Launch powers from {label} are unreasonable.")
        return launch

    def _expected_launch_from_settings() -> tuple[np.ndarray, str]:
        if launch_csv_path is not None:
            launch_from_csv = _load_launch_powers_csv(launch_csv_path, freqs)
            if launch_from_csv is not None:
                return _validate_launch("CSV", launch_from_csv), "CSV"
        return _validate_launch("TOML", _launch_from_toml()), "TOML"

    if use_profile:
        if profile_path is None:
            raise FileNotFoundError("Profile path required when use_profile=True.")
        launch_from_profile = _load_profile_launch_powers(profile_path, system.n_channels)
        if launch_from_profile is None:
            raise FileNotFoundError("Launch powers missing in Raman profile.")
        launch_from_profile = _validate_launch("profile", launch_from_profile)

        expected_launch, expected_label = _expected_launch_from_settings()
        profile_dbm = 10.0 * np.log10(launch_from_profile / 1e-3)
        expected_dbm = 10.0 * np.log10(expected_launch / 1e-3)
        max_abs_db = float(np.max(np.abs(profile_dbm - expected_dbm)))
        if max_abs_db > 0.1:
            raise ValueError(
                f"Profile launch powers do not match current {expected_label} setting "
                f"(max |Δ| = {max_abs_db:.3f} dB). Recompute profiles or update config."
            )
        lg.info("Using launch powers from Raman profile (validated against current settings).")
        return launch_from_profile

    # If profile not requested, fall back to CSV/TOML in that order.
    if launch_csv_path is not None:
        launch_from_csv = _load_launch_powers_csv(launch_csv_path, freqs)
        if launch_from_csv is not None:
            lg.info(f"Using launch powers from {launch_csv_path}")
            return _validate_launch("CSV", launch_from_csv)

    launch = _launch_from_toml()
    if hasattr(system.wdm, "band_specs") and system.wdm.band_specs:
        lg.info("Using per-band launch powers from TOML.")
    else:
        lg.info("Using uniform launch power from TOML.")
    return _validate_launch("TOML", launch)


def _resolve_signal_power(system: System,
                          profile_path: Path | str | None,
                          launch_override: np.ndarray | None) -> np.ndarray:
    """Return per-channel signal power (W) for GSNR computations."""
    """Use launch powers to keep TD and PCFM GSNR inputs identical."""
    freqs = system.wdm.frequency_grid()
    if launch_override is not None:
        return launch_override
    power_dbm = system.launch_power if system.launch_power is not None else -5.0
    return np.full_like(freqs, dBm2watt(power_dbm), dtype=float)


def _save_nlin_csv(path: Path | str,
                   freqs_hz: np.ndarray,
                   nlin_w: np.ndarray,
                   signal_power_w: np.ndarray) -> None:
    """Save NLIN + GSNR to CSV."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    nlin_flat = np.asarray(nlin_w, dtype=float).reshape(-1)
    if nlin_flat.size != freqs_hz.size:
        raise ValueError(f"NLIN array size {nlin_flat.size} != freq size {freqs_hz.size}")
    signal_power_w = np.asarray(signal_power_w, dtype=float).reshape(-1)
    if signal_power_w.size != freqs_hz.size:
        raise ValueError(f"Signal power size {signal_power_w.size} != freq size {freqs_hz.size}")
    denom = np.maximum(nlin_flat, 1e-18)
    gsnr_db = 10.0 * np.log10(signal_power_w / denom)
    data = np.column_stack([freqs_hz * 1e-12, nlin_flat, gsnr_db])
    header = "frequency_THz,nlin_W,gsnr_nli_dB"
    np.savetxt(out, data, delimiter=",", header=header, comments="")
    lg.success(f"Saved NLIN CSV to {out}")


def _load_or_compute_pcfm(system: System,
                          profile_path: Path | str,
                          launch_powers_w: np.ndarray,
                          output_path: Path,
                          cfg: PcfmConfig,
                          lumped_losses: list[tuple[float, float]] | None = None,
                          recompute: bool = False,
                          return_components: bool = False) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute or load PCFM NLIN and persist to .npy."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sci_path = output_path.with_name(f"{output_path.stem}_sci.npy") if return_components else None
    xci_path = output_path.with_name(f"{output_path.stem}_xci.npy") if return_components else None
    if output_path.exists() and not recompute:
        lg.info(f"Loading cached PCFM NLIN from {output_path}")
        cached = np.load(output_path)
        if np.all(np.isfinite(cached)):
            if return_components:
                if sci_path is None or xci_path is None:
                    raise ValueError("PCFM component paths not configured.")
                if sci_path.exists() and xci_path.exists():
                    sci = np.load(sci_path)
                    xci = np.load(xci_path)
                    if np.all(np.isfinite(sci)) and np.all(np.isfinite(xci)):
                        return cached, sci, xci
                    raise ValueError(
                        f"Cached PCFM components contain non-finite values; "
                        "re-run with recompute_pcfm=True."
                    )
                lg.warning("PCFM component cache missing; recomputing.")
            else:
                return cached
        raise ValueError(
            f"Cached PCFM NLIN at {output_path} contains non-finite values; "
            "re-run with recompute_pcfm=True."
        )
    lg.info(f"Computing PCFM NLIN (losses={lumped_losses})")
    if return_components:
        nlin, nlin_sci, nlin_xci = compute_pcfm_nlin(
            system,
            profile_path=profile_path,
            launch_powers_w=launch_powers_w,
            config=cfg,
            lumped_losses=lumped_losses,
            return_components=True,
        )
        np.save(output_path, nlin)
        if sci_path is not None and xci_path is not None:
            np.save(sci_path, nlin_sci)
            np.save(xci_path, nlin_xci)
            lg.success(f"Saved PCFM SCI to {sci_path}")
            lg.success(f"Saved PCFM XCI to {xci_path}")
        lg.success(f"Saved PCFM NLIN to {output_path}")
        return nlin, nlin_sci, nlin_xci
    nlin = compute_pcfm_nlin(
        system,
        profile_path=profile_path,
        launch_powers_w=launch_powers_w,
        config=cfg,
        lumped_losses=lumped_losses,
    )
    np.save(output_path, nlin)
    lg.success(f"Saved PCFM NLIN to {output_path}")
    return nlin


def _load_or_compute_gn(system: System,
                        profile_path: Path | str,
                        launch_powers_w: np.ndarray,
                        output_path: Path,
                        lumped_losses: list[tuple[float, float]] | None = None,
                        recompute: bool = False,
                        n_f: int = 40,
                        n_z: int = 200,
                        return_components: bool = False) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute or load numeric GN NLIN and persist to .npy."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sci_path = output_path.with_name(f"{output_path.stem}_sci.npy") if return_components else None
    xci_path = output_path.with_name(f"{output_path.stem}_xci.npy") if return_components else None
    if output_path.exists() and not recompute:
        lg.info(f"Loading cached GN NLIN from {output_path}")
        cached = np.load(output_path)
        if np.all(np.isfinite(cached)):
            if return_components:
                if sci_path is None or xci_path is None:
                    raise ValueError("GN component paths not configured.")
                if sci_path.exists() and xci_path.exists():
                    sci = np.load(sci_path)
                    xci = np.load(xci_path)
                    if np.all(np.isfinite(sci)) and np.all(np.isfinite(xci)):
                        return cached, sci, xci
                    raise ValueError(
                        f"Cached GN components contain non-finite values; "
                        "re-run with recompute_gn=True."
                    )
                lg.warning("GN component cache missing; recomputing.")
            else:
                return cached
        raise ValueError(
            f"Cached GN NLIN at {output_path} contains non-finite values; "
            "re-run with recompute_gn=True."
        )
    lg.info(f"Computing numeric GN NLIN (losses={lumped_losses})")
    if return_components:
        nlin, nlin_sci, nlin_xci = compute_gn_numeric(
            system,
            profile_path=profile_path,
            launch_powers_w=launch_powers_w,
            lumped_losses=lumped_losses,
            n_f=n_f,
            n_z=n_z,
            return_components=True,
        )
        np.save(output_path, nlin)
        if sci_path is not None and xci_path is not None:
            np.save(sci_path, nlin_sci)
            np.save(xci_path, nlin_xci)
            lg.success(f"Saved numeric GN SCI to {sci_path}")
            lg.success(f"Saved numeric GN XCI to {xci_path}")
        lg.success(f"Saved numeric GN NLIN to {output_path}")
        return nlin, nlin_sci, nlin_xci
    nlin = compute_gn_numeric(
        system,
        profile_path=profile_path,
        launch_powers_w=launch_powers_w,
        lumped_losses=lumped_losses,
        n_f=n_f,
        n_z=n_z,
    )
    np.save(output_path, nlin)
    lg.success(f"Saved numeric GN NLIN to {output_path}")
    return nlin


def _load_or_compute_gn_direct(system: System,
                               profile_path: Path | str,
                               launch_powers_w: np.ndarray,
                               output_path: Path,
                               lumped_losses: list[tuple[float, float]] | None = None,
                               recompute: bool = False,
                               n_f: int = 40,
                               return_components: bool = False
                               ) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute or load direct GN NLIN and persist to .npy."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sci_path = output_path.with_name(f"{output_path.stem}_sci.npy") if return_components else None
    xci_path = output_path.with_name(f"{output_path.stem}_xci.npy") if return_components else None
    if output_path.exists() and not recompute:
        lg.info(f"Loading cached direct GN NLIN from {output_path}")
        cached = np.load(output_path)
        if np.all(np.isfinite(cached)):
            if return_components:
                if sci_path is None or xci_path is None:
                    raise ValueError("Direct GN component paths not configured.")
                if sci_path.exists() and xci_path.exists():
                    sci = np.load(sci_path)
                    xci = np.load(xci_path)
                    if np.all(np.isfinite(sci)) and np.all(np.isfinite(xci)):
                        return cached, sci, xci
                    raise ValueError(
                        "Cached direct GN components contain non-finite values; "
                        "re-run with recompute_gn_direct=True."
                    )
                lg.warning("Direct GN component cache missing; recomputing.")
            else:
                return cached
        raise ValueError(
            f"Cached direct GN NLIN at {output_path} contains non-finite values; "
            "re-run with recompute_gn_direct=True."
        )
    lg.info(f"Computing direct GN NLIN (losses={lumped_losses})")
    if return_components:
        nlin, nlin_sci, nlin_xci = compute_gn_direct(
            system,
            profile_path=profile_path,
            launch_powers_w=launch_powers_w,
            lumped_losses=lumped_losses,
            n_f=n_f,
            return_components=True,
        )
        np.save(output_path, nlin)
        if sci_path is not None and xci_path is not None:
            np.save(sci_path, nlin_sci)
            np.save(xci_path, nlin_xci)
            lg.success(f"Saved direct GN SCI to {sci_path}")
            lg.success(f"Saved direct GN XCI to {xci_path}")
        lg.success(f"Saved direct GN NLIN to {output_path}")
        return nlin, nlin_sci, nlin_xci
    nlin = compute_gn_direct(
        system,
        profile_path=profile_path,
        launch_powers_w=launch_powers_w,
        lumped_losses=lumped_losses,
        n_f=n_f,
    )
    np.save(output_path, nlin)
    lg.success(f"Saved direct GN NLIN to {output_path}")
    return nlin


def plot_poggiolini_gsnr(freqs_hz: np.ndarray,
                         gsnr_td: np.ndarray,
                         gsnr_pcfm: dict[str, np.ndarray],
                         gsnr_gn: dict[str, np.ndarray] | None,
                         out_path: Path,
                         gsnr_gn_direct: dict[str, np.ndarray] | None = None,
                         title: str | None = None) -> None:
    """Plot GSNR_NLI overlays with the same style as existing noise plots."""
    dpi = 300
    grid = False
    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    if title:
        ax.set_title(title, fontsize=9)
    ax.plot(freqs_hz * 1e-12, gsnr_td, color="black", lw=0.9, label="TD")

    colors = ["tab:blue", "tab:orange", "tab:green"]
    for idx, (label, gsnr) in enumerate(gsnr_pcfm.items()):
        display = "" if label == "no_loss" else label
        suffix = f" {display}" if display else ""
        color = colors[idx % len(colors)]
        ax.plot(freqs_hz * 1e-12, gsnr, color=color, lw=0.8, label=f"PCFM{suffix}")

    if gsnr_gn:
        for idx, (label, gsnr) in enumerate(gsnr_gn.items()):
            display = "" if label == "no_loss" else label
            suffix = f" {display}" if display else ""
            color = colors[idx % len(colors)]
            ax.scatter(freqs_hz * 1e-12, gsnr, s=6, facecolors="none",
                       edgecolors=color, linewidths=0.5, label=f"GN{suffix}")

    if gsnr_gn_direct:
        for idx, (label, gsnr) in enumerate(gsnr_gn_direct.items()):
            display = "" if label == "no_loss" else label
            suffix = f" {display}" if display else ""
            color = colors[idx % len(colors)]
            ax.scatter(freqs_hz * 1e-12, gsnr, s=14, marker="x",
                       color=color, linewidths=0.6, label=f"GN dir{suffix}")

    ax.set_xlabel(r"$f \; [\mathrm{THz}]$")
    ax.set_ylabel(r"$GSNR_{NLI} \; [\mathrm{dB}]$")
    ax.grid(grid)
    ax.legend(loc="best", fontsize=7)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    lg.success(f"Saved GSNR plot to {out_path}")


def plot_poggiolini_nlin_power(freqs_hz: np.ndarray,
                               nlin_td_w: np.ndarray,
                               nlin_pcfm_w: dict[str, np.ndarray],
                               nlin_gn_w: dict[str, np.ndarray] | None,
                               out_path: Path,
                               nlin_td_mod_w: dict[str, np.ndarray] | None = None,
                               nlin_pcfm_xci_w: dict[str, np.ndarray] | None = None,
                               nlin_gn_xci_w: dict[str, np.ndarray] | None = None,
                               nlin_gn_direct_w: dict[str, np.ndarray] | None = None,
                               nlin_gn_direct_xci_w: dict[str, np.ndarray] | None = None) -> None:
    """Plot NLIN power per channel (dBm)."""
    dpi = 300
    grid = False
    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    if nlin_td_mod_w:
        styles = ["-", "--", ":"]
        for idx, (label, nlin) in enumerate(nlin_td_mod_w.items()):
            nlin_td_dbm = 10.0 * np.log10(np.maximum(nlin, 1e-18) / 1e-3)
            ax.plot(freqs_hz * 1e-12, nlin_td_dbm, color="black", lw=0.9,
                    ls=styles[idx % len(styles)], label=f"TD {label}")
    else:
        nlin_td_dbm = 10.0 * np.log10(np.maximum(nlin_td_w, 1e-18) / 1e-3)
        ax.plot(freqs_hz * 1e-12, nlin_td_dbm, color="black", lw=0.9, label="TD")

    colors = ["tab:blue", "tab:orange", "tab:green"]
    for idx, (label, nlin) in enumerate(nlin_pcfm_w.items()):
        display = "" if label == "no_loss" else label
        suffix = f" {display}" if display else ""
        color = colors[idx % len(colors)]
        nlin_dbm = 10.0 * np.log10(np.maximum(nlin, 1e-18) / 1e-3)
        ax.plot(freqs_hz * 1e-12, nlin_dbm, color=color, lw=0.8, label=f"PCFM{suffix}")

    if nlin_pcfm_xci_w:
        for idx, (label, nlin) in enumerate(nlin_pcfm_xci_w.items()):
            display = "" if label == "no_loss" else label
            suffix = f" {display}" if display else ""
            color = colors[idx % len(colors)]
            nlin_dbm = 10.0 * np.log10(np.maximum(nlin, 1e-18) / 1e-3)
            ax.plot(freqs_hz * 1e-12, nlin_dbm, color=color, lw=0.8, ls="--",
                    label=f"PCFM XCI{suffix}")

    if nlin_gn_w:
        for idx, (label, nlin) in enumerate(nlin_gn_w.items()):
            display = "" if label == "no_loss" else label
            suffix = f" {display}" if display else ""
            color = colors[idx % len(colors)]
            nlin_dbm = 10.0 * np.log10(np.maximum(nlin, 1e-18) / 1e-3)
            ax.scatter(freqs_hz * 1e-12, nlin_dbm, s=6, facecolors="none",
                       edgecolors=color, linewidths=0.5, label=f"GN{suffix}")

    if nlin_gn_direct_w:
        for idx, (label, nlin) in enumerate(nlin_gn_direct_w.items()):
            display = "" if label == "no_loss" else label
            suffix = f" {display}" if display else ""
            color = colors[idx % len(colors)]
            nlin_dbm = 10.0 * np.log10(np.maximum(nlin, 1e-18) / 1e-3)
            ax.scatter(freqs_hz * 1e-12, nlin_dbm, s=18, marker="^", facecolors="none",
                       edgecolors=color, linewidths=0.6, label=f"GN dir{suffix}")

    if nlin_gn_xci_w:
        for idx, (label, nlin) in enumerate(nlin_gn_xci_w.items()):
            display = "" if label == "no_loss" else label
            suffix = f" {display}" if display else ""
            color = colors[idx % len(colors)]
            nlin_dbm = 10.0 * np.log10(np.maximum(nlin, 1e-18) / 1e-3)
            ax.scatter(freqs_hz * 1e-12, nlin_dbm, s=14, marker="x",
                       color=color, linewidths=0.6, label=f"GN XCI{suffix}")

    if nlin_gn_direct_xci_w:
        for idx, (label, nlin) in enumerate(nlin_gn_direct_xci_w.items()):
            display = "" if label == "no_loss" else label
            suffix = f" {display}" if display else ""
            color = colors[idx % len(colors)]
            nlin_dbm = 10.0 * np.log10(np.maximum(nlin, 1e-18) / 1e-3)
            ax.scatter(freqs_hz * 1e-12, nlin_dbm, s=18, marker="+",
                       color=color, linewidths=0.7, label=f"GN dir XCI{suffix}")

    ax.set_xlabel(r"$f \; [\mathrm{THz}]$")
    ax.set_ylabel(r"$P_{NLI}\;[\mathrm{dBm}]$")
    ax.grid(grid)
    ax.legend(loc="best", fontsize=7)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    lg.success(f"Saved NLIN power plot to {out_path}")


def plot_poggiolini_diagnostics(system: System,
                                profile_path: Path | str,
                                launch_powers_w: np.ndarray,
                                out_dir: Path) -> None:
    """Generate diagnostic plots for intermediate quantities."""
    out_dir.mkdir(parents=True, exist_ok=True)
    freqs = system.wdm.frequency_grid()
    freqs_thz = freqs * 1e-12
    launch_dbm = 10.0 * np.log10(np.maximum(launch_powers_w, 1e-18) / 1e-3)
    out_dbm = None

    # Launch power plot
    fig, ax = plt.subplots(figsize=(3.6, 2.4))
    ax.plot(freqs_thz, launch_dbm, lw=0.8, color="black")
    ax.set_xlabel(r"$f \; [\mathrm{THz}]$")
    ax.set_ylabel(r"$P_\mathrm{launch}\;[\mathrm{dBm}]$")
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(out_dir / "poggiolini_launch_power.pdf", dpi=300)
    lg.success(f"Saved launch power plot to {out_dir / 'poggiolini_launch_power.pdf'}")
    plt.close(fig)

    # Profile powers (avg and output)
    try:
        sig_ch_z, z_axis = load_signal_profiles(profile_path, system)
        span = float(z_axis[-1] - z_axis[0]) if z_axis.size else 0.0
        if span > 0:
            avg_power = np.trapezoid(sig_ch_z, z_axis, axis=1) / span
        else:
            avg_power = sig_ch_z[:, 0]
        out_power = sig_ch_z[:, -1]
        avg_dbm = 10.0 * np.log10(np.maximum(avg_power, 1e-18) / 1e-3)
        out_dbm = 10.0 * np.log10(np.maximum(out_power, 1e-18) / 1e-3)

        fig, ax = plt.subplots(figsize=(3.6, 2.4))
        ax.plot(freqs_thz, avg_dbm, lw=0.8, color="tab:blue", label="avg")
        ax.plot(freqs_thz, out_dbm, lw=0.8, color="tab:orange", label="out")
        ax.set_xlabel(r"$f \; [\mathrm{THz}]$")
        ax.set_ylabel(r"$P\;[\mathrm{dBm}]$")
        ax.grid(False)
        ax.legend(loc="best", fontsize=7)
        fig.tight_layout()
        fig.savefig(out_dir / "poggiolini_profile_power.pdf", dpi=300)
        lg.success(f"Saved profile power plot to {out_dir / 'poggiolini_profile_power.pdf'}")
        plt.close(fig)

        # SPP fits (exact vs 9th-order) for center L/C/S channels
        spp = normalize_spp(sig_ch_z, z_axis)
        coeffs = fit_spp_polynomials(z_axis, spp, degree=9)
        L = float(z_axis[-1] - z_axis[0]) if z_axis.size else 1.0
        z_norm = (z_axis - z_axis[0]) / L if L > 0 else z_axis

        picks = []
        labels = []
        band_slices = getattr(system.wdm, "_band_slices", {}) if hasattr(system.wdm, "_band_slices") else {}
        if band_slices:
            for band_name in ("L", "C", "S"):
                for key, slc in band_slices.items():
                    if str(key).lower().startswith(band_name.lower()):
                        idx = int((slc.start + slc.stop - 1) // 2)
                        picks.append(idx)
                        labels.append(f"Ch {idx}: center-{band_name}")
                        break
        if not picks:
            # fallback: three evenly spaced channels
            n_ch = spp.shape[0]
            picks = [n_ch // 6, n_ch // 2, 5 * n_ch // 6]
            labels = [f"Ch {p}" for p in picks]

        fig, ax = plt.subplots(figsize=(4.0, 2.8))
        first = True
        for idx, label in zip(picks, labels):
            p_fit = np.polynomial.polynomial.polyval(z_norm, coeffs[idx])
            ax.plot(z_axis / 1e3, spp[idx], color="tab:blue", lw=0.9,
                    label="exact" if first else None)
            ax.plot(z_axis / 1e3, p_fit, color="tab:red", lw=0.9, ls="--",
                    label="polynomial Np=9" if first else None)
            # annotate at ~40% span
            z_anno = z_axis[int(0.4 * (len(z_axis) - 1))] / 1e3
            y_anno = np.interp(z_anno * 1e3, z_axis, spp[idx])
            ax.annotate(
                label,
                xy=(z_anno, y_anno),
                xytext=(z_anno + 5, y_anno + 0.08),
                textcoords="data",
                fontsize=7,
                arrowprops=dict(arrowstyle="->", lw=0.6),
            )
            first = False
        ax.set_xlabel(r"$z\;[\mathrm{km}]$")
        ax.set_ylabel(r"normalized power")
        ax.grid(False)
        ax.legend(loc="best", fontsize=7)
        fig.tight_layout()
        fig.savefig(out_dir / "poggiolini_spp_fit.pdf", dpi=300)
        lg.success(f"Saved SPP fit plot to {out_dir / 'poggiolini_spp_fit.pdf'}")
        plt.close(fig)

        # p(L) and poly_sum vs frequency
        p_L = np.array([np.polynomial.polynomial.polyval(1.0, c) for c in coeffs], dtype=float)
        poly_sum = np.array([np.sum(np.convolve(c, c) / (np.arange(len(c)*2-1) + 1.0)) for c in coeffs], dtype=float)
        fig, ax1 = plt.subplots(figsize=(3.6, 2.4))
        ax1.plot(freqs_thz, p_L, lw=0.8, color="tab:blue")
        ax1.set_xlabel(r"$f \; [\mathrm{THz}]$")
        ax1.set_ylabel(r"$p(L)$", color="tab:blue")
        ax2 = ax1.twinx()
        ax2.plot(freqs_thz, poly_sum, lw=0.8, color="tab:orange")
        ax2.set_ylabel(r"$\\sum a_n a_k/(n+k+1)$", color="tab:orange")
        ax1.grid(False)
        fig.tight_layout()
        fig.savefig(out_dir / "poggiolini_pcfm_terms.pdf", dpi=300)
        lg.success(f"Saved PCFM terms plot to {out_dir / 'poggiolini_pcfm_terms.pdf'}")
        plt.close(fig)
    except Exception as exc:
        lg.warning(f"Diagnostics skipped (profile-dependent): {exc}")

    # Launch spectrum scatter (signals + pumps), UWB-style (with output spectrum)
    try:
        pump_specs = system.pump_specs or []
        if pump_specs:
            pump_freqs_thz = np.array([3e8 / p.wavelength for p in pump_specs], dtype=float) * 1e-12
            pump_powers_dbm = np.array([p.power_dbm for p in pump_specs], dtype=float)
        else:
            pump_freqs_thz = np.array([])
            pump_powers_dbm = np.array([])
        fig, ax = plt.subplots(figsize=(3.6, 2.4))
        ax.scatter(freqs_thz, launch_dbm, s=6, alpha=0.7, label="signals (launch)")
        if out_dbm is not None:
            ax.scatter(freqs_thz, out_dbm, s=6, facecolors="none", edgecolors="tab:blue",
                       linewidths=0.6, label="signals (out)")
        if pump_powers_dbm.size:
            ax.scatter(pump_freqs_thz, pump_powers_dbm, marker="x", s=16, color="tab:red", label="pumps")
        ax.set_xlabel(r"$f \; [\mathrm{THz}]$")
        ax.set_ylabel(r"$P\;[\mathrm{dBm}]$")
        ax.grid(False)
        ax.legend(loc="best", fontsize=7)
        fig.tight_layout()
        fig.savefig(out_dir / "poggiolini_launch_spectrum.pdf", dpi=300)
        lg.success(f"Saved launch spectrum plot to {out_dir / 'poggiolini_launch_spectrum.pdf'}")
        plt.close(fig)
    except Exception as exc:
        lg.warning(f"Launch spectrum plot skipped: {exc}")

    # Fiber parameters (beta2, Aeff)
    try:
        wl = 3e8 / freqs
        beta2 = np.array([system.fiber.beta2_at(float(w)) for w in wl], dtype=float)
        aeff = np.array([system.fiber.effective_area_at(float(w)) for w in wl], dtype=float)
        fig, ax1 = plt.subplots(figsize=(3.6, 2.4))
        ax1.plot(freqs_thz, beta2 * 1e24, lw=0.8, color="tab:blue")
        ax1.set_xlabel(r"$f \; [\mathrm{THz}]$")
        ax1.set_ylabel(r"$\\beta_2\\;[10^{-24}\\,s^2/m]$", color="tab:blue")
        ax2 = ax1.twinx()
        ax2.plot(freqs_thz, aeff * 1e12, lw=0.8, color="tab:orange")
        ax2.set_ylabel(r"$A_{eff}\\;[\\mu m^2]$", color="tab:orange")
        ax1.grid(False)
        fig.tight_layout()
        fig.savefig(out_dir / "poggiolini_fiber_params.pdf", dpi=300)
        lg.success(f"Saved fiber parameters plot to {out_dir / 'poggiolini_fiber_params.pdf'}")
        plt.close(fig)
    except Exception as exc:
        lg.warning(f"Diagnostics skipped (fiber params): {exc}")


def run_poggiolini_workflow(cfg_path: Path | str = Path("./input/poggiolini_struct.toml"),
                            profile_path: Path | str = Path("results/poggiolini_power_profiles.npy"),
                            launch_csv_path: Path | str | None = Path("results/poggiolini_launch_power.csv"),
                            recompute_profiles: bool = False,
                            recompute_td: bool = False,
                            recompute_pcfm: bool = False,
                            recompute_gn: bool = False,
                            recompute_gn_direct: bool = False,
                            use_profile_launch_powers: bool = True,
                            compute_gn: bool = False,
                            compute_gn_direct: bool = False,
                            pcfm_numeric_xci: bool = False,
                            include_lumped_losses: bool = False,
                            plot: bool = True) -> None:
    """Run TD + PCFM (+ optional GN) workflow and plot GSNR overlays."""
    system = System.from_toml(cfg_path)
    freqs = system.wdm.frequency_grid()

    # Ensure Raman profiles exist (recompute only if explicitly requested)
    if Path(profile_path).exists():
        if recompute_profiles:
            jiang_cfg = JiangIterativeConfig(
                iterative_steps=120,
                pump_scale_start=1e-6,
                early_stop_rtol=1e-4,
            )
            lg.info("Recomputing Raman profiles with Jiang solver configuration.")
            compute_raman_profiles(
                system,
                save_path=profile_path,
                recompute=True,
                jiang_cfg=jiang_cfg,
                max_power_w=PROFILE_MAX_W,
            )
        else:
            compute_raman_profiles(system, save_path=profile_path, recompute=False)
    else:
        if not recompute_profiles:
            raise FileNotFoundError(
                f"Raman profile missing at {profile_path}. Re-run with recompute_profiles=True."
            )
        jiang_cfg = JiangIterativeConfig(
            iterative_steps=120,
            pump_scale_start=1e-6,
            early_stop_rtol=1e-4,
        )
        lg.info("Computing Raman profiles with Jiang solver configuration.")
        compute_raman_profiles(
            system,
            save_path=profile_path,
            recompute=True,
            jiang_cfg=jiang_cfg,
            max_power_w=PROFILE_MAX_W,
        )
    if _profile_needs_recompute(profile_path):
        raise ValueError(
            f"Raman profile at {profile_path} appears invalid; rerun with recompute_profiles=True."
        )
    launch_powers = _resolve_launch_powers(
        system, profile_path, launch_csv_path, use_profile=use_profile_launch_powers
    )
    # Log span-averaged and output powers for sanity (do not use in calculations).
    try:
        sig_ch_z, z_axis = load_signal_profiles(profile_path, system)
        span = float(z_axis[-1] - z_axis[0]) if z_axis.size else 0.0
        if span > 0:
            avg_power = np.trapezoid(sig_ch_z, z_axis, axis=1) / span
            out_power = sig_ch_z[:, -1]
            lg.info(
                "Profile power summary (per-channel): "
                f"avg_W min/med/max = {float(np.min(avg_power)):.3e} / "
                f"{float(np.median(avg_power)):.3e} / {float(np.max(avg_power)):.3e}; "
                f"out_W min/med/max = {float(np.min(out_power)):.3e} / "
                f"{float(np.median(out_power)):.3e} / {float(np.max(out_power)):.3e}"
            )
    except Exception as exc:
        lg.warning(f"Failed to compute profile power summary: {exc}")
    if plot:
        plot_power_profiles(system, profile_path)
        plot_poggiolini_diagnostics(
            system=system,
            profile_path=profile_path,
            launch_powers_w=launch_powers,
            out_dir=Path("media"),
        )
    # Log launch powers summary for sanity checks
    launch_dbm = 10.0 * np.log10(np.maximum(launch_powers, 1e-18) / 1e-3)
    lg.info(
        "Launch power summary (per-channel): dBm min/med/max = "
        f"{float(np.min(launch_dbm)):.2f} / {float(np.median(launch_dbm)):.2f} / {float(np.max(launch_dbm)):.2f}"
    )
    if hasattr(system.wdm, "band_specs") and system.wdm.band_specs:
        for name, slc in system.wdm._band_slices.items():
            spec = system.wdm.band_specs.get(name)
            band_dbm = launch_dbm[slc]
            lg.info(
                f"Band {name}: {slc.stop - slc.start} ch, "
                f"launch_dBm min/med/max = {float(np.min(band_dbm)):.2f} / "
                f"{float(np.median(band_dbm)):.2f} / {float(np.max(band_dbm)):.2f}"
            )
    signal_power = _resolve_signal_power(system, profile_path, launch_powers)

    # TD NLIN using existing pipeline
    ccfs = collision_coeffs_system_uwb(
        system,
        ipulse=1,
        recompute=recompute_td,
        profile_path=profile_path,
    )
    if not np.all(np.isfinite(ccfs)):
        lg.warning("Collision coefficients contain non-finite values; replacing NaNs with 0.")
        ccfs = np.nan_to_num(ccfs, nan=0.0, posinf=0.0, neginf=0.0)
    nlin_td = total_nlin_uwb(
        system,
        ccfs,
        use_kappa=True,
        use_x_mode=True,
        launch_powers_w=launch_powers,
        cache_path=_nlin_cache_path(profile_path, use_kappa=True, use_x_mode=True),
        recompute=recompute_td,
    )
    if not np.all(np.isfinite(nlin_td)):
        raise ValueError(
            "TD NLIN contains non-finite values. Re-run with recompute_td=True."
        )
    nlin_td_flat = np.asarray(nlin_td, dtype=float).reshape(-1)
    td_modulations = None
    try:
        qam_orders = [16, 64, 256]
        const_pref, sum_a, sum_b = _td_modulation_components(
            system,
            ccfs,
            launch_powers,
            use_kappa=True,
            use_x_mode=True,
        )
        td_modulations = {}
        for order in qam_orders:
            mu0 = _qam_mu0(order)
            nlin_mod = const_pref * (mu0 * sum_a + sum_b)
            td_modulations[f"{order}-QAM"] = np.asarray(nlin_mod, dtype=float).reshape(-1)
    except Exception as exc:
        lg.warning(f"TD modulation sweep skipped: {exc}")
    _save_nlin_csv(
        Path("results") / f"total_nlin_{Path(profile_path).stem}_k1_x1.csv",
        freqs,
        nlin_td_flat,
        signal_power,
    )

    # PCFM NLIN for loss cases (Fig. 14c style)
    if include_lumped_losses:
        loss_cases = {
            "no_loss": None,
            "loss_1db_10km": [(10e3, 1.0)],
            "loss_2db_5km_0p5db_97km": [(5e3, 2.0), (97e3, 0.5)],
        }
    else:
        loss_cases = {"no_loss": None}
    cfg = PcfmConfig(
        degree=9,
        include_mci=False,
        use_numeric_sci=True,
        use_numeric_xci=pcfm_numeric_xci,
    )

    gsnr_pcfm = {}
    gsnr_gn = {} if compute_gn else None
    gsnr_gn_direct = {} if compute_gn_direct else None
    nlin_pcfm = {}
    nlin_gn = {} if compute_gn else None
    nlin_gn_direct = {} if compute_gn_direct else None
    nlin_pcfm_xci = {}
    nlin_gn_xci = {} if compute_gn else None
    nlin_gn_direct_xci = {} if compute_gn_direct else None

    for label, losses in loss_cases.items():
        pcfm_path = Path("results") / f"total_nlin_{Path(profile_path).stem}_pcfm_{label}.npy"
        nlin_pcfm_arr, _, nlin_pcfm_xci_arr = _load_or_compute_pcfm(
            system,
            profile_path=profile_path,
            launch_powers_w=launch_powers,
            output_path=pcfm_path,
            cfg=cfg,
            lumped_losses=losses,
            recompute=recompute_pcfm,
            return_components=True,
        )
        nlin_pcfm_flat = np.asarray(nlin_pcfm_arr, dtype=float).reshape(-1)
        nlin_pcfm_xci_flat = np.asarray(nlin_pcfm_xci_arr, dtype=float).reshape(-1)
        _save_nlin_csv(
            Path("results") / f"total_nlin_{Path(profile_path).stem}_pcfm_{label}.csv",
            freqs,
            nlin_pcfm_flat,
            signal_power,
        )
        _save_nlin_csv(
            Path("results") / f"total_nlin_{Path(profile_path).stem}_pcfm_{label}_xci.csv",
            freqs,
            nlin_pcfm_xci_flat,
            signal_power,
        )
        gsnr_pcfm[label] = 10.0 * np.log10(signal_power / np.maximum(nlin_pcfm_flat, 1e-18))
        nlin_pcfm[label] = nlin_pcfm_flat
        nlin_pcfm_xci[label] = nlin_pcfm_xci_flat

        if compute_gn:
            gn_path = Path("results") / f"total_nlin_{Path(profile_path).stem}_gn_{label}.npy"
            nlin_gn_arr, _, nlin_gn_xci_arr = _load_or_compute_gn(
                system,
                profile_path=profile_path,
                launch_powers_w=launch_powers,
                output_path=gn_path,
                lumped_losses=losses,
                recompute=recompute_gn,
                return_components=True,
            )
            nlin_gn_flat = np.asarray(nlin_gn_arr, dtype=float).reshape(-1)
            nlin_gn_xci_flat = np.asarray(nlin_gn_xci_arr, dtype=float).reshape(-1)
            _save_nlin_csv(
                Path("results") / f"total_nlin_{Path(profile_path).stem}_gn_{label}.csv",
                freqs,
                nlin_gn_flat,
                signal_power,
            )
            _save_nlin_csv(
                Path("results") / f"total_nlin_{Path(profile_path).stem}_gn_{label}_xci.csv",
                freqs,
                nlin_gn_xci_flat,
                signal_power,
            )
            if gsnr_gn is not None:
                gsnr_gn[label] = 10.0 * np.log10(signal_power / np.maximum(nlin_gn_flat, 1e-18))
            if nlin_gn is not None:
                nlin_gn[label] = nlin_gn_flat
            if nlin_gn_xci is not None:
                nlin_gn_xci[label] = nlin_gn_xci_flat

        if compute_gn_direct:
            gn_direct_path = Path("results") / f"total_nlin_{Path(profile_path).stem}_gn_direct_{label}.npy"
            nlin_gn_direct_arr, _, nlin_gn_direct_xci_arr = _load_or_compute_gn_direct(
                system,
                profile_path=profile_path,
                launch_powers_w=launch_powers,
                output_path=gn_direct_path,
                lumped_losses=losses,
                recompute=recompute_gn_direct,
                return_components=True,
            )
            nlin_gn_direct_flat = np.asarray(nlin_gn_direct_arr, dtype=float).reshape(-1)
            nlin_gn_direct_xci_flat = np.asarray(nlin_gn_direct_xci_arr, dtype=float).reshape(-1)
            _save_nlin_csv(
                Path("results") / f"total_nlin_{Path(profile_path).stem}_gn_direct_{label}.csv",
                freqs,
                nlin_gn_direct_flat,
                signal_power,
            )
            _save_nlin_csv(
                Path("results") / f"total_nlin_{Path(profile_path).stem}_gn_direct_{label}_xci.csv",
                freqs,
                nlin_gn_direct_xci_flat,
                signal_power,
            )
            if gsnr_gn_direct is not None:
                gsnr_gn_direct[label] = 10.0 * np.log10(
                    signal_power / np.maximum(nlin_gn_direct_flat, 1e-18)
                )
            if nlin_gn_direct is not None:
                nlin_gn_direct[label] = nlin_gn_direct_flat
            if nlin_gn_direct_xci is not None:
                nlin_gn_direct_xci[label] = nlin_gn_direct_xci_flat

    gsnr_td = 10.0 * np.log10(signal_power / np.maximum(nlin_td_flat, 1e-18))
    if plot:
        plot_poggiolini_gsnr(
            freqs_hz=freqs,
            gsnr_td=gsnr_td,
            gsnr_pcfm=gsnr_pcfm,
            gsnr_gn=gsnr_gn,
            out_path=Path("media") / "poggiolini_fig14c.pdf",
            gsnr_gn_direct=gsnr_gn_direct,
        )
        plot_poggiolini_nlin_power(
            freqs_hz=freqs,
            nlin_td_w=nlin_td_flat,
            nlin_pcfm_w=nlin_pcfm,
            nlin_gn_w=nlin_gn,
            nlin_td_mod_w=td_modulations,
            nlin_pcfm_xci_w=nlin_pcfm_xci,
            nlin_gn_xci_w=nlin_gn_xci,
            nlin_gn_direct_w=nlin_gn_direct,
            nlin_gn_direct_xci_w=nlin_gn_direct_xci,
            out_path=Path("media") / "poggiolini_nlin_power.pdf",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Poggiolini PCFM/TD workflow runner")
    parser.add_argument("--recompute", action="store_true", help="Recompute Raman/TD/PCFM outputs.")
    parser.add_argument("--compute-gn", action="store_true", help="Compute numeric GN (SCI+XCI).")
    parser.add_argument("--compute-gn-direct", action="store_true", help="Compute direct GN using p(z) samples.")
    parser.add_argument("--recompute-gn", action="store_true", help="Force recompute of numeric GN outputs.")
    parser.add_argument("--recompute-gn-direct", action="store_true", help="Force recompute of direct GN outputs.")
    parser.add_argument("--pcfm-numeric-xci", action="store_true", help="Force PCFM to use numeric XCI.")
    args = parser.parse_args()

    run_poggiolini_workflow(
        cfg_path=Path("./input/poggiolini_struct.toml"),
        profile_path=Path("results/poggiolini_power_profiles.npy"),
        launch_csv_path=Path("results/poggiolini_launch_power.csv"),
        recompute_profiles=args.recompute,
        recompute_td=args.recompute,
        recompute_pcfm=args.recompute,
        recompute_gn=args.recompute or args.recompute_gn,
        recompute_gn_direct=args.recompute or args.recompute_gn_direct,
        use_profile_launch_powers=True,
        compute_gn=args.compute_gn,
        compute_gn_direct=args.compute_gn_direct,
        pcfm_numeric_xci=args.pcfm_numeric_xci,
        include_lumped_losses=False,
        plot=True,
    )
