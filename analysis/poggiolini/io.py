from pathlib import Path

import numpy as np
from loguru import logger as lg
from scipy.constants import c

from analysis.uwb_nlin import _load_profile_launch_powers
from pynlin.nlin.pcfm_gn import load_signal_profiles
from pynlin.system import System
from pynlin.utils import dBm2watt

from .config import PROFILE_MAX_W


def _profile_needs_recompute(
    profile_path: Path | str,
    max_power_w: float = PROFILE_MAX_W,
) -> bool:
    _ = max_power_w
    return not Path(profile_path).exists()


def _write_flat_profile(
    profile_path: Path | str,
    system: System,
    launch_powers_w: np.ndarray | None = None,
    n_z: int = 200,
) -> None:
    """Write a synthetic flat power profile with p(z)=1 for all channels."""
    path = Path(profile_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    freqs = system.wdm.frequency_grid()
    n_channels = freqs.size
    z_axis = np.linspace(0.0, float(system.fiber_length), max(int(n_z), 2))
    launch = (
        np.ones(n_channels, dtype=float)
        if launch_powers_w is None
        else np.asarray(launch_powers_w, dtype=float).reshape(-1)
    )
    signal_sol = np.tile(launch[None, :], (z_axis.size, 1))
    payload = {
        "signal_sol": signal_sol,
        "signal_wavelengths": c / freqs,
        "z": z_axis,
    }
    np.save(path, payload)
    lg.info(f"Saved flat SPP profile to {path}")


def _load_launch_powers_csv(path: Path | str, freqs_hz: np.ndarray) -> np.ndarray:
    """Load per-channel launch powers from a CSV with frequency_THz,launch_power_dbm."""
    data = np.genfromtxt(Path(path), delimiter=",", names=True)
    freqs_thz = np.atleast_1d(np.asarray(data["frequency_THz"], dtype=float))
    powers_dbm = np.atleast_1d(np.asarray(data["launch_power_dbm"], dtype=float))
    freqs_csv = freqs_thz * 1e12
    order = np.argsort(freqs_csv)
    freqs_csv = freqs_csv[order]
    powers_dbm = powers_dbm[order]
    powers_interp_dbm = np.interp(freqs_hz, freqs_csv, powers_dbm)
    return dBm2watt(powers_interp_dbm)


def _resolve_launch_powers(
    system: System,
    profile_path: Path | str | None,
    launch_csv_path: Path | str | None,
    use_profile: bool = True,
) -> np.ndarray:
    """Resolve per-channel launch powers (W)."""
    freqs = system.wdm.frequency_grid()

    if use_profile:
        if profile_path is None:
            raise FileNotFoundError("Profile path required when use_profile=True.")
        launch_from_profile = _load_profile_launch_powers(profile_path, system.n_channels)
        if launch_from_profile is None:
            raise FileNotFoundError("Launch powers missing in Raman profile.")
        launch_from_profile = np.asarray(launch_from_profile, dtype=float).reshape(-1)
        lg.info("Using launch powers from Raman profile.")
        return launch_from_profile

    if launch_csv_path is not None:
        lg.info(f"Using launch powers from {launch_csv_path}.")
        return np.asarray(_load_launch_powers_csv(launch_csv_path, freqs), dtype=float).reshape(-1)

    if hasattr(system.wdm, "band_specs") and system.wdm.band_specs:
        launch = np.zeros_like(freqs, dtype=float)
        for name, slc in system.wdm._band_slices.items():
            spec = system.wdm.band_specs.get(name)
            power_dbm = spec.launch_power_dbm if spec else system.launch_power
            launch[slc] = dBm2watt(power_dbm if power_dbm is not None else -5.0)
        return launch

    power_dbm = system.launch_power if system.launch_power is not None else -5.0
    return np.full_like(freqs, dBm2watt(power_dbm), dtype=float)


def _resolve_signal_power(
    system: System,
    profile_path: Path | str | None,
    launch_override: np.ndarray | None,
) -> np.ndarray:
    """Return per-channel end-of-fiber signal power (W) for GSNR/normalization."""
    freqs = system.wdm.frequency_grid()
    if profile_path is not None:
        sig_ch_z, _ = load_signal_profiles(profile_path, system)
        return np.asarray(sig_ch_z, dtype=float)[:, -1]
    if launch_override is not None:
        return np.asarray(launch_override, dtype=float).reshape(-1)
    power_dbm = system.launch_power if system.launch_power is not None else -5.0
    return np.full_like(freqs, dBm2watt(power_dbm), dtype=float)


def _save_nlin_csv(
    path: Path | str,
    freqs_hz: np.ndarray,
    nlin_w: np.ndarray,
    signal_power_w: np.ndarray,
) -> None:
    """Save NLIN + GSNR to CSV."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    nlin_flat = np.asarray(nlin_w, dtype=float).reshape(-1)
    signal_power_w = np.asarray(signal_power_w, dtype=float).reshape(-1)
    denom = np.maximum(nlin_flat, 1e-18)
    gsnr_db = 10.0 * np.log10(signal_power_w / denom)
    data = np.column_stack([freqs_hz * 1e-12, nlin_flat, gsnr_db])
    header = "frequency_THz,nlin_W,gsnr_nli_dB"
    np.savetxt(out, data, delimiter=",", header=header, comments="")
    lg.success(f"Saved NLIN CSV to {out}")
