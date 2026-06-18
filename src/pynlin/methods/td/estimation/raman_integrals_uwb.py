"""Raman profile and integral helpers for UWB profile-driven NLIN workflows."""

from typing import Tuple
from pathlib import Path

import numpy as np

from pynlin.log_init import init_logging
from loguru import logger as lg
from pynlin.system import System

init_logging()


def _load_signal_solution(path: Path):
    """Return (data, signal_solution, z_axis) from supported profile files."""
    lg.debug(f"[raman_integrals_uwb] Loading profile from {path}")
    payload = np.load(path, allow_pickle=True)
    if isinstance(payload, np.lib.npyio.NpzFile):
        data = payload
    elif isinstance(payload, np.ndarray) and payload.shape == ():
        data = payload.item()
    elif isinstance(payload, dict):
        data = payload
    else:
        raise TypeError(f"Unexpected Raman profile payload type: {type(payload)}")

    sig = None
    if hasattr(data, "get"):
        for key in ("signal_sol", "signal_solution", "signal_power"):
            if key in data:
                sig = data.get(key)
                if sig is not None:
                    break
    z = data.get("z") if hasattr(data, "get") else None
    return data, sig, z


def _coerce_signal_powers(signal_powers: np.ndarray, z_axis, system: System) -> np.ndarray:
    """Normalize supported profile layouts to ``(z, modes, channels)``."""
    arr = np.asarray(signal_powers, dtype=float)
    z_size = None if z_axis is None else np.asarray(z_axis).size
    n_channels = getattr(system, "n_channels", None)

    if arr.ndim == 2:
        if z_size is not None:
            if arr.shape[0] == z_size:
                sig_z_ch = arr
            elif arr.shape[1] == z_size:
                sig_z_ch = arr.T
            else:
                raise ValueError(f"Unrecognized signal_solution shape {arr.shape} for z size {z_size}.")
        elif n_channels is not None and arr.shape[0] == n_channels:
            sig_z_ch = arr.T
        else:
            sig_z_ch = arr
        return sig_z_ch[:, None, :]

    if arr.ndim == 3:
        if z_size is not None and arr.shape[0] == z_size:
            if n_channels is not None and arr.shape[2] == n_channels:
                return arr
            return np.swapaxes(arr, 1, 2)
        if z_size is not None and arr.shape[1] == z_size:
            return np.transpose(arr, (1, 2, 0))
        raise ValueError(f"Unrecognized signal_solution shape {arr.shape} for z size {z_size}.")

    raise ValueError(f"Unexpected signal_solution shape: {arr.shape}")


def load_fB(
    system: System,
    profile_path: Path | str | None = None,
    profile_channel_idx: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, callable, callable]:
    """Load normalized Raman gain profiles fB(z) from a Jiang-based profile if available."""
    if profile_path is not None:
        candidates = [Path(profile_path)]
    else:
        candidates = [
            Path("results/uwb_power_profiles.npy"),
            Path("results/dummy_power_profiles.npy"),
        ]
    sig = None
    z_axis = None
    data = None
    for path in candidates:
        if not path.exists():
            if profile_path is not None:
                raise FileNotFoundError(f"Requested Raman profile not found: {path}")
            continue
        data, sig, z_axis = _load_signal_solution(path)
        if sig is None:
            if profile_path is not None:
                raise FileNotFoundError(f"Requested Raman profile missing signal data: {path}")
            continue
        break
    if sig is None:
        raise FileNotFoundError("No suitable Raman profile file found for fB computation.")

    # lg.info(f"[raman_integrals_uwb] Using signal profiles from {source_path}")

    signal_powers = _coerce_signal_powers(sig, z_axis, system)

    fB = signal_powers / signal_powers[0, :, :]
    assert np.all(fB[0, :, :] == 1.0)
    if profile_channel_idx is None:
        fB_for_extrema = fB
        fB_max = np.max(fB_for_extrema, axis=(1, 2))
        fB_min = np.min(fB_for_extrema, axis=(1, 2))
    else:
        idx = int(profile_channel_idx)
        if idx < 0 or idx >= fB.shape[2]:
            raise IndexError(
                f"profile_channel_idx={idx} out of bounds for profile with {fB.shape[2]} channels"
            )
        fB_for_extrema = fB[:, :, idx]
        fB_max = np.max(fB_for_extrema, axis=1)
        fB_min = np.min(fB_for_extrema, axis=1)
    if z_axis is None:
        z_axis = np.linspace(0, system.fiber_length, len(fB_max))

    coeffs_max = np.polyfit(z_axis, fB_max, 6)
    coeffs_min = np.polyfit(z_axis, fB_min, 6)

    def fB_max_function(z):
        return np.polyval(coeffs_max, z)

    def fB_min_function(z):
        return np.polyval(coeffs_min, z)

    return fB, fB_min, fB_max, fB_min_function, fB_max_function


def raman_integral(system: System,
                   regime: str,
                   fB: np.ndarray):
    """Compute LO or HI Raman integrals for a given longitudinal gain profile."""
    fB = np.asarray(fB, dtype=np.float64)
    if fB.shape[0] < 2:
        raise ValueError("Raman integral requires at least two longitudinal samples.")
    if fB.size:
        fB_min = float(np.nanmin(fB))
        fB_max = float(np.nanmax(fB))
        fB_mean = float(np.nanmean(fB))
        lg.trace(
            f"[raman_integral] regime={regime} fB shape={fB.shape} "
            f"min={fB_min:.3e} max={fB_max:.3e} mean={fB_mean:.3e}"
        )
    z_axis = np.linspace(0, system.fiber_length, len(fB))
    regime = regime.upper()
    if regime == "LO":
        return (np.trapezoid(fB, z_axis, axis=0) / system.fiber_length) ** 2
    if regime == "HI":
        return np.trapezoid(fB ** 2, z_axis, axis=0) / system.fiber_length
    raise ValueError(f"Unsupported Raman integral regime: {regime!r}")


def load_raman_integral_extremes(system: System,
                                 profile_path: Path | str | None = None,
                                 profile_channel_idx: int | None = None) -> Tuple[float, float, float, float]:
    """Return LO/HI Raman integrals for minimum and maximum gain envelopes."""
    _, fB_min, fB_max, _, _ = load_fB(
        system,
        profile_path=profile_path,
        profile_channel_idx=profile_channel_idx,
    )
    r_lo_min = raman_integral(system, "LO", fB_min)
    r_lo_max = raman_integral(system, "LO", fB_max)
    r_hi_min = raman_integral(system, "HI", fB_min)
    r_hi_max = raman_integral(system, "HI", fB_max)
    return r_lo_min, r_lo_max, r_hi_min, r_hi_max


__all__ = ["load_fB", "raman_integral", "load_raman_integral_extremes"]
