from typing import Tuple
from pathlib import Path
import os

import numpy as np

from pynlin.log_init import init_logging
from loguru import logger as lg
from pynlin.system import System

init_logging()


def _load_signal_solution(path: Path):
    """Return (data, signal_solution, z_axis) from supported profile files."""
    if path.suffix == ".npz":
        lg.info(f"[raman_integrals_uwb] Loading profile from {path}")
        data = np.load(path, allow_pickle=True)
        sig = data.get("signal_solution")
        z = data.get("z")
        return data, sig, z
    lg.info(f"[raman_integrals_uwb] Loading profile from {path}")
    data = np.load(path, allow_pickle=True).item()
    sig = data.get("signal_sol")
    z = data.get("z")
    return data, sig, z


def load_fB(system: System, profile_path: Path | str | None = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, callable, callable]:
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
    source_path = None
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
        source_path = path
        break
    if sig is None:
        raise FileNotFoundError("No suitable Raman profile file found for fB computation.")

    # lg.info(f"[raman_integrals_uwb] Using signal profiles from {source_path}")

    signal_powers = np.array(sig, dtype=float)
    if signal_powers.ndim == 2:
        # (z, channels) -> add mode axis
        signal_powers = signal_powers[:, :, None]
    if signal_powers.ndim == 3:
        # (z, channels, modes) -> swap to (z, modes, channels)
        signal_powers = np.swapaxes(signal_powers, 1, 2)
    else:
        raise ValueError(f"Unexpected signal_solution shape: {signal_powers.shape}")

    fB = signal_powers / signal_powers[0, :, :]
    assert np.all(fB[0, :, :] == 1.0)
    fB_max = np.max(fB, axis=(1, 2))
    fB_min = np.min(fB, axis=(1, 2))
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
    if fB.size:
        fB_min = float(np.nanmin(fB))
        fB_max = float(np.nanmax(fB))
        fB_mean = float(np.nanmean(fB))
        lg.debug(
            f"[raman_integral] regime={regime} fB shape={fB.shape} "
            f"min={fB_min:.3e} max={fB_max:.3e} mean={fB_mean:.3e}"
        )
    z_axis = np.linspace(0, system.fiber_length, len(fB))
    dz = z_axis[1] - z_axis[0]
    # lg.info(f"max fB   = {np.max(fB):.3e}, min fB = {np.min(fB):.3e}")
    if regime == "LO":
        return (np.sum(fB) * dz / system.fiber_length) ** 2
    return np.sum(fB ** 2) * dz / system.fiber_length


def load_raman_integral_extremes(system: System,
                                 profile_path: Path | str | None = None) -> Tuple[float, float, float, float]:
    """Return LO/HI Raman integrals for minimum and maximum gain envelopes."""
    _, fB_min, fB_max, _, _ = load_fB(system, profile_path=profile_path)
    r_lo_min = raman_integral(system, "LO", fB_min)
    r_lo_max = raman_integral(system, "LO", fB_max)
    r_hi_min = raman_integral(system, "HI", fB_min)
    r_hi_max = raman_integral(system, "HI", fB_max)
    return r_lo_min, r_lo_max, r_hi_min, r_hi_max


__all__ = ["load_fB", "raman_integral", "load_raman_integral_extremes"]
