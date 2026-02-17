from pathlib import Path

import numpy as np
from loguru import logger as lg
from scipy.constants import c

from pynlin.nlin.pcfm_gn import load_signal_profiles
from pynlin.system import System
from pynlin.utils import dBm2watt

from .config import PROFILE_MAX_W

try:
    from analysis.uwb_nlin import _load_profile_launch_powers
except ModuleNotFoundError:
    from uwb_nlin import _load_profile_launch_powers  # type: ignore[no-redef]


def _profile_needs_recompute(
    profile_path: Path | str,
    max_power_w: float = PROFILE_MAX_W,
) -> bool:
    path = Path(profile_path)
    if not path.exists():
        return True
    try:
        data = np.load(path, allow_pickle=True)
        if isinstance(data, np.lib.npyio.NpzFile):
            signal = data.get("signal_sol")
        else:
            if isinstance(data, np.ndarray) and data.shape == ():
                data = data.item()
            signal = data.get("signal_sol") if isinstance(data, dict) else None
    except Exception as exc:
        lg.warning(f"Failed to read profile for validation: {exc}")
        return True
    if signal is None:
        lg.warning("Profile missing signal_sol; recompute required.")
        return True
    signal = np.asarray(signal, dtype=float)
    if not np.all(np.isfinite(signal)):
        lg.warning("Profile contains non-finite values; recompute required.")
        return True
    max_val = float(np.nanmax(signal))
    if max_val > max_power_w:
        lg.warning(
            f"Profile max power {max_val:.2e} W exceeds {max_power_w:.2e} W; "
            "recompute required."
        )
        return True
    return False


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
    n_z = max(int(n_z), 2)
    z_axis = np.linspace(0.0, float(system.fiber_length), n_z)
    if launch_powers_w is None:
        launch_powers_w = np.ones(n_channels, dtype=float)
    else:
        launch_powers_w = np.asarray(launch_powers_w, dtype=float).reshape(-1)
        if launch_powers_w.size != n_channels:
            raise ValueError(
                f"Flat profile launch powers size {launch_powers_w.size} != "
                f"n_channels {n_channels}"
            )
    signal_sol = np.tile(launch_powers_w[None, :], (n_z, 1))
    payload = {
        "signal_sol": signal_sol,
        "signal_wavelengths": (c / freqs),
        "z": z_axis,
    }
    np.save(path, payload)
    lg.info(f"Saved flat SPP profile to {path}")


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
        lg.warning(
            f"CSV {csv_path} missing required headers frequency_THz, launch_power_dbm."
        )
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


def _resolve_launch_powers(
    system: System,
    profile_path: Path | str | None,
    launch_csv_path: Path | str | None,
    use_profile: bool = True,
) -> np.ndarray:
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
        if (
            not np.all(np.isfinite(launch))
            or np.any(launch <= 0)
            or np.max(launch) > PROFILE_MAX_W
        ):
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


def _resolve_signal_power(
    system: System,
    profile_path: Path | str | None,
    launch_override: np.ndarray | None,
) -> np.ndarray:
    """Return per-channel end-of-fiber signal power (W) for GSNR/normalization."""
    freqs = system.wdm.frequency_grid()
    if profile_path is not None:
        try:
            sig_ch_z, _ = load_signal_profiles(profile_path, system)
            sig_ch_z = np.asarray(sig_ch_z, dtype=float)
            if sig_ch_z.ndim == 2 and sig_ch_z.shape[0] == freqs.size:
                out_power = sig_ch_z[:, -1]
            else:
                out_power = None
            if (
                out_power is not None
                and out_power.size == freqs.size
                and np.all(np.isfinite(out_power))
            ):
                lg.info("Using end-of-fiber signal power from profile for GSNR/normalization.")
                return out_power
            lg.warning("Profile signal power shape mismatch; falling back to launch powers.")
        except Exception as exc:
            lg.warning(f"Failed to read end-of-fiber signal power from profile: {exc}")
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
    if nlin_flat.size != freqs_hz.size:
        raise ValueError(f"NLIN array size {nlin_flat.size} != freq size {freqs_hz.size}")
    signal_power_w = np.asarray(signal_power_w, dtype=float).reshape(-1)
    if signal_power_w.size != freqs_hz.size:
        raise ValueError(
            f"Signal power size {signal_power_w.size} != freq size {freqs_hz.size}"
        )
    denom = np.maximum(nlin_flat, 1e-18)
    gsnr_db = 10.0 * np.log10(signal_power_w / denom)
    data = np.column_stack([freqs_hz * 1e-12, nlin_flat, gsnr_db])
    header = "frequency_THz,nlin_W,gsnr_nli_dB"
    np.savetxt(out, data, delimiter=",", header=header, comments="")
    lg.success(f"Saved NLIN CSV to {out}")
