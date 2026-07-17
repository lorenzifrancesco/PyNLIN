from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger as lg

from pynlin.utils import BaseModel, ConfigDict, EXTRA_IGNORE_CONFIG, _toml_load


def _fmt_c(val: complex) -> str:
    return f"{val.real:.3e}{val.imag:+.3e}j"


class FwmEstimationConfig(BaseModel):
    config: str = "input/studies.toml"
    out: Optional[str] = None
    f_min_ghz: Optional[float] = None
    f_max_ghz: Optional[float] = None
    f_points: Optional[int] = None
    f_offset_ghz: float = 0.0
    f_both_signs: bool = False
    f_grid_spacing_ghz: Optional[float] = None
    target_wl_nm: float = 1310.0
    center_wl_nm: Optional[float] = None
    signal_wl_nm: Optional[float] = None
    pump_wl_nm: Optional[float] = None
    pump_power_dbm: Optional[float] = None
    rho_pol: float = 2 / 3
    rho_model: str = "raman"
    logy: Optional[bool] = None
    normalize_leff: Optional[bool] = None
    db: Optional[bool] = None
    db_floor: Optional[float] = None
    band: str = "O"
    animate_band: bool = False
    animate_frames: int = 30
    animate_fps: int = 4
    animate_dpi: int = 200

    if ConfigDict:
        model_config = ConfigDict(extra="ignore")
    else:
        class Config:
            extra = "ignore"


def load_fwm_config(path: Optional[str | Path] = None) -> FwmEstimationConfig:
    cfg_path = Path(path) if path is not None else Path("input/fwm_estimation.toml")
    if not cfg_path.exists():
        lg.warning("FWM config file not found at {}; using defaults.", cfg_path)
        return FwmEstimationConfig()
    data = _toml_load(cfg_path)
    if isinstance(data, dict) and "fwm_estimation" in data:
        data = data["fwm_estimation"]
    if isinstance(data, dict):
        omega_keys = {
            "omega_min_ghz": "f_min_ghz",
            "omega_max_ghz": "f_max_ghz",
            "omega_offset_ghz": "f_offset_ghz",
            "omega_points": "f_points",
            "omega_both_signs": "f_both_signs",
        }
        converted = False
        for old_key, new_key in omega_keys.items():
            if old_key in data and new_key not in data:
                val = data[old_key]
                if old_key in ("omega_min_ghz", "omega_max_ghz", "omega_offset_ghz") and val is not None:
                    val = float(val) / (2.0 * np.pi)
                data[new_key] = val
                converted = True
        if converted:
            lg.warning("Deprecated omega_* config keys detected; converted to f_* (frequency GHz).")
    cfg = FwmEstimationConfig(**(data or {}))
    lg.info("Loaded FWM config from {}.", cfg_path)
    return cfg


def rho_undepleted(
    mu: np.ndarray,
    A: complex,
    alpha_p: float,
    length: float,
    *,
    log_points: bool = False,
    dps: int = 80,
) -> np.ndarray:
    """Compute rho from the closed-form undepleted-pump expression."""
    try:
        import mpmath as mp
    except ImportError as exc:
        raise ImportError(
            "mpmath is required for complex incomplete Gamma evaluation. "
            "Install it or provide real-valued inputs."
        ) from exc
    mp.mp.dps = dps
    lg.debug("mpmath precision set to {dps} digits.", dps=mp.mp.dps)
    mu_arr = np.asarray(mu, dtype=complex)
    A_mp = mp.mpc(A)
    exp_factor = mp.e ** (2.0 * alpha_p * length)
    lg.debug(
        "rho inputs: alpha_p={alpha_p:.3e}, L={length:.3e}, A={A}, exp_factor={exp_factor}",
        alpha_p=alpha_p,
        length=length,
        A=_fmt_c(complex(A)),
        exp_factor=str(exp_factor),
    )
    out = np.empty(mu_arr.shape, dtype=float)
    fail_count = 0
    for idx, mu_val in np.ndenumerate(mu_arr):
        mu_mp = mp.mpc(mu_val)
        if log_points:
            lg.debug(
                "gammainc call idx={} mu={} A={} Aexp={}",
                idx,
                _fmt_c(complex(mu_val)),
                _fmt_c(complex(A_mp)),
                _fmt_c(complex(A_mp * exp_factor)),
            )
        try:
            exp_mu = mp.e ** (-2.0 * alpha_p * length * mu_mp)
            term = mp.expint(1.0 + mu_mp, A_mp) - exp_mu * mp.expint(1.0 + mu_mp, A_mp * exp_factor)
            rho_val = mp.e ** (A_mp) / (2.0 * alpha_p) * abs(term) ** 2
            out_val = float(mp.re(rho_val))
            out[idx] = out_val
            if log_points and not np.isfinite(out_val):
                lg.debug(
                    "non-finite rho at idx={} mu={} rho={}",
                    idx,
                    _fmt_c(complex(mu_val)),
                    _fmt_c(complex(rho_val)),
                )
        except Exception:
            fail_count += 1
            if log_points:
                lg.debug("gammainc failed at idx={} mu={}", idx, _fmt_c(complex(mu_val)))
            out[idx] = np.nan
    if fail_count:
        lg.warning("gammainc failed for {} of {} points; values set to NaN.", fail_count, out.size)
    return out


def rho_attenuation(alpha: float, delta_beta: np.ndarray, length: float) -> np.ndarray:
    """Attenuation-only efficiency with full dispersion-based delta_beta."""
    alpha = float(alpha)
    delta_beta = np.asarray(delta_beta, dtype=float)
    denom = 2.0 * alpha - 1j * delta_beta
    numer = 1.0 - np.exp(-(2.0 * alpha - 1j * delta_beta) * length)
    return np.abs(numer / denom) ** 2


def effective_length(alpha: float, length: float) -> float:
    """Return the effective interaction length for power attenuation."""
    alpha = float(alpha)
    length = float(length)
    if alpha <= 0.0:
        return length
    return (1.0 - np.exp(-2.0 * alpha * length)) / (2.0 * alpha)
