"""Closed-form undepleted-pump Raman helpers."""

from __future__ import annotations

import numpy as np

__all__ = [
    "db_per_km_to_np_per_m",
    "effective_raman_gain",
    "pump_power_for_flat_signal",
    "pump_power_coprop",
    "pump_power_counterprop",
    "signal_power_undepleted_coprop",
    "signal_power_undepleted_counterprop",
]


def db_per_km_to_np_per_m(val_db_km: float) -> float:
    """Convert attenuation from dB/km (power) to Np/m."""
    return (val_db_km * np.log(10.0) / 10.0) / 1e3


def effective_raman_gain(g_r: float, rho_pol: float, a_eff: float) -> float:
    """Return g_eff = rho_pol * g_r / a_eff."""
    return rho_pol * g_r / a_eff


def pump_power_for_flat_signal(alpha_s: float, alpha_p: float, g_eff: float, length: float) -> float:
    """Return pump power that yields P_s(L) = P_s(0) under undepleted pump."""
    if np.isclose(g_eff, 0.0):
        raise ValueError("g_eff must be non-zero for flat-signal pump power.")
    if np.isclose(alpha_p, 0.0):
        return alpha_s / g_eff
    denom = g_eff * (-np.expm1(-alpha_p * length))
    if np.isclose(denom, 0.0):
        raise ValueError("Invalid parameters for flat-signal pump power.")
    return alpha_s * length * alpha_p / denom


def pump_power_coprop(z: np.ndarray, pump_in: float, alpha_p: float) -> np.ndarray:
    """Co-propagating pump profile Pp(z) = Pp_in * exp(-alpha_p z)."""
    z = np.asarray(z, dtype=float)
    return pump_in * np.exp(-alpha_p * z)


def pump_power_counterprop(z: np.ndarray, pump_in: float, alpha_p: float, length: float) -> np.ndarray:
    """Counter-propagating pump profile launched at z=L."""
    z = np.asarray(z, dtype=float)
    return pump_in * np.exp(-alpha_p * (length - z))


def signal_power_undepleted_coprop(
    z: np.ndarray,
    signal_in: float,
    alpha_s: float,
    g_eff: float,
    pump_in: float,
    alpha_p: float,
) -> np.ndarray:
    """Closed-form signal power with an undepleted co-propagating pump."""
    z = np.asarray(z, dtype=float)
    if np.isclose(alpha_p, 0.0):
        integral = g_eff * pump_in * z
    else:
        integral = (g_eff * pump_in / alpha_p) * (-np.expm1(-alpha_p * z))
    return signal_in * np.exp(-alpha_s * z + integral)


def signal_power_undepleted_counterprop(
    z: np.ndarray,
    signal_in: float,
    alpha_s: float,
    g_eff: float,
    pump_in: float,
    alpha_p: float,
    length: float,
) -> np.ndarray:
    """Closed-form signal power with an undepleted counter-propagating pump."""
    z = np.asarray(z, dtype=float)
    if np.isclose(alpha_p, 0.0):
        integral = g_eff * pump_in * z
    else:
        integral = (g_eff * pump_in * np.exp(-alpha_p * length) / alpha_p) * np.expm1(alpha_p * z)
    return signal_in * np.exp(-alpha_s * z + integral)
