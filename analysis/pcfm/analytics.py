from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
from scipy.constants import c
from scipy.integrate import cumulative_trapezoid
from scipy.special import sici

from pynlin.nlin import pcfm_gn
from pynlin.system import System


def _xci_log_term(delta_abs_hz: float, bandwidth_hz: float, order: int | None = None) -> float:
    """Return the exact or truncated log term in the flat-profile XCI kernel.

    For ``|Delta f| > B/2``:

    log((|Delta f| + B/2) / (|Delta f| - B/2)) = 2 * atanh(B / (2 |Delta f|))

    Using ``atanh`` is numerically cleaner than forming the ratio directly when
    ``B / |Delta f|`` is small.
    """
    delta_abs_hz = float(abs(delta_abs_hz))
    bandwidth_hz = float(abs(bandwidth_hz))
    if delta_abs_hz <= bandwidth_hz / 2.0:
        raise ValueError("Closed-form XCI requires |Delta f| > B/2.")

    x = bandwidth_hz / (2.0 * delta_abs_hz)
    if order is None:
        return float(2.0 * np.arctanh(x))
    if order == 1:
        return float(2.0 * x)
    if order == 3:
        return float(2.0 * (x + x**3 / 3.0))
    if order == 5:
        return float(2.0 * (x + x**3 / 3.0 + x**5 / 5.0))
    raise ValueError(f"Unsupported asymptotic order: {order}")


def flat_profile_xci_kernel(
    *,
    length_m: float,
    beta2_eff_s2_per_m: float,
    bandwidth_hz: float,
    delta_f_hz: float,
    poly_sum: float = 1.0,
    log_order: int | None = None,
) -> float:
    """Return the flat-profile closed-form XCI kernel.

    ``log_order=None`` gives the exact closed form. ``log_order in {1,3,5}``
    applies the low-``B/|Delta f|`` asymptotic truncation to the log term.
    """
    beta2_eff = max(abs(float(beta2_eff_s2_per_m)), pcfm_gn.MIN_BETA2)
    log_term = _xci_log_term(abs(delta_f_hz), bandwidth_hz, order=log_order)
    return float(length_m / (2.0 * math.pi * beta2_eff) * log_term * float(poly_sum))


_J_SERIES_CUTOFF = 1e-3
_J_GRID_STEP_TARGET = 0.05
_J_GRID_MIN_POINTS = 2049
_J_GRID_MAX_POINTS = 250_001


def _j_bucket_upper(x_abs: float) -> float:
    x_abs = float(abs(x_abs))
    if x_abs <= 1.0:
        return 1.0
    return float(2.0 ** math.ceil(math.log2(x_abs)))


@lru_cache(maxsize=None)
def _j_lookup_grid(x_upper: float) -> tuple[np.ndarray, np.ndarray]:
    x_upper = float(max(abs(x_upper), 1.0))
    n_points = int(math.ceil(x_upper / _J_GRID_STEP_TARGET)) + 1
    n_points = min(max(n_points, _J_GRID_MIN_POINTS), _J_GRID_MAX_POINTS)
    x_grid = np.linspace(0.0, x_upper, n_points, dtype=float)
    si_grid, _ = sici(x_grid)
    integrand = np.empty_like(x_grid)
    integrand[0] = 1.0
    integrand[1:] = si_grid[1:] / x_grid[1:]
    # Older calculation kept here as trace: this primitive used to be evaluated
    # point-by-point with `quad(lambda t: Si(t)/t, 0, x_abs, ...)`, which was
    # accurate but too slow and fragile for the full PCFM Eq. 18 channel sweep.
    j_grid = cumulative_trapezoid(integrand, x_grid, initial=0.0)
    return x_grid, j_grid


def _j_series(x_value: np.ndarray) -> np.ndarray:
    x_value = np.asarray(x_value, dtype=float)
    x2 = x_value * x_value
    return x_value * (1.0 - x2 / 54.0 + (x2 * x2) / 3000.0)


def _j_of_x(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    out = np.empty_like(values)
    abs_values = np.abs(values)

    zero_mask = np.isclose(values, 0.0)
    out[zero_mask] = 0.0

    small_mask = (~zero_mask) & (abs_values < _J_SERIES_CUTOFF)
    if np.any(small_mask):
        out[small_mask] = _j_series(values[small_mask])

    large_mask = ~(zero_mask | small_mask)
    if np.any(large_mask):
        x_grid, j_grid = _j_lookup_grid(_j_bucket_upper(float(np.max(abs_values[large_mask]))))
        interp = np.interp(abs_values[large_mask], x_grid, j_grid)
        out[large_mask] = np.copysign(interp, values[large_mask])

    return out


def _g_kernel(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    out = np.zeros_like(z)
    si_value, _ = sici(z)
    out[:] = _j_of_x(z) - si_value
    nonzero_mask = ~np.isclose(z, 0.0)
    out[nonzero_mask] += (1.0 - np.cos(z[nonzero_mask])) / z[nonzero_mask]
    return out


def flat_profile_xci_kernel_eq18(
    *,
    length_m: float,
    beta2_eff_s2_per_m: float,
    bandwidth_hz: float,
    delta_f_hz: float,
) -> float:
    """Return the dimensional flat-profile Eq. 18 XCI kernel for PCFM.

    The standalone analytical script works with the normalized quantity

        K * T^2 / L^2

    as a function of x = Delta f / B. Here we convert that normalized kernel
    back to the dimensional PCFM kernel K before applying the usual PCFM
    XCI prefactors.
    """
    delta_abs_hz = float(abs(delta_f_hz))
    bandwidth_hz = float(abs(bandwidth_hz))
    if delta_abs_hz <= bandwidth_hz / 2.0:
        raise ValueError("Eq. 18 flat-profile XCI requires |Delta f| > B/2.")

    beta2_eff = max(abs(float(beta2_eff_s2_per_m)), pcfm_gn.MIN_BETA2)
    x = delta_abs_hz / bandwidth_hz
    l_over_ld = beta2_eff * float(length_m) * (bandwidth_hz**2)
    z_plus = np.pi * l_over_ld * (x + 0.5)
    z_minus = np.pi * l_over_ld * (x - 0.5)
    k_normalized = float(
        ((2.0 / (np.pi * l_over_ld)) * (_g_kernel(np.array([z_plus])) - _g_kernel(np.array([z_minus]))))[0]
        / (2.0 * np.pi)
    )
    return float(k_normalized * (float(length_m) * bandwidth_hz) ** 2)


def _flat_profile_xci_kernel_eq18_vectorized(
    *,
    length_m: float,
    beta2_eff_s2_per_m: np.ndarray,
    bandwidth_hz: float,
    delta_f_hz: np.ndarray,
) -> np.ndarray:
    delta_abs_hz = np.abs(np.asarray(delta_f_hz, dtype=float))
    bandwidth_hz = float(abs(bandwidth_hz))
    if np.any(delta_abs_hz <= bandwidth_hz / 2.0):
        raise ValueError("Eq. 18 flat-profile XCI requires |Delta f| > B/2.")

    beta2_eff = np.maximum(np.abs(np.asarray(beta2_eff_s2_per_m, dtype=float)), pcfm_gn.MIN_BETA2)
    x = delta_abs_hz / bandwidth_hz
    l_over_ld = beta2_eff * float(length_m) * (bandwidth_hz**2)
    z_plus = np.pi * l_over_ld * (x + 0.5)
    z_minus = np.pi * l_over_ld * (x - 0.5)
    k_normalized = (_g_kernel(z_plus) - _g_kernel(z_minus)) / (np.pi**2 * l_over_ld)
    return k_normalized * (float(length_m) * bandwidth_hz) ** 2


def _pcfm_xci_poly_sums(
    system: System,
    *,
    profile_path: str | None,
    degree: int,
) -> np.ndarray:
    """Return the same per-interferer polynomial sums used by PCFM XCI."""
    n_channels = system.wdm.frequency_grid().size
    if profile_path is None:
        return np.ones(n_channels, dtype=float)

    signal_power_ch_z, z = pcfm_gn.load_signal_profiles(profile_path, system)
    spp = pcfm_gn.normalize_spp(signal_power_ch_z, z)
    coeffs = pcfm_gn.fit_spp_polynomials(z, spp, degree=int(degree))
    return np.array([pcfm_gn.poly_sum(coeffs[i]) for i in range(n_channels)], dtype=float)


def flat_profile_pcfm_xci_channel_power(
    system: System,
    *,
    channel_idx: int,
    launch_powers_w: np.ndarray,
    profile_path: str | None = None,
    degree: int = 9,
    use_beta2_eff: bool = True,
    log_order: int | None = None,
    xci_model: str = "closed_form",
) -> float:
    """Return the current-implementation PCFM-XCI power for a flat profile.

    This mirrors the normalization currently used by ``compute_pcfm_nlin``:
    doubled ``g_ch`` input scaling, the extra ``/2`` in the XCI PSD term, and
    the final per-polarization conversion.
    """
    freqs = system.wdm.frequency_grid()
    n_channels = freqs.size
    idx = int(channel_idx)
    if idx < 0 or idx >= n_channels:
        raise IndexError(f"channel_idx={idx} out of bounds for n_channels={n_channels}")

    launch = np.asarray(launch_powers_w, dtype=float).reshape(-1)
    if launch.size != n_channels:
        raise ValueError(f"launch_powers_w size {launch.size} != n_channels {n_channels}")

    bandwidth_hz = float(system.pulse.baud_rate)
    length_m = float(system.fiber_length)
    beta2 = pcfm_gn._beta2_array(system, freqs)
    aeff = pcfm_gn._aeff_array(system, freqs)
    fc_hz = float(system.center_frequency) if system.center_frequency is not None else float(np.mean(freqs))
    beta_coeffs = pcfm_gn._beta_coeffs_from_profile(system, fc_hz) if use_beta2_eff else None
    poly_sums = _pcfm_xci_poly_sums(
        system,
        profile_path=profile_path,
        degree=int(degree),
    )

    # Mirror compute_pcfm_nlin exactly.
    g_ch = launch / bandwidth_hz * 2.0
    total_psd = 0.0
    if xci_model == "eq18":
        valid = np.ones(n_channels, dtype=bool)
        valid[idx] = False
        delta_f_all = np.asarray(freqs - freqs[idx], dtype=float)
        valid &= np.abs(delta_f_all) > bandwidth_hz / 2.0
        if not np.any(valid):
            return 0.0

        if beta_coeffs:
            beta2_xci = np.array(
                [pcfm_gn._beta2_eff(float(freqs[idx]), float(freqs[j]), beta_coeffs) for j in np.flatnonzero(valid)],
                dtype=float,
            )
        else:
            beta2_xci = np.asarray(beta2[valid], dtype=float)
        delta_f_valid = delta_f_all[valid]
        aeff_valid = np.asarray(aeff[valid], dtype=float)
        gamma_xci = (
            2.0
            * math.pi
            * float(freqs[idx])
            / c
            * (2.0 * pcfm_gn.N2_SIO2 / (float(aeff[idx]) + aeff_valid))
        )
        k_xci = _flat_profile_xci_kernel_eq18_vectorized(
            length_m=length_m,
            beta2_eff_s2_per_m=beta2_xci,
            bandwidth_hz=bandwidth_hz,
            delta_f_hz=delta_f_valid,
        )
        poly_sum_valid = np.asarray(poly_sums[valid], dtype=float)
        g_j = np.asarray(g_ch[valid], dtype=float)
        total_psd = float(
            np.sum(
                (32.0 / 27.0)
                * float(g_ch[idx])
                * (g_j**2)
                * (gamma_xci**2)
                * k_xci
                * poly_sum_valid
            )
        )
        return float(pcfm_gn._to_per_polarization_power(total_psd * bandwidth_hz))

    for j in range(n_channels):
        if j == idx:
            continue
        delta_f = float(freqs[j] - freqs[idx])
        if abs(delta_f) <= bandwidth_hz / 2.0:
            continue
        beta2_xci = (
            pcfm_gn._beta2_eff(float(freqs[idx]), float(freqs[j]), beta_coeffs)
            if beta_coeffs
            else float(beta2[j])
        )
        if xci_model == "closed_form":
            k_xci = flat_profile_xci_kernel(
                length_m=length_m,
                beta2_eff_s2_per_m=beta2_xci,
                bandwidth_hz=bandwidth_hz,
                delta_f_hz=delta_f,
                poly_sum=float(poly_sums[j]),
                log_order=log_order,
            )
        elif xci_model != "closed_form":
            raise ValueError(f"Unsupported xci_model={xci_model!r}.")
        gamma_xci = (
            2.0
            * math.pi
            * float(freqs[idx])
            / c
            * (2.0 * pcfm_gn.N2_SIO2 / (float(aeff[idx]) + float(aeff[j])))
        )
        total_psd += (
            (32.0 / 27.0)
            * float(g_ch[idx])
            * float(g_ch[j] ** 2)
            * (gamma_xci**2)
            * k_xci
        )

    return float(pcfm_gn._to_per_polarization_power(total_psd * bandwidth_hz))
