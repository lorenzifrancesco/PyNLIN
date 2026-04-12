from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
from scipy.constants import c
from scipy.integrate import quad
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


@lru_cache(maxsize=None)
def _j_of_x_scalar(x_value: float) -> float:
    if np.isclose(x_value, 0.0):
        return 0.0
    sign = 1.0 if x_value >= 0.0 else -1.0
    x_abs = abs(x_value)

    def integrand(t: float) -> float:
        if np.isclose(t, 0.0):
            return 1.0
        si_value, _ = sici(t)
        return si_value / t

    value, _ = quad(integrand, 0.0, x_abs, limit=300, epsabs=1e-11, epsrel=1e-11)
    return sign * value


def _g_kernel(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    for idx, zi in np.ndenumerate(z):
        if np.isclose(zi, 0.0):
            out[idx] = 0.0
            continue
        si_value, _ = sici(float(zi))
        out[idx] = _j_of_x_scalar(float(zi)) - si_value + (1.0 - np.cos(zi)) / zi
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


def flat_profile_pcfm_xci_channel_power(
    system: System,
    *,
    channel_idx: int,
    launch_powers_w: np.ndarray,
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

    # Mirror compute_pcfm_nlin exactly.
    g_ch = launch / bandwidth_hz * 2.0
    total_psd = 0.0
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
                poly_sum=1.0,
                log_order=log_order,
            )
        elif xci_model == "eq18":
            k_xci = flat_profile_xci_kernel_eq18(
                length_m=length_m,
                beta2_eff_s2_per_m=beta2_xci,
                bandwidth_hz=bandwidth_hz,
                delta_f_hz=delta_f,
            )
        else:
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
            # / 2.0 # FIXME this is too much and it is something from the past
        )

    return float(pcfm_gn._to_per_polarization_power(total_psd * bandwidth_hz))
