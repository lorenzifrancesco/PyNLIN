from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
from scipy.constants import c
from scipy.integrate import cumulative_trapezoid
from scipy.special import sici

from pynlin.methods import pcfm as pcfm_gn
from pynlin.system import System


def _xci_log_term(
    delta_abs_hz: float, bandwidth_hz: float, order: int | None = None
) -> float:
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


def pcfm_I(
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
    # accurate but too slow and fragile for the full PCFM-II channel sweep.
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
        x_grid, j_grid = _j_lookup_grid(
            _j_bucket_upper(float(np.max(abs_values[large_mask])))
        )
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


_SBAR_SERIES_CUTOFF = 0.5


def _sbar_series(order: int, x_value: float) -> float:
    """Return ``int_0^1 y^order sin(x*y) dy`` using its small-argument series."""
    total = 0.0
    x_power = float(x_value)
    sign = 1.0
    factorial = 1.0
    for r in range(80):
        term = sign * x_power / (factorial * (order + 2 * r + 2))
        total += term
        if abs(term) <= 1e-15 * max(1.0, abs(total)):
            break
        sign *= -1.0
        x_power *= x_value * x_value
        factorial *= (2 * r + 2) * (2 * r + 3)
    return float(total)


@lru_cache(maxsize=200_000)
def _sbar(order: int, x_value: float) -> float:
    """Return the normalized sine moment ``int_0^1 y^order sin(x*y) dy``."""
    order = int(order)
    x_value = float(x_value)
    if order < 0:
        raise ValueError("Sine-moment order must be non-negative.")
    if abs(x_value) < _SBAR_SERIES_CUTOFF:
        return _sbar_series(order, x_value)

    sin_x = math.sin(x_value)
    cos_x = math.cos(x_value)
    s_prev = (1.0 - cos_x) / x_value
    if order == 0:
        return float(s_prev)
    c_prev = sin_x / x_value
    for current_order in range(1, order + 1):
        s_cur = -cos_x / x_value + current_order * c_prev / x_value
        c_cur = sin_x / x_value - current_order * s_prev / x_value
        s_prev, c_prev = s_cur, c_cur
    return float(s_prev)


@lru_cache(maxsize=200_000)
def _jbar(order: int, x_value: float) -> float:
    """Return the normalized ``J_order`` term from any_island.pdf Eq. 17."""
    order = int(order)
    x_value = float(x_value)
    if order < 0:
        raise ValueError("J order must be non-negative.")
    if order == 0:
        return float(_j_of_x(np.array([x_value], dtype=float))[0])
    si_value, _ = sici(x_value)
    return float((si_value - _sbar(order - 1, x_value)) / order)


@lru_cache(maxsize=500_000)
def _ipq_normalized(p_order: int, q_order: int, x_value: float) -> float:
    """Return ``L``-normalized ``I_{p,q}`` from any_island.pdf Eq. 17.

    With ``x = lambda * L``, the dimensional integral is
    ``I_{p,q}(L; lambda) = L**(p+q) * _ipq_normalized(p, q, x)`` for
    ``p >= 1`` and ``L**q * _ipq_normalized(0, q, x)`` for ``p == 0``.
    """
    p_order = int(p_order)
    q_order = int(q_order)
    x_value = float(x_value)
    if p_order < 0 or q_order < 0:
        raise ValueError("I_{p,q} orders must be non-negative.")

    total = 0.0
    if p_order == 0:
        for r in range(q_order + 1):
            total += math.comb(q_order, r) * ((-1.0) ** r) * _jbar(r, x_value)
        return float(total)

    si_value, _ = sici(x_value)
    beta_value = math.exp(
        math.lgamma(p_order)
        + math.lgamma(q_order + 1)
        - math.lgamma(p_order + q_order + 1)
    )
    total = float(si_value) * beta_value
    for r in range(q_order + 1):
        total -= (
            math.comb(q_order, r)
            * ((-1.0) ** r)
            * _sbar(p_order + r - 1, x_value)
            / (p_order + r)
        )
    return float(total)


def _nonflat_profile_pcfm_II_from_coeffs(
    *,
    length_m: float,
    beta2_eff_s2_per_m: float,
    bandwidth_hz: float,
    delta_f_hz: float,
    coeffs: np.ndarray,
) -> float:
    """Evaluate any_island.pdf Eq. 18 for a single XCI rectangle.

    ``coeffs`` must be the interferer SPP polynomial in normalized distance,
    using the ascending convention returned by ``pcfm_gn.fit_spp_polynomials``.
    """
    length_m = float(length_m)
    bandwidth_hz = float(abs(bandwidth_hz))
    delta_f_hz = float(delta_f_hz)
    if length_m <= 0.0:
        raise ValueError("PCFM-II XCI requires a positive length.")
    if abs(delta_f_hz) <= bandwidth_hz / 2.0:
        raise ValueError("PCFM-II XCI requires |Delta f| > B/2.")

    beta2_eff = max(abs(float(beta2_eff_s2_per_m)), pcfm_gn.MIN_BETA2)
    # Eq. 18 is evaluated with the PCFM/GN phase convention
    # lambda = 4*pi^2*beta2_eff*L*f1*f2.  With x=Delta f/B and
    # a=|beta2_eff|*L*B^2, the flat normalized result equals
    # 2*pi * k_xci_eq18_normalized(x, 2*pi*a) from the analytical script.
    # The inner 2*pi is in the phase arguments below; the outer 2*pi is the
    # resulting prefactor difference relative to that script's normalization.
    phase_beta = 4.0 * math.pi**2 * beta2_eff
    coeffs = np.asarray(coeffs, dtype=float).reshape(-1)
    if coeffs.size == 0:
        raise ValueError("PCFM-II XCI requires at least one SPP coefficient.")
    if np.any(~np.isfinite(coeffs)):
        coeffs = np.nan_to_num(coeffs, nan=0.0, posinf=0.0, neginf=0.0)

    ak = -bandwidth_hz / 2.0
    bk = bandwidth_hz / 2.0
    cm = delta_f_hz - bandwidth_hz / 2.0
    dm = delta_f_hz + bandwidth_hz / 2.0
    lambdas_l = (
        phase_beta
        * length_m
        * np.array([ak * dm, ak * cm, bk * cm, bk * dm], dtype=float)
    )
    lambda_signs = (-1.0, 1.0, -1.0, 1.0)

    total = 0.0
    degree = coeffs.size - 1
    for n_order in range(degree + 1):
        coeff_n = float(coeffs[n_order])
        if coeff_n == 0.0:
            continue
        for m_order in range(n_order, degree + 1):
            coeff_nm = coeff_n * float(coeffs[m_order])
            if coeff_nm == 0.0:
                continue
            pair_sum = 0.0
            for sign, lambda_l in zip(lambda_signs, lambdas_l):
                bracket = 0.0
                for i_order in range(n_order + 1):
                    q_order = m_order + n_order - i_order + 1
                    bracket += (
                        math.comb(n_order, i_order)
                        * _ipq_normalized(i_order, q_order, float(lambda_l))
                        / q_order
                    )
                for j_order in range(m_order + 1):
                    q_order = m_order + n_order - j_order + 1
                    bracket += (
                        math.comb(m_order, j_order)
                        * _ipq_normalized(j_order, q_order, float(lambda_l))
                        / q_order
                    )
                pair_sum += sign * bracket
            symmetry = 1.0 if n_order == m_order else 2.0
            total += symmetry * coeff_nm * pair_sum

    return float((length_m / phase_beta) * total) # / (2 * np.pi) # FIXME I added 2 pi for seeing what happens


def flat_profile_pcfm_II(
    *,
    length_m: float,
    beta2_eff_s2_per_m: float,
    bandwidth_hz: float,
    delta_f_hz: float,
) -> float:
    """Return the dimensional flat-profile PCFM-II XCI kernel for PCFM."""
    return _nonflat_profile_pcfm_II_from_coeffs(
        length_m=length_m,
        beta2_eff_s2_per_m=beta2_eff_s2_per_m,
        bandwidth_hz=bandwidth_hz,
        delta_f_hz=delta_f_hz,
        coeffs=np.ones(1, dtype=float),
    )


def pcfm_II(
    system: System,
    *,
    profile_path: str | None,
    interferer_idx: int,
    beta2_eff_s2_per_m: float,
    delta_f_hz: float,
    bandwidth_hz: float | None = None,
    length_m: float | None = None,
    degree: int = 9,
) -> float:
    """Return the true PCFM-II XCI kernel from any_island.pdf Eq. 18.

    The signal power profile is loaded, normalized, fitted with a degree-9
    polynomial by default, and the interferer profile is used directly in
    Eq. 18.  When ``profile_path`` is ``None``, a flat unit profile is used.
    """
    n_channels = system.wdm.frequency_grid().size
    idx = int(interferer_idx)
    if idx < 0 or idx >= n_channels:
        raise IndexError(
            f"interferer_idx={idx} out of bounds for n_channels={n_channels}"
        )

    if profile_path is None:
        coeffs = np.ones(1, dtype=float)
    else:
        signal_power_ch_z, z = pcfm_gn.load_signal_profiles(profile_path, system)
        spp = pcfm_gn.normalize_spp(signal_power_ch_z, z)
        coeffs = pcfm_gn.fit_spp_polynomials(z, spp, degree=int(degree))[idx]

    return _nonflat_profile_pcfm_II_from_coeffs(
        length_m=float(system.fiber_length if length_m is None else length_m),
        beta2_eff_s2_per_m=beta2_eff_s2_per_m,
        bandwidth_hz=float(
            system.pulse.baud_rate if bandwidth_hz is None else bandwidth_hz
        ),
        delta_f_hz=delta_f_hz,
        coeffs=coeffs,
    )


def _flat_profile_pcfm_II_vectorized(
    *,
    length_m: float,
    beta2_eff_s2_per_m: np.ndarray,
    bandwidth_hz: float,
    delta_f_hz: np.ndarray,
) -> np.ndarray:
    delta_f_values = np.asarray(delta_f_hz, dtype=float).reshape(-1)
    delta_abs_hz = np.abs(delta_f_values)
    bandwidth_hz = float(abs(bandwidth_hz))
    if np.any(delta_abs_hz <= bandwidth_hz / 2.0):
        raise ValueError("PCFM-II flat-profile XCI requires |Delta f| > B/2.")

    beta2_eff = np.maximum(
        np.abs(np.asarray(beta2_eff_s2_per_m, dtype=float).reshape(-1)),
        pcfm_gn.MIN_BETA2,
    )
    if beta2_eff.size == 1 and delta_abs_hz.size != 1:
        beta2_eff = np.full(delta_abs_hz.size, float(beta2_eff[0]), dtype=float)
    if beta2_eff.size != delta_abs_hz.size:
        raise ValueError("beta2_eff_s2_per_m and delta_f_hz must have matching sizes.")
    return np.array(
        [
            _nonflat_profile_pcfm_II_from_coeffs(
                length_m=length_m,
                beta2_eff_s2_per_m=float(beta2_eff[pos]),
                bandwidth_hz=bandwidth_hz,
                delta_f_hz=float(delta_f_values[pos]),
                coeffs=np.ones(1, dtype=float),
            )
            for pos in range(delta_abs_hz.size)
        ],
        dtype=float,
    )


def _pcfm_I_poly_sums(
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
    return np.array(
        [pcfm_gn.poly_sum(coeffs[i]) for i in range(n_channels)], dtype=float
    )


def _pcfm_I_profile_coeffs(
    system: System,
    *,
    profile_path: str | None,
    degree: int,
) -> np.ndarray | None:
    """Return fitted normalized SPP coefficients, or ``None`` for a flat profile."""
    if profile_path is None:
        return None
    signal_power_ch_z, z = pcfm_gn.load_signal_profiles(profile_path, system)
    spp = pcfm_gn.normalize_spp(signal_power_ch_z, z)
    return pcfm_gn.fit_spp_polynomials(z, spp, degree=int(degree))


def pcfm_general(
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
    """Return the current-implementation analytic PCFM-XCI power.

    This mirrors the normalization currently used by ``compute_pcfm_nlin``:
    doubled ``g_ch`` input scaling and the final per-polarization conversion.
    For ``xci_model="eq18"``, a supplied ``profile_path`` uses the fitted
    interferer SPP directly in any_island.pdf Eq. 18.
    """
    freqs = system.wdm.frequency_grid()
    n_channels = freqs.size
    idx = int(channel_idx)
    if idx < 0 or idx >= n_channels:
        raise IndexError(f"channel_idx={idx} out of bounds for n_channels={n_channels}")

    launch = np.asarray(launch_powers_w, dtype=float).reshape(-1)
    if launch.size != n_channels:
        raise ValueError(
            f"launch_powers_w size {launch.size} != n_channels {n_channels}"
        )

    bandwidth_hz = float(system.pulse.baud_rate)
    length_m = float(system.fiber_length)
    beta2 = pcfm_gn._beta2_array(system, freqs)
    aeff = pcfm_gn._aeff_array(system, freqs)
    fc_hz = (
        float(system.center_frequency)
        if system.center_frequency is not None
        else float(np.mean(freqs))
    )
    beta_coeffs = (
        pcfm_gn._beta_coeffs_from_profile(system, fc_hz) if use_beta2_eff else None
    )
    profile_coeffs = _pcfm_I_profile_coeffs(
        system, profile_path=profile_path, degree=int(degree)
    )
    poly_sums = (
        np.ones(n_channels, dtype=float)
        if profile_coeffs is None
        else np.array(
            [pcfm_gn.poly_sum(profile_coeffs[i]) for i in range(n_channels)],
            dtype=float,
        )
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
                [
                    pcfm_gn._beta2_eff(float(freqs[idx]), float(freqs[j]), beta_coeffs)
                    for j in np.flatnonzero(valid)
                ],
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
        k_xci = (
            _flat_profile_pcfm_II_vectorized(
                length_m=length_m,
                beta2_eff_s2_per_m=beta2_xci,
                bandwidth_hz=bandwidth_hz,
                delta_f_hz=delta_f_valid,
            ) # THIS IS THE CORE
            if profile_coeffs is None
            else np.array(
                [
                    _nonflat_profile_pcfm_II_from_coeffs(
                        length_m=length_m,
                        beta2_eff_s2_per_m=float(beta2_xci[pos]),
                        bandwidth_hz=bandwidth_hz,
                        delta_f_hz=float(delta_f_valid[pos]),
                        coeffs=profile_coeffs[j],
                    )
                    for pos, j in enumerate(np.flatnonzero(valid))
                ],
                dtype=float,
            )
        )
        g_j = np.asarray(g_ch[valid], dtype=float)
        total_psd = float(
            np.sum(
                (32.0 / 27.0)
                * float(g_ch[idx])
                * (g_j**2)
                * (gamma_xci**2)
                * k_xci
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
            k_xci = pcfm_I(
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
