"""
PCFM/GN NLIN estimation for SMF systems (SCI+XCI, no MCI).

This module is intended to be used by analysis/uwb_nlin.py for the
PCFM-style case studies, leveraging existing System/Fiber/WDM
infrastructure and Raman power profile files.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger as lg
from scipy.constants import c

from pynlin.system import System
from pynlin.utils import dBm2watt

N2_SIO2 = 2.6e-20
MIN_POWER_W = 1e-12
MAX_POWER_W = 10.0
MAX_SPP = 1e3
MIN_BETA2 = 1e-30
POLARIZATION_COUNT = 2.0
_WARNED_ONCE: set[str] = set()


def _warn_once(key: str, message: str) -> None:
    """Emit a warning message only once for a given key.

    Parameters
    ----------
    key:
        Stable identifier used to deduplicate repeated warnings.
    message:
        Warning text forwarded to the module logger.
    """
    if key in _WARNED_ONCE:
        return
    _WARNED_ONCE.add(key)
    lg.warning(message)


def _to_per_polarization_power(power_w: np.ndarray | float) -> np.ndarray | float:
    """Convert channel power from dual-pol normalization to per-polarization."""
    return power_w / POLARIZATION_COUNT


def _to_per_channel_power(power_w: np.ndarray | float) -> np.ndarray | float:
    """Convert per-polarization power to dual-pol channel normalization."""
    return power_w * POLARIZATION_COUNT


@dataclass
class PcfmConfig:
    """Configuration knobs for PCFM/GN kernel evaluation and integration grids."""
    degree: int = 9
    include_mci: bool = False
    use_numeric_sci: bool = True
    use_numeric_xci: bool = False
    use_beta2_eff: bool = True
    n_f: int = 40
    n_z: int = 200
    phase_coeff: float = 4.0 * np.pi ** 2


def _load_profile_payload(profile_path: Path | str):
    """Load a Raman/power-profile payload from ``.npy``/``.npz`` files.

    Parameters
    ----------
    profile_path:
        Path to a cached profile file produced by analysis workflows.

    Returns
    -------
    object
        A mapping-like payload exposing keys such as ``signal_sol`` and ``z``.

    Raises
    ------
    FileNotFoundError
        If the path does not exist.
    TypeError
        If the file content is not a supported payload type.
    """
    path = Path(profile_path)
    if not path.exists():
        raise FileNotFoundError(f"Profile file not found: {path}")
    payload = np.load(path, allow_pickle=True)
    if isinstance(payload, np.lib.npyio.NpzFile):
        return payload
    if isinstance(payload, np.ndarray) and payload.shape == ():
        return payload.item()
    if isinstance(payload, dict):
        return payload
    raise TypeError(f"Unexpected profile payload type: {type(payload)}")


def load_signal_profiles(profile_path: Path | str, system: System) -> tuple[np.ndarray, np.ndarray]:
    """Load channel power profiles and longitudinal grid from a profile file.

    The function accepts the different historical array layouts used by saved
    solutions and normalizes them to a common shape.

    Parameters
    ----------
    profile_path:
        Path to profile cache containing at least ``signal_sol`` and ``z``.
    system:
        Active system configuration, used to validate channel count.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(signal_power_ch_z, z)``, where:
        - ``signal_power_ch_z`` has shape ``(n_channels, n_z)`` in Watt.
        - ``z`` has shape ``(n_z,)`` in meter.
    """
    data = _load_profile_payload(profile_path)
    sig = None
    if hasattr(data, "get"):
        for key in ("signal_sol", "signal_solution", "signal_power"):
            if key in data:
                sig = data.get(key)
                if sig is not None:
                    break
    if sig is None:
        raise ValueError("Profile data missing signal power solution.")
    z = data.get("z") if hasattr(data, "get") else None
    if z is None:
        raise ValueError("Profile data missing z grid.")

    sig = np.asarray(sig, dtype=float)
    z = np.asarray(z, dtype=float)

    if sig.ndim == 2:
        if sig.shape[0] == z.size:
            sig_z_ch = sig
        elif sig.shape[1] == z.size:
            sig_z_ch = sig.T
        else:
            raise ValueError(f"Unrecognized signal_sol shape {sig.shape} for z size {z.size}.")
    elif sig.ndim == 3:
        if sig.shape[0] == z.size:
            if sig.shape[1] == system.n_channels:
                sig_z_ch = np.sum(sig, axis=2)
            elif sig.shape[2] == system.n_channels:
                sig_z_ch = np.sum(sig, axis=1)
            else:
                raise ValueError(f"signal_sol shape {sig.shape} does not match channel count {system.n_channels}.")
        elif sig.shape[1] == z.size and sig.shape[0] == system.n_channels:
            sig_z_ch = np.sum(sig, axis=2).T
        else:
            raise ValueError(f"Unrecognized signal_sol shape {sig.shape} for z size {z.size}.")
    else:
        raise ValueError(f"Unexpected signal_sol shape: {sig.shape}")

    if sig_z_ch.shape[1] != system.n_channels:
        lg.warning(
            f"signal_sol channels ({sig_z_ch.shape[1]}) != system.n_channels ({system.n_channels}); "
            "continuing with profile ordering."
        )

    signal_power_ch_z = sig_z_ch.T
    return signal_power_ch_z, z


def normalize_spp(signal_power_ch_z: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Normalize signal power profiles to unit launch value per channel.

    Parameters
    ----------
    signal_power_ch_z:
        Absolute channel powers, shape ``(n_channels, n_z)`` in Watt.
    z:
        Longitudinal grid in meter.

    Returns
    -------
    np.ndarray
        Normalized profiles ``p(z) = P(z)/P(0)`` with shape ``(n_channels, n_z)``.
        Non-finite values are sanitized and values are clipped to ``[0, MAX_SPP]``.
    """
    p_ch_z = signal_power_ch_z
    p0 = p_ch_z[:, 0].copy()
    p0[p0 <= MIN_POWER_W] = np.nan
    p = p_ch_z / p0[:, None]
    p = np.clip(p, 0.0, MAX_SPP)
    if np.any(~np.isfinite(p)):
        p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
    return p


def fit_spp_polynomials(z: np.ndarray, p_ch_z: np.ndarray, degree: int) -> np.ndarray:
    """Fit normalized channel profiles with polynomials in normalized distance.

    The fit variable is ``z_norm = (z - z0) / L`` in ``[0, 1]``.

    Parameters
    ----------
    z:
        Longitudinal grid in meter.
    p_ch_z:
        Normalized channel profiles with shape ``(n_channels, n_z)``.
    degree:
        Polynomial degree.

    Returns
    -------
    np.ndarray
        Polynomial coefficients with shape ``(n_channels, degree+1)``, in
        ascending order ``a0 + a1 z + ...`` for use with ``numpy.polynomial``.
    """
    if z.size < degree + 1:
        raise ValueError(f"Not enough z samples ({z.size}) for degree {degree} polynomial fit.")
    L = float(z[-1] - z[0]) if z.size > 1 else float(z[-1])
    if L <= 0:
        raise ValueError("Invalid fiber length derived from z grid.")
    z_norm = (z - z[0]) / L
    coeffs = np.zeros((p_ch_z.shape[0], degree + 1), dtype=float)
    for idx, p in enumerate(p_ch_z):
        fit = np.polyfit(z_norm, p, degree)
        coeffs[idx, :] = fit[::-1]
    if np.any(~np.isfinite(coeffs)):
        coeffs = np.nan_to_num(coeffs, nan=0.0, posinf=0.0, neginf=0.0)
    return coeffs


def poly_eval(coeffs: np.ndarray, x: np.ndarray | float) -> np.ndarray:
    """Evaluate a polynomial defined by ascending coefficients.

    Parameters
    ----------
    coeffs:
        Coefficients ``[a0, a1, ...]``.
    x:
        Evaluation point(s).

    Returns
    -------
    np.ndarray
        Evaluated polynomial values.
    """
    return np.polynomial.polynomial.polyval(x, coeffs)


def poly_sum(coeffs: np.ndarray) -> float:
    """Compute ``sum_{n,k} a_n a_k / (n+k+1)`` from polynomial coefficients.

    This term appears in the closed-form XCI approximation.

    Parameters
    ----------
    coeffs:
        Ascending polynomial coefficients ``[a0, a1, ...]``.

    Returns
    -------
    float
        Scalar weighted convolution sum.
    """
    conv = np.convolve(coeffs, coeffs)
    denom = np.arange(conv.size, dtype=float) + 1.0
    return float(np.sum(conv / denom))


def _beta2_array(system: System, freqs: np.ndarray) -> np.ndarray:
    """Sample channel-wise ``beta2`` from the fiber model.

    Parameters
    ----------
    system:
        Active system/fiber container.
    freqs:
        Channel frequencies in Hz.

    Returns
    -------
    np.ndarray
        Dispersion values ``beta2`` in SI units at each channel frequency.
    """
    wl = c / freqs
    beta2 = np.array([system.fiber.beta2_at(float(w)) for w in wl], dtype=float)
    if np.any(~np.isfinite(beta2)):
        fallback = getattr(system.fiber, "beta2", None)
        if fallback is None:
            raise ValueError("beta2 profile missing and no fallback beta2 available.")
        beta2 = np.where(np.isfinite(beta2), beta2, float(fallback))
    return beta2


def _beta_coeffs_from_profile(system: System, fc_hz: float) -> tuple[float, float, float, float] | None:
    """Extract ``(beta2, beta3, beta4, fc_hz)`` from a ``beta(omega)`` spline.

    Parameters
    ----------
    system:
        Active system whose fiber may provide ``beta_spline_omega``.
    fc_hz:
        Reference center frequency in Hz where derivatives are evaluated.

    Returns
    -------
    tuple[float, float, float, float] | None
        ``(beta2, beta3, beta4, fc_hz)`` if available, otherwise ``None``.
    """
    fiber = getattr(system, "fiber", None)
    if fiber is None or not hasattr(fiber, "beta_spline_omega"):
        _warn_once(
            "beta_spline_missing",
            "beta_spline_omega unavailable; using beta2_at only (Eq. 28 disabled).",
        )
        return None
    try:
        spline = fiber.beta_spline_omega(s=0.0, k=5)
    except Exception as exc:
        _warn_once(
            "beta_spline_missing",
            f"beta_spline_omega unavailable; using beta2_at only ({exc}).",
        )
        return None
    omega_c = 2.0 * np.pi * fc_hz
    try:
        beta2 = float(spline.derivative(2)(omega_c))
        beta3 = float(spline.derivative(3)(omega_c))
        beta4 = float(spline.derivative(4)(omega_c))
    except Exception as exc:
        lg.warning(f"Failed beta derivatives from spline; using beta2_at only ({exc}).")
        return None
    return beta2, beta3, beta4, fc_hz


def _check_loss_profile(system: System) -> None:
    """Warn when frequency-dependent attenuation data are unavailable.

    The PCFM Eq. (30)-style loss approximation is more accurate with an
    attenuation profile. This helper emits one-time warnings for fallbacks.
    """
    fiber = getattr(system, "fiber", None)
    if fiber is None:
        _warn_once(
            "alpha_profile_missing",
            "Fiber missing; Eq. (30) loss approximation unavailable.",
        )
        return
    if getattr(fiber, "_attenuation_profile", None) is None:
        _warn_once(
            "alpha_profile_missing",
            "Attenuation profile missing; Eq. (30) uses channel-center loss. "
            "Falling back to fiber.loss_profile/constant attenuation.",
        )


def _beta2_eff(fm_hz: float, fk_hz: float, coeffs: tuple[float, float, float, float]) -> float:
    """Compute effective ``beta2`` for a channel pair using Poggiolini Eq. (28).

    Parameters
    ----------
    fm_hz:
        Channel-under-test frequency in Hz.
    fk_hz:
        Interferer channel frequency in Hz.
    coeffs:
        Tuple ``(beta2, beta3, beta4, fc_hz)``.

    Returns
    -------
    float
        Effective dispersion coefficient for the integration island.
    """
    # Eq. (28): beta2_eff = beta2 + πβ3(fm+fk−2fc) + (2/3)π^2β4[(fm−fc)^2+(fm−fc)(fk−fc)+(fk−fc)^2]
    beta2, beta3, beta4, fc_hz = coeffs
    dfm = fm_hz - fc_hz
    dfk = fk_hz - fc_hz
    term3 = dfm + dfk
    term4 = (dfm * dfm + dfm * dfk + dfk * dfk)
    return beta2 + np.pi * beta3 * term3 + (2.0 / 3.0) * (np.pi ** 2) * beta4 * term4


def _aeff_array(system: System, freqs: np.ndarray) -> np.ndarray:
    """Sample effective area ``A_eff`` for each channel frequency.

    Parameters
    ----------
    system:
        Active system/fiber container.
    freqs:
        Channel frequencies in Hz.

    Returns
    -------
    np.ndarray
        ``A_eff`` array in square meter, one value per channel.
    """
    wl = c / freqs
    return np.array([system.fiber.effective_area_at(float(w)) for w in wl], dtype=float)


def _launch_powers_from_system(system: System) -> np.ndarray:
    """Resolve per-channel launch powers from the active ``System`` config.

    Supports either per-band launch powers (when WDM band specs are present) or
    a single global launch power.

    Parameters
    ----------
    system:
        Active system configuration.

    Returns
    -------
    np.ndarray
        Launch powers in Watt, shape ``(n_channels,)``.
    """
    freqs = system.wdm.frequency_grid()
    if hasattr(system.wdm, "band_specs") and system.wdm.band_specs:
        p = np.zeros_like(freqs, dtype=float)
        for name, slc in system.wdm._band_slices.items():
            spec = system.wdm.band_specs.get(name)
            power_dbm = spec.launch_power_dbm if spec else None
            if power_dbm is None:
                power_dbm = system.launch_power if system.launch_power is not None else -5.0
            p[slc] = dBm2watt(power_dbm)
        return p
    power_dbm = system.launch_power if system.launch_power is not None else -5.0
    return np.full_like(freqs, dBm2watt(power_dbm), dtype=float)


def compute_sci_numeric(coeffs: np.ndarray, L: float, beta2_eff: float, B: float,
                        n_f: int, n_z: int, phase_coeff: float) -> float:
    """Numerically compute the SCI kernel over a rectangular frequency island.

    Parameters
    ----------
    coeffs:
        Ascending polynomial coefficients for normalized profile ``p(z/L)``.
    L:
        Span length in meter.
    beta2_eff:
        Effective dispersion for SCI in SI units.
    B:
        Channel bandwidth (baud rate) in Hz.
    n_f:
        Frequency samples per axis.
    n_z:
        Longitudinal integration samples.
    phase_coeff:
        Multiplicative constant in the phase term, typically ``4*pi^2``.

    Returns
    -------
    float
        SCI kernel value.
    """
    z = np.linspace(0.0, L, int(n_z))
    z_norm = z / L if L > 0 else z
    p_z = poly_eval(coeffs, z_norm)
    f1 = np.linspace(-B / 2.0, B / 2.0, int(n_f))
    f2 = np.linspace(-B / 2.0, B / 2.0, int(n_f))
    f_prod = np.outer(f1, f2)
    phase = phase_coeff * beta2_eff * f_prod[..., None] * z[None, None, :]
    integrand = p_z[None, None, :] * np.exp(1j * phase)
    inner = np.trapezoid(integrand, z, axis=2)
    inner_sq = np.abs(inner) ** 2
    k_val = np.trapezoid(np.trapezoid(inner_sq, f2, axis=1), f1, axis=0)
    return float(k_val)


def compute_sci_numeric_direct(p_z: np.ndarray, z: np.ndarray, beta2_eff: float, B: float,
                               n_f: int, phase_coeff: float) -> float:
    """Numerically compute SCI kernel directly from sampled ``p(z)``.

    Parameters
    ----------
    p_z:
        Normalized power profile samples for one channel.
    z:
        Longitudinal grid in meter.
    beta2_eff:
        Effective dispersion for SCI in SI units.
    B:
        Channel bandwidth (baud rate) in Hz.
    n_f:
        Frequency samples per axis.
    phase_coeff:
        Multiplicative constant in the phase term, typically ``4*pi^2``.

    Returns
    -------
    float
        SCI kernel value.
    """
    p_z = np.asarray(p_z, dtype=float)
    z = np.asarray(z, dtype=float)
    if p_z.size != z.size:
        raise ValueError("p_z and z must have the same length for direct SCI.")
    f1 = np.linspace(-B / 2.0, B / 2.0, int(n_f))
    f2 = np.linspace(-B / 2.0, B / 2.0, int(n_f))
    f_prod = np.outer(f1, f2)
    phase = phase_coeff * beta2_eff * f_prod[..., None] * z[None, None, :]
    integrand = p_z[None, None, :] * np.exp(1j * phase)
    inner = np.trapezoid(integrand, z, axis=2)
    inner_sq = np.abs(inner) ** 2
    k_val = np.trapezoid(np.trapezoid(inner_sq, f2, axis=1), f1, axis=0)
    return float(k_val)


def compute_xci_numeric(coeffs: np.ndarray, L: float, beta2_eff: float, B_cut: float,
                        delta_f: float, B_int: float,
                        n_f: int, n_z: int, phase_coeff: float) -> float:
    """Numerically compute the XCI kernel over cut/interferer frequency islands.

    Parameters
    ----------
    coeffs:
        Ascending polynomial coefficients for interferer profile ``p(z/L)``.
    L:
        Span length in meter.
    beta2_eff:
        Effective dispersion for the CUT/interferer pair.
    B_cut:
        CUT bandwidth in Hz.
    delta_f:
        Interferer center offset from CUT in Hz.
    B_int:
        Interferer bandwidth in Hz.
    n_f:
        Frequency samples per axis.
    n_z:
        Longitudinal integration samples.
    phase_coeff:
        Multiplicative constant in the phase term, typically ``4*pi^2``.

    Returns
    -------
    float
        XCI kernel value.
    """
    z = np.linspace(0.0, L, int(n_z))
    z_norm = z / L if L > 0 else z
    p_z = poly_eval(coeffs, z_norm)
    f1 = np.linspace(-B_cut / 2.0, B_cut / 2.0, int(n_f))
    f2 = np.linspace(delta_f - B_int / 2.0, delta_f + B_int / 2.0, int(n_f))
    f_prod = np.outer(f1, f2)
    phase = phase_coeff * beta2_eff * f_prod[..., None] * z[None, None, :]
    integrand = p_z[None, None, :] * np.exp(1j * phase)
    inner = np.trapezoid(integrand, z, axis=2)
    inner_sq = np.abs(inner) ** 2
    k_val = np.trapezoid(np.trapezoid(inner_sq, f2, axis=1), f1, axis=0)
    return float(k_val)


def compute_xci_numeric_direct(p_z: np.ndarray, z: np.ndarray, beta2_eff: float, B_cut: float,
                               delta_f: float, B_int: float,
                               n_f: int, phase_coeff: float) -> float:
    """Numerically compute XCI kernel directly from sampled ``p(z)``.

    Parameters
    ----------
    p_z:
        Normalized interferer profile samples.
    z:
        Longitudinal grid in meter.
    beta2_eff:
        Effective dispersion for the CUT/interferer pair.
    B_cut:
        CUT bandwidth in Hz.
    delta_f:
        Interferer center offset from CUT in Hz.
    B_int:
        Interferer bandwidth in Hz.
    n_f:
        Frequency samples per axis.
    phase_coeff:
        Multiplicative constant in the phase term, typically ``4*pi^2``.

    Returns
    -------
    float
        XCI kernel value.
    """
    p_z = np.asarray(p_z, dtype=float)
    z = np.asarray(z, dtype=float)
    if p_z.size != z.size:
        raise ValueError("p_z and z must have the same length for direct XCI.")
    f1 = np.linspace(-B_cut / 2.0, B_cut / 2.0, int(n_f))
    f2 = np.linspace(delta_f - B_int / 2.0, delta_f + B_int / 2.0, int(n_f))
    f_prod = np.outer(f1, f2)
    phase = phase_coeff * beta2_eff * f_prod[..., None] * z[None, None, :]
    integrand = p_z[None, None, :] * np.exp(1j * phase)
    inner = np.trapezoid(integrand, z, axis=2)
    inner_sq = np.abs(inner) ** 2
    k_val = np.trapezoid(np.trapezoid(inner_sq, f2, axis=1), f1, axis=0)
    return float(k_val)

def compute_pcfm_nlin(
    system: System,
    profile_path: Path | str,
    launch_powers_w: Optional[np.ndarray] = None,
    config: Optional[PcfmConfig] = None,
    return_components: bool = False,
) -> np.ndarray:
    """Compute per-channel PCFM NLIN power (SCI + XCI, MCI disabled).

    This is the PCFM-style semi-analytical workflow using polynomial fits
    of Raman power profiles and optional numerical/closed-form XCI kernels.

    Parameters
    ----------
    system:
        Active system definition (WDM grid, fiber, baud rate, etc.).
    profile_path:
        Path to saved profile containing channel powers versus ``z``.
    launch_powers_w:
        Optional per-channel launch powers in Watt. If ``None``, values are
        resolved from ``system``.
    config:
        Optional :class:`PcfmConfig` with integration and modeling settings.
    return_components:
        If ``True``, returns total power plus SCI and XCI components.

    Returns
    -------
    np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]
        NLIN power per channel with shape ``(1, n_channels)`` in Watt, or
        ``(total, sci, xci)`` when ``return_components=True``.
    """
    cfg = config or PcfmConfig()
    if cfg.include_mci:
        lg.warning("PCFM MCI disabled by design in this workflow; ignoring include_mci=True.")

    freqs = system.wdm.frequency_grid()
    n_channels = freqs.size
    B_ch = float(system.pulse.baud_rate)
    # lg.warning("Bch = {:.3e} Hz from system pulse baud_rate; ensure this matches profile generation.".format(B_ch))
    L = float(system.fiber_length)

    signal_power_ch_z, z = load_signal_profiles(profile_path, system)
    spp = normalize_spp(signal_power_ch_z, z)
    coeffs = fit_spp_polynomials(z, spp, cfg.degree)

    poly_sums = np.array([poly_sum(coeffs[i]) for i in range(n_channels)], dtype=float)
    p_L = np.array([poly_eval(coeffs[i], 1.0) for i in range(n_channels)], dtype=float)
    p_L = np.maximum(p_L, 0.0)

    beta2 = _beta2_array(system, freqs)
    fc_hz = float(system.center_frequency) if system.center_frequency is not None else float(np.mean(freqs))
    beta_coeffs = _beta_coeffs_from_profile(system, fc_hz) if cfg.use_beta2_eff else None
    _check_loss_profile(system)
    aeff = _aeff_array(system, freqs)

    if launch_powers_w is None:
        launch_powers_w = _launch_powers_from_system(system)
    launch_powers_w = np.asarray(launch_powers_w, dtype=float)
    if launch_powers_w.size != n_channels:
        raise ValueError(
            f"launch_powers_w size {launch_powers_w.size} != n_channels {n_channels}"
        )
    if (not np.all(np.isfinite(launch_powers_w)) or
            np.any(launch_powers_w <= 0) or np.any(launch_powers_w > MAX_POWER_W)):
        raise ValueError("Launch powers are unreasonable for PCFM.")

    g_ch = _to_per_channel_power(launch_powers_w) / B_ch

    g_sci_psd = np.zeros((1, n_channels), dtype=float)
    g_xci_psd = np.zeros((1, n_channels), dtype=float)
    lg.info(
        "PCFM inputs: P_launch_W[min/max]=({:.3e}, {:.3e}), G_ch[min/max]=({:.3e}, {:.3e}), "
        "beta2[min/max]=({:.3e}, {:.3e}), Aeff[min/max]=({:.3e}, {:.3e})".format(
            float(np.min(launch_powers_w)), float(np.max(launch_powers_w)),
            float(np.min(g_ch)), float(np.max(g_ch)),
            float(np.min(beta2)), float(np.max(beta2)),
            float(np.min(aeff)), float(np.max(aeff)),
        )
    )
    lg.info(
        "PCFM SPP stats: p_L[min/max]=({:.3e}, {:.3e}), poly_sum[min/max]=({:.3e}, {:.3e})".format(
            float(np.min(p_L)), float(np.max(p_L)),
            float(np.min(poly_sums)), float(np.max(poly_sums)),
        )
    )
    for i in range(n_channels):
        g_cut = g_ch[i]
        p_cut_L = p_L[i]
        gamma_sci = 2.0 * np.pi * freqs[i] / c * (N2_SIO2 / aeff[i])

        beta2_sci = _beta2_eff(freqs[i], freqs[i], beta_coeffs) if beta_coeffs else beta2[i]
        if cfg.use_numeric_sci:
            k_sci = compute_sci_numeric(
                coeffs[i], L, beta2_sci, B_ch, cfg.n_f, cfg.n_z, cfg.phase_coeff
            )
        else:
            k_sci = compute_sci_numeric(
                coeffs[i], L, beta2_sci, B_ch, cfg.n_f, cfg.n_z, cfg.phase_coeff
            )
        # The normalized profile already shapes the longitudinal kernel. Leave
        # this launch-referenced; the workflow applies P_signal,out/P_launch.
        g_sci = (16.0 / 27.0) * (g_cut ** 3) * (gamma_sci ** 2) * k_sci

        g_xci_sum = 0.0
        for j in range(n_channels):
            if j == i:
                continue
            delta_f = freqs[j] - freqs[i]
            delta_abs = abs(delta_f)
            if delta_abs <= B_ch / 2.0:
                continue
            gamma_xci = 2.0 * np.pi * freqs[i] / c * (2.0 * N2_SIO2 / (aeff[i] + aeff[j]))
            beta2_xci = _beta2_eff(freqs[i], freqs[j], beta_coeffs) if beta_coeffs else beta2[j]
            if cfg.use_numeric_xci:
                k_xci = compute_xci_numeric(
                    coeffs[j], L, beta2_xci, B_ch, delta_f, B_ch,
                    cfg.n_f, cfg.n_z, cfg.phase_coeff
                )
            else:
                log_term = np.abs(np.log((delta_abs + B_ch / 2.0) / (delta_abs - B_ch / 2.0)))
                beta2_eff = max(abs(beta2_xci), MIN_BETA2)
                k_xci = (L / (2.0 * np.pi * beta2_eff)) * log_term * poly_sums[j] # FIXME (just for reference, nothing to fix really): here is the 2 pi at the denominator.  
                # lg.warning("in K_XCI, log: {:.1e} and baud: {:.1e}, Delta f: {:.1e}, Beta2eff: {:.1e}. ".format(log_term, B_ch, delta_f, beta2_eff))
            # omit p(L) scaling in PCFM NLI PSD.
            # 
            g_xci = (32.0 / 27.0) * g_cut * (g_ch[j] ** 2) * (gamma_xci ** 2) * k_xci # this used to be divided by 2, but now no more
            g_xci_sum += g_xci

        g_sci_psd[0, i] = g_sci
        g_xci_psd[0, i] = g_xci_sum # summed over all the interferers
        if i in (0, n_channels // 2, n_channels - 1):
            nli_power = _to_per_polarization_power((g_sci + g_xci_sum) * B_ch)
            lg.info(
                f"PCFM ch {i}: f={freqs[i]*1e-12:.2f} THz, "
                f"p_L={p_cut_L:.3e}, g_cut={g_cut:.3e}, "
                f"gamma_sci={gamma_sci:.3e}, k_sci={k_sci:.3e}, "
                f"g_sci={g_sci:.3e}, g_xci_sum={g_xci_sum:.3e}, "
                f"nli_psd={(g_sci + g_xci_sum):.3e}, nli_power={nli_power:.3e}"
            )

    nlin_psd = g_sci_psd + g_xci_psd
    # Return per-polarization channel NLIN power, still launch-referenced.
    nlin_power = _to_per_polarization_power(nlin_psd * B_ch)
    if return_components:
        return (
            nlin_power,
            _to_per_polarization_power(g_sci_psd * B_ch),
            _to_per_polarization_power(g_xci_psd * B_ch),
        )
    return nlin_power


def compute_gn_numeric(
    system: System,
    profile_path: Path | str,
    launch_powers_w: Optional[np.ndarray] = None,
    n_f: int = 40,
    n_z: int = 200,
    phase_coeff: float = 4.0 * np.pi ** 2,
    use_beta2_eff: bool = True,
    return_components: bool = False,
) -> np.ndarray:
    """Compute per-channel GN NLIN using fully numerical SCI/XCI kernels.

    Parameters
    ----------
    system:
        Active system definition (WDM grid, fiber, baud rate, etc.).
    profile_path:
        Path to saved profile containing channel powers versus ``z``.
    launch_powers_w:
        Optional per-channel launch powers in Watt. If ``None``, values are
        resolved from ``system``.
    n_f:
        Frequency samples per axis for 2D frequency integration.
    n_z:
        Longitudinal samples for profile-based kernels.
    phase_coeff:
        Multiplicative constant in the phase term, typically ``4*pi^2``.
    use_beta2_eff:
        If ``True``, use Eq. (28)-style ``beta2_eff``; otherwise use sampled
        channel ``beta2`` values.
    return_components:
        If ``True``, returns total power plus SCI and XCI components.

    Returns
    -------
    np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]
        NLIN power per channel with shape ``(1, n_channels)`` in Watt, or
        ``(total, sci, xci)`` when ``return_components=True``.
    """
    freqs = system.wdm.frequency_grid()
    n_channels = freqs.size
    B_ch = float(system.pulse.baud_rate)
    L = float(system.fiber_length)

    signal_power_ch_z, z = load_signal_profiles(profile_path, system)
    spp = normalize_spp(signal_power_ch_z, z)
    coeffs = fit_spp_polynomials(z, spp, degree=6)

    beta2 = _beta2_array(system, freqs)
    fc_hz = float(system.center_frequency) if system.center_frequency is not None else float(np.mean(freqs))
    beta_coeffs = _beta_coeffs_from_profile(system, fc_hz) if use_beta2_eff else None
    _check_loss_profile(system)
    aeff = _aeff_array(system, freqs)

    if launch_powers_w is None:
        launch_powers_w = _launch_powers_from_system(system)
    launch_powers_w = np.asarray(launch_powers_w, dtype=float)
    if launch_powers_w.size != n_channels:
        raise ValueError(
            f"launch_powers_w size {launch_powers_w.size} != n_channels {n_channels}"
        )
    if (not np.all(np.isfinite(launch_powers_w)) or
            np.any(launch_powers_w <= 0) or np.any(launch_powers_w > MAX_POWER_W)):
        raise ValueError("Launch powers are unreasonable for GN numeric.")

    g_ch = launch_powers_w / B_ch
    g_sci_psd = np.zeros((1, n_channels), dtype=float)
    g_xci_psd = np.zeros((1, n_channels), dtype=float)

    for i in range(n_channels):
        g_cut = g_ch[i]
        gamma_sci = 2.0 * np.pi * freqs[i] / c * (N2_SIO2 / aeff[i])
        beta2_sci = _beta2_eff(freqs[i], freqs[i], beta_coeffs) if beta_coeffs else beta2[i]
        k_sci = compute_sci_numeric(coeffs[i], L, beta2_sci, B_ch, n_f, n_z, phase_coeff)
        # Keep launch-referenced output here; workflow applies the endpoint
        # signal-power ratio consistently with TD and PCFM. # FIXME check
        g_sci = (16.0 / 27.0) * (g_cut ** 3) * (gamma_sci ** 2) * k_sci

        g_xci_sum = 0.0
        for j in range(n_channels):
            if j == i:
                continue
            delta_f = freqs[j] - freqs[i]
            if abs(delta_f) <= B_ch / 2.0:
                continue
            gamma_xci = 2.0 * np.pi * freqs[i] / c * (2.0 * N2_SIO2 / (aeff[i] + aeff[j]))
            beta2_xci = _beta2_eff(freqs[i], freqs[j], beta_coeffs) if beta_coeffs else beta2[j]
            k_xci = compute_xci_numeric(
                coeffs[j], L, beta2_xci, B_ch, delta_f, B_ch, n_f, n_z, phase_coeff
            )
            # Keep launch-referenced output here; workflow applies the endpoint
            # signal-power ratio consistently with TD and PCFM.
            g_xci = (32.0 / 27.0) * g_cut * (g_ch[j] ** 2) * (gamma_xci ** 2) * k_xci
            g_xci_sum += g_xci

        g_sci_psd[0, i] = g_sci
        g_xci_psd[0, i] = g_xci_sum

    nlin_psd = g_sci_psd + g_xci_psd
    nlin_power = _to_per_polarization_power(nlin_psd * B_ch)
    if return_components:
        return (
            nlin_power,
            _to_per_polarization_power(g_sci_psd * B_ch),
            _to_per_polarization_power(g_xci_psd * B_ch),
        )
    return nlin_power


def compute_gn_direct(
    system: System,
    profile_path: Path | str,
    launch_powers_w: Optional[np.ndarray] = None,
    n_f: int = 40,
    phase_coeff: float = 4.0 * np.pi ** 2,
    use_beta2_eff: bool = True,
    return_components: bool = False,
) -> np.ndarray:
    """Compute per-channel GN NLIN using direct sampled profiles ``p(z)``.

    Compared to :func:`compute_gn_numeric`, this path avoids polynomial fitting
    and integrates kernels directly on the stored longitudinal samples.

    Parameters
    ----------
    system:
        Active system definition (WDM grid, fiber, baud rate, etc.).
    profile_path:
        Path to saved profile containing channel powers versus ``z``.
    launch_powers_w:
        Optional per-channel launch powers in Watt. If ``None``, values are
        resolved from ``system``.
    n_f:
        Frequency samples per axis for 2D frequency integration.
    phase_coeff:
        Multiplicative constant in the phase term, typically ``4*pi^2``.
    use_beta2_eff:
        If ``True``, use Eq. (28)-style ``beta2_eff``; otherwise use sampled
        channel ``beta2`` values.
    return_components:
        If ``True``, returns total power plus SCI and XCI components.

    Returns
    -------
    np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]
        NLIN power per channel with shape ``(1, n_channels)`` in Watt, or
        ``(total, sci, xci)`` when ``return_components=True``.
    """
    freqs = system.wdm.frequency_grid()
    n_channels = freqs.size
    B_ch = float(system.pulse.baud_rate)

    signal_power_ch_z, z = load_signal_profiles(profile_path, system)
    spp = normalize_spp(signal_power_ch_z, z)

    beta2 = _beta2_array(system, freqs)
    fc_hz = float(system.center_frequency) if system.center_frequency is not None else float(np.mean(freqs))
    beta_coeffs = _beta_coeffs_from_profile(system, fc_hz) if use_beta2_eff else None
    _check_loss_profile(system)
    aeff = _aeff_array(system, freqs)

    if launch_powers_w is None:
        launch_powers_w = _launch_powers_from_system(system)
    launch_powers_w = np.asarray(launch_powers_w, dtype=float)
    if launch_powers_w.size != n_channels:
        raise ValueError(
            f"launch_powers_w size {launch_powers_w.size} != n_channels {n_channels}"
        )
    if (not np.all(np.isfinite(launch_powers_w)) or
            np.any(launch_powers_w <= 0) or np.any(launch_powers_w > MAX_POWER_W)):
        raise ValueError("Launch powers are unreasonable for GN direct.")

    g_ch = launch_powers_w / B_ch
    g_sci_psd = np.zeros((1, n_channels), dtype=float)
    g_xci_psd = np.zeros((1, n_channels), dtype=float)

    for i in range(n_channels):
        g_cut = g_ch[i]
        gamma_sci = 2.0 * np.pi * freqs[i] / c * (N2_SIO2 / aeff[i])
        beta2_sci = _beta2_eff(freqs[i], freqs[i], beta_coeffs) if beta_coeffs else beta2[i]
        k_sci = compute_sci_numeric_direct(spp[i], z, beta2_sci, B_ch, n_f, phase_coeff)
        # Keep launch-referenced output here; workflow applies the endpoint
        # signal-power ratio consistently with TD and PCFM.
        g_sci = (16.0 / 27.0) * (g_cut ** 3) * (gamma_sci ** 2) * k_sci

        g_xci_sum = 0.0
        for j in range(n_channels):
            if j == i:
                continue
            delta_f = freqs[j] - freqs[i]
            if abs(delta_f) <= B_ch / 2.0:
                continue
            gamma_xci = 2.0 * np.pi * freqs[i] / c * (2.0 * N2_SIO2 / (aeff[i] + aeff[j]))
            beta2_xci = _beta2_eff(freqs[i], freqs[j], beta_coeffs) if beta_coeffs else beta2[j]
            k_xci = compute_xci_numeric_direct(
                spp[j], z, beta2_xci, B_ch, delta_f, B_ch, n_f, phase_coeff
            )
            # Keep launch-referenced output here; workflow applies the endpoint
            # signal-power ratio consistently with TD and PCFM.
            g_xci = (32.0 / 27.0) * g_cut * (g_ch[j] ** 2) * (gamma_xci ** 2) * k_xci
            g_xci_sum += g_xci

        g_sci_psd[0, i] = g_sci
        g_xci_psd[0, i] = g_xci_sum

    nlin_psd = g_sci_psd + g_xci_psd
    nlin_power = _to_per_polarization_power(nlin_psd * B_ch)
    if return_components:
        return (
            nlin_power,
            _to_per_polarization_power(g_sci_psd * B_ch),
            _to_per_polarization_power(g_xci_psd * B_ch),
        )
    return nlin_power


__all__ = [
    "PcfmConfig",
    "compute_pcfm_nlin",
    "compute_gn_numeric",
    "compute_gn_direct",
    "load_signal_profiles",
    "normalize_spp",
    "fit_spp_polynomials",
]
