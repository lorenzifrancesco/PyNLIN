"""Low-level collision-resolved XPM time-domain kernels.

This module is intentionally independent from the production TD-NLIN fitted
estimators. It provides benchmark-quality direct and FFT-sliding-overlap
evaluators for the fundamental pulse-overlap coefficients.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import scipy.integrate

from pynlin.pulses import Pulse


@dataclass(frozen=True)
class XPMKernelGrid:
    """Numerical grid and physical parameters for XPM kernel evaluation."""

    t: np.ndarray
    z: np.ndarray
    dt: float
    z_weights: np.ndarray
    baud_rate: float
    dgd: float
    gvda: float
    gvdb: float
    convention: str = "Q(tau)=int C(t)D(t-tau)dt"


@dataclass(frozen=True)
class XPMKernelResult:
    """Collision-resolved kernel table for X[h,r,m] = X[h,m+r,m]."""

    X: np.ndarray
    h_values: np.ndarray
    r_values: np.ndarray
    m_values: np.ndarray
    metadata: dict[str, object]


def _uniform_dt(t: np.ndarray) -> float:
    t = np.asarray(t, dtype=float).reshape(-1)
    if t.size < 2:
        raise ValueError("time grid must contain at least two samples")
    dt_values = np.diff(t)
    dt = float(dt_values[0])
    if dt <= 0.0 or not np.allclose(dt_values, dt, rtol=1e-9, atol=1e-15):
        raise ValueError("time grid must be uniformly spaced and increasing")
    return dt


def _z_trapezoid_weights(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float).reshape(-1)
    if z.size == 0:
        raise ValueError("z grid must not be empty")
    if z.size == 1:
        return np.ones(1, dtype=float)
    dz = np.diff(z)
    if np.any(dz < 0.0):
        raise ValueError("z grid must be increasing")
    weights = np.empty_like(z, dtype=float)
    weights[0] = dz[0] / 2.0
    weights[-1] = dz[-1] / 2.0
    if z.size > 2:
        weights[1:-1] = (z[2:] - z[:-2]) / 2.0
    return weights


def kernel_grid_from_pulse(
    pulse: Pulse,
    z: np.ndarray,
    *,
    dgd: float,
    gvda: float,
    gvdb: float,
) -> XPMKernelGrid:
    """Build an :class:`XPMKernelGrid` from a pulse and longitudinal samples."""
    _, t = pulse.data()
    t = np.asarray(t, dtype=float).reshape(-1)
    z = np.asarray(z, dtype=float).reshape(-1)
    return XPMKernelGrid(
        t=t,
        z=z,
        dt=_uniform_dt(t),
        z_weights=_z_trapezoid_weights(z),
        baud_rate=float(pulse.baud_rate),
        dgd=float(dgd),
        gvda=float(gvda),
        gvdb=float(gvdb),
    )

## FIXME this code is repeated wrt the other code
def _pulse_spectrum(pulse: Pulse) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    g, t = pulse.data()
    g = np.asarray(g, dtype=complex).reshape(-1)
    t = np.asarray(t, dtype=float).reshape(-1)
    if g.size != t.size:
        raise ValueError("pulse samples and time grid have different lengths")
    dt = _uniform_dt(t)
    omega = np.fft.fftshift(2.0 * np.pi * np.fft.fftfreq(g.size, d=dt))
    gf = np.fft.fftshift(np.fft.fft(g))
    return gf, omega, t, dt


def linear_pulse_at_z_from_spectrum(gf: np.ndarray, omega: np.ndarray, gvd: float, z: float) -> np.ndarray:
    """Return a linearly dispersed pulse using the legacy TD sign convention."""
    phase = np.exp((-1j * float(gvd) / 2.0) * float(z) * omega * omega)
    return np.fft.ifft(np.fft.fftshift(gf * phase))


def linear_pulse_at_z(pulse: Pulse, z: float, gvd: float) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(g_z, t)`` for a pulse propagated by chromatic dispersion."""
    gf, omega, t, _ = _pulse_spectrum(pulse)
    return linear_pulse_at_z_from_spectrum(gf, omega, gvd, z), t


def shift_samples(values: np.ndarray, delay: float, t: np.ndarray) -> np.ndarray:
    """Sample ``values(t-delay)`` on ``t`` with zero outside the known grid."""
    values = np.asarray(values, dtype=complex).reshape(-1)
    t = np.asarray(t, dtype=float).reshape(-1)
    query = t - float(delay)
    real = np.interp(query, t, values.real, left=0.0, right=0.0)
    imag = np.interp(query, t, values.imag, left=0.0, right=0.0)
    return real + 1j * imag


def build_C(g_a_z: np.ndarray, h: int, baud_rate: float, t: np.ndarray) -> np.ndarray:
    """Build ``C_h(t) = g_A,z*(t) g_A,z(t-hT)``."""
    delay = int(h) / float(baud_rate)
    return np.conj(g_a_z) * shift_samples(g_a_z, delay, t)


def build_D(g_b_z: np.ndarray, r: int, baud_rate: float, t: np.ndarray) -> np.ndarray:
    """Build ``D_r(t) = g_B,z*(t-rT) g_B,z(t)``."""
    delay = int(r) / float(baud_rate)
    return np.conj(shift_samples(g_b_z, delay, t)) * g_b_z


def _next_pow_two_at_least(n: int) -> int:
    return 1 << (int(n) - 1).bit_length()


def sliding_overlap_fft(C: np.ndarray, D: np.ndarray, dt: float, *, n_fft: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Compute ``Q(tau)=int C(t)D(t-tau)dt`` using linear FFT convolution.

    No conjugation is applied to either operand. Returned lags are physical
    delays in seconds and follow ``tau = k*dt``.
    """
    C = np.asarray(C, dtype=complex).reshape(-1)
    D = np.asarray(D, dtype=complex).reshape(-1)
    if C.size != D.size:
        raise ValueError("C and D must have the same length")
    n_linear = 2 * C.size - 1
    if n_fft is None:
        n_fft = _next_pow_two_at_least(n_linear)
    if n_fft < n_linear:
        raise ValueError(f"n_fft={n_fft} is too small for a linear overlap of length {n_linear}")
    conv = np.fft.ifft(np.fft.fft(C, n_fft) * np.fft.fft(D[::-1], n_fft))[:n_linear]
    lags = (np.arange(n_linear, dtype=float) - (C.size - 1)) * float(dt)
    return lags, conv * float(dt)


def sample_correlation(lags: np.ndarray, values: np.ndarray, tau_values: np.ndarray) -> np.ndarray:
    """Linearly sample a complex overlap array at arbitrary delays."""
    lags = np.asarray(lags, dtype=float).reshape(-1)
    values = np.asarray(values, dtype=complex).reshape(-1)
    tau_values = np.asarray(tau_values, dtype=float)
    real = np.interp(tau_values, lags, values.real, left=0.0, right=0.0)
    imag = np.interp(tau_values, lags, values.imag, left=0.0, right=0.0)
    return real + 1j * imag


def compute_xpm_time_integrand_direct(
    pulse: Pulse,
    z: np.ndarray,
    *,
    h: int,
    k: int,
    m: int,
    dgd: float,
    gvda: float,
    gvdb: float,
) -> np.ndarray:
    """Directly compute the time integral integrand as a function of ``z``."""
    grid = kernel_grid_from_pulse(pulse, z, dgd=dgd, gvda=gvda, gvdb=gvdb)
    gf, omega, _, _ = _pulse_spectrum(pulse)
    out = np.empty(grid.z.size, dtype=complex)
    T = 1.0 / grid.baud_rate
    for idx, z_val in enumerate(grid.z):
        g_a = linear_pulse_at_z_from_spectrum(gf, omega, grid.gvda, float(z_val))
        g_b = linear_pulse_at_z_from_spectrum(gf, omega, grid.gvdb, float(z_val))
        delay_k = int(k) * T + grid.dgd * float(z_val)
        delay_m = int(m) * T + grid.dgd * float(z_val)
        integrand = (
            np.conj(g_a)
            * shift_samples(g_a, int(h) * T, grid.t)
            * np.conj(shift_samples(g_b, delay_k, grid.t))
            * shift_samples(g_b, delay_m, grid.t)
        )
        out[idx] = scipy.integrate.trapezoid(integrand, dx=grid.dt)
    return out


def compute_xpm_coefficient_direct(
    pulse: Pulse,
    z: np.ndarray,
    *,
    h: int,
    k: int,
    m: int,
    dgd: float,
    gvda: float,
    gvdb: float,
    amplification_function: Callable[[np.ndarray], np.ndarray] | None = None,
) -> complex:
    """Directly evaluate the full z-integrated ``X_{h,k,m}`` coefficient."""
    z = np.asarray(z, dtype=float).reshape(-1)
    I = compute_xpm_time_integrand_direct(
        pulse,
        z,
        h=h,
        k=k,
        m=m,
        dgd=dgd,
        gvda=gvda,
        gvdb=gvdb,
    )
    if amplification_function is not None:
        I = I * amplification_function(z)
    return complex(scipy.integrate.trapezoid(I, z))


def compute_x0mm_time_integrals_fft(
    pulse: Pulse,
    z: np.ndarray,
    m_values: np.ndarray,
    *,
    dgd: float,
    gvda: float,
    gvdb: float,
) -> np.ndarray:
    """Compute low-level ``X0mm`` time integrals for all ``m`` via FFT overlaps."""
    grid = kernel_grid_from_pulse(pulse, z, dgd=dgd, gvda=gvda, gvdb=gvdb)
    m_values = np.asarray(m_values, dtype=int).reshape(-1)
    gf, omega, _, _ = _pulse_spectrum(pulse)
    out = np.empty((m_values.size, grid.z.size), dtype=complex)
    for z_idx, z_val in enumerate(grid.z):
        g_a = linear_pulse_at_z_from_spectrum(gf, omega, grid.gvda, float(z_val))
        g_b = linear_pulse_at_z_from_spectrum(gf, omega, grid.gvdb, float(z_val))
        C = np.abs(g_a) ** 2
        D = np.abs(g_b) ** 2
        lags, Q = sliding_overlap_fft(C, D, grid.dt)
        tau = m_values / grid.baud_rate + grid.dgd * float(z_val)
        out[:, z_idx] = sample_correlation(lags, Q, tau)
    return out


def compute_x0mm_fft(
    pulse: Pulse,
    z: np.ndarray,
    m_values: np.ndarray,
    *,
    dgd: float,
    gvda: float,
    gvdb: float,
    amplification_function: Callable[[np.ndarray], np.ndarray] | None = None,
) -> np.ndarray:
    """Compute z-integrated ``X0mm`` coefficients for all ``m`` via FFT overlaps."""
    z = np.asarray(z, dtype=float).reshape(-1)
    I = compute_x0mm_time_integrals_fft(
        pulse,
        z,
        m_values,
        dgd=dgd,
        gvda=gvda,
        gvdb=gvdb,
    )
    if amplification_function is not None:
        I = I * amplification_function(z)[None, :]
    return scipy.integrate.trapezoid(I, z, axis=1)


def compute_xpm_kernel_fft(
    pulse: Pulse,
    z: np.ndarray,
    h_values: np.ndarray,
    r_values: np.ndarray,
    m_values: np.ndarray,
    *,
    dgd: float,
    gvda: float,
    gvdb: float,
    amplification_function: Callable[[np.ndarray], np.ndarray] | None = None,
) -> XPMKernelResult:
    """Compute a finite ``X[h,r,m] = X_{h,m+r,m}`` table by FFT overlaps.

    This is a low-level benchmark evaluator. It preserves collision identities
    and does not change the production TD-NLIN fitted estimators.
    """
    grid = kernel_grid_from_pulse(pulse, z, dgd=dgd, gvda=gvda, gvdb=gvdb)
    h_values = np.asarray(h_values, dtype=int).reshape(-1)
    r_values = np.asarray(r_values, dtype=int).reshape(-1)
    m_values = np.asarray(m_values, dtype=int).reshape(-1)
    if h_values.size == 0 or r_values.size == 0 or m_values.size == 0:
        raise ValueError("h_values, r_values, and m_values must all be non-empty")

    amp = (
        np.ones_like(grid.z, dtype=complex)
        if amplification_function is None
        else np.asarray(amplification_function(grid.z), dtype=complex).reshape(-1)
    )
    if amp.size != grid.z.size:
        raise ValueError("amplification_function must return one value per z sample")

    gf, omega, _, _ = _pulse_spectrum(pulse)
    X = np.zeros((h_values.size, r_values.size, m_values.size), dtype=complex)

    for z_idx, z_val in enumerate(grid.z):
        g_a = linear_pulse_at_z_from_spectrum(gf, omega, grid.gvda, float(z_val))
        g_b = linear_pulse_at_z_from_spectrum(gf, omega, grid.gvdb, float(z_val))
        C_by_h = [build_C(g_a, int(h), grid.baud_rate, grid.t) for h in h_values]
        D_by_r = [build_D(g_b, int(r), grid.baud_rate, grid.t) for r in r_values]
        tau = m_values / grid.baud_rate + grid.dgd * float(z_val)
        z_weight = grid.z_weights[z_idx] * amp[z_idx]
        for h_idx, C_h in enumerate(C_by_h):
            for r_idx, D_r in enumerate(D_by_r):
                lags, Q = sliding_overlap_fft(C_h, D_r, grid.dt)
                X[h_idx, r_idx, :] += z_weight * sample_correlation(lags, Q, tau)

    metadata = {
        "dgd": grid.dgd,
        "gvda": grid.gvda,
        "gvdb": grid.gvdb,
        "baud_rate": grid.baud_rate,
        "z_min": float(grid.z[0]),
        "z_max": float(grid.z[-1]),
        "n_z": int(grid.z.size),
        "n_t": int(grid.t.size),
        "dt": grid.dt,
        "h_values": h_values.copy(),
        "r_values": r_values.copy(),
        "m_values": m_values.copy(),
        "convention": "X[h,r,m]=X_{h,m+r,m}; Q(tau)=int C(t)D(t-tau)dt",
    }
    return XPMKernelResult(
        X=X,
        h_values=h_values,
        r_values=r_values,
        m_values=m_values,
        metadata=metadata,
    )


def classify_collision(h: int, r: int, m: int) -> str:
    """Classify ``X_{h,k,m}``, represented as ``X[h,r,m]`` with ``k=m+r``.

    Classes follow the collision identities:

    - ``2pc``: ``h=0, k=m``.
    - ``3pc``: exactly one symbol-index coincidence (including ``hmh``).
    - ``4pc``: all three offsets ``h``, ``k``, and ``m`` are distinct.
    """
    h = int(h)
    r = int(r)
    m = int(m)
    k = m + r
    if h == 0 and k == m:
        return "2pc"
    if h == 0 or k == m or h == k or h == m:
        return "3pc"
    return "4pc"


def collision_class_masks(h_values: np.ndarray, r_values: np.ndarray, m_values: np.ndarray) -> dict[str, np.ndarray]:
    """Return exhaustive boolean masks for ``2pc``, ``3pc``, and ``4pc`` entries."""
    h_values = np.asarray(h_values, dtype=int).reshape(-1)
    r_values = np.asarray(r_values, dtype=int).reshape(-1)
    m_values = np.asarray(m_values, dtype=int).reshape(-1)
    masks = {
        "2pc": np.zeros((h_values.size, r_values.size, m_values.size), dtype=bool),
        "3pc": np.zeros((h_values.size, r_values.size, m_values.size), dtype=bool),
        "4pc": np.zeros((h_values.size, r_values.size, m_values.size), dtype=bool),
    }
    for h_idx, h in enumerate(h_values):
        for r_idx, r in enumerate(r_values):
            for m_idx, m in enumerate(m_values):
                masks[classify_collision(int(h), int(r), int(m))][h_idx, r_idx, m_idx] = True
    return masks


__all__ = [
    "XPMKernelGrid",
    "XPMKernelResult",
    "build_C",
    "build_D",
    "classify_collision",
    "collision_class_masks",
    "compute_x0mm_fft",
    "compute_x0mm_time_integrals_fft",
    "compute_xpm_kernel_fft",
    "compute_xpm_coefficient_direct",
    "compute_xpm_time_integrand_direct",
    "kernel_grid_from_pulse",
    "linear_pulse_at_z",
    "sample_correlation",
    "shift_samples",
    "sliding_overlap_fft",
]
