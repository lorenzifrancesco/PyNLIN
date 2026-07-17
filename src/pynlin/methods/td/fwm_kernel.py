"""Reference generic FWM time-domain kernels.

This module intentionally starts with a direct evaluator.  It keeps the
frequency mismatch term explicit so off-grid wideband FWM combinations are
computed and suppressed by the oscillatory time integral instead of being
filtered by channel-center matching.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import scipy.integrate
from loguru import logger as lg

from pynlin.methods.td.xpm_kernel import (
    _pulse_spectrum,
    _uniform_dt,
    linear_pulse_at_z_from_spectrum,
    shift_samples,
)
from pynlin.pulses import Pulse


# Default parameters for z-grid auto-refinement.
_DEFAULT_MIN_PTS_PER_PERIOD: float = 3.0
_DEFAULT_MAX_Z_POINTS: int = 2000
_DEFAULT_MAX_WALKOFF_FRACTION: float = 0.8


@dataclass(frozen=True)
class FWMChannels:
    """Channel/mode parameters for ``d* a b c*`` FWM coefficients."""

    omega_a: float
    omega_b: float
    omega_c: float
    omega_d: float
    beta0_a: float = 0.0
    beta0_b: float = 0.0
    beta0_c: float = 0.0
    beta0_d: float = 0.0
    beta1_a: float = 0.0
    beta1_b: float = 0.0
    beta1_c: float = 0.0
    beta1_d: float = 0.0
    gvd_a: float = 0.0
    gvd_b: float = 0.0
    gvd_c: float = 0.0
    gvd_d: float = 0.0
    beta3_a: float = 0.0
    beta3_b: float = 0.0
    beta3_c: float = 0.0
    beta3_d: float = 0.0
    beta4_a: float = 0.0
    beta4_b: float = 0.0
    beta4_c: float = 0.0
    beta4_d: float = 0.0

    @property
    def delta_omega(self) -> float:
        """Return ``omega_a + omega_b - omega_c - omega_d``."""
        return float(self.omega_a + self.omega_b - self.omega_c - self.omega_d)

    @property
    def delta_beta0(self) -> float:
        """Return ``beta0_a + beta0_b - beta0_c - beta0_d``."""
        return float(self.beta0_a + self.beta0_b - self.beta0_c - self.beta0_d)


@dataclass(frozen=True)
class FWMKernelResult:
    """Collision-resolved generic FWM table."""

    X: np.ndarray
    h_values: np.ndarray
    k_values: np.ndarray
    m_values: np.ndarray
    metadata: dict[str, object]


def _amplification_values(
    z: np.ndarray,
    amplification_function: Callable[[np.ndarray], np.ndarray] | None,
) -> np.ndarray:
    if amplification_function is None:
        return np.ones_like(z, dtype=complex)
    amp = np.asarray(amplification_function(z), dtype=complex).reshape(-1)
    if amp.size != z.size:
        raise ValueError("amplification_function must return one value per z sample")
    return amp


def _metadata(
    t: np.ndarray,
    z: np.ndarray,
    channels: FWMChannels,
    *,
    include_frequency_mismatch: bool,
    include_phase_mismatch: bool,
) -> dict[str, object]:
    delta_omega = channels.delta_omega
    delta_beta0 = channels.delta_beta0
    return {
        "omega_a": float(channels.omega_a),
        "omega_b": float(channels.omega_b),
        "omega_c": float(channels.omega_c),
        "omega_d": float(channels.omega_d),
        "delta_omega": float(delta_omega),
        "delta_beta0": float(delta_beta0),
        "frequency_matched": bool(np.isclose(delta_omega, 0.0, rtol=0.0, atol=1e-12)),
        "include_frequency_mismatch": bool(include_frequency_mismatch),
        "include_phase_mismatch": bool(include_phase_mismatch),
        "beta0_a": float(channels.beta0_a),
        "beta0_b": float(channels.beta0_b),
        "beta0_c": float(channels.beta0_c),
        "beta0_d": float(channels.beta0_d),
        "beta1_a": float(channels.beta1_a),
        "beta1_b": float(channels.beta1_b),
        "beta1_c": float(channels.beta1_c),
        "beta1_d": float(channels.beta1_d),
        "gvd_a": float(channels.gvd_a),
        "gvd_b": float(channels.gvd_b),
        "gvd_c": float(channels.gvd_c),
        "gvd_d": float(channels.gvd_d),
        "beta3_a": float(channels.beta3_a),
        "beta3_b": float(channels.beta3_b),
        "beta3_c": float(channels.beta3_c),
        "beta3_d": float(channels.beta3_d),
        "beta4_a": float(channels.beta4_a),
        "beta4_b": float(channels.beta4_b),
        "beta4_c": float(channels.beta4_c),
        "beta4_d": float(channels.beta4_d),
        "n_t": int(t.size),
        "n_z": int(z.size),
        "dt": _uniform_dt(t),
        "z_min": float(z[0]),
        "z_max": float(z[-1]),
        "convention": (
            "X[h,k,m]=int dz dt g_d*(t) g_a(t-hT-tau_a) "
            "g_b(t-kT-tau_b) g_c*(t-mT-tau_c) "
            "exp(i delta_beta0 z - i delta_omega t)"
        ),
    }


def compute_fwm_coefficient_direct(
    pulse: Pulse,
    z: np.ndarray,
    *,
    h: int,
    k: int,
    m: int,
    channels: FWMChannels,
    amplification_function: Callable[[np.ndarray], np.ndarray] | None = None,
    include_frequency_mismatch: bool = True,
    include_phase_mismatch: bool = True,
) -> complex:
    """Directly evaluate one generic wideband FWM coefficient.

    The nonlinear product convention is ``g_d* g_a g_b g_c*``.  The explicit
    factor ``exp(-1j * delta_omega * t)`` lets non-center-matched channel
    tuples contribute according to their actual pulse spectra and numerical
    time window.
    """
    if any(
        value != 0.0
        for value in (
            channels.beta3_a,
            channels.beta3_b,
            channels.beta3_c,
            channels.beta3_d,
            channels.beta4_a,
            channels.beta4_b,
            channels.beta4_c,
            channels.beta4_d,
        )
    ):
        raise NotImplementedError(
            "beta3/beta4 are supported by the Dar frequency-domain FWM estimator, "
            "not by the direct time-domain pulse propagator"
        )
    z = np.asarray(z, dtype=float).reshape(-1)
    if z.size == 0:
        raise ValueError("z grid must not be empty")

    gf, omega, t, dt = _pulse_spectrum(pulse)
    amp = _amplification_values(z, amplification_function)
    T = 1.0 / float(pulse.baud_rate)
    delta_omega = channels.delta_omega
    delta_beta0 = channels.delta_beta0
    time_phase = np.exp(-1j * delta_omega * t) if include_frequency_mismatch else 1.0

    integrand_z = np.empty(z.size, dtype=complex)
    for z_idx, z_val in enumerate(z):
        z_float = float(z_val)
        g_a = linear_pulse_at_z_from_spectrum(gf, omega, channels.gvd_a, z_float)
        g_b = linear_pulse_at_z_from_spectrum(gf, omega, channels.gvd_b, z_float)
        g_c = linear_pulse_at_z_from_spectrum(gf, omega, channels.gvd_c, z_float)
        g_d = linear_pulse_at_z_from_spectrum(gf, omega, channels.gvd_d, z_float)

        tau_a = (channels.beta1_a - channels.beta1_d) * z_float
        tau_b = (channels.beta1_b - channels.beta1_d) * z_float
        tau_c = (channels.beta1_c - channels.beta1_d) * z_float
        z_phase = np.exp(1j * delta_beta0 * z_float) if include_phase_mismatch else 1.0

        integrand_t = (
            np.conj(g_d)
            * shift_samples(g_a, int(h) * T + tau_a, t)
            * shift_samples(g_b, int(k) * T + tau_b, t)
            * np.conj(shift_samples(g_c, int(m) * T + tau_c, t))
            * time_phase
        )
        integrand_z[z_idx] = (
            amp[z_idx]
            * z_phase
            * scipy.integrate.trapezoid(integrand_t, dx=dt)
        )

    return complex(scipy.integrate.trapezoid(integrand_z, z))


def _effective_llw_and_period(
    channels: FWMChannels,
    length: float,
    baud_rate: float,
) -> tuple[float, float, float]:
    """Return (LLW, phase_period, max_walkoff).

    LLW = length * baud_rate * max_beta1_spread  (walk-off in symbol units)
    phase_period = 2π / |delta_beta0|  (z-oscillation period from phase mismatch)
    max_walkoff = max_beta1_spread * length  (total physical walk-off)
    """
    beta1s = [channels.beta1_a, channels.beta1_b, channels.beta1_c, channels.beta1_d]
    beta1_spread = float(max(beta1s) - min(beta1s))
    llw = length * baud_rate * beta1_spread if beta1_spread > 0 else 0.0
    db0 = float(channels.delta_beta0)
    period = 2.0 * np.pi / abs(db0) if abs(db0) > 0.0 else float("inf")
    walkoff = beta1_spread * length
    return llw, period, walkoff


def _check_fwm_discretization(
    z: np.ndarray,
    channels: FWMChannels,
    length: float,
    baud_rate: float,
    *,
    min_pts_per_period: float = _DEFAULT_MIN_PTS_PER_PERIOD,
    max_walkoff_fraction: float = _DEFAULT_MAX_WALKOFF_FRACTION,
    action: str = "warn",
) -> None:
    """Warn or assert when the FWM z-grid is too coarse."""
    n_z = z.size
    dz = float(np.mean(np.diff(z)))
    llw, phase_period, walkoff = _effective_llw_and_period(channels, length, baud_rate)

    # Phase-oscillation resolution
    if np.isfinite(phase_period):
        n_per_period = float(n_z) * phase_period / length if length > 0 else float("inf")
    else:
        n_per_period = float("inf")

    # Walk-off resolution
    pts_per_collision = float(n_z) / llw if llw > 0 else float("inf")

    # Walk-off fraction of time window
    time_window = float(z[-1] - z[0])
    total_walkoff = walkoff
    walkoff_fraction = total_walkoff / time_window if time_window > 0 else float("inf")

    issues: list[str] = []
    if n_per_period < min_pts_per_period and np.isfinite(phase_period):
        issues.append(
            f"z-resolution: pts/phase-period={n_per_period:.2f} < {min_pts_per_period:.0f} "
            f"(delta_beta0*L/2π={abs(channels.delta_beta0)*length/(2*np.pi):.2f} periods, "
            f"n_z={n_z}, dz={dz:.3g}m)"
        )
    if pts_per_collision < min_pts_per_period:
        issues.append(
            f"z-resolution: pts/collision={pts_per_collision:.2f} < {min_pts_per_period:.0f} "
            f"(LLW={llw:.1f}, n_z={n_z}, dz={dz:.3g}m)"
        )
    if walkoff_fraction > max_walkoff_fraction:
        issues.append(
            f"time-window wrap: walk-off consumes {100*walkoff_fraction:.0f}% of "
            f"time window (walk-off={walkoff:.2e}m, max allowed {max_walkoff_fraction*100:.0f}%)"
        )

    if action == "silent":
        return
    msg = " | ".join(issues) if issues else ""
    if not msg:
        return
    if action == "assert":
        raise ValueError(f"FWM discretisation violation: {msg}")
    lg.warning(f"FWM discretisation issue: {msg}")


def _refine_fwm_z_grid(
    z: np.ndarray,
    channels: FWMChannels,
    length: float,
    baud_rate: float,
    *,
    min_pts_per_period: float = _DEFAULT_MIN_PTS_PER_PERIOD,
    max_z_points: int = _DEFAULT_MAX_Z_POINTS,
) -> np.ndarray:
    """Densify the ``z`` grid so that both phase-oscillation and walk-off are resolved."""
    n_z = z.size
    llw, phase_period, _ = _effective_llw_and_period(channels, length, baud_rate)

    required_nz = n_z
    if np.isfinite(phase_period) and phase_period > 0.0:
        needed = int(np.ceil(min_pts_per_period * length / phase_period))
        required_nz = max(required_nz, needed)
    if llw > 0.0:
        needed = int(np.ceil(min_pts_per_period * llw))
        required_nz = max(required_nz, needed)

    required_nz = min(max_z_points, required_nz)
    if required_nz <= n_z:
        return z

    new_z = np.linspace(float(z[0]), float(z[-1]), required_nz)
    print(f"  [fwm auto-refine z] {n_z} → {required_nz} pts  "
          f"(LLW={llw:.1f}, phase_period={phase_period:.2e}m)")
    return new_z


def compute_fwm_kernel_direct(
    pulse: Pulse,
    z: np.ndarray,
    h_values: np.ndarray,
    k_values: np.ndarray,
    m_values: np.ndarray,
    *,
    channels: FWMChannels,
    amplification_function: Callable[[np.ndarray], np.ndarray] | None = None,
    include_frequency_mismatch: bool = True,
    include_phase_mismatch: bool = True,
    auto_refine: bool = False,
    discretization_action: str = "warn",
    min_pts_per_period: float = _DEFAULT_MIN_PTS_PER_PERIOD,
    max_z_points: int = _DEFAULT_MAX_Z_POINTS,
    max_walkoff_fraction: float = _DEFAULT_MAX_WALKOFF_FRACTION,
) -> FWMKernelResult:
    """Compute a finite ``X[h,k,m]`` table using the direct FWM evaluator.

    Parameters
    ----------
    auto_refine : bool
        Automatically densify the ``z`` grid to resolve phase oscillations and
        walk-off (capped at ``max_z_points``).
    discretization_action : ``"warn"``, ``"assert"``, or ``"silent"``
        How to handle any remaining discretisation violations after
        auto-refinement.
    min_pts_per_period : float
        Minimum acceptable number of ``z`` samples per phase-oscillation period
        (also used for the walk-off collision criterion).
    max_z_points : int
        Hard cap on the number of ``z`` points after auto-refinement.
    max_walkoff_fraction : float
        Maximum acceptable fraction of the time window consumed by walk-off.
    """
    z = np.asarray(z, dtype=float).reshape(-1)
    if z.size == 0:
        raise ValueError("z grid must not be empty")
    h_values = np.asarray(h_values, dtype=int).reshape(-1)
    k_values = np.asarray(k_values, dtype=int).reshape(-1)
    m_values = np.asarray(m_values, dtype=int).reshape(-1)
    if h_values.size == 0 or k_values.size == 0 or m_values.size == 0:
        raise ValueError("h_values, k_values, and m_values must all be non-empty")

    # Auto-refine z-grid before entering the coefficient loop.
    length = float(z[-1] - z[0])
    baud_rate = float(pulse.baud_rate)
    if auto_refine:
        z = _refine_fwm_z_grid(
            z, channels, length, baud_rate,
            min_pts_per_period=min_pts_per_period,
            max_z_points=max_z_points,
        )
    _check_fwm_discretization(
        z, channels, length, baud_rate,
        min_pts_per_period=min_pts_per_period,
        max_walkoff_fraction=max_walkoff_fraction,
        action=discretization_action,
    )

    # Validate once before entering the coefficient loop.
    _amplification_values(z, amplification_function)

    X = np.empty((h_values.size, k_values.size, m_values.size), dtype=complex)
    for h_idx, h in enumerate(h_values):
        for k_idx, k in enumerate(k_values):
            for m_idx, m in enumerate(m_values):
                X[h_idx, k_idx, m_idx] = compute_fwm_coefficient_direct(
                    pulse,
                    z,
                    h=int(h),
                    k=int(k),
                    m=int(m),
                    channels=channels,
                    amplification_function=amplification_function,
                    include_frequency_mismatch=include_frequency_mismatch,
                    include_phase_mismatch=include_phase_mismatch,
                )

    _, _, t, _ = _pulse_spectrum(pulse)
    metadata = _metadata(
        t,
        z,
        channels,
        include_frequency_mismatch=include_frequency_mismatch,
        include_phase_mismatch=include_phase_mismatch,
    )
    llw, phase_period, total_walkoff = _effective_llw_and_period(channels, length, baud_rate)
    time_window = float(t[-1] - t[0])
    n_z = z.size
    pts_per_period = float(n_z) * phase_period / length if (np.isfinite(phase_period) and length > 0) else float("inf")
    pts_per_collision = float(n_z) / llw if llw > 0 else float("inf")
    walkoff_fraction = total_walkoff / time_window if time_window > 0 else float("inf")
    metadata.update(
        {
            "h_values": h_values.copy(),
            "k_values": k_values.copy(),
            "m_values": m_values.copy(),
            "LLW": float(llw),
            "phase_period": float(phase_period),
            "pts_per_phase_period": float(pts_per_period),
            "pts_per_collision": float(pts_per_collision),
            "walkoff_fraction": float(walkoff_fraction),
            "n_z": int(n_z),
            "dz": float(np.mean(np.diff(z))),
        }
    )
    return FWMKernelResult(
        X=X,
        h_values=h_values,
        k_values=k_values,
        m_values=m_values,
        metadata=metadata,
    )


__all__ = [
    "FWMChannels",
    "FWMKernelResult",
    "compute_fwm_coefficient_direct",
    "compute_fwm_kernel_direct",
    "_DEFAULT_MIN_PTS_PER_PERIOD",
    "_DEFAULT_MAX_Z_POINTS",
    "_DEFAULT_MAX_WALKOFF_FRACTION",
]
