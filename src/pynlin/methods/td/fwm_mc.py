from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from pynlin.methods.td.fwm_kernel import (
    FWMChannels,
    _check_fwm_discretization,
    _refine_fwm_z_grid,
    compute_fwm_coefficient_direct,
)
from pynlin.pulses import Pulse


@dataclass(frozen=True)
class FWMTermMCSum:
    """Finite-window Monte Carlo estimate of one generic FWM h/k/m sum."""

    total: float
    total_stderr: float
    locked: float
    locked_stderr: float
    partial: float
    partial_stderr: float
    unlocked: float
    unlocked_stderr: float
    n_samples: int
    n_entries: int
    seed: int | None
    metadata: dict[str, object]


@dataclass(frozen=True)
class FWMDarMCSum:
    """Dar-style frequency-domain MC estimate of an all-index FWM sum."""

    total: float
    total_stderr: float
    n_samples: int
    seed: int | None
    metadata: dict[str, object]


def _safe_mean_stderr(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float).reshape(-1)
    mean = float(np.mean(values))
    if values.size < 2:
        return mean, float("nan")
    return mean, float(np.std(values, ddof=1) / np.sqrt(values.size))


def _draw_uniform_band_frequencies(n_samples: int, seed: int | None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return 2.0 * np.pi * (rng.random((3, int(n_samples))) - 0.5)


def _propagator(delta_beta: np.ndarray, length: float, alpha: float) -> np.ndarray:
    denom = 1j * delta_beta - float(alpha)
    out = np.empty_like(denom, dtype=complex)
    near_zero = np.abs(denom) < 1e-14
    out[near_zero] = complex(length)
    out[~near_zero] = (
        np.exp((1j * delta_beta[~near_zero] - float(alpha)) * float(length)) - 1.0
    ) / denom[~near_zero]
    return out


def _beta_local(
    beta0: float,
    beta1: float,
    gvd: float,
    beta3: float,
    omega_local: np.ndarray,
) -> np.ndarray:
    return (
        float(beta0)
        + float(beta1) * omega_local
        + 0.5 * float(gvd) * omega_local**2
        + (float(beta3) / 6.0) * omega_local**3
    )


def estimate_fwm_term_sum_dar_mc(
    *,
    channels: FWMChannels,
    baud_rate: float,
    length: float,
    n_samples: int,
    alpha: float = 0.0,
    seed: int | None = None,
    random_variables: np.ndarray | None = None,
) -> FWMDarMCSum:
    """Estimate the all-index generic FWM sum by frequency-domain MC.

    The estimator samples three local normalized angular frequencies
    ``Ω_a, Ω_b, Ω_c ∈ [-π,π]`` and enforces energy conservation for the
    generated output frequency

    ``Ω_d = Ω_a + Ω_b - Ω_c + (ω_a + ω_b - ω_c - ω_d) / B``.

    Samples with ``Ω_d`` outside ``[-π,π]`` do not contribute.  The remaining
    samples evaluate the exact flat-profile z propagator

    ``∫_0^L exp((i Δβ - α) z) dz``

    with ``Δβ = β_a + β_b - β_c - β_d`` using each channel's local Taylor
    coefficients in ``FWMChannels``.  This is the generic-FWM analogue of the
    existing Dar MC estimator and removes the explicit h/k/m truncation.
    """
    n_samples = int(n_samples)
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    baud_rate = float(baud_rate)
    length = float(length)
    if random_variables is None:
        R = _draw_uniform_band_frequencies(n_samples, seed)
    else:
        R = np.asarray(random_variables, dtype=float)
        if R.shape != (3, n_samples):
            raise ValueError(f"random_variables must have shape (3, {n_samples})")

    omega_a_norm = R[0]
    omega_b_norm = R[1]
    omega_c_norm = R[2]
    center_mismatch_norm = channels.delta_omega / baud_rate
    omega_d_norm = omega_a_norm + omega_b_norm - omega_c_norm + center_mismatch_norm
    mask = (omega_d_norm > -np.pi) & (omega_d_norm < np.pi)

    omega_a = omega_a_norm * baud_rate
    omega_b = omega_b_norm * baud_rate
    omega_c = omega_c_norm * baud_rate
    omega_d = omega_d_norm * baud_rate
    delta_beta = (
        _beta_local(channels.beta0_a, channels.beta1_a, channels.gvd_a, channels.beta3_a, omega_a)
        + _beta_local(channels.beta0_b, channels.beta1_b, channels.gvd_b, channels.beta3_b, omega_b)
        - _beta_local(channels.beta0_c, channels.beta1_c, channels.gvd_c, channels.beta3_c, omega_c)
        - _beta_local(channels.beta0_d, channels.beta1_d, channels.gvd_d, channels.beta3_d, omega_d)
    )
    kernel = _propagator(delta_beta, length, alpha)
    samples = np.abs(kernel) ** 2 * mask
    total, total_stderr = _safe_mean_stderr(samples)
    return FWMDarMCSum(
        total=total,
        total_stderr=total_stderr,
        n_samples=n_samples,
        seed=seed,
        metadata={
            "calculation": "dar_frequency_domain_fwm_sum_mc",
            "baud_rate": baud_rate,
            "length": length,
            "alpha": float(alpha),
            "omega_a": float(channels.omega_a),
            "omega_b": float(channels.omega_b),
            "omega_c": float(channels.omega_c),
            "omega_d": float(channels.omega_d),
            "delta_omega": float(channels.delta_omega),
            "beta3_a": float(channels.beta3_a),
            "beta3_b": float(channels.beta3_b),
            "beta3_c": float(channels.beta3_c),
            "beta3_d": float(channels.beta3_d),
            "support_fraction": float(np.mean(mask)),
        },
    )


def _same_channel_sector(target: str, term: str, h: int, k: int, m: int) -> str:
    """Classify one ``(h,k,m)`` entry as locked, partial, or unlocked."""
    labels = [target, term[0], term[1], term[2]]
    indices = [0, int(h), int(k), int(m)]

    locked = True
    any_pair = False
    for label in sorted(set(labels)):
        positions = [idx for idx, value in enumerate(labels) if value == label]
        if len(positions) < 2:
            continue
        first = indices[positions[0]]
        group_locked = True
        for pos_i, pos in enumerate(positions[1:], start=1):
            current = indices[pos]
            group_locked = group_locked and current == first
            for prev in positions[:pos_i]:
                any_pair = any_pair or current == indices[prev]
        locked = locked and group_locked

    if locked:
        return "locked"
    if any_pair:
        return "partial"
    return "unlocked"


def estimate_fwm_term_sum_mc(
    pulse: Pulse,
    z: np.ndarray,
    h_values: np.ndarray,
    k_values: np.ndarray,
    m_values: np.ndarray,
    *,
    channels: FWMChannels,
    n_samples: int,
    seed: int | None = None,
    target: str | None = None,
    term: str | None = None,
    amplification_function: Callable[[np.ndarray], np.ndarray] | None = None,
    include_frequency_mismatch: bool = True,
    include_phase_mismatch: bool = True,
    auto_refine: bool = True,
    discretization_action: str = "warn",
    min_pts_per_period: float = 3.0,
    max_z_points: int = 2000,
) -> FWMTermMCSum:
    """Estimate ``sum_hkm |X_hkm|^2`` by sampling finite-window FWM entries.

    This estimator samples the same finite ``h/k/m`` table that
    ``compute_fwm_kernel_direct`` would evaluate exhaustively.  It therefore
    includes both frequency mismatch and phase mismatch exactly as implemented
    in the direct time-domain coefficient evaluator.

    When ``target`` and ``term`` are supplied, the sampled entries are also
    split into the same locked / partial / unlocked sectors used by the
    wideband FWM diagnostic plots.
    """
    n_samples = int(n_samples)
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")

    z = np.asarray(z, dtype=float).reshape(-1)
    h_values = np.asarray(h_values, dtype=int).reshape(-1)
    k_values = np.asarray(k_values, dtype=int).reshape(-1)
    m_values = np.asarray(m_values, dtype=int).reshape(-1)
    if z.size == 0:
        raise ValueError("z grid must not be empty")
    if h_values.size == 0 or k_values.size == 0 or m_values.size == 0:
        raise ValueError("h_values, k_values, and m_values must all be non-empty")
    if (target is None) != (term is None):
        raise ValueError("target and term must be provided together")
    if term is not None and len(term) != 3:
        raise ValueError("term must be a three-character string like 'abb'")

    length = float(z[-1] - z[0])
    baud_rate = float(pulse.baud_rate)
    if auto_refine:
        z = _refine_fwm_z_grid(
            z,
            channels,
            length,
            baud_rate,
            min_pts_per_period=min_pts_per_period,
            max_z_points=max_z_points,
        )
    _check_fwm_discretization(
        z,
        channels,
        length,
        baud_rate,
        min_pts_per_period=min_pts_per_period,
        action=discretization_action,
    )

    rng = np.random.default_rng(seed)
    n_entries = int(h_values.size * k_values.size * m_values.size)
    h_idx = rng.integers(0, h_values.size, size=n_samples)
    k_idx = rng.integers(0, k_values.size, size=n_samples)
    m_idx = rng.integers(0, m_values.size, size=n_samples)

    total_samples = np.empty(n_samples, dtype=float)
    locked_samples = np.zeros(n_samples, dtype=float)
    partial_samples = np.zeros(n_samples, dtype=float)
    unlocked_samples = np.zeros(n_samples, dtype=float)

    for idx in range(n_samples):
        h = int(h_values[h_idx[idx]])
        k = int(k_values[k_idx[idx]])
        m = int(m_values[m_idx[idx]])
        coeff = compute_fwm_coefficient_direct(
            pulse,
            z,
            h=h,
            k=k,
            m=m,
            channels=channels,
            amplification_function=amplification_function,
            include_frequency_mismatch=include_frequency_mismatch,
            include_phase_mismatch=include_phase_mismatch,
        )
        scaled_abs2 = float(n_entries * abs(coeff) ** 2)
        total_samples[idx] = scaled_abs2
        if target is not None and term is not None:
            sector = _same_channel_sector(target, term, h, k, m)
            if sector == "locked":
                locked_samples[idx] = scaled_abs2
            elif sector == "partial":
                partial_samples[idx] = scaled_abs2
            else:
                unlocked_samples[idx] = scaled_abs2

    total, total_stderr = _safe_mean_stderr(total_samples)
    if target is None:
        locked = locked_stderr = partial = partial_stderr = unlocked = unlocked_stderr = float("nan")
    else:
        locked, locked_stderr = _safe_mean_stderr(locked_samples)
        partial, partial_stderr = _safe_mean_stderr(partial_samples)
        unlocked, unlocked_stderr = _safe_mean_stderr(unlocked_samples)

    return FWMTermMCSum(
        total=total,
        total_stderr=total_stderr,
        locked=locked,
        locked_stderr=locked_stderr,
        partial=partial,
        partial_stderr=partial_stderr,
        unlocked=unlocked,
        unlocked_stderr=unlocked_stderr,
        n_samples=n_samples,
        n_entries=n_entries,
        seed=seed,
        metadata={
            "calculation": "finite_window_fwm_hkm_sum_mc",
            "n_z": int(z.size),
            "h_values_size": int(h_values.size),
            "k_values_size": int(k_values.size),
            "m_values_size": int(m_values.size),
            "target": target,
            "term": term,
            "include_frequency_mismatch": bool(include_frequency_mismatch),
            "include_phase_mismatch": bool(include_phase_mismatch),
            "auto_refine": bool(auto_refine),
            "min_pts_per_period": float(min_pts_per_period),
            "max_z_points": int(max_z_points),
        },
    )


__all__ = [
    "FWMDarMCSum",
    "FWMTermMCSum",
    "estimate_fwm_term_sum_dar_mc",
    "estimate_fwm_term_sum_mc",
]
