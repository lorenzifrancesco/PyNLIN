from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class XhkmMCSums:
    """Prefactor-free Monte Carlo estimates of Dar-style Xhkm sums."""

    n1: float
    n2: float
    n1_stderr: float
    n2_stderr: float
    n_samples: int
    seed: int | None
    metadata: dict[str, object]

    @property
    def n2_over_n1(self) -> float:
        if self.n1 <= 0.0:
            return float("nan")
        return self.n2 / self.n1


def assert_flat_signal_power_profile(system: object) -> None:
    """Reject system configs that request non-flat signal power profiles."""
    raw = getattr(system, "raw_config", None)
    if not isinstance(raw, Mapping):
        return

    profiles = raw.get("profiles")
    if isinstance(profiles, Mapping):
        mode = str(profiles.get("mode", "flat")).strip().lower()
        if mode != "flat":
            raise ValueError(
                "Prefactor-free Dar MC validation supports only flat signal power profiles; "
                f"got profiles.mode={mode!r}."
            )

    pcfm = raw.get("pcfm")
    run = pcfm.get("run") if isinstance(pcfm, Mapping) else None
    if isinstance(run, Mapping) and "power_profiles_mode" in run:
        mode = str(run["power_profiles_mode"]).strip().lower()
        if mode != "flat":
            raise ValueError(
                "Prefactor-free Dar MC validation supports only flat signal power profiles; "
                f"got pcfm.run.power_profiles_mode={mode!r}."
            )


def _draw_uniform_phases(n_samples: int, seed: int | None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return 2.0 * np.pi * (rng.random((4, int(n_samples))) - 0.5)


def _safe_mean_stderr(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float).reshape(-1)
    mean = float(np.mean(values))
    if values.size < 2:
        return mean, float("nan")
    return mean, float(np.std(values, ddof=1) / np.sqrt(values.size))


def _span_factor(phase: np.ndarray, nspan: int, sign: int) -> np.ndarray:
    signed = 1j * int(sign) * np.asarray(phase, dtype=float)
    numerator = 1.0 - np.exp(nspan * signed)
    denominator = 1.0 - np.exp(signed)
    out = np.empty_like(numerator, dtype=complex)
    near_zero = np.abs(denominator) < 1e-14
    out[near_zero] = complex(nspan)
    out[~near_zero] = numerator[~near_zero] / denominator[~near_zero]
    return out


def estimate_xhkm_sums_mc(
    *,
    beta2: float,
    alpha: float,
    length: float,
    channel_spacing_over_baud: float,
    n_samples: int,
    nspan: int = 1,
    phase_delay: float = 0.0,
    seed: int | None = None,
    random_variables: np.ndarray | None = None,
    system: object | None = None,
) -> XhkmMCSums:
    """Estimate prefactor-free ``N1`` and ``N2`` using Dar MC integrands.

    This is the stripped-down inter-channel Dar Monte Carlo path: no launch
    powers, no nonlinear coefficient, no modulation cumulants, and no final
    NLIN variance are included.  The returned quantities match the historical
    Dar ``chi1`` and ``chi2`` components after dividing out the common physical
    prefactor ``4*gamma**2*P0**3``.
    """
    if system is not None:
        assert_flat_signal_power_profile(system)
    n_samples = int(n_samples)
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    if random_variables is None:
        R = _draw_uniform_phases(n_samples, seed)
    else:
        R = np.asarray(random_variables, dtype=float)
        if R.shape != (4, n_samples):
            raise ValueError(f"random_variables must have shape (4, {n_samples})")

    beta2 = float(beta2)
    alpha = float(alpha)
    length = float(length)
    phase_delay = float(phase_delay)
    q = float(channel_spacing_over_baud)
    nspan = int(nspan)

    w0 = R[0, :] - R[1, :] + R[2, :]
    arg1 = (R[1, :] - R[2, :]) * (R[1, :] + 2.0 * np.pi * q - R[0, :])
    mask1 = (w0 < np.pi) & (w0 > -np.pi)
    denom1 = 1j * beta2 * arg1 - alpha
    ss1 = (
        np.exp(1j * arg1 * phase_delay)
        * (np.exp(1j * beta2 * arg1 * length - alpha * length) - 1.0)
        / denom1
        * mask1
    )
    span1 = _span_factor(arg1 * beta2 * length, nspan, sign=1)
    n1_samples = np.abs(ss1 * span1) ** 2

    w3p = -R[1, :] + R[3, :] + R[2, :] + 2.0 * np.pi * q
    arg2 = (R[1, :] - R[2, :]) * (R[3, :] - R[0, :] + 2.0 * np.pi * q)
    mask2 = (w3p > -np.pi + 2.0 * np.pi * q) & (w3p < np.pi + 2.0 * np.pi * q)
    denom2 = -1j * beta2 * arg2 - alpha
    ss2 = (
        np.exp(-1j * arg2 * phase_delay)
        * (np.exp(-1j * beta2 * arg2 * length - alpha * length) - 1.0)
        / denom2
        * mask2
    )
    span2 = _span_factor(arg2 * beta2 * length, nspan, sign=-1)
    n2_samples = np.real(span1 * ss1 * span2 * ss2)

    n1, n1_stderr = _safe_mean_stderr(n1_samples)
    n2, n2_stderr = _safe_mean_stderr(n2_samples)
    return XhkmMCSums(
        n1=n1,
        n2=n2,
        n1_stderr=n1_stderr,
        n2_stderr=n2_stderr,
        n_samples=n_samples,
        seed=seed,
        metadata={
            "calculation": "prefactor_free_dar_mc_xhkm_sums",
            "beta2": beta2,
            "alpha": alpha,
            "length": length,
            "channel_spacing_over_baud": q,
            "nspan": nspan,
            "phase_delay": phase_delay,
        },
    )


__all__ = ["XhkmMCSums", "assert_flat_signal_power_profile", "estimate_xhkm_sums_mc"]
