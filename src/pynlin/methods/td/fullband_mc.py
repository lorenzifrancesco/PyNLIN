from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pynlin.methods.td.fwm_kernel import FWMChannels
from pynlin.methods.td.fwm_mc import estimate_fwm_term_sum_dar_mc
from pynlin.methods.td.xhkm_mc import estimate_xhkm_sums_mc
from pynlin.system import System
from pynlin.utils import lambda2nu


@dataclass(frozen=True)
class FWMPruningStats:
    naive_tuples: int
    support_survivors: int
    evaluated_tuples: int


@dataclass(frozen=True)
class FullbandMCDiagnostic:
    target_indices: np.ndarray
    target_frequencies: np.ndarray
    xpm: np.ndarray
    fwm: np.ndarray
    total: np.ndarray
    fwm_tuple_count: np.ndarray
    fwm_support_count: np.ndarray
    pruning: FWMPruningStats
    metadata: dict[str, object]


def estimate_zdw_frequency(system: System) -> float | None:
    beta2_profile = getattr(system.fiber, "_beta2_profile", None)
    if beta2_profile is None:
        return None
    wavelengths, beta2 = beta2_profile
    wavelengths = np.asarray(wavelengths, dtype=float).reshape(-1)
    beta2 = np.asarray(beta2, dtype=float).reshape(-1)
    if wavelengths.size < 2 or wavelengths.size != beta2.size:
        return None

    order = np.argsort(wavelengths)
    wavelengths = wavelengths[order]
    beta2 = beta2[order]
    sign_changes = np.where(np.signbit(beta2[:-1]) != np.signbit(beta2[1:]))[0]
    if sign_changes.size == 0:
        return None

    idx = int(sign_changes[0])
    wl0 = float(wavelengths[idx])
    wl1 = float(wavelengths[idx + 1])
    b0 = float(beta2[idx])
    b1 = float(beta2[idx + 1])
    if b1 == b0:
        zdw_wavelength = 0.5 * (wl0 + wl1)
    else:
        zdw_wavelength = wl0 - b0 * (wl1 - wl0) / (b1 - b0)
    return float(lambda2nu(zdw_wavelength))


def decimated_frequency_grid(system: System, factor: int = 1) -> tuple[np.ndarray, np.ndarray]:
    freqs = np.asarray(system.wdm.frequency_grid(), dtype=float).reshape(-1)
    factor = int(factor)
    if factor < 1:
        raise ValueError("decimation factor must be >= 1")
    indices = np.arange(0, freqs.size, factor, dtype=int)
    if indices.size == 0:
        raise ValueError("decimation removed all channels")
    return indices, freqs[indices]


def _beta0_offsets_from_fiber(system: System, freqs: np.ndarray, target: int, beta1: np.ndarray) -> np.ndarray:
    beta_profile = getattr(system.fiber, "_beta_profile", None)
    freq_profile = getattr(system.fiber, "_freq_profile", None)
    if beta_profile is not None and freq_profile is not None:
        beta_abs = np.interp(
            np.asarray(freqs, dtype=float),
            np.asarray(freq_profile, dtype=float),
            np.asarray(beta_profile[1], dtype=float),
        )
        return beta_abs - float(beta_abs[int(target)])

    omega = 2.0 * np.pi * np.asarray(freqs, dtype=float)
    order = np.argsort(omega)
    beta_sorted = np.zeros_like(omega, dtype=float)
    omega_sorted = omega[order]
    beta1_sorted = np.asarray(beta1, dtype=float)[order]
    increments = 0.5 * (beta1_sorted[1:] + beta1_sorted[:-1]) * np.diff(omega_sorted)
    beta_sorted[1:] = np.cumsum(increments)
    beta_abs = np.empty_like(beta_sorted)
    beta_abs[order] = beta_sorted
    return beta_abs - float(beta_abs[int(target)])


def _channel_params(
    freqs: np.ndarray,
    beta0_offsets: np.ndarray,
    beta1: np.ndarray,
    beta2: np.ndarray,
    target: int,
    a: int,
    b: int,
    c: int,
) -> FWMChannels:
    # Local angular-frequency origins are channel centers.  beta0 is set from a
    # quadratic expansion around the global reference only for phase mismatch;
    # beta1/beta2 are the local Taylor coefficients at each channel.
    f0 = float(freqs[target])
    return FWMChannels(
        omega_a=2.0 * np.pi * float(freqs[a] - f0),
        omega_b=2.0 * np.pi * float(freqs[b] - f0),
        omega_c=2.0 * np.pi * float(freqs[c] - f0),
        omega_d=0.0,
        beta0_a=float(beta0_offsets[a]),
        beta0_b=float(beta0_offsets[b]),
        beta0_c=float(beta0_offsets[c]),
        beta0_d=0.0,
        beta1_a=float(beta1[a]),
        beta1_b=float(beta1[b]),
        beta1_c=float(beta1[c]),
        beta1_d=float(beta1[target]),
        gvd_a=float(beta2[a]),
        gvd_b=float(beta2[b]),
        gvd_c=float(beta2[c]),
        gvd_d=float(beta2[target]),
    )


def fwm_support_survives(freqs: np.ndarray, baud_rate: float, target: int, a: int, b: int, c: int) -> bool:
    mismatch = float(freqs[a] + freqs[b] - freqs[c] - freqs[target])
    return abs(mismatch) <= 2.0 * float(baud_rate)


def iter_support_pruned_fwm_tuples(freqs: np.ndarray, baud_rate: float, target: int):
    n = int(freqs.size)
    for a in range(n):
        for b in range(n):
            for c in range(n):
                if fwm_support_survives(freqs, baud_rate, target, a, b, c):
                    yield a, b, c


def collect_support_pruned_fwm_tuples(
    freqs: np.ndarray,
    baud_rate: float,
    target: int,
    *,
    max_tuples: int | None = None,
    seed: int | None = None,
    selection_mode: str = "reservoir",
    beta1: np.ndarray | None = None,
    beta2: np.ndarray | None = None,
    beta0_offsets: np.ndarray | None = None,
) -> tuple[int, list[tuple[int, int, int]]]:
    """Count and optionally sample FWM tuples passing exact output support.

    The support condition is

    ``|f_a + f_b - f_c - f_target| <= 2B``.

    This implementation sorts the channel frequencies once and uses binary
    search to find all valid ``c`` indices for each ``(a,b)``. If
    ``max_tuples`` is provided, it returns a uniform reservoir sample of the
    support-surviving tuples instead of the first tuples in lexical order.
    """
    freqs = np.asarray(freqs, dtype=float).reshape(-1)
    n = int(freqs.size)
    max_tuples = None if max_tuples is None else int(max_tuples)
    if max_tuples is not None and max_tuples < 1:
        return 0, []

    order = np.argsort(freqs)
    sorted_freqs = freqs[order]
    ft = float(freqs[int(target)])
    width = 2.0 * float(baud_rate)
    if selection_mode not in {"reservoir", "phase_proxy"}:
        raise ValueError("selection_mode must be 'reservoir' or 'phase_proxy'")
    if selection_mode == "phase_proxy" and beta0_offsets is None:
        raise ValueError("phase_proxy selection requires beta0_offsets")
    rng = np.random.default_rng(seed)
    selected: list[tuple[int, int, int]] = []
    count = 0

    if selection_mode == "phase_proxy":
        beta0 = np.asarray(beta0_offsets, dtype=float).reshape(-1)
        scored: list[tuple[float, tuple[int, int, int]]] = []

    for a in range(n):
        fa = float(freqs[a])
        for b in range(n):
            center = fa + float(freqs[b]) - ft
            lo = int(np.searchsorted(sorted_freqs, center - width, side="left"))
            hi = int(np.searchsorted(sorted_freqs, center + width, side="right"))
            if hi <= lo:
                continue
            for c in order[lo:hi]:
                tup = (a, b, int(c))
                if selection_mode == "phase_proxy":
                    delta_beta0 = abs(float(beta0[a] + beta0[b] - beta0[int(c)] - beta0[int(target)]))
                    scored.append((delta_beta0, tup))
                elif max_tuples is None:
                    selected.append(tup)
                elif len(selected) < max_tuples:
                    selected.append(tup)
                else:
                    j = int(rng.integers(0, count + 1))
                    if j < max_tuples:
                        selected[j] = tup
                count += 1

    if selection_mode == "phase_proxy":
        if max_tuples is None:
            selected = [tup for _, tup in sorted(scored, key=lambda item: item[0])]
        else:
            scored.sort(key=lambda item: item[0])
            selected = [tup for _, tup in scored[:max_tuples]]
    return count, selected


def _propagator_abs2(delta_beta: np.ndarray, length: float, alpha: float) -> np.ndarray:
    denom = 1j * delta_beta - float(alpha)
    out = np.empty_like(denom, dtype=complex)
    near_zero = np.abs(denom) < 1e-14
    out[near_zero] = complex(length)
    out[~near_zero] = (
        np.exp((1j * delta_beta[~near_zero] - float(alpha)) * float(length)) - 1.0
    ) / denom[~near_zero]
    return np.abs(out) ** 2


def estimate_target_fwm_joint_reservoir_mc(
    *,
    freqs: np.ndarray,
    beta0_offsets: np.ndarray,
    beta1: np.ndarray,
    beta2: np.ndarray,
    baud_rate: float,
    length: float,
    target: int,
    n_samples: int,
    seed: int,
    alpha: float = 0.0,
) -> tuple[float, float, int, int]:
    """Unbiased joint tuple/frequency MC estimate for one target FWM sum."""
    n_samples = int(n_samples)
    survivor_count, survivors = collect_support_pruned_fwm_tuples(
        freqs,
        baud_rate,
        int(target),
        max_tuples=n_samples,
        seed=seed,
        selection_mode="reservoir",
    )
    if not survivors:
        return 0.0, 0.0, survivor_count, 0

    tuples = np.asarray(survivors, dtype=int)
    a = tuples[:, 0]
    b = tuples[:, 1]
    c = tuples[:, 2]
    t = int(target)
    rng = np.random.default_rng(seed + 17_171)
    r = 2.0 * np.pi * (rng.random((3, tuples.shape[0])) - 0.5)
    omega_a_norm = r[0]
    omega_b_norm = r[1]
    omega_c_norm = r[2]

    delta_omega = 2.0 * np.pi * (freqs[a] + freqs[b] - freqs[c] - freqs[t])
    omega_d_norm = omega_a_norm + omega_b_norm - omega_c_norm + delta_omega / float(baud_rate)
    mask = (omega_d_norm > -np.pi) & (omega_d_norm < np.pi)

    omega_a = omega_a_norm * float(baud_rate)
    omega_b = omega_b_norm * float(baud_rate)
    omega_c = omega_c_norm * float(baud_rate)
    omega_d = omega_d_norm * float(baud_rate)
    delta_beta = (
        beta0_offsets[a]
        + beta1[a] * omega_a
        + 0.5 * beta2[a] * omega_a**2
        + beta0_offsets[b]
        + beta1[b] * omega_b
        + 0.5 * beta2[b] * omega_b**2
        - beta0_offsets[c]
        - beta1[c] * omega_c
        - 0.5 * beta2[c] * omega_c**2
        - beta1[t] * omega_d
        - 0.5 * beta2[t] * omega_d**2
    )
    values = _propagator_abs2(delta_beta, length, alpha) * mask
    mean = float(np.mean(values))
    stderr = float(np.std(values, ddof=1) / np.sqrt(values.size)) if values.size > 1 else 0.0
    scale = float(survivor_count)
    return scale * mean, scale * stderr, survivor_count, int(tuples.shape[0])


def _support_tuple_arrays(freqs: np.ndarray, baud_rate: float, target: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    freqs = np.asarray(freqs, dtype=float).reshape(-1)
    n = int(freqs.size)
    order = np.argsort(freqs)
    sorted_freqs = freqs[order]
    ft = float(freqs[int(target)])
    width = 2.0 * float(baud_rate)
    a_parts = []
    b_parts = []
    c_parts = []
    for a in range(n):
        fa = float(freqs[a])
        for b in range(n):
            center = fa + float(freqs[b]) - ft
            lo = int(np.searchsorted(sorted_freqs, center - width, side="left"))
            hi = int(np.searchsorted(sorted_freqs, center + width, side="right"))
            if hi <= lo:
                continue
            c_vals = order[lo:hi].astype(int, copy=False)
            a_parts.append(np.full(c_vals.size, a, dtype=int))
            b_parts.append(np.full(c_vals.size, b, dtype=int))
            c_parts.append(c_vals.copy())
    if not a_parts:
        empty = np.array([], dtype=int)
        return empty, empty, empty
    return np.concatenate(a_parts), np.concatenate(b_parts), np.concatenate(c_parts)


def estimate_target_fwm_exhaustive_support_mc(
    *,
    freqs: np.ndarray,
    beta0_offsets: np.ndarray,
    beta1: np.ndarray,
    beta2: np.ndarray,
    baud_rate: float,
    length: float,
    target: int,
    n_frequency_samples: int,
    seed: int,
    alpha: float = 0.0,
    chunk_size: int = 50_000,
) -> tuple[float, float, int, int]:
    """Sum all support-pruned tuples; use MC only for in-channel frequencies."""
    a_all, b_all, c_all = _support_tuple_arrays(freqs, baud_rate, int(target))
    support_count = int(a_all.size)
    if support_count == 0:
        return 0.0, 0.0, 0, 0
    q = int(n_frequency_samples)
    if q <= 0:
        raise ValueError("n_frequency_samples must be positive")
    rng = np.random.default_rng(seed + 17_171)
    r = 2.0 * np.pi * (rng.random((3, q)) - 0.5)
    per_frequency_sum = np.zeros(q, dtype=float)
    t = int(target)
    chunk_size = max(int(chunk_size), 1)
    for start in range(0, support_count, chunk_size):
        stop = min(start + chunk_size, support_count)
        a = a_all[start:stop]
        b = b_all[start:stop]
        c = c_all[start:stop]
        delta_omega = 2.0 * np.pi * (freqs[a] + freqs[b] - freqs[c] - freqs[t])
        for qi in range(q):
            omega_a_norm = r[0, qi]
            omega_b_norm = r[1, qi]
            omega_c_norm = r[2, qi]
            omega_d_norm = omega_a_norm + omega_b_norm - omega_c_norm + delta_omega / float(baud_rate)
            mask = (omega_d_norm > -np.pi) & (omega_d_norm < np.pi)
            omega_a = omega_a_norm * float(baud_rate)
            omega_b = omega_b_norm * float(baud_rate)
            omega_c = omega_c_norm * float(baud_rate)
            omega_d = omega_d_norm * float(baud_rate)
            delta_beta = (
                beta0_offsets[a]
                + beta1[a] * omega_a
                + 0.5 * beta2[a] * omega_a**2
                + beta0_offsets[b]
                + beta1[b] * omega_b
                + 0.5 * beta2[b] * omega_b**2
                - beta0_offsets[c]
                - beta1[c] * omega_c
                - 0.5 * beta2[c] * omega_c**2
                - beta1[t] * omega_d
                - 0.5 * beta2[t] * omega_d**2
            )
            per_frequency_sum[qi] += float(np.sum(_propagator_abs2(delta_beta, length, alpha) * mask))
    estimate = float(np.mean(per_frequency_sum))
    stderr = float(np.std(per_frequency_sum, ddof=1) / np.sqrt(q)) if q > 1 else 0.0
    return estimate, stderr, support_count, support_count


def compute_fullband_prefactor_free_mc(
    system: System,
    *,
    decimation: int = 1,
    target_indices: np.ndarray | None = None,
    include_xpm: bool = True,
    include_fwm: bool = True,
    xpm_samples: int = 50_000,
    fwm_samples: int = 20_000,
    seed: int = 1234,
    max_fwm_tuples_per_target: int | None = None,
    fwm_tuple_selection: str = "reservoir",
) -> FullbandMCDiagnostic:
    kept_indices, freqs = decimated_frequency_grid(system, decimation)
    baud_rate = float(system.pulse.baud_rate)
    length = float(system.fiber_length)
    beta1_grid, beta2_grid = system.beta_grids(freqs=freqs)
    beta1 = np.asarray(beta1_grid[0], dtype=float)
    beta2 = np.asarray(beta2_grid[0], dtype=float)
    beta2_ref = float(np.median(beta2))
    beta2_norm = beta2_ref / (1.0 / baud_rate) ** 2

    if target_indices is None:
        targets = np.arange(freqs.size, dtype=int)
    else:
        targets = np.asarray(target_indices, dtype=int).reshape(-1)
    if np.any((targets < 0) | (targets >= freqs.size)):
        raise ValueError("target_indices refer to decimated-grid positions")

    xpm = np.zeros(targets.size, dtype=float)
    fwm = np.zeros(targets.size, dtype=float)
    fwm_counts = np.zeros(targets.size, dtype=int)
    fwm_support_counts = np.zeros(targets.size, dtype=int)

    naive = int(targets.size * freqs.size**3)
    survivors_total = 0
    evaluated_total = 0

    for out_i, target in enumerate(targets):
        beta0_offsets = _beta0_offsets_from_fiber(system, freqs, int(target), beta1)
        if include_xpm:
            for b in range(freqs.size):
                if b == target:
                    continue
                q = abs(float(freqs[b] - freqs[target])) / baud_rate
                est = estimate_xhkm_sums_mc(
                    beta2=beta2_norm,
                    alpha=0.0,
                    length=length,
                    channel_spacing_over_baud=q,
                    n_samples=xpm_samples,
                    seed=seed + 10_000 * int(target) + int(b),
                    system=system,
                )
                xpm[out_i] += est.n1

        if include_fwm and fwm_tuple_selection == "exhaustive_support_mc":
            estimate, _, survivor_count, sampled_count = estimate_target_fwm_exhaustive_support_mc(
                freqs=freqs,
                beta0_offsets=beta0_offsets,
                beta1=beta1,
                beta2=beta2,
                baud_rate=baud_rate,
                length=length,
                target=int(target),
                n_frequency_samples=int(fwm_samples),
                seed=seed + 999_983 * int(target),
                alpha=0.0,
            )
            survivors_total += survivor_count
            fwm_support_counts[out_i] = survivor_count
            fwm_counts[out_i] = sampled_count
            evaluated_total += sampled_count
            fwm[out_i] = estimate
        elif include_fwm and fwm_tuple_selection == "joint_reservoir":
            estimate, _, survivor_count, sampled_count = estimate_target_fwm_joint_reservoir_mc(
                freqs=freqs,
                beta0_offsets=beta0_offsets,
                beta1=beta1,
                beta2=beta2,
                baud_rate=baud_rate,
                length=length,
                target=int(target),
                n_samples=int(max_fwm_tuples_per_target or fwm_samples),
                seed=seed + 999_983 * int(target),
                alpha=0.0,
            )
            survivors_total += survivor_count
            fwm_support_counts[out_i] = survivor_count
            fwm_counts[out_i] = sampled_count
            evaluated_total += sampled_count
            fwm[out_i] = estimate
        elif include_fwm:
            survivor_count, survivors = collect_support_pruned_fwm_tuples(
                freqs,
                baud_rate,
                int(target),
                max_tuples=max_fwm_tuples_per_target,
                seed=seed + 999_983 * int(target),
                selection_mode=fwm_tuple_selection,
                beta0_offsets=beta0_offsets,
            )
            survivors_total += survivor_count
            fwm_support_counts[out_i] = survivor_count
            fwm_counts[out_i] = len(survivors)
            evaluated_total += len(survivors)
            sampled_sum = 0.0
            for tuple_i, (a, b, c) in enumerate(survivors):
                channels = _channel_params(freqs, beta0_offsets, beta1, beta2, int(target), int(a), int(b), int(c))
                est = estimate_fwm_term_sum_dar_mc(
                    channels=channels,
                    baud_rate=baud_rate,
                    length=length,
                    n_samples=fwm_samples,
                    alpha=0.0,
                    seed=seed + 1_000_000 * int(target) + 10_000 * int(a) + 100 * int(b) + int(c) + tuple_i,
                )
                sampled_sum += est.total
            if survivors and fwm_tuple_selection == "reservoir":
                # Uniform reservoir sampling estimates the full support-pruned
                # FWM tuple sum by scaling the sampled mean by the survivor count.
                fwm[out_i] = sampled_sum * float(survivor_count) / float(len(survivors))
            elif survivors:
                # Phase-proxy mode intentionally returns a lower-bound / dominant
                # contribution from the most phase-matched tuples. Do not scale it
                # to all support survivors because the sample is not uniform.
                fwm[out_i] = sampled_sum

    total = xpm + fwm
    return FullbandMCDiagnostic(
        target_indices=kept_indices[targets],
        target_frequencies=freqs[targets],
        xpm=xpm,
        fwm=fwm,
        total=total,
        fwm_tuple_count=fwm_counts,
        fwm_support_count=fwm_support_counts,
        pruning=FWMPruningStats(
            naive_tuples=naive,
            support_survivors=survivors_total,
            evaluated_tuples=evaluated_total,
        ),
        metadata={
            "calculation": "prefactor_free_fullband_xpm_fwm_mc",
            "decimation": int(decimation),
            "n_channels_decimated": int(freqs.size),
            "baud_rate": baud_rate,
            "length": length,
            "zdw_frequency": estimate_zdw_frequency(system),
            "xpm_samples": int(xpm_samples),
            "fwm_samples": int(fwm_samples),
            "seed": int(seed),
            "include_xpm": bool(include_xpm),
            "include_fwm": bool(include_fwm),
            "max_fwm_tuples_per_target": max_fwm_tuples_per_target,
            "fwm_tuple_selection": fwm_tuple_selection,
        },
    )


__all__ = [
    "FWMPruningStats",
    "FullbandMCDiagnostic",
    "compute_fullband_prefactor_free_mc",
    "decimated_frequency_grid",
    "estimate_zdw_frequency",
    "estimate_target_fwm_joint_reservoir_mc",
    "estimate_target_fwm_exhaustive_support_mc",
    "fwm_support_survives",
    "collect_support_pruned_fwm_tuples",
    "iter_support_pruned_fwm_tuples",
]
