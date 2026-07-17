from __future__ import annotations

import argparse
import csv
import heapq
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["text.usetex"] = False
matplotlib.rcParams["mathtext.fontset"] = "cm"
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
import numpy as np

from pynlin.methods.td.fwm_kernel import FWMChannels
from pynlin.methods.td.fwm_mc import estimate_fwm_term_sum_dar_mc
from pynlin.system import System
from pynlin.utils import nu2lambda


OUT_MEDIA = Path("media/fwm/mc-gradient-scaling")
OUT_RESULTS = Path("results/fwm/mc-gradient-scaling")


@dataclass(frozen=True)
class SurrogateGeometry:
    delta_beta_center: float
    gradient: np.ndarray
    gradient_norm: float
    x_grad: float
    x_phase: float
    x_combined: float
    x_quadrature: float
    epsilon_support: float
    separation_ac: float
    separation_bc: float
    separation_asymmetry: float
    curvature: float
    phase_matched_domain: bool


def constant_beta2_delta_beta(
    omega_a: np.ndarray | float,
    omega_b: np.ndarray | float,
    omega_c: np.ndarray | float,
    beta2_const: float,
) -> np.ndarray:
    """Constant-beta2 FWM mismatch after eliminating omega_d by energy conservation."""
    return -float(beta2_const) * (np.asarray(omega_a) - np.asarray(omega_c)) * (
        np.asarray(omega_b) - np.asarray(omega_c)
    )


def phase_matched_domain_intersects(freqs: np.ndarray, baud_rate: float) -> bool:
    """Return whether a constant-beta2 phase-matched branch crosses four passbands.

    ``freqs`` follows ``(d, a, b, c)``. Each ideal Nyquist passband has full
    width ``baud_rate``. The two branches are ``f_a=f_c, f_b=f_d`` and
    ``f_b=f_c, f_a=f_d``.
    """
    fd, fa, fb, fc = np.asarray(freqs, dtype=float)
    width = float(baud_rate)
    branch_ac = abs(fa - fc) <= width and abs(fb - fd) <= width
    branch_bc = abs(fb - fc) <= width and abs(fa - fd) <= width
    return bool(branch_ac or branch_bc)


def surrogate_geometry(
    freqs: np.ndarray,
    *,
    baud_rate: float,
    length: float,
    beta2_const: float,
) -> SurrogateGeometry:
    """Evaluate the proposed constant-beta2 scale at channel-center inputs.

    Frequencies are ordered ``(d, a, b, c)`` in Hz. The gradient is with
    respect to the three independent local angular frequencies and has units
    s/m. ``x_grad`` and ``x_phase`` are dimensionless.
    """
    fd, fa, fb, fc = np.asarray(freqs, dtype=float)
    omega_a, omega_b, omega_c = 2.0 * np.pi * np.array([fa, fb, fc], dtype=float)
    beta2_const = float(beta2_const)
    separation_ac = omega_a - omega_c
    separation_bc = omega_b - omega_c
    gradient = beta2_const * np.array(
        [-separation_bc, -separation_ac, separation_ac + separation_bc],
        dtype=float,
    )
    gradient_norm = float(np.linalg.norm(gradient))
    delta_beta_center = float(
        constant_beta2_delta_beta(omega_a, omega_b, omega_c, beta2_const)
    )
    delta_omega_center = 2.0 * np.pi * (fa + fb - fc - fd)
    scale = max(abs(separation_ac), abs(separation_bc))
    separation_asymmetry = (
        abs(abs(separation_ac) - abs(separation_bc)) / scale if scale > 0.0 else 0.0
    )
    curvature = (
        float(baud_rate) * abs(beta2_const) / gradient_norm
        if gradient_norm > 0.0
        else float("inf")
    )
    return SurrogateGeometry(
        delta_beta_center=delta_beta_center,
        gradient=gradient,
        gradient_norm=gradient_norm,
        x_grad=float(length) * float(baud_rate) * gradient_norm,
        x_phase=float(length) * abs(delta_beta_center),
        x_combined=float(length) * (abs(delta_beta_center) + float(baud_rate) * gradient_norm),
        x_quadrature=float(length)
        * float(np.hypot(abs(delta_beta_center), float(baud_rate) * gradient_norm)),
        epsilon_support=delta_omega_center / float(baud_rate),
        separation_ac=separation_ac / float(baud_rate),
        separation_bc=separation_bc / float(baud_rate),
        separation_asymmetry=separation_asymmetry,
        curvature=curvature,
        phase_matched_domain=phase_matched_domain_intersects(freqs, baud_rate),
    )


def _interp_linear_extrapolation(x: np.ndarray, xp: np.ndarray, fp: np.ndarray) -> np.ndarray:
    order = np.argsort(xp)
    xp = np.asarray(xp, dtype=float)[order]
    fp = np.asarray(fp, dtype=float)[order]
    out = np.interp(np.asarray(x, dtype=float), xp, fp)
    if xp.size < 2:
        return out
    left = x < xp[0]
    right = x > xp[-1]
    out[left] = fp[0] + (x[left] - xp[0]) * (fp[1] - fp[0]) / (xp[1] - xp[0])
    out[right] = fp[-1] + (x[right] - xp[-1]) * (fp[-1] - fp[-2]) / (xp[-1] - xp[-2])
    return out


def beta0_grid(system: System, freqs: np.ndarray, beta1: np.ndarray) -> np.ndarray:
    """Build beta at channel centers without depending on fullband_mc helpers."""
    beta_profile = getattr(system.fiber, "_beta_profile", None)
    freq_profile = getattr(system.fiber, "_freq_profile", None)
    if beta_profile is not None and freq_profile is not None:
        spline_factory = getattr(system.fiber, "beta_spline_omega", None)
        if spline_factory is not None:
            try:
                return np.asarray(spline_factory(s=0.0, k=3)(2.0 * np.pi * freqs), dtype=float)
            except (TypeError, ValueError):
                pass
        return _interp_linear_extrapolation(
            np.asarray(freqs, dtype=float),
            np.asarray(freq_profile, dtype=float),
            np.asarray(beta_profile[1], dtype=float),
        )

    omega = 2.0 * np.pi * np.asarray(freqs, dtype=float)
    order = np.argsort(omega)
    omega_sorted = omega[order]
    beta1_sorted = np.asarray(beta1, dtype=float)[order]
    beta_sorted = np.zeros_like(omega_sorted)
    beta_sorted[1:] = np.cumsum(
        0.5 * (beta1_sorted[1:] + beta1_sorted[:-1]) * np.diff(omega_sorted)
    )
    beta = np.empty_like(beta_sorted)
    beta[order] = beta_sorted
    return beta


def enumerate_fwm_tuples(
    freqs: np.ndarray,
    *,
    baud_rate: float,
    target: int,
    cap: int | None,
    seed: int,
    selection_mode: str = "reservoir",
    beta0: np.ndarray | None = None,
    beta1: np.ndarray | None = None,
    beta2: np.ndarray | None = None,
) -> tuple[int, list[tuple[int, int, int]]]:
    """Count strict FWM tuples and select uniformly or by spectral proximity."""
    freqs = np.asarray(freqs, dtype=float)
    target = int(target)
    if selection_mode not in {"reservoir", "near_resonant", "profile_resonant", "nearby"}:
        raise ValueError(
            "selection_mode must be 'reservoir', 'near_resonant', 'profile_resonant', or 'nearby'"
        )
    if selection_mode == "profile_resonant" and any(value is None for value in (beta0, beta1, beta2)):
        raise ValueError("profile_resonant selection requires beta0, beta1, and beta2")
    order = np.argsort(freqs)
    sorted_freqs = freqs[order]
    width = 2.0 * float(baud_rate)
    rng = np.random.default_rng(seed)
    selected: list[tuple[int, int, int]] = []
    ranked: list[tuple[float, int, tuple[int, int, int]]] = []
    count = 0
    for a in range(freqs.size):
        for b in range(freqs.size):
            center = float(freqs[a] + freqs[b] - freqs[target])
            lo = int(np.searchsorted(sorted_freqs, center - width, side="left"))
            hi = int(np.searchsorted(sorted_freqs, center + width, side="right"))
            for c_raw in order[lo:hi]:
                c = int(c_raw)
                if target in (a, b, c) or len({a, b, c}) != 3:
                    continue
                item = (a, b, c)
                if selection_mode == "reservoir" and (cap is None or len(selected) < cap):
                    selected.append(item)
                elif selection_mode == "reservoir":
                    replacement = int(rng.integers(0, count + 1))
                    if replacement < cap:
                        selected[replacement] = item
                elif selection_mode == "near_resonant":
                    score = abs((freqs[a] - freqs[c]) * (freqs[b] - freqs[c]))
                    if cap is None:
                        ranked.append((-float(score), -count, item))
                    elif len(ranked) < cap:
                        heapq.heappush(ranked, (-float(score), -count, item))
                    elif score < -ranked[0][0]:
                        heapq.heapreplace(ranked, (-float(score), -count, item))
                elif selection_mode == "profile_resonant":
                    delta_omega = 2.0 * np.pi * (
                        freqs[a] + freqs[b] - freqs[c] - freqs[target]
                    )
                    score = abs(
                        beta0[a]
                        + beta0[b]
                        - beta0[c]
                        - beta0[target]
                        - beta1[target] * delta_omega
                        - 0.5 * beta2[target] * delta_omega**2
                    )
                    if cap is None:
                        ranked.append((-float(score), -count, item))
                    elif len(ranked) < cap:
                        heapq.heappush(ranked, (-float(score), -count, item))
                    elif score < -ranked[0][0]:
                        heapq.heapreplace(ranked, (-float(score), -count, item))
                else:
                    score = float(np.ptp(freqs[[target, a, b, c]]))
                    if cap is None:
                        ranked.append((-score, -count, item))
                    elif len(ranked) < cap:
                        heapq.heappush(ranked, (-score, -count, item))
                    elif score < -ranked[0][0]:
                        heapq.heapreplace(ranked, (-score, -count, item))
                count += 1
    if selection_mode != "reservoir":
        selected = [entry[2] for entry in sorted(ranked, key=lambda entry: (-entry[0], -entry[1]))]
    return count, selected


def _channels_for_tuple(
    freqs: np.ndarray,
    beta0: np.ndarray,
    beta1: np.ndarray,
    beta2: np.ndarray,
    target: int,
    a: int,
    b: int,
    c: int,
) -> FWMChannels:
    omega = 2.0 * np.pi * (np.asarray(freqs, dtype=float) - float(freqs[target]))
    beta0_offset = np.asarray(beta0, dtype=float) - float(beta0[target])
    return FWMChannels(
        omega_a=float(omega[a]),
        omega_b=float(omega[b]),
        omega_c=float(omega[c]),
        omega_d=0.0,
        beta0_a=float(beta0_offset[a]),
        beta0_b=float(beta0_offset[b]),
        beta0_c=float(beta0_offset[c]),
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


def _estimate_job(job: tuple) -> tuple[float, float, np.ndarray, np.ndarray]:
    channels, baud_rate, length, n_samples, seeds = job
    estimates = []
    internal_errors = []
    for seed in seeds:
        rng = np.random.default_rng(int(seed))
        paired_count = int(n_samples) // 2
        base = 2.0 * np.pi * (rng.random((3, paired_count)) - 0.5)
        random_variables = np.concatenate((base, base[[1, 0, 2]]), axis=1)
        if int(n_samples) % 2:
            extra = 2.0 * np.pi * (rng.random((3, 1)) - 0.5)
            extra[1] = extra[0]
            random_variables = np.concatenate((random_variables, extra), axis=1)
        result = estimate_fwm_term_sum_dar_mc(
            channels=channels,
            baud_rate=baud_rate,
            length=length,
            n_samples=n_samples,
            seed=None,
            random_variables=random_variables,
        )
        estimates.append(result.total)
        internal_errors.append(result.total_stderr)
    estimates_array = np.asarray(estimates, dtype=float)
    internal_array = np.asarray(internal_errors, dtype=float)
    if estimates_array.size > 1:
        seed_error = float(np.std(estimates_array, ddof=1) / np.sqrt(estimates_array.size))
        mc_error = float(np.sqrt(np.mean(internal_array**2) / estimates_array.size))
        stderr = float(np.hypot(seed_error, mc_error))
    else:
        stderr = float(internal_array[0])
    return float(np.mean(estimates_array)), stderr, estimates_array, internal_array


def _band_labels(system: System, original_indices: np.ndarray) -> np.ndarray:
    labels = np.full(original_indices.size, "unassigned", dtype="U32")
    slices = getattr(system.wdm, "_band_slices", {})
    for name, band_slice in slices.items():
        mask = (original_indices >= int(band_slice.start)) & (original_indices < int(band_slice.stop))
        labels[mask] = str(name)
    return labels


def surrogate_beta2_from_system(system: System) -> float:
    """Prefer the explicitly configured scalar used by the surrogate model."""
    raw = getattr(system, "raw_config", {})
    if isinstance(raw, dict):
        fiber_section = raw.get("fiber", {})
        if isinstance(fiber_section, dict) and fiber_section.get("beta2") is not None:
            return float(fiber_section["beta2"])
    return float(system.fiber.beta2)


def _estimate_zdw(freqs: np.ndarray, beta2: np.ndarray) -> float:
    order = np.argsort(freqs)
    f = np.asarray(freqs, dtype=float)[order]
    b = np.asarray(beta2, dtype=float)[order]
    changes = np.flatnonzero(np.signbit(b[:-1]) != np.signbit(b[1:]))
    if changes.size == 0:
        return float("nan")
    i = int(changes[0])
    return float(f[i] - b[i] * (f[i + 1] - f[i]) / (b[i + 1] - b[i]))


def compute_dataset(
    system: System,
    *,
    decimation: int,
    targets: np.ndarray,
    tuple_cap: int | None,
    n_samples: int,
    n_seeds: int,
    seed: int,
    beta2_const: float,
    workers: int,
    tuple_selection: str = "reservoir",
    include_o_in_fits: bool = False,
) -> dict[str, np.ndarray]:
    full_freqs = np.asarray(system.wdm.frequency_grid(), dtype=float)
    original_indices = np.arange(0, full_freqs.size, int(decimation), dtype=int)
    freqs = full_freqs[original_indices]
    labels = _band_labels(system, original_indices)
    beta1_grid, beta2_grid = system.beta_grids(freqs=freqs)
    beta1 = np.asarray(beta1_grid[0], dtype=float)
    beta2 = np.asarray(beta2_grid[0], dtype=float)
    beta0 = beta0_grid(system, freqs, beta1)
    length = float(system.fiber_length)
    baud_rate = float(system.pulse.baud_rate)
    zdw = _estimate_zdw(freqs, beta2)

    records: list[dict[str, object]] = []
    jobs = []
    for target in np.asarray(targets, dtype=int):
        support_count, tuples = enumerate_fwm_tuples(
            freqs,
            baud_rate=baud_rate,
            target=int(target),
            cap=tuple_cap,
            seed=seed + 999_983 * int(target),
            selection_mode=tuple_selection,
            beta0=beta0,
            beta1=beta1,
            beta2=beta2,
        )
        common_seeds = np.arange(
            seed + 100_003 * int(target),
            seed + 100_003 * int(target) + int(n_seeds),
            dtype=int,
        )
        for a, b, c in tuples:
            channels = _channels_for_tuple(
                freqs, beta0, beta1, beta2, int(target), a, b, c
            )
            geometry = surrogate_geometry(
                freqs[[target, a, b, c]],
                baud_rate=baud_rate,
                length=length,
                beta2_const=beta2_const,
            )
            output_offset = channels.delta_omega
            beta1_d_output = channels.beta1_d + channels.gvd_d * output_offset
            actual_delta_beta_center = (
                channels.delta_beta0
                - channels.beta1_d * output_offset
                - 0.5 * channels.gvd_d * output_offset**2
            )
            actual_gradient = np.array(
                [
                    channels.beta1_a - beta1_d_output,
                    channels.beta1_b - beta1_d_output,
                    -channels.beta1_c + beta1_d_output,
                ],
                dtype=float,
            )
            actual_gradient_norm = float(np.linalg.norm(actual_gradient))
            records.append(
                {
                    "target": int(target),
                    "a": int(a),
                    "b": int(b),
                    "c": int(c),
                    "support_count": int(support_count),
                    "target_frequency": float(freqs[target]),
                    "a_frequency": float(freqs[a]),
                    "b_frequency": float(freqs[b]),
                    "c_frequency": float(freqs[c]),
                    "target_band": labels[target],
                    "a_band": labels[a],
                    "b_band": labels[b],
                    "c_band": labels[c],
                    "canonical_ab": bool(a < b),
                    "contains_o_band": bool(
                        "O" in {str(labels[index]).upper() for index in (target, a, b, c)}
                    ),
                    "delta_beta_center": geometry.delta_beta_center,
                    "gradient_a": geometry.gradient[0],
                    "gradient_b": geometry.gradient[1],
                    "gradient_c": geometry.gradient[2],
                    "gradient_norm": geometry.gradient_norm,
                    "x_grad": geometry.x_grad,
                    "x_phase": geometry.x_phase,
                    "x_combined": geometry.x_combined,
                    "x_quadrature": geometry.x_quadrature,
                    "epsilon_support": geometry.epsilon_support,
                    "separation_ac": geometry.separation_ac,
                    "separation_bc": geometry.separation_bc,
                    "separation_asymmetry": geometry.separation_asymmetry,
                    "curvature": geometry.curvature,
                    "phase_matched_domain": geometry.phase_matched_domain,
                    "actual_delta_beta_center": actual_delta_beta_center,
                    "actual_gradient_norm": actual_gradient_norm,
                    "x_actual_phase": length * abs(actual_delta_beta_center),
                    "x_actual_grad": length * baud_rate * actual_gradient_norm,
                    "x_actual_combined": length
                    * (abs(actual_delta_beta_center) + baud_rate * actual_gradient_norm),
                    "zdw_distance": abs(float(freqs[target]) - zdw) if np.isfinite(zdw) else float("nan"),
                }
            )
            jobs.append(
                (
                    channels,
                    baud_rate,
                    length,
                    int(n_samples),
                    common_seeds,
                )
            )

    worker_count = int(workers)
    if worker_count < 1:
        worker_count = os.cpu_count() or 1
    if worker_count == 1 or len(jobs) < 2:
        estimates = [_estimate_job(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as pool:
            estimates = list(pool.map(_estimate_job, jobs))

    for record, (value, stderr, seed_values, seed_errors) in zip(records, estimates, strict=True):
        record["value"] = value
        record["stderr"] = stderr
        record["seed_values"] = seed_values
        record["seed_errors"] = seed_errors

    fields = list(records[0]) if records else []
    data = {field: np.asarray([record[field] for record in records]) for field in fields}
    data.update(
        {
            "length": np.array(length),
            "baud_rate": np.array(baud_rate),
            "beta2_const": np.array(float(beta2_const)),
            "decimation": np.array(int(decimation)),
            "n_samples": np.array(int(n_samples)),
            "n_seeds": np.array(int(n_seeds)),
            "seed": np.array(int(seed)),
            "tuple_selection": np.array(str(tuple_selection)),
            "include_o_in_fits": np.array(bool(include_o_in_fits)),
            "zdw_frequency": np.array(zdw),
            "original_indices": original_indices,
            "calculation": np.array("standalone_fixed_tuple_fwm_gradient_scaling"),
            "mc_value_convention": np.array("N_times_T_squared"),
            "vertical_normalization": np.array("mc_value/L^2 = N*T^2/L^2"),
        }
    )
    return data


def save_dataset(data: dict[str, np.ndarray], results_dir: Path) -> tuple[Path, Path]:
    results_dir.mkdir(parents=True, exist_ok=True)
    npz_path = results_dir / "fwm_mc_gradient_scaling.npz"
    np.savez(npz_path, **data)
    csv_path = results_dir / "fwm_mc_gradient_scaling.csv"
    row_fields = [
        key
        for key, value in data.items()
        if np.asarray(value).ndim == 1 and np.asarray(value).shape[0] == np.asarray(data.get("value", [])).size
        and key not in {"seed_values", "seed_errors", "original_indices"}
    ]
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(row_fields)
        for i in range(np.asarray(data.get("value", [])).size):
            writer.writerow([data[key][i] for key in row_fields])
    return npz_path, csv_path


def _fit_log_power(x: np.ndarray, y: np.ndarray, stderr: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    valid = mask & np.isfinite(x) & np.isfinite(y) & (x > 0.0) & (y > 0.0)
    if np.count_nonzero(valid) < 3:
        return {"slope": float("nan"), "intercept": float("nan"), "rmse": float("nan"), "r2": float("nan")}
    lx = np.log(x[valid])
    ly = np.log(y[valid])
    relative = np.divide(stderr[valid], y[valid], out=np.ones_like(ly), where=stderr[valid] > 0.0)
    weights = 1.0 / np.maximum(relative, 1e-3)
    slope, intercept = np.polyfit(lx, ly, 1, w=weights)
    predicted = slope * lx + intercept
    residual = ly - predicted
    rmse = float(np.sqrt(np.average(residual**2, weights=weights**2)))
    center = float(np.average(ly, weights=weights**2))
    total = float(np.sum(weights**2 * (ly - center) ** 2))
    r2 = 1.0 - float(np.sum(weights**2 * residual**2)) / total if total > 0.0 else float("nan")
    return {"slope": float(slope), "intercept": float(intercept), "rmse": rmse, "r2": r2}


def fit_metrics(data: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    value = np.asarray(data["value"], dtype=float)
    stderr = np.asarray(data["stderr"], dtype=float)
    y = value / float(data["length"]) ** 2
    include_o = bool(np.asarray(data.get("include_o_in_fits", False)).item())
    canonical_non_o = np.asarray(data["canonical_ab"], dtype=bool)
    if not include_o:
        canonical_non_o &= ~np.asarray(data["contains_o_band"], dtype=bool)
    x_grad = np.asarray(data["x_grad"], dtype=float)
    positive_x = x_grad[canonical_non_o & np.isfinite(x_grad) & (x_grad > 0.0)]
    threshold = float(np.quantile(positive_x, 0.75)) if positive_x.size else float("inf")
    matched = np.asarray(data["phase_matched_domain"], dtype=bool)
    metrics = {
        "all_non_o": _fit_log_power(x_grad, y, stderr / float(data["length"]) ** 2, canonical_non_o),
        "all_non_o_phase": _fit_log_power(
            np.asarray(data["x_phase"], dtype=float),
            y,
            stderr / float(data["length"]) ** 2,
            canonical_non_o,
        ),
        "all_non_o_combined": _fit_log_power(
            np.asarray(data["x_combined"], dtype=float),
            y,
            stderr / float(data["length"]) ** 2,
            canonical_non_o,
        ),
        "all_non_o_quadrature": _fit_log_power(
            np.asarray(data["x_quadrature"], dtype=float),
            y,
            stderr / float(data["length"]) ** 2,
            canonical_non_o,
        ),
        "all_non_o_actual_phase": _fit_log_power(
            np.asarray(data["x_actual_phase"], dtype=float),
            y,
            stderr / float(data["length"]) ** 2,
            canonical_non_o,
        ),
        "all_non_o_actual_combined": _fit_log_power(
            np.asarray(data["x_actual_combined"], dtype=float),
            y,
            stderr / float(data["length"]) ** 2,
            canonical_non_o,
        ),
        "large_x_grad": _fit_log_power(
            x_grad, y, stderr / float(data["length"]) ** 2, canonical_non_o & (x_grad >= threshold)
        ),
        "phase_matched": _fit_log_power(
            x_grad, y, stderr / float(data["length"]) ** 2, canonical_non_o & matched
        ),
        "fully_mismatched": _fit_log_power(
            x_grad, y, stderr / float(data["length"]) ** 2, canonical_non_o & ~matched
        ),
    }
    normalized_stderr = stderr / float(data["length"]) ** 2
    for field in ("epsilon_support", "separation_asymmetry"):
        values = np.asarray(data[field], dtype=float)
        finite = canonical_non_o & np.isfinite(values)
        if np.count_nonzero(finite) < 6:
            continue
        edges = np.unique(np.quantile(values[finite], [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0]))
        for bin_index, (low, high) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
            upper = values <= high if bin_index == len(edges) - 2 else values < high
            mask = finite & (values >= low) & upper
            metrics[f"{field}_bin_{bin_index + 1}"] = _fit_log_power(
                x_grad, y, normalized_stderr, mask
            )
    return metrics


def save_metrics(metrics: dict[str, dict[str, float]], results_dir: Path) -> Path:
    path = results_dir / "fwm_mc_gradient_scaling_metrics.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["subset", "slope", "intercept", "rmse", "r2"])
        for name, values in metrics.items():
            writer.writerow(
                [name, values["slope"], values["intercept"], values["rmse"], values["r2"]]
            )
    return path


def convergence_metrics(datasets: dict[int, dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Compare per-tuple estimates and fitted slopes with the finest MC run."""
    sample_counts = np.array(sorted(datasets), dtype=int)
    if sample_counts.size == 0:
        raise ValueError("at least one convergence dataset is required")
    reference = datasets[int(sample_counts[-1])]
    reference_keys = list(
        zip(reference["target"], reference["a"], reference["b"], reference["c"], strict=True)
    )
    reference_lookup = {tuple(map(int, key)): i for i, key in enumerate(reference_keys)}
    include_o = bool(np.asarray(reference.get("include_o_in_fits", False)).item())
    reference_mask = np.asarray(reference["canonical_ab"], dtype=bool)
    if not include_o:
        reference_mask &= ~np.asarray(reference["contains_o_band"], dtype=bool)
    rows = []
    reference_slope = fit_metrics(reference)["all_non_o"]["slope"]
    for sample_count in sample_counts:
        data = datasets[int(sample_count)]
        keys = list(zip(data["target"], data["a"], data["b"], data["c"], strict=True))
        current_lookup = {tuple(map(int, key)): i for i, key in enumerate(keys)}
        common_keys = [key for key in reference_lookup if key in current_lookup]
        ref_indices = np.array([reference_lookup[key] for key in common_keys], dtype=int)
        current_indices = np.array([current_lookup[key] for key in common_keys], dtype=int)
        keep = reference_mask[ref_indices]
        ref_indices = ref_indices[keep]
        current_indices = current_indices[keep]
        ref_values = np.asarray(reference["value"], dtype=float)[ref_indices]
        values = np.asarray(data["value"], dtype=float)[current_indices]
        ref_stderr = np.asarray(reference["stderr"], dtype=float)[ref_indices]
        stderr = np.asarray(data["stderr"], dtype=float)[current_indices]
        relative = np.divide(
            np.abs(values - ref_values),
            np.maximum(np.abs(ref_values), np.finfo(float).tiny),
        )
        combined_error = np.hypot(stderr, ref_stderr)
        z_score = np.divide(
            np.abs(values - ref_values),
            combined_error,
            out=np.zeros_like(values),
            where=combined_error > 0.0,
        )
        slope = fit_metrics(data)["all_non_o"]["slope"]
        rows.append(
            (
                int(sample_count),
                int(ref_indices.size),
                float(np.median(relative)) if relative.size else float("nan"),
                float(np.quantile(relative, 0.9)) if relative.size else float("nan"),
                float(np.mean(z_score <= 2.0)) if z_score.size else float("nan"),
                slope,
                abs(slope - reference_slope) if np.isfinite(slope) and np.isfinite(reference_slope) else float("nan"),
            )
        )
    values = np.asarray(rows, dtype=float)
    return {
        "n_samples": values[:, 0].astype(int),
        "n_tuples": values[:, 1].astype(int),
        "median_relative_change": values[:, 2],
        "p90_relative_change": values[:, 3],
        "fraction_within_2sigma": values[:, 4],
        "slope": values[:, 5],
        "slope_change": values[:, 6],
    }


def save_convergence(summary: dict[str, np.ndarray], results_dir: Path) -> Path:
    path = results_dir / "fwm_mc_gradient_scaling_convergence.csv"
    fields = list(summary)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        for row in zip(*(summary[field] for field in fields), strict=True):
            writer.writerow(row)
    return path


def plot_convergence(summary: dict[str, np.ndarray], out_dir: Path) -> Path:
    samples = np.asarray(summary["n_samples"], dtype=float)
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.2))
    axes[0].plot(samples, summary["median_relative_change"], marker="o", label="median")
    axes[0].plot(samples, summary["p90_relative_change"], marker="s", label="90th percentile")
    axes[0].axhline(0.05, color="0.35", ls=":", lw=0.8)
    axes[0].axhline(0.20, color="0.35", ls="--", lw=0.8)
    axes[0].set_ylabel("change from finest run")
    axes[0].set_yscale("log")
    axes[0].legend(fontsize=7, frameon=False)
    axes[1].plot(samples, summary["fraction_within_2sigma"], marker="o")
    axes[1].axhline(0.8, color="0.35", ls="--", lw=0.8)
    axes[1].set_ylim(0.0, 1.05)
    axes[1].set_ylabel(r"fraction within $2\sigma$")
    axes[2].plot(samples, summary["slope"], marker="o", label="fitted slope")
    axes[2].axhline(-1.0, color="0.35", ls="--", lw=0.8, label=r"$-1$")
    axes[2].axhline(-2.0, color="0.35", ls=":", lw=0.8, label=r"$-2$")
    axes[2].set_ylabel("gradient-fit slope")
    axes[2].legend(fontsize=7, frameon=False)
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("MC samples per tuple")
        ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    path = out_dir / "mc_convergence.pdf"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def convergence_passes(summary: dict[str, np.ndarray]) -> bool:
    """Apply a compact decision rule to the second-finest run."""
    if np.asarray(summary["n_samples"]).size < 2:
        return False
    index = -2
    return bool(
        summary["median_relative_change"][index] <= 0.05
        and summary["p90_relative_change"][index] <= 0.20
        and summary["fraction_within_2sigma"][index] >= 0.80
        and summary["slope_change"][index] <= 0.15
    )


def _tail_guide(x: np.ndarray, y: np.ndarray, slope: float) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(x)
    count = min(20, max(1, order.size // 4))
    tail = order[-count:]
    scale = float(np.exp(np.mean(np.log(y[tail]) - slope * np.log(x[tail]))))
    grid = np.geomspace(float(np.min(x)), float(np.max(x)), 100)
    return grid, scale * grid**slope


def plot_dataset(data: dict[str, np.ndarray], out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    value = np.asarray(data["value"], dtype=float)
    stderr = np.asarray(data["stderr"], dtype=float)
    length = float(data["length"])
    y = value / length**2
    yerr = stderr / length**2
    canonical = np.asarray(data["canonical_ab"], dtype=bool)
    include_o = bool(np.asarray(data.get("include_o_in_fits", False)).item())
    non_o = np.ones(value.size, dtype=bool) if include_o else ~np.asarray(data["contains_o_band"], dtype=bool)
    matched = np.asarray(data["phase_matched_domain"], dtype=bool)
    valid = canonical & non_o & np.isfinite(y) & (y > 0.0)
    paths = []

    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    for subset, label, marker in ((valid & matched, "phase-matched domain", "o"), (valid & ~matched, "fully mismatched", "s")):
        ax.errorbar(
            np.asarray(data["x_grad"])[subset], y[subset], yerr=yerr[subset],
            marker=marker, ls="none", ms=3, alpha=0.55, capsize=1, label=label,
        )
    guide_mask = valid & (np.asarray(data["x_grad"]) > 0.0)
    if np.count_nonzero(guide_mask) >= 4:
        for slope, style in ((-1.0, "--"), (-2.0, ":")):
            gx, gy = _tail_guide(np.asarray(data["x_grad"])[guide_mask], y[guide_mask], slope)
            ax.plot(gx, gy, color="0.25", ls=style, lw=0.9, label=rf"tail guide $x^{{{slope:g}}}$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$x_\nabla=L B\|\nabla_{\Omega}\Delta\beta\|_2$")
    ax.set_ylabel(r"$N_{dabc}T^2/L^2$")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    path = out_dir / "gradient_collapse.pdf"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    combined = np.asarray(data["x_combined"], dtype=float)
    for subset, label, marker in (
        (valid & matched, "phase-matched domain", "o"),
        (valid & ~matched, "fully mismatched", "s"),
    ):
        ax.errorbar(
            combined[subset],
            y[subset],
            yerr=yerr[subset],
            marker=marker,
            ls="none",
            ms=3,
            alpha=0.55,
            capsize=1,
            label=label,
        )
    guide_mask = valid & np.isfinite(combined) & (combined > 0.0)
    if np.count_nonzero(guide_mask) >= 4:
        for slope, style in ((-1.0, "--"), (-2.0, ":")):
            gx, gy = _tail_guide(combined[guide_mask], y[guide_mask], slope)
            ax.plot(gx, gy, color="0.25", ls=style, lw=0.9, label=rf"tail guide $x^{{{slope:g}}}$")
    combined_fit = fit_metrics(data)["all_non_o_combined"]
    ax.text(
        0.03,
        0.04,
        rf"slope={combined_fit['slope']:.2f}, $R^2$={combined_fit['r2']:.3f}",
        transform=ax.transAxes,
        fontsize=8,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$L(|\Delta\beta_0|+B\|\nabla_\Omega\Delta\beta\|_2)$")
    ax.set_ylabel(r"$N_{dabc}T^2/L^2$")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    path = out_dir / "combined_collapse.pdf"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    paths.append(path)

    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.5))
    axes[0].scatter(np.asarray(data["x_phase"])[valid], y[valid], c=np.asarray(data["separation_asymmetry"])[valid], s=12)
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel(r"$L|\Delta\beta_{\rm center}|$")
    axes[0].set_ylabel(r"$N_{dabc}T^2/L^2$")
    axes[1].scatter(
        np.asarray(data["x_combined"])[valid],
        y[valid],
        c=np.asarray(data["separation_asymmetry"])[valid],
        s=12,
    )
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel(r"$L(|\Delta\beta_0|+B\|\nabla_\Omega\Delta\beta\|_2)$")
    axes[1].set_ylabel(r"$N_{dabc}T^2/L^2$")
    plateau = np.asarray(data["x_grad"])[valid] * y[valid]
    axes[2].scatter(np.asarray(data["x_grad"])[valid], plateau, c=np.asarray(data["epsilon_support"])[valid], s=12)
    axes[2].set_xscale("log")
    axes[2].set_yscale("log")
    axes[2].set_xlabel(r"$x_\nabla$")
    axes[2].set_ylabel(r"$x_\nabla N_{dabc}T^2/L^2$")
    for ax in axes:
        ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    path = out_dir / "competing_scales.pdf"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    paths.append(path)

    metrics = fit_metrics(data)
    fit = metrics["all_non_o"]
    x = np.asarray(data["x_grad"], dtype=float)
    fit_valid = valid & np.isfinite(x) & (x > 0.0)
    residual = np.full_like(y, np.nan)
    if np.isfinite(fit["slope"]):
        residual[fit_valid] = np.log(y[fit_valid]) - (fit["slope"] * np.log(x[fit_valid]) + fit["intercept"])
    fig, axes = plt.subplots(2, 3, figsize=(11.4, 6.0))
    diagnostics = (
        ("epsilon_support", r"$\epsilon_{\rm support}$"),
        ("separation_asymmetry", "separation asymmetry"),
        ("x_phase", r"$L|\Delta\beta_{\rm center}|$"),
        ("target_frequency", "target frequency [THz]"),
        ("zdw_distance", "target distance from ZDW [THz]"),
    )
    for ax, (name, label) in zip(axes.ravel(), diagnostics, strict=False):
        xv = np.asarray(data[name], dtype=float).copy()
        if name in {"target_frequency", "zdw_distance"}:
            xv *= 1e-12
        ax.scatter(xv[fit_valid], residual[fit_valid], s=10, alpha=0.55)
        if name == "x_phase":
            ax.set_xscale("log")
        ax.axhline(0.0, color="0.3", lw=0.7, ls=":")
        ax.set_xlabel(label)
        ax.set_ylabel("log-fit residual")
        ax.grid(True, alpha=0.25)
    axes.ravel()[-1].axis("off")
    fig.tight_layout()
    path = out_dir / "fit_residuals.pdf"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    paths.append(path)
    return paths


def _parse_targets(value: str | None, n_channels: int) -> np.ndarray:
    if value:
        targets = np.array([int(part) for part in value.split(",") if part.strip()], dtype=int)
    else:
        targets = np.arange(n_channels, dtype=int)
    if np.any((targets < 0) | (targets >= n_channels)):
        raise ValueError("targets refer to positions on the decimated channel grid")
    return targets


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone fixed-channel FWM gradient-scaling study.")
    parser.add_argument("--config", type=Path, default=Path("input/studies.toml"))
    parser.add_argument("--out-dir", type=Path, default=OUT_MEDIA)
    parser.add_argument("--results-dir", type=Path, default=OUT_RESULTS)
    parser.add_argument("--decimation", type=int, default=100)
    parser.add_argument("--targets", type=str, default=None, help="Decimated-grid positions, comma separated.")
    parser.add_argument("--target-band", type=str, default=None, help="Configured WDM band label.")
    parser.add_argument("--tuple-cap", type=int, default=2000)
    parser.add_argument(
        "--tuple-selection",
        choices=("reservoir", "near_resonant", "profile_resonant", "nearby"),
        default="reservoir",
        help="Uniform sample, smallest constant-beta2 mismatch, smallest measured-profile mismatch, or smallest span.",
    )
    parser.add_argument("--n-samples", type=int, default=5000)
    parser.add_argument("--n-seeds", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--workers", type=int, default=1, help="0 uses all CPUs.")
    parser.add_argument("--beta2-const", type=float, default=None)
    parser.add_argument(
        "--include-o-in-fits",
        action="store_true",
        help="Include tuples touching the configured O band in fits and plots.",
    )
    parser.add_argument(
        "--quick-convergence",
        action="store_true",
        help="Compare three MC sample counts and use the finest run for the scaling plots.",
    )
    parser.add_argument(
        "--convergence-samples",
        type=str,
        default=None,
        help="Comma-separated MC sample counts, e.g. 250,1000,4000.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.decimation < 1 or args.n_samples < 1 or args.n_seeds < 1:
        raise ValueError("decimation, n-samples, and n-seeds must be positive")
    system = System.from_toml(args.config)
    original_indices = np.arange(0, len(system.wdm.frequency_grid()), args.decimation, dtype=int)
    labels = _band_labels(system, original_indices)
    targets = _parse_targets(args.targets, original_indices.size)
    if args.target_band:
        targets = targets[labels[targets] == args.target_band]
        if targets.size == 0:
            raise ValueError(f"no targets remain in configured band {args.target_band!r}")
    beta2_const = float(
        surrogate_beta2_from_system(system) if args.beta2_const is None else args.beta2_const
    )
    convergence_mode = args.quick_convergence or args.convergence_samples is not None
    if args.convergence_samples:
        sample_grid = sorted(
            {int(part) for part in args.convergence_samples.split(",") if part.strip()}
        )
    elif convergence_mode:
        sample_grid = sorted(
            {
                max(1, min(args.n_samples, max(64, args.n_samples // 16))),
                max(1, min(args.n_samples, max(128, args.n_samples // 4))),
                int(args.n_samples),
            }
        )
    else:
        sample_grid = [int(args.n_samples)]
    if not sample_grid or any(value <= 0 for value in sample_grid):
        raise ValueError("convergence sample counts must be positive")
    datasets = {
        sample_count: compute_dataset(
            system,
            decimation=args.decimation,
            targets=targets,
            tuple_cap=None if args.tuple_cap is None or args.tuple_cap <= 0 else args.tuple_cap,
            n_samples=sample_count,
            n_seeds=args.n_seeds,
            seed=args.seed,
            beta2_const=beta2_const,
            workers=args.workers,
            tuple_selection=args.tuple_selection,
            include_o_in_fits=args.include_o_in_fits,
        )
        for sample_count in sample_grid
    }
    data = datasets[sample_grid[-1]]
    if np.asarray(data.get("value", [])).size == 0:
        raise RuntimeError("no support-compatible fixed-channel FWM tuples were selected")
    npz_path, csv_path = save_dataset(data, args.results_dir)
    paths = plot_dataset(data, args.out_dir)
    metrics = fit_metrics(data)
    metrics_path = save_metrics(metrics, args.results_dir)
    print(f"saved {npz_path}")
    print(f"saved {csv_path}")
    print(f"saved {metrics_path}")
    for path in paths:
        print(f"saved {path}")
    if convergence_mode:
        summary = convergence_metrics(datasets)
        convergence_path = save_convergence(summary, args.results_dir)
        convergence_plot = plot_convergence(summary, args.out_dir)
        print(f"saved {convergence_path}")
        print(f"saved {convergence_plot}")
        for index, sample_count in enumerate(summary["n_samples"]):
            print(
                f"convergence n={sample_count}: median={summary['median_relative_change'][index]:.3%}, "
                f"p90={summary['p90_relative_change'][index]:.3%}, "
                f"within_2sigma={summary['fraction_within_2sigma'][index]:.1%}, "
                f"slope={summary['slope'][index]:.4g}"
            )
        print(f"quick convergence: {'PASS' if convergence_passes(summary) else 'NOT CONVERGED'}")
    for name, values in metrics.items():
        print(
            f"{name}: slope={values['slope']:.4g}, rmse={values['rmse']:.4g}, "
            f"R2={values['r2']:.4g}"
        )


if __name__ == "__main__":
    main()
