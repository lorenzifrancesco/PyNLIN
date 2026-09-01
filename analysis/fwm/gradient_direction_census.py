"""Exact angular census of strict-FWM linear phase-gradient directions.

Enumerates the full support-surviving ordered strict-FWM tuple population on
the physical channel grid, but accumulates only angle histograms and moments.
No tuple sampling or interferer-grid decimation is used.

The signed gradient vector is

    c = (nu_a, nu_b, -nu_c)

and its direction is reported by azimuth atan2(c_b, c_a) and elevation
asin(c_c / ||c||). Uniformity is assessed against equal probability per unit
solid angle, not against equal counts in raw elevation bins.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pynlin  # noqa: E402,F401
from loguru import logger as lg  # noqa: E402

from analysis.log_init import init_logging  # noqa: E402
from pynlin.methods.td.fullband_mc import decimated_frequency_grid  # noqa: E402
from pynlin.system import System  # noqa: E402

SCHEMA_VERSION = 1
CODE_OR_MODEL_VERSION = "gradient_direction_census_v1"
_MASK_DIRECTION = np.array([1.0, 1.0, -1.0]) / np.sqrt(3.0)
# Equation (10.4.7), permuted from companion-note slots (1, 2, 3) to the
# census convention (a, b, conjugated c) = (1, 3, 2).
_PLANE_DIRECTIONS = (
    (r"$\mathbf u_1$", np.array([1.0, 0.0, -1.0]) / np.sqrt(2.0)),
    (r"$\mathbf u_2$", np.array([0.0, -1.0, 1.0]) / np.sqrt(2.0)),
    (r"$\mathbf u_3$", np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)),
)

_WORKER_FREQUENCIES_HZ: np.ndarray | None = None
_WORKER_BETA1_S_PER_M: np.ndarray | None = None
_WORKER_SYMBOL_RATE_BAUD: float | None = None
_WORKER_N_AZIMUTH_BINS: int | None = None
_WORKER_N_ELEVATION_BINS: int | None = None
_WORKER_N_ALIGNMENT_BINS: int | None = None
_WORKER_A_CHUNK_SIZE: int | None = None


def _init_worker(
    frequencies_hz: np.ndarray,
    beta1_s_per_m: np.ndarray,
    symbol_rate_baud: float,
    n_azimuth_bins: int,
    n_elevation_bins: int,
    n_alignment_bins: int,
    a_chunk_size: int,
) -> None:
    global _WORKER_FREQUENCIES_HZ, _WORKER_BETA1_S_PER_M
    global _WORKER_SYMBOL_RATE_BAUD, _WORKER_N_AZIMUTH_BINS
    global _WORKER_N_ELEVATION_BINS, _WORKER_N_ALIGNMENT_BINS
    global _WORKER_A_CHUNK_SIZE
    _WORKER_FREQUENCIES_HZ = frequencies_hz
    _WORKER_BETA1_S_PER_M = beta1_s_per_m
    _WORKER_SYMBOL_RATE_BAUD = symbol_rate_baud
    _WORKER_N_AZIMUTH_BINS = n_azimuth_bins
    _WORKER_N_ELEVATION_BINS = n_elevation_bins
    _WORKER_N_ALIGNMENT_BINS = n_alignment_bins
    _WORKER_A_CHUNK_SIZE = a_chunk_size


def _bin_directions(
    component_a: np.ndarray,
    component_b: np.ndarray,
    component_c_conjugated: np.ndarray,
    n_azimuth_bins: int,
    n_elevation_bins: int,
    n_alignment_bins: int,
) -> tuple[np.ndarray, np.ndarray, int, np.ndarray, np.ndarray]:
    """Return angular counts, alignment counts, zero count, and moments."""
    gradient_norm = np.sqrt(
        component_a**2 + component_b**2 + component_c_conjugated**2
    )
    defined = gradient_norm > 0.0
    undefined_count = int(np.sum(~defined))
    if not np.any(defined):
        return (
            np.zeros((n_elevation_bins, n_azimuth_bins), dtype=np.int64),
            np.zeros(n_alignment_bins, dtype=np.int64),
            undefined_count,
            np.zeros(3),
            np.zeros((3, 3)),
        )

    inverse_norm = 1.0 / gradient_norm[defined]
    directions = np.column_stack((
        component_a[defined] * inverse_norm,
        component_b[defined] * inverse_norm,
        component_c_conjugated[defined] * inverse_norm,
    ))
    azimuth_rad = np.arctan2(directions[:, 1], directions[:, 0])
    elevation_rad = np.arcsin(np.clip(directions[:, 2], -1.0, 1.0))
    azimuth_bin = np.floor(
        (azimuth_rad + np.pi) * n_azimuth_bins / (2.0 * np.pi)
    ).astype(np.int64)
    elevation_bin = np.floor(
        (elevation_rad + 0.5 * np.pi) * n_elevation_bins / np.pi
    ).astype(np.int64)
    np.clip(azimuth_bin, 0, n_azimuth_bins - 1, out=azimuth_bin)
    np.clip(elevation_bin, 0, n_elevation_bins - 1, out=elevation_bin)
    flat_bin = elevation_bin * n_azimuth_bins + azimuth_bin
    angular_counts = np.bincount(
        flat_bin, minlength=n_elevation_bins * n_azimuth_bins
    ).reshape(n_elevation_bins, n_azimuth_bins)

    alignment_cosine = directions @ _MASK_DIRECTION
    alignment_bin = np.floor(
        (alignment_cosine + 1.0) * n_alignment_bins / 2.0
    ).astype(np.int64)
    np.clip(alignment_bin, 0, n_alignment_bins - 1, out=alignment_bin)
    alignment_counts = np.bincount(
        alignment_bin, minlength=n_alignment_bins
    )
    direction_sum = np.sum(directions, axis=0)
    direction_outer_sum = directions.T @ directions
    return (
        angular_counts,
        alignment_counts,
        undefined_count,
        direction_sum,
        direction_outer_sum,
    )


def census_target(
    target_idx: int,
    frequencies_hz: np.ndarray,
    beta1_s_per_m: np.ndarray,
    symbol_rate_baud: float,
    *,
    n_azimuth_bins: int,
    n_elevation_bins: int,
    n_alignment_bins: int,
    a_chunk_size: int,
) -> tuple[int, np.ndarray, np.ndarray, int, int, np.ndarray, np.ndarray, float]:
    """Enumerate and bin every ordered strict tuple for one target channel."""
    started = time.perf_counter()
    frequencies_hz = np.asarray(frequencies_hz, dtype=float)
    beta1_s_per_m = np.asarray(beta1_s_per_m, dtype=float)
    n_channels = frequencies_hz.size
    target_idx = int(target_idx)
    target_frequency_hz = frequencies_hz[target_idx]
    target_beta1_s_per_m = beta1_s_per_m[target_idx]
    sorted_positions = np.argsort(frequencies_hz)
    sorted_frequencies_hz = frequencies_hz[sorted_positions]
    support_halfwidth_hz = 2.0 * float(symbol_rate_baud)
    b_all = np.arange(n_channels, dtype=np.int64)

    angular_counts = np.zeros(
        (n_elevation_bins, n_azimuth_bins), dtype=np.int64
    )
    alignment_counts = np.zeros(n_alignment_bins, dtype=np.int64)
    tuple_count = 0
    undefined_direction_count = 0
    direction_sum = np.zeros(3)
    direction_outer_sum = np.zeros((3, 3))

    for a_start in range(0, n_channels, a_chunk_size):
        a_parts: list[np.ndarray] = []
        b_parts: list[np.ndarray] = []
        c_parts: list[np.ndarray] = []
        for a_idx in range(a_start, min(a_start + a_chunk_size, n_channels)):
            if a_idx == target_idx:
                continue
            centers_hz = frequencies_hz[a_idx] + frequencies_hz - target_frequency_hz
            lower = np.searchsorted(
                sorted_frequencies_hz,
                centers_hz - support_halfwidth_hz,
                side="left",
            )
            upper = np.searchsorted(
                sorted_frequencies_hz,
                centers_hz + support_halfwidth_hz,
                side="right",
            )
            counts = upper - lower
            counts[target_idx] = 0
            counts[a_idx] = 0
            surviving_b = np.flatnonzero(counts > 0)
            if surviving_b.size == 0:
                continue
            surviving_counts = counts[surviving_b]
            expanded_count = int(np.sum(surviving_counts))
            starts = np.repeat(lower[surviving_b], surviving_counts)
            ends = np.cumsum(surviving_counts)
            within = np.arange(expanded_count) - np.repeat(
                ends - surviving_counts, surviving_counts
            )
            c_idx = sorted_positions[starts + within]
            b_idx = np.repeat(b_all[surviving_b], surviving_counts)
            strict = (
                (c_idx != target_idx)
                & (c_idx != a_idx)
                & (c_idx != b_idx)
            )
            if not np.any(strict):
                continue
            c_idx = c_idx[strict]
            b_idx = b_idx[strict]
            a_parts.append(np.full(c_idx.size, a_idx, dtype=np.int32))
            b_parts.append(b_idx.astype(np.int32))
            c_parts.append(c_idx.astype(np.int32))

        if not a_parts:
            continue
        a_indices = np.concatenate(a_parts)
        b_indices = np.concatenate(b_parts)
        c_indices = np.concatenate(c_parts)
        component_a = beta1_s_per_m[a_indices] - target_beta1_s_per_m
        component_b = beta1_s_per_m[b_indices] - target_beta1_s_per_m
        component_c_conjugated = -(
            beta1_s_per_m[c_indices] - target_beta1_s_per_m
        )
        (
            chunk_angular_counts,
            chunk_alignment_counts,
            chunk_undefined_count,
            chunk_direction_sum,
            chunk_direction_outer_sum,
        ) = _bin_directions(
            component_a,
            component_b,
            component_c_conjugated,
            n_azimuth_bins,
            n_elevation_bins,
            n_alignment_bins,
        )
        chunk_count = int(a_indices.size)
        tuple_count += chunk_count
        undefined_direction_count += chunk_undefined_count
        angular_counts += chunk_angular_counts
        alignment_counts += chunk_alignment_counts
        direction_sum += chunk_direction_sum
        direction_outer_sum += chunk_direction_outer_sum

    return (
        target_idx,
        angular_counts,
        alignment_counts,
        tuple_count,
        undefined_direction_count,
        direction_sum,
        direction_outer_sum,
        time.perf_counter() - started,
    )


def _worker_census(target_idx: int):
    assert _WORKER_FREQUENCIES_HZ is not None
    assert _WORKER_BETA1_S_PER_M is not None
    assert _WORKER_SYMBOL_RATE_BAUD is not None
    assert _WORKER_N_AZIMUTH_BINS is not None
    assert _WORKER_N_ELEVATION_BINS is not None
    assert _WORKER_N_ALIGNMENT_BINS is not None
    assert _WORKER_A_CHUNK_SIZE is not None
    return census_target(
        target_idx,
        _WORKER_FREQUENCIES_HZ,
        _WORKER_BETA1_S_PER_M,
        _WORKER_SYMBOL_RATE_BAUD,
        n_azimuth_bins=_WORKER_N_AZIMUTH_BINS,
        n_elevation_bins=_WORKER_N_ELEVATION_BINS,
        n_alignment_bins=_WORKER_N_ALIGNMENT_BINS,
        a_chunk_size=_WORKER_A_CHUNK_SIZE,
    )


def _uniform_angular_probabilities(
    azimuth_edges_rad: np.ndarray, elevation_edges_rad: np.ndarray
) -> np.ndarray:
    azimuth_fraction = np.diff(azimuth_edges_rad) / (2.0 * np.pi)
    elevation_fraction = 0.5 * np.diff(np.sin(elevation_edges_rad))
    return elevation_fraction[:, None] * azimuth_fraction[None, :]


def _save_results(
    path: Path,
    *,
    frequencies_hz: np.ndarray,
    symbol_rate_baud: float,
    fiber_length_m: float,
    target_indices_in_full_grid: np.ndarray,
    done: np.ndarray,
    angular_counts: np.ndarray,
    alignment_counts: np.ndarray,
    tuple_count: int,
    undefined_direction_count: int,
    direction_sum: np.ndarray,
    direction_outer_sum: np.ndarray,
    azimuth_edges_rad: np.ndarray,
    elevation_edges_rad: np.ndarray,
    alignment_cosine_edges: np.ndarray,
    target_tuple_counts: np.ndarray,
    target_wall_times_s: np.ndarray,
    target_grid_stride: int,
) -> None:
    units = {
        "frequencies_hz": "Hz",
        "target_indices_in_full_grid": "1",
        "done": "1",
        "angular_tuple_counts": "1",
        "alignment_tuple_counts": "1",
        "strict_fwm_tuple_count": "1",
        "undefined_direction_tuple_count": "1",
        "direction_sum": "1",
        "direction_outer_sum": "1",
        "azimuth_edges_rad": "rad",
        "elevation_edges_rad": "rad",
        "alignment_cosine_edges": "1",
        "target_tuple_counts": "1",
        "target_wall_times_s": "s",
        "symbol_rate_baud": "baud",
        "fiber_length_m": "m",
        "target_grid_stride": "1",
        "interferer_grid_stride": "1",
    }
    axes = {
        "frequencies_hz": ["full_channel"],
        "target_indices_in_full_grid": ["target_position"],
        "done": ["full_channel"],
        "angular_tuple_counts": ["elevation_bin", "azimuth_bin"],
        "alignment_tuple_counts": ["alignment_cosine_bin"],
        "strict_fwm_tuple_count": [],
        "undefined_direction_tuple_count": [],
        "direction_sum": ["gradient_component"],
        "direction_outer_sum": ["gradient_component", "gradient_component"],
        "azimuth_edges_rad": ["azimuth_edge"],
        "elevation_edges_rad": ["elevation_edge"],
        "alignment_cosine_edges": ["alignment_cosine_edge"],
        "target_tuple_counts": ["full_channel"],
        "target_wall_times_s": ["full_channel"],
        "symbol_rate_baud": [],
        "fiber_length_m": [],
        "target_grid_stride": [],
        "interferer_grid_stride": [],
    }
    frequency_grid_hash = hashlib.sha256(
        np.ascontiguousarray(frequencies_hz, dtype="<f8").tobytes()
    ).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        schema_version=SCHEMA_VERSION,
        units=json.dumps(units, sort_keys=True),
        axes=json.dumps(axes, sort_keys=True),
        phase_model="linear_gradient_direction",
        power_basis="single_pol",
        reference_plane="launch_ref",
        tuple_ordering="ordered_ab",
        frequency_grid_hash=frequency_grid_hash,
        symbol_rate_baud=symbol_rate_baud,
        fiber_length_m=fiber_length_m,
        gamma_model="not_applied_direction_only",
        code_or_model_version=CODE_OR_MODEL_VERSION,
        target_grid_stride=target_grid_stride,
        interferer_grid_stride=1,
        frequencies_hz=frequencies_hz,
        target_indices_in_full_grid=target_indices_in_full_grid,
        done=done,
        angular_tuple_counts=angular_counts,
        alignment_tuple_counts=alignment_counts,
        strict_fwm_tuple_count=np.int64(tuple_count),
        undefined_direction_tuple_count=np.int64(undefined_direction_count),
        direction_sum=direction_sum,
        direction_outer_sum=direction_outer_sum,
        azimuth_edges_rad=azimuth_edges_rad,
        elevation_edges_rad=elevation_edges_rad,
        alignment_cosine_edges=alignment_cosine_edges,
        target_tuple_counts=target_tuple_counts,
        target_wall_times_s=target_wall_times_s,
    )


def _load_checkpoint(
    path: Path,
    n_channels: int,
    n_elevation_bins: int,
    n_azimuth_bins: int,
    n_alignment_bins: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    int,
    int,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    if not path.exists():
        return (
            np.zeros(n_channels, dtype=bool),
            np.zeros((n_elevation_bins, n_azimuth_bins), dtype=np.int64),
            0,
            0,
            np.zeros(3),
            np.zeros((3, 3)),
            np.zeros(n_alignment_bins, dtype=np.int64),
            np.zeros(n_channels, dtype=np.int64),
            np.full(n_channels, np.nan),
        )
    with np.load(path) as saved:
        if int(saved["schema_version"]) != SCHEMA_VERSION:
            raise ValueError("checkpoint schema version does not match this script")
        if str(saved["code_or_model_version"]) != CODE_OR_MODEL_VERSION:
            raise ValueError("checkpoint model version does not match this script")
        angular_counts = saved["angular_tuple_counts"]
        alignment_counts = saved["alignment_tuple_counts"]
        if angular_counts.shape != (n_elevation_bins, n_azimuth_bins):
            raise ValueError("checkpoint angular bin shape does not match this run")
        if alignment_counts.shape != (n_alignment_bins,):
            raise ValueError("checkpoint alignment bin shape does not match this run")
        return (
            saved["done"].astype(bool),
            angular_counts.astype(np.int64),
            int(saved["strict_fwm_tuple_count"]),
            int(saved["undefined_direction_tuple_count"]),
            saved["direction_sum"].astype(float),
            saved["direction_outer_sum"].astype(float),
            alignment_counts.astype(np.int64),
            saved["target_tuple_counts"].astype(np.int64),
            saved["target_wall_times_s"].astype(float),
        )


def _uniformity_statistics(
    angular_counts: np.ndarray,
    expected_probabilities: np.ndarray,
    direction_sum: np.ndarray,
    direction_outer_sum: np.ndarray,
) -> dict[str, float | list[float]]:
    defined_count = int(np.sum(angular_counts))
    observed_probability = angular_counts / max(defined_count, 1)
    positive = observed_probability > 0.0
    ratio = observed_probability / expected_probabilities
    total_variation = 0.5 * float(
        np.sum(np.abs(observed_probability - expected_probabilities))
    )
    kl_divergence = float(np.sum(
        observed_probability[positive]
        * np.log(observed_probability[positive] / expected_probabilities[positive])
    ))
    mean_direction = direction_sum / max(defined_count, 1)
    second_moment = direction_outer_sum / max(defined_count, 1)
    eigenvalues = np.linalg.eigvalsh(second_moment)
    return {
        "defined_tuple_count": defined_count,
        "total_variation_distance": total_variation,
        "kl_divergence_nats": kl_divergence,
        "minimum_observed_over_uniform": float(np.min(ratio)),
        "maximum_observed_over_uniform": float(np.max(ratio)),
        "mean_direction_resultant": float(np.linalg.norm(mean_direction)),
        "mean_direction": mean_direction.tolist(),
        "direction_second_moment_eigenvalues": eigenvalues.tolist(),
        "second_moment_max_deviation_from_one_third": float(
            np.max(np.abs(eigenvalues - 1.0 / 3.0))
        ),
    }


def _plot_results(
    out_path: Path,
    angular_counts: np.ndarray,
    alignment_counts: np.ndarray,
    azimuth_edges_rad: np.ndarray,
    elevation_edges_rad: np.ndarray,
    alignment_cosine_edges: np.ndarray,
) -> None:
    defined_count = int(np.sum(angular_counts))
    expected_probability = _uniform_angular_probabilities(
        azimuth_edges_rad, elevation_edges_rad
    )
    observed_probability = angular_counts / max(defined_count, 1)
    ratio = observed_probability / expected_probability
    log_ratio = np.log10(np.maximum(ratio, 1e-12))
    degree_azimuth_edges = np.rad2deg(azimuth_edges_rad)
    degree_elevation_edges = np.rad2deg(elevation_edges_rad)

    matplotlib.rcParams.update({
        "font.size": 13,
        "axes.labelsize": 14,
        "axes.titlesize": 15,
        "figure.titlesize": 17,
    })
    fig, axes = plt.subplots(1, 3, figsize=(18.0, 5.2))
    probability_density_per_sr = observed_probability / (
        4.0 * np.pi * expected_probability
    )
    positive_density = probability_density_per_sr[probability_density_per_sr > 0]
    image = axes[0].pcolormesh(
        degree_azimuth_edges,
        degree_elevation_edges,
        np.log10(np.maximum(probability_density_per_sr, positive_density.min())),
        shading="auto",
        cmap="viridis",
    )
    fig.colorbar(image, ax=axes[0], label=r"$\log_{10}$ probability density [sr$^{-1}$]")
    axes[0].set_title("Observed gradient directions")
    axes[0].set_xlabel("azimuth atan2($c_b,c_a$) [deg]")
    axes[0].set_ylabel("elevation asin($c_c/\|c\|$) [deg]")

    limit = max(float(np.percentile(np.abs(log_ratio), 99.0)), 0.1)
    image = axes[1].pcolormesh(
        degree_azimuth_edges,
        degree_elevation_edges,
        log_ratio,
        shading="auto",
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit,
    )
    fig.colorbar(image, ax=axes[1], label=r"$\log_{10}$(observed / uniform-solid-angle)")
    axes[1].set_title("Departure from a uniform sphere")
    axes[1].set_xlabel("azimuth [deg]")
    axes[1].set_ylabel("elevation [deg]")

    annotation_offsets = ((8, -16), (8, 8), (8, 8))
    for (label, direction), offset in zip(
        _PLANE_DIRECTIONS, annotation_offsets, strict=True
    ):
        azimuth_deg = np.rad2deg(np.arctan2(direction[1], direction[0]))
        elevation_deg = np.rad2deg(np.arcsin(direction[2]))
        for ax in axes[:2]:
            ax.scatter(
                azimuth_deg,
                elevation_deg,
                marker="*",
                s=190,
                facecolor="#ffd84d",
                edgecolor="black",
                linewidth=0.9,
                zorder=5,
            )
            ax.annotate(
                label,
                (azimuth_deg, elevation_deg),
                xytext=offset,
                textcoords="offset points",
                color="black",
                fontsize=12,
                fontweight="bold",
                bbox={"boxstyle": "round,pad=0.15", "fc": "white", "alpha": 0.8, "ec": "none"},
                zorder=6,
            )

    alignment_centers = 0.5 * (
        alignment_cosine_edges[:-1] + alignment_cosine_edges[1:]
    )
    alignment_width = np.diff(alignment_cosine_edges)
    alignment_probability_density = (
        alignment_counts / max(np.sum(alignment_counts), 1) / alignment_width
    )
    axes[2].plot(
        alignment_centers,
        alignment_probability_density,
        color="#8b1e3f",
        lw=2.0,
        label="system tuples",
    )
    axes[2].axhline(0.5, color="black", ls="--", label="uniform sphere")
    axes[2].set_xlabel(r"alignment $\hat{\mathbf c}\cdot(1,1,-1)/\sqrt3$")
    axes[2].set_ylabel("probability density")
    axes[2].set_title("Alignment with the mask normal")
    axes[2].legend()
    axes[2].grid(alpha=0.25)
    alignment_by_label = {
        label: float(direction @ _MASK_DIRECTION)
        for label, direction in _PLANE_DIRECTIONS
    }
    grouped_alignments = (
        (r"$\mathbf u_2$", alignment_by_label[r"$\mathbf u_2$"]),
        (
            r"$\mathbf u_1,\mathbf u_3$",
            alignment_by_label[r"$\mathbf u_1$"],
        ),
    )
    for label, alignment in grouped_alignments:
        density = float(np.interp(
            alignment, alignment_centers, alignment_probability_density
        ))
        axes[2].scatter(
            alignment,
            density,
            marker="*",
            s=190,
            facecolor="#ffd84d",
            edgecolor="black",
            linewidth=0.9,
            zorder=5,
        )
        axes[2].annotate(
            label,
            (alignment, density),
            xytext=(7, 7),
            textcoords="offset points",
            fontsize=12,
            fontweight="bold",
        )

    fig.suptitle(
        f"Strict-FWM linear phase-gradient directions ({defined_count:,} defined tuples)"
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    init_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("input/studies.toml"))
    parser.add_argument("--out-dir", type=Path, default=Path("media/lorenzi-fast"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--target-stride", type=int, default=1)
    parser.add_argument("--n-azimuth-bins", type=int, default=144)
    parser.add_argument("--n-elevation-bins", type=int, default=72)
    parser.add_argument("--n-alignment-bins", type=int, default=120)
    parser.add_argument("--a-chunk-size", type=int, default=128)
    parser.add_argument("--checkpoint-every", type=int, default=20)
    args = parser.parse_args()

    system = System.from_toml(args.config)
    _, frequencies_hz = decimated_frequency_grid(system, 1)
    beta1_grid, _ = system.beta_grids(freqs=frequencies_hz)
    beta1_s_per_m = np.asarray(beta1_grid[0], dtype=float)
    symbol_rate_baud = float(system.pulse.baud_rate)
    fiber_length_m = float(system.fiber_length)
    n_channels = frequencies_hz.size
    target_stride = max(int(args.target_stride), 1)
    targets = np.arange(0, n_channels, target_stride, dtype=np.int64)
    workers = args.workers if args.workers > 0 else min(max((os.cpu_count() or 2) // 2, 1), 8)
    checkpoint_path = args.checkpoint or args.out_dir / "gradient_direction_census.npz"

    azimuth_edges_rad = np.linspace(-np.pi, np.pi, args.n_azimuth_bins + 1)
    elevation_edges_rad = np.linspace(
        -0.5 * np.pi, 0.5 * np.pi, args.n_elevation_bins + 1
    )
    alignment_cosine_edges = np.linspace(-1.0, 1.0, args.n_alignment_bins + 1)
    (
        done,
        angular_counts,
        tuple_count,
        undefined_direction_count,
        direction_sum,
        direction_outer_sum,
        alignment_counts,
        target_tuple_counts,
        target_wall_times_s,
    ) = _load_checkpoint(
        checkpoint_path,
        n_channels,
        args.n_elevation_bins,
        args.n_azimuth_bins,
        args.n_alignment_bins,
    )
    remaining = targets[~done[targets]]
    lg.info(
        f"{n_channels} channels on the full interferer grid; "
        f"target_stride={target_stride}, remaining targets={remaining.size}/{targets.size}, "
        f"workers={workers}"
    )

    completed_since_checkpoint = 0
    started = time.perf_counter()
    if remaining.size:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_worker,
            initargs=(
                frequencies_hz,
                beta1_s_per_m,
                symbol_rate_baud,
                args.n_azimuth_bins,
                args.n_elevation_bins,
                args.n_alignment_bins,
                args.a_chunk_size,
            ),
        ) as pool:
            futures = {pool.submit(_worker_census, int(t)): int(t) for t in remaining}
            for future in as_completed(futures):
                (
                    target_idx,
                    target_angular_counts,
                    target_alignment_counts,
                    target_tuple_count,
                    target_undefined_count,
                    target_direction_sum,
                    target_direction_outer_sum,
                    target_wall_time_s,
                ) = future.result()
                angular_counts += target_angular_counts
                alignment_counts += target_alignment_counts
                tuple_count += target_tuple_count
                undefined_direction_count += target_undefined_count
                direction_sum += target_direction_sum
                direction_outer_sum += target_direction_outer_sum
                target_tuple_counts[target_idx] = target_tuple_count
                target_wall_times_s[target_idx] = target_wall_time_s
                done[target_idx] = True
                completed_since_checkpoint += 1
                if completed_since_checkpoint >= args.checkpoint_every:
                    _save_results(
                        checkpoint_path,
                        frequencies_hz=frequencies_hz,
                        symbol_rate_baud=symbol_rate_baud,
                        fiber_length_m=fiber_length_m,
                        target_indices_in_full_grid=targets,
                        done=done,
                        angular_counts=angular_counts,
                        alignment_counts=alignment_counts,
                        tuple_count=tuple_count,
                        undefined_direction_count=undefined_direction_count,
                        direction_sum=direction_sum,
                        direction_outer_sum=direction_outer_sum,
                        azimuth_edges_rad=azimuth_edges_rad,
                        elevation_edges_rad=elevation_edges_rad,
                        alignment_cosine_edges=alignment_cosine_edges,
                        target_tuple_counts=target_tuple_counts,
                        target_wall_times_s=target_wall_times_s,
                        target_grid_stride=target_stride,
                    )
                    completed_since_checkpoint = 0
                    lg.info(
                        f"checkpoint: {int(np.sum(done[targets]))}/{targets.size} targets, "
                        f"{tuple_count:,} tuples"
                    )

    _save_results(
        checkpoint_path,
        frequencies_hz=frequencies_hz,
        symbol_rate_baud=symbol_rate_baud,
        fiber_length_m=fiber_length_m,
        target_indices_in_full_grid=targets,
        done=done,
        angular_counts=angular_counts,
        alignment_counts=alignment_counts,
        tuple_count=tuple_count,
        undefined_direction_count=undefined_direction_count,
        direction_sum=direction_sum,
        direction_outer_sum=direction_outer_sum,
        azimuth_edges_rad=azimuth_edges_rad,
        elevation_edges_rad=elevation_edges_rad,
        alignment_cosine_edges=alignment_cosine_edges,
        target_tuple_counts=target_tuple_counts,
        target_wall_times_s=target_wall_times_s,
        target_grid_stride=target_stride,
    )
    expected_probability = _uniform_angular_probabilities(
        azimuth_edges_rad, elevation_edges_rad
    )
    statistics = _uniformity_statistics(
        angular_counts,
        expected_probability,
        direction_sum,
        direction_outer_sum,
    )
    statistics.update({
        "schema_version": SCHEMA_VERSION,
        "units": {
            "angles": "rad",
            "counts": "1",
            "direction_moments": "1",
            "fiber_length_m": "m",
            "symbol_rate_baud": "baud",
        },
        "axes": {
            "angular_tuple_counts": ["elevation_bin", "azimuth_bin"],
            "alignment_tuple_counts": ["alignment_cosine_bin"],
        },
        "phase_model": "linear_gradient_direction",
        "power_basis": "single_pol",
        "reference_plane": "launch_ref",
        "tuple_ordering": "ordered_ab",
        "frequency_grid_hash": hashlib.sha256(
            np.ascontiguousarray(frequencies_hz, dtype="<f8").tobytes()
        ).hexdigest(),
        "symbol_rate_baud": symbol_rate_baud,
        "fiber_length_m": fiber_length_m,
        "gamma_model": "not_applied_direction_only",
        "code_or_model_version": CODE_OR_MODEL_VERSION,
        "strict_fwm_tuple_count": tuple_count,
        "undefined_direction_tuple_count": undefined_direction_count,
        "completed_target_count": int(np.sum(done[targets])),
        "requested_target_count": int(targets.size),
        "target_grid_stride": target_stride,
        "interferer_grid_stride": 1,
        "wall_time_s_this_invocation": time.perf_counter() - started,
    })
    summary_path = args.out_dir / "gradient_direction_census_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(statistics, indent=2, sort_keys=True) + "\n")
    plot_paths = [args.out_dir / "gradient_direction_census.png"]
    canonical_media_dir = REPO_ROOT / "media" / "lorenzi-fast"
    if args.out_dir.resolve() == canonical_media_dir.resolve():
        docs_static_dir = REPO_ROOT / "docs" / "source" / "_static" / "lorenzi-fast"
        if docs_static_dir.exists():
            plot_paths.append(docs_static_dir / "gradient_direction_census.png")
    for plot_path in plot_paths:
        _plot_results(
            plot_path,
            angular_counts,
            alignment_counts,
            azimuth_edges_rad,
            elevation_edges_rad,
            alignment_cosine_edges,
        )
    lg.success(
        f"census complete: {tuple_count:,} tuples; "
        f"TV distance from uniform={statistics['total_variation_distance']:.4f}; "
        f"outputs in {args.out_dir}"
    )


if __name__ == "__main__":
    main()
