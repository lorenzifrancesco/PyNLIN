"""Benchmark TD-NLIN runtime from analysis-level entry points.

This script reports:
1. the legacy coarse timings used previously, and
2. a more explicit S2/S3 split aligned with the algorithm description.

Legacy/coarse buckets are retained for reference:
- ``precompute_correction_factors_s`` wraps the full collision-coefficient path
  (cache load or full recompute).
- ``noise_sum_*`` wraps the final ``total_nlin(...)`` reductions.

Detailed buckets expose the main reusable-precompute (S2) and pairwise-apply
and reduction (S3) substeps.
"""

from __future__ import annotations

import argparse
import csv
import itertools as it
import json
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger as lg

try:
    # Works when invoked as a module: `python -m analysis.benchmark`
    from analysis.system_nlin import plot_case_study_noise
except ModuleNotFoundError:
    # Works when invoked as a script: `python analysis/benchmark.py`
    from system_nlin import plot_case_study_noise

import pynlin.wdm
from pynlin.fiber import MMFiber
from pynlin.fiber_data.load_fiber_values import load_group_delay
from pynlin.log_init import init_logging
from pynlin.nlin.cache_names import (
    s2b_lo_extrema_path,
    s3_pair_nlin_kernel_path,
)
from pynlin.nlin import nlin_estimator as td_estimator
from pynlin.nlin.nlin_estimation.ideal_fits import ideal_fit_coefficients
from pynlin.nlin.nlin_estimation.lo_correction import build_lookup_integral_table_with_raman
from pynlin.nlin.nlin_estimation.raman_integrals import load_fB, load_raman_integral_extremes
from pynlin.nlin.nlin_estimator import total_nlin
from pynlin.system import System

init_logging()

BENCHMARK_M_LO_TRUNCATION = 3

LEGACY_TIMING_KEYS = [
    "precompute_correction_factors_s",
    "noise_sum_interacting_s",
    "noise_sum_noninteracting_s",
    "end_to_end_interacting_s",
    "end_to_end_with_noninteracting_s",
]

DETAILED_TIMING_KEYS = [
    "collision_coeffs_cache_load_s",
    "s2_beta_dispersion_setup_s",
    "s2_lookup_table_raman_gvd_s",
    "s2_load_fB_profiles_s",
    "s2_raman_integral_extremes_s",
    "s2_ideal_fit_params_s",
    "s2_worker_input_staging_s",
    "s2_total_prepare_context_s",
    "s3_pairwise_fit_eval_s",
    "s3_collision_coeffs_save_s",
    "s3_total_interacting_s",
    "s3_total_with_noninteracting_s",
]

TIMING_KEYS = LEGACY_TIMING_KEYS + DETAILED_TIMING_KEYS

TIMING_METADATA = {
    "precompute_correction_factors_s": {
        "label": "Legacy precompute",
        "functions": "collision_coeffs_system",
        "description": "End-to-end time spent in collision-coefficient generation/loading.",
    },
    "noise_sum_interacting_s": {
        "label": "Legacy reduce (interacting)",
        "functions": "total_nlin(use_x_mode=True)",
        "description": "End-to-end time spent in total_nlin with cross-mode terms enabled.",
    },
    "noise_sum_noninteracting_s": {
        "label": "Legacy reduce (noninteracting)",
        "functions": "total_nlin(use_x_mode=False)",
        "description": "End-to-end time spent in total_nlin with cross-mode terms disabled.",
    },
    "end_to_end_interacting_s": {
        "label": "Legacy end-to-end (interacting)",
        "functions": "collision_coeffs_system + total_nlin(use_x_mode=True)",
        "description": "Combined time: precompute_correction_factors_s + noise_sum_interacting_s.",
    },
    "end_to_end_with_noninteracting_s": {
        "label": "Legacy end-to-end (+ noninteracting)",
        "functions": "collision_coeffs_system + total_nlin(True/False)",
        "description": (
            "Combined time: precompute_correction_factors_s + noise_sum_interacting_s "
            "+ noise_sum_noninteracting_s."
        ),
    },
    "collision_coeffs_cache_load_s": {
        "label": "Cache load",
        "functions": "np.load(s3_pair_nlin_kernel_*.npy)",
        "description": (
            "Time spent loading precomputed collision coefficients when recomputation is skipped."
        ),
    },
    "s2_beta_dispersion_setup_s": {
        "label": "S2 beta/dispersion setup",
        "functions": "load_group_delay + WDM.frequency_grid + MMFiber.evaluate_beta1/beta2",
        "description": "Build beta1/beta2 grids for all mode-channel combinations.",
    },
    "s2_lookup_table_raman_gvd_s": {
        "label": "S2 Raman/GVD lookup",
        "functions": f"build_lookup_integral_table_with_raman(m_lo_truncation={BENCHMARK_M_LO_TRUNCATION})",
        "description": (
            "Build or load the Raman-inclusive low-DGD lookup tables "
            f"using m_lo = 0..{BENCHMARK_M_LO_TRUNCATION}."
        ),
    },
    "s2_load_fB_profiles_s": {
        "label": "S2 Raman profiles",
        "functions": "load_fB",
        "description": "Load normalized Raman power profiles fB(z).",
    },
    "s2_raman_integral_extremes_s": {
        "label": "S2 Raman extremes",
        "functions": "load_raman_integral_extremes",
        "description": "Load LO/HI Raman integral extremes used for profile interpolation.",
    },
    "s2_ideal_fit_params_s": {
        "label": "S2 ideal fit params",
        "functions": "ideal_fit_coefficients",
        "description": "Load and fit the ideal dispersionless baseline softplus parameters.",
    },
    "s2_worker_input_staging_s": {
        "label": "S2 worker staging",
        "functions": "np.save(beta1,beta2,fB) + itertools.product",
        "description": "Stage shared arrays and task lists for the pairwise worker pool.",
    },
    "s2_total_prepare_context_s": {
        "label": "S2 total",
        "functions": "S2 subtotal",
        "description": (
            "Aggregate of reusable system-specific preparation before pairwise application."
        ),
    },
    "s3_pairwise_fit_eval_s": {
        "label": "S3 pairwise fit/apply",
        "functions": "ProcessPoolExecutor(init=_init_worker, fn=work_A)",
        "description": (
            "Apply corrections per channel pair and build the collision-coefficient tensor."
        ),
    },
    "s3_collision_coeffs_save_s": {
        "label": "S3 cache save",
        "functions": "np.save(collision_coeffs)",
        "description": "Persist the collision-coefficient tensor for later cached reuse.",
    },
    "s3_total_interacting_s": {
        "label": "S3 total (interacting)",
        "functions": "work_A + np.save + total_nlin(use_x_mode=True)",
        "description": "Pairwise corrected-fit evaluation plus interacting-mode reduction.",
    },
    "s3_total_with_noninteracting_s": {
        "label": "S3 total (+ noninteracting)",
        "functions": "work_A + np.save + total_nlin(True/False)",
        "description": (
            "Pairwise corrected-fit evaluation plus both interacting and noninteracting reductions."
        ),
    },
}


def _ensure_legacy_fb_solution(auto_prepare: bool) -> Path:
    """Ensure the legacy TD Raman profile file exists where core expects it.

    The legacy path used by ``src/pynlin/nlin/nlin_estimation/raman_integrals.py``
    is hardcoded to ``results/ct_solution-5_gain_0.0.npy``.
    """
    target = Path("results/ct_solution-5_gain_0.0.npy")
    if target.exists():
        return target

    fallback = Path("results/old/ct_solution-5_gain_0.0.npy")
    if auto_prepare and fallback.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(fallback, target)
        lg.warning(
            "Prepared missing legacy fB source by copying {} -> {}",
            fallback,
            target,
        )
        return target

    msg = (
        "Missing required legacy Raman profile file: "
        f"{target}. "
        "The TD legacy path hardcodes this filename for load_fB(). "
    )
    if fallback.exists():
        msg += (
            f"A compatible fallback exists at {fallback}. "
            f"Either copy it manually (`cp {fallback} {target}`) "
            "or rerun benchmark with `--auto-prepare-fb`."
        )
    else:
        msg += (
            "No fallback file found in results/old. "
            "Generate that file with the legacy Raman optimization flow first."
        )
    raise FileNotFoundError(msg)


def _summarize(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {"mean_s": np.nan, "std_s": np.nan, "min_s": np.nan, "max_s": np.nan}
    return {
        "mean_s": float(np.mean(arr)),
        "std_s": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        "min_s": float(np.min(arr)),
        "max_s": float(np.max(arr)),
    }


def _collision_recompute_for_iteration(mode: str, idx_zero_based: int) -> bool:
    if mode == "cached":
        return False
    if mode == "recompute_each":
        return True
    if mode == "recompute_first":
        return idx_zero_based == 0
    raise ValueError(f"Unknown collision_mode: {mode}")


def _lookup_drop_for_iteration(mode: str, idx_zero_based: int) -> bool:
    if mode == "keep":
        return False
    if mode == "drop_each":
        return True
    if mode == "drop_first":
        return idx_zero_based == 0
    raise ValueError(f"Unknown lookup_cache_mode: {mode}")


def _drop_lookup_cache_files(ipulse: int) -> int:
    pulse_name = "gaussian" if int(ipulse) == 0 else "nyquist"
    results_dir = Path("results")
    pattern = s2b_lo_extrema_path(
        pulse_shape=pulse_name,
        m_lo_truncation=BENCHMARK_M_LO_TRUNCATION,
        fiber_length=0.0,
        lld_max=0.0,
    ).name.replace("mtrunc3_L0.0km_lldmax0.00.npz", "*.npz")
    removed = 0
    for path in results_dir.glob(pattern):
        path.unlink(missing_ok=True)
        removed += 1
    return removed


def _collision_coeffs_filename(cf: System, ipulse: int) -> Path:
    fiber_type = "smf" if cf.n_modes == 1 else "mmf"
    return s3_pair_nlin_kernel_path(
        ipulse=ipulse,
        fiber_type=fiber_type,
        br_hz=float(cf.baud_rate),
        n_ch=int(cf.n_channels),
        spacing_hz=getattr(cf, "channel_spacing", None),
    )


def _empty_detailed_timing() -> dict[str, float]:
    return {key: 0.0 for key in DETAILED_TIMING_KEYS}


def _compute_mmf_td_nlin_with_detailed_timing(
    cf_mmf: System,
    ipulse: int,
    recompute_collisions: bool,
):
    """Compute MMF TD NLIN with both coarse legacy and detailed S2/S3 timings."""
    timings = _empty_detailed_timing()
    collision_path = _collision_coeffs_filename(cf_mmf, ipulse)
    cache_hit = collision_path.exists() and not recompute_collisions

    t_pre_start = time.perf_counter()
    if cache_hit:
        t0 = time.perf_counter()
        ccfs_mmf = np.load(collision_path)
        timings["collision_coeffs_cache_load_s"] = time.perf_counter() - t0
    else:
        lg.info("Computing collision coefficients from scratch via detailed benchmark path.")
        n_samples_numeric_n = td_estimator._get_n_samples_numeric_n(cf_mmf)

        t0 = time.perf_counter()
        oi_fit = np.load("results/oi_fit.npy")
        beta1_params = load_group_delay()
        wdm = pynlin.wdm.WDM(
            spacing=cf_mmf.channel_spacing,
            num_channels=cf_mmf.n_channels,
            center_frequency=cf_mmf.center_frequency,
        )
        freqs = wdm.frequency_grid()
        fiber = MMFiber(
            effective_area=cf_mmf.effective_area,
            overlap_integrals=oi_fit,
            group_delay=beta1_params,
            length=cf_mmf.fiber_length,
        )
        beta1 = np.zeros((cf_mmf.n_modes, len(freqs)))
        beta2 = np.zeros((cf_mmf.n_modes, len(freqs)))
        for i in range(cf_mmf.n_modes):
            beta1[i, :] = fiber.group_delay.evaluate_beta1(i, freqs)
            beta2[i, :] = fiber.group_delay.evaluate_beta2(i, freqs)
        timings["s2_beta_dispersion_setup_s"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        max_lld = td_estimator._max_lld_from_beta2(cf_mmf, beta2)
        raman_gvd_correction_max, raman_gvd_correction_min = build_lookup_integral_table_with_raman(
            cf_mmf,
            m_lo_truncation=BENCHMARK_M_LO_TRUNCATION,
            ipulse=ipulse,
            max_lld=max_lld,
        )
        timings["s2_lookup_table_raman_gvd_s"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        fB, _, _, _, _ = load_fB(cf_mmf)
        timings["s2_load_fB_profiles_s"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        raman_extremes = load_raman_integral_extremes(cf_mmf)
        timings["s2_raman_integral_extremes_s"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        ps_ideal = ideal_fit_coefficients(
            0.0,
            0.0,
            ipulse=ipulse,
            fiber_length=float(cf_mmf.fiber_length),
            baud_rate=float(cf_mmf.baud_rate),
            n_samples_numeric_n=n_samples_numeric_n,
        )
        timings["s2_ideal_fit_params_s"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        beta1_path = "/tmp/beta1_grid.npy"
        beta2_path = "/tmp/beta2_grid.npy"
        fB_path = "/tmp/fB.npy"
        np.save(beta1_path, beta1)
        np.save(beta2_path, beta2)
        np.save(fB_path, fB)
        a_tasks = list(it.product(range(cf_mmf.n_modes), range(len(freqs))))
        n_workers = os.cpu_count() or 1
        timings["s2_worker_input_staging_s"] = time.perf_counter() - t0

        timings["s2_total_prepare_context_s"] = (
            timings["s2_beta_dispersion_setup_s"]
            + timings["s2_lookup_table_raman_gvd_s"]
            + timings["s2_load_fB_profiles_s"]
            + timings["s2_raman_integral_extremes_s"]
            + timings["s2_ideal_fit_params_s"]
            + timings["s2_worker_input_staging_s"]
        )

        collision_coeffs = np.zeros((cf_mmf.n_modes, len(freqs), cf_mmf.n_modes, len(freqs)))

        t0 = time.perf_counter()
        with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=td_estimator._init_worker,
            initargs=(
                beta1_path,
                beta2_path,
                fB_path,
                cf_mmf.n_modes,
                len(freqs),
                cf_mmf,
                ipulse,
                raman_gvd_correction_min,
                raman_gvd_correction_max,
                n_workers,
                raman_extremes,
                ps_ideal,
            ),
        ) as ex:
            futures = [ex.submit(td_estimator.work_A, a) for a in a_tasks]
            for fut in as_completed(futures):
                mA, nuA, block, elapsed = fut.result()
                collision_coeffs[mA, nuA, :, :] = block
                lg.trace(
                    "Finished NLIN for A(m={},nu={:>5}) in {:>6.2f} s",
                    mA,
                    nuA,
                    elapsed,
                )
        timings["s3_pairwise_fit_eval_s"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        np.save(collision_path, collision_coeffs)
        timings["s3_collision_coeffs_save_s"] = time.perf_counter() - t0
        ccfs_mmf = collision_coeffs

    t_pre = time.perf_counter() - t_pre_start

    t0 = time.perf_counter()
    nlin_mmf = total_nlin(
        cf_mmf,
        ccfs_mmf,
        use_kappa=True,
        use_x_mode=True,
    )
    t_sum_interacting = time.perf_counter() - t0

    t0 = time.perf_counter()
    nlin_mmf_noninteracting = total_nlin(
        cf_mmf,
        ccfs_mmf,
        use_kappa=True,
        use_x_mode=False,
    )
    t_sum_noninteracting = time.perf_counter() - t0

    timings["s3_total_interacting_s"] = (
        timings["s3_pairwise_fit_eval_s"]
        + timings["s3_collision_coeffs_save_s"]
        + t_sum_interacting
    )
    timings["s3_total_with_noninteracting_s"] = (
        timings["s3_pairwise_fit_eval_s"]
        + timings["s3_collision_coeffs_save_s"]
        + t_sum_interacting
        + t_sum_noninteracting
    )

    timings.update(
        {
            "recompute_collisions": bool(recompute_collisions),
            "collision_coeffs_cache_hit": bool(cache_hit),
            "collision_coeffs_path": str(collision_path),
            "precompute_correction_factors_s": t_pre,
            "noise_sum_interacting_s": t_sum_interacting,
            "noise_sum_noninteracting_s": t_sum_noninteracting,
            "end_to_end_interacting_s": t_pre + t_sum_interacting,
            "end_to_end_with_noninteracting_s": t_pre + t_sum_interacting + t_sum_noninteracting,
        }
    )
    return ccfs_mmf, nlin_mmf, nlin_mmf_noninteracting, timings


def run_td_benchmark(
    system_path: Path,
    runs: int,
    warmup: int,
    ipulse: int,
    collision_mode: str,
    lookup_cache_mode: str,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, float]]]:
    if runs <= 0:
        raise ValueError(f"runs must be > 0, got {runs}")
    if warmup < 0:
        raise ValueError(f"warmup must be >= 0, got {warmup}")

    lg.info("Loading system from {}", system_path)
    system = System.from_toml(system_path)

    total_iters = warmup + runs
    measured_rows: list[dict[str, Any]] = []

    for idx in range(total_iters):
        is_warmup = idx < warmup
        phase = "warmup" if is_warmup else "measured"
        recompute_requested = _collision_recompute_for_iteration(collision_mode, idx)
        drop_lookup = _lookup_drop_for_iteration(lookup_cache_mode, idx)

        if drop_lookup:
            removed = _drop_lookup_cache_files(ipulse=ipulse)
            lg.info(
                "Lookup-table cache purge for run {} [{}]: removed {} files.",
                idx + 1,
                phase,
                removed,
            )

        # If we drop lookup caches but keep collision coefficients cached, the
        # lookup table would not be rebuilt (collision path returns early).
        # Force recompute so the benchmark reflects actual lookup regeneration.
        recompute = recompute_requested or drop_lookup

        lg.info(
            "Benchmark run {}/{} [{}] with collision_mode={} lookup_cache_mode={} "
            "(recompute_requested={}, recompute_effective={})",
            idx + 1,
            total_iters,
            phase,
            collision_mode,
            lookup_cache_mode,
            recompute_requested,
            recompute,
        )

        run_start = time.perf_counter()
        _, nlin_interacting, nlin_noninteracting, timings = _compute_mmf_td_nlin_with_detailed_timing(
            system,
            ipulse=ipulse,
            recompute_collisions=recompute,
        )
        run_total_wall_s = time.perf_counter() - run_start

        checksum = float(np.sum(nlin_interacting) + np.sum(nlin_noninteracting))

        row = {
            "iteration": idx + 1,
            "phase": phase,
            "recompute_collisions_requested": recompute_requested,
            "lookup_cache_dropped": drop_lookup,
            "recompute_collisions_effective": recompute,
            "run_total_wall_s": run_total_wall_s,
            "checksum": checksum,
            **timings,
        }

        if not is_warmup:
            measured_rows.append(row)

        lg.info(
            "Run {} [{}] summary: precompute={:.3f}s, S2={:.3f}s, S3(pairwise)={:.3f}s, "
            "sum(interacting)={:.3f}s, total={:.3f}s",
            idx + 1,
            phase,
            row["precompute_correction_factors_s"],
            row["s2_total_prepare_context_s"],
            row["s3_pairwise_fit_eval_s"],
            row["noise_sum_interacting_s"],
            row["end_to_end_with_noninteracting_s"],
        )

    stats: dict[str, dict[str, float]] = {}
    for key in TIMING_KEYS + ["run_total_wall_s"]:
        stats[key] = _summarize([float(r[key]) for r in measured_rows])

    return measured_rows, stats


def _print_timing_definitions() -> None:
    print("Timed functionalities:")
    for key in TIMING_KEYS:
        meta = TIMING_METADATA[key]
        print(
            f"- {key}: {meta['label']} | {meta['functions']} | {meta['description']}"
        )


def _print_stats_table(
    stats: dict[str, dict[str, float]],
    rows: list[tuple[str, str, str]],
    title: str,
) -> None:
    metric_w = max(len("Metric"), max(len(label) for _, label, _ in rows))
    func_w = max(len("Functions"), max(len(functions) for _, _, functions in rows))
    num_w = 12

    print(f"\n{title} (measured runs only, seconds):")
    header = (
        f"{'Metric':<{metric_w}} "
        f"{'Functions':<{func_w}} "
        f"{'Mean':>{num_w}} "
        f"{'Std':>{num_w}} "
        f"{'Min':>{num_w}} "
        f"{'Max':>{num_w}}"
    )
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)

    for key, label, functions in rows:
        s = stats[key]
        print(
            f"{label:<{metric_w}} "
            f"{functions:<{func_w}} "
            f"{s['mean_s']:>{num_w}.6f} "
            f"{s['std_s']:>{num_w}.6f} "
            f"{s['min_s']:>{num_w}.6f} "
            f"{s['max_s']:>{num_w}.6f}"
        )
    print(sep)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("No rows to write.")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    lg.info("Saved per-run benchmark rows to {}", path)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    lg.info("Saved benchmark summary to {}", path)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Benchmark MMF TD-NLIN execution time from analysis-level calls."
    )
    ap.add_argument(
        "--system",
        type=Path,
        default=Path("input/mmf_struct.toml"),
        help="Path to MMF system TOML.",
    )
    ap.add_argument("--runs", type=int, default=5, help="Number of measured runs.")
    ap.add_argument("--warmup", type=int, default=1, help="Number of warmup runs.")
    ap.add_argument("--ipulse", type=int, default=1, choices=[0, 1], help="Pulse type index.")
    ap.add_argument(
        "--collision-mode",
        type=str,
        default="cached",
        choices=["cached", "recompute_first", "recompute_each"],
        help=(
            "How to handle collision coefficients during repeated runs: "
            "cached (always load if present), recompute_first (first iteration only), "
            "recompute_each (every iteration)."
        ),
    )
    ap.add_argument(
        "--lookup-cache-mode",
        type=str,
        default="keep",
        choices=["keep", "drop_first", "drop_each"],
        help=(
            "How to handle Raman/GVD lookup-table cache files "
            "(results/s2b_lo_extrema_<pulse>_*.npz): "
            "keep (never delete), drop_first (delete before first iteration), "
            "drop_each (delete before every iteration). "
            "When deletion is requested, effective collision recompute is forced "
            "so lookup tables are rebuilt."
        ),
    )
    ap.add_argument(
        "--csv-out",
        type=Path,
        default=Path("results/benchmark_td_runs.csv"),
        help="Output CSV path for per-run timing rows.",
    )
    ap.add_argument(
        "--json-out",
        type=Path,
        default=Path("results/benchmark_td_summary.json"),
        help="Output JSON path for aggregate statistics and metadata.",
    )
    ap.add_argument(
        "--auto-prepare-fb",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Auto-copy missing legacy fB source from "
            "results/old/ct_solution-5_gain_0.0.npy to results/. "
            "Disable with --no-auto-prepare-fb."
        ),
    )
    ap.add_argument(
        "--plot-realistic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Generate the realistic NLIN figure after benchmark execution. "
            "Disable with --no-plot-realistic."
        ),
    )
    ap.add_argument(
        "--plot-name",
        type=str,
        default="realistic_benchmark",
        help=(
            "Suffix used by plot_case_study_noise when saving "
            "media/nlin<name>.pdf."
        ),
    )
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    fb_path = _ensure_legacy_fb_solution(auto_prepare=bool(args.auto_prepare_fb))
    lg.info("Using legacy fB source at {}", fb_path)
    _print_timing_definitions()

    rows, stats = run_td_benchmark(
        system_path=args.system,
        runs=int(args.runs),
        warmup=int(args.warmup),
        ipulse=int(args.ipulse),
        collision_mode=str(args.collision_mode),
        lookup_cache_mode=str(args.lookup_cache_mode),
    )

    legacy_rows = [
        (
            "precompute_correction_factors_s",
            TIMING_METADATA["precompute_correction_factors_s"]["label"],
            TIMING_METADATA["precompute_correction_factors_s"]["functions"],
        ),
        (
            "noise_sum_interacting_s",
            TIMING_METADATA["noise_sum_interacting_s"]["label"],
            TIMING_METADATA["noise_sum_interacting_s"]["functions"],
        ),
        (
            "noise_sum_noninteracting_s",
            TIMING_METADATA["noise_sum_noninteracting_s"]["label"],
            TIMING_METADATA["noise_sum_noninteracting_s"]["functions"],
        ),
        (
            "end_to_end_interacting_s",
            TIMING_METADATA["end_to_end_interacting_s"]["label"],
            TIMING_METADATA["end_to_end_interacting_s"]["functions"],
        ),
        (
            "end_to_end_with_noninteracting_s",
            TIMING_METADATA["end_to_end_with_noninteracting_s"]["label"],
            TIMING_METADATA["end_to_end_with_noninteracting_s"]["functions"],
        ),
        ("run_total_wall_s", "Outer wall time", "benchmark.run_td_benchmark"),
    ]
    detailed_rows = [
        (
            "collision_coeffs_cache_load_s",
            TIMING_METADATA["collision_coeffs_cache_load_s"]["label"],
            TIMING_METADATA["collision_coeffs_cache_load_s"]["functions"],
        ),
        (
            "s2_beta_dispersion_setup_s",
            TIMING_METADATA["s2_beta_dispersion_setup_s"]["label"],
            TIMING_METADATA["s2_beta_dispersion_setup_s"]["functions"],
        ),
        (
            "s2_lookup_table_raman_gvd_s",
            TIMING_METADATA["s2_lookup_table_raman_gvd_s"]["label"],
            TIMING_METADATA["s2_lookup_table_raman_gvd_s"]["functions"],
        ),
        (
            "s2_load_fB_profiles_s",
            TIMING_METADATA["s2_load_fB_profiles_s"]["label"],
            TIMING_METADATA["s2_load_fB_profiles_s"]["functions"],
        ),
        (
            "s2_raman_integral_extremes_s",
            TIMING_METADATA["s2_raman_integral_extremes_s"]["label"],
            TIMING_METADATA["s2_raman_integral_extremes_s"]["functions"],
        ),
        (
            "s2_ideal_fit_params_s",
            TIMING_METADATA["s2_ideal_fit_params_s"]["label"],
            TIMING_METADATA["s2_ideal_fit_params_s"]["functions"],
        ),
        (
            "s2_worker_input_staging_s",
            TIMING_METADATA["s2_worker_input_staging_s"]["label"],
            TIMING_METADATA["s2_worker_input_staging_s"]["functions"],
        ),
        (
            "s2_total_prepare_context_s",
            TIMING_METADATA["s2_total_prepare_context_s"]["label"],
            TIMING_METADATA["s2_total_prepare_context_s"]["functions"],
        ),
        (
            "s3_pairwise_fit_eval_s",
            TIMING_METADATA["s3_pairwise_fit_eval_s"]["label"],
            TIMING_METADATA["s3_pairwise_fit_eval_s"]["functions"],
        ),
        (
            "s3_collision_coeffs_save_s",
            TIMING_METADATA["s3_collision_coeffs_save_s"]["label"],
            TIMING_METADATA["s3_collision_coeffs_save_s"]["functions"],
        ),
        (
            "noise_sum_interacting_s",
            "S3 reduce (interacting)",
            "total_nlin(use_x_mode=True)",
        ),
        (
            "noise_sum_noninteracting_s",
            "S3 reduce (noninteracting)",
            "total_nlin(use_x_mode=False)",
        ),
        (
            "s3_total_interacting_s",
            TIMING_METADATA["s3_total_interacting_s"]["label"],
            TIMING_METADATA["s3_total_interacting_s"]["functions"],
        ),
        (
            "s3_total_with_noninteracting_s",
            TIMING_METADATA["s3_total_with_noninteracting_s"]["label"],
            TIMING_METADATA["s3_total_with_noninteracting_s"]["functions"],
        ),
    ]

    _print_stats_table(stats, legacy_rows, "Legacy benchmark statistics")
    _print_stats_table(stats, detailed_rows, "Detailed S2/S3 benchmark statistics")
    _write_csv(args.csv_out, rows)

    summary_payload = {
        "system": str(args.system),
        "runs": int(args.runs),
        "warmup": int(args.warmup),
        "ipulse": int(args.ipulse),
        "collision_mode": str(args.collision_mode),
        "lookup_cache_mode": str(args.lookup_cache_mode),
        "timing_metadata": TIMING_METADATA,
        "stats": stats,
    }
    _write_json(args.json_out, summary_payload)

    if bool(args.plot_realistic):
        plot_name = str(args.plot_name)
        lg.info(
            "Generating realistic NLIN plot from benchmark flow as media/nlin{}.pdf",
            plot_name,
        )
        plot_case_study_noise(
            use_dBm_scale=True,
            also_plot_smf=True,
            also_plot_noninteracting=True,
            name=plot_name,
            report_timing=False,
            recompute_collisions=False,
        )


if __name__ == "__main__":
    main()
