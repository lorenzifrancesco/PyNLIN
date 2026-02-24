"""Benchmark TD-NLIN runtime from analysis-level entry points.

This script intentionally does not modify or instrument ``src/pynlin`` internals.
It measures end-to-end call durations around already-defined analysis functions.

Timed functionalities
1. precompute_correction_factors_s:
   Wall-time of ``collision_coeffs_system(...)`` as called by
   ``analysis.system_nlin._compute_mmf_td_nlin_with_timing``.
   This includes either:
   - cache loading (when recompute=False), or
   - full TD collision-coefficient recomputation (when recompute=True).
2. noise_sum_interacting_s:
   Wall-time of ``total_nlin(..., use_x_mode=True)``.
3. noise_sum_noninteracting_s:
   Wall-time of ``total_nlin(..., use_x_mode=False)``.
4. end_to_end_interacting_s:
   ``precompute_correction_factors_s + noise_sum_interacting_s``.
5. end_to_end_with_noninteracting_s:
   ``precompute_correction_factors_s + noise_sum_interacting_s + noise_sum_noninteracting_s``.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger as lg

try:
    # Works when invoked as a module: `python -m analysis.benchmark`
    from analysis.system_nlin import (
        _compute_mmf_td_nlin_with_timing,
        plot_case_study_noise,
    )
except ModuleNotFoundError:
    # Works when invoked as a script: `python analysis/benchmark.py`
    from system_nlin import _compute_mmf_td_nlin_with_timing, plot_case_study_noise
from pynlin.log_init import init_logging
from pynlin.system import System

init_logging()

TIMING_KEYS = [
    "precompute_correction_factors_s",
    "noise_sum_interacting_s",
    "noise_sum_noninteracting_s",
    "end_to_end_interacting_s",
    "end_to_end_with_noninteracting_s",
]

TIMING_DESCRIPTIONS = {
    "precompute_correction_factors_s": (
        "End-to-end time spent in collision-coefficient generation/loading "
        "(collision_coeffs_system)."
    ),
    "noise_sum_interacting_s": (
        "End-to-end time spent in total_nlin with cross-mode terms enabled "
        "(use_x_mode=True)."
    ),
    "noise_sum_noninteracting_s": (
        "End-to-end time spent in total_nlin with cross-mode terms disabled "
        "(use_x_mode=False)."
    ),
    "end_to_end_interacting_s": (
        "Combined time: precompute_correction_factors_s + noise_sum_interacting_s."
    ),
    "end_to_end_with_noninteracting_s": (
        "Combined time: precompute_correction_factors_s + noise_sum_interacting_s "
        "+ noise_sum_noninteracting_s."
    ),
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
    pattern = f"raman_correction_grid_{pulse_name}_*.npy"
    removed = 0
    for path in results_dir.glob(pattern):
        path.unlink(missing_ok=True)
        removed += 1
    return removed


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
        _, nlin_interacting, nlin_noninteracting, timings = _compute_mmf_td_nlin_with_timing(
            system,
            ipulse=ipulse,
            recompute_collisions=recompute,
        )
        run_total_wall_s = time.perf_counter() - run_start

        # Keep a simple deterministic numeric signature for sanity checks.
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
            "Run {} [{}] summary: precompute={:.3f}s, sum(interacting)={:.3f}s, "
            "sum(noninteracting)={:.3f}s, total={:.3f}s",
            idx + 1,
            phase,
            row["precompute_correction_factors_s"],
            row["noise_sum_interacting_s"],
            row["noise_sum_noninteracting_s"],
            row["end_to_end_with_noninteracting_s"],
        )

    stats: dict[str, dict[str, float]] = {}
    for key in TIMING_KEYS + ["run_total_wall_s"]:
        stats[key] = _summarize([float(r[key]) for r in measured_rows])

    return measured_rows, stats


def _print_timing_definitions() -> None:
    print("Timed functionalities:")
    for key in TIMING_KEYS:
        print(f"- {key}: {TIMING_DESCRIPTIONS[key]}")


def _print_stats_table(stats: dict[str, dict[str, float]]) -> None:
    display_rows = [
        ("precompute_correction_factors_s", "Precompute correction factors"),
        ("noise_sum_interacting_s", "Noise sums (interacting)"),
        ("noise_sum_noninteracting_s", "Noise sums (noninteracting)"),
        ("end_to_end_interacting_s", "End-to-end (interacting)"),
        ("end_to_end_with_noninteracting_s", "End-to-end (+ noninteracting)"),
        ("run_total_wall_s", "Total wall time (outer run)"),
    ]

    metric_w = max(len("Metric"), max(len(label) for _, label in display_rows))
    num_w = 14

    print("\nBenchmark statistics (measured runs only, seconds):")
    header = (
        f"{'Metric':<{metric_w}} "
        f"{'Mean':>{num_w}} "
        f"{'Std':>{num_w}} "
        f"{'Min':>{num_w}} "
        f"{'Max':>{num_w}}"
    )
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)

    for key, label in display_rows:
        s = stats[key]
        print(
            f"{label:<{metric_w}} "
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
            "(results/raman_correction_grid_<pulse>_*.npy): "
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

    _print_stats_table(stats)
    _write_csv(args.csv_out, rows)

    summary_payload = {
        "system": str(args.system),
        "runs": int(args.runs),
        "warmup": int(args.warmup),
        "ipulse": int(args.ipulse),
        "collision_mode": str(args.collision_mode),
        "lookup_cache_mode": str(args.lookup_cache_mode),
        "timing_definitions": TIMING_DESCRIPTIONS,
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
