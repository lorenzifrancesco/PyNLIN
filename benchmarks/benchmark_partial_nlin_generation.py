#!/usr/bin/env python3
"""Benchmark full regeneration of results/partial_nlin_*_perfect_... files.

This script times the end-to-end call to:
    pynlin.nlin.validation.compute_numeric_nlin(...)

It is intended for before/after performance comparisons when changing
collision-integral kernels (e.g., m_th_time_integral_general).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import tomllib
from pathlib import Path
from statistics import mean, median, stdev
from types import SimpleNamespace

import numpy as np
from loguru import logger as lg


def _configure_logging(level: str) -> None:
    lg.remove()
    lg.add(sys.stderr, level=level.upper())


def _patch_legacy_config_loader() -> None:
    # validation.compute_numeric_nlin currently calls io_utils.load_toml_to_struct
    # with flat legacy TOML files (input/mmf.toml).
    import pynlin.io_utils as cfg

    def _legacy_load_toml_to_struct(path: str | Path):
        with open(path, "rb") as f:
            return SimpleNamespace(**tomllib.load(f))

    cfg.load_toml_to_struct = _legacy_load_toml_to_struct


def _patch_nlin_package_exports() -> None:
    # validation imports symbols from pynlin.nlin package level, while the
    # implementations currently live in the flat module loaded as _nlin_module.
    import pynlin.nlin as nlin_pkg

    if not hasattr(nlin_pkg, "_nlin_module"):
        raise RuntimeError("pynlin.nlin._nlin_module is missing.")
    flat = nlin_pkg._nlin_module
    sys.modules["pynlin.nlin_module"] = flat
    nlin_pkg.X0mm_space_integral = flat.X0mm_space_integral
    nlin_pkg.compute_all_collisions_time_integrals = flat.compute_all_collisions_time_integrals


def _stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {
            "mean_s": float("nan"),
            "median_s": float("nan"),
            "std_s": float("nan"),
            "min_s": float("nan"),
            "max_s": float("nan"),
        }
    return {
        "mean_s": mean(values),
        "median_s": median(values),
        "std_s": stdev(values) if len(values) > 1 else 0.0,
        "min_s": min(values),
        "max_s": max(values),
    }


def _generation_count_for_runs(outfile: Path, total_runs: int, recompute: bool) -> int:
    if total_runs <= 0:
        return 0
    if recompute:
        return total_runs
    return 0 if outfile.exists() else 1


def _estimate_m_integral_counts(ipulse: int) -> dict:
    from pynlin.collisions import get_m_values
    from pynlin.pulses import GaussianPulse, NyquistPulse

    with open("input/mmf.toml", "rb") as f:
        cf = SimpleNamespace(**tomllib.load(f))
    with open("input/numerical_config.toml", "rb") as f:
        nc = SimpleNamespace(**tomllib.load(f))

    nc.dgd1 = td_dgd1 = 0.01 / (cf.fiber_length * cf.baud_rate)
    td_dgd2 = 100.0 / (cf.fiber_length * cf.baud_rate)
    nc.dgd2_n = td_dgd2
    nc.dgd2_g = td_dgd2

    if ipulse == 0:
        n_samples_numeric = int(nc.n_samples_numeric_g)
        dgd2 = float(nc.dgd2_g)
        pulse = GaussianPulse(
            baud_rate=cf.baud_rate,
            num_symbols=1e2,
            samples_per_symbol=2**5,
        )
    else:
        n_samples_numeric = int(nc.n_samples_numeric_n)
        dgd2 = float(nc.dgd2_n)
        pulse = NyquistPulse(
            baud_rate=cf.baud_rate,
            num_symbols=1e3,
            samples_per_symbol=2**5,
            rolloff=0.0,
        )

    collision_margin = int(getattr(cf, "collision_margin", 5))
    dgds_numeric = np.logspace(np.log10(td_dgd1), np.log10(dgd2), n_samples_numeric)
    fiber = SimpleNamespace(length=float(cf.fiber_length))

    m_counts: list[int] = []
    m_ranges: list[dict[str, int]] = []
    for dgd in dgds_numeric:
        m_values = np.asarray(
            get_m_values(
                fiber,
                pulse,
                collision_margin,
                float(dgd),
            )
        )
        m_counts.append(int(m_values.size))
        m_ranges.append(
            {
                "m_min": int(m_values.min()),
                "m_max": int(m_values.max()),
            }
        )

    m_counts_arr = np.asarray(m_counts, dtype=int)
    return {
        "collision_margin": collision_margin,
        "n_dgd_points": int(dgds_numeric.size),
        "dgd_values_s": [float(x) for x in dgds_numeric],
        "m_integrals_per_dgd": [int(x) for x in m_counts],
        "m_ranges_per_dgd": m_ranges,
        "total_m_integrals_per_generation": int(np.sum(m_counts_arr)),
        "min_m_integrals_per_dgd": int(np.min(m_counts_arr)) if m_counts else 0,
        "max_m_integrals_per_dgd": int(np.max(m_counts_arr)) if m_counts else 0,
        "mean_m_integrals_per_dgd": float(np.mean(m_counts_arr)) if m_counts else float("nan"),
    }


def run_benchmark(
    ipulse: int,
    gvda: float,
    gvdb: float,
    repeats: int,
    warmup: int,
    recompute: bool,
    perfect_only: bool,
) -> dict:
    _patch_legacy_config_loader()
    _patch_nlin_package_exports()

    from pynlin.nlin.validation import compute_numeric_nlin

    if ipulse not in (0, 1):
        raise ValueError(f"ipulse must be 0 or 1, got {ipulse}")

    pulse = "gaussian" if ipulse == 0 else "nyquist"
    outfile = Path(f"results/partial_nlin_{pulse}_perfect_{gvda}_{gvdb}.npy")
    total_runs = warmup + repeats
    m_integral_counts = _estimate_m_integral_counts(ipulse=ipulse)
    executed_generations = _generation_count_for_runs(
        outfile=outfile,
        total_runs=total_runs,
        recompute=recompute,
    )

    measured: list[float] = []
    for i in range(total_runs):
        t0 = time.perf_counter()
        compute_numeric_nlin(
            gvda=gvda,
            gvdb=gvdb,
            ipulse=ipulse,
            recompute=recompute,
            perfect_only=perfect_only,
        )
        dt = time.perf_counter() - t0
        if i >= warmup:
            measured.append(dt)

    if not outfile.exists():
        raise FileNotFoundError(f"Expected output file not found: {outfile}")

    arr = np.load(outfile)
    result = {
        "ipulse": ipulse,
        "pulse": pulse,
        "gvda": gvda,
        "gvdb": gvdb,
        "recompute": recompute,
        "perfect_only": perfect_only,
        "repeats": repeats,
        "warmup": warmup,
        "times_s": measured,
        "stats": _stats(measured),
        "outfile": str(outfile),
        "n_points": int(arr.size),
        "checksum_sum": float(np.sum(arr)),
        "checksum_l2": float(np.linalg.norm(arr)),
        "m_integral_counts": m_integral_counts,
        "executed_generations": int(executed_generations),
        "total_m_integrals_executed": int(
            executed_generations * m_integral_counts["total_m_integrals_per_generation"]
        ),
    }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark full partial_nlin_perfect generation."
    )
    parser.add_argument("--ipulse", type=int, choices=[0, 1], default=1, help="0=gaussian, 1=nyquist")
    parser.add_argument("--gvda", type=float, default=0.0, help="GVD A [s^2/m]")
    parser.add_argument("--gvdb", type=float, default=0.0, help="GVD B [s^2/m]")
    parser.add_argument("--repeats", type=int, default=3, help="Measured runs")
    parser.add_argument("--warmup", type=int, default=0, help="Warmup runs (not measured)")
    parser.add_argument("--no-recompute", action="store_true", help="Do not force file regeneration")
    parser.add_argument(
        "--with-extremes",
        action="store_true",
        help="Also compute/save *_min_* and *_max_* datasets (default is perfect-only).",
    )
    parser.add_argument("--log-level", type=str, default="ERROR", help="Loguru level")
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path("results/benchmark_partial_nlin_generation.json"),
        help="Path to write JSON output (default: results/benchmark_partial_nlin_generation.json)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _configure_logging(args.log_level)

    if args.repeats <= 0:
        raise ValueError(f"--repeats must be > 0, got {args.repeats}")
    if args.warmup < 0:
        raise ValueError(f"--warmup must be >= 0, got {args.warmup}")

    result = run_benchmark(
        ipulse=args.ipulse,
        gvda=args.gvda,
        gvdb=args.gvdb,
        repeats=args.repeats,
        warmup=args.warmup,
        recompute=not args.no_recompute,
        perfect_only=not args.with_extremes,
    )

    payload = json.dumps(result, indent=2)
    print(payload)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(payload + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
