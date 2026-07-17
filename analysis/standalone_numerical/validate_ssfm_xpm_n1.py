"""Validate prefactor-free Dar XPM N1 against controlled scalar SSFM."""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
logging.getLogger("fontTools.subset").setLevel(logging.WARNING)
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.methods.ssfm_interface import (
    compute_scalar_ssfm_xpm_n1,
    fit_scalar_ssfm_xpm_sweep,
    prepare_ssfm_runtime_with_template,
)
from pynlin.methods.td.xhkm_mc import estimate_xhkm_sums_mc
from pynlin.system import System
from pynlin.utils import _toml_load


def _section(config_path: Path) -> dict:
    raw = _toml_load(config_path)
    return dict(raw.get("ssfm", {}).get("n1_validation", {}))


def _value(cli_value, config: dict, key: str, default):
    return cli_value if cli_value is not None else config.get(key, default)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("input/studies.toml"))
    parser.add_argument("--template", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--target-index", type=int)
    parser.add_argument("--interferer-shift", type=int)
    parser.add_argument("--cut-power-w", type=float)
    parser.add_argument("--interferer-powers-w", type=float, nargs="+")
    parser.add_argument("--n-trials", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--n-symbols", type=int)
    parser.add_argument("--samples-per-symbol", type=int)
    parser.add_argument("--fiber-length-m", type=float)
    parser.add_argument("--max-step-m", type=float)
    parser.add_argument("--max-nonlinear-phase-deg", type=float)
    parser.add_argument("--mc-samples", type=int)
    parser.add_argument(
        "--symbol-distribution", choices=["qam", "circular_complex_gaussian"]
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = _section(args.config)
    system = System.from_toml(args.config)
    frequencies = np.asarray(system.wdm.frequency_grid(), dtype=float)

    target = int(_value(args.target_index, cfg, "target_index", frequencies.size // 2))
    shift = int(_value(args.interferer_shift, cfg, "interferer_shift", 1))
    interferer = target + shift
    if not (0 <= target < frequencies.size and 0 <= interferer < frequencies.size):
        raise ValueError("target/interferer indices fall outside the WDM grid")
    if shift == 0:
        raise ValueError("interferer_shift must be nonzero")
    baud_rate = float(system.pulse.baud_rate)
    spacing = abs(float(frequencies[interferer] - frequencies[target]))
    if spacing < 2.0 * baud_rate:
        raise ValueError(
            "Controlled N1 validation requires spacing >= 2*baud_rate so the "
            "interferer's cubic spectrum cannot overlap the CUT band"
        )

    cut_power = float(_value(args.cut_power_w, cfg, "cut_power_w", 0.2e-3))
    powers = np.asarray(
        _value(
            args.interferer_powers_w,
            cfg,
            "interferer_powers_w",
            [0.08e-3, 0.12e-3, 0.16e-3],
        ),
        dtype=float,
    )
    if powers.size < 2 or np.any(powers <= 0.0) or np.unique(powers).size != powers.size:
        raise ValueError("At least two distinct positive interferer powers are required")

    n_trials = int(_value(args.n_trials, cfg, "n_trials", 8))
    seed = int(_value(args.seed, cfg, "seed", 1234))
    n_symbols = int(_value(args.n_symbols, cfg, "n_symbols", 2048))
    samples_per_symbol = int(
        _value(args.samples_per_symbol, cfg, "samples_per_symbol", 8)
    )
    fiber_length = float(
        _value(args.fiber_length_m, cfg, "fiber_length_m", system.fiber_length)
    )
    max_step = float(_value(args.max_step_m, cfg, "max_step_m", 500.0))
    max_phase = float(
        _value(args.max_nonlinear_phase_deg, cfg, "max_nonlinear_phase_deg", 0.05)
    )
    mc_samples = int(_value(args.mc_samples, cfg, "mc_samples", 200_000))
    symbol_distribution = str(
        _value(
            args.symbol_distribution,
            cfg,
            "symbol_distribution",
            "circular_complex_gaussian",
        )
    )
    out_dir = Path(
        _value(args.out_dir, cfg, "out_dir", "results/ssfm_xpm_n1")
    ).resolve()
    template = Path(
        _value(
            args.template,
            cfg,
            "template",
            "/home/lorenzi/sw/gnlse-python/input/wdm_nli_config.toml",
        )
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    ssfm_ctx = prepare_ssfm_runtime_with_template(out_dir, template)
    if ssfm_ctx is None:
        raise RuntimeError("The external gnlse-python SSFM runtime is unavailable")

    results = []
    for index, power in enumerate(powers):
        print(f"Running paired SSFM power {index + 1}/{powers.size}: {power:.6g} W")
        results.append(
            compute_scalar_ssfm_xpm_n1(
                ssfm_ctx,
                system=system,
                cut_power_w=cut_power,
                interferer_power_w=float(power),
                channel_idx=target,
                interferer_idx=interferer,
                n_trials=n_trials,
                rng_seed=seed,
                sweep_tag=f"p{index:02d}",
                n_symbols=n_symbols,
                samples_per_symbol=samples_per_symbol,
                max_step_m=max_step,
                max_nonlinear_phase_deg=max_phase,
                fiber_length_m=fiber_length,
                symbol_distribution=symbol_distribution,
            )
        )

    power_squared = powers**2
    fit = fit_scalar_ssfm_xpm_sweep(
        results, powers, cut_power_w=cut_power
    )
    slope = fit.slope_w_inv
    slope_stderr = fit.slope_stderr_w_inv
    gamma = results[0].gamma_w_inv_m
    n1_ssfm = fit.n1_ssfm_m2
    n1_ssfm_stderr = fit.n1_ssfm_stderr_m2
    mean_excess = fit.mean_excess_power_w
    r_squared = fit.r_squared

    _, beta2_grid = system.beta_grids(frequencies)
    beta2_si = float(np.asarray(beta2_grid)[0, target])
    dar = estimate_xhkm_sums_mc(
        beta2=beta2_si * baud_rate**2,
        alpha=0.0,
        length=fiber_length,
        channel_spacing_over_baud=spacing / baud_rate,
        n_samples=mc_samples,
        seed=seed,
    )
    combined_stderr = float(np.hypot(n1_ssfm_stderr, dar.n1_stderr))
    z_score = (
        (n1_ssfm - dar.n1) / combined_stderr if combined_stderr > 0.0 else float("nan")
    )

    payload = {
        "calculation": "controlled_scalar_ssfm_vs_prefactor_free_dar_xpm_n1",
        "config": str(args.config),
        "template": str(template),
        "target_index": target,
        "interferer_index": interferer,
        "target_frequency_hz": float(frequencies[target]),
        "interferer_frequency_hz": float(frequencies[interferer]),
        "spacing_hz": spacing,
        "baud_rate_hz": baud_rate,
        "beta2_s2_per_m": beta2_si,
        "fiber_length_m": fiber_length,
        "gamma_w_inv_m": gamma,
        "cut_power_w": cut_power,
        "interferer_powers_w": powers.tolist(),
        "n_trials": n_trials,
        "seed": seed,
        "n_symbols": n_symbols,
        "samples_per_symbol": samples_per_symbol,
        "max_step_m": max_step,
        "max_nonlinear_phase_deg": max_phase,
        "mc_samples": mc_samples,
        "symbol_distribution": symbol_distribution,
        "mean_excess_power_w": mean_excess.tolist(),
        "mean_excess_power_stderr_w": fit.mean_excess_power_stderr_w.tolist(),
        "fit_slope_w_inv": slope,
        "fit_slope_stderr_w_inv": slope_stderr,
        "fit_intercept_w": fit.intercept_w,
        "fit_r_squared": r_squared,
        "n1_ssfm_m2": n1_ssfm,
        "n1_ssfm_stderr_m2": n1_ssfm_stderr,
        "n1_dar_m2": dar.n1,
        "n1_dar_stderr_m2": dar.n1_stderr,
        "ssfm_over_dar": n1_ssfm / dar.n1,
        "difference_z_score": z_score,
    }
    json_path = out_dir / "ssfm_xpm_n1_validation.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    csv_path = out_dir / "ssfm_xpm_n1_power_sweep.csv"
    point_stderr = fit.mean_excess_power_stderr_w
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["interferer_power_w", "interferer_power_squared_w2", "excess_power_w", "stderr_w"])
        writer.writerows(zip(powers, power_squared, mean_excess, point_stderr))

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.errorbar(power_squared, mean_excess, yerr=point_stderr, fmt="o", label="paired SSFM")
    x_plot = np.linspace(0.0, 1.05 * float(np.max(power_squared)), 200)
    ax.plot(x_plot, slope * x_plot + fit.intercept_w, label="linear fit")
    ax.plot(
        x_plot,
        4.0 * gamma**2 * cut_power * dar.n1 * x_plot,
        "--",
        label="Dar $N_1$",
    )
    ax.set_xlabel(r"Interferer power squared $P_i^2$ (W$^2$)")
    ax.set_ylabel("Paired excess CUT error power (W)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    figure_path = out_dir / "ssfm_xpm_n1_validation.pdf"
    fig.savefig(figure_path)
    plt.close(fig)

    print(f"SSFM N1 = {n1_ssfm:.6e} +/- {n1_ssfm_stderr:.2e} m^2")
    print(f"Dar  N1 = {dar.n1:.6e} +/- {dar.n1_stderr:.2e} m^2")
    print(f"ratio = {n1_ssfm / dar.n1:.6g}, z = {z_score:.3g}, R^2 = {r_squared:.6g}")
    print(f"Saved {json_path}, {csv_path}, and {figure_path}")


if __name__ == "__main__":
    main()
