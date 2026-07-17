"""Sample the physical XPM spectrum with controlled scalar SSFM."""

from __future__ import annotations

import argparse
import csv
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
from pynlin.methods.td.fullband_mc import estimate_xpm_n1_local_taylor_mc
from pynlin.system import System
from pynlin.utils import _toml_load


def select_spectrum_target_indices(
    system: System, *, target_count: int, interferer_shift: int
) -> np.ndarray:
    """Select approximately uniform real WDM targets inside the beta profile."""
    frequencies = np.asarray(system.wdm.frequency_grid(), dtype=float)
    indices = np.arange(frequencies.size)
    grid_direction = 1 if float(np.median(np.diff(frequencies))) > 0.0 else -1
    interferers = indices + grid_direction * int(interferer_shift)
    valid = (interferers >= 0) & (interferers < frequencies.size)
    profile = getattr(system.fiber, "_freq_profile", None)
    if profile is not None:
        lower = float(np.min(profile))
        upper = float(np.max(profile))
        clipped = np.clip(interferers, 0, frequencies.size - 1)
        valid &= (frequencies >= lower) & (frequencies <= upper)
        valid &= (frequencies[clipped] >= lower) & (frequencies[clipped] <= upper)
    candidates = indices[valid]
    if candidates.size == 0:
        raise ValueError("No real WDM target/interferer pairs lie inside the beta profile")
    candidates = candidates[np.argsort(frequencies[candidates])]
    count = min(int(target_count), candidates.size)
    if count < 1:
        raise ValueError("target_count must be positive")
    positions = np.rint(np.linspace(0, candidates.size - 1, count)).astype(int)
    return candidates[np.unique(positions)]


def local_taylor_coefficients(
    system: System, target_frequency_hz: float, interferer_frequency_hz: float
) -> tuple[list[float], tuple[np.ndarray, ...]]:
    """Build one target-centered beta2-beta4 polynomial for SSFM and Dar."""
    spline_factory = getattr(system.fiber, "beta_spline_omega", None)
    if spline_factory is None:
        beta2 = float(np.asarray(system.beta_grids(np.array([target_frequency_hz]))[1])[0, 0])
        beta3 = 0.0
        beta4 = 0.0
    else:
        spline = spline_factory(s=0.0, k=5)
        omega_target = 2.0 * np.pi * float(target_frequency_hz)
        beta2 = float(spline.derivative(2)(omega_target))
        beta3 = float(spline.derivative(3)(omega_target))
        beta4 = float(spline.derivative(4)(omega_target))

    delta = 2.0 * np.pi * (float(interferer_frequency_hz) - float(target_frequency_hz))
    beta0 = np.array([
        0.0,
        0.5 * beta2 * delta**2 + beta3 * delta**3 / 6.0 + beta4 * delta**4 / 24.0,
    ])
    beta1 = np.array([
        0.0,
        beta2 * delta + 0.5 * beta3 * delta**2 + beta4 * delta**3 / 6.0,
    ])
    beta2_grid = np.array([beta2, beta2 + beta3 * delta + 0.5 * beta4 * delta**2])
    beta3_grid = np.array([beta3, beta3 + beta4 * delta])
    beta4_grid = np.array([beta4, beta4])
    ssfm_betas = [beta2 * 1e24, beta3 * 1e36, beta4 * 1e48]
    return ssfm_betas, (beta0, beta1, beta2_grid, beta3_grid, beta4_grid)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("input/studies.toml"))
    parser.add_argument("--template", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--target-count", type=int)
    parser.add_argument("--target-frequencies-thz", type=float, nargs="+")
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


def _setting(args, config: dict, argument: str, key: str, default):
    value = getattr(args, argument)
    return config.get(key, default) if value is None else value


def main() -> None:
    args = _parse_args()
    raw = _toml_load(args.config)
    config = dict(raw.get("ssfm", {}).get("xpm_spectrum_validation", {}))
    system = System.from_toml(args.config)
    frequencies = np.asarray(system.wdm.frequency_grid(), dtype=float)
    baud_rate = float(system.pulse.baud_rate)

    shift = int(_setting(args, config, "interferer_shift", "interferer_shift", 2))
    target_count = int(_setting(args, config, "target_count", "target_count", 5))
    cut_power = float(_setting(args, config, "cut_power_w", "cut_power_w", 0.2e-3))
    powers = np.asarray(
        _setting(
            args,
            config,
            "interferer_powers_w",
            "interferer_powers_w",
            [0.04e-3, 0.10e-3, 0.16e-3],
        ),
        dtype=float,
    )
    n_trials = int(_setting(args, config, "n_trials", "n_trials", 4))
    seed = int(_setting(args, config, "seed", "seed", 1234))
    n_symbols = int(_setting(args, config, "n_symbols", "n_symbols", 1024))
    samples_per_symbol = int(
        _setting(args, config, "samples_per_symbol", "samples_per_symbol", 8)
    )
    length = float(
        _setting(args, config, "fiber_length_m", "fiber_length_m", system.fiber_length)
    )
    max_step = float(_setting(args, config, "max_step_m", "max_step_m", 500.0))
    max_phase = float(
        _setting(
            args,
            config,
            "max_nonlinear_phase_deg",
            "max_nonlinear_phase_deg",
            0.2,
        )
    )
    mc_samples = int(_setting(args, config, "mc_samples", "mc_samples", 200_000))
    symbol_distribution = str(
        _setting(
            args,
            config,
            "symbol_distribution",
            "symbol_distribution",
            "circular_complex_gaussian",
        )
    )
    out_dir = Path(
        _setting(
            args,
            config,
            "out_dir",
            "out_dir",
            "results/ssfm_xpm_spectrum",
        )
    ).resolve()
    template = Path(
        _setting(
            args,
            config,
            "template",
            "template",
            "/home/lorenzi/sw/gnlse-python/input/wdm_nli_config.toml",
        )
    )
    if shift == 0 or powers.size < 2 or np.any(powers <= 0.0):
        raise ValueError("Use a nonzero shift and at least two positive power points")
    spacing = abs(float(system.wdm.spacing) * shift)
    if spacing < 2.0 * baud_rate:
        raise ValueError("interferer spacing must be at least twice the baud rate")

    configured_frequencies = _setting(
        args,
        config,
        "target_frequencies_thz",
        "target_frequencies_thz",
        None,
    )
    if configured_frequencies is None:
        target_indices = select_spectrum_target_indices(
            system, target_count=target_count, interferer_shift=shift
        )
        target_frequencies = frequencies[target_indices]
    else:
        target_frequencies = np.asarray(configured_frequencies, dtype=float) * 1e12
        target_indices = np.full(target_frequencies.size, -1, dtype=int)
    interferer_frequencies = target_frequencies + shift * float(system.wdm.spacing)
    profile = np.asarray(getattr(system.fiber, "_freq_profile", frequencies), dtype=float)
    if (
        np.any(target_frequencies < np.min(profile))
        or np.any(interferer_frequencies > np.max(profile))
    ):
        raise ValueError("target/interferer frequencies must remain inside the beta profile")
    out_dir.mkdir(parents=True, exist_ok=True)
    ssfm_ctx = prepare_ssfm_runtime_with_template(out_dir, template)
    if ssfm_ctx is None:
        raise RuntimeError("The external gnlse-python SSFM runtime is unavailable")
    rows = []
    for target_number, (target_frequency, interferer_frequency) in enumerate(
        zip(target_frequencies, interferer_frequencies, strict=True)
    ):
        target = int(target_indices[target_number])
        point_seed = seed + 1_000_003 * target_number
        print(
            f"Target {target_number + 1}/{target_frequencies.size}: "
            f"{target_frequency * 1e-12:.6f} THz"
        )
        ssfm_betas, local_grids = local_taylor_coefficients(
            system, target_frequency, interferer_frequency
        )
        results = []
        for power_number, power in enumerate(powers):
            results.append(
                compute_scalar_ssfm_xpm_n1(
                    ssfm_ctx,
                    system=system,
                    cut_power_w=cut_power,
                    interferer_power_w=float(power),
                    channel_idx=0,
                    interferer_idx=1,
                    n_trials=n_trials,
                    rng_seed=point_seed,
                    sweep_tag=f"t{target_number:03d}_p{power_number:02d}",
                    n_symbols=n_symbols,
                    samples_per_symbol=samples_per_symbol,
                    max_step_m=max_step,
                    max_nonlinear_phase_deg=max_phase,
                    fiber_length_m=length,
                    target_frequency_hz=float(target_frequency),
                    interferer_frequency_hz=float(interferer_frequency),
                    dispersion_betas_ps_per_m=ssfm_betas,
                    symbol_distribution=symbol_distribution,
                )
            )
        fit = fit_scalar_ssfm_xpm_sweep(results, powers, cut_power_w=cut_power)
        beta0, beta1, beta2_grid, beta3_grid, beta4_grid = local_grids
        dar_value, dar_stderr = estimate_xpm_n1_local_taylor_mc(
            beta0_offsets=beta0,
            beta1=beta1,
            beta2=beta2_grid,
            beta3=beta3_grid,
            beta4=beta4_grid,
            baud_rate=baud_rate,
            length=length,
            target=0,
            interferer=1,
            n_samples=mc_samples,
            seed=point_seed,
            alpha=0.0,
        )
        combined = float(np.hypot(fit.n1_ssfm_stderr_m2, dar_stderr))
        rows.append(
            {
                "target_index": int(target),
                "interferer_index": -1,
                "target_frequency_hz": float(target_frequency),
                "interferer_frequency_hz": float(interferer_frequency),
                "beta2_s2_per_m": float(beta2_grid[0]),
                "beta3_s3_per_m": float(beta3_grid[0]),
                "beta4_s4_per_m": float(beta4_grid[0]),
                "n1_ssfm_m2": fit.n1_ssfm_m2,
                "n1_ssfm_stderr_m2": fit.n1_ssfm_stderr_m2,
                "n1_dar_local_taylor_m2": dar_value,
                "n1_dar_local_taylor_stderr_m2": dar_stderr,
                "ssfm_over_dar": fit.n1_ssfm_m2 / dar_value,
                "difference_z_score": (
                    (fit.n1_ssfm_m2 - dar_value) / combined if combined > 0.0 else np.nan
                ),
                "fit_r_squared": fit.r_squared,
            }
        )

    fields = tuple(rows[0])
    arrays = {field: np.asarray([row[field] for row in rows]) for field in fields}
    metadata = {
        "calculation": np.array("controlled_scalar_ssfm_xpm_spectrum"),
        "config": np.array(str(args.config)),
        "template": np.array(str(template)),
        "xpm_shift_channels": np.array(shift),
        "baud_rate_hz": np.array(baud_rate),
        "channel_spacing_hz": np.array(float(system.wdm.spacing)),
        "fiber_length_m": np.array(length),
        "cut_power_w": np.array(cut_power),
        "interferer_powers_w": powers,
        "n_trials": np.array(n_trials),
        "n_symbols": np.array(n_symbols),
        "samples_per_symbol": np.array(samples_per_symbol),
        "max_step_m": np.array(max_step),
        "max_nonlinear_phase_deg": np.array(max_phase),
        "mc_samples": np.array(mc_samples),
        "symbol_distribution": np.array(symbol_distribution),
        "dispersion_model": np.array("target_centered_local_taylor_beta2_beta3_beta4"),
    }
    npz_path = out_dir / "ssfm_xpm_spectrum_validation.npz"
    np.savez(npz_path, **arrays, **metadata)
    csv_path = out_dir / "ssfm_xpm_spectrum_validation.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        writer.writerows(zip(*(arrays[field] for field in fields), strict=True))

    order = np.argsort(arrays["target_frequency_hz"])
    frequency_thz = arrays["target_frequency_hz"][order] * 1e-12
    normalization = length**2
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.0), sharex=True)
    axes[0].errorbar(
        frequency_thz,
        arrays["n1_ssfm_m2"][order] / normalization,
        yerr=arrays["n1_ssfm_stderr_m2"][order] / normalization,
        fmt="o",
        label="scalar SSFM",
    )
    axes[0].errorbar(
        frequency_thz,
        arrays["n1_dar_local_taylor_m2"][order] / normalization,
        yerr=arrays["n1_dar_local_taylor_stderr_m2"][order] / normalization,
        fmt=".-",
        label=r"Dar local $\beta_2$-$\beta_4$",
    )
    ratio_stderr = np.hypot(
        arrays["n1_ssfm_stderr_m2"][order]
        / arrays["n1_dar_local_taylor_m2"][order],
        arrays["n1_ssfm_m2"][order]
        * arrays["n1_dar_local_taylor_stderr_m2"][order]
        / arrays["n1_dar_local_taylor_m2"][order] ** 2,
    )
    axes[1].errorbar(
        frequency_thz, arrays["ssfm_over_dar"][order], yerr=ratio_stderr, fmt="o-"
    )
    axes[1].axhline(1.0, color="0.3", ls="--", lw=0.8)
    axes[0].set_yscale("log")
    axes[0].set_ylabel(r"$N_1/L^2$")
    axes[1].set_ylabel("SSFM / Dar")
    axes[1].set_xlabel("target frequency [THz]")
    axes[0].legend(frameon=False)
    for axis in axes:
        axis.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    pdf_path = out_dir / "ssfm_xpm_spectrum_validation.pdf"
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"Saved {npz_path}, {csv_path}, and {pdf_path}")


if __name__ == "__main__":
    main()
