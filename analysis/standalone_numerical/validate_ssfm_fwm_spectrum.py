"""Sample the degenerate-FWM coefficient across the band with scalar SSFM.

At each target frequency the tuple (d, d+s, d+s, d+2s) is validated exactly as
in `validate_ssfm_fwm.py`: four seed-paired SSFM runs isolate the FWM excess
via inclusion-exclusion, a power sweep fits the P_pump^2 * P_source slope, and
the coefficient is compared with the single-tuple Dar MC estimate. The cache
written here can be overlaid on the `validate_fwm_mc_real_tuples` plots.
"""

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
    compute_scalar_ssfm_fwm_excess,
    fit_scalar_ssfm_fwm_sweep,
    prepare_ssfm_runtime_with_template,
)
from analysis.standalone_numerical.validate_ssfm_fwm import degenerate_fwm_channels
from analysis.standalone_numerical.validate_ssfm_xpm_spectrum import (
    local_taylor_coefficients,
    select_spectrum_target_indices,
)
from pynlin.methods.td.fwm_mc import estimate_fwm_term_sum_dar_mc
from pynlin.system import System
from pynlin.utils import _toml_load


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("input/studies.toml"))
    parser.add_argument("--template", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--target-count", type=int)
    parser.add_argument("--target-frequencies-thz", type=float, nargs="+")
    parser.add_argument("--fwm-shift-channels", type=int)
    parser.add_argument("--cut-power-w", type=float)
    parser.add_argument("--pump-power-w", type=float)
    parser.add_argument("--source-power-w", type=float)
    parser.add_argument("--power-scales", type=float, nargs="+")
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
    config = dict(raw.get("ssfm", {}).get("fwm_spectrum_validation", {}))
    system = System.from_toml(args.config)
    frequencies = np.asarray(system.wdm.frequency_grid(), dtype=float)
    baud_rate = float(system.pulse.baud_rate)

    shift = int(_setting(args, config, "fwm_shift_channels", "fwm_shift_channels", 1))
    target_count = int(_setting(args, config, "target_count", "target_count", 5))
    idler_power = float(_setting(args, config, "cut_power_w", "cut_power_w", 0.2e-3))
    pump_base = float(_setting(args, config, "pump_power_w", "pump_power_w", 0.2e-3))
    source_base = float(
        _setting(args, config, "source_power_w", "source_power_w", 0.2e-3)
    )
    scales = np.asarray(
        _setting(args, config, "power_scales", "power_scales", [0.7, 1.0, 1.3]),
        dtype=float,
    )
    n_trials = int(_setting(args, config, "n_trials", "n_trials", 8))
    seed = int(_setting(args, config, "seed", "seed", 1234))
    n_symbols = int(_setting(args, config, "n_symbols", "n_symbols", 2048))
    samples_per_symbol = int(
        _setting(args, config, "samples_per_symbol", "samples_per_symbol", 8)
    )
    length = float(
        _setting(args, config, "fiber_length_m", "fiber_length_m", system.fiber_length)
    )
    max_step = float(_setting(args, config, "max_step_m", "max_step_m", 500.0))
    max_phase = float(
        _setting(
            args, config, "max_nonlinear_phase_deg", "max_nonlinear_phase_deg", 0.05
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
        _setting(args, config, "out_dir", "out_dir", "results/ssfm_fwm_spectrum")
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
    if shift == 0 or scales.size < 1 or np.any(scales <= 0.0):
        raise ValueError("Use a nonzero shift and at least one positive power scale")
    spacing = abs(shift) * float(system.wdm.spacing)
    if spacing < 2.0 * baud_rate:
        raise ValueError("pump spacing must be at least twice the baud rate")
    pump_powers = pump_base * scales
    source_powers = source_base * scales

    configured_frequencies = _setting(
        args, config, "target_frequencies_thz", "target_frequencies_thz", None
    )
    if configured_frequencies is None:
        # The farthest active channel is the source at two shifts above the target.
        target_indices = select_spectrum_target_indices(
            system, target_count=target_count, interferer_shift=2 * shift
        )
        target_frequencies = frequencies[target_indices]
    else:
        target_frequencies = np.asarray(configured_frequencies, dtype=float) * 1e12
        target_indices = np.full(target_frequencies.size, -1, dtype=int)
    source_frequencies = target_frequencies + 2.0 * spacing * np.sign(shift)
    profile = np.asarray(getattr(system.fiber, "_freq_profile", frequencies), dtype=float)
    if (
        np.any(target_frequencies < np.min(profile))
        or np.any(source_frequencies > np.max(profile))
    ):
        raise ValueError("target/source frequencies must remain inside the beta profile")
    out_dir.mkdir(parents=True, exist_ok=True)
    ssfm_ctx = prepare_ssfm_runtime_with_template(out_dir, template)
    if ssfm_ctx is None:
        raise RuntimeError("The external gnlse-python SSFM runtime is unavailable")

    rows = []
    for target_number, (target_frequency, source_frequency) in enumerate(
        zip(target_frequencies, source_frequencies, strict=True)
    ):
        point_seed = seed + 1_000_003 * target_number
        print(
            f"Target {target_number + 1}/{target_frequencies.size}: "
            f"{target_frequency * 1e-12:.6f} THz"
        )
        ssfm_betas, _ = local_taylor_coefficients(
            system, target_frequency, source_frequency
        )
        beta2 = ssfm_betas[0] * 1e-24
        beta3 = ssfm_betas[1] * 1e-36
        beta4 = ssfm_betas[2] * 1e-48
        results = []
        idler_only_payload = None
        for scale_number, (pump_power, source_power) in enumerate(
            zip(pump_powers, source_powers, strict=True)
        ):
            result = compute_scalar_ssfm_fwm_excess(
                ssfm_ctx,
                system=system,
                idler_power_w=idler_power,
                pump_power_w=float(pump_power),
                source_power_w=float(source_power),
                target_frequency_hz=float(target_frequency),
                spacing_hz=spacing,
                n_trials=n_trials,
                rng_seed=point_seed,
                sweep_tag=f"t{target_number:03d}_s{scale_number:02d}",
                n_symbols=n_symbols,
                samples_per_symbol=samples_per_symbol,
                max_step_m=max_step,
                max_nonlinear_phase_deg=max_phase,
                fiber_length_m=length,
                dispersion_betas_ps_per_m=ssfm_betas,
                idler_only_payload=idler_only_payload,
                symbol_distribution=symbol_distribution,
            )
            idler_only_payload = result.idler_only_payload
            results.append(result)
        if scales.size == 1:
            # Single-point mode: apply the 2*gamma^2*P_pump^2*P_source prefactor
            # directly instead of fitting a slope across scales.
            c_ssfm = results[0].c_ssfm_m2
            c_ssfm_stderr = results[0].c_ssfm_stderr_m2
            fit_r_squared = float("nan")
        else:
            fit = fit_scalar_ssfm_fwm_sweep(results, pump_powers, source_powers)
            c_ssfm = fit.c_ssfm_m2
            c_ssfm_stderr = fit.c_ssfm_stderr_m2
            fit_r_squared = fit.r_squared
        channels = degenerate_fwm_channels(
            beta2=beta2, beta3=beta3, beta4=beta4, spacing_hz=spacing
        )
        dar = estimate_fwm_term_sum_dar_mc(
            channels=channels,
            baud_rate=baud_rate,
            length=length,
            n_samples=mc_samples,
            alpha=0.0,
            seed=point_seed,
        )
        # Unsubtracted center-channel NLI at the scale closest to 1.0, expressed
        # with the same FWM prefactor so it plots on the coefficient axis.
        reference = int(np.argmin(np.abs(scales - 1.0)))
        total_trials = results[reference].full_nli_power_trials_w
        total_mean = float(np.mean(total_trials))
        total_stderr = (
            float(np.std(total_trials, ddof=1) / np.sqrt(total_trials.size))
            if total_trials.size > 1 else 0.0
        )
        total_denominator = (
            2.0
            * results[reference].gamma_w_inv_m ** 2
            * float(pump_powers[reference]) ** 2
            * float(source_powers[reference])
        )
        # Interferer-induced NLI: the seed-paired difference removes the
        # SCI/SPM residual that survives the receiver's ideal backprop.
        interferer_trials = (
            total_trials - results[reference].idler_only_nli_power_trials_w
        )
        interferer_mean = float(np.mean(interferer_trials))
        interferer_stderr = (
            float(np.std(interferer_trials, ddof=1) / np.sqrt(interferer_trials.size))
            if interferer_trials.size > 1 else 0.0
        )
        combined = float(np.hypot(c_ssfm_stderr, dar.total_stderr))
        rows.append(
            {
                "target_index": int(target_indices[target_number]),
                "target_frequency_hz": float(target_frequency),
                "source_frequency_hz": float(source_frequency),
                "beta2_s2_per_m": beta2,
                "beta3_s3_per_m": beta3,
                "beta4_s4_per_m": beta4,
                "c_ssfm_m2": c_ssfm,
                "c_ssfm_stderr_m2": c_ssfm_stderr,
                "c_dar_m2": dar.total,
                "c_dar_stderr_m2": dar.total_stderr,
                "ssfm_over_dar": c_ssfm / dar.total,
                "difference_z_score": (
                    (c_ssfm - dar.total) / combined if combined > 0.0 else np.nan
                ),
                "fit_r_squared": fit_r_squared,
                "total_nli_power_w": total_mean,
                "total_nli_power_stderr_w": total_stderr,
                "c_total_m2": total_mean / total_denominator,
                "c_total_stderr_m2": total_stderr / total_denominator,
                "interferer_nli_power_w": interferer_mean,
                "interferer_nli_power_stderr_w": interferer_stderr,
                "c_interferer_m2": interferer_mean / total_denominator,
                "c_interferer_stderr_m2": interferer_stderr / total_denominator,
            }
        )

    fields = tuple(rows[0])
    arrays = {field: np.asarray([row[field] for row in rows]) for field in fields}
    metadata = {
        "calculation": np.array("controlled_scalar_ssfm_fwm_spectrum"),
        "config": np.array(str(args.config)),
        "template": np.array(str(template)),
        "tuple": np.array("(d,d+s,d+s,d+2s)"),
        "fwm_shift_channels": np.array(shift),
        "baud_rate_hz": np.array(baud_rate),
        "channel_spacing_hz": np.array(float(system.wdm.spacing)),
        "fiber_length_m": np.array(length),
        "cut_power_w": np.array(idler_power),
        "pump_powers_w": pump_powers,
        "source_powers_w": source_powers,
        "n_trials": np.array(n_trials),
        "n_symbols": np.array(n_symbols),
        "samples_per_symbol": np.array(samples_per_symbol),
        "max_step_m": np.array(max_step),
        "max_nonlinear_phase_deg": np.array(max_phase),
        "mc_samples": np.array(mc_samples),
        "symbol_distribution": np.array(symbol_distribution),
        "dispersion_model": np.array("target_centered_local_taylor_beta2_beta3_beta4"),
    }
    npz_path = out_dir / "ssfm_fwm_spectrum_validation.npz"
    np.savez(npz_path, **arrays, **metadata)
    csv_path = out_dir / "ssfm_fwm_spectrum_validation.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        writer.writerows(zip(*(arrays[field] for field in fields), strict=True))

    order = np.argsort(arrays["target_frequency_hz"])
    frequency_thz = arrays["target_frequency_hz"][order] * 1e-12
    normalization = length**2
    ratio = arrays["ssfm_over_dar"][order]
    ratio_stderr = np.hypot(
        arrays["c_ssfm_stderr_m2"][order] / arrays["c_dar_m2"][order],
        arrays["c_ssfm_m2"][order]
        * arrays["c_dar_stderr_m2"][order]
        / arrays["c_dar_m2"][order] ** 2,
    )
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.0), sharex=True)
    axes[0].errorbar(
        frequency_thz,
        arrays["c_ssfm_m2"][order] / normalization,
        yerr=arrays["c_ssfm_stderr_m2"][order] / normalization,
        fmt="o",
        label="scalar SSFM",
    )
    axes[0].errorbar(
        frequency_thz,
        arrays["c_dar_m2"][order] / normalization,
        yerr=arrays["c_dar_stderr_m2"][order] / normalization,
        fmt=".-",
        label=r"Dar local $\beta_2$-$\beta_4$",
    )
    axes[1].errorbar(frequency_thz, ratio, yerr=ratio_stderr, fmt="o-")
    axes[1].axhline(1.0, color="0.3", ls="--", lw=0.8)
    axes[0].set_yscale("log")
    axes[0].set_ylabel(r"$C_{\mathrm{FWM}}/L^2$")
    axes[1].set_ylabel("SSFM / Dar")
    axes[1].set_xlabel("target frequency [THz]")
    axes[0].legend(frameon=False)
    for axis in axes:
        axis.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    pdf_path = out_dir / "ssfm_fwm_spectrum_validation.pdf"
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"Saved {npz_path}, {csv_path}, and {pdf_path}")


if __name__ == "__main__":
    main()
