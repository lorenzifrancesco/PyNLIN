"""Validate the single-tuple Dar FWM coefficient against controlled scalar SSFM.

Three active channels are simulated: the idler/CUT at the target frequency,
the pump one spacing above, and the source two spacings above, so the
degenerate product 2*f_pump - f_source lands exactly on the idler. The FWM
excess is isolated by inclusion-exclusion over four seed-paired runs and
fitted against P_pump^2 * P_source; the resulting coefficient
slope / (2*gamma^2) is compared with `estimate_fwm_term_sum_dar_mc`.
"""

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
    compute_scalar_ssfm_fwm_excess,
    fit_scalar_ssfm_fwm_sweep,
    prepare_ssfm_runtime_with_template,
)
from analysis.standalone_numerical.validate_ssfm_xpm_spectrum import (
    local_taylor_coefficients,
)
from pynlin.methods.td.fwm_kernel import FWMChannels
from pynlin.methods.td.fwm_mc import estimate_fwm_term_sum_dar_mc
from pynlin.system import System
from pynlin.utils import _toml_load


def degenerate_fwm_channels(
    *, beta2: float, beta3: float, beta4: float, spacing_hz: float
) -> FWMChannels:
    """Build the degenerate tuple (d, d+s, d+s, d+2s) from target-local betas."""
    delta = 2.0 * np.pi * float(spacing_hz)

    def beta0(offset):
        return 0.5 * beta2 * offset**2 + beta3 * offset**3 / 6.0 + beta4 * offset**4 / 24.0

    def beta1(offset):
        return beta2 * offset + 0.5 * beta3 * offset**2 + beta4 * offset**3 / 6.0

    def beta2_local(offset):
        return beta2 + beta3 * offset + 0.5 * beta4 * offset**2

    return FWMChannels(
        omega_a=delta,
        omega_b=delta,
        omega_c=2.0 * delta,
        omega_d=0.0,
        beta0_a=beta0(delta),
        beta0_b=beta0(delta),
        beta0_c=beta0(2.0 * delta),
        beta0_d=0.0,
        beta1_a=beta1(delta),
        beta1_b=beta1(delta),
        beta1_c=beta1(2.0 * delta),
        beta1_d=0.0,
        gvd_a=beta2_local(delta),
        gvd_b=beta2_local(delta),
        gvd_c=beta2_local(2.0 * delta),
        gvd_d=beta2,
        beta3_a=beta3 + beta4 * delta,
        beta3_b=beta3 + beta4 * delta,
        beta3_c=beta3 + 2.0 * beta4 * delta,
        beta3_d=beta3,
        beta4_a=beta4,
        beta4_b=beta4,
        beta4_c=beta4,
        beta4_d=beta4,
    )


def _section(config_path: Path) -> dict:
    raw = _toml_load(config_path)
    return dict(raw.get("ssfm", {}).get("fwm_validation", {}))


def _value(cli_value, config: dict, key: str, default):
    return cli_value if cli_value is not None else config.get(key, default)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("input/studies.toml"))
    parser.add_argument("--template", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--target-frequency-thz", type=float)
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


def main() -> None:
    args = _parse_args()
    cfg = _section(args.config)
    system = System.from_toml(args.config)
    baud_rate = float(system.pulse.baud_rate)

    target = float(
        _value(args.target_frequency_thz, cfg, "target_frequency_thz", 228.265378)
    ) * 1e12
    shift = int(_value(args.fwm_shift_channels, cfg, "fwm_shift_channels", 1))
    if shift == 0:
        raise ValueError("fwm_shift_channels must be nonzero")
    spacing = abs(shift) * float(system.wdm.spacing)
    if spacing < 2.0 * baud_rate:
        raise ValueError(
            "Controlled FWM validation requires spacing >= 2*baud_rate so the "
            "pump/source cubic spectra cannot overlap the idler band"
        )
    pump_frequency = target + spacing
    source_frequency = target + 2.0 * spacing

    idler_power = float(_value(args.cut_power_w, cfg, "cut_power_w", 0.2e-3))
    pump_base = float(_value(args.pump_power_w, cfg, "pump_power_w", 0.2e-3))
    source_base = float(_value(args.source_power_w, cfg, "source_power_w", 0.2e-3))
    scales = np.asarray(
        _value(args.power_scales, cfg, "power_scales", [0.7, 1.0, 1.3]), dtype=float
    )
    if scales.size < 2 or np.any(scales <= 0.0) or np.unique(scales).size != scales.size:
        raise ValueError("At least two distinct positive power scales are required")
    pump_powers = pump_base * scales
    source_powers = source_base * scales

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
        _value(args.out_dir, cfg, "out_dir", "results/ssfm_fwm_n1")
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

    ssfm_betas, _ = local_taylor_coefficients(system, target, source_frequency)
    beta2 = ssfm_betas[0] * 1e-24
    beta3 = ssfm_betas[1] * 1e-36
    beta4 = ssfm_betas[2] * 1e-48

    results = []
    idler_only_payload = None
    for index, (pump_power, source_power) in enumerate(
        zip(pump_powers, source_powers, strict=True)
    ):
        print(
            f"Running paired SSFM scale {index + 1}/{scales.size}: "
            f"pump {pump_power:.6g} W, source {source_power:.6g} W"
        )
        result = compute_scalar_ssfm_fwm_excess(
            ssfm_ctx,
            system=system,
            idler_power_w=idler_power,
            pump_power_w=float(pump_power),
            source_power_w=float(source_power),
            target_frequency_hz=target,
            spacing_hz=spacing,
            n_trials=n_trials,
            rng_seed=seed,
            sweep_tag=f"s{index:02d}",
            n_symbols=n_symbols,
            samples_per_symbol=samples_per_symbol,
            max_step_m=max_step,
            max_nonlinear_phase_deg=max_phase,
            fiber_length_m=fiber_length,
            dispersion_betas_ps_per_m=ssfm_betas,
            idler_only_payload=idler_only_payload,
            symbol_distribution=symbol_distribution,
        )
        idler_only_payload = result.idler_only_payload
        results.append(result)

    fit = fit_scalar_ssfm_fwm_sweep(results, pump_powers, source_powers)
    gamma = results[0].gamma_w_inv_m
    power_cubed = pump_powers**2 * source_powers

    channels = degenerate_fwm_channels(
        beta2=beta2, beta3=beta3, beta4=beta4, spacing_hz=spacing
    )
    dar = estimate_fwm_term_sum_dar_mc(
        channels=channels,
        baud_rate=baud_rate,
        length=fiber_length,
        n_samples=mc_samples,
        alpha=0.0,
        seed=seed,
    )
    combined_stderr = float(np.hypot(fit.c_ssfm_stderr_m2, dar.total_stderr))
    z_score = (
        (fit.c_ssfm_m2 - dar.total) / combined_stderr
        if combined_stderr > 0.0 else float("nan")
    )

    payload = {
        "calculation": "controlled_scalar_ssfm_vs_prefactor_free_dar_fwm",
        "config": str(args.config),
        "template": str(template),
        "tuple": "(d,d+s,d+s,d+2s)",
        "target_frequency_hz": target,
        "pump_frequency_hz": pump_frequency,
        "source_frequency_hz": source_frequency,
        "spacing_hz": spacing,
        "baud_rate_hz": baud_rate,
        "beta2_s2_per_m": beta2,
        "beta3_s3_per_m": beta3,
        "beta4_s4_per_m": beta4,
        "fiber_length_m": fiber_length,
        "gamma_w_inv_m": gamma,
        "cut_power_w": idler_power,
        "pump_powers_w": pump_powers.tolist(),
        "source_powers_w": source_powers.tolist(),
        "n_trials": n_trials,
        "seed": seed,
        "n_symbols": n_symbols,
        "samples_per_symbol": samples_per_symbol,
        "max_step_m": max_step,
        "max_nonlinear_phase_deg": max_phase,
        "mc_samples": mc_samples,
        "symbol_distribution": symbol_distribution,
        "mean_excess_power_w": fit.mean_excess_power_w.tolist(),
        "mean_excess_power_stderr_w": fit.mean_excess_power_stderr_w.tolist(),
        "fit_slope_w_inv2": fit.slope_w_inv2,
        "fit_slope_stderr_w_inv2": fit.slope_stderr_w_inv2,
        "fit_intercept_w": fit.intercept_w,
        "fit_r_squared": fit.r_squared,
        "c_ssfm_m2": fit.c_ssfm_m2,
        "c_ssfm_stderr_m2": fit.c_ssfm_stderr_m2,
        "c_dar_m2": dar.total,
        "c_dar_stderr_m2": dar.total_stderr,
        "ssfm_over_dar": fit.c_ssfm_m2 / dar.total,
        "difference_z_score": z_score,
    }
    json_path = out_dir / "ssfm_fwm_validation.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    csv_path = out_dir / "ssfm_fwm_power_sweep.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "pump_power_w",
            "source_power_w",
            "pump_sq_source_w3",
            "excess_power_w",
            "stderr_w",
        ])
        writer.writerows(zip(
            pump_powers,
            source_powers,
            power_cubed,
            fit.mean_excess_power_w,
            fit.mean_excess_power_stderr_w,
        ))

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.errorbar(
        power_cubed,
        fit.mean_excess_power_w,
        yerr=fit.mean_excess_power_stderr_w,
        fmt="o",
        label="paired SSFM",
    )
    x_plot = np.linspace(0.0, 1.05 * float(np.max(power_cubed)), 200)
    ax.plot(x_plot, fit.slope_w_inv2 * x_plot + fit.intercept_w, label="linear fit")
    ax.plot(x_plot, 2.0 * gamma**2 * dar.total * x_plot, "--", label="Dar FWM")
    ax.set_xlabel(r"$P_{\mathrm{pump}}^2 P_{\mathrm{source}}$ (W$^3$)")
    ax.set_ylabel("Paired excess idler error power (W)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    figure_path = out_dir / "ssfm_fwm_validation.pdf"
    fig.savefig(figure_path)
    plt.close(fig)

    print(f"SSFM C = {fit.c_ssfm_m2:.6e} +/- {fit.c_ssfm_stderr_m2:.2e} m^2")
    print(f"Dar  C = {dar.total:.6e} +/- {dar.total_stderr:.2e} m^2")
    print(
        f"ratio = {fit.c_ssfm_m2 / dar.total:.6g}, z = {z_score:.3g}, "
        f"R^2 = {fit.r_squared:.6g}"
    )
    print(f"Saved {json_path}, {csv_path}, and {figure_path}")


if __name__ == "__main__":
    main()
