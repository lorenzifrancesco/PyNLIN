"""Run the three-active-channel degenerate-FWM SSFM case at the ZDW."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.methods.ssfm_interface import (
    _build_scalar_fwm_runtime_config,
    _write_toml,
    prepare_ssfm_runtime_with_template,
)
from analysis.standalone_numerical.validate_ssfm_fwm import degenerate_fwm_channels
from analysis.standalone_numerical.validate_ssfm_xpm_spectrum import (
    local_taylor_coefficients,
)
from pynlin.methods.td.fullband_mc import gamma_grid
from pynlin.methods.td.fwm_mc import estimate_fwm_term_sum_dar_mc
from pynlin.system import System
from pynlin.utils import _toml_load


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("input/studies.toml"))
    parser.add_argument("--target-frequency-thz", type=float, default=228.265378)
    parser.add_argument("--out-dir", type=Path, default=Path("results/ssfm_fwm_constellation"))
    args = parser.parse_args()

    raw = _toml_load(args.config)
    config = dict(raw.get("ssfm", {}).get("fwm_constellation", {}))
    system = System.from_toml(args.config)
    target = float(args.target_frequency_thz) * 1e12
    shift = int(config.get("fwm_shift_channels", 1))
    spacing = shift * float(system.wdm.spacing)
    pump = target + spacing
    source = target + 2.0 * spacing
    powers = [
        float(config.get("cut_power_w", 0.2e-3)),
        float(config.get("pump_power_w", 0.2e-3)),
        float(config.get("source_power_w", 0.2e-3)),
    ]
    ssfm_betas, _ = local_taylor_coefficients(system, target, source)
    out_dir = args.out_dir.resolve()
    template = Path(config.get("template", "/home/lorenzi/sw/gnlse-python/input/wdm_nli_config.toml"))
    context = prepare_ssfm_runtime_with_template(out_dir, template)
    if context is None:
        raise RuntimeError("The external SSFM runtime is unavailable")

    runtime = _build_scalar_fwm_runtime_config(
        template_cfg=context["template_cfg"],
        system=system,
        idler_power_w=powers[0],
        pump_power_w=powers[1],
        source_power_w=powers[2],
        target_frequency_hz=target,
        spacing_hz=spacing,
        out_dir=out_dir,
        scenario_tag="fwm_zdw_three_active_channels",
        n_trials=int(config.get("n_trials", 1)),
        rng_seed=int(config.get("seed", 1234)),
        n_symbols=int(config.get("n_symbols", 2048)),
        samples_per_symbol=int(config.get("samples_per_symbol", 8)),
        max_step_m=float(config.get("max_step_m", 500.0)),
        max_nonlinear_phase_deg=float(config.get("max_nonlinear_phase_deg", 0.2)),
        fiber_length_m=float(config.get("fiber_length_m", system.fiber_length)),
        dispersion_betas_ps_per_m=ssfm_betas,
        save_plots=True,
    )
    runtime_path = out_dir / "runtime" / "fwm_zdw_three_active_channels.toml"
    _write_toml(runtime_path, runtime)
    payload = context["run_fn"](runtime_path)
    beta2 = ssfm_betas[0] * 1e-24
    beta3 = ssfm_betas[1] * 1e-36
    beta4 = ssfm_betas[2] * 1e-48

    channels = degenerate_fwm_channels(
        beta2=beta2, beta3=beta3, beta4=beta4, spacing_hz=spacing
    )
    dar = estimate_fwm_term_sum_dar_mc(
        channels=channels,
        baud_rate=float(system.pulse.baud_rate),
        length=float(runtime["fiber"]["fiber_length_m"]),
        n_samples=int(config.get("mc_samples", 200_000)),
        alpha=0.0,
        seed=int(config.get("seed", 1234)),
    )
    gamma = float(gamma_grid(system, np.array([target]))[0])
    predicted_fwm_power = 2.0 * gamma**2 * powers[1] ** 2 * powers[2] * dar.total
    summary = {
        "interpretation": "total received CUT NLI containing SCI, two XPM terms, and degenerate FWM",
        "active_frequencies_thz": [target * 1e-12, pump * 1e-12, source * 1e-12],
        "active_powers_w": powers,
        "tuple": "(d,d+s,d+s,d+2s)",
        "dar_fwm_coefficient_m2": dar.total,
        "dar_fwm_coefficient_over_l2": dar.total / float(runtime["fiber"]["fiber_length_m"]) ** 2,
        "predicted_first_order_fwm_power_w": predicted_fwm_power,
        "fwm_power_prefactor": "2*gamma^2*P_pump^2*P_source",
        "power_report": payload["power_report"],
        "plot_outputs": payload["plot_outputs"],
        "runtime_toml": str(runtime_path),
    }
    summary_path = out_dir / "fwm_zdw_three_active_channels.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()
