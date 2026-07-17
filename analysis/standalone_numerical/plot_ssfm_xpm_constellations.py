"""Generate native SSFM received-cloud plots for cached XPM spectrum dots."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.methods.ssfm_interface import (
    _build_scalar_xpm_runtime_config,
    _write_toml,
    prepare_ssfm_runtime_with_template,
)
from analysis.standalone_numerical.validate_ssfm_xpm_spectrum import (
    local_taylor_coefficients,
)
from pynlin.system import System
from pynlin.utils import _toml_load


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("input/studies.toml"))
    parser.add_argument("--cache", type=Path, default=Path("results/ssfm_xpm_spectrum/ssfm_xpm_spectrum_validation.npz"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/ssfm_xpm_constellations"))
    parser.add_argument("--interferer-power-w", type=float, default=None)
    parser.add_argument("--n-symbols", type=int, default=None)
    parser.add_argument(
        "--symbol-distribution", choices=["qam", "circular_complex_gaussian"]
    )
    args = parser.parse_args()

    raw = _toml_load(args.config)
    config = dict(raw.get("ssfm", {}).get("xpm_spectrum_validation", {}))
    system = System.from_toml(args.config)
    with np.load(args.cache, allow_pickle=False) as cache:
        target_frequencies = np.asarray(cache["target_frequency_hz"], dtype=float)
        shift = int(cache["xpm_shift_channels"])
        powers = np.asarray(cache["interferer_powers_w"], dtype=float)
        length = float(cache["fiber_length_m"])
        cached_distribution = (
            str(cache["symbol_distribution"])
            if "symbol_distribution" in cache.files
            else "circular_complex_gaussian"
        )
    symbol_distribution = (
        cached_distribution
        if args.symbol_distribution is None
        else args.symbol_distribution
    )

    interferer_power = (
        float(np.max(powers))
        if args.interferer_power_w is None
        else float(args.interferer_power_w)
    )
    cut_power = float(config.get("cut_power_w", 0.2e-3))
    n_symbols = int(config.get("n_symbols", 1024) if args.n_symbols is None else args.n_symbols)
    samples_per_symbol = int(config.get("samples_per_symbol", 8))
    max_step = float(config.get("max_step_m", 500.0))
    max_phase = float(config.get("max_nonlinear_phase_deg", 0.2))
    template = Path(config.get("template", "/home/lorenzi/sw/gnlse-python/input/wdm_nli_config.toml"))
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    context = prepare_ssfm_runtime_with_template(out_dir, template)
    if context is None:
        raise RuntimeError("The external SSFM runtime is unavailable")

    rows = []
    spacing = shift * float(system.wdm.spacing)
    for index, target_frequency in enumerate(target_frequencies):
        interferer_frequency = target_frequency + spacing
        ssfm_betas, _ = local_taylor_coefficients(
            system, target_frequency, interferer_frequency
        )
        tag = f"xpm_{target_frequency * 1e-12:.6f}THz"
        runtime = _build_scalar_xpm_runtime_config(
            template_cfg=context["template_cfg"],
            system=system,
            cut_power_w=cut_power,
            interferer_power_w=interferer_power,
            channel_idx=0,
            interferer_idx=1,
            out_dir=out_dir,
            scenario_tag=tag,
            n_trials=1,
            rng_seed=int(config.get("seed", 1234)) + 1_000_003 * index,
            n_symbols=n_symbols,
            samples_per_symbol=samples_per_symbol,
            max_step_m=max_step,
            max_nonlinear_phase_deg=max_phase,
            fiber_length_m=length,
            target_frequency_hz=float(target_frequency),
            interferer_frequency_hz=float(interferer_frequency),
            dispersion_betas_ps_per_m=ssfm_betas,
            save_plots=True,
            symbol_distribution=symbol_distribution,
        )
        runtime_path = out_dir / "runtime" / f"{tag}.toml"
        _write_toml(runtime_path, runtime)
        payload = context["run_fn"](runtime_path)
        rows.append(
            {
                "target_frequency_thz": target_frequency * 1e-12,
                "interferer_power_w": interferer_power,
                "nli_power_w": payload["power_report"]["center_channel"]["nli_power_avg_w"],
                "constellation_plot": payload["plot_outputs"]["constellation_plot"],
                "constellation_plot_no_ticks": payload["plot_outputs"]["constellation_plot_no_ticks"],
            }
        )
        print(f"Plotted {index + 1}/{target_frequencies.size}: {tag}")

    manifest = out_dir / "xpm_constellation_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {manifest}")


if __name__ == "__main__":
    main()
