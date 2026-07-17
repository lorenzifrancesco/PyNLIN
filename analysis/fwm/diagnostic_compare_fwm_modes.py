from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from loguru import logger as lg

from analysis.log_init import init_logging
from pynlin.methods.td.fullband_mc import (
    compute_fullband_prefactor_free_mc,
    decimated_frequency_grid,
    estimate_zdw_frequency,
)
from pynlin.system import System
from pynlin.utils import nu2lambda


def find_representative_targets(freqs, zdw_freq, n=3):
    """Pick n targets: near-ZDW, mid-grid, and far (U-band)."""
    if np.isfinite(zdw_freq):
        zdw_idx = int(np.argmin(np.abs(freqs - zdw_freq)))
    else:
        zdw_idx = len(freqs) // 2

    # near ZDW
    t1 = zdw_idx
    # far U-band (lowest freq / longest wavelength)
    t2 = 0
    # mid C-band
    t3 = len(freqs) // 2

    targets = sorted(set([t1, t2, t3]))
    return np.array(targets, dtype=int)


def main():
    init_logging()
    config_path = Path("input/pcfm_1200ch_oesclu.toml")
    system = System.from_toml(config_path)

    channel_decimation = 4
    xpm_samples = 20000
    fwm_samples = 10000
    max_fwm_tuples = 5000
    seed = 1234

    _, freqs = decimated_frequency_grid(system, channel_decimation)
    zdw_freq = estimate_zdw_frequency(system)
    lg.info(f"ZDW frequency: {zdw_freq*1e-12:.3f} THz ({nu2lambda(zdw_freq)*1e9:.1f} nm)")
    lg.info(f"Grid: {freqs[0]*1e-12:.2f} - {freqs[-1]*1e-12:.2f} THz, {freqs.size} channels")

    targets = find_representative_targets(freqs, zdw_freq)
    lg.info(f"Targets (decimated indices): {targets}")
    for ti in targets:
        lg.info(f"  index {ti}: {freqs[ti]*1e-12:.3f} THz ({nu2lambda(freqs[ti])*1e9:.1f} nm)")

    results = {}

    for mode in ("joint_reservoir", "phase_proxy"):
        lg.info(f"--- Running FWM selection mode: {mode} ---")
        diag = compute_fullband_prefactor_free_mc(
            system,
            decimation=channel_decimation,
            target_indices=targets,
            include_xpm=True,
            include_fwm=True,
            xpm_samples=xpm_samples,
            fwm_samples=fwm_samples,
            seed=seed,
            max_fwm_tuples_per_target=max_fwm_tuples,
            fwm_tuple_selection=mode,
        )
        results[mode] = diag

        for i, ti in enumerate(diag.target_indices):
            supp = diag.fwm_support_count[i]
            ev = diag.fwm_tuple_count[i]
            xpm = diag.xpm[i]
            fwm = diag.fwm[i]
            total = diag.total[i]
            fwm_pct = 100 * fwm / total if total > 0 else 0
            xpm_pct = 100 * xpm / total if total > 0 else 0
            f_thz = diag.target_frequencies[i] * 1e-12
            lg.info(
                f"  [{mode}] target {ti} @ {f_thz:.2f} THz: "
                f"XPM={xpm:.4e} ({xpm_pct:.1f}%), "
                f"FWM={fwm:.4e} ({fwm_pct:.1f}%), "
                f"total={total:.4e}, "
                f"support={supp}, evaluated={ev}"
            )

    print("")
    print("=" * 80)
    print("COMPARISON: joint_reservoir vs diagnostic phase_proxy")
    print("=" * 80)
    for i, ti in enumerate(results["joint_reservoir"].target_indices):
        f_thz = results["joint_reservoir"].target_frequencies[i] * 1e-12
        for mode in ("joint_reservoir", "phase_proxy"):
            diag = results[mode]
            xpm = diag.xpm[i]
            fwm = diag.fwm[i]
            total = diag.total[i]
            fwm_pct = 100 * fwm / total if total > 0 else 0
            xpm_pct = 100 * xpm / total if total > 0 else 0
            supp = diag.fwm_support_count[i]
            ev = diag.fwm_tuple_count[i]
            print(
                f"  target {ti} @ {f_thz:.2f} THz | {mode:20s} | "
                f"XPM={xpm:.4e} ({xpm_pct:.1f}%)  "
                f"FWM={fwm:.4e} ({fwm_pct:.1f}%)  "
                f"total={total:.4e}  "
                f"support={supp:6d}  evaluated={ev:5d}"
            )

    out = results["joint_reservoir"]
    fwm_ratio = results["phase_proxy"].fwm / np.maximum(results["joint_reservoir"].fwm, 1e-30)
    for i, ti in enumerate(out.target_indices):
        f_thz = out.target_frequencies[i] * 1e-12
        print(f"  target {ti} @ {f_thz:.2f} THz | FWM ratio phase/joint = {fwm_ratio[i]:.3f}")


if __name__ == "__main__":
    main()
