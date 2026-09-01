"""Lorenzi Fast S6: physical NLIN spectrum from the fast full-band sums.

Reads the S5 prefactor-free spectrum and applies the SSFM-convention
physical layer (gamma(f), flat launch power, Gaussian symbols) to produce
the NLIN variance and NSR per channel across the band.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import sys as _sys
_sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent / '.'))
import pubstyle

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pynlin  # noqa: F401
from loguru import logger as lg

from analysis.log_init import init_logging
from pynlin.methods.td.fast_nlin import physical_nlin_spectrum
from pynlin.methods.td.fullband_mc import estimate_zdw_frequency, gamma_grid
from pynlin.system import System

def main() -> None:
    init_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    pubstyle.add_argument(parser)
    parser.add_argument("--config", type=Path, default=Path("input/studies.toml"))
    parser.add_argument("--out-dir", type=Path, default=Path("media/lorenzi-fast"))
    parser.add_argument("--s5-npz", type=Path, default=Path("media/lorenzi-fast/s5_fullband.npz"))
    args = parser.parse_args()
    pubstyle.apply(args)

    system = System.from_toml(args.config)
    data = np.load(args.s5_npz)
    freqs = data["freqs"]
    xpm_m2 = data["xpm_fast"]
    fwm_m2 = data["fwm_fast"]
    # With targets-only decimation the S5 arrays hold NaN at non-target
    # positions; restrict everything downstream to the computed targets.
    computed = np.isfinite(xpm_m2) & np.isfinite(fwm_m2)
    if not np.all(computed):
        lg.info(f"S5 data covers {int(np.sum(computed))}/{computed.size} targets")
    freqs = freqs[computed]
    xpm_m2 = xpm_m2[computed]
    fwm_m2 = fwm_m2[computed]

    gamma = gamma_grid(system, freqs)
    lp_dbm = float(system.launch_power)
    launch_power_w = 10.0 ** (lp_dbm / 10.0) * 1e-3
    xpm_w, fwm_w = physical_nlin_spectrum(gamma, launch_power_w, xpm_m2, fwm_m2)
    total_w = xpm_w + fwm_w
    nsr_db = 10.0 * np.log10(total_w / launch_power_w)
    lg.info(
        f"launch {lp_dbm:.1f} dBm; NSR range [{np.min(nsr_db):.1f}, {np.max(nsr_db):.1f}] dB; "
        f"FWM/total share range [{np.min(fwm_w/total_w):.2e}, {np.max(fwm_w/total_w):.2e}]"
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out_dir / "s6_physical.npz",
        freqs=freqs, gamma=gamma, launch_power_w=launch_power_w,
        xpm_w=xpm_w, fwm_w=fwm_w, total_w=total_w, nsr_db=nsr_db,
    )

    zdw = estimate_zdw_frequency(system)
    f_thz = freqs * 1e-12
    fig, axes = plt.subplots(2, 1, figsize=pubstyle.figsize(7.0, 5.4), sharex=True)
    axes[0].plot(f_thz, xpm_w, lw=1.2, color="tab:blue", label="XPM")
    axes[0].plot(f_thz, fwm_w, lw=1.2, color="tab:red", label="FWM (strict)")
    axes[0].plot(f_thz, total_w, lw=1.0, color="black", alpha=0.6, label="total")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("NLIN variance [W]")
    axes[0].grid(True, which="both", alpha=0.25)
    axes[0].legend(fontsize=8, frameon=False)
    axes[1].plot(f_thz, nsr_db, lw=1.2, color="black")
    axes[1].set_xlabel("channel frequency [THz]")
    axes[1].set_ylabel("NLIN-to-signal [dB]")
    axes[1].grid(True, alpha=0.25)
    if zdw is not None and np.isfinite(zdw):
        for a in axes:
            a.axvline(zdw * 1e-12, color="crimson", lw=0.8, ls=":", alpha=0.7)
    fig.suptitle(
        "Lorenzi Fast physical NLIN spectrum\n"
        f"gamma(f) per-channel, flat launch {lp_dbm:.0f} dBm, Gaussian symbols"
    )
    fig.tight_layout()
    fig.savefig(args.out_dir / "s6_physical.png", dpi=pubstyle.dpi(200))
    plt.close(fig)
    lg.success(f"S6 saved to {args.out_dir}")


if __name__ == "__main__":
    main()
