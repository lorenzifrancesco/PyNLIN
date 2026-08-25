"""Lorenzi Fast S3 gate: certified tube + analytic branches vs reference.

For a set of probe targets on the FULL grid, sweeps the minimum admissible
efficiency epsilon and compares the analytic production path
(``target_analytic_sums``: epsilon-tube selection with exact discarded
certificate + sheet/far closed forms + quadrature fallback) against the
reference pipeline (``target_fast_sums``: exhaustive tuples, bulk quadrature
+ exact-acceptance refinement).

Reported per (target, epsilon): FWM sum ratio analytic/reference, survivor
fraction, certificate/kept ratio (the self-certified truncation error),
branch shares by count and by mass, and wall time for both paths.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pynlin  # noqa: F401
from loguru import logger as lg

from analysis.log_init import init_logging
from pynlin.methods.td.fast_analytic import target_analytic_sums
from pynlin.methods.td.fast_nlin import target_fast_sums
from pynlin.methods.td.fullband_mc import (
    _beta0_abs_from_fiber,
    decimated_frequency_grid,
)
from pynlin.system import System


def main() -> None:
    init_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("input/studies.toml"))
    parser.add_argument("--out-dir", type=Path, default=Path("media/lorenzi-fast"))
    parser.add_argument(
        "--targets", type=int, nargs="+", default=[0, 380, 1141, 1720, 2283],
        help="Full-grid probe targets (default: O edge, near-ZDW, mid-E, mid-C, U edge).",
    )
    parser.add_argument(
        "--epsilons", type=float, nargs="+",
        default=[1e-4, 1e-6, 1e-8, 1e-10],
    )
    args = parser.parse_args()

    system = System.from_toml(args.config)
    _, freqs = decimated_frequency_grid(system, 1)
    baud_rate = float(system.pulse.baud_rate)
    length = float(system.fiber_length)
    beta1_grid, beta2_grid = system.beta_grids(freqs=freqs)
    beta1 = np.asarray(beta1_grid[0], dtype=float)
    beta2 = np.asarray(beta2_grid[0], dtype=float)
    beta0_abs = _beta0_abs_from_fiber(system, freqs, beta1)

    rows = []
    for t in args.targets:
        t0 = time.perf_counter()
        ref = target_fast_sums(freqs, beta0_abs, beta1, beta2, baud_rate, length, int(t))
        t_ref = time.perf_counter() - t0
        lg.info(
            f"target {t} (f={freqs[t]*1e-12:.1f} THz): reference fwm={ref.fwm:.6e} "
            f"({ref.fwm_tuples} tuples, {t_ref:.1f} s)"
        )
        for eps in args.epsilons:
            t0 = time.perf_counter()
            ana = target_analytic_sums(
                freqs, beta0_abs, beta1, beta2, baud_rate, length, int(t),
                epsilon=eps,
            )
            t_ana = time.perf_counter() - t0
            ratio = ana.fwm / ref.fwm if ref.fwm > 0 else np.nan
            cert_rel = ana.certificate / ana.fwm if ana.fwm > 0 else np.inf
            kept_frac = ana.fwm_tuples_kept / max(ana.fwm_tuples_total, 1)
            bc, bm = ana.branch_counts, ana.branch_mass
            mass_tot = max(sum(bm), 1e-300)
            lg.info(
                f"  eps={eps:.0e}: fwm={ana.fwm:.6e} ratio={ratio:.4f} "
                f"kept={ana.fwm_tuples_kept}/{ana.fwm_tuples_total} "
                f"({kept_frac:.2e}) cert/kept={cert_rel:.2e} "
                f"branches sheet/far/fallback n=({bc[0]},{bc[1]},{bc[2]}) "
                f"mass=({bm[0]/mass_tot:.1%},{bm[1]/mass_tot:.1%},{bm[2]/mass_tot:.1%}) "
                f"t={t_ana:.1f}s (ref {t_ref:.1f}s)"
            )
            rows.append((
                t, eps, ref.fwm, ana.fwm, ratio, kept_frac, cert_rel,
                bc[0], bc[1], bc[2], bm[0], bm[1], bm[2], t_ref, t_ana,
            ))

    arr = np.array(rows, dtype=float)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out_dir / "s3_tube_gate.npz",
        rows=arr, targets=np.array(args.targets), epsilons=np.array(args.epsilons),
        columns=np.array(
            "target eps ref_fwm ana_fwm ratio kept_frac cert_rel "
            "n_sheet n_far n_fallback m_sheet m_far m_fallback t_ref t_ana".split()
        ),
    )

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4))
    for t in args.targets:
        sel = arr[:, 0] == t
        eps = arr[sel, 1]
        label = f"t={int(t)} ({freqs[int(t)]*1e-12:.0f} THz)"
        axes[0].semilogx(eps, arr[sel, 4], "o-", ms=4, label=label)
        axes[1].loglog(eps, arr[sel, 5], "o-", ms=4)
        axes[2].loglog(eps, np.maximum(arr[sel, 6], 1e-18), "o-", ms=4)
    axes[0].axhline(1.0, color="gray", lw=0.8, ls=":")
    axes[0].set_ylabel("FWM sum: analytic/reference")
    axes[1].set_ylabel("survivor fraction")
    axes[2].set_ylabel("certificate / kept sum")
    for ax in axes:
        ax.set_xlabel(r"$\varepsilon$ (min efficiency)")
        ax.grid(True, alpha=0.25)
    axes[0].legend(fontsize=7, frameon=False)
    fig.suptitle("S3 gate: certified tube + analytic branches vs reference")
    fig.tight_layout()
    fig.savefig(args.out_dir / "s3_tube_gate.png", dpi=200)
    plt.close(fig)
    lg.success(f"S3 gate saved to {args.out_dir}")


if __name__ == "__main__":
    main()
