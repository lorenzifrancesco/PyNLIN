"""Lorenzi Fast S4: fast per-target sums vs the repo exhaustive-support MC.

Runs ``compute_fullband_prefactor_free_mc`` (exhaustive support tuples, MC
over in-channel frequencies) at probe targets and compares against the fast
path (analytic regimes + top-K QMC refinement), component by component.
Also reports wall-time for both paths.

Decimation note: both sides see the SAME decimated grid, so the fast-vs-MC
ratios are a valid estimator check at any decimation -- but the absolute
per-target values at decimation > 1 are those of a coarser-spaced system,
not the real one (interferer decimation changes the physics). Use
--decimation 1 for physically quotable values.
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
from pynlin.methods.td.fast_nlin import target_fast_sums
from pynlin.methods.td.fullband_mc import (
    _beta0_abs_from_fiber,
    compute_fullband_prefactor_free_mc,
    decimated_frequency_grid,
)
from pynlin.system import System


def main() -> None:
    init_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("input/studies.toml"))
    parser.add_argument("--out-dir", type=Path, default=Path("media/lorenzi-fast"))
    parser.add_argument("--decimation", type=int, default=8)
    parser.add_argument("--n-targets", type=int, default=7)
    parser.add_argument("--fwm-frequency-samples", type=int, default=200)
    parser.add_argument("--xpm-samples", type=int, default=200_000)
    parser.add_argument("--n-refine", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    system = System.from_toml(args.config)
    _, freqs = decimated_frequency_grid(system, args.decimation)
    baud_rate = float(system.pulse.baud_rate)
    length = float(system.fiber_length)
    beta1_grid, beta2_grid = system.beta_grids(freqs=freqs)
    beta1 = np.asarray(beta1_grid[0], dtype=float)
    beta2 = np.asarray(beta2_grid[0], dtype=float)
    beta0_abs = _beta0_abs_from_fiber(system, freqs, beta1)
    n = freqs.size
    targets = np.unique(np.linspace(0, n - 1, args.n_targets).astype(int))
    lg.info(f"decimation={args.decimation}: {n} channels, probes={targets}")

    t0 = time.perf_counter()
    fast_rows = []
    for t in targets:
        res = target_fast_sums(
            freqs, beta0_abs, beta1, beta2, baud_rate, length, int(t),
            n_refine=args.n_refine,
        )
        fast_rows.append(res)
    t_fast = time.perf_counter() - t0
    lg.info(f"fast path: {t_fast:.2f} s for {targets.size} targets")

    t0 = time.perf_counter()
    diag = compute_fullband_prefactor_free_mc(
        system,
        decimation=args.decimation,
        target_indices=targets,
        include_xpm=True,
        include_fwm=True,
        xpm_samples=args.xpm_samples,
        fwm_frequency_samples=args.fwm_frequency_samples,
        fwm_tuple_selection="exhaustive_support_mc",
        n_workers=args.workers,
    )
    t_mc = time.perf_counter() - t0
    lg.info(f"MC reference: {t_mc:.2f} s ({t_mc / max(t_fast, 1e-9):.0f}x fast path)")

    L2 = length**2
    rows = []
    for i, t in enumerate(targets):
        fast = fast_rows[i]
        xpm_fast = fast.xpm * L2
        fwm_fast = fast.fwm * L2
        xpm_mc = diag.xpm[i]
        fwm_mc = diag.fwm[i]
        rows.append((t, xpm_fast, xpm_mc, fwm_fast, fwm_mc))
        lg.info(
            f"target {t:>4} (f={freqs[t]*1e-12:.1f} THz): "
            f"XPM fast={xpm_fast:.4e} MC={xpm_mc:.4e} ratio={xpm_fast/xpm_mc:.3f} | "
            f"FWM fast={fwm_fast:.4e} MC={fwm_mc:.4e} ratio={fwm_fast/fwm_mc:.3f} "
            f"(tuples={fast.fwm_tuples}, refined mass={fast.refined_mass_fraction:.0%})"
        )
    arr = np.array(rows, dtype=float)
    xr = arr[:, 1] / arr[:, 2]
    fr = arr[:, 3] / arr[:, 4]
    lg.info(
        f"XPM fast/MC: mean={np.mean(xr):.3f}, worst={xr[np.argmax(np.abs(xr-1))]:.3f}; "
        f"FWM fast/MC: mean={np.mean(fr):.3f}, worst={fr[np.argmax(np.abs(fr-1))]:.3f}"
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out_dir / "s4_targets.npz",
        rows=arr, freqs=freqs, targets=targets,
        t_fast=t_fast, t_mc=t_mc, decimation=args.decimation,
    )
    plot_s4_targets(arr, freqs, targets, t_fast, t_mc, args.out_dir)
    lg.success(f"S4 saved to {args.out_dir}")


def plot_s4_targets(
    rows: np.ndarray,
    freqs: np.ndarray,
    targets: np.ndarray,
    t_fast: float,
    t_mc: float,
    out_dir: Path,
) -> None:
    """Render the fast-vs-MC per-target comparison.

    ``rows`` columns: target, xpm_fast, xpm_mc, fwm_fast, fwm_mc.  Works
    standalone on a saved ``s4_targets.npz``.
    """
    rows = np.asarray(rows, dtype=float)
    f_thz = np.asarray(freqs)[np.asarray(targets, dtype=int)] * 1e-12
    xpm_fast, xpm_mc = rows[:, 1], rows[:, 2]
    fwm_fast, fwm_mc = rows[:, 3], rows[:, 4]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6))

    ax.plot(f_thz, xpm_fast, "o-", ms=5, lw=1.2, color="tab:blue", label="XPM fast")
    ax.plot(f_thz, xpm_mc, "o", ms=7, mfc="none", color="tab:blue", label="XPM MC")
    ax.plot(f_thz, fwm_fast, "s-", ms=5, lw=1.2, color="tab:red", label="FWM fast")
    ax.plot(f_thz, fwm_mc, "s", ms=7, mfc="none", color="tab:red", label="FWM MC")
    ax.set_yscale("log")
    ax.set_xlabel("target frequency [THz]")
    ax.set_ylabel(r"prefactor-free sum [m$^2$]")
    ax.set_title("per-target sums: fast path vs exhaustive-support MC")
    ax.legend(fontsize=8)

    xr = xpm_fast / np.maximum(xpm_mc, 1e-300)
    fr = fwm_fast / np.maximum(fwm_mc, 1e-300)
    ax2.axhspan(0.9, 1.1, color="gray", alpha=0.2, lw=0)
    ax2.axhline(1.0, color="black", lw=0.8)
    ax2.plot(f_thz, xr, "o-", ms=5, lw=1.0, color="tab:blue", label="XPM fast/MC")
    ax2.plot(f_thz, fr, "s-", ms=5, lw=1.0, color="tab:red", label="FWM fast/MC")
    ax2.set_xlabel("target frequency [THz]")
    ax2.set_ylabel("fast / MC ratio")
    ax2.set_title("agreement per target (band: $\\pm$10%)")
    ax2.legend(fontsize=8)
    worst = max(np.max(np.abs(np.log(xr))), np.max(np.abs(np.log(fr))))
    ax2.set_ylim(
        max(np.exp(-3.0 * worst), 0.5), min(np.exp(3.0 * worst), 2.0)
    )

    fig.suptitle(
        f"Lorenzi Fast S4: estimator validation on probe targets "
        f"(MC reference {t_mc / max(t_fast, 1e-9):.0f}x slower than fast path)"
    )
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "s4_targets.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
