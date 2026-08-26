"""Dispatch-plane census: tuple populations over the regime-dispatch regions.

For each probe target, plots 2-D histograms in the (W, |u0|) plane -- W the
total in-band width sum and |u0| the center mismatch -- of (a) all
support-surviving tuples,
(b) the same tuples weighted by their bulk-model FWM value (where the mass
lives), and (c) the epsilon-tube survivors, overlaid with the dispatch
boundaries of ``linear_tuple_estimate`` / ``analytic_tuple_values``:

* far:   |u0| = 3 W + 3000
* wide:  W = 3000
* sheet: W > 2000 and |u0| < W - 200

The current tube selector is mask-aware and also depends on the signed
projection of the mismatch coefficients and on d.  It therefore has no
single boundary in this projection; survivor points are plotted directly.

Companion figure to docs/source/lorenzi_fast_cost_anatomy.md.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pynlin  # noqa: F401
from loguru import logger as lg

from analysis.log_init import init_logging
from pynlin.methods.td.fast_analytic import (
    masked_linear_phase_outer_interval,
    select_tube,
)
from pynlin.methods.td.fast_nlin import (
    fwm_tuple_variables,
    linear_tuple_estimate,
)
from pynlin.methods.td.fullband_mc import (
    _beta0_abs_from_fiber,
    decimated_frequency_grid,
)
from pynlin.system import System

FLOOR = 1e-1  # rad; log-plane floor for W and |u0|


def _hist(logw, logu, xedges, yedges, weights=None):
    h, _, _ = np.histogram2d(logw, logu, bins=(xedges, yedges), weights=weights)
    return h


def _overlay(ax):
    w = np.logspace(np.log10(FLOOR), 6.5, 200)
    ax.plot(w, 3.0 * w + 3000.0, color="#D55E00", lw=1.4, label=r"far: $|u_0|=3W+3000$")
    ax.axvline(3000.0, color="#0072B2", lw=1.4, label=r"wide: $W=3000$")
    ws = w[w > 2000.0]
    ax.plot(
        ws, ws - 200.0, color="#009E73", lw=1.4, ls="-.", label=r"sheet: $|u_0|=W-200$"
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(FLOOR, 3e6)
    ax.set_ylim(FLOOR, 3e7)


def main() -> None:
    init_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("input/studies.toml"))
    parser.add_argument("--out-dir", type=Path, default=Path("media/lorenzi-fast"))
    parser.add_argument("--targets", type=int, nargs="+", default=[380, 1720])
    parser.add_argument("--epsilon", type=float, default=1e-6)
    args = parser.parse_args()

    system = System.from_toml(args.config)
    _, freqs = decimated_frequency_grid(system, 1)
    baud_rate = float(system.pulse.baud_rate)
    length = float(system.fiber_length)
    beta1_grid, beta2_grid = system.beta_grids(freqs=freqs)
    beta1 = np.asarray(beta1_grid[0], dtype=float)
    beta2 = np.asarray(beta2_grid[0], dtype=float)
    beta0_abs = _beta0_abs_from_fiber(system, freqs, beta1)

    n_t = len(args.targets)
    fig, axes = plt.subplots(n_t, 3, figsize=(12.5, 3.9 * n_t), squeeze=False)
    xedges = np.linspace(np.log10(FLOOR), 6.5, 170)
    yedges = np.linspace(np.log10(FLOOR), 7.5, 170)

    for row, t in enumerate(args.targets):
        t = int(t)
        v = fwm_tuple_variables(freqs, beta0_abs, beta1, beta2, baud_rate, length, t)
        coeffs = np.stack([v.nu_a, v.nu_b, -v.nu_c], axis=-1)
        est = linear_tuple_estimate(v.u0, coeffs, v.d)
        keep, _ = select_tube(v, args.epsilon)
        lower, upper = masked_linear_phase_outer_interval(v)
        mask_outer_crossing = (lower <= 0.0) & (upper >= 0.0)
        W = np.maximum(np.sum(v.widths, axis=-1), FLOOR)
        U = np.maximum(np.abs(v.u0), FLOOR)
        logw, logu = np.log10(W), np.log10(U)
        mass_frac = np.sum(est.values[keep]) / np.sum(est.values)
        lg.info(
            f"target {t}: {v.u0.size} tuples, {keep.size} survivors "
            f"({mass_frac:.4%} of bulk mass inside the tube), "
            f"{np.sum(mask_outer_crossing)} mask-outer-interval crossings"
        )

        panels = (
            (
                f"all tuples (n={v.u0.size / 1e6:.1f}M)",
                _hist(logw, logu, xedges, yedges),
            ),
            ("bulk-model FWM mass", _hist(logw, logu, xedges, yedges, est.values)),
            (
                f"tube survivors (n={keep.size})",
                _hist(logw[keep], logu[keep], xedges, yedges),
            ),
        )
        for col, (title, h) in enumerate(panels):
            ax = axes[row][col]
            hm = np.ma.masked_less_equal(h, 0.0)
            ax.pcolormesh(
                10.0**xedges,
                10.0**yedges,
                hm.T,
                norm=LogNorm(vmin=max(hm.min(), hm.max() * 1e-9), vmax=hm.max()),
                cmap="Blues",
                rasterized=True,
            )
            _overlay(ax)
            ax.set_title(f"t={t} ({freqs[t] * 1e-12:.1f} THz): {title}", fontsize=9)
            if col == 0:
                ax.set_ylabel(r"$|u_0|$ [rad]")
            ax.set_xlabel(r"$W = \sum_j w_j$ [rad]")
            ax.grid(True, which="major", alpha=0.15)
        if row == 0:
            axes[0][0].legend(fontsize=6.5, frameon=False, loc="upper left")

    fig.suptitle(
        "Dispatch regions vs tuple population, FWM mass, and tube survivors",
        fontsize=11,
    )
    fig.tight_layout()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_dir / "dispatch_regions.png", dpi=200)
    plt.close(fig)
    lg.success(f"dispatch-plane census saved to {args.out_dir}")


if __name__ == "__main__":
    main()
