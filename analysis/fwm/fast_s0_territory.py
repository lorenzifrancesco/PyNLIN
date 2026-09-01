"""Lorenzi Fast S0: map where FWM tuples live in the normalized variables.

For a set of target channels, computes (u0, nu, q, d) for every
support-surviving strict FWM tuple and reports where the estimated
contribution mass concentrates, both by tuple count and by fast-pass weight.

Plots both the raw (|u0|, W) view (u0 = L*Delta_beta_center, W = pi*sum|nu_j|,
as used internally by the regime dispatch) and the natural, decorrelated
coordinates from the prior single-tuple scaling derivation
(docs/source/fwm_dispersion_scales_and_coordinates.md).

Tuples are always enumerated on the FULL channel grid: decimating the grid
biases the census toward near-exact zero-sum combinations (at decimation 8,
59% of surviving tuples had |d| < 0.01) and makes the population that of a
coarser-spaced system, not the real one. Only the number of *target*
channels is a knob here.

Mass panels are PER-TARGET NORMALIZED (each target's F sums to 1 before
histogramming): per-target masses span many orders of magnitude across the
band (a near-ZDW target once carried 99.7% of the raw overlay), so without
normalization the "mass" panels degenerate into a single target's territory.

Rendering: 2-D log-log histograms on data-driven bin ranges (so no tuple is
ever silently dropped from a panel) with a light Gaussian smoothing in bin
units before the log10 color mapping -- the territory is sparse (a few % of
bins occupied, most of each target's mass in its top ~10-100 tuples), so raw
bins render as isolated pixels.

    x       = L*B*||grad Delta_beta||_2 = sqrt(nu_a^2 + nu_b^2 + nu_c^2)
    mu  = Delta_beta_center / (B*||grad Delta_beta||_2) = u0 / x

so that L*Delta_beta_center = mu * x exactly. Since W ~ x up to an L1/L2
norm-conversion factor, the raw (u0, W) axes are entangled -- u0 already
contains x as a multiplicative factor -- which is almost certainly why the
raw territory histogram shows a strong diagonal correlation band rather than
revealing independent structure. (x, mu) separates "how loud is this
tuple" from "how close to phase-matched", and mu directly exposes the
exact classification boundary |mu| = W/x (equivalently |u0| = W) derived
in docs/source/fwm_single_tuple_scaling.md for whether the phase-matched
surface crosses the admissible (unmasked) box.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pynlin  # noqa: F401
from loguru import logger as lg

from analysis.log_init import init_logging
from pynlin.methods.td.fast_nlin import (
    fwm_tuple_variables,
    linear_tuple_estimate,
)
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
    parser.add_argument("--n-targets", type=int, default=7)
    parser.add_argument(
        "--replot", type=Path, default=None,
        help="re-render the figure from an existing s0_territory.npz and exit, "
             "without recomputing the census (leaves the npz untouched)",
    )
    args = parser.parse_args()

    if args.replot is not None:
        replot_from_npz(args.replot, args.out_dir)
        return

    system = System.from_toml(args.config)
    # Full grid always: interferer decimation changes the tuple population.
    _, freqs = decimated_frequency_grid(system, 1)
    baud_rate = float(system.pulse.baud_rate)
    length = float(system.fiber_length)
    beta1_grid, beta2_grid = system.beta_grids(freqs=freqs)
    beta1 = np.asarray(beta1_grid[0], dtype=float)
    beta2 = np.asarray(beta2_grid[0], dtype=float)
    beta0_abs = _beta0_abs_from_fiber(system, freqs, beta1)

    n = freqs.size
    targets = np.unique(np.linspace(0, n - 1, args.n_targets).astype(int))
    lg.info(f"{n} channels (full grid), targets={targets}")

    all_u0, all_W, all_sigma, all_q, all_d, all_F, all_regime, all_target = (
        [], [], [], [], [], [], [], []
    )
    for t in targets:
        v = fwm_tuple_variables(freqs, beta0_abs, beta1, beta2, baud_rate, length, t)
        if v.u0.size == 0:
            lg.warning(f"target {t}: no surviving tuples")
            continue
        coeffs = np.stack([v.nu_a, v.nu_b, -v.nu_c], axis=-1)
        est = linear_tuple_estimate(v.u0, coeffs, v.d)
        W = np.sum(v.widths, axis=-1)
        q_eff = np.abs(v.q_a) + np.abs(v.q_b) + np.abs(v.q_c) + abs(v.q_t)
        all_u0.append(v.u0)
        all_W.append(W)
        all_sigma.append(v.sigma)
        all_q.append(q_eff)
        all_d.append(v.d)
        all_F.append(est.values)
        all_regime.append(est.regime)
        all_target.append(np.full(v.u0.size, t, dtype=int))
        top = np.argsort(est.values)[-10:][::-1]
        mass = np.sum(est.values)
        top_share = np.sum(est.values[top]) / mass if mass > 0 else 0.0
        lg.info(
            f"target {t} (f={freqs[t]*1e-12:.1f} THz): {v.u0.size} tuples, "
            f"F_sum={mass:.3e}, top-10 share={top_share:.1%}, "
            f"regimes near/far/wide="
            f"{np.sum(est.regime == 0)}/{np.sum(est.regime == 1)}/{np.sum(est.regime == 2)}"
        )

    u0 = np.concatenate(all_u0)
    W = np.concatenate(all_W)
    sigma = np.concatenate(all_sigma)
    q_eff = np.concatenate(all_q)
    d = np.concatenate(all_d)
    F = np.concatenate(all_F)
    regime = np.concatenate(all_regime)
    target_of = np.concatenate(all_target)

    # Per-target normalized weights: each target's mass sums to 1, so the
    # overlaid mass panels and aggregate statistics weight every target
    # equally instead of being dominated by the loudest (near-ZDW) one.
    F_norm = F.copy()
    for tt in np.unique(target_of):
        sel = target_of == tt
        total_t = float(np.sum(F[sel]))
        if total_t > 0:
            F_norm[sel] = F[sel] / total_t

    # Natural, decorrelated coordinates: x = L B ||grad Delta_beta||_2 =
    # sqrt(nu_a^2+nu_b^2+nu_c^2). FWMTupleVariables.sigma = sqrt(sum(widths^2)/3)
    # with widths_j = pi|nu_j|, so sigma = (pi/sqrt(3)) * x -- no need to
    # recompute nu_a/nu_b/nu_c separately.
    x_grad = sigma * (np.sqrt(3.0) / np.pi)
    mu = np.divide(u0, x_grad, out=np.zeros_like(u0), where=x_grad > 1e-300)

    # Aggregate statistics use the per-target normalized weights so every
    # target counts equally (raw masses differ by orders of magnitude).
    mass_total = np.sum(F_norm)
    for name, sel in (
        ("near", regime == 0),
        ("far", regime == 1),
        ("wide", regime == 2),
    ):
        lg.info(
            f"regime {name}: {np.sum(sel)} tuples ({np.mean(sel):.1%}), "
            f"normalized mass share {np.sum(F_norm[sel]) / mass_total:.2%}"
        )
    # Mass-ranked cumulative curve: how many tuples carry 50/90/99% of the sum.
    order = np.argsort(F_norm)[::-1]
    cum = np.cumsum(F_norm[order]) / mass_total
    for level in (0.5, 0.9, 0.99):
        k = int(np.searchsorted(cum, level)) + 1
        lg.info(
            f"{level:.0%} of the normalized mass in the top {k} tuples "
            f"({k / F.size:.2e} of all)"
        )
    small_q = q_eff < 0.5
    lg.info(
        f"tuples with q_eff < 0.5 (quadratics negligible): {np.mean(small_q):.1%}; "
        f"their normalized mass share {np.sum(F_norm[small_q]) / mass_total:.2%}"
    )
    # Exact (unmasked-box) phase-matched-surface-crosses-domain classification:
    # |u0| < W. This is a NECESSARY condition for the mask-restricted domain
    # too (the true admissible region is a subset of the box), so |u0| > W
    # always means gapped/far-asymptotic; |u0| < W only means "maybe near".
    crosses = np.abs(u0) < W
    lg.info(
        f"exact unmasked-box test |u0|<W: {np.mean(crosses):.1%} of tuples, "
        f"normalized mass share {np.sum(F_norm[crosses]) / mass_total:.2%} "
        f"(equivalently |mu| < W/x, a per-tuple-direction threshold)"
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out_dir / "s0_territory.npz",
        u0=u0, W=W, sigma=sigma, q_eff=q_eff, d=d, F=F, F_norm=F_norm,
        regime=regime, target_of=target_of, targets=targets, freqs=freqs,
        decimation=1, x_grad=x_grad, mu=mu,
    )

    plot_territory(u0, W, F_norm, x_grad, mu, args.out_dir)
    lg.success(f"S0 saved to {args.out_dir}")


def _set_readable_fonts() -> None:
    """Keep labels legible after the multi-panel image is scaled in docs."""
    matplotlib.rcParams.update({
        "font.size": 14,
        "axes.labelsize": 15,
        "axes.titlesize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 13,
        "legend.title_fontsize": 13,
        "figure.titlesize": 17,
        "xtick.major.size": 5,
        "ytick.major.size": 5,
        "xtick.major.width": 1,
        "ytick.major.width": 1,
    })


# Histogram resolution (x-bins, y-bins) per panel.
_N_BINS = (140, 110)
# Absolute floor for log-spaced axes: zero / sub-floor values are clipped into
# the first bin instead of being dropped by histogram2d.
_VALUE_FLOOR = 1e-6


def _log_edges(values: np.ndarray, n_bins: int, *, floor: float = _VALUE_FLOOR) -> np.ndarray:
    """Log-spaced edges covering pre-floored ``values`` exactly.

    Both ends carry a small headroom factor. Unlike fixed ranges this can
    never silently drop a sample (the previous hard ``|mu| <= 10**2.5``
    cap discarded ~77% of the tuples from the natural-axes panels).
    """
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    lo = vmin / 1.05 if vmin > floor else floor
    hi = vmax * 1.05
    if not np.isfinite(hi) or hi <= lo:
        hi = lo * 10.0
    return np.logspace(np.log10(lo), np.log10(hi), n_bins + 1)


def _territory_panel(
    ax: Axes,
    xs: np.ndarray,
    ys: np.ndarray,
    weights: np.ndarray | None,
    *,
    title: str,
) -> None:
    xs = np.maximum(np.asarray(xs, dtype=float), _VALUE_FLOOR)
    ys = np.maximum(np.asarray(ys, dtype=float), _VALUE_FLOOR)
    xe = _log_edges(xs, _N_BINS[0])
    ye = _log_edges(ys, _N_BINS[1])
    h_count, _, _ = np.histogram2d(xs, ys, bins=[xe, ye])
    if h_count.sum() != xs.size:
        lg.warning(f"{title}: {xs.size - h_count.sum():d} samples fell outside the bin range")
    h = h_count
    if weights is not None:
        h, _, _ = np.histogram2d(xs, ys, bins=[xe, ye], weights=weights)
    img = np.log10(np.maximum(h.T, 1e-300))
    vmax = float(img.max())
    vmin = max(vmax - 6.0, -12.0)
    pcm = ax.pcolormesh(xe, ye, img, cmap="viridis", vmin=vmin, vmax=vmax)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(title)
    fig = ax.get_figure()
    if fig is not None:
        fig.colorbar(pcm, ax=ax, label=r"$\log_{10}$")


def _overlay_phase_regions(ax) -> None:
    """Figure 10's demarcations, which are straight in (x_grad, |u0|).

    The data limits are restored afterwards: the coherence line x_grad = 1 sits
    outside the populated range on this grid, and letting it stretch the axis
    would open an empty strip.
    """
    x_lim, y_lim = ax.get_xlim(), ax.get_ylim()
    x_line = np.geomspace(max(x_lim[0], 1e-12), x_lim[1], 300)
    ax.plot(x_line, np.pi * x_line / np.sqrt(3.0), color="w", lw=1.4)
    ax.plot(x_line, np.pi * np.sqrt(3.0) * x_line, color="w", lw=1.0, ls=":")
    if x_lim[0] <= 1.0 <= x_lim[1]:
        ax.axvline(1.0, color="w", lw=1.0, ls="--", alpha=0.8)
    ax.set_xlim(*x_lim)
    ax.set_ylim(*y_lim)


def plot_territory(
    u0: np.ndarray,
    W: np.ndarray,
    F: np.ndarray,
    x_grad: np.ndarray,
    mu: np.ndarray,
    out_dir: Path,
) -> None:
    """Render the six territory panels from per-tuple arrays.

    Also usable standalone to re-render ``s0_territory.png`` from a saved
    ``s0_territory.npz`` without recomputing the tuples -- see ``--replot``.

    Rows, in the coordinates of doc §4.1:
      0  raw       (|u0|, W)        -- both axes carry x_grad, hence entangled
      1  fundamental (x_grad, |u0|) -- the axes of Figure 10; the population is
                                       just as correlated here (u0 = mu x_grad),
                                       but the region boundaries are rays, so
                                       the census can be read against them
      2  derived   (x_grad, |mu|)   -- the pair that actually decorrelates
    """
    if u0.size == 0:
        lg.warning("plot_territory: no tuples to plot")
        return
    _set_readable_fonts()
    fig, axes = plt.subplots(3, 2, figsize=(9.4, 11.1))
    abs_u0 = np.abs(u0)
    abs_mu = np.abs(mu)

    _territory_panel(axes[0, 0], abs_u0, W, None, title="tuple count (raw axes)")
    _territory_panel(
        axes[0, 1], abs_u0, W, F,
        title="fast-pass mass, per-target normalized (raw axes)",
    )
    for ax in axes[0]:
        ax.set_xlabel(r"$|u_0| = L\,|\Delta\beta_0|$ [rad]")
    axes[0, 0].set_ylabel(r"$W = \pi(|\nu_a|+|\nu_b|+|\nu_c|)$ [rad]")

    _territory_panel(axes[1, 0], x_grad, abs_u0, None,
                     title="tuple count (fundamental axes)")
    _territory_panel(
        axes[1, 1], x_grad, abs_u0, F,
        title="fast-pass mass, per-target normalized (fundamental)",
    )
    for ax in axes[1]:
        ax.set_xlabel(r"$x_\nabla = LB\|\nabla\Delta\beta\|_2$ [rad]")
        _overlay_phase_regions(ax)
    axes[1, 0].set_ylabel(r"$|u_0| = L\,|\Delta\beta_0|$ [rad]")

    _territory_panel(axes[2, 0], x_grad, abs_mu, None,
                     title="tuple count (decorrelating axes)")
    _territory_panel(
        axes[2, 1], x_grad, abs_mu, F,
        title="fast-pass mass, per-target normalized (decorrelating)",
    )
    for ax in axes[2]:
        ax.set_xlabel(r"$x_\nabla = LB\|\nabla\Delta\beta\|_2$ [rad]")
    axes[2, 0].set_ylabel(r"$|\mu| = |u_0| / x_\nabla$")

    fig.suptitle(
        f"Lorenzi Fast S0: FWM tuple territory (full grid, {u0.size:,} tuples, "
        "mass per-target normalized)\n"
        "raw $(|u_0|,W)$ -- fundamental $(x_\\nabla,|u_0|)$ with the Figure 10 "
        "rays -- decorrelating $(x_\\nabla,|\\mu|)$"
    )
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "s0_territory.png", dpi=200)
    plt.close(fig)


def replot_from_npz(npz_path: Path, out_dir: Path) -> None:
    """Re-render the figure from a saved census, without recomputing tuples."""
    lg.info(f"re-plotting from {npz_path}")
    with np.load(npz_path, allow_pickle=False) as z:
        u0 = z["u0"]; W = z["W"]; F = z["F_norm"]
        x_grad = z["x_grad"]; mu = z["mu"]
        lg.info(f"loaded {u0.size:,} tuples, {np.unique(z['target_of']).size} targets")
        plot_territory(u0, W, F, x_grad, mu, out_dir)
    lg.success(f"S0 territory figure re-rendered into {out_dir}")


if __name__ == "__main__":
    main()
