"""Lorenzi Fast S0-syn: synthetic per-tuple FWM mass map, no real system.

Evaluates the fast-pass linear model :func:`linear_tuple_estimate` on a
by-hand grid of the natural, decorrelated coordinates

    x      = L B ||grad Delta_beta||_2 = sqrt(nu_a^2 + nu_b^2 + nu_c^2)
    mu = Delta_beta_center / (B ||grad Delta_beta||_2),  u0 = mu * x

so the per-tuple mass N*T^2/L^2 (x, mu) is decoupled from any channel plan,
tuple-count density, or real dispersion profile.  The mass also depends on

* how the walk-off splits among the three legs -- selected with
  ``--direction``: ``equal`` (nu_a = nu_b = nu_c = x/sqrt(3), default),
  ``sphere`` (average over random unit directions), ``single`` (all walk-off
  on one leg);
* the normalized support shift ``d`` (default 0, i.e. the combination lands
  on the target-channel center).

Outputs:
  * left panel: log10 F heatmap over (x, |mu|) with the exact
    unmasked-box classification boundary |u0| = W, i.e.
    |mu| = pi * ||nu||_1 / ||nu||_2 (pi*sqrt(3) for the equal split),
    below which the phase-matched surface crosses the admissible box;
  * right panel: N*T^2/L^2 vs |u0| at fixed x values (log-log), evaluated on a
    DENSE u0 grid (independent of the heatmap mesh) so the small-x coherent
    fringes render faithfully: for W <~ pi the decay is genuinely
    nonmonotonic (F inherits the finite-span 4 sin^2(u/2)/u^2 nulls at
    u0 = 2 pi k; verified: 160% relative oscillation at x = 1 with minima on
    the kernel nulls), washing out with the per-leg sinc contrast to ~1% by
    x ~ 100. NOTE: the heatmap itself remains coarsely sampled in u0, so
    any fringe-like pattern visible there at high |u0| is aliasing (moire),
    not the true fringe geometry.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger as lg

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pynlin  # noqa: F401

from analysis.log_init import init_logging
from pynlin.methods.td.fast_nlin import linear_tuple_estimate

# Equal-split coeffs (nu_a, nu_b, -nu_c) with ||coeffs||_2 = x: the FWM
# coefficient signs are (nu_a, nu_b, -nu_c) but only their magnitudes enter
# the widths; keep the production sign pattern for the mask model.
_EQUAL_DIR = np.array([1.0, 1.0, -1.0]) / np.sqrt(3.0)
_SINGLE_DIR = np.array([1.0, 0.0, 0.0])


def _directions(mode: str, n_directions: int, seed: int) -> np.ndarray:
    """Unit walk-off directions (n_dir, 3), sign pattern (+, +, -)."""
    if mode == "equal":
        return _EQUAL_DIR[None, :]
    if mode == "single":
        return _SINGLE_DIR[None, :]
    if mode == "sphere":
        rng = np.random.default_rng(seed)
        v = rng.normal(size=(n_directions, 3))
        v /= np.linalg.norm(v, axis=-1, keepdims=True)
        v[:, 2] *= -1.0
        return v
    raise ValueError(f"unknown direction mode: {mode}")


def synthetic_mass_map(
    x_grid: np.ndarray,
    mu_grid: np.ndarray,
    directions: np.ndarray,
    d: float,
) -> tuple[np.ndarray, np.ndarray]:
    """F and regime on the (x, mu) mesh, averaged over ``directions``.

    Returns arrays of shape ``(len(mu_grid), len(x_grid))`` matching a
    ``pcolormesh(x_grid, mu_grid, ...)`` orientation.
    """
    xx, mm = np.meshgrid(x_grid, mu_grid, indexing="xy")
    x_flat = xx.reshape(-1)
    u0 = (mm * xx).reshape(-1)
    n = x_flat.size
    acc = np.zeros(n, dtype=float)
    regime_acc = np.zeros(n, dtype=int)
    for s in directions:
        coeffs = x_flat[:, None] * s[None, :]
        est = linear_tuple_estimate(u0, coeffs, np.full(n, d))
        acc += est.values
        regime_acc |= est.regime
    f = (acc / directions.shape[0]).reshape(xx.shape)
    regime = regime_acc.reshape(xx.shape)
    return f, regime


def main() -> None:
    init_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("media/lorenzi-fast"))
    parser.add_argument("--x-min", type=float, default=1e-1)
    parser.add_argument("--x-max", type=float, default=1e4)
    parser.add_argument("--n-x", type=int, default=160)
    parser.add_argument("--mu-min", type=float, default=1e-2)
    parser.add_argument("--mu-max", type=float, default=1e3)
    parser.add_argument("--n-mu", type=int, default=120)
    parser.add_argument("--direction", choices=("equal", "sphere", "single"), default="equal")
    parser.add_argument("--n-directions", type=int, default=16)
    parser.add_argument("--d", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--s-min", type=float, default=1e-1)
    parser.add_argument("--s-max", type=float, default=3e3)
    parser.add_argument("--n-s", type=int, default=1600)
    parser.add_argument("--n-mu-dense", type=int, default=400)
    args = parser.parse_args()

    x_grid = np.logspace(np.log10(args.x_min), np.log10(args.x_max), args.n_x)
    mu_grid = np.logspace(np.log10(args.mu_min), np.log10(args.mu_max), args.n_mu)
    directions = _directions(args.direction, args.n_directions, args.seed)
    lg.info(
        f"synthetic map: {args.n_x}x{args.n_mu} grid over "
        f"x in [{args.x_min:g}, {args.x_max:g}], |mu| in "
        f"[{args.mu_min:g}, {args.mu_max:g}], direction={args.direction} "
        f"({directions.shape[0]} realization(s)), d={args.d:g}"
    )

    f, regime = synthetic_mass_map(x_grid, mu_grid, directions, args.d)

    finite = f[np.isfinite(f) & (f > 0)]
    lg.info(
        f"F range: [{finite.min():.3e}, {finite.max():.3e}] "
        f"({100 * np.mean(np.isfinite(f) & (f > 0)):.1f}% positive); "
        f"regimes near/far/wide = {np.mean(regime == 0):.1%}/"
        f"{np.mean(regime == 1):.1%}/{np.mean(regime == 2):.1%}"
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out_dir / "s0_synthetic.npz",
        x_grid=x_grid,
        mu_grid=mu_grid,
        F=f,
        regime=regime,
        directions=directions,
        d=args.d,
    )

    # +3pt on every numeric font size (the user matplotlibrc pins numeric
    # values for labels/ticks/legend, which a bare font.size bump misses).
    for key in (
        "font.size", "axes.labelsize", "axes.titlesize", "xtick.labelsize",
        "ytick.labelsize", "legend.fontsize", "legend.title_fontsize",
        "figure.titlesize",
    ):
        value = matplotlib.rcParams.get(key)
        if isinstance(value, (int, float)):
            matplotlib.rcParams[key] = value + 3.0
    fig, (ax, ax2, ax3, ax4, ax5, ax6) = plt.subplots(1, 6, figsize=(33.0, 4.8))
    img = np.log10(np.maximum(f, 1e-300))
    pcm = ax.pcolormesh(x_grid, mu_grid, img, cmap="viridis", shading="auto")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$x = LB\|\nabla\Delta\beta\|_2$ [rad]")
    ax.set_ylabel(
        r"$|\mu| = |\Delta\beta_{\rm center}|/(B\|\nabla\Delta\beta\|_2)$"
    )
    fig.colorbar(pcm, ax=ax, label=r"$\log_{10} N\,T^2\!/L^2$")
    # Exact unmasked-box boundary |u0| = W  <=>  |mu| = pi*||nu||_1/||nu||_2.
    l1_over_l2 = np.mean(np.sum(np.abs(directions), axis=-1))
    ax.axhline(np.pi * l1_over_l2, color="w", ls="--", lw=1.0, alpha=0.8)
    ax.text(
        1.25 * args.x_min,
        1.15 * np.pi * l1_over_l2,
        r"$|u_0|=W$",
        color="w",
    )
    ax.set_title(f"per-tuple mass $N\\,T^2/L^2$, direction={args.direction}, d={args.d:g}")

    n_slices = 6
    sel = np.unique(np.geomspace(1, args.n_x, n_slices).astype(int) - 1)
    colors = plt.cm.viridis(np.linspace(0.15, 0.9, sel.size))
    for j, c in zip(sel, colors):
        # Dense, slice-local u0 grid: the fringe period is 2*pi, far below
        # the heatmap's log-spaced resolution at high |u0|.
        mu_lo = args.mu_min * x_grid[j]
        mu_hi = args.mu_max * x_grid[j]
        n_dense = int(min(max(2000, 4 * (mu_hi - mu_lo) / (2 * np.pi)), 20000))
        mu_dense = np.geomspace(mu_lo, mu_hi, n_dense)
        est = linear_tuple_estimate(
            mu_dense,
            x_grid[j] * np.mean(directions, axis=0)[None, :] * np.ones((n_dense, 1)),
            np.full(n_dense, args.d),
        )
        ax2.plot(mu_dense, est.values, lw=0.9, color=c,
                 label=rf"$x={x_grid[j]:.3g}$")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel(r"$|u_0| = |\mu|\,x = L\,|\Delta\beta_{\rm center}|$ [rad]")
    ax2.set_ylabel(r"$N\,T^2\!/L^2$")
    ax2.legend(title="iso-$x$ slices")
    ax2.set_title(r"$N\,T^2\!/L^2$ vs $|u_0|$ at fixed $x$")

    # Third panel: the single-tuple-scaling cut, F(x) at fixed mu --
    # the exact coordinates of docs/source/fwm_single_tuple_scaling.md, so
    # the x^-1 (surface-crossing, |mu| below the boundary) and x^-2
    # (gapped) collapse laws are exhibited directly by the production model.
    boundary = np.pi * l1_over_l2
    mu_slices = (0.5, 2.0, boundary, 20.0, 100.0)
    colors3 = plt.cm.plasma(np.linspace(0.1, 0.8, len(mu_slices)))
    for md, c in zip(mu_slices, colors3):
        # Fringe period in x at fixed mu is 2*pi/mu (the high-u0
        # oscillations note, eq. 5): sample densely enough at low x.
        n_dense = int(min(max(3000, 4 * (args.x_max - args.x_min) * md / (2 * np.pi)),
                          20000))
        x_dense = np.geomspace(args.x_min, args.x_max, n_dense)
        est = linear_tuple_estimate(
            md * x_dense,
            x_dense[:, None] * np.mean(directions, axis=0)[None, :],
            np.full(n_dense, args.d),
        )
        lbl = (r"$|\mu| = \pi\|\nu\|_1/\|\nu\|_2$ (boundary)"
               if md == boundary else rf"$|\mu|={md:g}$")
        # Compensated coordinate of the single-tuple scaling note: its boxed
        # result is x*N(x) -> 2*pi*g(0) on the surface-crossing class, so
        # x*F renders the -1 law as a FLAT plateau (a far more sensitive
        # collapse test than a slope), while gapped tuples fall as
        # x*F ~ 2A/(mu^2 x) (thin dashed guides).
        ax3.plot(x_dense, x_dense * est.values, lw=0.9, color=c, label=lbl)
        if md > boundary:
            ax3.plot(x_dense, 2.0 * (2.0 / 3.0) / (md**2 * x_dense),
                     color=c, ls="--", lw=0.7, alpha=0.5)
    x_ref = np.geomspace(10.0, args.x_max, 50)
    ax3.plot(x_ref, 60.0 / x_ref, color="gray", ls=":", lw=1.2)
    ax3.text(x_ref[8], 90.0 / x_ref[8], r"$x\,N\,T^2\!/L^2 \propto x^{-1}$ (gapped)",
             color="gray")
    ax3.set_xscale("log")
    ax3.set_yscale("log")
    ax3.set_ylim(1e-8, 8.0)
    ax3.set_xlabel(r"$x = LB\|\nabla\Delta\beta\|_2$ [rad]")
    ax3.set_ylabel(r"$x\,N\,T^2\!/L^2$   (plateau $\Leftrightarrow$ $x^{-1}$ law)")
    ax3.legend(title=r"iso-$\mu$ slices", fontsize=matplotlib.rcParams["legend.fontsize"] - 1)
    ax3.set_title(r"compensated single-tuple collapse: $x\,N\,T^2\!/L^2$ vs $x$")

    # Fourth panel: combined-axis collapse, mirroring media/fwm/
    # mc-gradient-scaling/combined_collapse.pdf: every mesh point plotted
    # against s = x + |u0| = L(B||grad dbeta||_2 + |dbeta_0|). The
    # surface-crossing branch follows s^-1 and the gapped tail s^-2, so the
    # whole (x, mu) mesh falls onto one wedge between the two guide slopes.
    xx_c, mm_c = np.meshgrid(x_grid, mu_grid, indexing="xy")
    combined = (xx_c * (1.0 + mm_c)).reshape(-1)  # x + |u0|, u0 = mu * x
    f_flat = f.reshape(-1)
    good = np.isfinite(f_flat) & (f_flat > 0)
    sc = ax4.scatter(
        combined[good], f_flat[good], c=np.log10(mm_c.reshape(-1)[good]),
        s=6, cmap="plasma", alpha=0.7, linewidths=0,
    )
    fig.colorbar(sc, ax=ax4, label=r"$\log_{10}|\mu|$")
    s_guide = np.geomspace(max(combined[good].min(), 1e-2), combined[good].max(), 50)
    ax4.plot(s_guide, 2.0 / s_guide, "--", color="gray", lw=1.0, label=r"$\propto s^{-1}$")
    ax4.plot(s_guide, 8.0 / s_guide**2, ":", color="gray", lw=1.2, label=r"$\propto s^{-2}$")
    ax4.set_xscale("log")
    ax4.set_yscale("log")
    ax4.set_ylim(max(f_flat[good].min() * 0.5, 1e-14), 2.0)
    ax4.set_xlabel(r"$s = x + |u_0| = L(B\|\nabla\Delta\beta\|_2 + |\Delta\beta_0|)$ [rad]")
    ax4.set_ylabel(r"$N\,T^2/L^2$")
    ax4.legend()
    ax4.set_title("combined-axis collapse (cf. mc-gradient-scaling)")

    # Fifth panel: the same mass map as panel 1, but over (s, |mu|) with
    # s = x + |u0| = x (1 + |mu|). The (x, mu) mesh maps to a warped but
    # monotone quadrilateral mesh, which pcolormesh accepts directly as 2-D
    # coordinate arrays -- no rebinning. In these axes the surface-crossing
    # region (below the boundary) becomes a function of s alone (vertical
    # iso-mass stripes), which is the heatmap face of panel 4's collapse.
    s_2d = xx_c * (1.0 + mm_c)
    pcm5 = ax5.pcolormesh(s_2d, mm_c, img, cmap="viridis", shading="auto")
    fig.colorbar(pcm5, ax=ax5, label=r"$\log_{10} N\,T^2\!/L^2$")
    ax5.axhline(np.pi * l1_over_l2, color="w", ls="--", lw=1.0, alpha=0.8)
    ax5.text(
        1.25 * s_2d.min(),
        1.15 * np.pi * l1_over_l2,
        r"$|u_0|=W$",
        color="w",
    )
    ax5.set_xscale("log")
    ax5.set_yscale("log")
    ax5.set_xlabel(r"$s = x + |u_0| = L(B\|\nabla\Delta\beta\|_2 + |\Delta\beta_0|)$ [rad]")
    ax5.set_ylabel(r"$|\mu| = |\Delta\beta_{\rm center}|/(B\|\nabla\Delta\beta\|_2)$")
    ax5.set_title(r"per-tuple mass over $(s, |\mu|)$")

    # Sixth panel: dense RECTANGULAR grid directly in (s, |mu|) — no missing
    # wedges, every point computed from x = s/(1+|mu|), u0 = s|mu|/(1+|mu|).
    # Dense enough in s to resolve the cos(u0) fringes (period 2pi in u0);
    # since the fringe amplitude only survives where x <~ 3, i.e. mu >~ 1
    # where u0 ~= s, the fringes should appear as near-vertical stripes at
    # s ~= 2 pi k. Also saved standalone as s0_synthetic_smu_dense.png.
    s_grid = np.logspace(
        np.log10(args.s_min), np.log10(args.s_max), args.n_s
    )
    mu_g2 = np.logspace(np.log10(args.mu_min), np.log10(args.mu_max), args.n_mu_dense)
    ss6, mm6 = np.meshgrid(s_grid, mu_g2, indexing="xy")
    x6 = (ss6 / (1.0 + mm6)).reshape(-1)
    u06 = (ss6 * mm6 / (1.0 + mm6)).reshape(-1)
    dir6 = directions[0]
    f6 = np.empty(x6.size)
    chunk = 200_000
    for lo in range(0, x6.size, chunk):
        hi = min(lo + chunk, x6.size)
        coeffs6 = x6[lo:hi, None] * dir6[None, :]
        f6[lo:hi] = linear_tuple_estimate(
            u06[lo:hi], coeffs6, np.full(hi - lo, args.d)
        ).values
    img6 = np.log10(np.maximum(f6.reshape(ss6.shape), 1e-300))
    pcm6 = ax6.pcolormesh(s_grid, mu_g2, img6, cmap="viridis", shading="auto")
    fig.colorbar(pcm6, ax=ax6, label=r"$\log_{10} N\,T^2\!/L^2$")
    ax6.axhline(np.pi * l1_over_l2, color="w", ls="--", lw=1.0, alpha=0.8)
    ax6.set_xscale("log")
    ax6.set_yscale("log")
    ax6.set_xlabel(r"$s = x + |u_0|$ [rad]")
    ax6.set_ylabel(r"$|\mu|$")
    ax6.set_title(rf"dense $(s, |\mu|)$ grid ({args.n_s}$\times${args.n_mu_dense})")

    fig6, ax6b = plt.subplots(figsize=(9.5, 7.0))
    pcm6b = ax6b.pcolormesh(s_grid, mu_g2, img6, cmap="viridis", shading="auto")
    fig6.colorbar(pcm6b, ax=ax6b, label=r"$\log_{10} N\,T^2\!/L^2$")
    ax6b.axhline(np.pi * l1_over_l2, color="w", ls="--", lw=1.0, alpha=0.8)
    ax6b.set_xscale("log")
    ax6b.set_yscale("log")
    ax6b.set_xlabel(r"$s = x + |u_0| = L(B\|\nabla\Delta\beta\|_2 + |\Delta\beta_0|)$ [rad]")
    ax6b.set_ylabel(r"$|\mu| = |\Delta\beta_{\rm center}|/(B\|\nabla\Delta\beta\|_2)$")
    ax6b.set_title(rf"per-tuple mass, dense $(s, |\mu|)$ grid ({args.n_s}$\times${args.n_mu_dense})")
    fig6.tight_layout()
    fig6.savefig(args.out_dir / "s0_synthetic_smu_dense.png", dpi=250)
    plt.close(fig6)

    fig.suptitle("Lorenzi Fast S0-syn: synthetic FWM mass territory (by-hand variables)")
    fig.tight_layout()
    fig.savefig(args.out_dir / "s0_synthetic.png", dpi=200)
    plt.close(fig)

    lg.success(f"S0-syn saved to {args.out_dir}")


if __name__ == "__main__":
    main()
