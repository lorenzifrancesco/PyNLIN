"""Assessment figures for the (x_grad, |u0|) four-region phase diagram (doc §10.2).

Validates the closed-form region laws of lorenzi_fast_method.md §10.2 against
the exact linear model (equal split, d = 0, hard-mask window |v| <= w), in the
intrinsic coordinates

  x_grad = ||nu||_2   (walk-off spread sampled across the legs)
  u0                  (center accumulated mismatch phase)

with w = pi x_grad / sqrt3.  In these coordinates the four laws are

  region 1 (coherent plateau):  N T^2/L^2 = 2/3,   |u0| + pi x/sqrt3 <~ pi
  region 2 (sheet):             (3 sqrt3 / 4x) (1 - u0^2/(pi^2 x^2)),  |u0| < w
  region 3 (gapped, dephased):  4/(3 u0^2)   (fringe-averaged; x drops out)
  region 4 (gapped, coherent):  (2/3) Khat(u0),  x <~ 1

and every boundary is a straight line: the plateau edge |u0| + pi x/sqrt3 = pi,
the sheet/gap ray |u0| = pi x/sqrt3, the coherence line x = 1, and the unmasked
reference ray |u0| = W = pi sqrt3 x.

Outputs (media/lorenzi-fast/ and, if present, docs/source/_static/lorenzi-fast/):
  xu0_phase_diagram.png -- cmaps: exact model | piecewise prediction | ratio,
                           with region boundaries and labels
  xu0_phase_cuts.png    -- line plots: iso-x cuts vs |u0| + laws; compensated
                           u0^2 N; iso-|u0| cuts vs x; region-4 fringe zoom;
                           fringe contrast vs x; plateau-edge collapse
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

# These wide figures are substantially downscaled in the documentation.
# Explicit values also override tiny numeric sizes pinned by matplotlibrc.
matplotlib.rcParams.update({
    "font.size": 16,
    "axes.labelsize": 17,
    "axes.titlesize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 14,
    "legend.title_fontsize": 14,
    "figure.titlesize": 20,
    "xtick.major.size": 6,
    "ytick.major.size": 6,
    "xtick.major.width": 1.2,
    "ytick.major.width": 1.2,
})
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pynlin  # noqa: F401
from loguru import logger as lg

from analysis.log_init import init_logging
from pynlin.methods.td.fast_nlin import linear_tuple_estimate

_EQUAL_DIR = np.array([1.0, 1.0, -1.0]) / np.sqrt(3.0)
SLOPE_23 = np.pi / np.sqrt(3.0)        # |u0| = SLOPE_23 * x : masked sheet/gap ray
SLOPE_UNMASKED = np.pi * np.sqrt(3.0)  # |u0| = W : unmasked box reference
PLATEAU = 2.0 / 3.0


def model(x: np.ndarray, u0: np.ndarray, chunk: int = 200_000) -> np.ndarray:
    """Exact linear-model N T^2/L^2 at (x_grad, u0), equal split, d = 0."""
    x = np.asarray(x, float).reshape(-1)
    u0 = np.asarray(u0, float).reshape(-1)
    out = np.empty(x.size)
    for lo in range(0, x.size, chunk):
        hi = min(lo + chunk, x.size)
        coeffs = x[lo:hi, None] * _EQUAL_DIR[None, :]
        out[lo:hi] = linear_tuple_estimate(u0[lo:hi], coeffs, np.zeros(hi - lo)).values
    return out


def khat(u: np.ndarray) -> np.ndarray:
    u = np.asarray(u, float)
    out = np.ones_like(u)
    nz = np.abs(u) > 1e-12
    out[nz] = 4.0 * np.sin(u[nz] / 2.0) ** 2 / u[nz] ** 2
    return out


def plateau_edge(x: np.ndarray) -> np.ndarray:
    """Region-1 edge |u0| = pi - pi x/sqrt3 (empty for x > sqrt3)."""
    return np.pi * (1.0 - np.asarray(x, float) / np.sqrt(3.0))


def prediction(x: np.ndarray, u0: np.ndarray) -> np.ndarray:
    """Piecewise four-region closed form (region 3 fringe-averaged)."""
    x = np.asarray(x, float)
    u0 = np.abs(np.asarray(u0, float))
    sheet = (3.0 * np.sqrt(3.0) / (4.0 * x)) * (1.0 - u0**2 / (np.pi**2 * x**2))
    smooth3 = 4.0 / (3.0 * np.maximum(u0, 1e-300) ** 2)
    fringe4 = PLATEAU * khat(u0)
    gapped = np.where(x < 1.0, fringe4, smooth3)
    pred = np.where(u0 < SLOPE_23 * x, sheet, gapped)
    return np.minimum(pred, PLATEAU)


def draw_boundaries(ax, x_lim, u_lim, label: bool) -> None:
    x_g = np.geomspace(x_lim[0], x_lim[1], 400)
    # sheet/gap ray and unmasked reference ray
    ax.plot(x_g, SLOPE_23 * x_g, color="w", lw=1.6)
    ax.plot(x_g, SLOPE_UNMASKED * x_g, color="w", lw=0.9, ls=":")
    # coherence line x = 1
    ax.plot([1.0, 1.0], list(u_lim), color="w", lw=1.6, ls="--")
    # plateau edge: straight line in linear axes, curved on the log-log frame
    x_p = np.linspace(x_lim[0], np.sqrt(3.0) * (1 - 1e-9), 600)
    ax.plot(x_p, plateau_edge(x_p), color="w", lw=1.6)
    ax.set_xlim(*x_lim)
    ax.set_ylim(*u_lim)
    if label:
        kw = dict(color="w", fontsize=17, fontweight="bold", ha="center")
        ax.text(0.05, 0.05, "1", **kw)
        ax.text(600.0, 8.0, "2", **kw)
        ax.text(6.0, 900.0, "3", **kw)
        ax.text(0.05, 900.0, "4", **kw)
        # ray labels placed along their own slope (45 deg on a log-log frame)
        ax.text(30.0, SLOPE_UNMASKED * 30.0 * 1.5, r"$|u_0|=W$",
                color="w", fontsize=13, rotation=45, rotation_mode="anchor",
                ha="left", va="bottom")
        ax.text(120.0, SLOPE_23 * 120.0 * 0.62,
                r"$|u_0|=\pi x_\nabla/\sqrt{3}$",
                color="w", fontsize=13, rotation=45, rotation_mode="anchor",
                ha="left", va="top")
        ax.text(1.25, 4.0, r"$x_\nabla=1$", color="w", fontsize=13, rotation=90,
                va="bottom")
        ax.text(0.013, 1.05, r"$|u_0|+\pi x_\nabla/\sqrt3=\pi$",
                color="w", fontsize=13, ha="left", va="bottom")


def fig_cmaps(args, out_dirs) -> None:
    x_grid = np.geomspace(args.x_min, args.x_max, args.n_x)
    u_grid = np.geomspace(args.u_min, args.u_max, args.n_u)
    xx, uu = np.meshgrid(x_grid, u_grid, indexing="xy")
    lg.info(f"cmap grid {args.n_x}x{args.n_u} = {xx.size:,} model evaluations")
    e_model = model(xx.reshape(-1), uu.reshape(-1)).reshape(xx.shape)
    e_pred = prediction(xx, uu)
    ratio = np.log10(np.maximum(e_model, 1e-300) / np.maximum(e_pred, 1e-300))

    fig, axes = plt.subplots(1, 3, figsize=pubstyle.figsize(18.0, 5.4))
    img_lims = dict(vmin=np.log10(e_pred[e_pred > 0].min()), vmax=np.log10(PLATEAU))
    for ax, img, title in (
        (axes[0], np.log10(np.maximum(e_model, 1e-300)), "(a) exact linear model"),
        (axes[1], np.log10(np.maximum(e_pred, 1e-300)), "(b) piecewise four-region prediction"),
    ):
        pcm = ax.pcolormesh(x_grid, u_grid, img, cmap="viridis", shading="auto", **img_lims)
        cb = fig.colorbar(pcm, ax=ax, label=r"$\log_{10} N\,T^2\!/L^2$")
        cb.ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
        ax.set_title(title)
    pcm = axes[2].pcolormesh(
        x_grid, u_grid, ratio, cmap="RdBu_r", shading="auto", vmin=-0.5, vmax=0.5
    )
    cb = fig.colorbar(pcm, ax=axes[2], label=r"$\log_{10}$ model/prediction")
    cb.ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
    axes[2].set_title("(c) ratio (region-3 residual stripes = real fringes,\n"
                      "prediction there is fringe-averaged)")
    for k, ax in enumerate(axes):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$x_\nabla$ [rad]")
        ax.set_ylabel(r"$|u_0|$ [rad]")
        draw_boundaries(ax, (args.x_min, args.x_max), (args.u_min, args.u_max),
                        label=(k == 0))
    fig.suptitle(r"Four-region phase diagram of the FWM kernel over $(x_\nabla, |u_0|)$ "
                 r"(equal split, $d=0$) — doc §10.2")
    fig.tight_layout()
    for d in out_dirs:
        fig.savefig(d / "xu0_phase_diagram.png", dpi=pubstyle.dpi(220))
    plt.close(fig)

    inside = {
        1: uu + np.pi * xx / np.sqrt(3.0) < 0.5 * np.pi,
        2: (uu < 0.5 * SLOPE_23 * xx) & (xx > 4.0 * np.sqrt(3.0)),
        3: (uu > 3.0 * SLOPE_23 * xx) & (xx > 4.0),
        4: (xx < 0.3) & (uu > 4.0 * np.pi),
    }
    for r, sel in inside.items():
        dev = np.abs(ratio[sel])
        lg.info(f"region {r} interior ({sel.sum():,} px): median |log10 ratio| = "
                f"{np.median(dev):.4f}, p95 = {np.percentile(dev, 95):.3f}")


def fig_cuts(args, out_dirs) -> None:
    fig, axes = plt.subplots(2, 3, figsize=pubstyle.figsize(18.0, 9.6))

    # (a) iso-x cuts vs |u0|, laws overlaid
    ax = axes[0, 0]
    x_cuts = (0.1, 1.0, 10.0, 100.0, 1000.0)
    colors = plt.cm.plasma(np.linspace(0.05, 0.85, len(x_cuts)))
    u_g = np.geomspace(args.u_min, args.u_max, 1500)
    for x_c, c in zip(x_cuts, colors):
        x_v = np.full(u_g.size, x_c)
        ax.plot(u_g, model(x_v, u_g), color=c, lw=1.0, label=rf"$x_\nabla={x_c:g}$")
        ax.plot(u_g, prediction(x_v, u_g), color="k", lw=0.7, ls="--", alpha=0.7)
    ax.plot(u_g, 4.0 / (3.0 * u_g**2), color="gray", ls=":", lw=1.0)
    ax.text(3e2, 4.0 / (3.0 * 3e2**2) * 3, r"$4/3u_0^2$", color="gray", fontsize=14)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_ylim(1e-10, 2.0)
    ax.set_xlabel(r"$|u_0|$ [rad]"); ax.set_ylabel(r"$N\,T^2\!/L^2$")
    ax.legend(loc="lower left")
    ax.set_title("(a) iso-$x_\\nabla$ cuts vs $|u_0|$; dashed = region laws")

    # (b) compensated u0^2 N -- region 3 collapses on the universal 4/3
    ax = axes[0, 1]
    for x_c, c in zip((10.0, 100.0, 1000.0), colors[2:]):
        x_v = np.full(u_g.size, x_c)
        ax.plot(u_g, u_g**2 * model(x_v, u_g), color=c, lw=0.9,
                label=rf"$x_\nabla={x_c:g}$")
    ax.axhline(4.0 / 3.0, color="k", ls="--", lw=1.0)
    ax.axhline(8.0 / 3.0, color="gray", ls=":", lw=1.0)
    ax.text(args.u_max * 0.35, 4.0 / 3.0 * 1.15, r"region 3: $4/3$ (universal)",
            color="k", fontsize=14, ha="right")
    ax.text(args.u_max * 0.35, 8.0 / 3.0 * 1.15, r"envelope $8/3$",
            color="gray", fontsize=14, ha="right")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(3.0, args.u_max); ax.set_ylim(1e-3, 6.0)
    ax.set_xlabel(r"$|u_0|$ [rad]"); ax.set_ylabel(r"$u_0^2\, N\,T^2\!/L^2$")
    ax.legend(loc="lower left")
    ax.set_title("(b) compensated gapped cuts: plateau $=4/3$, no $x_\\nabla$ left")

    # (c) iso-|u0| cuts vs x
    ax = axes[1, 0]
    u_cuts = (3.0, 30.0, 300.0, 3000.0)
    colors_u = plt.cm.viridis(np.linspace(0.1, 0.85, len(u_cuts)))
    x_g = np.geomspace(args.x_min, args.x_max, 1200)
    for u_c, c in zip(u_cuts, colors_u):
        u_v = np.full(x_g.size, u_c)
        ax.plot(x_g, model(x_g, u_v), color=c, lw=1.0, label=rf"$|u_0|={u_c:g}$")
        ax.plot(x_g, prediction(x_g, u_v), color="k", lw=0.7, ls="--", alpha=0.7)
        ax.axvline(u_c / SLOPE_23, color=c, ls="-", lw=0.6, alpha=0.5)
    ax.axvline(1.0, color="gray", ls="--", lw=0.8)
    ax.text(1.15, 1.2, r"$x_\nabla=1$", color="gray", fontsize=14)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_ylim(1e-9, 3.0)
    ax.set_xlabel(r"$x_\nabla$ [rad]"); ax.set_ylabel(r"$N\,T^2\!/L^2$")
    ax.legend(loc="upper left")
    ax.set_title("(c) iso-$|u_0|$ cuts vs $x_\\nabla$: flat gapped plateau,\n"
                 "then the $3\\sqrt3/4x_\\nabla$ sheet past $|u_0|=\\pi x_\\nabla/\\sqrt3$")

    # (d) region-4 fringes -- nulls exactly at u0 = 2 pi k, no mu correction
    ax = axes[1, 1]
    x_c = 1e-3
    u_lin = np.linspace(4.0, 45.0, 1600)
    ax.semilogy(u_lin, model(np.full(u_lin.size, x_c), u_lin), color="C0", lw=1.2,
                label="exact model")
    ax.semilogy(u_lin, PLATEAU * khat(u_lin), color="k", ls="--", lw=0.9,
                label=r"$\frac{2}{3}\hat K(u_0)$")
    for k in range(1, 8):
        ax.axvline(2.0 * np.pi * k, color="gray", ls=":", lw=0.6)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
    ax.set_xlabel(r"$|u_0|$ [rad] (linear)"); ax.set_ylabel(r"$N\,T^2\!/L^2$")
    ax.legend()
    ax.set_title(rf"(d) region-4 fringes at $x_\nabla=10^{{-3}}$; nulls at $u_0=2\pi k$")

    # (e) fringe contrast vs x
    ax = axes[1, 2]
    x_grid = np.geomspace(0.3, 100.0, 26)
    # Sweep u0 over one kernel period at a fixed multiple C of the sheet edge,
    # so every point stays gapped (|u0| > w) as x_grad grows.
    for c_mult, mk in ((4.0, "o"), (20.0, "s"), (100.0, "^")):
        contrast = []
        for x_t in x_grid:
            u_c = c_mult * SLOPE_23 * x_t
            uu = np.linspace(u_c, u_c + 2.0 * np.pi, 81)
            e = model(np.full(uu.size, x_t), uu)
            contrast.append((e.max() - e.min()) / (e.max() + e.min()))
        ax.plot(x_grid, contrast, mk, ms=3.5, lw=0.0,
                label=rf"$|u_0|={c_mult:g}\,\pi x_\nabla/\sqrt3$")
    ax.plot(x_grid, np.minimum(1.0, 3.0 * np.sqrt(3.0) / (4.0 * np.pi * x_grid)),
            "k--", lw=1.0, label=r"$\frac{3\sqrt3}{4\pi x_\nabla}$ (mask-edge envelope)")
    x_fine = np.geomspace(0.3, 100.0, 2000)
    ax.plot(x_fine, np.minimum(1.0, 3.0 * np.sqrt(3.0)
            / (4.0 * np.pi * x_fine) * np.abs(np.sin(np.pi * x_fine / np.sqrt(3.0)))),
            color="k", lw=0.5, alpha=0.45,
            label=r"$\times\,|\sin(\pi x_\nabla/\sqrt3)|$")
    ax.plot(x_grid, np.abs(np.sinc(x_grid / np.sqrt(3.0))) ** 3, color="gray", ls=":",
            lw=1.0, label=r"$|\mathrm{sinc}^3(\pi x_\nabla/\sqrt3)|$ (unmasked)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_ylim(1e-4, 1.5)
    ax.set_xlabel(r"$x_\nabla$ [rad]"); ax.set_ylabel("fringe contrast")
    ax.legend()
    ax.set_title("(e) fringe contrast: mask-edge $1/x_\\nabla$ envelope\n"
                 "(dips = frozen $|\\sin w|$), not the unmasked $\\mathrm{sinc}^3$")

    # (f) plateau-edge collapse on the straight edge |u0| + pi x/sqrt3 = pi
    ax = axes[0, 2]
    # The plateau edge is a straight line, so the collapse is along rays through
    # the origin.  A ray is labelled by its slope |u0|/x_grad (the derived |mu|).
    p_g = np.geomspace(0.05, 30.0, 500)
    for slope, c in zip((0.05, 1.0, 8.0, 100.0, 1e4), colors):
        x_v = np.pi * p_g / (slope + np.pi / np.sqrt(3.0))
        u_v = slope * x_v
        ax.plot(p_g, model(x_v, u_v) / PLATEAU, color=c, lw=1.0,
                label=rf"$|u_0|/x_\nabla={slope:g}$")
    ax.axhline(1.0, color="gray", ls=":", lw=1.0)
    ax.axvline(1.0, color="gray", ls="-", lw=0.8)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_ylim(1e-4, 2.0)
    ax.set_xlabel(r"$p = \left(|u_0| + \pi x_\nabla/\sqrt3\right)/\pi$")
    ax.set_ylabel(r"$N\,T^2\!/L^2 \;/\; (2/3)$")
    ax.legend(loc="lower left")
    ax.set_title("(f) plateau edge collapses on the straight line $p=1$\n"
                 "(rays through the origin, six decades of slope)")

    fig.suptitle("Quantitative cuts through the four-region phase diagram (doc §10.2)")
    fig.tight_layout()
    for d in out_dirs:
        fig.savefig(d / "xu0_phase_cuts.png", dpi=pubstyle.dpi(220))
    plt.close(fig)


def main() -> None:
    init_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    pubstyle.add_argument(parser)
    parser.add_argument("--out-dir", type=Path, default=Path("media/lorenzi-fast"))
    parser.add_argument("--docs-dir", type=Path,
                        default=Path("docs/source/_static/lorenzi-fast"))
    parser.add_argument("--x-min", type=float, default=1e-2)
    parser.add_argument("--x-max", type=float, default=3e3)
    parser.add_argument("--n-x", type=int, default=420)
    parser.add_argument("--u-min", type=float, default=1e-2)
    parser.add_argument("--u-max", type=float, default=3e3)
    parser.add_argument("--n-u", type=int, default=900)
    args = parser.parse_args()
    pubstyle.apply(args)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_dirs = [args.out_dir]
    if args.docs_dir.is_dir():
        out_dirs.append(args.docs_dir)

    fig_cmaps(args, out_dirs)
    fig_cuts(args, out_dirs)
    lg.success(f"phase-diagram figures saved to {', '.join(str(d) for d in out_dirs)}")


if __name__ == "__main__":
    main()
