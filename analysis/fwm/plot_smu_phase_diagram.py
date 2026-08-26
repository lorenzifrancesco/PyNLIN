"""Assessment figures for the (s, |mu|) four-region phase diagram (doc §10.2).

Validates the closed-form region laws of lorenzi_fast_method.md §10.2 against
the exact linear model (equal split, d = 0, hard-mask window |v| <= w):

  region 1 (coherent plateau):  N T^2/L^2 = 2/3,   s <~ s1(mu)
  region 2 (sheet):             (3 sqrt3/4)(1+mu)(1-mu^2/pi^2)/s,  |mu| < pi/sqrt3
  region 3 (gapped, dephased):  4(1+mu)^2/(3 mu^2 s^2)  (fringe-averaged)
  region 4 (gapped, coherent):  (2/3) Khat(u0),  x = s/(1+mu) <~ 1

Outputs (media/lorenzi-fast/ and, if present, docs/source/_static/lorenzi-fast/):
  smu_phase_diagram.png -- cmaps: exact model | piecewise prediction | ratio,
                           with region boundaries and labels
  smu_phase_cuts.png    -- line plots: iso-mu cuts + laws; compensated s^2 N;
                           iso-s cuts vs mu; region-4 fringe zoom; fringe
                           contrast vs x; plateau-edge collapse vs s/s1(mu)
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import matplotlib

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
MU_23 = np.pi / np.sqrt(3.0)          # masked reachability (2<->3 boundary)
MU_UNMASKED = np.pi * np.sqrt(3.0)    # unmasked-box |u0| = W (reference)
PLATEAU = 2.0 / 3.0


def model(s: np.ndarray, mu: np.ndarray, chunk: int = 200_000) -> np.ndarray:
    """Exact linear-model N T^2/L^2 at (s, mu), equal split, d = 0."""
    s = np.asarray(s, float).reshape(-1)
    mu = np.asarray(mu, float).reshape(-1)
    x = s / (1.0 + mu)
    u0 = s * mu / (1.0 + mu)
    out = np.empty(s.size)
    for lo in range(0, s.size, chunk):
        hi = min(lo + chunk, s.size)
        coeffs = x[lo:hi, None] * _EQUAL_DIR[None, :]
        out[lo:hi] = linear_tuple_estimate(u0[lo:hi], coeffs, np.zeros(hi - lo)).values
    return out


def khat(u: np.ndarray) -> np.ndarray:
    u = np.asarray(u, float)
    out = np.ones_like(u)
    nz = np.abs(u) > 1e-12
    out[nz] = 4.0 * np.sin(u[nz] / 2.0) ** 2 / u[nz] ** 2
    return out


def s1_boundary(mu: np.ndarray) -> np.ndarray:
    """Region-1 edge s1(mu) = pi (1+mu)/(mu + pi/sqrt3)."""
    return np.pi * (1.0 + mu) / (mu + MU_23)


def prediction(s: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """Piecewise four-region closed form (region 3 fringe-averaged)."""
    s = np.asarray(s, float)
    mu = np.asarray(mu, float)
    x = s / (1.0 + mu)
    u0 = s * mu / (1.0 + mu)
    sheet = (3.0 * np.sqrt(3.0) / 4.0) * (1.0 + mu) * (1.0 - mu**2 / np.pi**2) / s
    smooth3 = 4.0 * (1.0 + mu) ** 2 / (3.0 * mu**2 * np.maximum(s, 1e-300) ** 2)
    fringe4 = PLATEAU * khat(u0)
    gapped = np.where(x < 1.0, fringe4, smooth3)
    pred = np.where(mu < MU_23, sheet, gapped)
    return np.minimum(pred, PLATEAU)


def draw_boundaries(ax, s_lim, mu_lim, label: bool) -> None:
    mu_g = np.geomspace(mu_lim[0], mu_lim[1], 200)
    ax.plot(s1_boundary(mu_g), mu_g, color="w", lw=1.6)
    s_g = np.geomspace(s1_boundary(np.array([MU_23]))[0], s_lim[1], 100)
    ax.plot(s_g, np.full(s_g.size, MU_23), color="w", lw=1.6)
    ax.plot(s_g, np.full(s_g.size, MU_UNMASKED), color="w", lw=0.9, ls=":")
    diag = np.geomspace(max(1.0 + MU_23, s_lim[0]), min(s_lim[1], 1.0 + mu_lim[1]), 100)
    ax.plot(diag, diag - 1.0, color="w", lw=1.6, ls="--")
    ax.set_xlim(*s_lim)
    ax.set_ylim(*mu_lim)
    if label:
        kw = dict(color="w", fontsize=16, fontweight="bold", ha="center")
        ax.text(0.35, 1.0, "1", **kw)
        ax.text(60.0, 0.15, "2", **kw)
        ax.text(600.0, 12.0, "3", **kw)
        ax.text(20.0, 400.0, "4", **kw)
        ax.text(s_lim[1] * 0.5, MU_UNMASKED * 1.3, r"$|u_0|=W$", color="w", fontsize=14, ha="right")
        ax.text(s_lim[1] * 0.5, MU_23 * 0.55, r"$|\mu|=\pi/\sqrt{3}$", color="w", fontsize=14, ha="right")
        ax.text(300.0, 190.0, r"$x_\nabla=1$", color="w", fontsize=14, rotation=45)


def fig_cmaps(args, out_dirs) -> None:
    s_grid = np.geomspace(args.s_min, args.s_max, args.n_s)
    mu_grid = np.geomspace(args.mu_min, args.mu_max, args.n_mu)
    ss, mm = np.meshgrid(s_grid, mu_grid, indexing="xy")
    lg.info(f"cmap grid {args.n_s}x{args.n_mu} = {ss.size:,} model evaluations")
    e_model = model(ss.reshape(-1), mm.reshape(-1)).reshape(ss.shape)
    e_pred = prediction(ss, mm)
    ratio = np.log10(np.maximum(e_model, 1e-300) / np.maximum(e_pred, 1e-300))

    fig, axes = plt.subplots(1, 3, figsize=(18.0, 5.4))
    img_lims = dict(vmin=np.log10(e_pred[e_pred > 0].min()), vmax=np.log10(PLATEAU))
    for ax, img, title in (
        (axes[0], np.log10(np.maximum(e_model, 1e-300)), "(a) exact linear model"),
        (axes[1], np.log10(np.maximum(e_pred, 1e-300)), "(b) piecewise four-region prediction"),
    ):
        pcm = ax.pcolormesh(s_grid, mu_grid, img, cmap="viridis", shading="auto", **img_lims)
        cb = fig.colorbar(pcm, ax=ax, label=r"$\log_{10} N\,T^2\!/L^2$")
        cb.ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
        ax.set_title(title)
    pcm = axes[2].pcolormesh(
        s_grid, mu_grid, ratio, cmap="RdBu_r", shading="auto", vmin=-0.5, vmax=0.5
    )
    cb = fig.colorbar(pcm, ax=axes[2], label=r"$\log_{10}$ model/prediction")
    cb.ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
    axes[2].set_title("(c) ratio (region-3 residual stripes = real fringes,\n"
                      "prediction there is fringe-averaged)")
    for k, ax in enumerate(axes):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$s = x_\nabla + |u_0|$ [rad]")
        ax.set_ylabel(r"$|\mu|$")
        draw_boundaries(ax, (args.s_min, args.s_max), (args.mu_min, args.mu_max), label=(k == 0))
    fig.suptitle(r"Four-region phase diagram of the FWM kernel over $(s, |\mu|)$ "
                 r"(equal split, $d=0$) — doc §10.2")
    fig.tight_layout()
    for d in out_dirs:
        fig.savefig(d / "smu_phase_diagram.png", dpi=220)
    plt.close(fig)

    inside = {
        1: ss < 0.5 * s1_boundary(mm),
        2: (ss > 4.0 * s1_boundary(mm)) & (mm < 0.5 * MU_23),
        3: (mm > 3.0 * MU_23) & (ss / (1.0 + mm) > 4.0),
        4: (ss / (1.0 + mm) < 0.3) & (mm > 3.0 * MU_23) & (ss > 4.0 * np.pi),
    }
    for r, sel in inside.items():
        dev = np.abs(ratio[sel])
        lg.info(f"region {r} interior ({sel.sum():,} px): median |log10 ratio| = "
                f"{np.median(dev):.4f}, p95 = {np.percentile(dev, 95):.3f}")


def fig_cuts(args, out_dirs) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(18.0, 9.6))

    # (a) iso-mu cuts vs s, laws overlaid
    ax = axes[0, 0]
    mu_cuts = (0.05, 1.0, 8.0, 100.0, 1e4)
    colors = plt.cm.plasma(np.linspace(0.05, 0.85, len(mu_cuts)))
    s_g = np.geomspace(args.s_min, args.s_max, 1500)
    for mu_c, c in zip(mu_cuts, colors):
        mu_v = np.full(s_g.size, mu_c)
        ax.plot(s_g, model(s_g, mu_v), color=c, lw=1.0, label=rf"$|\mu|={mu_c:g}$")
        ax.plot(s_g, prediction(s_g, mu_v), color="k", lw=0.7, ls="--", alpha=0.7)
    ax.plot(s_g, 2.0 / s_g, color="gray", ls=":", lw=1.0)
    ax.plot(s_g, 20.0 / s_g**2, color="gray", ls=":", lw=1.0)
    ax.text(2e3, 1.6e-3, r"$\propto s^{-1}$", color="gray", fontsize=14)
    ax.text(1.5e2, 2e-4, r"$\propto s^{-2}$", color="gray", fontsize=14)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_ylim(1e-10, 2.0)
    ax.set_xlabel(r"$s$ [rad]"); ax.set_ylabel(r"$N\,T^2\!/L^2$")
    ax.legend()
    ax.set_title("(a) iso-$\\mu$ cuts vs $s$; dashed = region laws")

    # (b) compensated s^2 N for gapped cuts
    ax = axes[0, 1]
    for mu_c, c in zip((8.0, 100.0, 1e4), colors[2:]):
        mu_v = np.full(s_g.size, mu_c)
        ax.plot(s_g, s_g**2 * model(s_g, mu_v), color=c, lw=0.9, label=rf"$|\mu|={mu_c:g}$")
        ax.axhline(4.0 * (1.0 + mu_c) ** 2 / (3.0 * mu_c**2), color=c, ls="--", lw=0.8)
    ax.axhline(8.0 / 3.0, color="gray", ls=":", lw=1.0)
    ax.text(args.s_max * 0.3, 8.0 / 3.0 * 1.15, r"envelope $\frac{2}{3}\cdot\frac{4}{s^2}s^2 = \frac{8}{3}$",
            color="gray", fontsize=14, ha="right")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(3.0, args.s_max); ax.set_ylim(1e-4, 6.0)
    ax.set_xlabel(r"$s$ [rad]"); ax.set_ylabel(r"$s^2\, N\,T^2\!/L^2$")
    ax.legend()
    ax.set_title("(b) compensated gapped cuts: plateau = $\\frac{4}{3}(1+1/\\mu)^2$")

    # (c) iso-s cuts vs mu
    ax = axes[1, 0]
    s_cuts = (3.0, 30.0, 300.0, 3000.0)
    colors_s = plt.cm.viridis(np.linspace(0.1, 0.85, len(s_cuts)))
    mu_g = np.geomspace(args.mu_min, args.mu_max, 1200)
    for s_c, c in zip(s_cuts, colors_s):
        s_v = np.full(mu_g.size, s_c)
        ax.plot(mu_g, model(s_v, mu_g), color=c, lw=1.0, label=rf"$s={s_c:g}$")
        ax.plot(mu_g, prediction(s_v, mu_g), color="k", lw=0.7, ls="--", alpha=0.7)
    ax.axvline(MU_23, color="gray", ls="-", lw=0.8)
    ax.axvline(MU_UNMASKED, color="gray", ls=":", lw=0.8)
    ax.text(MU_23 * 0.85, 1.2, r"$\pi/\sqrt3$", color="gray", fontsize=14, ha="right")
    ax.text(MU_UNMASKED * 1.2, 1.2, r"$\pi\sqrt3$", color="gray", fontsize=14)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_ylim(1e-9, 3.0)
    ax.set_xlabel(r"$|\mu|$"); ax.set_ylabel(r"$N\,T^2\!/L^2$")
    ax.legend(loc="lower left")
    ax.set_title("(c) iso-$s$ cuts vs $|\\mu|$: flat inside regions,\n"
                 "step at $\\pi/\\sqrt3$, fringes appear near $\\mu \\approx s$")

    # (d) region-4 fringe zoom
    ax = axes[1, 1]
    mu_c = 1e3
    s_lin = np.linspace(4.0, 45.0, 1600)
    mu_v = np.full(s_lin.size, mu_c)
    u0 = s_lin * mu_c / (1.0 + mu_c)
    ax.semilogy(s_lin, model(s_lin, mu_v), color="C0", lw=1.2, label="exact model")
    ax.semilogy(s_lin, PLATEAU * khat(u0), color="k", ls="--", lw=0.9,
                label=r"$\frac{2}{3}\hat K(u_0)$")
    for k in range(1, 8):
        ax.axvline(2.0 * np.pi * k * (1.0 + 1.0 / mu_c), color="gray", ls=":", lw=0.6)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
    ax.set_xlabel(r"$s$ [rad] (linear)"); ax.set_ylabel(r"$N\,T^2\!/L^2$")
    ax.legend()
    ax.set_title(rf"(d) region-4 fringes at $|\mu|=10^3$; nulls at $u_0 = 2\pi k$")

    # (e) fringe contrast vs x
    ax = axes[1, 2]
    x_grid = np.geomspace(0.3, 100.0, 26)
    for mu_c, mk in ((20.0, "o"), (50.0, "s"), (200.0, "^")):
        contrast = []
        for x_t in x_grid:
            s_c = x_t * (1.0 + mu_c)
            period = 2.0 * np.pi * (1.0 + mu_c) / mu_c
            ss = np.linspace(s_c, s_c + period, 81)
            e = model(ss, np.full(ss.size, mu_c))
            contrast.append((e.max() - e.min()) / (e.max() + e.min()))
        ax.plot(x_grid, contrast, mk, ms=3.5, lw=0.0, label=rf"$|\mu|={mu_c:g}$")
    ax.plot(x_grid, np.minimum(1.0, 3.0 * np.sqrt(3.0) / (4.0 * np.pi * x_grid)),
            "k--", lw=1.0, label=r"$\frac{3\sqrt3}{4\pi x_\nabla}$ (mask-edge envelope)")
    x_fine = np.geomspace(0.3, 100.0, 2000)
    ax.plot(x_fine, np.minimum(1.0, 3.0 * np.sqrt(3.0)
            / (4.0 * np.pi * x_fine) * np.abs(np.sin(np.pi * x_fine / np.sqrt(3.0)))),
            color="k", lw=0.5, alpha=0.45,
            label=r"$\times\,|\sin(\pi x_\nabla/\sqrt3)|$ (frozen edge, high $\mu$)")
    ax.plot(x_grid, np.abs(np.sinc(x_grid / np.sqrt(3.0))) ** 3, color="gray", ls=":",
            lw=1.0, label=r"$|\mathrm{sinc}^3(\pi x_\nabla/\sqrt3)|$ (unmasked)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_ylim(1e-4, 1.5)
    ax.set_xlabel(r"$x_\nabla = s/(1+|\mu|)$ [rad]"); ax.set_ylabel("fringe contrast")
    ax.legend()
    ax.set_title("(e) fringe contrast: mask-edge $1/x_\\nabla$ envelope\n"
                 "(high-$\\mu$ dips = frozen $|\\sin w|$), not the unmasked $\\mathrm{sinc}^3$")

    # (f) plateau-edge collapse vs s/s1(mu)
    ax = axes[0, 2]
    for mu_c, c in zip((0.05, 1.0, 8.0, 100.0, 1e4), colors):
        r_g = np.geomspace(0.05, 30.0, 500)
        s_v = r_g * s1_boundary(np.array([mu_c]))[0]
        ax.plot(r_g, model(s_v, np.full(s_v.size, mu_c)) / PLATEAU,
                color=c, lw=1.0, label=rf"$|\mu|={mu_c:g}$")
    ax.axhline(1.0, color="gray", ls=":", lw=1.0)
    ax.axvline(1.0, color="gray", ls="-", lw=0.8)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_ylim(1e-4, 2.0)
    ax.set_xlabel(r"$s / s_1(\mu)$,   $s_1 = \pi(1+\mu)/(\mu+\pi/\sqrt3)$")
    ax.set_ylabel(r"$N\,T^2\!/L^2 \;/\; (2/3)$")
    ax.legend(loc="lower left")
    ax.set_title("(f) plateau edge collapses on $s = s_1(\\mu)$")

    fig.suptitle("Quantitative cuts through the four-region phase diagram (doc §10.2)")
    fig.tight_layout()
    for d in out_dirs:
        fig.savefig(d / "smu_phase_cuts.png", dpi=220)
    plt.close(fig)


def main() -> None:
    init_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("media/lorenzi-fast"))
    parser.add_argument("--docs-dir", type=Path,
                        default=Path("docs/source/_static/lorenzi-fast"))
    parser.add_argument("--s-min", type=float, default=1e-1)
    parser.add_argument("--s-max", type=float, default=3e3)
    parser.add_argument("--n-s", type=int, default=1400)
    parser.add_argument("--mu-min", type=float, default=1e-2)
    parser.add_argument("--mu-max", type=float, default=1e3)
    parser.add_argument("--n-mu", type=int, default=360)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_dirs = [args.out_dir]
    if args.docs_dir.is_dir():
        out_dirs.append(args.docs_dir)

    fig_cmaps(args, out_dirs)
    fig_cuts(args, out_dirs)
    lg.success(f"phase-diagram figures saved to {', '.join(str(d) for d in out_dirs)}")


if __name__ == "__main__":
    main()
