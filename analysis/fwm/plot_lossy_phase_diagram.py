"""Phase diagram with a decaying power profile, decay tied to u0.

Exploratory companion to §10.2 (not yet referenced from the note).

The lossless note uses a flat longitudinal power profile, for which the link
power kernel is Khat(u) = 4 sin^2(u/2)/u^2.  For an exponentially decaying
profile the amplitude profile is exp(-a z) over the normalized span z in [0,1],
with a = alpha L / 2 the accumulated AMPLITUDE decay in nepers, and the kernel
becomes the squared Fourier transform of that profile,

    K_a(u) = | int_0^1 exp(-a z) exp(i u z) dz |^2
           = [1 - 2 exp(-a) cos u + exp(-2a)] / (u^2 + a^2),

i.e. exactly the standard lossy FWM efficiency; a -> 0 recovers Khat, and the
loss enters as an imaginary part of the mismatch, u -> u + i a.

The figure has two rows, two slices of the (x_grad, u0, a) family.  Equal
split, d = 0 throughout, so the masked density is the central Irwin-Hall
branch of §10.2 and both maps are directly comparable to Figure 10.

TOP ROW -- the diagnostic diagonal a = |u0|: the accumulated decay equals the
accumulated center mismatch at every point, alpha L = |Delta beta_0| L, i.e.
loss and dephasing terminate the interaction at the same scale.  The four
regions collapse to three:

  * fringe contrast collapses to sech(a) = sech(|u0|).  At the first lossless
    null u0 = 2 pi it is 3.7e-3, so the region-4 fringes are gone and the old
    regions 3 and 4 merge into one law.
  * narrow-window / gapped universal law   E = (2/3) K_{u0}(u0),
    which is 2/3 as u0 -> 0 and 1/(3 u0^2) at large u0.  The lossless plateau
    2/3 therefore survives only for |u0| <~ 0.69 (where K(u0,u0) = 1/2) --
    a HORIZONTAL edge, replacing the lossless slanted |u0| + w = pi.
  * gapped tail 4/(3 u0^2) -> 1/(3 u0^2), exactly a factor 1/4.

BOTTOM ROW -- a real span, a = A_SPAN = 2.30 (100 km at 0.2 dB/km).  Here ALL
FOUR regions survive, each rescaled by its own constant: plateau -> (2/3)K_a(0)
= 0.102, sheet x L_eff/L = 0.215, gapped x (1+e^-2a)/2 = 0.505, fringes damped
to contrast sech(a) = 0.198 rather than erased.  Because the factors differ
per region, loss is NOT a single overall efficiency factor.  This is the slice
to use for a link budget.

Common to both, the sheet law is

    E_sheet = 2 pi (L_eff/L) rho(-u0),   L_eff/L = (1-e^-2a)/(2a),

the delta-limit weight 2 pi multiplied by the effective-length ratio (Parseval).
It is 1 at a = 0, recovering §10.2.3.  On the diagonal cut it behaves as
1/(2 u0) at large u0, so E -> 3 sqrt3 / (8 u0 x_grad) and the sheet decays
along BOTH axes; at constant a it is just a constant rescaling.

NOTE on the closed form: in the crossing region the sheet law must be capped by
the PLATEAU (2/3)K_a(0), not by the narrow-window law -- inside a wide crossing
window (2/3)K_a(u0) lies below the sheet value and would wrongly win a minimum.

Output (media/lorenzi-fast/ and docs/source/_static/lorenzi-fast/):
  lossy_phase_diagram.png
"""
from __future__ import annotations

import argparse
import sys
from functools import lru_cache
from pathlib import Path

import matplotlib
import sys as _sys
_sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent / '.'))
import pubstyle

matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams.update({
    "font.size": 15,
    "axes.labelsize": 16,
    "axes.titlesize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 12,
    "figure.titlesize": 18,
})
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pynlin  # noqa: F401
from loguru import logger as lg

from analysis.log_init import init_logging
from analysis.fwm.plot_xu0_phase_diagram import model as lossless_model

SQRT3 = np.sqrt(3.0)
PLATEAU = 2.0 / 3.0
U_HALF = 0.693752  # K_a(a) = 1/2 ; loss-limited plateau edge (diagonal cut)
A_SPAN = 2.3026    # 100 km at 0.2 dB/km: alpha L = 4.605 Np power, a = alphaL/2


@lru_cache(maxsize=64)
def _leggauss(n: int) -> tuple[np.ndarray, np.ndarray]:
    return np.polynomial.legendre.leggauss(n)


def kernel(u: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Lossy link power kernel; a is the amplitude decay over the span."""
    u = np.asarray(u, float)
    a = np.asarray(a, float)
    num = 1.0 - 2.0 * np.exp(-a) * np.cos(u) + np.exp(-2.0 * a)
    den = u**2 + a**2
    return np.where(den < 1e-300, 1.0, num / np.maximum(den, 1e-300))


def density(v: np.ndarray, w: float) -> np.ndarray:
    """Central Irwin-Hall branch retained by the equal-split mask, mass 2/3."""
    return (0.75 - (v / (2.0 * w)) ** 2) / (2.0 * w)


def efficiency_column(x: float, u0: np.ndarray, decay: np.ndarray) -> np.ndarray:
    """Exact masked efficiency at one x_grad, vectorized over |u0|.

    `decay` is the amplitude decay a, broadcast against u0 (a scalar for the
    constant-loss slice, or |u0| itself for the diagonal cut).  The integrand is
    nonnegative (a squared modulus times a density), so Gauss-Legendre is
    cancellation-free.
    """
    w = np.pi * x / SQRT3
    n_raw = int(np.clip(200 + 40.0 * w / np.pi, 200, 8192))
    n_nodes = min(1 << (n_raw - 1).bit_length(), 8192)
    t, wt = _leggauss(n_nodes)
    v = w * t
    weight = w * wt * density(v, w)
    a = np.broadcast_to(np.asarray(decay, float), u0.shape)
    return kernel(u0[:, None] + v[None, :], a[:, None]) @ weight


def model(x_grid: np.ndarray, u_grid: np.ndarray, decay) -> np.ndarray:
    a = np.abs(u_grid) if decay is None else np.full(u_grid.shape, float(decay))
    out = np.empty((u_grid.size, x_grid.size))
    for c, xv in enumerate(x_grid):
        out[:, c] = efficiency_column(float(xv), u_grid, a)
    return out


def effective_length_ratio(a: np.ndarray) -> np.ndarray:
    """L_eff/L = int_0^1 exp(-2 a z) dz = (1-exp(-2a))/(2a), -> 1 as a -> 0."""
    a = np.asarray(a, float)
    small = a < 1e-8
    return np.where(small, 1.0 - a, (1.0 - np.exp(-2.0 * np.where(small, 1.0, a)))
                    / (2.0 * np.where(small, 1.0, a)))


def prediction(x: np.ndarray, u0: np.ndarray, decay) -> np.ndarray:
    """Closed form for either slice; `decay` None means the diagonal a = |u0|.

    coherent / narrow window : (2/3) K_a(u0)                 (keeps the fringes)
    gapped, fringe-averaged  : (2/3)(1+e^-2a)/(u0^2+a^2)
    sheet                    : 2 pi (L_eff/L) rho(-u0)

    The delta-limit weight of the lossy kernel is not 2 pi but, by Parseval,
    2 pi times L_eff/L; that factor is what makes the sheet law reduce to the
    lossless 2 pi rho(-u0) as the decay goes to zero.
    """
    x = np.asarray(x, float)
    u0 = np.abs(np.asarray(u0, float))
    a = u0 if decay is None else np.full_like(u0, float(decay))
    w = np.pi * x / SQRT3
    narrow = PLATEAU * kernel(u0, a)
    gapped_avg = PLATEAU * (1.0 + np.exp(-2.0 * a)) / (u0**2 + a**2)
    gapped = np.where(x < 1.0, narrow, gapped_avg)
    rho_at_match = np.maximum(3.0 * w**2 - u0**2, 0.0) / (8.0 * w**3)
    sheet = 2.0 * np.pi * effective_length_ratio(a) * rho_at_match
    # Cap the sheet by the plateau (2/3)K_a(0): a rigorous upper bound, since
    # K_a peaks at u=0 and the accepted mass is 2/3.  Capping by the
    # narrow-window law instead would be wrong -- inside a WIDE crossing window
    # (2/3)K_a(u0) sits BELOW the sheet value and would win the minimum.
    plateau_a = PLATEAU * kernel(np.zeros_like(u0), a)
    return np.where(u0 < w, np.minimum(sheet, plateau_a), gapped)


def draw_boundaries(ax, x_lim, u_lim, diagonal: bool, label: bool) -> None:
    x_g = np.geomspace(x_lim[0], x_lim[1], 400)
    ax.plot(x_g, np.pi * x_g / SQRT3, color="w", lw=1.8)            # sheet/gap ray
    x_p = np.linspace(x_lim[0], SQRT3 * (1 - 1e-9), 400)            # lossless edge
    ax.plot(x_p, np.pi - np.pi * x_p / SQRT3, color="w", lw=1.0, ls=":", alpha=0.9)
    if diagonal:
        ax.axhline(U_HALF, color="w", lw=1.8)                       # loss plateau edge
        ax.axvline(1.0, color="w", lw=1.0, ls="--", alpha=0.5)      # meaningless here
    else:
        ax.axvline(1.0, color="w", lw=1.5, ls="--")                 # coherence line lives
    ax.set_xlim(*x_lim)
    ax.set_ylim(*u_lim)
    if label:
        kw = dict(color="w", fontsize=15, fontweight="bold", ha="center")
        ax.text(0.03, 0.06, "1", **kw)
        ax.text(400.0, 30.0, "2", **kw)
        if diagonal:
            ax.text(0.06, 200.0, "3", **kw)
            ax.text(0.013, U_HALF * 1.4, r"$|u_0|=0.69$", color="w", fontsize=11,
                    ha="left", va="bottom")
        else:
            ax.text(20.0, 900.0, "3", **kw)
            ax.text(0.05, 900.0, "4", **kw)


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
    parser.add_argument("--a-const", type=float, default=A_SPAN,
                        help="amplitude decay of the constant-loss row (nepers)")
    args = parser.parse_args()
    pubstyle.apply(args)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_dirs = [args.out_dir]
    if args.docs_dir.is_dir():
        out_dirs.append(args.docs_dir)

    x_grid = np.geomspace(args.x_min, args.x_max, args.n_x)
    u_grid = np.geomspace(args.u_min, args.u_max, args.n_u)
    xx, uu = np.meshgrid(x_grid, u_grid, indexing="xy")
    lg.info(f"grid {args.n_x}x{args.n_u} = {xx.size:,} points per row")
    lg.info("evaluating the lossless reference")
    e_flat = lossless_model(xx.reshape(-1), uu.reshape(-1)).reshape(xx.shape)

    rows = (
        (None, r"(diagonal cut) $a=|u_0|$"),
        (args.a_const, rf"(physical span) $a={args.a_const:.2f}$"),
    )
    x_lim, u_lim = (args.x_min, args.x_max), (args.u_min, args.u_max)
    fig, axes = plt.subplots(2, 3, figsize=pubstyle.figsize(18.5, 11.0))

    for r, (decay, tag) in enumerate(rows):
        lg.info(f"row {r}: {tag}")
        e = model(x_grid, u_grid, decay)
        pred = prediction(xx, uu, decay)
        ratio_pred = np.log10(np.maximum(e, 1e-300) / np.maximum(pred, 1e-300))
        ratio_flat = np.log10(np.maximum(e, 1e-300) / np.maximum(e_flat, 1e-300))
        diagonal = decay is None

        pcm = axes[r, 0].pcolormesh(x_grid, u_grid, np.log10(np.maximum(e, 1e-300)),
                                    cmap="viridis", shading="auto",
                                    vmin=-10.0, vmax=np.log10(PLATEAU))
        cb = fig.colorbar(pcm, ax=axes[r, 0], label=r"$\log_{10} N\,T^2\!/L^2$")
        cb.ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
        axes[r, 0].set_title(f"({chr(97+3*r)}) exact model, {tag}")

        pcm = axes[r, 1].pcolormesh(x_grid, u_grid, ratio_pred, cmap="RdBu_r",
                                    shading="auto", vmin=-0.5, vmax=0.5)
        cb = fig.colorbar(pcm, ax=axes[r, 1], label=r"$\log_{10}$ model/prediction")
        cb.ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
        axes[r, 1].set_title(f"({chr(98+3*r)}) vs the closed form")

        pcm = axes[r, 2].pcolormesh(x_grid, u_grid, ratio_flat, cmap="PuOr_r",
                                    shading="auto", vmin=-4.0, vmax=0.5)
        cb = fig.colorbar(pcm, ax=axes[r, 2], label=r"$\log_{10}$ lossy/lossless")
        cb.ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
        axes[r, 2].set_title(f"({chr(99+3*r)}) effect of the decay: / Figure 10")

        for k in range(3):
            ax = axes[r, k]
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_xlabel(r"$x_\nabla$ [rad]"); ax.set_ylabel(r"$|u_0|$ [rad]")
            draw_boundaries(ax, x_lim, u_lim, diagonal, label=(k == 0))

        interiors = {
            1: (uu < 0.3) & (xx < 0.4),
            2: (uu < 0.4 * np.pi * xx / SQRT3) & (xx > 20.0) & (uu > 1.0),
            3: (uu > 3.0 * np.pi * xx / SQRT3) & (uu > 5.0) & (xx > 4.0),
        }
        if not diagonal:
            interiors[4] = (uu > 3.0 * np.pi * xx / SQRT3) & (uu > 5.0) & (xx < 0.3)
        for reg, sel in interiors.items():
            dev = np.abs(ratio_pred[sel])
            lg.info(f"   region {reg} ({sel.sum():,} px): median |log10| = "
                    f"{np.median(dev):.4f}, p95 = {np.percentile(dev, 95):.3f}")
        gap = uu > 3.0 * np.pi * xx / SQRT3
        a_eff = uu[gap] if diagonal else args.a_const
        lg.info(f"   gapped lossy/lossless median = "
                f"{np.median(10**ratio_flat[gap]):.4f}  (law: (1+e^-2a)/2 "
                f"with a={'|u0|' if diagonal else args.a_const})")
        sheet = (uu < 0.3 * np.pi * xx / SQRT3) & (xx > 50.0)
        lg.info(f"   sheet lossy/lossless median = "
                f"{np.median(10**ratio_flat[sheet]):.4f}"
                + ("" if diagonal else
                   f"  (law: L_eff/L = {float(effective_length_ratio(args.a_const)):.4f})"))

    fig.suptitle(r"Equal-split phase diagram with an exponentially decaying power "
                 r"profile ($d=0$): the diagonal cut $a=|u_0|$ (top) and a real "
                 r"span (bottom)")
    fig.tight_layout()
    for d in out_dirs:
        fig.savefig(d / "lossy_phase_diagram.png", dpi=pubstyle.dpi(200))
    plt.close(fig)
    lg.success(f"figure saved to {', '.join(str(d) for d in out_dirs)}")


if __name__ == "__main__":
    main()
