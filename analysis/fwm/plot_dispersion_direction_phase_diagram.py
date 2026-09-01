"""Phase diagrams of the three zero-phase plane normals u1, u2, u3 (doc §10.4.1).

Companion-note convention (`analysis/standalone_analytical/phase_matching_planes.md`):
coordinates (nu1, nu2, nu3) with energy conservation w1 - w2 + w3 = w4, so legs 1
and 3 are the unconjugated pumps and leg 2 is the conjugated one.  The
frequency-matching slab is |nu1 - nu2 + nu3| < pi, i.e. mask normal m = (1,-1,1).

Truncated at beta_3 the phase-matching locus is exactly three planes, and the
UNIT NORMALS to those planes are

    u1 = (1,-1,0)/sqrt2     P1: nu1 = nu2            (XPM sector)
    u2 = (0, 1,-1)/sqrt2    P2: nu2 = nu3            (XPM sector)
    u3 = (1, 0, 1)/sqrt2    Q : nu1 + nu3 = 2 nu_ZDF (genuine FWM sheet)

Because the surfaces are level sets of Delta-beta, the normal IS the walk-off
gradient, so each u_i doubles as the walk-off direction c on its own plane.

Structure exploited here.  Each u_i has one zero slot, and the mask normal has
+-1 in every slot, so the leg absent from the mismatch always enters the mask
linearly with unit coefficient.  Writing t for the unit-scale mismatch
(v = x_grad * t) and S for the mask form, one finds S = (m . u_i) t = eps
sqrt2 t with eps = +1 for u1, u3 and eps = -1 for u2.  Hence

    rho_d(t) = [sqrt2 (2pi - sqrt2|t|) / (4 pi^2)]           <- triangular pair
             * [max(0, 2pi - |eps sqrt2 t + d|) / (2 pi)]    <- free third leg

on |t| <= h = sqrt2 pi.  At d = 0 this is the cusped law (h-|t|)^2/h^3 for all
three, so the three phase diagrams COINCIDE EXACTLY.  At d != 0 the sign eps
separates them: u1 and u3 stay identical while u2 is the exact mirror,
E_u2(x, u0) = E_u1(x, -u0).

Verified against direct 3-D masked Monte-Carlo over the cube (<=5.5e-3, MC
noise) and, independently, by reproducing A(d) = 2/3, 23/48, 1/6, 1/48 at
d/pi = 0, 1, 2, 3.

Output (media/lorenzi-fast/ and docs/source/_static/lorenzi-fast/):
  dispersion_direction_phase_diagram.png
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
    "font.size": 14,
    "axes.labelsize": 15,
    "axes.titlesize": 15,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
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

SQRT2 = np.sqrt(2.0)
H_UNIT = SQRT2 * np.pi          # support half-width of t at x_grad = 1
MASK_NORMAL = np.array([1.0, -1.0, 1.0])

# plane normals in the companion note's (nu1, nu2, nu3) coordinates
PLANES = (
    (r"$\mathbf u_1=(1,-1,0)/\sqrt2$", r"$P_1:\ \nu_1=\nu_2$",
     np.array([1.0, -1.0, 0.0])),
    (r"$\mathbf u_2=(0,1,-1)/\sqrt2$", r"$P_2:\ \nu_2=\nu_3$",
     np.array([0.0, 1.0, -1.0])),
    (r"$\mathbf u_3=(1,0,1)/\sqrt2$", r"$Q:\ \nu_1+\nu_3=2\nu_{\rm ZDF}$",
     np.array([1.0, 0.0, 1.0])),
)


def unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def mask_sign(normal: np.ndarray) -> float:
    """eps such that S = eps*sqrt2*t; derived from the direction itself."""
    u = unit(normal)
    k = int(np.argmin(np.abs(u)))
    if abs(u[k]) > 1e-12:
        raise ValueError("direction has no zero slot; general case not handled")
    if abs(abs(MASK_NORMAL[k]) - 1.0) > 1e-12:
        raise ValueError("free leg does not enter the mask with unit coefficient")
    return float(np.dot(MASK_NORMAL, u) / SQRT2)


def khat(u: np.ndarray) -> np.ndarray:
    u = np.asarray(u, float)
    out = np.ones_like(u)
    nz = np.abs(u) > 1e-12
    out[nz] = 4.0 * np.sin(u[nz] / 2.0) ** 2 / u[nz] ** 2
    return out


def rho_masked(t: np.ndarray, eps: float, d: float) -> np.ndarray:
    """Masked density of the unit-scale mismatch t, support |t| <= H_UNIT."""
    t = np.asarray(t, float)
    triangular = SQRT2 * (2.0 * np.pi - SQRT2 * np.abs(t)) / (4.0 * np.pi**2)
    acceptance = np.clip(2.0 * np.pi - np.abs(eps * SQRT2 * t + d), 0.0, None) / (2.0 * np.pi)
    return np.where(np.abs(t) <= H_UNIT, triangular * acceptance, 0.0)


def accepted_support(eps: float, d: float) -> tuple[float, float]:
    """Interval of t retained by the shifted mask (intersected with the cube)."""
    lo, hi = (-2.0 * np.pi - d) / (eps * SQRT2), (2.0 * np.pi - d) / (eps * SQRT2)
    if lo > hi:
        lo, hi = hi, lo
    return max(lo, -H_UNIT), min(hi, H_UNIT)


def accepted_moments(eps: float, d: float, n: int = 200_001) -> tuple[float, float, float]:
    """(A(d), mean t, sigma_t) of the masked density."""
    t = np.linspace(-H_UNIT, H_UNIT, n)
    r = rho_masked(t, eps, d)
    a = np.trapezoid(r, t)
    if a <= 0:
        return 0.0, 0.0, 0.0
    m1 = np.trapezoid(t * r, t) / a
    m2 = np.trapezoid(t**2 * r, t) / a
    return a, m1, float(np.sqrt(max(m2 - m1**2, 0.0)))


@lru_cache(maxsize=64)
def _leggauss(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Cached nodes/weights; leggauss is O(n^2) and would dominate otherwise."""
    return np.polynomial.legendre.leggauss(n)


def efficiency_column(x: float, u0: np.ndarray, eps: float, d: float) -> np.ndarray:
    """Exact masked efficiency at one x_grad, vectorized over signed u0.

    Nonnegative integrand, so Gauss-Legendre is cancellation-free; the node
    count tracks the number of kernel periods across the accepted window.
    """
    n_raw = int(np.clip(200 + 40.0 * H_UNIT * abs(x) / np.pi, 200, 12000))
    n_nodes = 1 << (n_raw - 1).bit_length()  # snap to powers of two so the cache hits
    tn, w = _leggauss(n_nodes)
    t = H_UNIT * tn
    weight = H_UNIT * w * rho_masked(t, eps, d)
    return khat(u0[:, None] + x * t[None, :]) @ weight


def efficiency_grid(x_grid: np.ndarray, u_grid: np.ndarray, eps: float,
                    d: float) -> np.ndarray:
    out = np.empty((u_grid.size, x_grid.size))
    for c, xv in enumerate(x_grid):
        out[:, c] = efficiency_column(float(xv), u_grid, eps, d)
    return out


def symlog_grid(u_max: float, linthresh: float, n_side: int) -> np.ndarray:
    tail = np.geomspace(linthresh, u_max, n_side)
    core = np.linspace(-linthresh, linthresh, max(9, n_side // 8))[1:-1]
    return np.concatenate([-tail[::-1], core, tail])


def draw_boundaries(ax, eps: float, d: float, x_lim, u_lim) -> None:
    x_line = np.geomspace(x_lim[0], x_lim[1], 400)
    t_lo, t_hi = accepted_support(eps, d)
    # phase matching reachable  <=>  -u0/x in [t_lo, t_hi]
    for t_edge in (t_lo, t_hi):
        ax.plot(x_line, -t_edge * x_line, color="w", lw=1.6)
    _, _, sigma_unit = accepted_moments(eps, d)
    if sigma_unit > 0:
        ax.axvline(1.0 / sigma_unit, color="w", lw=1.3, ls="--")
    ax.set_xlim(*x_lim)
    ax.set_ylim(*u_lim)


def main() -> None:
    init_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    pubstyle.add_argument(parser)
    parser.add_argument("--out-dir", type=Path, default=Path("media/lorenzi-fast"))
    parser.add_argument("--docs-dir", type=Path,
                        default=Path("docs/source/_static/lorenzi-fast"))
    parser.add_argument("--x-min", type=float, default=1e-2)
    parser.add_argument("--x-max", type=float, default=1e2)
    parser.add_argument("--n-x", type=int, default=260)
    parser.add_argument("--u-max", type=float, default=300.0)
    parser.add_argument("--linthresh", type=float, default=0.3)
    parser.add_argument("--n-u", type=int, default=190)
    args = parser.parse_args()
    pubstyle.apply(args)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_dirs = [args.out_dir]
    if args.docs_dir.is_dir():
        out_dirs.append(args.docs_dir)

    x_grid = np.geomspace(args.x_min, args.x_max, args.n_x)
    u_grid = symlog_grid(args.u_max, args.linthresh, args.n_u)
    x_lim = (args.x_min, args.x_max)
    u_lim = (-args.u_max, args.u_max)
    shifts = (0.0, np.pi)

    lg.info(f"grid {args.n_x}x{u_grid.size} per panel, "
            f"{len(PLANES) * len(shifts)} panels")

    results = {}
    for row, d in enumerate(shifts):
        for col, (name, plane, normal) in enumerate(PLANES):
            eps = mask_sign(normal)
            results[(row, col)] = (efficiency_grid(x_grid, u_grid, eps, d), eps)
            a, m1, s1 = accepted_moments(eps, d)
            lg.info(f"d={d/np.pi:g}pi  {plane}: eps={eps:+.0f}  A(d)={a:.6f}  "
                    f"mean t={m1:+.4f}  sigma_t={s1:.4f}  "
                    f"coherence x_grad={1.0/s1:.4f}")

    fig, axes = plt.subplots(len(shifts), len(PLANES), figsize=pubstyle.figsize(18.5, 10.6),
                             sharex=True, sharey=True)
    vmin, vmax = -8.0, np.log10(2.0 / 3.0)
    for row, d in enumerate(shifts):
        for col, (name, plane, normal) in enumerate(PLANES):
            ax = axes[row, col]
            values, eps = results[(row, col)]
            image = ax.pcolormesh(x_grid, u_grid,
                                  np.log10(np.maximum(values, 1e-300)),
                                  cmap="viridis", shading="auto",
                                  vmin=vmin, vmax=vmax)
            draw_boundaries(ax, eps, d, x_lim, u_lim)
            ax.set_xscale("log")
            ax.set_yscale("symlog", linthresh=args.linthresh)
            tag = f"({chr(97 + row * len(PLANES) + col)})"
            ax.set_title(f"{tag} {name}\n{plane},  $d={d/np.pi:g}\\pi$")
            if row == len(shifts) - 1:
                ax.set_xlabel(r"$x_\nabla$ [rad]")
            if col == 0:
                ax.set_ylabel(r"signed $u_0$ [rad]")
    cbar = fig.colorbar(image, ax=axes, shrink=0.9, pad=0.015)
    cbar.set_label(r"$\log_{10} N\,T^2\!/L^2$")
    fig.suptitle(
        "Phase diagrams of the three zero-phase plane normals (doc §10.4.1)\n"
        r"top: $d=0$, all three coincide exactly;  bottom: $d=\pi$, "
        r"$\mathbf u_1\equiv\mathbf u_3$ while $\mathbf u_2$ is the mirror "
        r"$u_0\mapsto-u_0$"
    )
    for out_dir in out_dirs:
        fig.savefig(out_dir / "dispersion_direction_phase_diagram.png",
                    dpi=pubstyle.dpi(200), bbox_inches=None if pubstyle.current() != "screen" else "tight")
    plt.close(fig)

    # ---- quantitative identity / mirror checks quoted in the caption --------
    for row, d in enumerate(shifts):
        e1, e2, e3 = (results[(row, c)][0] for c in range(3))
        keep = e1 > 1e-12
        lg.info(f"d={d/np.pi:g}pi: max|u1/u3 - 1| = "
                f"{np.abs(e1[keep] / e3[keep] - 1).max():.3e}")
        mirrored = e2[::-1, :]
        keep_m = e1 > 1e-12
        lg.info(f"d={d/np.pi:g}pi: max|u2(-u0)/u1 - 1| = "
                f"{np.abs(mirrored[keep_m] / e1[keep_m] - 1).max():.3e}"
                "   (u_grid is symmetric, so row reversal is u0 -> -u0)")
    lg.success(f"figure saved to {', '.join(str(d) for d in out_dirs)}")


if __name__ == "__main__":
    main()
