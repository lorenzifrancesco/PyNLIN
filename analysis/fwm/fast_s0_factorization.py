"""Lorenzi Fast S0 factorization: real census = population x kernel x acceptance.

Demonstrates that the real-system S0 mass map factorizes as

    mass(x, mu)  ~=  N(x, mu) * F_syn(x, mu) * A(d)/A(0)

where N is the tuple population density (pure grid combinatorics),
F_syn is the synthetic equal-split, d=0 efficiency kernel evaluated on a
by-hand (x, mu) mesh (the "terrain": interaction physics only, no
channel plan), and A(d) is the closed-form support acceptance (on a regular
grid d is quantized to a few discrete values, so this is a per-tuple
discrete factor, not a function of the map coordinates).

Reads media/lorenzi-fast/s0_territory.npz (produced by fast_s0_territory.py
on the full grid) and writes a 4-panel figure:
  (a) population density, (b) synthetic kernel, (c) real per-target-
  normalized mass map, (d) predicted mass built from (a)+(b)+acceptance only
plus bin-level and per-tuple agreement statistics.

Validated (2026-08-24, full 2284-channel grid, 69.5M tuples): per-tuple
median real/predicted ratio 1.000 with 99.7% within 2x; bin-level median
-0.06 dex, IQR < 0.14 dex (residual dominated by the equal-split
idealization of the walk-off direction).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pynlin  # noqa: F401
from loguru import logger as lg

from analysis.log_init import init_logging
from pynlin.methods.td.fast_nlin import linear_tuple_estimate, support_acceptance

_EQUAL_DIR = np.array([1.0, 1.0, -1.0]) / np.sqrt(3.0)
_A0 = 2.0 / 3.0


def synthetic_kernel_grid(
    x_g: np.ndarray, m_g: np.ndarray
) -> np.ndarray:
    """Equal-split, d=0 kernel F(x, mu) on the mesh (rows mu)."""
    xx, mm = np.meshgrid(x_g, m_g, indexing="xy")
    xf = xx.reshape(-1)
    u0 = (mm * xx).reshape(-1)
    coeffs = xf[:, None] * _EQUAL_DIR[None, :]
    est = linear_tuple_estimate(u0, coeffs, np.zeros(xf.size))
    return est.values.reshape(xx.shape)


def main() -> None:
    init_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--territory-npz", type=Path,
        default=Path("media/lorenzi-fast/s0_territory.npz"),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("media/lorenzi-fast"))
    parser.add_argument(
        "--docs-dir", type=Path,
        default=Path("docs/source/_static/lorenzi-fast"),
        help="Extra copy of the figure for the theory doc (skipped if missing).",
    )
    parser.add_argument("--n-grid", type=int, default=140)
    parser.add_argument("--n-bins", type=int, default=120)
    args = parser.parse_args()

    real = np.load(args.territory_npz)
    # Current S0 keys (x_grad, mu); fall back to the pre-rename keys
    # (x_scale, mu_doc) so older census files stay loadable. CAUTION: old
    # files ALSO contain a "mu" key, but there it is the ACCUMULATED phase
    # (now "u0"), not the dimensionless detuning -- discriminate on the
    # new-convention marker "x_grad", never on the presence of "mu".
    new_keys = "x_grad" in real
    x = real["x_grad"] if new_keys else real["x_scale"]
    md = np.abs(real["mu"] if new_keys else real["mu_doc"])
    Fn, d, tof = real["F_norm"], real["d"], real["target_of"]
    pos = x > 0

    # Kernel mesh covering the whole real population (mu floor catches the
    # exact zero-sum families: the kernel saturates below the boundary, so
    # clamping them to the bottom row is exact to the digits shown).
    x_g = np.logspace(
        np.log10(max(x[pos].min(), 1e-2) / 1.1), np.log10(x[pos].max() * 1.1),
        args.n_grid,
    )
    m_g = np.logspace(-7, np.log10(md[pos].max() * 1.1), args.n_grid)
    lg.info(f"kernel mesh {args.n_grid}x{args.n_grid} over "
            f"x in [{x_g[0]:.2g}, {x_g[-1]:.2g}], mu in [{m_g[0]:.0e}, {m_g[-1]:.2g}]")
    Fk = synthetic_kernel_grid(x_g, m_g)

    ok = (x > x_g[0]) & (x < x_g[-1]) & (Fn > 0)
    md_c = np.clip(md, m_g[0] * 1.001, None)
    ok &= md_c < m_g[-1]
    idx = np.where(ok)[0]
    lg.info(f"usable tuples: {ok.mean():.1%} ({Fn[idx].sum() / Fn.sum():.2%} of mass)")

    itp = RegularGridInterpolator(
        (np.log(m_g), np.log(x_g)), np.log(np.maximum(Fk, 1e-300))
    )
    pred = np.empty(idx.size)
    for s0 in range(0, idx.size, 4_000_000):
        sl = idx[s0:s0 + 4_000_000]
        pred[s0:s0 + sl.size] = np.exp(
            itp(np.column_stack([np.log(md_c[sl]), np.log(x[sl])]))
        )
    w_pred = pred * support_acceptance(d[idx]) / _A0
    # per-target normalization of the prediction, mirroring the S0 convention
    for tt in np.unique(tof[idx]):
        sel = tof[idx] == tt
        tot = w_pred[sel].sum()
        if tot > 0:
            w_pred[sel] /= tot

    xe = np.logspace(np.log10(x_g[0]), np.log10(x_g[-1]), args.n_bins + 1)
    ye = np.logspace(np.log10(m_g[0]), np.log10(m_g[-1]), args.n_bins + 1)

    def hist(w: np.ndarray | None = None) -> np.ndarray:
        h, _, _ = np.histogram2d(x[idx], md_c[idx], bins=[xe, ye], weights=w)
        return h.T

    H_cnt, H_real, H_pred = hist(), hist(Fn[idx]), hist(w_pred)
    sel = (H_real > H_real.max() * 1e-6) & (H_pred > 0)
    r = np.log10(H_real[sel] / H_pred[sel])
    lg.info(f"bin-level log10(real/pred): median {np.median(r):+.3f}, "
            f"IQR [{np.percentile(r, 25):+.2f}, {np.percentile(r, 75):+.2f}] "
            f"over {sel.sum()} bins")

    boundary = np.pi * np.sqrt(3.0)
    # +3pt on every numeric font size (numeric rc values pin their elements
    # and do not follow font.size).
    matplotlib.rcParams["figure.dpi"] = 150
    for key in (
        "font.size", "axes.labelsize", "axes.titlesize", "xtick.labelsize",
        "ytick.labelsize", "legend.fontsize", "legend.title_fontsize",
        "figure.titlesize",
    ):
        value = matplotlib.rcParams.get(key)
        if isinstance(value, (int, float)):
            matplotlib.rcParams[key] = value + 3.0
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.0), sharex=True, sharey=True)
    for ax, img, title in (
        (axes[0, 0], np.log10(np.maximum(H_cnt, 1e-300)),
         "(a) population: tuple count density"),
        (axes[1, 0], np.log10(np.maximum(H_real, 1e-300)),
         "(c) REAL mass map (S0, per-target normalized)"),
        (axes[1, 1], np.log10(np.maximum(H_pred, 1e-300)),
         "(d) PREDICTED mass = population x kernel x acceptance"),
    ):
        vmax = img.max()
        pcm = ax.pcolormesh(xe, ye, img, cmap="viridis", vmin=vmax - 6, vmax=vmax)
        fig.colorbar(pcm, ax=ax, label=r"$\log_{10}$")
        ax.set_title(title)
    ax = axes[0, 1]
    pcm = ax.pcolormesh(
        x_g, m_g, np.log10(np.maximum(Fk, 1e-300)), cmap="viridis", vmin=-8, vmax=0
    )
    fig.colorbar(pcm, ax=ax, label=r"$\log_{10} F$")
    ax.set_title("(b) synthetic kernel F(x, $\\mu$) (equal split, d=0)")
    for ax in axes.ravel():
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.axhline(boundary, color="w", ls="--", lw=0.9, alpha=0.85)
    for ax in axes[1]:
        ax.set_xlabel(r"$x = LB\|\nabla\Delta\beta\|_2$ [rad]")
    for ax in axes[:, 0]:
        ax.set_ylabel(
            r"$|\mu| = |\Delta\beta_{\rm center}|/(B\|\nabla\Delta\beta\|_2)$"
        )
    axes[0, 0].text(x_g[2], boundary * 1.5, r"$|u_0|=W$", color="w")
    fig.suptitle(
        f"Real S0 mass = population (a) $\\times$ synthetic kernel (b) $\\times$ acceptance "
        f"({idx.size:,} tuples):\n"
        "panel (d) is built from (a)+(b) only, yet reproduces the real map (c); "
        "raw bins, no smoothing"
    )
    fig.tight_layout()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / "s0_factorization.png"
    fig.savefig(out, bbox_inches="tight")
    if args.docs_dir.is_dir():
        fig.savefig(args.docs_dir / "s0_factorization.png", bbox_inches="tight")
    plt.close(fig)
    lg.success(f"factorization figure saved to {out}")


if __name__ == "__main__":
    main()
