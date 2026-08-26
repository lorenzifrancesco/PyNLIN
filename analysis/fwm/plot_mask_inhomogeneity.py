"""Figure for the mask-inhomogeneity analysis in doc §10.3.

Compares the exact masked mismatch densities, efficiencies, and fringe
contrasts of three walk-off width splits at fixed gradient scale x_nabla and
d = 0: the equal split (w,w,w), the one-leg direction (W,0,0), and the
two-leg direction (w,w,0). The first two share the same parabolic masked
density up to a sqrt(3) width rescaling; the two-leg direction is the
genuinely different cusped class.

Output (media/lorenzi-fast and docs/source/_static/lorenzi-fast):
  mask_inhomogeneity.png -- (a) masked densities vs Monte Carlo,
                            (b) sheet-regime efficiency vs u0,
                            (c) gapped fringe contrast vs x_nabla
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

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

SQRT3 = np.sqrt(3.0)
SQRT2 = np.sqrt(2.0)

# Fixed identity order: equal, one-leg, two-leg (Okabe-Ito, CVD-safe), with
# distinct line styles as the color-independent secondary encoding.
DIRECTIONS = (
    ("equal $(w,w,w)$", "#0072B2", "-"),
    ("one-leg $(W,0,0)$", "#D55E00", "--"),
    ("two-leg $(w,w,0)$", "#009E73", "-."),
)


def half_width(direction: int, x_grad: float) -> float:
    """Support half-width h of the masked density at gradient scale x_grad."""
    return np.pi * x_grad * (1.0 / SQRT3, 1.0, SQRT2)[direction]


def masked_density(direction: int, v: np.ndarray, x_grad: float) -> np.ndarray:
    """Exact d=0 masked mismatch density (integrates to A(0) = 2/3)."""
    h = half_width(direction, x_grad)
    inside = np.abs(v) < h
    if direction in (0, 1):
        density = 3.0 / (8.0 * h) - v**2 / (8.0 * h**3)
    else:
        density = (h - np.abs(v)) ** 2 / h**3
    return np.where(inside, density, 0.0)


def link_power_kernel(phase_mismatch: np.ndarray) -> np.ndarray:
    return np.sinc(phase_mismatch / (2.0 * np.pi)) ** 2


def masked_efficiency(
    direction: int, u_const: np.ndarray, x_grad: float, n_nodes: int = 40001
) -> np.ndarray:
    """E = int K(u0+v) rho_masked(v) dv by dense trapezoid quadrature."""
    h = half_width(direction, x_grad)
    v = np.linspace(-h, h, n_nodes)
    density = masked_density(direction, v, x_grad)
    kernel = link_power_kernel(u_const[:, None] + v[None, :])
    return np.trapezoid(kernel * density[None, :], v, axis=1)


def fringe_contrast(direction: int, x_grad: np.ndarray) -> np.ndarray:
    """|phi_v(1)| / A(0) for the masked density (gapped fringe contrast)."""
    a = np.pi * x_grad * (1.0 / SQRT3, 1.0, SQRT2)[direction]
    if direction in (0, 1):
        cf = 0.5 * (np.sin(a) / a - np.cos(a) / a**2 + np.sin(a) / a**3)
    else:
        cf = 4.0 * (a - np.sin(a)) / a**3
    return np.abs(cf) / (2.0 / 3.0)


def monte_carlo_density(
    direction: int, x_grad: float, rng: np.random.Generator, n_samples: int
) -> tuple[np.ndarray, np.ndarray]:
    """Histogram of the masked linear offset for the given direction."""
    xa, xb, xc = rng.uniform(-np.pi, np.pi, (3, n_samples))
    mask = np.abs(xa + xb - xc) < np.pi
    coefficients = (
        (x_grad / SQRT3, x_grad / SQRT3, x_grad / SQRT3),
        (x_grad, 0.0, 0.0),
        (x_grad / SQRT2, x_grad / SQRT2, 0.0),
    )[direction]
    v = coefficients[0] * xa + coefficients[1] * xb - coefficients[2] * xc
    h = half_width(direction, x_grad)
    counts, edges = np.histogram(v[mask], bins=48, range=(-h, h))
    centers = 0.5 * (edges[1:] + edges[:-1])
    density = counts / (n_samples * (edges[1] - edges[0]))
    return centers, density


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("media/lorenzi-fast"))
    parser.add_argument(
        "--docs-dir", type=Path, default=Path("docs/source/_static/lorenzi-fast")
    )
    parser.add_argument("--x-grad-sheet", type=float, default=12.0)
    parser.add_argument("--n-mc", type=int, default=2_000_000)
    args = parser.parse_args()

    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 3, figsize=(18.0, 5.4), constrained_layout=True)

    # (a) masked densities in units of pi * x_nabla, with MC verification.
    ax = axes[0]
    scale = np.pi  # x_grad = 1 => v is in units of pi * x_nabla directly
    v_grid = np.linspace(-SQRT2 * np.pi, SQRT2 * np.pi, 1201)
    for direction, (label, color, style) in enumerate(DIRECTIONS):
        ax.plot(
            v_grid / scale,
            scale * masked_density(direction, v_grid, 1.0),
            color=color,
            ls=style,
            lw=2.2,
            label=label,
        )
        centers, density = monte_carlo_density(direction, 1.0, rng, args.n_mc)
        ax.plot(
            centers[::2] / scale,
            scale * density[::2],
            "o",
            ms=4.5,
            mfc="none",
            mec=color,
            mew=1.2,
        )
    ax.set_xlabel(r"$v/(\pi x_\nabla)$")
    ax.set_ylabel(r"$\pi x_\nabla\,\rho_{\rm masked}(v)$")
    ax.set_title("(a) masked densities ($d=0$; dots: MC)")
    ax.legend(loc="upper right")

    # (b) sheet-regime efficiency vs u0 at fixed x_nabla.
    ax = axes[1]
    u_grid = np.linspace(0.0, 9.0 * args.x_grad_sheet, 700)
    for direction, (label, color, style) in enumerate(DIRECTIONS):
        ax.semilogy(
            u_grid,
            np.maximum(
                masked_efficiency(direction, u_grid, args.x_grad_sheet), 1e-12
            ),
            color=color,
            ls=style,
            lw=2.2,
            label=label,
        )
    tail = np.linspace(2.0 * np.pi * args.x_grad_sheet, u_grid[-1], 200)
    ax.semilogy(
        tail,
        2.0 * (2.0 / 3.0) / tail**2,
        color="0.35",
        ls=":",
        lw=1.8,
        label=r"common tail $2A/u_0^2$",
    )
    ax.set_xlabel(r"$u_0$")
    ax.set_ylabel(r"$E(u_0, x_\nabla)$")
    ax.set_title(rf"(b) sheet regime, $x_\nabla={args.x_grad_sheet:g}$")
    ax.set_ylim(1e-6, 1.0)
    ax.legend(loc="upper right")

    # (c) gapped fringe contrast |phi_v(1)|/A vs x_nabla.
    ax = axes[2]
    x_grid = np.geomspace(1.0, 300.0, 4000)
    for direction, (label, color, style) in enumerate(DIRECTIONS):
        ax.loglog(
            x_grid,
            np.maximum(fringe_contrast(direction, x_grid), 1e-12),
            color=color,
            ls=style,
            lw=1.6,
            label=label,
        )
    ax.loglog(x_grid, 0.8 / x_grid, color="0.35", ls=":", lw=1.6)
    ax.loglog(x_grid, 2.0 / x_grid**2, color="0.35", ls=":", lw=1.6)
    ax.text(120.0, 1.3e-2, r"$\propto x_\nabla^{-1}$", color="0.25")
    ax.text(120.0, 4e-4, r"$\propto x_\nabla^{-2}$", color="0.25")
    ax.set_xlabel(r"$x_\nabla$")
    ax.set_ylabel(r"$|\varphi_v(1)|/A(0)$")
    ax.set_title("(c) gapped fringe contrast")
    ax.set_ylim(1e-6, 2.0)
    ax.legend(loc="lower left")

    fig.suptitle(
        "Mask inhomogeneity across the width simplex at $d=0$: "
        "equal and one-leg splits share one parabolic class; "
        "the two-leg split is cusped"
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    output_dirs = [args.out_dir]
    if args.docs_dir.is_dir():
        output_dirs.append(args.docs_dir)
    for output_dir in output_dirs:
        fig.savefig(
            output_dir / "mask_inhomogeneity.png", dpi=220, bbox_inches="tight"
        )
    plt.close(fig)
    print(f"mask-inhomogeneity figure saved to {', '.join(map(str, output_dirs))}")


if __name__ == "__main__":
    main()
