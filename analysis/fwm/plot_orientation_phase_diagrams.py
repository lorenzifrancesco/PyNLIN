"""Orientation phase diagrams for doc §10.3 (Figure-10 analogues).

Exact linear-model phase diagrams in the (x_grad, signed u0) plane for the three
sign-aligned walk-off orientations of §10.3 at d = 0: equal split (w,w,w),
one-leg (W,0,0), and two-leg (w,w,0). All three masked mismatch densities
are piecewise quadratic, so the efficiency integral is evaluated with the
same closed-form kernel primitives used for the support-shift diagrams.
The one-leg panel is the equal-split diagram under the exact remap
(10.3.2); the two-leg panel is the genuinely different cusped class, and
the last panel shows its log-ratio to the equal split.

Output (media/lorenzi-fast and docs/source/_static/lorenzi-fast):
  orientation_phase_diagrams.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import sici

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

SQRT2 = np.sqrt(2.0)
SQRT3 = np.sqrt(3.0)
PLATEAU = 2.0 / 3.0

# (label, half-width factor h / (pi x_nabla))
ORIENTATIONS = (
    (r"equal $(w,w,w)$", 1.0 / SQRT3),
    (r"one-leg $(W,0,0)$", 1.0),
    (r"two-leg $(w,w,0)$", SQRT2),
)


def _kernel_primitives(phase_mismatch: np.ndarray) -> tuple[np.ndarray, ...]:
    """Primitives of (1-cos(u)), (1-cos(u))/u, and (1-cos(u))/u^2."""
    phase_mismatch = np.asarray(phase_mismatch, dtype=float)
    abs_phase = np.abs(phase_mismatch)
    small = abs_phase < 1e-3

    primitive_0 = phase_mismatch - np.sin(phase_mismatch)
    _, cosine_integral = sici(abs_phase)
    with np.errstate(divide="ignore", invalid="ignore"):
        primitive_1 = np.euler_gamma + np.log(abs_phase) - cosine_integral
        primitive_2 = sici(phase_mismatch)[0] - (
            2.0 * np.sin(phase_mismatch / 2.0) ** 2 / phase_mismatch
        )

    u2 = phase_mismatch * phase_mismatch
    primitive_0 = np.where(
        small,
        phase_mismatch**3 / 6.0
        - phase_mismatch**5 / 120.0
        + phase_mismatch**7 / 5040.0,
        primitive_0,
    )
    primitive_1 = np.where(
        small,
        u2 / 4.0 - u2**2 / 96.0 + u2**3 / 4320.0,
        primitive_1,
    )
    primitive_2 = np.where(
        small,
        phase_mismatch / 2.0
        - phase_mismatch**3 / 72.0
        + phase_mismatch**5 / 3600.0,
        primitive_2,
    )
    return primitive_0, primitive_1, primitive_2


def _quadratic_segment(
    u_const: np.ndarray,
    coeff_v2: np.ndarray,
    coeff_v1: np.ndarray,
    coeff_v0: np.ndarray,
    v_lo: np.ndarray,
    v_hi: np.ndarray,
) -> np.ndarray:
    """Integrate K(u_const+v)(a v^2 + b v + c) over [v_lo, v_hi] exactly."""
    coeff_u2 = coeff_v2
    coeff_u1 = coeff_v1 - 2.0 * coeff_v2 * u_const
    coeff_u0 = coeff_v2 * u_const**2 - coeff_v1 * u_const + coeff_v0

    def primitive(v: np.ndarray) -> np.ndarray:
        primitive_0, primitive_1, primitive_2 = _kernel_primitives(u_const + v)
        return 2.0 * (
            coeff_u2 * primitive_0
            + coeff_u1 * primitive_1
            + coeff_u0 * primitive_2
        )

    return primitive(v_hi) - primitive(v_lo)


def orientation_efficiency(
    orientation: int, gradient_scale: np.ndarray, u_0: np.ndarray
) -> np.ndarray:
    """Exact d=0 masked linear efficiency on arrays of (x_grad, signed u0)."""
    gradient_scale, u_const = np.broadcast_arrays(
        np.asarray(gradient_scale, dtype=float), np.asarray(u_0, dtype=float)
    )
    h = np.pi * gradient_scale * ORIENTATIONS[orientation][1]

    if orientation in (0, 1):
        efficiency = _quadratic_segment(
            u_const,
            -1.0 / (8.0 * h**3),
            np.zeros_like(h),
            3.0 / (8.0 * h),
            -h,
            h,
        )
    else:
        efficiency = _quadratic_segment(
            u_const, 1.0 / h**3, 2.0 / h**2, 1.0 / h, -h, np.zeros_like(h)
        ) + _quadratic_segment(
            u_const, 1.0 / h**3, -2.0 / h**2, 1.0 / h, np.zeros_like(h), h
        )
    return np.maximum(efficiency, 0.0)


def _draw_boundaries(ax: plt.Axes, orientation: int, x_limits) -> None:
    """Sheet edges are the rays u0 = +-h x_grad; coherence line is x_grad = 1."""
    slope = np.pi * ORIENTATIONS[orientation][1]
    x_line = np.geomspace(x_limits[0], x_limits[1], 400)
    for signed_slope in (-slope, slope):
        ax.plot(x_line, signed_slope * x_line, color="w", lw=1.2)
    ax.axvline(1.0, color="w", lw=1.0, ls="--")


def _symlog_grid(u_max: float, linthresh: float, n_side: int) -> np.ndarray:
    """Signed u0 samples spaced logarithmically away from zero."""
    tail = np.geomspace(linthresh, u_max, n_side)
    core = np.linspace(-linthresh, linthresh, max(9, n_side // 8))[1:-1]
    return np.concatenate([-tail[::-1], core, tail])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("media/lorenzi-fast"))
    parser.add_argument(
        "--docs-dir", type=Path, default=Path("docs/source/_static/lorenzi-fast")
    )
    parser.add_argument("--x-min", type=float, default=0.05)
    parser.add_argument("--x-max", type=float, default=300.0)
    parser.add_argument("--n-x", type=int, default=900)
    parser.add_argument("--u-max", type=float, default=300.0)
    parser.add_argument("--linthresh", type=float, default=0.5)
    parser.add_argument("--n-u", type=int, default=260)
    args = parser.parse_args()

    x_grid = np.geomspace(args.x_min, args.x_max, args.n_x)
    u_grid = _symlog_grid(args.u_max, args.linthresh, args.n_u)
    xx, uu = np.meshgrid(x_grid, u_grid, indexing="xy")
    efficiencies = [
        orientation_efficiency(orientation, xx, uu) for orientation in range(3)
    ]

    positive = np.concatenate([values[values > 0.0] for values in efficiencies])
    color_limits = dict(
        vmin=max(-8.0, np.log10(positive).min()), vmax=np.log10(PLATEAU)
    )
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(14.0, 10.0),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    for panel, (ax, efficiency) in enumerate(zip(axes.flat[:3], efficiencies)):
        image = ax.pcolormesh(
            x_grid,
            u_grid,
            np.log10(np.maximum(efficiency, 1e-300)),
            cmap="viridis",
            shading="auto",
            **color_limits,
        )
        _draw_boundaries(ax, panel, (args.x_min, args.x_max))
        ax.set_xscale("log")
        ax.set_yscale("symlog", linthresh=args.linthresh)
        ax.set_xlim(args.x_min, args.x_max)
        ax.set_ylim(-args.u_max, args.u_max)
        ax.set_title(f"({chr(97 + panel)}) {ORIENTATIONS[panel][0]}")
        ax.set_xlabel(r"$x_\nabla$ [rad]")
        ax.set_ylabel(r"signed $u_0$ [rad]")
    colorbar = fig.colorbar(image, ax=axes.flat[:3].tolist(), shrink=0.94, pad=0.02)
    colorbar.set_label(r"$\log_{10} E$")

    ax = axes.flat[3]
    log_ratio = np.log10(
        np.maximum(efficiencies[2], 1e-14) / np.maximum(efficiencies[0], 1e-14)
    )
    image = ax.pcolormesh(
        x_grid,
        u_grid,
        log_ratio,
        cmap="RdBu_r",
        shading="auto",
        vmin=-2.0,
        vmax=2.0,
    )
    ax.contour(x_grid, u_grid, log_ratio, levels=[0.0], colors="k", linewidths=0.6)
    for orientation in (0, 2):
        _draw_boundaries(ax, orientation, (args.x_min, args.x_max))
    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=args.linthresh)
    ax.set_xlim(args.x_min, args.x_max)
    ax.set_ylim(-args.u_max, args.u_max)
    ax.set_title("(d) ratio two-leg / equal")
    ax.set_xlabel(r"$x_\nabla$ [rad]")
    ax.set_ylabel(r"signed $u_0$ [rad]")
    colorbar = fig.colorbar(image, ax=[ax], shrink=0.94, pad=0.02)
    colorbar.set_label(r"$\log_{10}(E_{(w,w,0)}/E_{(w,w,w)})$")

    fig.suptitle(
        "Exact $d=0$ phase diagrams across walk-off orientations\n"
        r"solid: sheet-edge rays $|u_0|=h\,x_\nabla$; dashed: $x_\nabla=1$; "
        "(b) is (a) under the exact remap (10.3.2)"
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    output_dirs = [args.out_dir]
    if args.docs_dir.is_dir():
        output_dirs.append(args.docs_dir)
    for output_dir in output_dirs:
        fig.savefig(
            output_dir / "orientation_phase_diagrams.png",
            dpi=220,
            bbox_inches="tight",
        )
    plt.close(fig)
    print(
        "orientation phase diagrams saved to "
        f"{', '.join(map(str, output_dirs))}"
    )


if __name__ == "__main__":
    main()
