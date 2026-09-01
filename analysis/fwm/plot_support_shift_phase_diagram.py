"""Figures for the nonzero-support-shift equal-split analysis in doc §10.2.8.

The evaluator integrates the full piecewise-quadratic Irwin-Hall density
analytically against the linear-model link power kernel. It therefore retains
the exact mask-mismatch correlation of the equal-split direction and does not
use the production estimator's marginal-mask approximation.

Outputs (media/lorenzi-fast and docs/source/_static/lorenzi-fast):
  support_shift_phase_slices.png   -- exact signed-u0 slices at d/pi=0,1,2,3
  support_shift_marginal_error.png -- error of E_0 A(d)/A(0) at d/pi=1,2,3

Coordinates are the two intrinsic accumulated-phase scales (x_grad, signed u0)
of doc §4.1; every mask/density boundary is then a ray u0 = slope * x_grad.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import sys as _sys
_sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent / '.'))
import pubstyle

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import sici


# Both outputs are wide multi-panel figures and are downscaled in the docs.
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
PLATEAU = 2.0 / 3.0
MU_UNMASKED = np.pi * SQRT3


def link_power_kernel(phase_mismatch: np.ndarray) -> np.ndarray:
    """Return 4 sin^2(u/2)/u^2 with its continuous value at zero."""
    phase_mismatch = np.asarray(phase_mismatch, dtype=float)
    return np.sinc(phase_mismatch / (2.0 * np.pi)) ** 2


def support_acceptance(support_shift: float | np.ndarray) -> np.ndarray:
    """Exact marginal acceptance for three normalized uniform coordinates."""
    support_shift = np.asarray(support_shift, dtype=float)
    shift_over_pi = np.abs(support_shift) / np.pi
    central = PLATEAU - shift_over_pi**2 / 4.0 + shift_over_pi**3 / 16.0
    edge = (4.0 - shift_over_pi) ** 3 / 48.0
    return np.where(
        shift_over_pi <= 2.0,
        central,
        np.where(shift_over_pi < 4.0, edge, 0.0),
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


def _quadratic_density_segment(
    u_const: np.ndarray,
    width: np.ndarray,
    y_lo: float,
    y_hi: float,
    branch: str,
) -> np.ndarray:
    """Integrate one quadratic density branch over v/width in [y_lo, y_hi]."""
    if y_hi <= y_lo:
        return np.zeros_like(u_const)

    if branch == "central":
        coeff_v2 = -1.0 / (8.0 * width**3)
        coeff_v1 = np.zeros_like(width)
        coeff_v0 = 3.0 / (8.0 * width)
    elif branch == "negative":
        coeff_v2 = 1.0 / (16.0 * width**3)
        coeff_v1 = 3.0 / (8.0 * width**2)
        coeff_v0 = 9.0 / (16.0 * width)
    elif branch == "positive":
        coeff_v2 = 1.0 / (16.0 * width**3)
        coeff_v1 = -3.0 / (8.0 * width**2)
        coeff_v0 = 9.0 / (16.0 * width)
    else:
        raise ValueError(f"unknown density branch: {branch}")

    # Substitute v = u - u_const in rho(v) = a*v^2 + b*v + c.
    coeff_u2 = coeff_v2
    coeff_u1 = coeff_v1 - 2.0 * coeff_v2 * u_const
    coeff_u0 = (
        coeff_v2 * u_const**2 - coeff_v1 * u_const + coeff_v0
    )

    def primitive(v: np.ndarray) -> np.ndarray:
        phase = u_const + v
        primitive_0, primitive_1, primitive_2 = _kernel_primitives(phase)
        return 2.0 * (
            coeff_u2 * primitive_0
            + coeff_u1 * primitive_1
            + coeff_u0 * primitive_2
        )

    return primitive(width * y_hi) - primitive(width * y_lo)


def equal_split_efficiency(
    gradient_scale: np.ndarray,
    u_0: np.ndarray,
    support_shift: float,
) -> np.ndarray:
    """Exact equal-split linear efficiency on arrays of (x_grad, signed u0)."""
    gradient_scale, u_const = np.broadcast_arrays(
        np.asarray(gradient_scale, dtype=float), np.asarray(u_0, dtype=float)
    )
    width = np.pi * gradient_scale / SQRT3

    accepted_lo = max(-3.0, -1.0 - support_shift / np.pi)
    accepted_hi = min(3.0, 1.0 - support_shift / np.pi)
    if accepted_lo >= accepted_hi:
        return np.zeros_like(gradient_scale)

    efficiency = np.zeros_like(gradient_scale)
    for branch_lo, branch_hi, branch in (
        (-3.0, -1.0, "negative"),
        (-1.0, 1.0, "central"),
        (1.0, 3.0, "positive"),
    ):
        y_lo = max(accepted_lo, branch_lo)
        y_hi = min(accepted_hi, branch_hi)
        efficiency += _quadratic_density_segment(
            u_const, width, y_lo, y_hi, branch
        )

    # Roundoff in differences of analytic primitives can produce tiny
    # negative values near deep kernel nulls.
    return np.maximum(efficiency, 0.0)


def _draw_boundaries(ax: plt.Axes, support_shift: float, x_limits) -> None:
    """All mask/density boundaries are rays u0 = slope * x_grad."""
    x_line = np.geomspace(x_limits[0], x_limits[1], 400)
    sheet_lo = (support_shift - np.pi) / SQRT3
    sheet_hi = (support_shift + np.pi) / SQRT3
    for slope in (sheet_lo, sheet_hi):
        ax.plot(x_line, slope * x_line, color="w", lw=1.2)
    for slope in (-MU_UNMASKED, MU_UNMASKED):
        ax.plot(x_line, slope * x_line, color="w", lw=0.8, ls=":")
    ax.axvline(1.0, color="w", lw=1.0, ls="--")


def _symlog_grid(u_max: float, linthresh: float, n_side: int) -> np.ndarray:
    """Signed u0 samples spaced logarithmically away from zero."""
    tail = np.geomspace(linthresh, u_max, n_side)
    core = np.linspace(-linthresh, linthresh, max(9, n_side // 8))[1:-1]
    return np.concatenate([-tail[::-1], core, tail])


def plot_phase_slices(args, output_dirs: list[Path]) -> None:
    x_grid = np.geomspace(args.x_min, args.x_max, args.n_x)
    u_grid = _symlog_grid(args.u_max, args.linthresh, args.n_u)
    xx, uu = np.meshgrid(x_grid, u_grid, indexing="xy")
    support_shifts = np.pi * np.array([0.0, 1.0, 2.0, 3.0])
    efficiencies = [equal_split_efficiency(xx, uu, d) for d in support_shifts]

    positive = np.concatenate([values[values > 0.0] for values in efficiencies])
    color_limits = dict(vmin=max(-8.0, np.log10(positive).min()), vmax=np.log10(PLATEAU))
    fig, axes = plt.subplots(
        2,
        2,
        figsize=pubstyle.figsize(14.0, 10.0),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    for panel, (ax, support_shift, efficiency) in enumerate(
        zip(axes.flat, support_shifts, efficiencies)
    ):
        image = ax.pcolormesh(
            x_grid,
            u_grid,
            np.log10(np.maximum(efficiency, 1e-300)),
            cmap="viridis",
            shading="auto",
            **color_limits,
        )
        _draw_boundaries(ax, support_shift, (args.x_min, args.x_max))
        ax.set_xscale("log")
        ax.set_yscale("symlog", linthresh=args.linthresh)
        ax.set_xlim(args.x_min, args.x_max)
        ax.set_ylim(-args.u_max, args.u_max)
        ax.set_title(rf"({chr(97 + panel)}) $d/\pi={support_shift / np.pi:g}$")
        ax.set_xlabel(r"$x_\nabla$ [rad]")
        ax.set_ylabel(r"signed $u_0$ [rad]")

    colorbar = fig.colorbar(image, ax=axes, shrink=0.94, pad=0.02)
    colorbar.set_label(r"$\log_{10} E_d$")
    fig.suptitle(
        "Exact equal-split phase diagrams under a translated output-support mask\n"
        r"solid: $u_0=(d\pm\pi)x_\nabla/\sqrt{3}$; "
        r"dotted: $|u_0|=\pi\sqrt{3}\,x_\nabla$; dashed: $x_\nabla=1$"
    )
    for output_dir in output_dirs:
        fig.savefig(
            output_dir / "support_shift_phase_slices.png",
            dpi=pubstyle.dpi(220),
            bbox_inches=None if pubstyle.current() != "screen" else "tight",
        )
    plt.close(fig)


def plot_marginal_error(args, output_dirs: list[Path]) -> None:
    x_grid = np.geomspace(args.x_min, args.x_max, args.n_x)
    u_grid = _symlog_grid(args.u_max, args.linthresh, args.n_u)
    xx, uu = np.meshgrid(x_grid, u_grid, indexing="xy")
    centered = equal_split_efficiency(xx, uu, 0.0)
    support_shifts = np.pi * np.array([1.0, 2.0, 3.0])

    fig, axes = plt.subplots(
        1,
        3,
        figsize=pubstyle.figsize(18.0, 5.4),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    for panel, (ax, support_shift) in enumerate(zip(axes, support_shifts)):
        exact = equal_split_efficiency(xx, uu, support_shift)
        marginal = centered * support_acceptance(support_shift) / PLATEAU
        log_ratio = np.log10(
            np.maximum(exact, 1e-14) / np.maximum(marginal, 1e-14)
        )
        image = ax.pcolormesh(
            x_grid,
            u_grid,
            log_ratio,
            cmap="RdBu_r",
            shading="auto",
            vmin=-3.0,
            vmax=3.0,
        )
        ax.contour(x_grid, u_grid, log_ratio, levels=[0.0], colors="k", linewidths=0.6)
        _draw_boundaries(ax, support_shift, (args.x_min, args.x_max))
        ax.set_xscale("log")
        ax.set_yscale("symlog", linthresh=args.linthresh)
        ax.set_xlim(args.x_min, args.x_max)
        ax.set_ylim(-args.u_max, args.u_max)
        ax.set_title(rf"({chr(97 + panel)}) $d/\pi={support_shift / np.pi:g}$")
        ax.set_xlabel(r"$x_\nabla$ [rad]")
        ax.set_ylabel(r"signed $u_0$ [rad]")

    colorbar = fig.colorbar(image, ax=axes, shrink=0.92, pad=0.02)
    colorbar.set_label(r"$\log_{10}\{E_d/[E_0 A(d)/A(0)]\}$")
    fig.suptitle(
        "Failure of marginal support rescaling for the correlated equal-split direction\n"
        "black contour: equality; color range clipped at factors $10^{-3}$ and $10^3$"
    )
    for output_dir in output_dirs:
        fig.savefig(
            output_dir / "support_shift_marginal_error.png",
            dpi=pubstyle.dpi(220),
            bbox_inches=None if pubstyle.current() != "screen" else "tight",
        )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    pubstyle.add_argument(parser)
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
    pubstyle.apply(args)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    output_dirs = [args.out_dir]
    if args.docs_dir.is_dir():
        output_dirs.append(args.docs_dir)

    plot_phase_slices(args, output_dirs)
    plot_marginal_error(args, output_dirs)
    print(f"support-shift figures saved to {', '.join(map(str, output_dirs))}")


if __name__ == "__main__":
    main()
