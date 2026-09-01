"""Verify the equal-split linear model in characteristic-function form.

The product-of-sincs integral is the unmasked expectation. For the masked
phase diagram, the full three-uniform characteristic function is replaced by
the cosine transform of the density retained by the output-support mask.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.fwm.plot_support_shift_phase_diagram import equal_split_efficiency
from pynlin.methods.td.fast_nlin import cf_gauss_legendre


SQRT3 = np.sqrt(3.0)
MASKED_REACHABILITY = np.pi / SQRT3
UNMASKED_REACHABILITY = np.pi * SQRT3

matplotlib.rcParams.update({
    "font.size": 14,
    "axes.labelsize": 15,
    "axes.titlesize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 12,
    "figure.titlesize": 18,
    "xtick.major.size": 5,
    "ytick.major.size": 5,
    "xtick.major.width": 1,
    "ytick.major.width": 1,
})


def retained_offset_transform(argument: np.ndarray) -> np.ndarray:
    """Cosine transform of the masked equal-split offset density.

    The transform includes the mask acceptance, so its value at zero is 2/3.
    """
    argument = np.asarray(argument, dtype=float)
    out = np.empty_like(argument)
    small = np.abs(argument) < 1e-3
    a = argument[small]
    out[small] = 2.0 / 3.0 - a**2 / 10.0 + a**4 / 210.0 - a**6 / 9072.0
    a = argument[~small]
    out[~small] = 0.5 * (
        np.sin(a) / a - np.cos(a) / a**2 + np.sin(a) / a**3
    )
    return out


def characteristic_function_efficiencies(
    gradient_scale: np.ndarray,
    u_0: np.ndarray,
    *,
    n_nodes: int,
    chunk_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Return masked and unmasked equal-split CF integrals.

    Inputs are the intrinsic coordinates (x_grad, |u0|) as dimensionless arrays
    with matching shapes. The returned arrays have the same shape and contain
    dimensionless interaction efficiencies.
    """
    gradient_scale, u_0 = np.broadcast_arrays(
        np.asarray(gradient_scale, dtype=float), np.asarray(u_0, dtype=float)
    )
    shape = gradient_scale.shape
    s_flat = gradient_scale.reshape(-1)
    u_const = u_0.reshape(-1)
    mismatch_halfwidth = np.pi * s_flat / SQRT3

    lag, weights = cf_gauss_legendre(n_nodes)
    triangular_weight = 2.0 * (1.0 - lag) * weights
    masked = np.empty_like(s_flat)
    unmasked = np.empty_like(s_flat)

    for start in range(0, s_flat.size, chunk_size):
        stop = min(start + chunk_size, s_flat.size)
        phase = u_const[start:stop, None] * lag[None, :]
        width_lag = mismatch_halfwidth[start:stop, None] * lag[None, :]
        common = np.cos(phase) * triangular_weight[None, :]
        masked[start:stop] = np.sum(
            common * retained_offset_transform(width_lag), axis=1
        )
        unmasked[start:stop] = np.sum(common * np.sinc(width_lag / np.pi) ** 3, axis=1)

    return masked.reshape(shape), unmasked.reshape(shape)


def draw_boundaries(ax: plt.Axes, x_limits: tuple[float, float]) -> None:
    """Same demarcations as Figure 10, all straight in (x_grad, |u0|)."""
    x_line = np.geomspace(x_limits[0], x_limits[1], 500)
    ax.plot(x_line, MASKED_REACHABILITY * x_line, color="w", lw=1.2)
    ax.plot(x_line, UNMASKED_REACHABILITY * x_line, color="w", lw=0.8, ls=":")
    x_plateau = np.linspace(x_limits[0], SQRT3 * (1 - 1e-9), 400)
    ax.plot(x_plateau, np.pi * (1.0 - x_plateau / SQRT3), color="w", lw=1.2)
    ax.axvline(1.0, color="w", lw=1.0, ls="--")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("media/lorenzi-fast"))
    parser.add_argument(
        "--docs-dir", type=Path, default=Path("docs/source/_static/lorenzi-fast")
    )
    parser.add_argument("--x-max", type=float, default=300.0)
    parser.add_argument("--u-max", type=float, default=300.0)
    parser.add_argument("--n-x", type=int, default=220)
    parser.add_argument("--n-u", type=int, default=220)
    parser.add_argument("--n-nodes", type=int, default=2048)
    args = parser.parse_args()

    x_grid = np.geomspace(1e-2, args.x_max, args.n_x)
    u_grid = np.geomspace(1e-2, args.u_max, args.n_u)
    xx, uu = np.meshgrid(x_grid, u_grid, indexing="xy")
    masked_cf, unmasked_cf = characteristic_function_efficiencies(
        xx, uu, n_nodes=args.n_nodes
    )
    masked_reference = equal_split_efficiency(xx, uu, 0.0)

    reliable = masked_reference > 1e-10
    relative_error = np.full_like(masked_reference, np.nan)
    relative_error[reliable] = (
        masked_cf[reliable] / masked_reference[reliable] - 1.0
    )
    print(
        "masked CF vs mismatch-space reference: "
        f"median relative error={np.nanmedian(np.abs(relative_error)):.3e}, "
        f"max={np.nanmax(np.abs(relative_error)):.3e} "
        "for reference efficiency > 1e-10"
    )

    positive = masked_reference[masked_reference > 0.0]
    efficiency_limits = dict(vmin=max(-10.0, np.log10(positive).min()), vmax=0.0)
    fig, axes = plt.subplots(2, 2, figsize=(14.0, 10.0), constrained_layout=True)

    for ax, values, title in (
        (axes[0, 0], masked_reference, "(a) mismatch-space integral (reference)"),
        (axes[0, 1], masked_cf, "(b) mask-corrected characteristic-function integral"),
    ):
        image = ax.pcolormesh(
            x_grid,
            u_grid,
            np.log10(np.maximum(values, 1e-300)),
            cmap="viridis",
            shading="auto",
            **efficiency_limits,
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(title)
        ax.set_xlabel(r"$x_\nabla$ [rad]")
        ax.set_ylabel(r"$|u_0|$ [rad]")
        draw_boundaries(ax, (x_grid[0], x_grid[-1]))
        ax.set_xlim(x_grid[0], x_grid[-1])
        ax.set_ylim(u_grid[0], u_grid[-1])
        fig.colorbar(image, ax=ax, label=r"$\log_{10} E$")

    ax = axes[1, 0]
    log_error = np.log10(np.maximum(np.abs(relative_error), 1e-16))
    image = ax.pcolormesh(
        x_grid,
        u_grid,
        log_error,
        cmap="magma",
        shading="auto",
        vmin=-14.0,
        vmax=-4.0,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(x_grid[0], x_grid[-1])
    ax.set_ylim(u_grid[0], u_grid[-1])
    ax.set_title("(c) pointwise numerical agreement")
    ax.set_xlabel(r"$x_\nabla$ [rad]")
    ax.set_ylabel(r"$|u_0|$ [rad]")
    fig.colorbar(
        image,
        ax=ax,
        label=r"$\log_{10}|E_{\rm CF}/E_{u}-1|$",
        format="%g",
    )

    ax = axes[1, 1]
    cut_u = np.geomspace(1e-2, args.u_max, 500)
    colors = plt.cm.plasma(np.linspace(0.15, 0.85, 3))
    for gradient, color in zip((0.3, 3.0, 30.0), colors):
        cut_x = np.full_like(cut_u, gradient)
        cut_masked_cf, cut_unmasked_cf = characteristic_function_efficiencies(
            cut_x, cut_u, n_nodes=args.n_nodes
        )
        cut_reference = equal_split_efficiency(cut_x, cut_u, 0.0)
        ax.loglog(
            cut_u,
            cut_reference,
            color=color,
            lw=1.8,
            label=rf"masked, $x_\nabla={gradient:g}$",
        )
        ax.loglog(cut_u[::16], cut_masked_cf[::16], "o", color=color, ms=3)
        ax.loglog(cut_u, cut_unmasked_cf, color=color, ls="--", lw=1.0)
    ax.set_title("(d) values: masked (solid/points), unmasked sinc$^3$ (dashed)")
    ax.set_xlabel(r"$|u_0|$ [rad]")
    ax.set_ylabel("dimensionless efficiency")
    ax.set_ylim(1e-10, 2.0)
    ax.legend(loc="lower left")

    fig.suptitle(
        "Numerical verification of the linear-model characteristic-function representation"
    )
    output_dirs = [args.out_dir]
    if args.docs_dir.is_dir():
        output_dirs.append(args.docs_dir)
    for output_dir in output_dirs:
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir / "linear_cf_verification.png", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()
