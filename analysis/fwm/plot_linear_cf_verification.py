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
    s: np.ndarray,
    detuning_mu_abs: np.ndarray,
    *,
    n_nodes: int,
    chunk_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Return masked and unmasked equal-split CF integrals.

    Inputs are dimensionless arrays with matching shapes. The returned arrays
    have the same shape and contain dimensionless interaction efficiencies.
    """
    s, detuning_mu_abs = np.broadcast_arrays(
        np.asarray(s, dtype=float), np.asarray(detuning_mu_abs, dtype=float)
    )
    shape = s.shape
    s_flat = s.reshape(-1)
    mu_flat = detuning_mu_abs.reshape(-1)
    gradient_scale = s_flat / (1.0 + mu_flat)
    u_const = s_flat * mu_flat / (1.0 + mu_flat)
    mismatch_halfwidth = np.pi * gradient_scale / SQRT3

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


def draw_boundaries(ax: plt.Axes, s_limits: tuple[float, float]) -> None:
    detuning = np.geomspace(1e-2, 1e3, 500)
    plateau = np.pi * (1.0 + detuning) / (detuning + MASKED_REACHABILITY)
    ax.plot(plateau, detuning, color="w", lw=1.2)
    ax.axhline(MASKED_REACHABILITY, color="w", lw=1.2)
    ax.axhline(UNMASKED_REACHABILITY, color="w", lw=0.8, ls=":")
    coherence = 1.0 + detuning
    visible = (coherence >= s_limits[0]) & (coherence <= s_limits[1])
    ax.plot(coherence[visible], detuning[visible], color="w", lw=1.0, ls="--")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("media/lorenzi-fast"))
    parser.add_argument(
        "--docs-dir", type=Path, default=Path("docs/source/_static/lorenzi-fast")
    )
    parser.add_argument("--s-max", type=float, default=300.0)
    parser.add_argument("--n-s", type=int, default=220)
    parser.add_argument("--n-mu", type=int, default=170)
    parser.add_argument("--n-nodes", type=int, default=2048)
    args = parser.parse_args()

    s_grid = np.geomspace(0.1, args.s_max, args.n_s)
    mu_grid = np.geomspace(1e-2, 1e3, args.n_mu)
    ss, mm = np.meshgrid(s_grid, mu_grid, indexing="xy")
    masked_cf, unmasked_cf = characteristic_function_efficiencies(
        ss, mm, n_nodes=args.n_nodes
    )
    masked_reference = equal_split_efficiency(ss, mm, 0.0)

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
            s_grid,
            mu_grid,
            np.log10(np.maximum(values, 1e-300)),
            cmap="viridis",
            shading="auto",
            **efficiency_limits,
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(title)
        ax.set_xlabel(r"$s=x_\nabla+|u_{\rm const}|$")
        ax.set_ylabel(r"$|\mu|=|u_{\rm const}|/x_\nabla$")
        draw_boundaries(ax, (s_grid[0], s_grid[-1]))
        fig.colorbar(image, ax=ax, label=r"$\log_{10} E$")

    ax = axes[1, 0]
    log_error = np.log10(np.maximum(np.abs(relative_error), 1e-16))
    image = ax.pcolormesh(
        s_grid,
        mu_grid,
        log_error,
        cmap="magma",
        shading="auto",
        vmin=-14.0,
        vmax=-4.0,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("(c) pointwise numerical agreement")
    ax.set_xlabel(r"$s=x_\nabla+|u_{\rm const}|$")
    ax.set_ylabel(r"$|\mu|=|u_{\rm const}|/x_\nabla$")
    fig.colorbar(
        image,
        ax=ax,
        label=r"$\log_{10}|E_{\rm CF}/E_{u}-1|$",
        format="%g",
    )

    ax = axes[1, 1]
    cut_s = np.geomspace(0.1, args.s_max, 500)
    colors = plt.cm.plasma(np.linspace(0.15, 0.85, 3))
    for detuning, color in zip((0.1, 2.0, 20.0), colors):
        cut_mu = np.full_like(cut_s, detuning)
        cut_masked_cf, cut_unmasked_cf = characteristic_function_efficiencies(
            cut_s, cut_mu, n_nodes=args.n_nodes
        )
        cut_reference = equal_split_efficiency(cut_s, cut_mu, 0.0)
        ax.loglog(
            cut_s,
            cut_reference,
            color=color,
            lw=1.8,
            label=rf"masked, $|\mu|={detuning:g}$",
        )
        ax.loglog(cut_s[::16], cut_masked_cf[::16], "o", color=color, ms=3)
        ax.loglog(cut_s, cut_unmasked_cf, color=color, ls="--", lw=1.0)
    ax.set_title("(d) values: masked (solid/points), unmasked sinc$^3$ (dashed)")
    ax.set_xlabel(r"$s=x_\nabla+|u_{\rm const}|$")
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
