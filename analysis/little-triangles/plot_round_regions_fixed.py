#!/usr/bin/env python3
"""Plot regions of round(x+y), round(x), and round(y).

Outputs are written to ./media/:
    round_x_plus_y.png
    round_x.png
    round_y.png
    round_regions_combined.png

The combined figure uses the color map for round(x+y), overlays the
boundaries of round(x) and round(y), and labels every subregion with

    (round(x+y), round(x), round(y)).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


def nearest_integer(z: np.ndarray | float) -> np.ndarray | int:
    """Nearest-integer rounding using NumPy's tie-to-even convention."""
    value = np.rint(z).astype(int)
    return int(value) if np.ndim(value) == 0 else value


def discrete_cmap_and_norm(values: np.ndarray, cmap_name: str):
    vmin = int(np.min(values))
    vmax = int(np.max(values))
    levels = np.arange(vmin - 0.5, vmax + 1.5, 1.0)
    cmap = plt.get_cmap(cmap_name, vmax - vmin + 1)
    norm = BoundaryNorm(levels, cmap.N)
    return cmap, norm, levels, vmin, vmax


def half_integer_boundaries(lim: tuple[float, float]) -> np.ndarray:
    """Half-integers strictly inside the plotting interval."""
    n_min = int(np.ceil(lim[0] - 0.5))
    n_max = int(np.floor(lim[1] - 0.5))
    values = np.arange(n_min, n_max + 1, dtype=float) + 0.5
    return values[(values > lim[0]) & (values < lim[1])]


def integer_cells(lim: tuple[float, float]) -> range:
    """Integer labels whose nearest-integer cells intersect lim."""
    first = int(np.ceil(lim[0] - 0.5))
    last = int(np.floor(lim[1] + 0.5))
    return range(first, last + 1)


def save_scalar_region_plot(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    title: str,
    colorbar_label: str,
    output: Path,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    cmap_name: str,
) -> None:
    cmap, norm, _, vmin, vmax = discrete_cmap_and_norm(Z, cmap_name)

    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    mesh = ax.pcolormesh(X, Y, Z, cmap=cmap, norm=norm, shading="nearest")

    cbar = fig.colorbar(mesh, ax=ax, ticks=np.arange(vmin, vmax + 1), pad=0.025)
    cbar.set_label(colorbar_label)

    ax.set_title(title)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal")
    ax.grid(False)

    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def annotate_triplets(
    ax: plt.Axes,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> None:
    """Label each polygonal region by (round(x+y), round(x), round(y)).

    Within the cell round(x)=i, round(y)=j there are generally three pieces:
    a lower-left triangle, a central hexagon, and an upper-right triangle.
    Their centroids are at offsets (-1/3,-1/3), (0,0), and (1/3,1/3).
    """
    candidates = (
        (-1.0 / 3.0, -1.0 / 3.0, -1),
        (0.0, 0.0, 0),
        (1.0 / 3.0, 1.0 / 3.0, 1),
    )

    margin_x = 0.03 * (xlim[1] - xlim[0])
    margin_y = 0.03 * (ylim[1] - ylim[0])

    for i in integer_cells(xlim):
        for j in integer_cells(ylim):
            for dx, dy, delta in candidates:
                x = i + dx
                y = j + dy

                if not (
                    xlim[0] + margin_x <= x <= xlim[1] - margin_x
                    and ylim[0] + margin_y <= y <= ylim[1] - margin_y
                ):
                    continue

                k = i + j + delta
                ax.text(
                    x,
                    y,
                    f"({k}, {i}, {j})",
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    color="black",
                    bbox={
                        "boxstyle": "round,pad=0.13",
                        "facecolor": "white",
                        "edgecolor": "none",
                        "alpha": 0.72,
                    },
                    zorder=5,
                )


def save_combined_plot(
    X: np.ndarray,
    Y: np.ndarray,
    Zsum: np.ndarray,
    output: Path,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    cmap_name: str,
) -> None:
    cmap, norm, levels, vmin, vmax = discrete_cmap_and_norm(Zsum, cmap_name)

    fig, ax = plt.subplots(figsize=(9.0, 7.8))
    mesh = ax.pcolormesh(
        X,
        Y,
        Zsum,
        cmap=cmap,
        norm=norm,
        shading="nearest",
        rasterized=True,
    )

    round_x = nearest_integer(X)
    round_y = nearest_integer(Y)
    zero_sum = Zsum + round_x + round_y == 0
    ax.contourf(
        X,
        Y,
        zero_sum.astype(int),
        levels=[0.5, 1.5],
        colors=["#ffd166"],
        alpha=0.55,
        zorder=1,
    )
    ax.contour(
        X,
        Y,
        zero_sum.astype(int),
        levels=[0.5],
        colors=["#9c6500"],
        linewidths=1.5,
        zorder=2,
    )

    # Diagonal boundaries of round(x+y).
    ax.contour(
        X,
        Y,
        X + Y,
        levels=levels,
        colors="white",
        linewidths=0.7,
        alpha=0.8,
        zorder=2,
    )

    # Vertical and horizontal nearest-integer partitions.
    for xb in half_integer_boundaries(xlim):
        ax.axvline(xb, color="black", linewidth=1.0, alpha=0.85, zorder=3)
    for yb in half_integer_boundaries(ylim):
        ax.axhline(
            yb,
            color="black",
            linewidth=1.0,
            linestyle="--",
            alpha=0.85,
            zorder=3,
        )

    annotate_triplets(ax, xlim, ylim)

    cbar = fig.colorbar(mesh, ax=ax, ticks=np.arange(vmin, vmax + 1), pad=0.025)
    cbar.set_label(r"$\mathrm{round}(x+y)$")

    legend_handles = [
        Line2D([0], [0], color="black", linewidth=1.2,
               label=r"boundaries of $\mathrm{round}(x)$"),
        Line2D([0], [0], color="black", linewidth=1.2, linestyle="--",
               label=r"boundaries of $\mathrm{round}(y)$"),
        Line2D([0], [0], color="white", linewidth=1.2,
               label=r"boundaries of $\mathrm{round}(x+y)$"),
        Patch(
            facecolor="#ffd166",
            edgecolor="#9c6500",
            alpha=0.55,
            label=r"tuple sum $=0$",
        ),
    ]
    ax.legend(handles=legend_handles, loc="upper right", framealpha=0.95)

    ax.set_title(
        r"Regions labeled by "
        r"$(\mathrm{round}(x+y),\,\mathrm{round}(x),\,\mathrm{round}(y))$"
    )
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal")
    ax.grid(False)

    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_round_regions(
    xlim: tuple[float, float] = (-3.0, 3.0),
    ylim: tuple[float, float] = (-3.0, 3.0),
    resolution: int = 900,
    cmap_name: str = "Spectral_r",
    media_dir: str | Path = "media",
) -> None:
    media = Path(media_dir)
    media.mkdir(parents=True, exist_ok=True)

    x = np.linspace(*xlim, resolution)
    y = np.linspace(*ylim, resolution)
    X, Y = np.meshgrid(x, y)

    Zx = nearest_integer(X)
    Zy = nearest_integer(Y)
    Zsum = nearest_integer(X + Y)

    save_scalar_region_plot(
        X, Y, Zsum,
        title=r"Regions of $\mathrm{round}(x+y)$",
        colorbar_label=r"$\mathrm{round}(x+y)$",
        output=media / "round_x_plus_y.png",
        xlim=xlim,
        ylim=ylim,
        cmap_name=cmap_name,
    )
    save_scalar_region_plot(
        X, Y, Zx,
        title=r"Regions of $\mathrm{round}(x)$",
        colorbar_label=r"$\mathrm{round}(x)$",
        output=media / "round_x.png",
        xlim=xlim,
        ylim=ylim,
        cmap_name=cmap_name,
    )
    save_scalar_region_plot(
        X, Y, Zy,
        title=r"Regions of $\mathrm{round}(y)$",
        colorbar_label=r"$\mathrm{round}(y)$",
        output=media / "round_y.png",
        xlim=xlim,
        ylim=ylim,
        cmap_name=cmap_name,
    )
    save_combined_plot(
        X, Y, Zsum,
        output=media / "round_regions_combined.png",
        xlim=xlim,
        ylim=ylim,
        cmap_name=cmap_name,
    )

    for path in sorted(media.glob("round*.png")):
        print(path)


if __name__ == "__main__":
    plot_round_regions()
