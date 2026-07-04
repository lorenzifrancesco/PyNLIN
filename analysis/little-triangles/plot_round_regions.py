#!/usr/bin/env python3
"""
Plot the regions defined by round(x + y), with the partitions induced by
round(x) and round(y) superimposed.

The diagonal colored bands correspond to round(x + y).
Vertical solid lines separate regions of constant round(x).
Horizontal dashed lines separate regions of constant round(y).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm


def nearest_integer(z: np.ndarray) -> np.ndarray:
    """Round to the nearest integer.

    np.rint uses NumPy's round-to-even convention exactly at half-integers.
    This convention affects only the boundary lines, not the open regions.
    """
    return np.rint(z).astype(int)


def plot_round_regions(
    xlim: tuple[float, float] = (-4.0, 4.0),
    ylim: tuple[float, float] = (-4.0, 4.0),
    resolution: int = 900,
    cmap_name: str = "Spectral_r",
    output: str | None = None,
) -> None:
    x = np.linspace(*xlim, resolution)
    y = np.linspace(*ylim, resolution)
    X, Y = np.meshgrid(x, y)

    Z = nearest_integer(X + Y)

    zmin = int(Z.min())
    zmax = int(Z.max())
    levels = np.arange(zmin - 0.5, zmax + 1.5, 1)

    cmap = plt.get_cmap(cmap_name, zmax - zmin + 1)
    norm = BoundaryNorm(levels, cmap.N)

    fig, ax = plt.subplots(figsize=(8.2, 7.0), constrained_layout=True)

    mesh = ax.pcolormesh(
        X,
        Y,
        Z,
        cmap=cmap,
        norm=norm,
        shading="nearest",
        rasterized=True,
    )

    # Boundaries where round(x) changes: x = n + 1/2.
    x_boundaries = np.arange(
        np.ceil(xlim[0] - 0.5) + 0.5,
        xlim[1],
        1.0,
    )
    for xb in x_boundaries:
        ax.axvline(xb, linewidth=0.9, alpha=0.75, color="black")

    # Boundaries where round(y) changes: y = n + 1/2.
    y_boundaries = np.arange(
        np.ceil(ylim[0] - 0.5) + 0.5,
        ylim[1],
        1.0,
    )
    for yb in y_boundaries:
        ax.axhline(
            yb,
            linewidth=0.9,
            linestyle="--",
            alpha=0.75,
            color="black",
        )

    # Optional diagonal boundary emphasis for round(x + y).
    ax.contour(
        X,
        Y,
        X + Y,
        levels=levels,
        linewidths=0.45,
        colors="white",
        alpha=0.55,
    )

    cbar = fig.colorbar(
        mesh,
        ax=ax,
        ticks=np.arange(zmin, zmax + 1),
        pad=0.025,
    )
    cbar.set_label(r"$\mathrm{round}(x+y)$")

    # Legend proxies.
    ax.plot([], [], color="black", linewidth=1.2,
            label=r"boundaries of $\mathrm{round}(x)$")
    ax.plot([], [], color="black", linewidth=1.2, linestyle="--",
            label=r"boundaries of $\mathrm{round}(y)$")

    ax.set(
        xlabel=r"$x$",
        ylabel=r"$y$",
        xlim=xlim,
        ylim=ylim,
        title=(
            r"Regions of $\mathrm{round}(x+y)$, with "
            r"$\mathrm{round}(x)$ and $\mathrm{round}(y)$ partitions"
        ),
    )
    ax.set_aspect("equal")
    ax.legend(loc="upper right", framealpha=0.92)
    ax.grid(False)

    if output:
        fig.savefig(output, dpi=300, bbox_inches="tight")

    plt.show()


if __name__ == "__main__":
    plot_round_regions(output="round_regions.png")
