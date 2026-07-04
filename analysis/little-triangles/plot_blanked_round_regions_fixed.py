#!/usr/bin/env python3
"""
Plot regions associated with round(x), round(y), and round(x+y), while
blanking a strip of half-width f around every rounding transition.

A transition of round(t) occurs at t = n + 1/2. A point (x, y) is blanked
whenever at least one of x, y, or x+y lies within distance f of a transition.

The combined plot labels each surviving region with the triplet

    (round(x+y), round(x), round(y)).

All figures are saved in media/.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm


def nearest_integer(z: np.ndarray) -> np.ndarray:
    """Nearest integer, using NumPy's round-to-even convention at ties."""
    return np.rint(z).astype(int)


def distance_to_round_transition(z: np.ndarray) -> np.ndarray:
    """
    Distance from z to the nearest transition of round(z).

    The transitions are the half-integers n + 1/2. Since the pattern has
    period 1, the distance is

        | (z mod 1) - 1/2 |.
    """
    return np.abs(np.mod(z, 1.0) - 0.5)


def blanked_round(
    z: np.ndarray,
    f: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return round(z) and a Boolean validity mask.

    valid[i] is False when z[i] lies within distance f of a half-integer
    transition of round(z).

    Parameters
    ----------
    z:
        Input array.
    f:
        Half-width of the blank strip around each transition.
        It must satisfy 0 <= f < 0.5.
    """
    if not 0.0 <= f < 0.5:
        raise ValueError("f must satisfy 0 <= f < 0.5")

    values = nearest_integer(z)
    valid = distance_to_round_transition(z) >= f
    return values, valid


def discrete_cmap_and_norm(
    values: np.ndarray,
    cmap_name: str,
) -> tuple:
    """Construct a discrete colormap and matching normalization."""
    finite_values = values[np.isfinite(values)]

    if finite_values.size == 0:
        raise ValueError("No unmasked values remain. Reduce f.")

    vmin = int(np.min(finite_values))
    vmax = int(np.max(finite_values))

    levels = np.arange(vmin - 0.5, vmax + 1.5, 1.0)
    cmap = plt.get_cmap(cmap_name, vmax - vmin + 1)
    norm = BoundaryNorm(levels, cmap.N)

    return cmap, norm, vmin, vmax


def save_single_field(
    X: np.ndarray,
    Y: np.ndarray,
    values: np.ndarray,
    valid: np.ndarray,
    title: str,
    colorbar_label: str,
    output_path: Path,
    cmap_name: str,
) -> None:
    """Save one masked discrete field."""
    masked_values = np.ma.masked_where(~valid, values)
    cmap, norm, vmin, vmax = discrete_cmap_and_norm(
        masked_values.astype(float).filled(np.nan),
        cmap_name,
    )

    fig, ax = plt.subplots(figsize=(8.0, 7.0))

    mesh = ax.pcolormesh(
        X,
        Y,
        masked_values,
        cmap=cmap,
        norm=norm,
        shading="nearest",
        rasterized=True,
    )

    cbar = fig.colorbar(
        mesh,
        ax=ax,
        ticks=np.arange(vmin, vmax + 1),
        pad=0.025,
    )
    cbar.set_label(colorbar_label)

    ax.set(
        xlabel=r"$x$",
        ylabel=r"$y$",
        title=title,
    )
    ax.set_aspect("equal")
    ax.set_xlim(X.min(), X.max())
    ax.set_ylim(Y.min(), Y.max())

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def region_label_positions(
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    f: float,
) -> list[tuple[float, float, tuple[int, int, int]]]:
    """
    Find one safe label point in each nonblank connected polygonal region.

    Candidate points are generated analytically from intersections of the
    central lines of the three rounding cells. They are then filtered using
    the same blanking condition as the plotted field.
    """
    xmin, xmax = xlim
    ymin, ymax = ylim

    ix_min = int(np.floor(xmin)) - 2
    ix_max = int(np.ceil(xmax)) + 2
    iy_min = int(np.floor(ymin)) - 2
    iy_max = int(np.ceil(ymax)) + 2
    is_min = int(np.floor(xmin + ymin)) - 2
    is_max = int(np.ceil(xmax + ymax)) + 2

    labels: list[tuple[float, float, tuple[int, int, int]]] = []

    # For each integer triplet, search a small set of natural interior
    # candidates and keep the one furthest from all three transition sets.
    for rx in range(ix_min, ix_max + 1):
        for ry in range(iy_min, iy_max + 1):
            for rs in range(is_min, is_max + 1):
                candidates = [
                    (float(rx), float(ry)),
                    (float(rx), float(rs - rx)),
                    (float(rs - ry), float(ry)),
                    (
                        0.5 * (rx + rs - ry),
                        0.5 * (ry + rs - rx),
                    ),
                ]

                best = None
                best_clearance = -np.inf

                for x0, y0 in candidates:
                    if not (xmin < x0 < xmax and ymin < y0 < ymax):
                        continue

                    vals = nearest_integer(
                        np.array([x0, y0, x0 + y0], dtype=float)
                    )
                    triplet = (int(vals[2]), int(vals[0]), int(vals[1]))

                    if triplet != (rs, rx, ry):
                        continue

                    distances = distance_to_round_transition(
                        np.array([x0, y0, x0 + y0], dtype=float)
                    )
                    clearance = float(np.min(distances))

                    if clearance < f:
                        continue

                    if clearance > best_clearance:
                        best = (x0, y0, triplet)
                        best_clearance = clearance

                if best is not None:
                    labels.append(best)

    # Remove duplicate labels caused by different candidate constructions.
    unique: dict[tuple[int, int, int], tuple[float, float, tuple[int, int, int]]] = {}
    for item in labels:
        unique[item[2]] = item

    return list(unique.values())


def plot_round_regions_with_blanking(
    f: float = 0.08,
    xlim: tuple[float, float] = (-4.0, 4.0),
    ylim: tuple[float, float] = (-4.0, 4.0),
    resolution: int = 1200,
    cmap_name: str = "Spectral_r",
    output_dir: str | Path = "media",
) -> None:
    """
    Generate the three individual fields and the combined labeled plot.

    The common combined mask is

        valid_x AND valid_y AND valid_(x+y).
    """
    if not 0.0 <= f < 0.5:
        raise ValueError("f must satisfy 0 <= f < 0.5")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    x = np.linspace(*xlim, resolution)
    y = np.linspace(*ylim, resolution)
    X, Y = np.meshgrid(x, y)

    round_x, valid_x = blanked_round(X, f)
    round_y, valid_y = blanked_round(Y, f)
    round_sum, valid_sum = blanked_round(X + Y, f)

    valid_all = valid_x & valid_y & valid_sum

    # Individual plots use their own masks.
    save_single_field(
        X,
        Y,
        round_x,
        valid_x,
        title=rf"Blanked regions of $\mathrm{{round}}(x)$, $f={f:g}$",
        colorbar_label=r"$\mathrm{round}(x)$",
        output_path=output_dir / "blanked_round_x.png",
        cmap_name=cmap_name,
    )
    save_single_field(
        X,
        Y,
        round_y,
        valid_y,
        title=rf"Blanked regions of $\mathrm{{round}}(y)$, $f={f:g}$",
        colorbar_label=r"$\mathrm{round}(y)$",
        output_path=output_dir / "blanked_round_y.png",
        cmap_name=cmap_name,
    )
    save_single_field(
        X,
        Y,
        round_sum,
        valid_sum,
        title=rf"Blanked regions of $\mathrm{{round}}(x+y)$, $f={f:g}$",
        colorbar_label=r"$\mathrm{round}(x+y)$",
        output_path=output_dir / "blanked_round_x_plus_y.png",
        cmap_name=cmap_name,
    )

    # Combined plot: color by round(x+y), but blank whenever any one of
    # x, y, or x+y is too close to a transition.
    combined_values = np.ma.masked_where(~valid_all, round_sum)
    cmap, norm, vmin, vmax = discrete_cmap_and_norm(
        combined_values.astype(float).filled(np.nan),
        cmap_name,
    )

    fig, ax = plt.subplots(figsize=(9.0, 7.8))

    mesh = ax.pcolormesh(
        X,
        Y,
        combined_values,
        cmap=cmap,
        norm=norm,
        shading="nearest",
        rasterized=True,
    )

    cbar = fig.colorbar(
        mesh,
        ax=ax,
        ticks=np.arange(vmin, vmax + 1),
        pad=0.025,
    )
    cbar.set_label(r"$\mathrm{round}(x+y)$")

    for x0, y0, triplet in region_label_positions(xlim, ylim, f):
        ax.text(
            x0,
            y0,
            rf"$({triplet[0]},{triplet[1]},{triplet[2]})$",
            ha="center",
            va="center",
            fontsize=7.5,
            bbox={
                "boxstyle": "round,pad=0.16",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.74,
            },
        )

    ax.set(
        xlabel=r"$x$",
        ylabel=r"$y$",
        title=(
            rf"Combined blanked regions, $f={f:g}$"
            "\n"
            r"labels: $(\mathrm{round}(x+y),"
            r"\mathrm{round}(x),\mathrm{round}(y))$"
        ),
    )
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")

    fig.tight_layout()
    fig.savefig(
        output_dir / "blanked_round_regions_combined.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    print(f"Saved figures in: {output_dir.resolve()}")


if __name__ == "__main__":
    plot_round_regions_with_blanking(
        f=0.08,
        xlim=(-4.0, 4.0),
        ylim=(-4.0, 4.0),
        resolution=1200,
        output_dir="media",
    )
