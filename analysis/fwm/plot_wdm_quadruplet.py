"""Draw a WDM grid with a four-wave-mixing quadruplet highlighted.

Schematic only (not tied to any fast_nlin/pynlin data structures): draws a
comb of WDM bands in normalized frequency nu/Omega_0 (Omega_0 = channel
spacing, same convention as analysis/standalone_analytical/the-domain-in-
freq-space-*.py) and marks the four frequencies of an FWM quadruplet
(nu1, nu2, nu3, nu4) with arrows rising from the frequency axis. By
convention nu1 and nu3 enter the FWM product with a plus sign (drawn red)
and nu2, nu4 enter with a minus sign / complex conjugate (drawn blue).
nu1, nu2, nu3 need not sit on integer channel centers (a pump can be
anywhere inside its channel's occupied bandwidth); nu4 must fall inside the
COI channel's own bandwidth, i.e. |nu4| <= bandwidth_norm/2.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from loguru import logger as lg

from analysis.methods.plotting import GNUPLOT_BLUE, GNUPLOT_GRAY, GNUPLOT_RED
from analysis.methods.figure_size import scale_figsize_to_ieee_column

LABELS = (r"$\nu_1$", r"$\nu_2$", r"$\nu_3$", r"$\nu_4$")
COLORS = (GNUPLOT_RED, GNUPLOT_BLUE, GNUPLOT_RED, GNUPLOT_BLUE)


def random_compatible_quadruplet(
    max_offset: int,
    rng: np.random.Generator | None = None,
) -> tuple[int, int, int, int]:
    """Sample a frequency-matched quadruplet (nu1, nu2, nu3, 0) by rejection.

    nu1, nu3 are drawn uniformly from nonzero channel offsets in
    [-max_offset, max_offset]; nu2 = nu1 + nu3 is accepted only if it also
    falls (nonzero) inside that range, so nu1 - nu2 + nu3 - 0 = 0 always
    holds.
    """
    if max_offset < 2:
        raise ValueError(f"max_offset must be >= 2, got {max_offset}")
    if rng is None:
        rng = np.random.default_rng()
    candidates = [o for o in range(-max_offset, max_offset + 1) if o != 0]
    while True:
        n1 = int(rng.choice(candidates))
        n3 = int(rng.choice(candidates))
        n2 = n1 + n3
        if n2 != 0 and abs(n2) <= max_offset:
            return (n1, n2, n3, 0)


def draw_wdm_quadruplet(
    quadruplet: tuple[float, float, float, float],
    bandwidth_norm: float = 0.6,
    n_channels: int | None = None,
    margin_channels: int = 1,
    freq_match_tol: float = 1e-6,
) -> plt.Figure:
    """Render the WDM comb + quadruplet arrows and return the figure.

    ``quadruplet`` gives (nu1, nu2, nu3, nu4) as frequency offsets from the
    COI, normalized by the channel spacing Omega_0 (a channel at index n
    sits at nu/Omega_0 = n; ``bandwidth_norm`` is B/Omega_0), matching the
    convention in the-domain-in-freq-space-*.py. nu1, nu2, nu3 need not be
    integers: a pump can sit anywhere inside its channel's occupied
    bandwidth. nu4 must land inside the COI channel's own bandwidth,
    i.e. |nu4| <= bandwidth_norm/2, since that is what makes it observable
    at the COI.
    """
    nu1, nu2, nu3, nu4 = quadruplet
    half_band = bandwidth_norm / 2.0
    if abs(nu4) > half_band + freq_match_tol:
        raise ValueError(
            f"nu4={nu4:g} must fall inside the COI band "
            f"(|nu4| <= bandwidth_norm/2 = {half_band:g})"
        )
    if bandwidth_norm > 1.0:
        lg.warning(
            f"bandwidth_norm={bandwidth_norm} exceeds 1 (the channel spacing "
            "Omega_0); bands will overlap in the drawing."
        )

    freq_residual = nu1 - nu2 + nu3 - nu4
    if abs(freq_residual) > freq_match_tol:
        lg.warning(
            f"quadruplet {quadruplet} does not satisfy frequency matching "
            f"(nu1 - nu2 + nu3 - nu4 = {freq_residual:.4g} != 0)"
        )

    offsets = (nu1, nu2, nu3, nu4)
    if n_channels is None:
        half = int(np.ceil(max(abs(o) for o in offsets))) + margin_channels
        low, high = -half, half
    else:
        half = n_channels // 2
        low, high = -half, n_channels - half - 1
        if min(offsets) < low or max(offsets) > high:
            raise ValueError(
                f"quadruplet {quadruplet} does not fit in a grid of "
                f"n_channels={n_channels} (index range [{low}, {high}])"
            )

    fig, ax = plt.subplots(figsize=scale_figsize_to_ieee_column(3.4, 2.0))

    band_height = 1.0
    for idx in range(low, high + 1):
        is_coi = idx == 0
        rect = Rectangle(
            (idx - bandwidth_norm / 2, 0.0),
            bandwidth_norm,
            band_height,
            facecolor="#e6e6e6" if not is_coi else "#fff2cc",
            edgecolor=GNUPLOT_GRAY,
            linewidth=1.1 if is_coi else 0.6,
        )
        ax.add_patch(rect)
        if is_coi:
            ax.text(
                idx,
                -0.62,
                "COI",
                ha="center",
                va="top",
                fontsize=7.5,
                color=GNUPLOT_GRAY,
            )

    arrow_top = band_height + 0.75
    for k, x in enumerate(offsets):
        ax.annotate(
            "",
            xy=(x, arrow_top),
            xytext=(x, 0.0),
            arrowprops=dict(arrowstyle="->", color=COLORS[k], lw=1.3),
            zorder=5,
        )
        ax.text(
            x,
            arrow_top + 0.08,
            LABELS[k],
            ha="center",
            va="bottom",
            fontsize=9.5,
            color=COLORS[k],
            zorder=5,
        )

    ax.set_xlim(low - 1, high + 1)
    ax.set_ylim(-1.1, arrow_top + 0.55)
    ax.tick_params(axis="x", pad=8)
    ax.set_xlabel(r"$\nu/\Omega_0$", fontsize=10.5, labelpad=22)
    ax.set_yticks([])
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_position(("data", 0.0))
    ax.set_title(
        rf"$B/\Omega_0={bandwidth_norm:g}$, "
        rf"$(\nu_1,\nu_2,\nu_3,\nu_4)/\Omega_0=({nu1:.2f},{nu2:.2f},{nu3:.2f},{nu4:.2f})$",
        fontsize=8.5,
    )

    fig.tight_layout(pad=0.4)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bandwidth-norm",
        type=float,
        default=0.6,
        help="Occupied bandwidth per channel as a fraction of the spacing Omega_0 (B/Omega_0).",
    )
    parser.add_argument(
        "--n-channels",
        type=int,
        default=None,
        help="Total channels to draw. Default: auto-sized around the quadruplet.",
    )
    parser.add_argument(
        "--quadruplet",
        type=float,
        nargs=4,
        metavar=("NU1", "NU2", "NU3", "NU4"),
        default=(2, 3, 1, 0),
        help="Frequency offsets from the COI, in nu/Omega_0 units, for "
        "(nu1, nu2, nu3, nu4). NU4 must fall within the COI band "
        "(|nu4| <= bandwidth_norm/2). Ignored if --random is set.",
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="Draw a randomly sampled frequency-matched quadruplet instead of --quadruplet.",
    )
    parser.add_argument(
        "--max-offset",
        type=int,
        default=4,
        help="Max |channel offset| when sampling with --random.",
    )
    parser.add_argument("--seed", type=int, default=None, help="RNG seed for --random.")
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "media" / "diagrams" / "wdm_quadruplet.svg",
        help="Output image path (extension selects the format, e.g. .svg/.pdf/.png).",
    )
    args = parser.parse_args()

    if args.random:
        quadruplet = random_compatible_quadruplet(
            args.max_offset, rng=np.random.default_rng(args.seed)
        )
        lg.info(f"Sampled random compatible quadruplet: {quadruplet}")
    else:
        quadruplet = tuple(args.quadruplet)

    fig = draw_wdm_quadruplet(
        quadruplet,
        bandwidth_norm=args.bandwidth_norm,
        n_channels=args.n_channels,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    lg.success(f"Saved WDM quadruplet diagram to {args.out}")


if __name__ == "__main__":
    main()
