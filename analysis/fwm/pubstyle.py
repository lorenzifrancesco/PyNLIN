"""Shared figure style for publication-quality output.

Rationale
---------
A figure is legible at the size it is *printed*, not the size it is drawn.
Generating a 18 in wide figure and including it with ``width=\\linewidth`` in a
3.49 in LaTeX column shrinks every font by the same factor: a 16 pt tick label
lands on the page at 3.2 pt.  The only fix is to generate the figure at the
width it will occupy and never rescale it afterwards.

This module centralises the sizes so the plotting scripts cannot drift apart.
Screen defaults are unchanged; publication mode is opt-in per script via
``--publication``.

Usage
-----
    from . import pubstyle            # or: import pubstyle

    parser = argparse.ArgumentParser()
    pubstyle.add_argument(parser)
    args = parser.parse_args()
    mode = pubstyle.apply(args)       # "screen" | "column" | "text"

    fig, axes = plt.subplots(2, 3, figsize=pubstyle.figsize(18.5, 11.0, mode))
    ...
    pubstyle.savefig(fig, out_path, mode)

Target widths were measured, not recalled::

    IEEEtran twocolumn:  \\columnwidth = 252 pt = 3.487 in = 88.6 mm
                         \\textwidth   = 516 pt = 7.140 in = 181.4 mm

Re-measure for any other class with ``\\typeout{\\the\\columnwidth}`` and divide
by 72.27 pt/in.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

PT_PER_IN = 72.27

#: IEEEtran two-column single-column width [in].
COLUMN_WIDTH_IN = 252.0 / PT_PER_IN  # 3.487
#: IEEEtran full text width (two-column-spanning figure*) [in].
TEXT_WIDTH_IN = 516.0 / PT_PER_IN  # 7.140

WIDTHS = {"column": COLUMN_WIDTH_IN, "text": TEXT_WIDTH_IN}

#: Base text size on the page [pt].  The manuscript-mechanics rule is ~8 pt,
#: never below 7 pt at final size, never above the body text.
BASE_PT = 8.0

_PUBLICATION_RC = {
    "font.size": BASE_PT,
    "axes.labelsize": BASE_PT,
    "axes.titlesize": BASE_PT,
    "xtick.labelsize": BASE_PT - 0.5,
    "ytick.labelsize": BASE_PT - 0.5,
    "legend.fontsize": BASE_PT - 1.0,
    "legend.title_fontsize": BASE_PT - 1.0,
    "figure.titlesize": BASE_PT + 1.0,
    # Lines and markers scale with the text, not with the canvas.
    "lines.linewidth": 1.0,
    "lines.markersize": 3.0,
    "axes.linewidth": 0.6,
    "grid.linewidth": 0.4,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.minor.size": 1.5,
    "ytick.minor.size": 1.5,
    # Keep text as text in the PDF (Type 42 = TrueType, selectable, embedded).
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    # Raster layers (pcolormesh, imshow) need >=300 dpi at final size.
    "savefig.dpi": 600,
    "figure.dpi": 600,
    # A tight bbox trims the canvas, so the saved width no longer equals the
    # requested width and \linewidth rescales it again.  Lay out instead.
    "savefig.bbox": None,
    # NB: no layout engine is forced here.  The scripts variously call
    # tight_layout() or pass constrained_layout=True, and both preserve the
    # requested figure size; forcing a third choice only raises warnings.
}


#: Minimum raster resolution at final size.  A heatmap saved at 200 dpi and
#: 3.49 in wide is only 697 px across, below the 300 dpi print floor.
MIN_DPI = 600


def dpi(screen_dpi: int, mode: str | None = None) -> int:
    """Raise the save dpi to the print floor in publication mode."""
    mode = _MODE if mode is None else mode
    return max(screen_dpi, MIN_DPI) if mode in WIDTHS else screen_dpi


def add_argument(parser) -> None:
    """Register ``--publication`` on an existing argparse parser."""
    parser.add_argument(
        "--publication",
        choices=("column", "text"),
        default=None,
        metavar="WIDTH",
        help="render at final printed size for the manuscript: 'column' "
             f"({COLUMN_WIDTH_IN:.2f} in, IEEEtran one column) or 'text' "
             f"({TEXT_WIDTH_IN:.2f} in, figure*). Default: screen sizes.",
    )


#: Mode set by the most recent :func:`apply` call, so the helpers below can be
#: used deep inside plotting functions without threading the mode through every
#: signature.
_MODE = "screen"


def current() -> str:
    """The active mode: ``"screen"``, ``"column"`` or ``"text"``."""
    return _MODE


def apply(args_or_mode) -> str:
    """Apply the publication rcParams if requested; return the active mode.

    Accepts either the parsed argparse namespace or the mode string itself.
    Returns ``"screen"`` when publication mode is off, leaving every script's
    own rcParams untouched.
    """
    global _MODE
    mode = getattr(args_or_mode, "publication", args_or_mode)
    if mode not in WIDTHS:
        _MODE = "screen"
        return _MODE
    matplotlib.rcParams.update(_PUBLICATION_RC)
    _MODE = mode
    return mode


def figsize(screen_w: float, screen_h: float, mode: str | None = None) -> tuple[float, float]:
    """Map a screen figsize onto the target width, preserving aspect ratio.

    In screen mode the input is returned unchanged, so wrapping an existing
    ``figsize=(w, h)`` is a no-op until ``--publication`` is passed.
    """
    mode = _MODE if mode is None else mode
    if mode not in WIDTHS:
        return (screen_w, screen_h)
    width = WIDTHS[mode]
    return (width, width * screen_h / screen_w)


def panel_width_in(ncols: int, mode: str | None = None) -> float:
    """Width available to each column of panels [in] — a legibility check.

    Below roughly 1.5 in a panel cannot carry readable tick labels at 8 pt,
    however correct the font size nominally is.
    """
    mode = _MODE if mode is None else mode
    return WIDTHS.get(mode, COLUMN_WIDTH_IN) / max(ncols, 1)


def savefig(fig, path: Path, mode: str | None = None, **kwargs) -> None:
    """Save without a tight bbox in publication mode, so width is exact."""
    mode = _MODE if mode is None else mode
    if mode in WIDTHS:
        kwargs.pop("bbox_inches", None)
        kwargs.pop("dpi", None)
    fig.savefig(path, **kwargs)
