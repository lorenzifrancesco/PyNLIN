"""Figure-7-style band spectrum on a density- and fill-preserving decimation.

Figure 7 of the theory note is the S5 prefactor-free XPM/FWM spectrum. At full
resolution that costs hours; this script produces the same view on the scaled
analogue defined by ``fullband_mc.decimated_system`` -- delta_f -> k delta_f
with B -> k B and P -> k P, which holds the filling factor B/delta_f, the
average power density P/delta_f and the support shift d fixed.

The result is a self-consistent *different* system, not an approximation to
the full-resolution one: absolute sums scale as k^2 and the per-target tuple
population is genuinely smaller. Read the shape, not the level.

Note this is NOT ``fast_s5_fullband.py --decimation k``: there ``--decimation``
thins the *target list* while the interferer grid stays full, which is the
operation section 9 endorses. This script scales the grid itself.

Output (media/lorenzi-fast and docs/source/_static/lorenzi-fast):
  scaled_band_spectrum_dec<k>.png / .npz
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from loguru import logger as lg

from analysis.log_init import init_logging
from pynlin.methods.td.fast_nlin import target_fast_sums
from pynlin.methods.td.fullband_mc import (
    _beta0_abs_from_fiber,
    decimated_system,
    estimate_zdw_frequency,
)
from pynlin.system import System

INK, MUTED, GRID = "#22262b", "#8b929a", "#d8d8d4"
BAND_RULE, BAND_TEXT = "#c9c9c3", "#6f757c"


def band_extents(system):
    """Named ITU band spans [THz], read from the grid's own metadata."""
    slices = getattr(system.wdm, "_band_slices", None)
    if not slices:
        return []
    grid = np.asarray(system.wdm.frequency_grid(), dtype=float)
    out = []
    for name, sl in slices.items():
        seg = grid[sl]
        if seg.size:
            out.append((name, seg.min() / 1e12, seg.max() / 1e12))
    return sorted(out, key=lambda r: r[1])


def draw_bands(ax, bands, label_axis):
    """Shade the occupied bands; letters along the top of one panel.

    Only the shaded spans carry channels.  Every boundary between two bands is
    a guard gap of 50-75 GHz -- far too narrow to resolve on a 57 THz axis, so
    the gaps appear as the rules themselves rather than as visible strips.
    They are where the narrow FWM notches sit.
    """
    if not bands:
        return
    for i, (name, lo, hi) in enumerate(bands):
        for a in ax:
            a.axvspan(lo, hi, color="#000000",
                      alpha=0.045 if i % 2 else 0.015, lw=0, zorder=0)
            a.axvline(lo, color=BAND_RULE, lw=0.9, zorder=0)
            a.axvline(hi, color=BAND_RULE, lw=0.9, zorder=0)
        label_axis.annotate(
            name, (0.5 * (lo + hi), 1.0), xycoords=("data", "axes fraction"),
            ha="center", va="bottom", fontsize=10, color=BAND_TEXT, weight="bold",
        )
C_XPM, C_FWM = "#1971c2", "#e8590c"


def main() -> None:
    init_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("input/studies.toml"))
    parser.add_argument("--out-dir", type=Path, default=Path("media/lorenzi-fast"))
    parser.add_argument(
        "--docs-dir", type=Path, default=Path("docs/source/_static/lorenzi-fast")
    )
    parser.add_argument("--decimation", type=int, default=8)
    parser.add_argument("--n-refine", type=int, default=256)
    args = parser.parse_args()

    system = System.from_toml(args.config)
    d = decimated_system(system, args.decimation)
    length = float(system.fiber_length)
    beta1_grid, beta2_grid = system.beta_grids(freqs=d.freqs)
    beta1 = np.asarray(beta1_grid[0], dtype=float)
    beta2 = np.asarray(beta2_grid[0], dtype=float)
    beta0 = _beta0_abs_from_fiber(system, d.freqs, beta1, beta2)
    zdw = float(estimate_zdw_frequency(system)) / 1e12
    bands = band_extents(system)

    lg.info(
        f"scaled grid k={d.factor}: {d.freqs.size} channels, "
        f"spacing {d.channel_spacing / 1e9:.2f} GHz, B {d.baud_rate / 1e9:.2f} GBd, "
        f"P {d.launch_power_dbm:+.2f} dBm, filling factor {d.filling_factor:.4f}"
    )

    xpm = np.zeros(d.freqs.size)
    fwm = np.zeros(d.freqs.size)
    t0 = time.time()
    for i in range(d.freqs.size):
        r = target_fast_sums(d.freqs, beta0, beta1, beta2, d.baud_rate, length,
                             i, n_refine=args.n_refine)
        xpm[i], fwm[i] = r.xpm, r.fwm
        if i % 20 == 0 or i == d.freqs.size - 1:
            lg.info(f"  [{i + 1}/{d.freqs.size}] {d.freqs[i] / 1e12:7.2f} THz "
                    f"xpm={r.xpm:.4e} fwm={r.fwm:.4e}  "
                    f"({time.time() - t0:.0f}s)")

    f_thz = d.freqs / 1e12
    order = np.argsort(f_thz)
    f_thz, xpm, fwm = f_thz[order], xpm[order], fwm[order]

    fig, axes = plt.subplots(3, 1, figsize=(10, 7.6), sharex=True,
                             gridspec_kw={"height_ratios": (2.1, 1, 0.16)},
                             constrained_layout=True)
    ax, rug = axes[:2], axes[2]
    ax[0].semilogy(f_thz, xpm, lw=1.6, color=C_XPM, label="XPM")
    ax[0].semilogy(f_thz, fwm, lw=1.6, color=C_FWM, label="strict FWM")
    ax[0].semilogy(f_thz, xpm + fwm, lw=1.0, color=INK, alpha=0.65, label="total")
    ax[0].set_ylabel("prefactor-free sum [m$^2$]")
    ax[0].legend(frameon=False, ncol=3)

    share = 100 * fwm / np.maximum(xpm + fwm, 1e-300)
    ax[1].plot(f_thz, share, lw=1.6, color=C_FWM)
    ax[1].set_ylabel("FWM share of total [\\%]")
    ax[1].set_ylim(0, 100)
    ax[1].set_yticks([0, 25, 50, 75, 100])
    ax[1].ticklabel_format(axis="y", style="plain", useOffset=False)
    rug.set_xlabel("target frequency [THz]")

    for a in ax:
        a.axvline(zdw, color=MUTED, ls=":", lw=1.6)
        a.ticklabel_format(axis="x", style="plain", useOffset=False)
        a.grid(True, which="both", color=GRID, lw=0.6, alpha=0.7)
        a.set_axisbelow(True)
        a.tick_params(colors=MUTED)
        a.yaxis.label.set_color(MUTED)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
    ax[1].xaxis.label.set_color(MUTED)
    # Occupancy rug. Each computed channel spans B, and neighbours are only
    # ~4 GHz apart -- far below one pixel on a 57 THz axis, so drawing them
    # individually produces aliased slivers rather than structure. Merge runs
    # of touching channels into blocks so only the real guard gaps (~54 GHz)
    # appear as breaks.
    half = 0.5 * d.baud_rate / 1e12
    spans = sorted((x - half, x + half) for x in f_thz)
    merge_below = 0.020                       # THz: above inter-channel, below guard
    blocks = [list(spans[0])]
    for lo, hi in spans[1:]:
        if lo - blocks[-1][1] <= merge_below:
            blocks[-1][1] = max(blocks[-1][1], hi)
        else:
            blocks.append([lo, hi])
    rug.broken_barh([(lo, hi - lo) for lo, hi in blocks], (0.0, 1.0),
                    facecolor="#3d4248", edgecolor="none")
    rug.set_ylim(0, 1)
    rug.set_yticks([])
    rug.set_ylabel("channels", rotation=0, ha="right", va="center",
                   fontsize=9, color=MUTED)
    for side in ("top", "right", "left"):
        rug.spines[side].set_visible(False)
    rug.tick_params(colors=MUTED)
    rug.ticklabel_format(axis="x", style="plain", useOffset=False)
    rug.annotate(f"{d.freqs.size} channels, {d.baud_rate / 1e9:.0f} GHz wide "
                 f"at {d.channel_spacing / 1e9:.0f} GHz pitch "
                 f"(breaks = guard gaps)",
                 (0.005, -1.5), xycoords="axes fraction", fontsize=8,
                 color=MUTED, ha="left", va="top")

    draw_bands(list(ax) + [rug], bands, ax[0])
    ax[0].annotate("ZDW", (zdw, 1.06), xycoords=("data", "axes fraction"),
                   ha="center", va="bottom", fontsize=9, color=MUTED)
    fig.suptitle(
        f"Band spectrum on the scaled grid $k={d.factor}$ "
        f"({d.freqs.size} ch, {d.baud_rate / 1e9:.0f} GBd, "
        f"$B/\\Delta f={d.filling_factor:.3f}$)", color=INK, fontsize=12)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.docs_dir.mkdir(parents=True, exist_ok=True)
    stem = f"scaled_band_spectrum_dec{d.factor}"
    np.savez(args.out_dir / f"{stem}.npz", f_thz=f_thz, xpm=xpm, fwm=fwm,
             zdw_thz=zdw, decimation=d.factor, baud_rate=d.baud_rate,
             filling_factor=d.filling_factor,
             launch_power_dbm=d.launch_power_dbm,
             band_names=np.array([b[0] for b in bands]),
             band_lo=np.array([b[1] for b in bands]),
             band_hi=np.array([b[2] for b in bands]))
    for path in (args.out_dir / f"{stem}.png", args.docs_dir / f"{stem}.png"):
        fig.savefig(path, dpi=200, facecolor="white")
        lg.info(f"wrote {path}")
    lg.success("scaled band spectrum saved")


if __name__ == "__main__":
    main()
