"""Load a saved fullband MC NPZ diagnostic and replot."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_from_npz(npz_path: Path, out_dir: Path | None = None) -> None:
    d = np.load(npz_path, allow_pickle=True)
    freq_thz = np.asarray(d["target_frequencies"], dtype=float) * 1e-12
    xpm = np.asarray(d["xpm"], dtype=float).reshape(-1)
    fwm = np.asarray(d["fwm"], dtype=float).reshape(-1)
    total = np.asarray(d["total"], dtype=float).reshape(-1)
    support = np.asarray(d["fwm_support_count"], dtype=float).reshape(-1)
    sampled = np.asarray(d["fwm_tuple_count"], dtype=float).reshape(-1)

    zdw_thz = None
    zdw = d.get("meta_zdw_frequency")
    if zdw is not None and np.isfinite(float(zdw.item())):
        zdw_thz = float(zdw.item()) * 1e-12

    decimation = d.get("meta_decimation", None)
    n_channels_dec = d.get("meta_n_channels_decimated", None)

    if out_dir is None:
        out_dir = npz_path.parent
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(6.2, 5.2), sharex=True)

    axes[0].plot(freq_thz, xpm, marker="o", lw=1.0, label="XPM MC")
    axes[0].plot(freq_thz, fwm, marker="s", lw=1.0, label="FWM Dar MC")
    axes[0].plot(freq_thz, total, marker="^", lw=1.0, label="total")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("prefactor-free sum [m$^2$]")
    axes[0].grid(True, which="both", alpha=0.25)
    axes[0].legend(fontsize=8, frameon=False)

    axes[1].bar(freq_thz, support, width=0.02, label="support-pruned")
    axes[1].bar(freq_thz, sampled, width=0.012, label="sampled")
    axes[1].set_xlabel("target frequency [THz]")
    axes[1].set_ylabel("evaluated FWM tuples")
    axes[1].grid(True, axis="y", alpha=0.25)
    axes[1].legend(fontsize=8, frameon=False)

    if zdw_thz is not None:
        for ax in axes:
            ax.axvline(zdw_thz, color="crimson", lw=0.8, ls=":", alpha=0.7)
        axes[1].scatter(
            [zdw_thz], [0.0],
            marker="*", s=110, color="crimson", edgecolor="black",
            linewidth=0.4, transform=axes[1].get_xaxis_transform(),
            clip_on=False, zorder=5, label="ZDW",
        )
        axes[1].annotate(
            "ZDW", xy=(zdw_thz, 0.0),
            xycoords=axes[1].get_xaxis_transform(),
            xytext=(0, -18), textcoords="offset points",
            ha="center", va="top", fontsize=8, color="crimson",
        )

    dec_label = f"decimation={decimation}" if decimation is not None else ""
    ch_label = f"channels={n_channels_dec}" if n_channels_dec is not None else ""
    title = "Prefactor-free fullband MC diagnostic"
    if dec_label or ch_label:
        title += f"\n{dec_label}, {ch_label}"
    fig.suptitle(title)
    fig.tight_layout()

    out_path = out_dir / "fullband_mc_prefactor_free.pdf"
    fig.savefig(out_path, dpi=300)
    fig.savefig(out_path.with_suffix(".png"), dpi=300)
    plt.close(fig)
    print(f"Saved {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot fullband MC diagnostic from saved NPZ.")
    parser.add_argument("npz_path", type=Path)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()
    plot_from_npz(args.npz_path.resolve(), args.out_dir)


if __name__ == "__main__":
    main()
