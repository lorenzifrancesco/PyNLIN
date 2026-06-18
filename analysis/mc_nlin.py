"""
Monte Carlo (MC) NLIN benchmark: single-span, 5-channel SMF.

Reads results produced by the studies system (studies.py) and provides
plotting and CLI convenience. Computation lives in analysis/methods/mc.py
and analysis/methods/td.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from loguru import logger as lg


def _print_comparison_table(freqs_hz: np.ndarray,
                            nlin_td_w: np.ndarray,
                            nlin_pcfm_w: np.ndarray,
                            signal_power_w: np.ndarray) -> None:
    freqs_thz = np.asarray(freqs_hz, dtype=float).reshape(-1) * 1e-12
    nlin_td = np.asarray(nlin_td_w, dtype=float).reshape(-1)
    nlin_pcfm = np.asarray(nlin_pcfm_w, dtype=float).reshape(-1)
    sig = np.asarray(signal_power_w, dtype=float).reshape(-1)
    if freqs_thz.size != nlin_td.size or nlin_td.size != nlin_pcfm.size or sig.size != nlin_td.size:
        raise ValueError("Comparison table arrays must have the same size.")
    denom_td = np.maximum(nlin_td, 1e-18)
    denom_pcfm = np.maximum(nlin_pcfm, 1e-18)
    gsnr_td = 10.0 * np.log10(sig / denom_td)
    gsnr_pcfm = 10.0 * np.log10(sig / denom_pcfm)
    ratio = nlin_pcfm / np.maximum(nlin_td, 1e-30)
    delta_db = 10.0 * np.log10(np.maximum(ratio, 1e-30))
    dgsnr = gsnr_pcfm - gsnr_td

    print("Comparison (TD vs PCFM):")
    print("ch  f_THz   NLIN_TD(W)   NLIN_PCFM(W)   ratio  dNLIN(dB)  GSNR_TD(dB)  GSNR_PCFM(dB)  dGSNR(dB)")
    for idx, f_thz in enumerate(freqs_thz):
        print(
            f"{idx:2d}  {f_thz:6.3f}  {nlin_td[idx]:11.3e}  {nlin_pcfm[idx]:13.3e}"
            f"  {ratio[idx]:7.3f}  {delta_db[idx]:9.3f}  {gsnr_td[idx]:11.3f}"
            f"  {gsnr_pcfm[idx]:13.3f}  {dgsnr[idx]:9.3f}"
        )


def _plot_nsr_vs_length(csv_path: Path, out_pdf: Path) -> None:
    """Plot NSR vs length from a sweep CSV with td_nlin_w and nlin_16qam_w columns."""
    if not csv_path.exists():
        lg.warning("CSV not found: {}", csv_path)
        return
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    if data.ndim == 1:
        data = data[None, :]
    lengths_km = data[:, 0]
    td_nlin = data[:, 4]
    mc_16qam = data[:, -1]
    has_chi = data.shape[1] >= 10
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    ax.plot(lengths_km, -10.0 * np.log10(np.maximum(td_nlin, 1e-18)),
            marker="o", ms=3, lw=0.9, label="TD total")
    if has_chi:
        ax.plot(lengths_km, -10.0 * np.log10(np.maximum(mc_16qam, 1e-18)),
                marker="^", ms=3, lw=0.9, label="MC (16-QAM)")
    ax.set_xlabel("Length [km]")
    ax.set_ylabel(r"$\mathrm{NSR}\;[\mathrm{dB}]$")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=7)
    fig.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, dpi=300)
    plt.close(fig)
    lg.success("Plot saved to {}", out_pdf)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MC NLIN benchmark: plot results from studies.")
    parser.add_argument("--csv", type=str, default="results/length_sweep/sweep_length.csv",
                        help="Path to sweep CSV produced by studies.")
    parser.add_argument("--out", type=str, default="media/mc/nsr_vs_length.pdf",
                        help="Output PDF path.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    _plot_nsr_vs_length(Path(args.csv), Path(args.out))
