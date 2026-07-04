#!/usr/bin/env python3
"""Standalone plotter for per-band DGD and GVD histograms.

Usage
-----
    python analysis/plot_band_histograms.py <config.toml>              # all bands
    python analysis/plot_band_histograms.py <config.toml> --bands O    # O-band only
    python analysis/plot_band_histograms.py <config.toml> --bands O E C

The script loads a *System* from the given TOML, computes the
L/L_W (DGD) and L/L_D (GVD) distributions for every optical band
present in the WDM configuration, and saves two figures:

    media/band_histograms/dgd_histogram_per_band.pdf
    media/band_histograms/gvd_histogram_per_band.pdf
"""
import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
for path in (REPO_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from analysis.methods.plotting import plot_band_dgd_gvd_histograms
from pynlin.system import System


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot per-band DGD (L/LW) and GVD (L/LD) histograms."
    )
    parser.add_argument("config", type=str, help="Path to system TOML config.")
    parser.add_argument(
        "--bands",
        type=str,
        nargs="*",
        default=None,
        help="Space-separated band names to plot (e.g. O E S C L U). Default: all bands.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory for the PDF figures (default: media/band_histograms).",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    system = System.from_toml(config_path)
    print(f"Loaded system from {config_path}")
    print(system.summary())

    out_dir = Path(args.out_dir) if args.out_dir else REPO_ROOT / "media" / "band_histograms"
    plot_band_dgd_gvd_histograms(system, out_dir=out_dir, bands=args.bands)


if __name__ == "__main__":
    main()
