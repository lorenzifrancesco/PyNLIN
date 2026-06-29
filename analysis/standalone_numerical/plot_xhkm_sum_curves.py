from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pynlin.methods.td.reference_curves import load_xhkm_sum_reference_curves


def plot_xhkm_sum_curves(path: str | Path, out_dir: str | Path = "media/n-PC") -> tuple[Path, Path, Path]:
    dataset = load_xhkm_sum_reference_curves(path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    llw = np.asarray(dataset["llw_grid"], dtype=float)
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    ax.plot(llw, dataset["ref_n1"], lw=1.0, label=r"$N_1$ (prefactor-free chi1 sum)")
    ax.plot(llw, dataset["ref_n2"], lw=1.0, label=r"$N_2$ (prefactor-free chi2 sum)")
    ax.plot(llw, dataset["ref_n_2pc"], lw=1.0, label="2PC slice")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$L/L_W$")
    ax.set_ylabel(r"$N\,T^2/L^2$")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=7)
    fig.tight_layout()
    main_path = out_dir / "xhkm_n1_n2_curves.pdf"
    fig.savefig(main_path, dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.6, 2.2))
    ax.plot(llw, dataset["n2_over_n1"], lw=1.0)
    ax.set_xscale("log")
    ax.set_xlabel(r"$L/L_W$")
    ax.set_ylabel(r"$N_2/N_1$")
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    ratio_path = out_dir / "xhkm_n2_over_n1.pdf"
    fig.savefig(ratio_path, dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.8, 2.8))
    ax.plot(llw, dataset["ref_n_2pc"], lw=1.0, marker="*", ms=5, label="2PC")
    ax.plot(llw, dataset["ref_n_3pca"], lw=1.0, marker="o", ms=3, label="3PCa: h=0, k!=m")
    ax.plot(llw, dataset["ref_n_3pcb"], lw=1.0, marker="s", ms=3, label="3PCb: h!=0, k=m")
    ax.plot(llw, dataset["ref_n_3pc_other"], lw=0.9, marker="^", ms=3, label="3PC other")
    ax.plot(llw, dataset["ref_n_4pc"], lw=1.0, marker="D", ms=3, label="4PC")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$L/L_W$")
    ax.set_ylabel(r"$N\,T^2/L^2$")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=6)
    fig.tight_layout()
    decomp_path = out_dir / "xhkm_collision_decomposition.pdf"
    fig.savefig(decomp_path, dpi=300)
    plt.close(fig)
    return main_path, ratio_path, decomp_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot prefactor-free Xhkm N1/N2 reference curves.")
    parser.add_argument("path", type=str, help="Path to xhkm_sum_ref_curve .npz dataset.")
    parser.add_argument("--out-dir", type=str, default="media/n-PC", help="Output directory for plots.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    plot_xhkm_sum_curves(args.path, args.out_dir)
