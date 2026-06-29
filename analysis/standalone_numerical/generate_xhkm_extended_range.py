"""Extend Nyquist Xhkm curves to L/L_W = 5000 with adaptive z resolution."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pynlin.collisions import get_m_values
from pynlin.fiber import SMFiber
from pynlin.methods.td.reference_curves import save_xhkm_sum_reference_curves
from pynlin.methods.td.xhkm_sums import compute_xhkm_sums
from pynlin.methods.td.xpm_kernel import compute_xpm_kernel_fft
from pynlin.pulses import NyquistPulse

OUT_MEDIA = Path("media/n-PC")
OUT_RESULTS = Path("results")
L = 400.0
B = 10e9
H_MAX = 5
R_MAX = 5
MARGIN = 10
NUM_SYMBOLS = 220
SPS = 16
LLW_OLD_MAX = 40.0
LLW_NEW_MAX = 5000.0
N_EXTRA = 6

FIBER = SMFiber(length=L, beta2=0.0)
Z_BASE = np.linspace(0.0, L, 81)


def _dataset_path_old(pulse_shape: str) -> Path:
    return OUT_RESULTS / f"xhkm_sum_ref_curve_{pulse_shape}_demo_h{H_MAX}_r{R_MAX}_marg{MARGIN}.npz"


def _dataset_path_ext(pulse_shape: str) -> Path:
    return OUT_RESULTS / f"xhkm_sum_ref_curve_{pulse_shape}_extended_h{H_MAX}_r{R_MAX}_marg{MARGIN}.npz"


def _annotation_ext() -> str:
    return f"$h,r\\in[-{H_MAX},{H_MAX}]$, m marg={MARGIN}$\\;$(ext. range)"


def _annotate(ax: plt.Axes) -> None:
    ax.text(0.98, 0.04, _annotation_ext(), transform=ax.transAxes,
            ha="right", va="bottom", fontsize=5.5, color="0.35",
            bbox=dict(facecolor="white", alpha=0.65, edgecolor="none", pad=1.5))


def _extended_points() -> np.ndarray:
    return np.geomspace(max(LLW_OLD_MAX, 1.0), LLW_NEW_MAX, N_EXTRA)


def compute_extended(case: dict) -> Path:
    path_ext = _dataset_path_ext(str(case["pulse_shape"]))
    if path_ext.exists():
        return path_ext

    h_values = np.arange(-H_MAX, H_MAX + 1)
    r_values = np.arange(-R_MAX, R_MAX + 1)
    pulse = case["pulse"]

    # Load existing dataset below LLW_OLD_MAX.
    path_old = _dataset_path_old(str(case["pulse_shape"]))
    old = np.load(path_old)
    llw_old = old["llw_grid"]
    keep = llw_old <= LLW_OLD_MAX
    llw_list = list(llw_old[keep])
    raw = {k: list(old[k][keep]) for k in
           ["raw_n1", "raw_n2", "raw_n_2pc", "raw_n_3pc_total",
            "raw_n_3pca", "raw_n_3pcb", "raw_n_3pc_other",
            "raw_n_3pc_k_eq_m", "raw_n_4pc", "raw_n_k_neq_m"]}

    # Append extended points.
    for llw in _extended_points():
        n_z = min(500, int(81 + (float(llw) - 40.0) * 0.08))
        z = np.linspace(0.0, L, n_z)
        dgd = float(llw / (L * B))
        m_values = get_m_values(FIBER, pulse, MARGIN, dgd)[::-1]
        result = compute_xpm_kernel_fft(
            pulse, z, h_values, r_values, m_values,
            dgd=dgd, gvda=float(case["gvda"]), gvdb=float(case["gvdb"]),
        )
        sums = compute_xhkm_sums(result.X, result.h_values, result.r_values, result.m_values)
        raw["raw_n1"].append(sums.n1)
        raw["raw_n2"].append(sums.n2)
        raw["raw_n_2pc"].append(sums.n_2pc)
        raw["raw_n_3pc_total"].append(sums.n_3pc_total)
        raw["raw_n_3pca"].append(sums.n_3pca)
        raw["raw_n_3pcb"].append(sums.n_3pcb)
        raw["raw_n_3pc_other"].append(sums.n_3pc_other)
        raw["raw_n_3pc_k_eq_m"].append(sums.n_3pc_k_eq_m)
        raw["raw_n_4pc"].append(sums.n_4pc)
        raw["raw_n_k_neq_m"].append(sums.n_k_neq_m)
        llw_list.append(float(llw))
        print(f"  {case['key']}: llw={float(llw):.2e}, n_z={n_z}, n_m={len(m_values)}")

    llw_grid = np.array(llw_list)
    return save_xhkm_sum_reference_curves(
        path_ext,
        llw_grid=llw_grid,
        raw_n1=np.asarray(raw["raw_n1"]),
        raw_n2=np.asarray(raw["raw_n2"]),
        raw_n_2pc=np.asarray(raw["raw_n_2pc"]),
        raw_n_3pc_total=np.asarray(raw["raw_n_3pc_total"]),
        raw_n_3pca=np.asarray(raw["raw_n_3pca"]),
        raw_n_3pcb=np.asarray(raw["raw_n_3pcb"]),
        raw_n_3pc_other=np.asarray(raw["raw_n_3pc_other"]),
        raw_n_3pc_k_eq_m=np.asarray(raw["raw_n_3pc_k_eq_m"]),
        raw_n_4pc=np.asarray(raw["raw_n_4pc"]),
        raw_n_k_neq_m=np.asarray(raw["raw_n_k_neq_m"]),
        fiber_length=L,
        baud_rate=B,
        pulse_shape=str(case["pulse_shape"]),
        mode="perfect",
        gvda=float(case["gvda"]),
        gvdb=float(case["gvdb"]),
        h_values=h_values,
        r_values=r_values,
        partial_collisions_margin=MARGIN,
        n_samples_numeric=len(llw_grid),
    )


def main() -> None:
    cases = [
        {"key": "nyquist", "pulse_shape": "nyquist",
         "pulse": NyquistPulse(baud_rate=B, num_symbols=NUM_SYMBOLS, samples_per_symbol=SPS, rolloff=0.0),
         "gvda": 0.0, "gvdb": 0.0},
    ]
    for ld_ratio in (1.0, 5.0, 10.0):
        gvd = ld_ratio / (L * B**2)
        cases.append({
            "key": f"nyquist_LD{int(ld_ratio)}",
            "pulse_shape": f"nyquist_LD{int(ld_ratio)}",
            "pulse": NyquistPulse(baud_rate=B, num_symbols=NUM_SYMBOLS, samples_per_symbol=SPS, rolloff=0.0),
            "gvda": gvd, "gvdb": gvd,
        })

    paths = {}
    for case in cases:
        print(f"Computing {case['key']}...")
        path = compute_extended(case)
        paths[case["key"]] = path
        print(f"  saved {path}")

    # Combine plot of all extended cases.
    colors = [(r"No disp.", "nyquist", "tab:orange"),
              (r"$L/L_D=1$", "nyquist_LD1", "tab:green"),
              (r"$L/L_D=5$", "nyquist_LD5", "tab:red"),
              (r"$L/L_D=10$", "nyquist_LD10", "tab:blue")]

    RELIABILITY_LLW = 80.0  # no-dispersion collision width = z spacing

    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    for label, key, color in colors:
        d = np.load(paths[key])
        llw = d["llw_grid"]
        ax.plot(llw, d["ref_n1"], color=color, marker="o", ms=2.5, lw=1.0,
                label=f"{label} N1")
        ax.plot(llw, d["ref_n2"], color=color, marker="s", ms=2.5, lw=1.0,
                ls="--", label=f"{label} N2")
        ax.plot(llw, d["ref_n_2pc"], color=color, marker="*", ms=3.5, lw=0.9,
                ls=":", label=f"{label} 2PC")
    ax.axvline(LLW_OLD_MAX, color="grey", lw=0.5, ls="--", alpha=0.5)
    ax.axvspan(RELIABILITY_LLW, 8000,
               color="grey", alpha=0.06, label=r"$\leftarrow$ z-res. limit")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$L/L_W$")
    ax.set_ylabel(r"$N\,T^2/L^2$")
    ax.set_title("Nyquist Xhkm sums: extended range")
    ax.grid(True, which="both", alpha=0.25)
    leg = ax.legend(fontsize=5.6, ncol=2)
    _annotate(ax)
    # Add z-resolution warning as a second annotation.
    ax.text(0.98, 0.92,
            "Shaded region: no-disp.\n"
            "collision width $<\\Delta z$;\n"
            "curves there may be\n"
            "numerically under-resolved\n"
            "and artificially split.",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=5.5, color="0.35",
            bbox=dict(facecolor="white", alpha=0.65, edgecolor="none", pad=1.5))
    fig.tight_layout()
    stem = f"xhkm_extended_range_h{H_MAX}_r{R_MAX}_marg{MARGIN}"
    fig.savefig(OUT_MEDIA / f"{stem}.pdf", dpi=300)
    fig.savefig(OUT_MEDIA / f"{stem}.png", dpi=300)
    plt.close(fig)
    print(f"plotted {stem}")

    # N2/N1 ratio inset zoom.
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for label, key, color in colors:
        d = np.load(paths[key])
        llw = d["llw_grid"]
        ax.plot(llw, d["n2_over_n1"], color=color, marker="o", ms=2.5, lw=1.0, label=label)
    ax.axvline(LLW_OLD_MAX, color="grey", lw=0.5, ls="--", alpha=0.5)
    ax.axvspan(RELIABILITY_LLW, 8000,
               color="grey", alpha=0.06)
    ax.axhline(1.0, color="k", lw=0.4, ls="-", alpha=0.3)
    ax.set_xscale("log")
    ax.set_xlabel(r"$L/L_W$")
    ax.set_ylabel(r"$N_2/N_1$")
    ax.set_title("Nyquist Xhkm: N2/N1 ratio, extended range")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=7)
    _annotate(ax)
    ax.text(0.98, 0.92,
            "Shaded: z-res. limit",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=5.5, color="0.35",
            bbox=dict(facecolor="white", alpha=0.65, edgecolor="none", pad=1.5))
    fig.tight_layout()
    stem_rat = f"xhkm_extended_ratio_h{H_MAX}_r{R_MAX}_marg{MARGIN}"
    fig.savefig(OUT_MEDIA / f"{stem_rat}.pdf", dpi=300)
    fig.savefig(OUT_MEDIA / f"{stem_rat}.png", dpi=300)
    plt.close(fig)
    print(f"plotted {stem_rat}")


if __name__ == "__main__":
    main()
