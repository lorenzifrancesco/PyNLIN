"""Generate h=±1, r=±1 Xhkm reference curves and plots WITH WARNING.

These use a small (h,r) window that is INSUFFICIENT for convergence.
The plots are kept to document why the larger h=±5, r=±5, margin=10
window is needed.
"""

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
from pynlin.pulses import GaussianPulse, NyquistPulse, RaisedCosinePulse, RootRaisedCosinePulse

OUT_MEDIA = Path("media/n-PC")
OUT_RESULTS = Path("results")
FIBER_LENGTH = 400.0
BAUD_RATE = 10e9
N_Z_POINTS = 81
N_LLW_POINTS = 22
LLW_MIN = 5e-2
LLW_MAX = 4e1
H_MAX = 1
R_MAX = 1
PARTIAL_COLLISIONS_MARGIN = 0
NUM_SYMBOLS = 220
SAMPLES_PER_SYMBOL = 16
WARN_TEXT = "WARNING: h,r=±1 window insufficient for convergence"


def _dataset_path(pulse_shape: str) -> Path:
    return OUT_RESULTS / (
        f"xhkm_sum_ref_curve_{pulse_shape}_demo_h{H_MAX}_r{R_MAX}.npz"
    )


def _annotation() -> str:
    return (
        rf"$h,r\in[-{H_MAX},{H_MAX}]$" "\n"
        rf"m margin={PARTIAL_COLLISIONS_MARGIN}"
    )


def _annotate(ax: plt.Axes) -> None:
    # Place warning prominently at top-left
    ax.text(
        0.02, 0.98, WARN_TEXT,
        transform=ax.transAxes, ha="left", va="top",
        fontsize=6.0, color="red", weight="bold",
        bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "red", "pad": 2.0, "linewidth": 0.8},
    )
    ax.text(
        0.98, 0.04,
        _annotation(),
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.5,
        color="0.35",
        bbox={"facecolor": "white", "alpha": 0.65, "edgecolor": "none", "pad": 1.5},
    )


def _cases() -> list[dict[str, object]]:
    cases: list[dict[str, object]] = [
        {"key": "gaussian", "pulse_shape": "gaussian",
         "pulse": GaussianPulse(baud_rate=BAUD_RATE, num_symbols=NUM_SYMBOLS, samples_per_symbol=SAMPLES_PER_SYMBOL),
         "gvda": 0.0, "gvdb": 0.0},
        {"key": "nyquist", "pulse_shape": "nyquist",
         "pulse": NyquistPulse(baud_rate=BAUD_RATE, num_symbols=NUM_SYMBOLS, samples_per_symbol=SAMPLES_PER_SYMBOL, rolloff=0.0),
         "gvda": 0.0, "gvdb": 0.0},
        {"key": "rc02", "pulse_shape": "raised_cosine_rolloff0p2",
         "pulse": RaisedCosinePulse(baud_rate=BAUD_RATE, num_symbols=NUM_SYMBOLS, samples_per_symbol=SAMPLES_PER_SYMBOL, rolloff=0.2),
         "gvda": 0.0, "gvdb": 0.0},
        {"key": "rc05", "pulse_shape": "raised_cosine_rolloff0p5",
         "pulse": RaisedCosinePulse(baud_rate=BAUD_RATE, num_symbols=NUM_SYMBOLS, samples_per_symbol=SAMPLES_PER_SYMBOL, rolloff=0.5),
         "gvda": 0.0, "gvdb": 0.0},
        {"key": "rrc02", "pulse_shape": "root_raised_cosine_rolloff0p2",
         "pulse": RootRaisedCosinePulse(baud_rate=BAUD_RATE, num_symbols=NUM_SYMBOLS, samples_per_symbol=SAMPLES_PER_SYMBOL, rolloff=0.2),
         "gvda": 0.0, "gvdb": 0.0},
        {"key": "rrc05", "pulse_shape": "root_raised_cosine_rolloff0p5",
         "pulse": RootRaisedCosinePulse(baud_rate=BAUD_RATE, num_symbols=NUM_SYMBOLS, samples_per_symbol=SAMPLES_PER_SYMBOL, rolloff=0.5),
         "gvda": 0.0, "gvdb": 0.0},
    ]
    for ld_ratio in (1.0, 5.0, 10.0):
        gvd = ld_ratio / (FIBER_LENGTH * BAUD_RATE**2)
        cases.append({"key": f"nyquist_LD{int(ld_ratio)}", "pulse_shape": f"nyquist_LD{int(ld_ratio)}",
                      "pulse": NyquistPulse(baud_rate=BAUD_RATE, num_symbols=NUM_SYMBOLS, samples_per_symbol=SAMPLES_PER_SYMBOL, rolloff=0.0),
                      "gvda": gvd, "gvdb": gvd})
    return cases


def compute_case(case: dict[str, object]) -> Path:
    OUT_RESULTS.mkdir(parents=True, exist_ok=True)
    path = _dataset_path(str(case["pulse_shape"]))
    fiber = SMFiber(length=FIBER_LENGTH, beta2=0.0)
    z = np.linspace(0.0, FIBER_LENGTH, N_Z_POINTS)
    llw_grid = np.geomspace(LLW_MIN, LLW_MAX, N_LLW_POINTS)
    h_values = np.arange(-H_MAX, H_MAX + 1)
    r_values = np.arange(-R_MAX, R_MAX + 1)
    pulse = case["pulse"]

    raw = {name: [] for name in (
        "n1", "n2", "n_2pc", "n_3pc_total", "n_3pca", "n_3pcb",
        "n_3pc_other", "n_3pc_k_eq_m", "n_4pc", "n_k_neq_m",
    )}
    for llw in llw_grid:
        dgd = float(llw / (FIBER_LENGTH * BAUD_RATE))
        m_values = get_m_values(fiber, pulse, PARTIAL_COLLISIONS_MARGIN, dgd)[::-1]
        result = compute_xpm_kernel_fft(
            pulse, z, h_values, r_values, m_values,
            dgd=dgd, gvda=float(case["gvda"]), gvdb=float(case["gvdb"]),
            auto_refine=True,
            min_pts_per_collision=3.0,
            discretization_action="silent",
        )
        sums = compute_xhkm_sums(result.X, result.h_values, result.r_values, result.m_values)
        raw["n1"].append(sums.n1)
        raw["n2"].append(sums.n2)
        raw["n_2pc"].append(sums.n_2pc)
        raw["n_3pc_total"].append(sums.n_3pc_total)
        raw["n_3pca"].append(sums.n_3pca)
        raw["n_3pcb"].append(sums.n_3pcb)
        raw["n_3pc_other"].append(sums.n_3pc_other)
        raw["n_3pc_k_eq_m"].append(sums.n_3pc_k_eq_m)
        raw["n_4pc"].append(sums.n_4pc)
        raw["n_k_neq_m"].append(sums.n_k_neq_m)

    return save_xhkm_sum_reference_curves(
        path, llw_grid=llw_grid,
        raw_n1=np.asarray(raw["n1"]), raw_n2=np.asarray(raw["n2"]),
        raw_n_2pc=np.asarray(raw["n_2pc"]), raw_n_3pc_total=np.asarray(raw["n_3pc_total"]),
        raw_n_3pca=np.asarray(raw["n_3pca"]), raw_n_3pcb=np.asarray(raw["n_3pcb"]),
        raw_n_3pc_other=np.asarray(raw["n_3pc_other"]), raw_n_3pc_k_eq_m=np.asarray(raw["n_3pc_k_eq_m"]),
        raw_n_4pc=np.asarray(raw["n_4pc"]), raw_n_k_neq_m=np.asarray(raw["n_k_neq_m"]),
        fiber_length=FIBER_LENGTH, baud_rate=BAUD_RATE, pulse_shape=str(case["pulse_shape"]),
        mode="perfect", gvda=float(case["gvda"]), gvdb=float(case["gvdb"]),
        h_values=h_values, r_values=r_values, partial_collisions_margin=PARTIAL_COLLISIONS_MARGIN,
        n_samples_numeric=N_LLW_POINTS,
    )


def _finish(fig: plt.Figure, stem: str) -> None:
    OUT_MEDIA.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT_MEDIA / f"{stem}.pdf", dpi=300)
    plt.close(fig)


def plot_single(path: Path, title: str, stem: str) -> None:
    data = np.load(path)
    llw = data["llw_grid"]
    fig, ax = plt.subplots(figsize=(4.4, 3.15))
    ax.plot(llw, data["ref_n1"], marker="o", ms=3, lw=1.0, label=r"$N_1$")
    ax.plot(llw, data["ref_n2"], marker="s", ms=3, lw=1.0, ls="--", label=r"$N_2$")
    ax.plot(llw, data["ref_n_2pc"], marker="*", ms=5, lw=0.9, ls=":", label="2PC")
    ax.plot(llw, data["ref_n_3pc_total"], marker="^", ms=3, lw=0.9, ls="-.", label="3PC")
    ax.plot(llw, data["ref_n_4pc"], marker="D", ms=3, lw=0.9, ls=(0, (3, 1, 1, 1)), label="4PC")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$L/L_W$"); ax.set_ylabel(r"$N\,T^2/L^2$")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)
    _annotate(ax)
    _finish(fig, stem)


def plot_decomposition(path: Path, title: str, stem: str) -> None:
    data = np.load(path)
    llw = data["llw_grid"]
    fig, ax = plt.subplots(figsize=(4.6, 3.25))
    ax.plot(llw, data["ref_n_2pc"], marker="*", ms=5, lw=0.9, label="2PC")
    ax.plot(llw, data["ref_n_3pca"], marker="o", ms=3, lw=1.0, label="3PCa")
    ax.plot(llw, data["ref_n_3pcb"], marker="s", ms=3, lw=1.0, label="3PCb")
    ax.plot(llw, data["ref_n_4pc"], marker="D", ms=3, lw=1.0, label="4PC")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$L/L_W$"); ax.set_ylabel(r"$N\,T^2/L^2$")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=7)
    _annotate(ax)
    _finish(fig, stem)


def plot_compare(items: list[tuple[str, Path, str]], stem: str, *, include_2pc: bool = True, legend_font: float = 6.0) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for label, path, color in items:
        data = np.load(path)
        llw = data["llw_grid"]
        ax.plot(llw, data["ref_n1"], color=color, marker="o", ms=3, lw=1.0, label=f"{label} N1")
        ax.plot(llw, data["ref_n2"], color=color, marker="s", ms=3, lw=1.0, ls="--", label=f"{label} N2")
        if include_2pc:
            ax.plot(llw, data["ref_n_2pc"], color=color, marker="*", ms=5, lw=0.9, ls=":", label=f"{label} 2PC")
            ax.plot(llw, data["ref_n_3pc_total"], color=color, marker="^", ms=3, lw=0.9, ls="-.", label=f"{label} 3PC")
            ax.plot(llw, data["ref_n_4pc"], color=color, marker="D", ms=3, lw=0.9, ls=(0, (3, 1, 1, 1)), label=f"{label} 4PC")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$L/L_W$"); ax.set_ylabel(r"$N\,T^2/L^2$")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=legend_font, ncol=3)
    _annotate(ax)
    _finish(fig, stem)


def plot_ratio(paths: dict[str, Path]) -> None:
    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    for label, key, color in (("Gaussian", "gaussian", "tab:purple"), ("Nyquist", "nyquist", "tab:blue")):
        data = np.load(paths[key])
        ax.plot(data["llw_grid"], data["n2_over_n1"], marker="o", ms=3, lw=1.0, color=color, label=label)
    ax.set_xscale("log")
    ax.set_xlabel(r"$L/L_W$"); ax.set_ylabel(r"$N_2/N_1$")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)
    _annotate(ax)
    _finish(fig, "xhkm_n2_over_n1_gaussian_nyquist")


def main() -> None:
    print("=== Mini-window (h,r=±1) Xhkm reference curves ===")
    cases = _cases()
    paths: dict[str, Path] = {}
    for case in cases:
        path = compute_case(case)
        paths[str(case["key"])] = path
        print(f"  saved {path}")

    plot_single(paths["gaussian"], "Gaussian, flat profile [MINI WINDOW]", "xhkm_n1_n2_gaussian")
    plot_single(paths["nyquist"], "Nyquist, flat profile [MINI WINDOW]", "xhkm_n1_n2_nyquist")
    plot_decomposition(paths["gaussian"], "Gaussian decomposition [MINI WINDOW]", "xhkm_collision_decomposition_gaussian")
    plot_decomposition(paths["nyquist"], "Nyquist decomposition [MINI WINDOW]", "xhkm_collision_decomposition_nyquist")
    plot_compare(
        [("Gaussian", paths["gaussian"], "tab:purple"), ("Nyquist", paths["nyquist"], "tab:blue")],
        "xhkm_n1_n2_2pc_gaussian_nyquist_combined", include_2pc=True)
    plot_compare(
        [("Gaussian", paths["gaussian"], "tab:purple"), ("Nyquist", paths["nyquist"], "tab:blue")],
        "xhkm_n1_n2_gaussian_nyquist_combined", include_2pc=False, legend_font=7)
    plot_ratio(paths)
    plot_compare([("Nyquist", paths["nyquist"], "tab:blue"), (r"RC $\rho=0.2$", paths["rc02"], "tab:orange")],
                 "xhkm_n1_n2_2pc_nyquist_vs_raised_cosine_rolloff0p2")
    plot_compare([("Nyquist", paths["nyquist"], "tab:blue"), (r"RC $\rho=0.5$", paths["rc05"], "tab:green")],
                 "xhkm_n1_n2_2pc_nyquist_vs_raised_cosine_rolloff0p5")
    plot_compare([
        ("Nyquist", paths["nyquist"], "tab:blue"),
        (r"RRC $\rho=0.2$", paths["rrc02"], "tab:orange"),
        (r"RRC $\rho=0.5$", paths["rrc05"], "tab:green"),
    ], "xhkm_n1_n2_2pc_nyquist_vs_rrc_rolloff", legend_font=5.5)

    for ld_ratio in (1, 5, 10):
        plot_single(paths[f"nyquist_LD{ld_ratio}"],
                    rf"Nyquist, $L/L_D={ld_ratio}$ [MINI WINDOW]",
                    f"xhkm_n1_n2_2pc_nyquist_LD{ld_ratio}")
    plot_compare([("No disp.", paths["nyquist"], "tab:orange"), (r"$L/L_D=10$", paths["nyquist_LD10"], "tab:blue")],
                 "xhkm_n1_n2_2pc_nyquist_no_disp_vs_LD10")
    plot_compare([
        ("No disp.", paths["nyquist"], "tab:orange"),
        (r"$L/L_D=1$", paths["nyquist_LD1"], "tab:green"),
        (r"$L/L_D=10$", paths["nyquist_LD10"], "tab:blue"),
    ], "xhkm_n1_n2_2pc_nyquist_no_disp_vs_LD1_vs_LD10", legend_font=5.7)
    plot_compare([
        ("No disp.", paths["nyquist"], "tab:orange"),
        (r"$L/L_D=1$", paths["nyquist_LD1"], "tab:green"),
        (r"$L/L_D=5$", paths["nyquist_LD5"], "tab:red"),
        (r"$L/L_D=10$", paths["nyquist_LD10"], "tab:blue"),
    ], "xhkm_n1_n2_2pc_nyquist_no_disp_vs_LD1_vs_LD5_vs_LD10", legend_font=5.4)

    print("  All mini-window plots saved in media/n-PC/")


if __name__ == "__main__":
    main()
