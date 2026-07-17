from __future__ import annotations

import argparse
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
from pynlin.pulses import GaussianPulse, NyquistPulse


OUT_MEDIA = Path("media/n-PC/window-convergence")
OUT_RESULTS = Path("results/window-convergence")
FIBER_LENGTH = 400.0
BAUD_RATE = 10e9
NUM_SYMBOLS = 220
SAMPLES_PER_SYMBOL = 16
N_Z_POINTS = 81

COMPONENTS = (
    ("ref_n1", r"$N_1$"),
    ("ref_n2", r"$N_2$"),
    ("ref_n_2pc", "2PC"),
    ("ref_n_3pc_total", "3PC"),
    ("ref_n_3pca", "3PCa"),
    ("ref_n_3pcb", "3PCb"),
    ("ref_n_4pc", "4PC"),
)


def _window_label(window: tuple[int, int, int]) -> str:
    h_max, r_max, margin = window
    return f"h,r=({h_max},{r_max}), marg={margin}"


def _window_token(window: tuple[int, int, int]) -> str:
    h_max, r_max, margin = window
    return f"h{h_max}_r{r_max}_marg{margin}"


def _pulse(pulse_shape: str):
    if pulse_shape == "gaussian":
        return GaussianPulse(
            baud_rate=BAUD_RATE,
            num_symbols=NUM_SYMBOLS,
            samples_per_symbol=SAMPLES_PER_SYMBOL,
        )
    if pulse_shape == "nyquist":
        return NyquistPulse(
            baud_rate=BAUD_RATE,
            num_symbols=NUM_SYMBOLS,
            samples_per_symbol=SAMPLES_PER_SYMBOL,
            rolloff=0.0,
        )
    raise ValueError(f"Unsupported pulse_shape={pulse_shape!r}")


def _dataset_path(
    pulse_shape: str,
    window: tuple[int, int, int],
    llw_min: float,
    llw_max: float,
    n_llw: int,
) -> Path:
    return OUT_RESULTS / (
        f"xhkm_window_convergence_{pulse_shape}_{_window_token(window)}"
        f"_llw{llw_min:g}-{llw_max:g}_n{n_llw}.npz"
    )


def compute_window(
    *,
    pulse_shape: str,
    window: tuple[int, int, int],
    llw_grid: np.ndarray,
    recompute: bool,
    max_z_points: int,
) -> Path:
    path = _dataset_path(pulse_shape, window, float(llw_grid[0]), float(llw_grid[-1]), llw_grid.size)
    if path.exists() and not recompute:
        return path

    OUT_RESULTS.mkdir(parents=True, exist_ok=True)
    h_max, r_max, margin = window
    h_values = np.arange(-h_max, h_max + 1)
    r_values = np.arange(-r_max, r_max + 1)
    pulse = _pulse(pulse_shape)
    fiber = SMFiber(length=FIBER_LENGTH, beta2=0.0)
    z = np.linspace(0.0, FIBER_LENGTH, N_Z_POINTS)

    raw = {
        name: []
        for name in (
            "n1",
            "n2",
            "n_2pc",
            "n_3pc_total",
            "n_3pca",
            "n_3pcb",
            "n_3pc_other",
            "n_3pc_k_eq_m",
            "n_4pc",
            "n_k_neq_m",
        )
    }
    n_m_values = []
    n_z_values = []
    for llw in llw_grid:
        dgd = float(llw / (FIBER_LENGTH * BAUD_RATE))
        m_values = get_m_values(fiber, pulse, margin, dgd)[::-1]
        result = compute_xpm_kernel_fft(
            pulse,
            z,
            h_values,
            r_values,
            m_values,
            dgd=dgd,
            gvda=0.0,
            gvdb=0.0,
            auto_refine=True,
            min_pts_per_collision=3.0,
            max_z_points=max_z_points,
            discretization_action="warn",
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
        n_m_values.append(len(m_values))
        n_z_values.append(int(result.metadata["n_z"]))
        print(
            f"{pulse_shape} {_window_label(window)}: "
            f"L/LW={llw:.3g}, n_m={len(m_values)}, n_z={result.metadata['n_z']}"
        )

    saved = save_xhkm_sum_reference_curves(
        path,
        llw_grid=llw_grid,
        raw_n1=np.asarray(raw["n1"]),
        raw_n2=np.asarray(raw["n2"]),
        raw_n_2pc=np.asarray(raw["n_2pc"]),
        raw_n_3pc_total=np.asarray(raw["n_3pc_total"]),
        raw_n_3pca=np.asarray(raw["n_3pca"]),
        raw_n_3pcb=np.asarray(raw["n_3pcb"]),
        raw_n_3pc_other=np.asarray(raw["n_3pc_other"]),
        raw_n_3pc_k_eq_m=np.asarray(raw["n_3pc_k_eq_m"]),
        raw_n_4pc=np.asarray(raw["n_4pc"]),
        raw_n_k_neq_m=np.asarray(raw["n_k_neq_m"]),
        fiber_length=FIBER_LENGTH,
        baud_rate=BAUD_RATE,
        pulse_shape=pulse_shape,
        mode="perfect",
        gvda=0.0,
        gvdb=0.0,
        h_values=h_values,
        r_values=r_values,
        partial_collisions_margin=margin,
        n_samples_numeric=llw_grid.size,
    )
    with np.load(saved) as data:
        payload = {key: data[key] for key in data.files}
    np.savez(saved, **payload, n_m_values=np.asarray(n_m_values), n_z_values=np.asarray(n_z_values))
    return saved


def _load_curves(paths: dict[tuple[int, int, int], Path]) -> dict[tuple[int, int, int], dict[str, np.ndarray]]:
    curves = {}
    for window, path in paths.items():
        with np.load(path) as data:
            curves[window] = {key: np.asarray(data[key]) for key in data.files}
    return curves


def plot_absolute(curves: dict[tuple[int, int, int], dict[str, np.ndarray]], out_dir: Path, pulse_shape: str) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2), sharex=True)
    plot_components = (
        ("ref_n_2pc", "2PC"),
        ("ref_n_3pc_total", "3PC"),
        ("ref_n_4pc", "4PC"),
        ("ref_n1", r"$N_1$"),
    )
    for ax, (component, title) in zip(axes.ravel(), plot_components, strict=True):
        for window, data in curves.items():
            ax.plot(data["llw_grid"], data[component], marker="o", ms=2.5, lw=1.0, label=_window_label(window))
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.25)
    for ax in axes[-1, :]:
        ax.set_xlabel(r"$L/L_W$")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$N\,T^2/L^2$")
    axes[0, 0].legend(fontsize=6)
    fig.suptitle(f"{pulse_shape.title()} xhkm window convergence")
    fig.tight_layout()
    path = out_dir / f"xhkm_window_convergence_absolute_{pulse_shape}.pdf"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def plot_relative_error(
    curves: dict[tuple[int, int, int], dict[str, np.ndarray]],
    out_dir: Path,
    pulse_shape: str,
) -> Path:
    reference_window = list(curves)[-1]
    reference = curves[reference_window]
    windows = [window for window in curves if window != reference_window]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2), sharex=True)
    plot_components = (
        ("ref_n_2pc", "2PC"),
        ("ref_n_3pc_total", "3PC"),
        ("ref_n_4pc", "4PC"),
        ("ref_n1", r"$N_1$"),
    )
    eps = 1e-300
    for ax, (component, title) in zip(axes.ravel(), plot_components, strict=True):
        ref = np.maximum(np.abs(reference[component]), eps)
        for window in windows:
            data = curves[window]
            relerr = np.abs(data[component] - reference[component]) / ref
            ax.plot(data["llw_grid"], relerr, marker="o", ms=2.5, lw=1.0, label=_window_label(window))
        ax.axhline(1e-2, color="0.35", lw=0.7, ls=":", label="1%" if component == "ref_n_2pc" else None)
        ax.axhline(1e-1, color="0.35", lw=0.7, ls="--", label="10%" if component == "ref_n_2pc" else None)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.25)
    for ax in axes[-1, :]:
        ax.set_xlabel(r"$L/L_W$")
    for ax in axes[:, 0]:
        ax.set_ylabel("relative error")
    axes[0, 0].legend(fontsize=6)
    fig.suptitle(f"Relative to largest window: {_window_label(reference_window)}")
    fig.tight_layout()
    path = out_dir / f"xhkm_window_convergence_relative_{pulse_shape}.pdf"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def _parse_windows(values: list[str]) -> list[tuple[int, int, int]]:
    windows = []
    for value in values:
        parts = value.split(",")
        if len(parts) != 3:
            raise argparse.ArgumentTypeError(f"Window {value!r} must be h,r,margin")
        windows.append(tuple(int(part) for part in parts))
    return windows


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check xhkm h/r/m window convergence for 2PC, 3PC, and 4PC.")
    parser.add_argument("--pulse", choices=("nyquist", "gaussian"), default="nyquist")
    parser.add_argument("--llw-min", type=float, default=5e-2)
    parser.add_argument("--llw-max", type=float, default=3e2)
    parser.add_argument("--n-llw", type=int, default=14)
    parser.add_argument(
        "--window",
        action="append",
        default=None,
        help="Window as h,r,margin. Repeatable. Default: 5,5,10; 7,7,15; 10,10,20.",
    )
    parser.add_argument("--max-z-points", type=int, default=500)
    parser.add_argument("--recompute", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    windows = _parse_windows(args.window) if args.window else [(5, 5, 10), (7, 7, 15), (10, 10, 20)]
    llw_grid = np.geomspace(args.llw_min, args.llw_max, args.n_llw)
    paths = {}
    for window in windows:
        paths[window] = compute_window(
            pulse_shape=args.pulse,
            window=window,
            llw_grid=llw_grid,
            recompute=args.recompute,
            max_z_points=args.max_z_points,
        )
        print(f"saved {paths[window]}")

    OUT_MEDIA.mkdir(parents=True, exist_ok=True)
    curves = _load_curves(paths)
    for path in (
        plot_absolute(curves, OUT_MEDIA, args.pulse),
        plot_relative_error(curves, OUT_MEDIA, args.pulse),
    ):
        print(f"saved {path}")


if __name__ == "__main__":
    main()
