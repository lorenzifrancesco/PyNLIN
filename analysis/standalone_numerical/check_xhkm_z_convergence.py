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


OUT_MEDIA = Path("media/n-PC/z-convergence")
OUT_RESULTS = Path("results/z-convergence")
FIBER_LENGTH = 400.0
BAUD_RATE = 10e9
NUM_SYMBOLS = 220
SAMPLES_PER_SYMBOL = 16
N_Z_SEED = 81

PLOT_COMPONENTS = (
    ("ref_n_2pc", "2PC"),
    ("ref_n_3pc_total", "3PC"),
    ("ref_n_4pc", "4PC"),
    ("ref_n1", r"$N_1$"),
)


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


def _window_label(window: tuple[int, int, int]) -> str:
    h_max, r_max, margin = window
    return f"h,r=({h_max},{r_max}), marg={margin}"


def _window_token(window: tuple[int, int, int]) -> str:
    h_max, r_max, margin = window
    return f"h{h_max}_r{r_max}_marg{margin}"


def _setting_label(setting: tuple[float, int]) -> str:
    min_pts, max_z = setting
    return f"pts/coll={min_pts:g}, max_z={max_z}"


def _setting_token(setting: tuple[float, int]) -> str:
    min_pts, max_z = setting
    pts = f"{min_pts:g}".replace(".", "p")
    return f"pts{pts}_maxz{max_z}"


def _dataset_path(
    pulse_shape: str,
    window: tuple[int, int, int],
    setting: tuple[float, int],
    llw_grid: np.ndarray,
) -> Path:
    return OUT_RESULTS / (
        f"xhkm_z_convergence_{pulse_shape}_{_window_token(window)}_{_setting_token(setting)}"
        f"_llw{float(llw_grid[0]):g}-{float(llw_grid[-1]):g}_n{llw_grid.size}.npz"
    )


def compute_setting(
    *,
    pulse_shape: str,
    window: tuple[int, int, int],
    setting: tuple[float, int],
    llw_grid: np.ndarray,
    recompute: bool,
) -> Path:
    path = _dataset_path(pulse_shape, window, setting, llw_grid)
    if path.exists() and not recompute:
        return path

    OUT_RESULTS.mkdir(parents=True, exist_ok=True)
    min_pts_per_collision, max_z_points = setting
    h_max, r_max, margin = window
    h_values = np.arange(-h_max, h_max + 1)
    r_values = np.arange(-r_max, r_max + 1)
    pulse = _pulse(pulse_shape)
    fiber = SMFiber(length=FIBER_LENGTH, beta2=0.0)
    z_seed = np.linspace(0.0, FIBER_LENGTH, N_Z_SEED)

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
    pts_per_collision = []
    walkoff_fraction = []

    for llw in llw_grid:
        dgd = float(llw / (FIBER_LENGTH * BAUD_RATE))
        m_values = get_m_values(fiber, pulse, margin, dgd)[::-1]
        result = compute_xpm_kernel_fft(
            pulse,
            z_seed,
            h_values,
            r_values,
            m_values,
            dgd=dgd,
            gvda=0.0,
            gvdb=0.0,
            auto_refine=True,
            min_pts_per_collision=float(min_pts_per_collision),
            max_z_points=int(max_z_points),
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
        pts_per_collision.append(float(result.metadata["pts_per_collision"]))
        walkoff_fraction.append(float(result.metadata["walkoff_fraction"]))
        print(
            f"{pulse_shape} {_window_label(window)} {_setting_label(setting)}: "
            f"L/LW={llw:.3g}, n_m={len(m_values)}, n_z={result.metadata['n_z']}, "
            f"pts/coll={result.metadata['pts_per_collision']:.2f}"
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
    np.savez(
        saved,
        **payload,
        min_pts_per_collision=np.array(float(min_pts_per_collision)),
        max_z_points=np.array(int(max_z_points)),
        n_m_values=np.asarray(n_m_values),
        n_z_values=np.asarray(n_z_values),
        pts_per_collision=np.asarray(pts_per_collision),
        walkoff_fraction=np.asarray(walkoff_fraction),
    )
    return saved


def _load_curves(paths: dict[tuple[float, int], Path]) -> dict[tuple[float, int], dict[str, np.ndarray]]:
    curves = {}
    for setting, path in paths.items():
        with np.load(path) as data:
            curves[setting] = {key: np.asarray(data[key]) for key in data.files}
    return curves


def plot_absolute(curves: dict[tuple[float, int], dict[str, np.ndarray]], out_dir: Path, pulse_shape: str) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2), sharex=True)
    for ax, (component, title) in zip(axes.ravel(), PLOT_COMPONENTS, strict=True):
        for setting, data in curves.items():
            ax.plot(data["llw_grid"], data[component], marker="o", ms=2.5, lw=1.0, label=_setting_label(setting))
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.25)
    for ax in axes[-1, :]:
        ax.set_xlabel(r"$L/L_W$")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$N\,T^2/L^2$")
    axes[0, 0].legend(fontsize=6)
    fig.suptitle(f"{pulse_shape.title()} xhkm z-resolution convergence")
    fig.tight_layout()
    path = out_dir / f"xhkm_z_convergence_absolute_{pulse_shape}.pdf"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def plot_relative_error(curves: dict[tuple[float, int], dict[str, np.ndarray]], out_dir: Path, pulse_shape: str) -> Path:
    reference_setting = list(curves)[-1]
    reference = curves[reference_setting]
    settings = [setting for setting in curves if setting != reference_setting]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2), sharex=True)
    eps = 1e-300
    for ax, (component, title) in zip(axes.ravel(), PLOT_COMPONENTS, strict=True):
        ref = np.maximum(np.abs(reference[component]), eps)
        for setting in settings:
            data = curves[setting]
            relerr = np.abs(data[component] - reference[component]) / ref
            ax.plot(data["llw_grid"], relerr, marker="o", ms=2.5, lw=1.0, label=_setting_label(setting))
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
    fig.suptitle(f"Relative to strictest z setting: {_setting_label(reference_setting)}")
    fig.tight_layout()
    path = out_dir / f"xhkm_z_convergence_relative_{pulse_shape}.pdf"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def plot_actual_nz(curves: dict[tuple[float, int], dict[str, np.ndarray]], out_dir: Path, pulse_shape: str) -> Path:
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    for setting, data in curves.items():
        ax.plot(data["llw_grid"], data["n_z_values"], marker="o", ms=2.5, lw=1.0, label=_setting_label(setting))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$L/L_W$")
    ax.set_ylabel("actual z points")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=6)
    fig.tight_layout()
    path = out_dir / f"xhkm_z_convergence_actual_nz_{pulse_shape}.pdf"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def _parse_window(value: str) -> tuple[int, int, int]:
    parts = value.split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"Window {value!r} must be h,r,margin")
    return tuple(int(part) for part in parts)


def _parse_setting(value: str) -> tuple[float, int]:
    parts = value.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"z setting {value!r} must be min_pts,max_z")
    return float(parts[0]), int(parts[1])


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check xhkm z-grid convergence for 2PC, 3PC, and 4PC.")
    parser.add_argument("--pulse", choices=("nyquist", "gaussian"), default="nyquist")
    parser.add_argument("--llw-min", type=float, default=5e-2)
    parser.add_argument("--llw-max", type=float, default=3e2)
    parser.add_argument("--n-llw", type=int, default=14)
    parser.add_argument("--window", type=_parse_window, default=(10, 10, 20), help="Fixed index window h,r,margin.")
    parser.add_argument(
        "--z-setting",
        action="append",
        type=_parse_setting,
        default=None,
        help="Auto-refine setting min_pts,max_z. Repeatable. Default: 3,500; 6,1000; 10,2000; 15,4000.",
    )
    parser.add_argument("--recompute", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    settings = args.z_setting if args.z_setting else [(3.0, 500), (6.0, 1000), (10.0, 2000), (15.0, 4000)]
    llw_grid = np.geomspace(args.llw_min, args.llw_max, args.n_llw)
    paths = {}
    for setting in settings:
        paths[setting] = compute_setting(
            pulse_shape=args.pulse,
            window=args.window,
            setting=setting,
            llw_grid=llw_grid,
            recompute=args.recompute,
        )
        print(f"saved {paths[setting]}")

    OUT_MEDIA.mkdir(parents=True, exist_ok=True)
    curves = _load_curves(paths)
    for path in (
        plot_absolute(curves, OUT_MEDIA, args.pulse),
        plot_relative_error(curves, OUT_MEDIA, args.pulse),
        plot_actual_nz(curves, OUT_MEDIA, args.pulse),
    ):
        print(f"saved {path}")


if __name__ == "__main__":
    main()
