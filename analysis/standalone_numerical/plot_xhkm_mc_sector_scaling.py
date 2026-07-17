from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pynlin.methods.td.xhkm_mc import estimate_xhkm_sums_mc


OUT_MEDIA = Path("media/n-PC/mc-sector-scaling")
OUT_RESULTS = Path("results/mc-sector-scaling")

COMPONENTS = (
    ("n1", r"$N_1$"),
    ("n2", r"$N_2$"),
    ("n_2pc", "2PC"),
    ("n_3pc_total", "3PC"),
    ("n_3pca", "3PCa"),
    ("n_3pcb", "3PCb"),
    ("n_4pc", "4PC"),
)

PLOT_COMPONENTS = (
    ("n1", r"$N_1$"),
    ("n2", r"$N_2$"),
    ("n_2pc", "2PC"),
    ("n_3pc_total", "3PC"),
    ("n_4pc", "4PC"),
)


def _beta2_for_llw(llw: float, *, length: float, channel_spacing_over_baud: float) -> float:
    # In the Dar MC normalization used here: L/L_W = 2*pi*q*beta2*L.
    return float(llw) / (2.0 * np.pi * float(channel_spacing_over_baud) * float(length))


def _params_for_llw(
    llw: float,
    *,
    length: float,
    spacing: float,
    lld: float | None,
) -> tuple[float, float]:
    if lld is None:
        return _beta2_for_llw(llw, length=length, channel_spacing_over_baud=spacing), float(spacing)
    beta2 = float(lld) / float(length)
    q = float(llw) / (2.0 * np.pi * float(lld))
    return beta2, q


def _safe_relative_stderr(values: np.ndarray, stderr: np.ndarray) -> np.ndarray:
    return np.divide(stderr, np.abs(values), out=np.full_like(stderr, np.nan, dtype=float), where=np.abs(values) > 0.0)


def compute_curves(args: argparse.Namespace, *, lld: float | None = None) -> dict[str, np.ndarray]:
    llw_grid = np.geomspace(args.llw_min, args.llw_max, args.n_llw)
    seeds = np.arange(args.seed, args.seed + args.n_seeds, dtype=int)
    sample_rows = []
    mean_rows = []
    stderr_rows = []
    spacing_rows = []

    for llw in llw_grid:
        beta2, spacing = _params_for_llw(llw, length=args.length, spacing=args.spacing, lld=lld)
        spacing_rows.append(spacing)
        per_seed = []
        per_seed_stderr = []
        for seed in seeds:
            mc = estimate_xhkm_sums_mc(
                beta2=beta2,
                alpha=args.alpha,
                length=args.length,
                channel_spacing_over_baud=spacing,
                nspan=args.nspan,
                phase_delay=args.phase_delay,
                n_samples=args.n_samples,
                seed=int(seed),
            )
            values = [float(getattr(mc, name)) for name, _ in COMPONENTS]
            stderrs = [float(getattr(mc, f"{name}_stderr")) for name, _ in COMPONENTS]
            per_seed.append(values)
            per_seed_stderr.append(stderrs)
            sample_rows.append([llw, beta2, spacing, seed, *values, *stderrs])
            print(
                f"L/LW={llw:.3g}, L/LD={'sweep' if lld is None else f'{lld:g}'}, q={spacing:.3g}, seed={seed}: "
                f"N1={mc.n1:.4e}, N2={mc.n2:.4e}, 2PC={mc.n_2pc:.4e}, "
                f"3PC={mc.n_3pc_total:.4e}, 4PC={mc.n_4pc:.4e}"
            )
        per_seed = np.asarray(per_seed, dtype=float)
        per_seed_stderr = np.asarray(per_seed_stderr, dtype=float)
        mean = np.mean(per_seed, axis=0)
        if args.n_seeds > 1:
            seed_stderr = np.std(per_seed, axis=0, ddof=1) / np.sqrt(args.n_seeds)
            mc_stderr = np.sqrt(np.mean(per_seed_stderr**2, axis=0) / args.n_seeds)
            stderr = np.sqrt(seed_stderr**2 + mc_stderr**2)
        else:
            stderr = per_seed_stderr[0]
        mean_rows.append([llw, beta2, spacing, *mean])
        stderr_rows.append([llw, beta2, spacing, *stderr])

    data = {
        "llw_grid": llw_grid,
        "spacing_grid": np.asarray(spacing_rows, dtype=float),
        "seeds": seeds,
        "sample_rows": np.asarray(sample_rows, dtype=float),
        "mean_rows": np.asarray(mean_rows, dtype=float),
        "stderr_rows": np.asarray(stderr_rows, dtype=float),
        "length": np.array(float(args.length)),
        "spacing": np.array(float(args.spacing)),
        "lld": np.array(np.nan if lld is None else float(lld)),
        "alpha": np.array(float(args.alpha)),
        "nspan": np.array(int(args.nspan)),
        "phase_delay": np.array(float(args.phase_delay)),
        "n_samples": np.array(int(args.n_samples)),
        "n_seeds": np.array(int(args.n_seeds)),
        "component_names": np.asarray([name for name, _ in COMPONENTS]),
    }
    for idx, (name, _) in enumerate(COMPONENTS):
        data[name] = data["mean_rows"][:, idx + 3]
        data[f"{name}_stderr"] = data["stderr_rows"][:, idx + 3]
    return data


def _plot_positive_with_error(ax: plt.Axes, x: np.ndarray, y: np.ndarray, err: np.ndarray, *, label: str) -> None:
    mask = np.isfinite(y) & (y > 0.0)
    if not np.any(mask):
        return
    yerr = np.where(np.isfinite(err[mask]), err[mask], 0.0)
    ax.errorbar(x[mask], y[mask], yerr=yerr, marker="o", ms=3, lw=1.0, capsize=2, label=label)


def plot_absolute(data: dict[str, np.ndarray], out_dir: Path) -> Path:
    llw = data["llw_grid"]
    fig, ax = plt.subplots(figsize=(5.4, 3.7))
    for name, label in PLOT_COMPONENTS:
        _plot_positive_with_error(ax, llw, data[name], data[f"{name}_stderr"], label=label)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$L/L_W$")
    ax.set_ylabel(r"prefactor-free MC sum")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=7)
    fig.tight_layout()
    path = out_dir / "absolute.pdf"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def plot_fractions(data: dict[str, np.ndarray], out_dir: Path) -> Path:
    llw = data["llw_grid"]
    total = data["n1"]
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for name, label in (("n_2pc", "2PC"), ("n_3pc_total", "3PC"), ("n_4pc", "4PC")):
        frac = np.divide(data[name], total, out=np.full_like(total, np.nan), where=total != 0.0)
        ax.plot(llw, frac, marker="o", ms=3, lw=1.0, label=label)
    ax.set_xscale("log")
    ax.set_xlabel(r"$L/L_W$")
    ax.set_ylabel(r"fraction of $N_1$")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = out_dir / "fractions.pdf"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def plot_local_slopes(data: dict[str, np.ndarray], out_dir: Path) -> Path:
    llw = data["llw_grid"]
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for name, label in PLOT_COMPONENTS:
        y = np.asarray(data[name], dtype=float)
        mask = np.isfinite(y) & (y > 0.0)
        if np.count_nonzero(mask) < 3:
            continue
        slope = np.gradient(np.log(y[mask]), np.log(llw[mask]))
        ax.plot(llw[mask], slope, marker="o", ms=3, lw=1.0, label=label)
    ax.axhline(0.0, color="0.35", lw=0.7, ls=":")
    ax.axhline(-1.0, color="0.35", lw=0.7, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel(r"$L/L_W$")
    ax.set_ylabel(r"local slope $d\log N/d\log(L/L_W)$")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=7)
    fig.tight_layout()
    path = out_dir / "local_slopes.pdf"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def plot_relative_stderr(data: dict[str, np.ndarray], out_dir: Path) -> Path:
    llw = data["llw_grid"]
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for name, label in PLOT_COMPONENTS:
        rel = _safe_relative_stderr(data[name], data[f"{name}_stderr"])
        ax.plot(llw, rel, marker="o", ms=3, lw=1.0, label=label)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$L/L_W$")
    ax.set_ylabel("relative standard error")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=7)
    fig.tight_layout()
    path = out_dir / "relative_stderr.pdf"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def plot_lld_absolute(datasets: dict[float, dict[str, np.ndarray]], out_dir: Path) -> Path:
    fig, axes = plt.subplots(2, 3, figsize=(9.6, 5.4), sharex=True)
    components = (
        ("n_2pc", "2PC"),
        ("n_3pc_total", "3PC"),
        ("n_4pc", "4PC"),
        ("n_3pca", "3PCa"),
        ("n_3pcb", "3PCb"),
        ("n1", r"$N_1$"),
    )
    for ax, (name, title) in zip(axes.ravel(), components, strict=True):
        for lld, data in datasets.items():
            _plot_positive_with_error(ax, data["llw_grid"], data[name], data[f"{name}_stderr"], label=rf"$L/L_D={lld:g}$")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.25)
    for ax in axes[-1, :]:
        ax.set_xlabel(r"$L/L_W$")
    for ax in axes[:, 0]:
        ax.set_ylabel("prefactor-free MC sum")
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    path = out_dir / "lld_absolute.pdf"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def plot_lld_fractions(datasets: dict[float, dict[str, np.ndarray]], out_dir: Path) -> Path:
    fig, axes = plt.subplots(2, 3, figsize=(9.6, 5.2), sharex=True)
    components = (
        ("n_2pc", "2PC"),
        ("n_3pc_total", "3PC"),
        ("n_4pc", "4PC"),
        ("n_3pca", "3PCa"),
        ("n_3pcb", "3PCb"),
    )
    axes_flat = axes.ravel()
    for ax, (name, title) in zip(axes_flat, components, strict=False):
        for lld, data in datasets.items():
            frac = np.divide(data[name], data["n1"], out=np.full_like(data["n1"], np.nan), where=data["n1"] != 0.0)
            mask = np.isfinite(frac)
            ax.plot(data["llw_grid"][mask], frac[mask], marker="o", ms=3, lw=1.0, label=rf"$L/L_D={lld:g}$")
        ax.set_xscale("log")
        ax.set_title(title)
        ax.set_xlabel(r"$L/L_W$")
        ax.grid(True, which="both", alpha=0.25)
        ax.set_ylim(-0.05, 1.05)
    axes_flat[-1].axis("off")
    axes[0, 0].set_ylabel(r"fraction of $N_1$")
    axes[1, 0].set_ylabel(r"fraction of $N_1$")
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    path = out_dir / "lld_fractions.pdf"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Nyquist MC sector scaling versus L/L_W.")
    parser.add_argument("--llw-min", type=float, default=5e-2)
    parser.add_argument("--llw-max", type=float, default=3e2)
    parser.add_argument("--n-llw", type=int, default=24)
    parser.add_argument("--n-samples", type=int, default=200_000)
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--length", type=float, default=1.0)
    parser.add_argument("--spacing", type=float, default=1.0, help="channel spacing over baud, q=Delta f/B")
    parser.add_argument(
        "--lld",
        type=float,
        nargs="+",
        default=None,
        help="Fixed L/L_D values. If set, q is varied as q=(L/L_W)/(2*pi*L/L_D).",
    )
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--nspan", type=int, default=1)
    parser.add_argument("--phase-delay", type=float, default=0.0)
    parser.add_argument("--out-dir", type=Path, default=OUT_MEDIA)
    parser.add_argument("--results-dir", type=Path, default=OUT_RESULTS)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    if args.lld is None:
        data = compute_curves(args)
        result_path = args.results_dir / "scaling.npz"
        np.savez(result_path, **data)
        print(f"saved {result_path}")
        paths = (
            plot_absolute(data, args.out_dir),
            plot_fractions(data, args.out_dir),
            plot_local_slopes(data, args.out_dir),
            plot_relative_stderr(data, args.out_dir),
        )
    else:
        datasets = {}
        for lld in args.lld:
            data = compute_curves(args, lld=float(lld))
            datasets[float(lld)] = data
            token = f"{float(lld):g}".replace(".", "p")
            result_path = args.results_dir / f"lld{token}.npz"
            np.savez(result_path, **data)
            print(f"saved {result_path}")
        paths = (
            plot_lld_absolute(datasets, args.out_dir),
            plot_lld_fractions(datasets, args.out_dir),
        )
    for path in paths:
        print(f"saved {path}")


if __name__ == "__main__":
    main()
