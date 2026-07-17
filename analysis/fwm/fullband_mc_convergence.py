from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pynlin  # noqa: F401
from loguru import logger as lg

from analysis.log_init import init_logging
from pynlin.methods.td.fullband_mc import compute_fullband_prefactor_free_mc
from pynlin.system import System


def _parse_ints(value: str) -> list[int]:
    return [int(part) for part in value.split(",") if part.strip()]


def run_convergence(
    *,
    system: System,
    targets: list[int],
    caps: list[int],
    seeds: list[int],
    fwm_samples: int,
    fwm_frequency_samples: int,
    decimation: int,
    selection_mode: str,
    workers: int,
) -> dict[str, np.ndarray]:
    rows = []
    for target in targets:
        for cap in caps:
            for seed in seeds:
                lg.info(
                    f"convergence run: target={target}, cap={cap}, "
                    f"fwm_samples={fwm_samples}, seed={seed}"
                )
                diag = compute_fullband_prefactor_free_mc(
                    system,
                    decimation=decimation,
                    target_indices=np.array([target], dtype=int),
                    include_xpm=False,
                    include_fwm=True,
                    xpm_samples=1,
                    fwm_samples=fwm_samples,
                    fwm_frequency_samples=fwm_frequency_samples,
                    seed=seed,
                    max_fwm_tuples_per_target=cap,
                    fwm_tuple_selection=selection_mode,
                    n_workers=workers,
                )
                rows.append(
                    (
                        target,
                        cap,
                        seed,
                        diag.fwm[0],
                        diag.fwm_support_count[0],
                        diag.fwm_tuple_count[0],
                        diag.target_frequencies[0],
                    )
                )
    data = np.asarray(rows, dtype=float)
    return {
        "target": data[:, 0].astype(int),
        "cap": data[:, 1].astype(int),
        "seed": data[:, 2].astype(int),
        "fwm": data[:, 3],
        "support_count": data[:, 4].astype(int),
        "sampled_count": data[:, 5].astype(int),
        "target_frequency": data[:, 6],
        "fwm_samples": np.array([int(fwm_samples)]),
        "fwm_frequency_samples": np.array([int(fwm_frequency_samples)]),
        "decimation": np.array([int(decimation)]),
        "selection_mode": np.array([selection_mode]),
        "workers": np.array([int(workers)]),
    }


def plot_convergence(data: dict[str, np.ndarray], out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    targets = np.unique(data["target"])
    fig, axes = plt.subplots(
        targets.size,
        1,
        figsize=(6.2, max(3.0, 2.4 * targets.size)),
        sharex=True,
        squeeze=False,
    )
    axes = axes[:, 0]
    for ax, target in zip(axes, targets, strict=True):
        mask_t = data["target"] == target
        caps = np.unique(data["cap"][mask_t])
        means = []
        stds = []
        for cap in caps:
            vals = data["fwm"][mask_t & (data["cap"] == cap)]
            means.append(float(np.mean(vals)))
            stds.append(float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0)
            ax.scatter(np.full(vals.size, cap), vals, s=14, color="C0", alpha=0.55)
        means = np.asarray(means)
        stds = np.asarray(stds)
        ax.errorbar(caps, means, yerr=stds, marker="o", lw=1.0, color="C1", label="seed mean ± std")
        if means[-1] > 0:
            ax2 = ax.twinx()
            ax2.plot(caps, means / means[-1], marker="s", lw=0.8, color="C2", label="mean / finest")
            ax2.axhline(1.0, color="0.4", lw=0.6, ls=":")
            ax2.set_ylabel("relative to finest", color="C2")
            ax2.tick_params(axis="y", labelcolor="C2")
        freq_thz = float(data["target_frequency"][mask_t][0]) * 1e-12
        support = int(data["support_count"][mask_t][0])
        ax.set_title(f"target {int(target)} ({freq_thz:.3f} THz), support tuples={support:,}")
        ax.set_ylabel("FWM prefactor-free sum [m$^2$]")
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.25)
    axes[-1].set_xlabel("sampled FWM tuples per COI")
    axes[-1].set_xscale("log")
    fig.suptitle(
        "Full-grid FWM tuple-sampling convergence, "
        f"frequency samples={int(data['fwm_frequency_samples'][0])}"
    )
    fig.tight_layout()
    pdf = out_dir / "fullband_fwm_tuple_convergence.pdf"
    png = pdf.with_suffix(".png")
    fig.savefig(pdf, dpi=300)
    fig.savefig(png, dpi=300)
    plt.close(fig)
    return [pdf, png]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convergence of fullband FWM tuple sampling.")
    parser.add_argument("--config", type=Path, default=Path("input/studies.toml"))
    parser.add_argument("--out-dir", type=Path, default=Path("media/fullband-mc/convergence"))
    parser.add_argument("--targets", type=str, default="0,9,12,16")
    parser.add_argument("--caps", type=str, default="500,1000,2000,5000")
    parser.add_argument("--seeds", type=str, default="1234,2234,3234")
    parser.add_argument("--fwm-samples", type=int, default=2000)
    parser.add_argument("--fwm-frequency-samples", type=int, default=50)
    parser.add_argument("--decimation", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1, help="Target-level worker processes per run; 0 uses all CPUs.")
    parser.add_argument(
        "--selection-mode",
        choices=("reservoir", "phase_proxy", "joint_reservoir", "exhaustive_support_mc"),
        default="joint_reservoir",
    )
    return parser.parse_args()


def main() -> None:
    init_logging()
    args = _parse_args()
    system = System.from_toml(args.config)
    data = run_convergence(
        system=system,
        targets=_parse_ints(args.targets),
        caps=_parse_ints(args.caps),
        seeds=_parse_ints(args.seeds),
        fwm_samples=args.fwm_samples,
        fwm_frequency_samples=args.fwm_frequency_samples,
        decimation=args.decimation,
        selection_mode=args.selection_mode,
        workers=args.workers,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    npz = args.out_dir / "fullband_fwm_tuple_convergence.npz"
    np.savez(npz, **data)
    lg.success(f"saved {npz}")
    for path in plot_convergence(data, args.out_dir):
        lg.success(f"saved {path}")

    print("target cap mean std rel_std")
    for target in np.unique(data["target"]):
        mask_t = data["target"] == target
        for cap in np.unique(data["cap"][mask_t]):
            vals = data["fwm"][mask_t & (data["cap"] == cap)]
            mean = float(np.mean(vals))
            std = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0
            print(f"{int(target)} {int(cap)} {mean:.6e} {std:.6e} {std/max(mean,1e-300):.3e}")


if __name__ == "__main__":
    main()
