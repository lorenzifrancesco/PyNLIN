from __future__ import annotations

import argparse
import csv
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
from pynlin.methods.td.fullband_mc import compute_fullband_prefactor_free_mc, decimated_frequency_grid
from pynlin.system import System
from pynlin.utils import OpticalBands, nu2lambda


def _parse_targets(value: str | None) -> np.ndarray | None:
    if value is None or not value.strip():
        return None
    return np.array([int(part) for part in value.split(",") if part.strip()], dtype=int)


def _fullband_section(system: System) -> dict:
    raw = system.raw_config if isinstance(system.raw_config, dict) else {}
    methods = raw.get("methods", {}) if isinstance(raw, dict) else {}
    section = methods.get("fullband_mc", {}) if isinstance(methods, dict) else {}
    if isinstance(section, dict) and section:
        return dict(section)
    mc = methods.get("mc", {}) if isinstance(methods, dict) else {}
    if isinstance(mc, dict) and str(mc.get("engine", "")).lower() == "fullband":
        return dict(mc)
    return {}


def _select_targets_from_grid(
    system: System,
    *,
    channel_decimation: int,
    target_decimation: int,
    target_offset: int,
    target_limit: int | None,
    target_band: str | None = None,
) -> np.ndarray:
    _, freqs = decimated_frequency_grid(system, channel_decimation)
    base = np.arange(freqs.size, dtype=int)
    if target_band:
        band_name = target_band.strip().upper()
        try:
            wl_min, wl_max = OpticalBands[band_name].value
        except KeyError as exc:
            valid = ", ".join(b.name for b in OpticalBands)
            raise ValueError(f"Unknown target band {target_band!r}; valid bands: {valid}") from exc
        wavelengths_nm = nu2lambda(freqs) * 1e9
        band_mask = (wavelengths_nm >= wl_min) & (wavelengths_nm < wl_max)
        base = base[band_mask]
        if base.size == 0:
            wl_all = wavelengths_nm
            raise ValueError(
                f"No COIs in {band_name}-band ({wl_min}-{wl_max} nm) after channel_decimation={channel_decimation}. "
                f"Current WDM wavelength range is {float(np.min(wl_all)):.1f}-{float(np.max(wl_all)):.1f} nm."
            )
    target_decimation = max(int(target_decimation), 1)
    target_offset = max(int(target_offset), 0)
    targets = base[target_offset::target_decimation]
    if target_limit is not None:
        targets = targets[: max(int(target_limit), 0)]
    return targets


def _save_pruning_csv(path: Path, diagnostic) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    p = diagnostic.pruning
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["naive_tuples", "support_survivors", "evaluated_tuples", "support_fraction", "evaluated_fraction"])
        writer.writerow([
            p.naive_tuples,
            p.support_survivors,
            p.evaluated_tuples,
            p.support_survivors / max(p.naive_tuples, 1),
            p.evaluated_tuples / max(p.naive_tuples, 1),
        ])


def _save_npz(path: Path, diagnostic) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        target_indices=diagnostic.target_indices,
        target_frequencies=diagnostic.target_frequencies,
        xpm=diagnostic.xpm,
        fwm=diagnostic.fwm,
        total=diagnostic.total,
        fwm_tuple_count=diagnostic.fwm_tuple_count,
        fwm_support_count=diagnostic.fwm_support_count,
        naive_tuples=np.array([diagnostic.pruning.naive_tuples]),
        support_survivors=np.array([diagnostic.pruning.support_survivors]),
        evaluated_tuples=np.array([diagnostic.pruning.evaluated_tuples]),
        **{f"meta_{k}": np.array([v]) for k, v in diagnostic.metadata.items() if v is not None},
    )


def _plot(path: Path, diagnostic) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    freq_thz = diagnostic.target_frequencies * 1e-12
    zdw_frequency = diagnostic.metadata.get("zdw_frequency")
    zdw_thz = None if zdw_frequency is None else float(zdw_frequency) * 1e-12
    fig, axes = plt.subplots(2, 1, figsize=(6.2, 5.2), sharex=True)
    axes[0].plot(freq_thz, diagnostic.xpm, marker="o", lw=1.0, label="XPM MC")
    axes[0].plot(freq_thz, diagnostic.fwm, marker="s", lw=1.0, label="FWM Dar MC")
    axes[0].plot(freq_thz, diagnostic.total, marker="^", lw=1.0, label="total")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("prefactor-free sum [m$^2$]")
    axes[0].grid(True, which="both", alpha=0.25)
    axes[0].legend(fontsize=8, frameon=False)

    axes[1].bar(freq_thz, diagnostic.fwm_support_count, width=0.02, label="support-pruned")
    axes[1].bar(freq_thz, diagnostic.fwm_tuple_count, width=0.012, label="sampled")
    axes[1].set_xlabel("target frequency [THz]")
    axes[1].set_ylabel("evaluated FWM tuples")
    axes[1].grid(True, axis="y", alpha=0.25)
    axes[1].legend(fontsize=8, frameon=False)
    if zdw_thz is not None and np.isfinite(zdw_thz):
        for ax in axes:
            ax.axvline(zdw_thz, color="crimson", lw=0.8, ls=":", alpha=0.7)
        axes[1].scatter(
            [zdw_thz],
            [0.0],
            marker="*",
            s=110,
            color="crimson",
            edgecolor="black",
            linewidth=0.4,
            transform=axes[1].get_xaxis_transform(),
            clip_on=False,
            zorder=5,
            label="ZDW",
        )
        axes[1].annotate(
            "ZDW",
            xy=(zdw_thz, 0.0),
            xycoords=axes[1].get_xaxis_transform(),
            xytext=(0, -18),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=8,
            color="crimson",
        )
        x_min = min(float(np.min(freq_thz)), zdw_thz)
        x_max = max(float(np.max(freq_thz)), zdw_thz)
        pad = max((x_max - x_min) * 0.03, 0.02)
        axes[1].set_xlim(x_min - pad, x_max + pad)
    fig.suptitle(
        "Prefactor-free fullband MC diagnostic\n"
        f"decimation={diagnostic.metadata['decimation']}, channels={diagnostic.metadata['n_channels_decimated']}"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    fig.savefig(path.with_suffix(".png"), dpi=300)
    plt.close(fig)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prefactor-free fullband XPM+FWM Dar MC diagnostic.")
    parser.add_argument("--config", type=Path, default=Path("input/studies.toml"))
    parser.add_argument("--out-dir", type=Path, default=Path("media/fullband-mc"))
    parser.add_argument("--decimation", type=int, default=None, help="Channel-grid decimation factor.")
    parser.add_argument(
        "--targets",
        type=str,
        default=None,
        help="Explicit COIs as decimated-grid positions, e.g. 0,3,9. If omitted, COIs are selected from the TOML WDM grid.",
    )
    parser.add_argument("--target-decimation", type=int, default=None, help="COI decimation after channel-grid decimation.")
    parser.add_argument("--target-offset", type=int, default=None, help="First decimated-grid COI position.")
    parser.add_argument("--target-limit", type=int, default=None, help="Maximum number of COIs to compute.")
    parser.add_argument("--target-band", type=str, default=None, help="Restrict COIs to an optical band: O, E, S, C, L, U.")
    parser.add_argument("--xpm-samples", type=int, default=None)
    parser.add_argument("--fwm-samples", type=int, default=None)
    parser.add_argument("--fwm-frequency-samples", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None, help="Target-level worker processes; 0 uses all CPUs.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-fwm-tuples-per-target", type=int, default=None)
    parser.add_argument(
        "--fwm-tuple-selection",
        choices=("joint_reservoir", "exhaustive_support_mc", "reservoir", "phase_proxy"),
        default=None,
    )
    parser.add_argument("--no-xpm", action="store_true")
    parser.add_argument("--no-fwm", action="store_true")
    return parser.parse_args()


def main() -> None:
    init_logging()
    args = _parse_args()
    system = System.from_toml(args.config)
    section = _fullband_section(system)
    channel_decimation = int(args.decimation if args.decimation is not None else section.get("channel_decimation", 100))
    target_decimation = int(args.target_decimation if args.target_decimation is not None else section.get("target_decimation", 1))
    target_offset = int(args.target_offset if args.target_offset is not None else section.get("target_offset", 0))
    target_limit = args.target_limit if args.target_limit is not None else section.get("target_limit")
    target_limit = None if target_limit is None else int(target_limit)
    target_band = args.target_band if args.target_band is not None else section.get("target_band")
    xpm_samples = int(args.xpm_samples if args.xpm_samples is not None else section.get("xpm_samples", 5000))
    fwm_samples = int(args.fwm_samples if args.fwm_samples is not None else section.get("fwm_samples", 2000))
    fwm_frequency_samples = int(
        args.fwm_frequency_samples
        if args.fwm_frequency_samples is not None
        else section.get("fwm_frequency_samples", 50)
    )
    seed = int(args.seed if args.seed is not None else section.get("seed", 1234))
    workers = int(args.workers if args.workers is not None else section.get("workers", 1))
    max_fwm_tuples = (
        args.max_fwm_tuples_per_target
        if args.max_fwm_tuples_per_target is not None
        else section.get("max_fwm_tuples_per_target")
    )
    max_fwm_tuples = None if max_fwm_tuples is None else int(max_fwm_tuples)
    fwm_tuple_selection = str(
        args.fwm_tuple_selection
        if args.fwm_tuple_selection is not None
        else section.get("fwm_tuple_selection", "joint_reservoir")
    )
    targets = _parse_targets(args.targets)
    if targets is None:
        targets = _select_targets_from_grid(
            system,
            channel_decimation=channel_decimation,
            target_decimation=target_decimation,
            target_offset=target_offset,
            target_limit=target_limit,
            target_band=target_band,
        )
    lg.info(
        f"fullband MC diagnostic: channel_decimation={channel_decimation}, targets={targets}, "
        f"target_band={target_band}, target_decimation={target_decimation}, "
        f"target_offset={target_offset}, target_limit={target_limit}, "
        f"xpm_samples={xpm_samples}, fwm_samples={fwm_samples}, "
        f"fwm_frequency_samples={fwm_frequency_samples}, tuple_selection={fwm_tuple_selection}, workers={workers}"
    )
    diagnostic = compute_fullband_prefactor_free_mc(
        system,
        decimation=channel_decimation,
        target_indices=targets,
        include_xpm=not args.no_xpm,
        include_fwm=not args.no_fwm,
        xpm_samples=xpm_samples,
        fwm_samples=fwm_samples,
        fwm_frequency_samples=fwm_frequency_samples,
        seed=seed,
        max_fwm_tuples_per_target=max_fwm_tuples,
        fwm_tuple_selection=fwm_tuple_selection,
        n_workers=workers,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _save_npz(args.out_dir / "fullband_mc_prefactor_free.npz", diagnostic)
    _save_pruning_csv(args.out_dir / "tuple_pruning_summary.csv", diagnostic)
    _plot(args.out_dir / "fullband_mc_prefactor_free.pdf", diagnostic)
    lg.success(f"saved fullband MC diagnostic to {args.out_dir}")
    lg.info(
        "FWM pruning: naive={} support_survivors={} evaluated={} support_fraction={:.3e}".format(
            diagnostic.pruning.naive_tuples,
            diagnostic.pruning.support_survivors,
            diagnostic.pruning.evaluated_tuples,
            diagnostic.pruning.support_survivors / max(diagnostic.pruning.naive_tuples, 1),
        )
    )


if __name__ == "__main__":
    main()
