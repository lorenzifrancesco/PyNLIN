from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pynlin  # noqa: F401  # initialize loguru fallback when needed
from loguru import logger as lg

from analysis.log_init import init_logging
from pynlin.methods.td.xhkm_sums import compute_xhkm_sums
from pynlin.methods.td.xhkm_mc import assert_flat_signal_power_profile, estimate_xhkm_sums_mc
from pynlin.methods.td.xpm_kernel import compute_xpm_kernel_fft
from pynlin.pulses import pulse_from_config
from pynlin.system import System


@dataclass(frozen=True)
class MCValidationConfig:
    sample_grid: tuple[int, ...] = (1000, 3000, 10000)
    n_seeds: int = 3
    seed: int = 1234
    spacing_grid: tuple[float, ...] = (0.75, 1.0, 1.25, 1.5, 2.0)
    mc_samples: int = 50000
    mc_alpha: float = 1e-4
    h_radius: int = 5
    r_radius: int = 5
    m_radius: int = 9
    z_points: int = 33
    num_symbols: int = 128
    samples_per_symbol: int = 16


def _config_from_system(system: System) -> MCValidationConfig:
    raw = system.raw_config if isinstance(system.raw_config, Mapping) else {}
    section = raw.get("mc_validation") if isinstance(raw, Mapping) else None
    if not isinstance(section, Mapping):
        return MCValidationConfig()
    defaults = MCValidationConfig()
    return MCValidationConfig(
        sample_grid=tuple(int(v) for v in section.get("sample_grid", defaults.sample_grid)),
        n_seeds=int(section.get("n_seeds", defaults.n_seeds)),
        seed=int(section.get("seed", defaults.seed)),
        spacing_grid=tuple(float(v) for v in section.get("spacing_grid", defaults.spacing_grid)),
        mc_samples=int(section.get("mc_samples", defaults.mc_samples)),
        mc_alpha=float(section.get("mc_alpha", defaults.mc_alpha)),
        h_radius=int(section.get("h_radius", defaults.h_radius)),
        r_radius=int(section.get("r_radius", defaults.r_radius)),
        m_radius=int(section.get("m_radius", defaults.m_radius)),
        z_points=int(section.get("z_points", defaults.z_points)),
        num_symbols=int(section.get("num_symbols", defaults.num_symbols)),
        samples_per_symbol=int(section.get("samples_per_symbol", defaults.samples_per_symbol)),
    )


def run_validation(args: argparse.Namespace) -> Path:
    system = System.from_toml(args.config)
    assert_flat_signal_power_profile(system)
    config = _config_from_system(system)
    lg.info(f"validated flat profile in {args.config}")
    lg.info(f"MC validation config: {config}")

    rows = []
    beta2_mc = _mc_beta2_from_system(system)
    length = float(system.fiber_length)
    for n_samples in config.sample_grid:
        for seed in range(config.seed, config.seed + config.n_seeds):
            sums = estimate_xhkm_sums_mc(
                beta2=beta2_mc,
                alpha=config.mc_alpha,
                length=length,
                channel_spacing_over_baud=config.spacing_grid[0],
                nspan=1,
                phase_delay=0.0,
                n_samples=n_samples,
                seed=seed,
                system=system,
            )
            rows.append((n_samples, seed, sums.n1, sums.n2, sums.n1_stderr, sums.n2_stderr))
            lg.info(
                f"n={n_samples:g}, seed={seed}: N1={sums.n1:.4e}, "
                f"N2={sums.n2:.4e}, N2/N1={sums.n2_over_n1:.4f}"
            )

    data = np.asarray(rows, dtype=float)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_npz = args.out_dir / "dar_mc_prefactor_free_xhkm_sums.npz"
    np.savez(
        out_npz,
        n_samples=data[:, 0].astype(int),
        seed=data[:, 1].astype(int),
        n1=data[:, 2],
        n2=data[:, 3],
        n1_stderr=data[:, 4],
        n2_stderr=data[:, 5],
        beta2=beta2_mc,
        alpha=config.mc_alpha,
        length=length,
        channel_spacing_over_baud=config.spacing_grid[0],
        nspan=1,
        phase_delay=0.0,
    )
    lg.success(f"saved {out_npz}")

    _plot_convergence(data, args.out_dir)
    comparison = _compute_mc_direct_comparison(system, config)
    _save_comparison(comparison, args.out_dir)
    _plot_mc_direct_comparison(comparison, args.out_dir)
    return out_npz


def _mc_beta2_from_system(system: System) -> float:
    beta2 = float(system.numerics.gvd if system.numerics is not None else system.raw_config["fiber"]["beta2"])
    T = 1.0 / float(system.pulse.baud_rate)
    return beta2 / (T * T)


def _diagnostic_pulse(system: System, config: MCValidationConfig):
    if system.pulse_config is None:
        return system.pulse
    return pulse_from_config(
        system.pulse_config,
        num_symbols=config.num_symbols,
        samples_per_symbol=config.samples_per_symbol,
    )


def _compute_mc_direct_comparison(system: System, config: MCValidationConfig) -> dict[str, np.ndarray]:
    pulse = _diagnostic_pulse(system, config)
    length = float(system.fiber_length)
    baud_rate = float(pulse.baud_rate)
    beta2_si = float(system.numerics.gvd if system.numerics is not None else system.raw_config["fiber"]["beta2"])
    beta2_mc = beta2_si / (1.0 / baud_rate) ** 2
    T = 1.0 / baud_rate

    rows = []
    for spacing in config.spacing_grid:
        dgd = abs(beta2_si) * 2.0 * np.pi * float(spacing) * baud_rate
        llw = length * baud_rate * dgd
        lg.info(f"direct Xhkm spacing/B={spacing:g}: LLW={llw:.3g}, dgd={dgd:.3e}")

        # Adaptive window: scale h/r/m radii linearly with LLW, capped to
        # keep computation manageable.  At LLW≈5.7 (q=0.75), h_r=5,m_r=9 gives
        # ~96% N1 coverage; at LLW≈15 (q=2.0) we cap at h_r=10,m_r=21.
        llw_ref = 5.66
        scale = max(1.0, llw / llw_ref)
        h_r = max(config.h_radius, min(10, int(np.ceil(config.h_radius * scale))))
        r_r = max(config.r_radius, min(10, int(np.ceil(config.r_radius * scale))))
        m_r = max(config.m_radius, min(21, int(np.ceil(config.m_radius * scale))))

        h_values = np.arange(-h_r, h_r + 1)
        r_values = np.arange(-r_r, r_r + 1)
        m_values = np.arange(-m_r, m_r + 1)
        n_entries = h_values.size * r_values.size * m_values.size

        z0 = np.linspace(0.0, length, max(config.z_points, 9))
        table = compute_xpm_kernel_fft(
            pulse,
            z0,
            h_values,
            r_values,
            m_values,
            dgd=dgd,
            gvda=beta2_si,
            gvdb=beta2_si,
            auto_refine=True,
            discretization_action="silent",
            min_pts_per_collision=3.0,
        )
        actual_nz = table.metadata["n_z"]
        direct = compute_xhkm_sums(table.X, table.h_values, table.r_values, table.m_values)

        mc = estimate_xhkm_sums_mc(
            beta2=beta2_mc,
            alpha=0.0,
            length=length,
            channel_spacing_over_baud=float(spacing),
            n_samples=config.mc_samples,
            seed=config.seed,
            system=system,
        )
        norm = T ** 2
        rows.append((spacing, llw, direct.n1 * norm, direct.n2 * norm,
                     mc.n1, mc.n2, mc.n1_stderr, mc.n2_stderr, h_r, m_r, actual_nz, n_entries))
        lg.info(f"  h_r={h_r} m_r={m_r} nz={actual_nz} entries={n_entries}  "
                f"N1/MC={direct.n1*norm/mc.n1:.4f}  N2/MC={direct.n2*norm/mc.n2:.4f}")

    data = np.asarray(rows, dtype=float)
    return {
        "spacing_over_baud": data[:, 0],
        "llw": data[:, 1],
        "direct_n1": data[:, 2],
        "direct_n2": data[:, 3],
        "mc_n1": data[:, 4],
        "mc_n2": data[:, 5],
        "mc_n1_stderr": data[:, 6],
        "mc_n2_stderr": data[:, 7],
        "h_radius": data[:, 8],
        "m_radius": data[:, 9],
        "nz_actual": data[:, 10],
        "n_entries": data[:, 11],
    }


def _save_comparison(comparison: dict[str, np.ndarray], out_dir: Path) -> None:
    path = out_dir / "mc_vs_direct_xhkm_sums.npz"
    np.savez(path, **comparison)
    lg.success(f"saved {path}")


def _plot_convergence(data: np.ndarray, out_dir: Path) -> None:
    sample_grid = np.unique(data[:, 0].astype(int))
    means = []
    stds = []
    for n_samples in sample_grid:
        block = data[data[:, 0] == n_samples]
        means.append([np.mean(block[:, 2]), np.mean(block[:, 3])])
        stds.append([np.std(block[:, 2], ddof=1), np.std(block[:, 3], ddof=1)] if block.shape[0] > 1 else [0.0, 0.0])
    means = np.asarray(means)
    stds = np.asarray(stds)

    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    ax.errorbar(sample_grid, means[:, 0], yerr=stds[:, 0], marker="o", lw=1.0, label=r"$N_1^{MC}$")
    ax.errorbar(sample_grid, means[:, 1], yerr=stds[:, 1], marker="s", lw=1.0, label=r"$N_2^{MC}$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("MC samples")
    ax.set_ylabel(r"prefactor-free sum [m$^2$]")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = out_dir / "dar_mc_xhkm_n1_n2_convergence.pdf"
    fig.savefig(path, dpi=300)
    fig.savefig(path.with_suffix(".png"), dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.0, 2.6))
    ratio = means[:, 1] / means[:, 0]
    ax.plot(sample_grid, ratio, marker="o", lw=1.0)
    ax.set_xscale("log")
    ax.set_xlabel("MC samples")
    ax.set_ylabel(r"$N_2^{MC}/N_1^{MC}$")
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    path = out_dir / "dar_mc_xhkm_n2_over_n1.pdf"
    fig.savefig(path, dpi=300)
    fig.savefig(path.with_suffix(".png"), dpi=300)
    plt.close(fig)
    lg.success(f"saved plots in {out_dir}")


def _plot_mc_direct_comparison(comparison: dict[str, np.ndarray], out_dir: Path) -> None:
    spacing = comparison["spacing_over_baud"]
    direct_n1 = comparison["direct_n1"]
    direct_n2 = comparison["direct_n2"]
    mc_n1 = comparison["mc_n1"]
    mc_n2 = comparison["mc_n2"]
    mc_n1_lo = mc_n1 - comparison["mc_n1_stderr"]
    mc_n1_hi = mc_n1 + comparison["mc_n1_stderr"]
    mc_n2_lo = mc_n2 - comparison["mc_n2_stderr"]
    mc_n2_hi = mc_n2 + comparison["mc_n2_stderr"]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharex=True)
    axes[0].fill_between(spacing, mc_n1_lo, mc_n1_hi, alpha=0.2, color="C1")
    axes[0].plot(spacing, direct_n1, marker="o", lw=1.0, label=r"direct $N_1$")
    axes[0].plot(spacing, mc_n1, marker="s", lw=1.0, ls="--", label=r"MC $N_1$")
    axes[0].set_title(r"$N_1 = T^2 \sum |X_{hkm}|^2 \quad (\alpha\!=\!0)$")
    axes[1].fill_between(spacing, mc_n2_lo, mc_n2_hi, alpha=0.2, color="C1")
    axes[1].plot(spacing, direct_n2, marker="o", lw=1.0, label=r"direct $N_2$")
    axes[1].plot(spacing, mc_n2, marker="s", lw=1.0, ls="--", label=r"MC $N_2$")
    axes[1].set_title(r"$N_2 = T^2(2\pi)^{-2} \sum |X_{h0m}|^2$")
    for ax in axes:
        ax.set_yscale("log")
        ax.set_xlabel(r"spacing $\Delta f/B$")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=8)
    axes[0].set_ylabel(r"prefactor-free sum [m$^2$]")
    fig.tight_layout()
    path = out_dir / "mc_vs_direct_xhkm_n1_n2.pdf"
    fig.savefig(path, dpi=300)
    fig.savefig(path.with_suffix(".png"), dpi=300)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharex=True)
    axes[0].plot(spacing, direct_n1 / mc_n1, marker="o", lw=1.0)
    axes[0].axhline(1.0, color="0.35", lw=0.8, ls=":")
    axes[0].set_title(r"$N_1$: direct / MC")
    axes[1].plot(spacing, direct_n2 / mc_n2, marker="o", lw=1.0)
    axes[1].axhline(1.0, color="0.35", lw=0.8, ls=":")
    axes[1].set_title(r"$N_2$: direct / MC")
    for ax in axes:
        ax.set_xlabel(r"spacing $\Delta f/B$")
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    path = out_dir / "mc_vs_direct_xhkm_ratio.pdf"
    fig.savefig(path, dpi=300)
    fig.savefig(path.with_suffix(".png"), dpi=300)
    plt.close(fig)
    lg.success(f"saved MC-vs-direct comparison plots in {out_dir}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate prefactor-free Dar MC N1/N2 estimates.")
    parser.add_argument("--config", type=Path, default=Path("input/studies.toml"))
    parser.add_argument("--out-dir", type=Path, default=Path("media/mc-validation"))
    return parser.parse_args()


def main() -> None:
    init_logging()
    run_validation(_parse_args())


if __name__ == "__main__":
    main()
