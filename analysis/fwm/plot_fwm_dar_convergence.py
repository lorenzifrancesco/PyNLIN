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

from analysis.fwm.compare_fwm_mc_direct import _beta2, _channel_params
from analysis.log_init import init_logging
from pynlin.methods.td.fwm_kernel import compute_fwm_kernel_direct
from pynlin.methods.td.fwm_mc import estimate_fwm_term_sum_dar_mc
from pynlin.pulses import pulse_from_config
from pynlin.system import System


def _diagnostic_pulse(system: System, num_symbols: int, samples_per_symbol: int):
    return pulse_from_config(
        system.pulse_config,
        num_symbols=num_symbols,
        samples_per_symbol=samples_per_symbol,
    )


def compute_convergence(
    *,
    system: System,
    target: str,
    term: str,
    spacing: float,
    radii: list[int],
    mc_samples: int,
    seed: int,
    z_points: int,
    num_symbols: int,
    samples_per_symbol: int,
    min_pts_per_period: float,
    max_z_points: int,
) -> dict[str, np.ndarray]:
    pulse = _diagnostic_pulse(system, num_symbols, samples_per_symbol)
    beta2 = _beta2(system)
    length = float(system.fiber_length)
    baud_rate = float(pulse.baud_rate)
    channels = _channel_params(target, term, spacing, baud_rate, beta2)
    z = np.linspace(0.0, length, z_points)
    norm = (1.0 / baud_rate) ** 2

    mc = estimate_fwm_term_sum_dar_mc(
        channels=channels,
        baud_rate=baud_rate,
        length=length,
        n_samples=mc_samples,
        seed=seed,
        alpha=0.0,
    )
    direct = []
    nz_actual = []
    entries = []
    for radius in radii:
        values = np.arange(-radius, radius + 1)
        lg.info(f"direct radius R={radius}, entries={values.size**3}")
        table = compute_fwm_kernel_direct(
            pulse,
            z,
            values,
            values,
            values,
            channels=channels,
            auto_refine=True,
            min_pts_per_period=min_pts_per_period,
            max_z_points=max_z_points,
            discretization_action="warn",
        )
        direct.append(float(np.sum(np.abs(table.X) ** 2) * norm))
        nz_actual.append(int(table.metadata["n_z"]))
        entries.append(int(values.size**3))

    return {
        "radii": np.asarray(radii, dtype=int),
        "direct": np.asarray(direct, dtype=float),
        "dar_mc": np.array([mc.total], dtype=float),
        "dar_mc_stderr": np.array([mc.total_stderr], dtype=float),
        "nz_actual": np.asarray(nz_actual, dtype=int),
        "entries": np.asarray(entries, dtype=int),
        "spacing_over_baud": np.array([spacing], dtype=float),
        "target": np.array([target]),
        "term": np.array([term]),
        "beta2_si": np.array([beta2], dtype=float),
        "mc_samples": np.array([mc_samples], dtype=int),
    }


def plot_convergence(data: dict[str, np.ndarray], out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    radii = data["radii"]
    direct = data["direct"]
    mc = float(data["dar_mc"][0])
    mc_stderr = float(data["dar_mc_stderr"][0])
    ratio = direct / mc
    term = str(data["term"][0])
    target = str(data["target"][0])
    spacing = float(data["spacing_over_baud"][0])
    beta2_ps2_per_km = float(data["beta2_si"][0] / 1e-27)

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.2))
    ax = axes[0]
    ax.fill_between(
        [radii[0], radii[-1]],
        [mc - mc_stderr, mc - mc_stderr],
        [mc + mc_stderr, mc + mc_stderr],
        color="C1",
        alpha=0.2,
        label="Dar MC stderr",
    )
    ax.axhline(mc, color="C1", lw=1.0, ls="--", label="Dar MC all-index")
    ax.plot(radii, direct, marker="o", lw=1.0, color="C0", label=r"direct $T^2\sum|X|^2$")
    ax.set_xlabel("direct h/k/m radius R")
    ax.set_ylabel(r"prefactor-free FWM sum [m$^2$]")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=7, frameon=False)

    ax = axes[1]
    ax.axhline(1.0, color="0.35", lw=0.8, ls=":")
    ax.fill_between(
        [radii[0], radii[-1]],
        [(mc - mc_stderr) / mc, (mc - mc_stderr) / mc],
        [(mc + mc_stderr) / mc, (mc + mc_stderr) / mc],
        color="C1",
        alpha=0.2,
    )
    ax.plot(radii, ratio, marker="o", lw=1.0, color="C0")
    for r, y, entries, nz in zip(radii, ratio, data["entries"], data["nz_actual"], strict=True):
        ax.annotate(f"{entries}\nnz={nz}", (r, y), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=6)
    ax.set_xlabel("direct h/k/m radius R")
    ax.set_ylabel("direct / Dar MC")
    ax.set_ylim(0.0, 1.15)
    ax.grid(True, alpha=0.25)

    fig.suptitle(
        rf"FWM $X^{{{term}\to {target}}}$, $\Delta f/B={spacing:g}$, "
        rf"$\beta_2={beta2_ps2_per_km:.1f}$ ps$^2$/km, MC={int(data['mc_samples'][0])}"
    )
    fig.tight_layout()
    stem = f"fwm_dar_convergence_{term}_to_{target}_q{spacing:g}".replace(".", "p")
    pdf = out_dir / f"{stem}.pdf"
    png = pdf.with_suffix(".png")
    fig.savefig(pdf, dpi=300)
    fig.savefig(png, dpi=300)
    plt.close(fig)
    return [pdf, png]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot convergence of direct FWM sums toward Dar frequency MC.")
    parser.add_argument("--config", type=Path, default=Path("input/studies.toml"))
    parser.add_argument("--out-dir", type=Path, default=Path("media/fwm"))
    parser.add_argument("--target", default="a")
    parser.add_argument("--term", default="abb")
    parser.add_argument("--spacing", type=float, default=1.0)
    parser.add_argument("--radii", type=int, nargs="+", default=[2, 3, 4, 5, 7])
    parser.add_argument("--mc-samples", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--z-points", type=int, default=7)
    parser.add_argument("--num-symbols", type=int, default=64)
    parser.add_argument("--samples-per-symbol", type=int, default=24)
    parser.add_argument("--min-pts-per-period", type=float, default=3.0)
    parser.add_argument("--max-z-points", type=int, default=500)
    return parser.parse_args()


def main() -> None:
    init_logging()
    args = _parse_args()
    system = System.from_toml(args.config)
    data = compute_convergence(
        system=system,
        target=args.target,
        term=args.term,
        spacing=args.spacing,
        radii=args.radii,
        mc_samples=args.mc_samples,
        seed=args.seed,
        z_points=args.z_points,
        num_symbols=args.num_symbols,
        samples_per_symbol=args.samples_per_symbol,
        min_pts_per_period=args.min_pts_per_period,
        max_z_points=args.max_z_points,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    spacing_label = f"{args.spacing:g}".replace(".", "p")
    npz = args.out_dir / f"fwm_dar_convergence_{args.term}_to_{args.target}_q{spacing_label}.npz"
    np.savez(npz, **data)
    lg.success(f"saved {npz}")
    for path in plot_convergence(data, args.out_dir):
        lg.success(f"saved {path}")


if __name__ == "__main__":
    main()
