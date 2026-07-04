from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pynlin  # noqa: F401  # initializes the project loguru fallback when needed
from loguru import logger as lg

from analysis.log_init import init_logging
from pynlin.methods.td.fwm_kernel import FWMChannels, compute_fwm_kernel_direct
from pynlin.methods.td.fwm_mc import estimate_fwm_term_sum_dar_mc, estimate_fwm_term_sum_mc
from pynlin.pulses import Pulse, pulse_from_config
from pynlin.system import System


@dataclass(frozen=True)
class FWMTermResult:
    target: str
    term: str
    pulse_shape: str
    hkm_radius: int
    spacing_over_baud: float
    delta_omega_over_2pi_baud: float
    delta_beta0: float
    total: float
    locked: float
    partial: float
    unlocked: float


@dataclass(frozen=True)
class FWMDiagnosticConfig:
    hkm_radius: int = 3
    compare_next_radius: bool = True
    z_points: int = 5
    num_symbols: int = 48
    samples_per_symbol: int = 16
    spacing_grid: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
    symmetry_spacing: float = 1.0
    workers: int = 1
    auto_refine_z: bool = True
    min_pts_per_period: float = 3.0
    max_z_points: int = 500
    mc_backend: str = "direct"
    mc_samples: int = 1000
    mc_seed: int = 1234


def _fwm_config_from_system(system: System) -> FWMDiagnosticConfig:
    raw = system.raw_config if isinstance(system.raw_config, Mapping) else {}
    section = raw.get("fwm") if isinstance(raw, Mapping) else None
    if not isinstance(section, Mapping):
        cfg = FWMDiagnosticConfig()
        _validate_fwm_config(cfg)
        return cfg

    defaults = FWMDiagnosticConfig()
    spacing_grid = section.get("spacing_grid", defaults.spacing_grid)
    cfg = FWMDiagnosticConfig(
        hkm_radius=int(section.get("hkm_radius", defaults.hkm_radius)),
        compare_next_radius=bool(
            section.get("compare_next_radius", defaults.compare_next_radius)
        ),
        z_points=int(section.get("z_points", defaults.z_points)),
        num_symbols=int(section.get("num_symbols", defaults.num_symbols)),
        samples_per_symbol=int(section.get("samples_per_symbol", defaults.samples_per_symbol)),
        spacing_grid=tuple(float(value) for value in spacing_grid),
        symmetry_spacing=float(section.get("symmetry_spacing", defaults.symmetry_spacing)),
        workers=int(section.get("workers", defaults.workers)),
        auto_refine_z=bool(section.get("auto_refine_z", defaults.auto_refine_z)),
        min_pts_per_period=float(
            section.get("min_pts_per_period", section.get("min_pts_per_collision", defaults.min_pts_per_period))
        ),
        max_z_points=int(section.get("max_z_points", defaults.max_z_points)),
        mc_backend=str(section.get("mc_backend", defaults.mc_backend)),
        mc_samples=int(section.get("mc_samples", defaults.mc_samples)),
        mc_seed=int(section.get("mc_seed", defaults.mc_seed)),
    )
    _validate_fwm_config(cfg)
    return cfg


def _validate_fwm_config(config: FWMDiagnosticConfig) -> None:
    if config.hkm_radius < 0:
        raise ValueError("fwm.hkm_radius must be non-negative")
    if config.z_points < 2:
        raise ValueError("fwm.z_points must be at least 2")
    if config.num_symbols <= 0:
        raise ValueError("fwm.num_symbols must be positive")
    if config.samples_per_symbol <= 0:
        raise ValueError("fwm.samples_per_symbol must be positive")
    if not config.spacing_grid:
        raise ValueError("fwm.spacing_grid must contain at least one value")
    if any(value <= 0.0 for value in config.spacing_grid):
        raise ValueError("fwm.spacing_grid values must be positive")
    if config.mc_backend not in {"direct", "finite_mc", "dar_mc", "mc"}:
        raise ValueError("fwm.mc_backend must be 'direct', 'finite_mc', or 'dar_mc'")
    if config.mc_samples <= 0:
        raise ValueError("fwm.mc_samples must be positive")


def _beta2_from_system(system: System) -> float:
    if system.numerics is not None and hasattr(system.numerics, "gvd"):
        return float(system.numerics.gvd)
    fiber_section = (
        system.raw_config.get("fiber", {}) if isinstance(system.raw_config, Mapping) else {}
    )
    if isinstance(fiber_section, Mapping) and "beta2" in fiber_section:
        return float(fiber_section["beta2"])
    return -21.0e-27


def _pulse_shape_name(system: System) -> str:
    if system.pulse_config is not None:
        return system.pulse_config.type.value
    return system.pulse.__class__.__name__


def _elapsed(start: float) -> str:
    seconds = time.perf_counter() - start
    if seconds < 60.0:
        return f"{seconds:.1f}s"
    minutes, rem = divmod(seconds, 60.0)
    return f"{int(minutes)}m{rem:04.1f}s"


def _progress_step(total: int) -> int:
    if total < 50:
        return total
    return max(1, total // 10)


def _diagnostic_pulse(system: System, config: FWMDiagnosticConfig) -> Pulse:
    if system.pulse_config is None:
        return system.pulse
    num_symbols = system.pulse_config.num_symbols
    samples_per_symbol = system.pulse_config.samples_per_symbol
    return pulse_from_config(
        system.pulse_config,
        num_symbols=config.num_symbols if num_symbols is None else num_symbols,
        samples_per_symbol=(
            config.samples_per_symbol if samples_per_symbol is None else samples_per_symbol
        ),
    )


def _channel_params(
    target: str,
    term: str,
    spacing_over_baud: float,
    baud_rate: float,
    beta2: float,
) -> FWMChannels:
    delta_omega = 2.0 * np.pi * spacing_over_baud * baud_rate
    omegas = {"a": 0.0, "b": delta_omega}

    def beta0(label: str) -> float:
        omega = omegas[label]
        return 0.5 * beta2 * omega * omega

    def beta1(label: str) -> float:
        return beta2 * omegas[label]

    a, b, c = term
    return FWMChannels(
        omega_a=omegas[a],
        omega_b=omegas[b],
        omega_c=omegas[c],
        omega_d=omegas[target],
        beta0_a=beta0(a),
        beta0_b=beta0(b),
        beta0_c=beta0(c),
        beta0_d=beta0(target),
        beta1_a=beta1(a),
        beta1_b=beta1(b),
        beta1_c=beta1(c),
        beta1_d=beta1(target),
        gvd_a=beta2,
        gvd_b=beta2,
        gvd_c=beta2,
        gvd_d=beta2,
    )


def _same_channel_lock_masks(
    target: str,
    term: str,
    h_values: np.ndarray,
    k_values: np.ndarray,
    m_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = [target, term[0], term[1], term[2]]
    H = h_values[:, None, None]
    K = k_values[None, :, None]
    M = m_values[None, None, :]
    indices = [np.zeros((1, 1, 1), dtype=int), H, K, M]
    shape = (h_values.size, k_values.size, m_values.size)

    locked = np.ones(shape, dtype=bool)
    any_pair = np.zeros(shape, dtype=bool)
    for label in sorted(set(labels)):
        positions = [idx for idx, value in enumerate(labels) if value == label]
        if len(positions) < 2:
            continue
        first = np.broadcast_to(indices[positions[0]], shape)
        group_locked = np.ones(shape, dtype=bool)
        for pos_i, pos in enumerate(positions[1:], start=1):
            current = np.broadcast_to(indices[pos], shape)
            group_locked &= current == first
            for prev in positions[:pos_i]:
                any_pair |= current == np.broadcast_to(indices[prev], shape)
        locked &= group_locked

    partial = any_pair & ~locked
    unlocked = ~(locked | partial)
    return locked, partial, unlocked


def _compute_result(
    pulse: Pulse,
    z: np.ndarray,
    h_values: np.ndarray,
    k_values: np.ndarray,
    m_values: np.ndarray,
    *,
    target: str,
    term: str,
    pulse_shape: str,
    hkm_radius: int,
    spacing_over_baud: float,
    beta2: float,
    auto_refine_z: bool = True,
    min_pts_per_period: float = 3.0,
    max_z_points: int = 500,
    mc_backend: str = "direct",
    mc_samples: int = 1000,
    mc_seed: int = 1234,
) -> FWMTermResult:
    channels = _channel_params(
        target, term, spacing_over_baud, pulse.baud_rate, beta2
    )
    if mc_backend == "dar_mc":
        seed = int(mc_seed + round(1_000_000 * spacing_over_baud) + ord(target) + sum(ord(ch) for ch in term))
        mc = estimate_fwm_term_sum_dar_mc(
            channels=channels,
            baud_rate=float(pulse.baud_rate),
            length=float(z[-1] - z[0]),
            n_samples=mc_samples,
            alpha=0.0,
            seed=seed,
        )
        total = float(mc.total)
        locked = partial = unlocked = float("nan")
    elif mc_backend in {"mc", "finite_mc"}:
        seed = int(mc_seed + round(1_000_000 * spacing_over_baud) + ord(target) + sum(ord(ch) for ch in term))
        mc = estimate_fwm_term_sum_mc(
            pulse,
            z,
            h_values,
            k_values,
            m_values,
            channels=channels,
            target=target,
            term=term,
            n_samples=mc_samples,
            seed=seed,
            auto_refine=auto_refine_z,
            min_pts_per_period=min_pts_per_period,
            max_z_points=max_z_points,
            discretization_action="warn",
        )
        total = float(mc.total)
        locked = float(mc.locked)
        partial = float(mc.partial)
        unlocked = float(mc.unlocked)
    else:
        kernel = compute_fwm_kernel_direct(
            pulse,
            z,
            h_values,
            k_values,
            m_values,
            channels=channels,
            auto_refine=auto_refine_z,
            min_pts_per_period=min_pts_per_period,
            max_z_points=max_z_points,
            discretization_action="warn",
        )
        abs2 = np.abs(kernel.X) ** 2
        locked_mask, partial_mask, unlocked_mask = _same_channel_lock_masks(
            target, term, h_values, k_values, m_values
        )
        total = float(np.sum(abs2))
        locked = float(np.sum(abs2[locked_mask]))
        partial = float(np.sum(abs2[partial_mask]))
        unlocked = float(np.sum(abs2[unlocked_mask]))
    return FWMTermResult(
        target=target,
        term=term,
        pulse_shape=pulse_shape,
        hkm_radius=int(hkm_radius),
        spacing_over_baud=float(spacing_over_baud),
        delta_omega_over_2pi_baud=float(
            channels.delta_omega / (2.0 * np.pi * pulse.baud_rate)
        ),
        delta_beta0=float(channels.delta_beta0),
        total=total,
        locked=locked,
        partial=partial,
        unlocked=unlocked,
    )


def _compute_result_from_job(job: tuple) -> FWMTermResult:
    return _compute_result(*job[:4], **job[4])


def compute_demo_dataset(
    *,
    system: System,
    fwm_config: FWMDiagnosticConfig,
) -> list[FWMTermResult]:
    pulse = _diagnostic_pulse(system, fwm_config)
    pulse_shape = _pulse_shape_name(system)
    beta2 = _beta2_from_system(system)
    length = float(system.fiber_length if system.fiber_length is not None else 1500.0)
    spacing_grid = np.asarray(fwm_config.spacing_grid, dtype=float)
    z = np.linspace(0.0, length, fwm_config.z_points)
    radii = [int(fwm_config.hkm_radius)]
    if fwm_config.compare_next_radius:
        radii.append(int(fwm_config.hkm_radius) + 1)
    terms = ["".join(labels) for labels in product("ab", repeat=3)]
    start = time.perf_counter()

    lg.info(
        f"FWM setup: pulse={pulse_shape}, R={radii}, spacings={spacing_grid.size}, "
        f"z={fwm_config.z_points}, Nt={len(pulse.data()[0])}, beta2={beta2:.3e}"
    )

    jobs = []
    for radius in radii:
        h_values = np.arange(-radius, radius + 1)
        k_values = np.arange(-radius, radius + 1)
        m_values = np.arange(-radius, radius + 1)
        for spacing in spacing_grid:
            for target in ("a", "b"):
                for term in terms:
                    jobs.append(
                        (
                            pulse,
                            z,
                            h_values,
                            k_values,
                            {
                                "m_values": m_values,
                                "target": target,
                                "term": term,
                                "pulse_shape": pulse_shape,
                                "hkm_radius": radius,
                                "spacing_over_baud": float(spacing),
                                "beta2": beta2,
                                "auto_refine_z": fwm_config.auto_refine_z,
                                "min_pts_per_period": fwm_config.min_pts_per_period,
                                "max_z_points": fwm_config.max_z_points,
                                "mc_backend": fwm_config.mc_backend,
                                "mc_samples": fwm_config.mc_samples,
                                "mc_seed": fwm_config.mc_seed,
                            },
                        )
                    )

    lg.info(f"FWM jobs: {len(jobs)} coefficient tables")

    if fwm_config.workers == 1 or len(jobs) <= 1:
        results = []
        progress_step = _progress_step(len(jobs))
        for idx, job in enumerate(jobs, start=1):
            results.append(_compute_result_from_job(job))
            if idx == len(jobs) or idx % progress_step == 0:
                lg.info(f"FWM progress: {idx}/{len(jobs)} ({_elapsed(start)})")
        lg.success(f"FWM coefficients done in {_elapsed(start)}")
        return results

    max_workers = os.cpu_count() if fwm_config.workers <= 0 else fwm_config.workers
    lg.info(f"FWM multiprocessing: workers={max_workers}")
    results = []
    progress_step = _progress_step(len(jobs))
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=mp.get_context("fork")) as executor:
        futures = [executor.submit(_compute_result_from_job, job) for job in jobs]
        for idx, future in enumerate(as_completed(futures), start=1):
            results.append(future.result())
            if idx == len(futures) or idx % progress_step == 0:
                lg.info(f"FWM progress: {idx}/{len(futures)} ({_elapsed(start)})")
    lg.success(f"FWM coefficients done in {_elapsed(start)}")
    return results


def _save_dataset(results: list[FWMTermResult], path: Path) -> None:
    start = time.perf_counter()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        target=np.array([r.target for r in results]),
        term=np.array([r.term for r in results]),
        pulse_shape=np.array([r.pulse_shape for r in results]),
        hkm_radius=np.array([r.hkm_radius for r in results]),
        spacing_over_baud=np.array([r.spacing_over_baud for r in results]),
        delta_omega_over_2pi_baud=np.array([r.delta_omega_over_2pi_baud for r in results]),
        delta_beta0=np.array([r.delta_beta0 for r in results]),
        total=np.array([r.total for r in results]),
        locked=np.array([r.locked for r in results]),
        partial=np.array([r.partial for r in results]),
        unlocked=np.array([r.unlocked for r in results]),
    )
    lg.success(f"saved {path} ({_elapsed(start)})")


def _records(
    results: list[FWMTermResult],
    *,
    target: str | None = None,
    term: str | None = None,
    hkm_radius: int | None = None,
):
    for result in results:
        if target is not None and result.target != target:
            continue
        if term is not None and result.term != term:
            continue
        if hkm_radius is not None and result.hkm_radius != hkm_radius:
            continue
        yield result


def _term_values(
    results: list[FWMTermResult], target: str, term: str, hkm_radius: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = sorted(
        _records(results, target=target, term=term, hkm_radius=hkm_radius),
        key=lambda item: item.spacing_over_baud,
    )
    return (
        np.array([row.spacing_over_baud for row in rows]),
        np.array([row.total for row in rows]),
        np.array([row.delta_omega_over_2pi_baud for row in rows]),
    )


def plot_results(
    results: list[FWMTermResult], out_dir: Path, *, symmetry_spacing: float = 1.0
) -> list[Path]:
    start = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    terms = ["".join(labels) for labels in product("ab", repeat=3)]
    colors = plt.cm.tab10(np.linspace(0.0, 1.0, len(terms)))
    radii = sorted({result.hkm_radius for result in results})
    display_radius = radii[-1]

    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    for color, term in zip(colors, terms, strict=True):
        spacing, total, delta = _term_values(results, "a", term, display_radius)
        line_style = "-" if np.allclose(delta, 0.0) else "--"
        max_total = float(np.max(total)) if total.size else 0.0
        normalized_total = total / max_total if max_total > 0.0 else np.full_like(total, np.nan, dtype=float)
        ax.plot(
            spacing,
            normalized_total,
            marker="o",
            ms=3,
            lw=1.0,
            ls=line_style,
            color=color,
            label=term,
        )
    pulse_shape = results[0].pulse_shape
    ax.set_title(fr"{pulse_shape}, observed $d=a$, $R={display_radius}$")
    ax.set_xlabel(r"carrier spacing $\Delta f/B$")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.25)
    ax.set_ylabel(r"$\sum_{hkm}|X^{abc\to a}_{hkm}|^2$ normalized per term")
    ax.legend(ncol=2, fontsize=7, frameon=False)
    fig.tight_layout()
    path = out_dir / "wideband_fwm_spacing_sweep.pdf"
    fig.savefig(path, dpi=300)
    png_path = path.with_suffix(".png")
    fig.savefig(png_path, dpi=300)
    plt.close(fig)
    paths.extend([path, png_path])

    spacing_pick = symmetry_spacing
    x = np.arange(len(terms))
    width = 0.38
    fig, axes = plt.subplots(
        len(radii), 1, figsize=(7.0, 2.8 * len(radii)), sharex=True, sharey=True
    )
    axes = np.atleast_1d(axes)
    for ax, radius in zip(axes, radii, strict=True):
        rows_a = {
            row.term: row
            for row in _records(results, target="a", hkm_radius=radius)
            if np.isclose(row.spacing_over_baud, spacing_pick)
        }
        rows_b = {
            row.term: row
            for row in _records(results, target="b", hkm_radius=radius)
            if np.isclose(row.spacing_over_baud, spacing_pick)
        }
        if set(rows_a) != set(terms) or set(rows_b) != set(terms):
            raise ValueError(
                f"No complete dataset found at spacing/B={spacing_pick:g}; "
                "include it in --spacing-grid or choose --symmetry-spacing accordingly."
            )
        vals_a = np.array([rows_a[term].total for term in terms])
        vals_b_mirror = np.array(
            [rows_b[term.translate(str.maketrans("ab", "ba"))].total for term in terms]
        )
        ax.bar(
            x - width / 2.0,
            vals_a / np.max(vals_a),
            width,
            label=r"$d=a$, term $abc$",
        )
        ax.bar(
            x + width / 2.0,
            vals_b_mirror / np.max(vals_b_mirror),
            width,
            label=r"$d=b$, mirrored $a\leftrightarrow b$",
        )
        ax.set_yscale("log")
        ax.set_ylabel(r"normalized $\sum_{hkm}|X|^2$")
        ax.set_title(fr"two-carrier symmetry at $\Delta f/B={spacing_pick:g}$, $R={radius}$")
        ax.grid(True, axis="y", which="both", alpha=0.25)
        ax.legend(fontsize=8, frameon=False)
    axes[-1].set_xticks(x, terms)
    fig.tight_layout()
    path = out_dir / "wideband_fwm_ab_symmetry.pdf"
    fig.savefig(path, dpi=300)
    png_path = path.with_suffix(".png")
    fig.savefig(png_path, dpi=300)
    plt.close(fig)
    paths.extend([path, png_path])

    representative = [("a", "abb"), ("a", "aab"), ("a", "bbb"), ("b", "baa")]
    if any(not np.isfinite(row.locked + row.partial + row.unlocked) for row in results):
        lg.warning("skipping FWM sector decomposition plot: backend does not provide sectors")
        lg.success(f"plots done: {len(paths)} files ({_elapsed(start)})")
        return paths
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0), sharex=True, sharey=True)
    for ax, (target, term) in zip(axes.flat, representative, strict=True):
        rows = sorted(
            _records(results, target=target, term=term, hkm_radius=display_radius),
            key=lambda item: item.spacing_over_baud,
        )
        spacing = np.array([row.spacing_over_baud for row in rows])
        locked = np.array([row.locked for row in rows])
        partial = np.array([row.partial for row in rows])
        unlocked = np.array([row.unlocked for row in rows])
        total = locked + partial + unlocked
        ax.stackplot(
            spacing,
            locked / total,
            partial / total,
            unlocked / total,
            labels=["same-channel locked", "partial lock", "unlocked"],
            alpha=0.85,
        )
        ax.set_title(fr"$X^{{{term}\to {target}}}$, $R={display_radius}$")
        ax.grid(True, alpha=0.25)
    for ax in axes[-1, :]:
        ax.set_xlabel(r"carrier spacing $\Delta f/B$")
    for ax in axes[:, 0]:
        ax.set_ylabel("fraction of summed power")
    axes[0, 1].legend(
        fontsize=7, frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5)
    )
    fig.tight_layout()
    path = out_dir / "wideband_fwm_hkm_sector_decomposition.pdf"
    fig.savefig(path, dpi=300)
    png_path = path.with_suffix(".png")
    fig.savefig(png_path, dpi=300)
    plt.close(fig)
    paths.extend([path, png_path])
    lg.success(f"plots done: {len(paths)} files ({_elapsed(start)})")
    return paths


def main() -> None:
    init_logging()
    start = time.perf_counter()
    parser = argparse.ArgumentParser(description="Plot small direct wideband FWM hkm-sum diagnostics.")
    parser.add_argument("--config", type=Path, default=Path("input/studies.toml"))
    parser.add_argument("--out-dir", type=Path, default=Path("media/fwm"))
    args = parser.parse_args()

    lg.info(f"loading config: {args.config}")
    system = System.from_toml(args.config)
    fwm_config = _fwm_config_from_system(system)
    lg.info(f"FWM config: {fwm_config}")
    results = compute_demo_dataset(system=system, fwm_config=fwm_config)
    data_path = args.out_dir / "wideband_fwm_terms_demo.npz"
    _save_dataset(results, data_path)
    paths = plot_results(results, args.out_dir, symmetry_spacing=fwm_config.symmetry_spacing)
    for path in paths:
        lg.info(f"saved {path}")
    lg.success(f"FWM workflow complete in {_elapsed(start)}")


if __name__ == "__main__":
    main()
