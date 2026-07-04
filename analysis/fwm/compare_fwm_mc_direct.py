from __future__ import annotations

import argparse
import sys
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

import pynlin  # noqa: F401
from loguru import logger as lg

from analysis.log_init import init_logging
from pynlin.methods.td.fwm_kernel import FWMChannels, compute_fwm_kernel_direct
from pynlin.methods.td.fwm_mc import estimate_fwm_term_sum_dar_mc, estimate_fwm_term_sum_mc
from pynlin.pulses import pulse_from_config
from pynlin.system import System


@dataclass(frozen=True)
class FWMCompareConfig:
    hkm_radius: int = 3
    z_points: int = 7
    num_symbols: int = 64
    samples_per_symbol: int = 24
    spacing_grid: tuple[float, ...] = (0.75, 1.0, 2.0)
    terms: tuple[str, ...] = ("abb", "aab", "bbb", "baa")
    targets: tuple[str, ...] = ("a", "b")
    mc_samples: int = 1000
    mc_seed: int = 1234
    auto_refine_z: bool = True
    min_pts_per_period: float = 3.0
    max_z_points: int = 500
    mc_kind: str = "dar"


def _config_from_system(system: System) -> FWMCompareConfig:
    raw = system.raw_config if isinstance(system.raw_config, Mapping) else {}
    section = raw.get("fwm") if isinstance(raw, Mapping) else None
    defaults = FWMCompareConfig()
    if not isinstance(section, Mapping):
        return defaults
    return FWMCompareConfig(
        hkm_radius=int(section.get("hkm_radius", defaults.hkm_radius)),
        z_points=int(section.get("z_points", defaults.z_points)),
        num_symbols=int(section.get("num_symbols", defaults.num_symbols)),
        samples_per_symbol=int(section.get("samples_per_symbol", defaults.samples_per_symbol)),
        spacing_grid=tuple(float(v) for v in section.get("spacing_grid", defaults.spacing_grid)),
        mc_samples=int(section.get("mc_samples", defaults.mc_samples)),
        mc_seed=int(section.get("mc_seed", defaults.mc_seed)),
        auto_refine_z=bool(section.get("auto_refine_z", defaults.auto_refine_z)),
        min_pts_per_period=float(
            section.get("min_pts_per_period", section.get("min_pts_per_collision", defaults.min_pts_per_period))
        ),
        max_z_points=int(section.get("max_z_points", defaults.max_z_points)),
        mc_kind=str(section.get("mc_kind", defaults.mc_kind)),
    )


def _channel_params(target: str, term: str, spacing_over_baud: float, baud_rate: float, beta2: float) -> FWMChannels:
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


def _same_channel_lock_masks(target: str, term: str, h_values: np.ndarray, k_values: np.ndarray, m_values: np.ndarray):
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


def _pulse(system: System, config: FWMCompareConfig):
    return pulse_from_config(
        system.pulse_config,
        num_symbols=config.num_symbols,
        samples_per_symbol=config.samples_per_symbol,
    )


def _beta2(system: System) -> float:
    if system.numerics is not None and hasattr(system.numerics, "gvd"):
        return float(system.numerics.gvd)
    return float(system.raw_config["fiber"]["beta2"])


def run_comparison(system: System, config: FWMCompareConfig) -> dict[str, np.ndarray]:
    pulse = _pulse(system, config)
    beta2 = _beta2(system)
    length = float(system.fiber_length)
    norm = (1.0 / float(pulse.baud_rate)) ** 2
    z = np.linspace(0.0, length, config.z_points)
    values = np.arange(-config.hkm_radius, config.hkm_radius + 1)

    rows = []
    for spacing in config.spacing_grid:
        for target in config.targets:
            for term in config.terms:
                if any(ch not in "ab" for ch in target + term):
                    raise ValueError("this diagnostic currently supports only labels 'a' and 'b'")
                channels = _channel_params(target, term, float(spacing), float(pulse.baud_rate), beta2)
                lg.info(f"FWM direct/MC: q={spacing:g}, {term}->{target}, R={config.hkm_radius}")
                direct = compute_fwm_kernel_direct(
                    pulse,
                    z,
                    values,
                    values,
                    values,
                    channels=channels,
                    auto_refine=config.auto_refine_z,
                    min_pts_per_period=config.min_pts_per_period,
                    max_z_points=config.max_z_points,
                    discretization_action="warn",
                )
                abs2 = np.abs(direct.X) ** 2
                locked_mask, partial_mask, unlocked_mask = _same_channel_lock_masks(
                    target, term, values, values, values
                )
                direct_total = float(np.sum(abs2))
                direct_locked = float(np.sum(abs2[locked_mask]))
                direct_partial = float(np.sum(abs2[partial_mask]))
                direct_unlocked = float(np.sum(abs2[unlocked_mask]))

                if config.mc_kind == "dar":
                    mc = estimate_fwm_term_sum_dar_mc(
                        channels=channels,
                        baud_rate=float(pulse.baud_rate),
                        length=length,
                        n_samples=config.mc_samples,
                        alpha=0.0,
                        seed=config.mc_seed,
                    )
                    mc_total = mc.total
                    mc_total_stderr = mc.total_stderr
                    mc_locked = mc_locked_stderr = float("nan")
                    mc_partial = mc_partial_stderr = float("nan")
                    mc_unlocked = mc_unlocked_stderr = float("nan")
                    direct_total_cmp = direct_total * norm
                    direct_locked_cmp = direct_locked * norm
                    direct_partial_cmp = direct_partial * norm
                    direct_unlocked_cmp = direct_unlocked * norm
                else:
                    mc = estimate_fwm_term_sum_mc(
                        pulse,
                        z,
                        values,
                        values,
                        values,
                        channels=channels,
                        target=target,
                        term=term,
                        n_samples=config.mc_samples,
                        seed=config.mc_seed,
                        auto_refine=config.auto_refine_z,
                        min_pts_per_period=config.min_pts_per_period,
                        max_z_points=config.max_z_points,
                        discretization_action="warn",
                    )
                    mc_total = mc.total
                    mc_total_stderr = mc.total_stderr
                    mc_locked = mc.locked
                    mc_locked_stderr = mc.locked_stderr
                    mc_partial = mc.partial
                    mc_partial_stderr = mc.partial_stderr
                    mc_unlocked = mc.unlocked
                    mc_unlocked_stderr = mc.unlocked_stderr
                    direct_total_cmp = direct_total
                    direct_locked_cmp = direct_locked
                    direct_partial_cmp = direct_partial
                    direct_unlocked_cmp = direct_unlocked
                rows.append(
                    (
                        float(spacing),
                        float(ord(target)),
                        float(sum(ord(ch) << (8 * idx) for idx, ch in enumerate(term))),
                        direct_total_cmp,
                        mc_total,
                        mc_total_stderr,
                        direct_locked_cmp,
                        mc_locked,
                        mc_locked_stderr,
                        direct_partial_cmp,
                        mc_partial,
                        mc_partial_stderr,
                        direct_unlocked_cmp,
                        mc_unlocked,
                        mc_unlocked_stderr,
                    )
                )
    data = np.asarray(rows, dtype=float)
    return {
        "spacing_over_baud": data[:, 0],
        "target_code": data[:, 1],
        "term_code": data[:, 2],
        "direct_total": data[:, 3],
        "mc_total": data[:, 4],
        "mc_total_stderr": data[:, 5],
        "direct_locked": data[:, 6],
        "mc_locked": data[:, 7],
        "mc_locked_stderr": data[:, 8],
        "direct_partial": data[:, 9],
        "mc_partial": data[:, 10],
        "mc_partial_stderr": data[:, 11],
        "direct_unlocked": data[:, 12],
        "mc_unlocked": data[:, 13],
        "mc_unlocked_stderr": data[:, 14],
        "hkm_radius": np.array([config.hkm_radius]),
        "mc_samples": np.array([config.mc_samples]),
        "mc_kind": np.array([config.mc_kind]),
    }


def plot_comparison(data: dict[str, np.ndarray], out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    ratio = data["direct_total"] / data["mc_total"]
    err = data["direct_total"] * data["mc_total_stderr"] / (data["mc_total"] ** 2)
    x = np.arange(ratio.size)
    labels = [
        f"q={q:g}\n{''.join(chr((int(code) >> (8*i)) & 255) for i in range(3))}->{chr(int(tgt))}"
        for q, tgt, code in zip(data["spacing_over_baud"], data["target_code"], data["term_code"], strict=True)
    ]
    fig, ax = plt.subplots(figsize=(max(6.0, 0.45 * ratio.size), 3.2))
    ax.errorbar(x, ratio, yerr=err, marker="o", lw=1.0, ls="none")
    ax.axhline(1.0, color="0.35", lw=0.8, ls=":")
    ax.set_ylabel("direct / MC total")
    ax.set_xticks(x, labels, rotation=75, ha="right", fontsize=6)
    ax.grid(True, axis="y", alpha=0.25)
    ax.set_title(
        f"FWM {str(data['mc_kind'][0])} MC check, R={int(data['hkm_radius'][0])}, "
        f"samples={int(data['mc_samples'][0])}"
    )
    fig.tight_layout()
    pdf = out_dir / "fwm_mc_vs_direct_ratio.pdf"
    png = pdf.with_suffix(".png")
    fig.savefig(pdf, dpi=300)
    fig.savefig(png, dpi=300)
    plt.close(fig)
    return [pdf, png]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare direct finite-window FWM sums to MC sampled sums.")
    parser.add_argument("--config", type=Path, default=Path("input/studies.toml"))
    parser.add_argument("--out-dir", type=Path, default=Path("media/fwm"))
    parser.add_argument("--spacing-grid", type=float, nargs="+")
    parser.add_argument("--terms", nargs="+", default=None)
    parser.add_argument("--targets", nargs="+", default=None)
    parser.add_argument("--radius", type=int, default=None)
    parser.add_argument("--mc-samples", type=int, default=None)
    parser.add_argument("--mc-kind", choices=("dar", "finite"), default=None)
    return parser.parse_args()


def main() -> None:
    init_logging()
    args = _parse_args()
    system = System.from_toml(args.config)
    config = _config_from_system(system)
    if any(value is not None for value in (args.spacing_grid, args.terms, args.targets, args.radius, args.mc_samples, args.mc_kind)):
        config = FWMCompareConfig(
            hkm_radius=args.radius if args.radius is not None else config.hkm_radius,
            z_points=config.z_points,
            num_symbols=config.num_symbols,
            samples_per_symbol=config.samples_per_symbol,
            spacing_grid=tuple(args.spacing_grid) if args.spacing_grid is not None else config.spacing_grid,
            terms=tuple(args.terms) if args.terms is not None else config.terms,
            targets=tuple(args.targets) if args.targets is not None else config.targets,
            mc_samples=args.mc_samples if args.mc_samples is not None else config.mc_samples,
            mc_seed=config.mc_seed,
            auto_refine_z=config.auto_refine_z,
            min_pts_per_period=config.min_pts_per_period,
            max_z_points=config.max_z_points,
            mc_kind=args.mc_kind if args.mc_kind is not None else config.mc_kind,
        )
    lg.info(f"FWM MC compare config: {config}")
    data = run_comparison(system, config)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    npz = args.out_dir / "fwm_mc_vs_direct_terms.npz"
    np.savez(npz, **data)
    lg.success(f"saved {npz}")
    for path in plot_comparison(data, args.out_dir):
        lg.success(f"saved {path}")


if __name__ == "__main__":
    main()
