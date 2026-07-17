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

import pynlin  # noqa: F401
from loguru import logger as lg
from analysis.log_init import init_logging
from pynlin.methods.td.xhkm_sums import compute_xhkm_sums
from pynlin.methods.td.xpm_kernel import compute_xpm_kernel_fft
from pynlin.pulses import pulse_from_config
from pynlin.system import System


@dataclass(frozen=True)
class XPMSectorConfig:
    spacing_grid: tuple[float, ...] = (0.75, 1.0, 1.25, 1.5, 2.0)
    h_radius: int = 5
    r_radius: int = 5
    m_radius: int = 9
    z_points: int = 33
    num_symbols: int = 128
    samples_per_symbol: int = 16
    auto_refine_z: bool = True
    min_pts_per_collision: float = 3.0
    max_z_points: int = 2000
    h_radius_cap: int = 10
    m_radius_cap: int = 21
    use_window_caps: bool = True
    max_entries: int = 100_000
    beta2_override_si: float | None = None


def _config_from_system(system: System, spacing_source: str) -> XPMSectorConfig:
    raw = system.raw_config if isinstance(system.raw_config, Mapping) else {}
    mc_section = raw.get("mc_validation") if isinstance(raw, Mapping) else None
    fwm_section = raw.get("fwm") if isinstance(raw, Mapping) else None
    num_section = raw.get("numerics") if isinstance(raw, Mapping) else None

    defaults = XPMSectorConfig()
    section = mc_section if isinstance(mc_section, Mapping) else {}
    if spacing_source == "fwm" and isinstance(fwm_section, Mapping):
        spacing_grid = fwm_section.get("spacing_grid", defaults.spacing_grid)
    else:
        spacing_grid = section.get("spacing_grid", defaults.spacing_grid)

    return XPMSectorConfig(
        spacing_grid=tuple(float(v) for v in spacing_grid),
        h_radius=int(section.get("h_radius", defaults.h_radius)),
        r_radius=int(section.get("r_radius", defaults.r_radius)),
        m_radius=int(section.get("m_radius", defaults.m_radius)),
        z_points=int(section.get("z_points", defaults.z_points)),
        num_symbols=int(section.get("num_symbols", defaults.num_symbols)),
        samples_per_symbol=int(section.get("samples_per_symbol", defaults.samples_per_symbol)),
        auto_refine_z=bool(
            (num_section or {}).get("auto_refine_z", defaults.auto_refine_z)
            if isinstance(num_section, Mapping)
            else defaults.auto_refine_z
        ),
        min_pts_per_collision=float(
            (num_section or {}).get("min_pts_per_collision", defaults.min_pts_per_collision)
            if isinstance(num_section, Mapping)
            else defaults.min_pts_per_collision
        ),
        max_z_points=int(
            (num_section or {}).get("max_z_points", defaults.max_z_points)
            if isinstance(num_section, Mapping)
            else defaults.max_z_points
        ),
    )


def _diagnostic_pulse(system: System, config: XPMSectorConfig):
    if system.pulse_config is None:
        return system.pulse
    return pulse_from_config(
        system.pulse_config,
        num_symbols=config.num_symbols,
        samples_per_symbol=config.samples_per_symbol,
    )


def compute_xpm_sectors(system: System, config: XPMSectorConfig) -> dict[str, np.ndarray]:
    pulse = _diagnostic_pulse(system, config)
    length = float(system.fiber_length)
    baud_rate = float(pulse.baud_rate)
    beta2_si = (
        float(config.beta2_override_si)
        if config.beta2_override_si is not None
        else float(system.numerics.gvd if system.numerics is not None else system.raw_config["fiber"]["beta2"])
    )

    rows = []
    for spacing in config.spacing_grid:
        dgd = abs(beta2_si) * 2.0 * np.pi * float(spacing) * baud_rate
        llw = length * baud_rate * dgd

        # Match the MC validation convention: scale the h/r/m truncation with LLW.
        llw_ref = 5.66
        scale = max(1.0, llw / llw_ref)
        h_target = int(np.ceil(config.h_radius * scale))
        r_target = int(np.ceil(config.r_radius * scale))
        m_target = int(np.ceil(config.m_radius * scale))
        if config.use_window_caps:
            h_r = max(config.h_radius, min(config.h_radius_cap, h_target))
            r_r = max(config.r_radius, min(config.h_radius_cap, r_target))
            m_r = max(config.m_radius, min(config.m_radius_cap, m_target))
        else:
            h_r = max(config.h_radius, h_target)
            r_r = max(config.r_radius, r_target)
            m_r = max(config.m_radius, m_target)

        n_entries = (2 * h_r + 1) * (2 * r_r + 1) * (2 * m_r + 1)
        if n_entries > config.max_entries:
            raise RuntimeError(
                "adaptive XPM window would be too large: "
                f"q={spacing:g}, LLW={llw:.2f}, h_r={h_r}, r_r={r_r}, m_r={m_r}, "
                f"entries={n_entries:,} > max_entries={config.max_entries:,}. "
                "Increase --max-entries or use caps."
            )

        h_values = np.arange(-h_r, h_r + 1)
        r_values = np.arange(-r_r, r_r + 1)
        m_values = np.arange(-m_r, m_r + 1)
        z = np.linspace(0.0, length, config.z_points)
        lg.info(
            f"XPM sectors q={spacing:g}: LLW={llw:.2f}, "
            f"h_r={h_r}, r_r={r_r}, m_r={m_r}, entries={n_entries}"
        )
        table = compute_xpm_kernel_fft(
            pulse,
            z,
            h_values,
            r_values,
            m_values,
            dgd=dgd,
            gvda=beta2_si,
            gvdb=beta2_si,
            auto_refine=config.auto_refine_z,
            min_pts_per_collision=config.min_pts_per_collision,
            max_z_points=config.max_z_points,
            discretization_action="warn",
        )
        sums = compute_xhkm_sums(table.X, table.h_values, table.r_values, table.m_values)
        rows.append(
            (
                float(spacing),
                float(llw),
                float(sums.n1),
                float(sums.n_2pc),
                float(sums.n_3pca),
                float(sums.n_3pcb),
                float(sums.n_4pc),
                h_r,
                r_r,
                m_r,
                int(table.metadata["n_z"]),
            )
        )

    data = np.asarray(rows, dtype=float)
    return {
        "spacing_over_baud": data[:, 0],
        "llw": data[:, 1],
        "n1": data[:, 2],
        "n_2pc": data[:, 3],
        "n_3pca": data[:, 4],
        "n_3pcb": data[:, 5],
        "n_4pc": data[:, 6],
        "h_radius": data[:, 7],
        "r_radius": data[:, 8],
        "m_radius": data[:, 9],
        "nz_actual": data[:, 10],
        "beta2_si": np.array([beta2_si], dtype=float),
        "lld": np.array([length * abs(beta2_si) * baud_rate**2], dtype=float),
    }


def plot_xpm_sectors(data: dict[str, np.ndarray], out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    spacing = data["spacing_over_baud"]
    total = data["n1"]
    sectors = [
        data["n_2pc"] / total,
        data["n_3pca"] / total,
        data["n_3pcb"] / total,
        data["n_4pc"] / total,
    ]
    labels = [
        r"2PC: $h=0,\ k=m$",
        r"3PCa: $h=0,\ k\ne m$",
        r"3PCb: $h\ne0,\ k=m$",
        r"4PC: unlocked",
    ]

    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    ax.stackplot(spacing, *sectors, labels=labels, alpha=0.85)
    ax.set_xlabel(r"carrier spacing $\Delta f/B$")
    ax.set_ylabel(r"fraction of $N_1=\sum_{hkm}|X_{hkm}|^2$")
    ax.set_title("XPM/Xhkm sector decomposition")
    beta2_si = float(np.ravel(data.get("beta2_si", np.array([np.nan])))[0])
    if np.isfinite(beta2_si):
        beta2_ps2_per_km = beta2_si / 1e-27
        lld = float(np.ravel(data.get("lld", np.array([np.nan])))[0])
        ax.text(
            0.02,
            0.04,
            rf"$\beta_2={beta2_ps2_per_km:.1f}\,$ps$^2$/km, $L/L_D={lld:.1f}$",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=7,
            color="0.25",
            bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none", "pad": 2.0},
        )
    ax.grid(True, alpha=0.25)
    ax.set_ylim(0.0, 1.0)
    ax.legend(fontsize=7, frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5))
    for q, hr, mr, nz in zip(spacing, data["h_radius"], data["m_radius"], data["nz_actual"], strict=True):
        ax.annotate(
            f"h={int(hr)},m={int(mr)},nz={int(nz)}",
            (q, 1.0),
            textcoords="offset points",
            xytext=(0, 4),
            ha="center",
            fontsize=6,
            rotation=35,
        )
    fig.tight_layout()
    pdf = out_dir / "xpm_hkm_sector_decomposition.pdf"
    fig.savefig(pdf, dpi=300)
    plt.close(fig)
    return [pdf]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot XPM/Xhkm 2PC/3PC/4PC sector decomposition.")
    parser.add_argument("--config", type=Path, default=Path("input/studies.toml"))
    parser.add_argument("--out-dir", type=Path, default=Path("media/fwm"))
    parser.add_argument("--spacing-source", choices=("mc_validation", "fwm"), default="mc_validation")
    parser.add_argument(
        "--spacing-grid",
        type=float,
        nargs="+",
        help="Override the TOML spacing grid, e.g. --spacing-grid 0.75 1 2 5 10",
    )
    parser.add_argument("--h-radius-cap", type=int, default=None)
    parser.add_argument("--m-radius-cap", type=int, default=None)
    parser.add_argument(
        "--no-window-cap",
        action="store_true",
        help="Use fully LLW-scaled h/r/m windows instead of cap-limited windows.",
    )
    parser.add_argument("--max-entries", type=int, default=None)
    parser.add_argument("--num-symbols", type=int, default=None)
    parser.add_argument(
        "--beta2-ps2-per-km",
        type=float,
        default=None,
        help="Override beta2 in ps^2/km, e.g. --beta2-ps2-per-km -5",
    )
    parser.add_argument(
        "--lld",
        type=float,
        default=None,
        help="Override beta2 by target L/L_D = L*|beta2|*B^2. Uses negative beta2 sign.",
    )
    return parser.parse_args()


def main() -> None:
    init_logging()
    args = _parse_args()
    system = System.from_toml(args.config)
    config = _config_from_system(system, args.spacing_source)
    if (
        args.spacing_grid is not None
        or args.h_radius_cap is not None
        or args.m_radius_cap is not None
        or args.no_window_cap
        or args.max_entries is not None
        or args.beta2_ps2_per_km is not None
        or args.lld is not None
        or args.num_symbols is not None
    ):
        beta2_override_si = config.beta2_override_si
        if args.beta2_ps2_per_km is not None:
            beta2_override_si = args.beta2_ps2_per_km * 1e-27
        if args.lld is not None:
            length = float(system.fiber_length)
            baud_rate = float(system.pulse.baud_rate)
            beta2_override_si = -abs(args.lld) / (length * baud_rate**2)
        config = XPMSectorConfig(
            spacing_grid=tuple(args.spacing_grid) if args.spacing_grid is not None else config.spacing_grid,
            h_radius=config.h_radius,
            r_radius=config.r_radius,
            m_radius=config.m_radius,
            z_points=config.z_points,
            num_symbols=args.num_symbols if args.num_symbols is not None else config.num_symbols,
            samples_per_symbol=config.samples_per_symbol,
            auto_refine_z=config.auto_refine_z,
            min_pts_per_collision=config.min_pts_per_collision,
            max_z_points=config.max_z_points,
            h_radius_cap=args.h_radius_cap if args.h_radius_cap is not None else config.h_radius_cap,
            m_radius_cap=args.m_radius_cap if args.m_radius_cap is not None else config.m_radius_cap,
            use_window_caps=not args.no_window_cap,
            max_entries=args.max_entries if args.max_entries is not None else config.max_entries,
            beta2_override_si=beta2_override_si,
        )
    lg.info(f"XPM sector config: {config}")
    data = compute_xpm_sectors(system, config)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    npz = args.out_dir / "xpm_hkm_sector_decomposition.npz"
    np.savez(npz, **data)
    lg.success(f"saved {npz}")
    for path in plot_xpm_sectors(data, args.out_dir):
        lg.success(f"saved {path}")


if __name__ == "__main__":
    main()
