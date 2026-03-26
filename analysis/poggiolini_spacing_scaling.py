import argparse
from pathlib import Path

import matplotlib
import numpy as np
from loguru import logger as lg

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pynlin.constellation_stats import gaussian_mu0
from pynlin.nlin.nlin_estimator_uwb import collision_coeffs_system_uwb, total_nlin_uwb
from pynlin.nlin.pcfm_gn import PcfmConfig
from pynlin.system import System
from pynlin.wdm import IrregularWDM, RegularWDM

from analysis.poggiolini.io import _resolve_launch_powers, _write_flat_profile
from analysis.poggiolini.models import _load_or_compute_pcfm
from analysis.poggiolini.td import _td_modulation_components
from analysis.poggiolini.workflow import MANAKOV_SCALE_POGGIOLINI
from analysis.uwb_nlin import _nlin_cache_path

plt.rcParams["text.usetex"] = False


def _parse_float_list(values: str) -> list[float]:
    return [float(x.strip()) for x in values.split(",") if x.strip()]


def _safe_tag(value: float) -> str:
    text = f"{value:.6e}"
    return text.replace("+", "").replace("-", "m").replace(".", "p")


def _center_channel(system: System) -> tuple[int, str]:
    freqs = system.wdm.frequency_grid()
    center_idx = int(freqs.size // 2)
    band_label = "overall"
    if hasattr(system.wdm, "_band_slices"):
        for name, slc in system.wdm._band_slices.items():
            if slc.start <= center_idx < slc.stop:
                band_label = str(name)
                break
    return center_idx, band_label


def _fit_exponent(x_values: np.ndarray, values_w: np.ndarray) -> float:
    mask = np.isfinite(x_values) & np.isfinite(values_w) & (x_values > 0.0) & (values_w > 0.0)
    if int(np.count_nonzero(mask)) < 2:
        return float("nan")
    coeffs = np.polyfit(np.log(x_values[mask]), np.log(values_w[mask]), 1)
    return float(coeffs[0])


def _rebuild_wdm_with_spacing(system: System, spacing_hz: float) -> None:
    spacing_hz = float(spacing_hz)
    wdm = system.wdm
    if isinstance(wdm, RegularWDM):
        system.wdm = RegularWDM(
            spacing=spacing_hz,
            num_channels=wdm.num_channels,
            center_frequency=wdm.central_frequency,
        )
        return
    if isinstance(wdm, IrregularWDM):
        band_specs = {
            name: {
                "n_channels": spec.n_channels,
                "launch_power_dbm": spec.launch_power_dbm,
                "start_nm": spec.start_nm,
                "modulation": spec.modulation,
            }
            for name, spec in wdm.band_specs.items()
        }
        system.wdm = IrregularWDM.from_bands_mapping(
            band_specs,
            wdm_data={"spacing": spacing_hz},
            root_data=None,
        )
        return
    raise TypeError(f"Unsupported WDM type for spacing sweep: {type(wdm)!r}")


def _scaling_series(rows: list[dict]) -> dict[str, np.ndarray]:
    return {
        "TD 64-QAM": np.array([row["td_channel_w"] for row in rows], dtype=float),
        "TD Gaussian": np.array([row["td_gaussian_channel_w"] for row in rows], dtype=float),
        "PCFM total": np.array([row["pcfm_channel_w"] for row in rows], dtype=float),
        "PCFM SCI": np.array([row["pcfm_sci_channel_w"] for row in rows], dtype=float),
        "PCFM XCI": np.array([row["pcfm_xci_channel_w"] for row in rows], dtype=float),
    }


def _save_figure(fig: plt.Figure, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=260)
    if out_path.suffix.lower() == ".png":
        fig.savefig(out_path.with_suffix(".pdf"))


def _plot_scaling(rows: list[dict], out_path: Path, channel_idx: int) -> None:
    spacing_ghz = np.array([row["channel_spacing_ghz"] for row in rows], dtype=float)
    series = _scaling_series(rows)
    freqs = np.array([row["channel_freq_thz"] for row in rows], dtype=float)
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    colors = ["black", "tab:red", "tab:blue", "tab:green", "tab:orange"]
    for (label, values), color in zip(series.items(), colors):
        ax.plot(
            spacing_ghz,
            values,
            marker="o",
            markersize=3,
            markerfacecolor="none",
            markeredgewidth=0.9,
            linewidth=1.0,
            color=color,
            label=f"{label} (p={_fit_exponent(spacing_ghz * 1e9, values):.3f})",
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Channel spacing [GHz]")
    ax.set_ylabel("Center-channel NLI power [W]")
    ax.set_title(
        "Center channel idx={}, f={:.3f}-{:.3f} THz".format(
            channel_idx,
            float(np.min(freqs)),
            float(np.max(freqs)),
        )
    )
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=7)
    _save_figure(fig, out_path)
    plt.close(fig)


def _plot_normalized_scaling(rows: list[dict], out_path: Path, channel_idx: int) -> None:
    spacing_ghz = np.array([row["channel_spacing_ghz"] for row in rows], dtype=float)
    series = _scaling_series(rows)
    freqs = np.array([row["channel_freq_thz"] for row in rows], dtype=float)
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    colors = ["black", "tab:red", "tab:blue", "tab:green", "tab:orange"]
    for (label, values), color in zip(series.items(), colors):
        ref = float(values[0]) if values.size else 1.0
        normalized = values / ref if ref > 0.0 else values
        ax.plot(
            spacing_ghz,
            normalized,
            marker="o",
            markersize=3,
            markerfacecolor="none",
            markeredgewidth=0.9,
            linewidth=1.0,
            color=color,
            label=label,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Channel spacing [GHz]")
    ax.set_ylabel("Normalized center-channel NLI power")
    ax.set_title(
        "Center channel idx={}, f={:.3f}-{:.3f} THz".format(
            channel_idx,
            float(np.min(freqs)),
            float(np.max(freqs)),
        )
    )
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=7)
    _save_figure(fig, out_path)
    plt.close(fig)


def run_spacing_sweep(
    cfg_path: Path,
    spacing_ghz_values: list[float],
    out_dir: Path,
    launch_dbm: float | None,
    pcfm_numeric_xci: bool,
    recompute_td: bool,
    recompute_pcfm: bool,
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    skipped_spacings: list[tuple[float, str]] = []

    base_system = System.from_toml(cfg_path)
    center_idx, band_label = _center_channel(base_system)
    lg.info("Selected center channel idx={} in band={}.".format(center_idx, band_label))

    for spacing_ghz in spacing_ghz_values:
        system = System.from_toml(cfg_path)
        spacing_hz = float(spacing_ghz) * 1e9
        try:
            _rebuild_wdm_with_spacing(system, spacing_hz)
        except ValueError as exc:
            skipped_spacings.append((float(spacing_ghz), str(exc)))
            lg.warning("Skipping S={:.2f} GHz: {}".format(float(spacing_ghz), exc))
            continue
        freqs = system.wdm.frequency_grid()
        channel_freq_thz = float(freqs[center_idx] * 1e-12)

        if launch_dbm is not None:
            launch_w = 10 ** ((float(launch_dbm) - 30.0) / 10.0)
            launch_vec = np.full(system.n_channels, launch_w, dtype=float)
        else:
            launch_vec = _resolve_launch_powers(
                system,
                profile_path=None,
                launch_csv_path=None,
                use_profile=False,
            )

        spacing_tag = f"S{_safe_tag(spacing_ghz)}GHz"
        profile_path = out_dir / f"flat_profile_{spacing_tag}.npy"
        _write_flat_profile(profile_path, system, launch_powers_w=launch_vec)

        ccfs = collision_coeffs_system_uwb(
            system,
            ipulse=1,
            recompute=recompute_td,
            profile_path=profile_path,
        )
        td_cache = _nlin_cache_path(
            profile_path=profile_path,
            use_kappa=False,
            use_x_mode=True,
            extra_tag=spacing_tag,
        )
        nlin_td = total_nlin_uwb(
            system,
            ccfs,
            use_kappa=False,
            use_x_mode=True,
            launch_powers_w=launch_vec,
            cache_path=td_cache,
            recompute=recompute_td,
        )
        td_vec = np.asarray(nlin_td, dtype=float).reshape(-1) * float(MANAKOV_SCALE_POGGIOLINI)

        const_pref, sum_a, sum_b = _td_modulation_components(
            system,
            ccfs,
            launch_vec,
            use_kappa=False,
            use_x_mode=True,
        )
        const_pref = np.asarray(const_pref, dtype=float) * float(MANAKOV_SCALE_POGGIOLINI)
        td_gaussian_vec = np.asarray(
            const_pref * (gaussian_mu0() * sum_a + sum_b),
            dtype=float,
        ).reshape(-1)

        pcfm_cfg = PcfmConfig(
            degree=9,
            include_mci=False,
            use_numeric_sci=True,
            use_numeric_xci=bool(pcfm_numeric_xci),
        )
        pcfm_path = out_dir / f"pcfm_{spacing_tag}.npy"
        pcfm_total, pcfm_sci, pcfm_xci = _load_or_compute_pcfm(
            system=system,
            profile_path=profile_path,
            launch_powers_w=launch_vec,
            output_path=pcfm_path,
            cfg=pcfm_cfg,
            lumped_losses=None,
            recompute=recompute_pcfm,
            return_components=True,
        )
        pcfm_total = np.asarray(pcfm_total, dtype=float).reshape(-1)
        pcfm_sci = np.asarray(pcfm_sci, dtype=float).reshape(-1)
        pcfm_xci = np.asarray(pcfm_xci, dtype=float).reshape(-1)

        row = {
            "channel_spacing_ghz": float(spacing_ghz),
            "channel_idx": int(center_idx),
            "band_label": band_label,
            "channel_freq_thz": channel_freq_thz,
            "launch_channel_w": float(launch_vec[center_idx]),
            "td_channel_w": float(td_vec[center_idx]),
            "td_gaussian_channel_w": float(td_gaussian_vec[center_idx]),
            "pcfm_channel_w": float(pcfm_total[center_idx]),
            "pcfm_sci_channel_w": float(pcfm_sci[center_idx]),
            "pcfm_xci_channel_w": float(pcfm_xci[center_idx]),
        }
        rows.append(row)
        lg.info(
            "S={:.2f} GHz -> f={:.6f} THz, TD={:.3e} W, TD(Gauss)={:.3e} W, "
            "PCFM={:.3e} W, PCFM_SCI={:.3e} W, PCFM_XCI={:.3e} W".format(
                row["channel_spacing_ghz"],
                row["channel_freq_thz"],
                row["td_channel_w"],
                row["td_gaussian_channel_w"],
                row["pcfm_channel_w"],
                row["pcfm_sci_channel_w"],
                row["pcfm_xci_channel_w"],
            )
        )

    rows.sort(key=lambda item: item["channel_spacing_ghz"])
    if not rows:
        raise RuntimeError("No valid spacing points were available for the sweep.")
    spacing_hz = np.array([row["channel_spacing_ghz"] * 1e9 for row in rows], dtype=float)
    for key in (
        "td_channel_w",
        "td_gaussian_channel_w",
        "pcfm_channel_w",
        "pcfm_sci_channel_w",
        "pcfm_xci_channel_w",
    ):
        exponent = _fit_exponent(spacing_hz, np.array([row[key] for row in rows], dtype=float))
        for row in rows:
            row[f"{key}_scaling_exp"] = exponent

    csv_path = out_dir / "center_channel_spacing_scaling.csv"
    header = [
        "channel_spacing_ghz",
        "channel_idx",
        "band_label",
        "channel_freq_thz",
        "launch_channel_w",
        "td_channel_w",
        "td_gaussian_channel_w",
        "pcfm_channel_w",
        "pcfm_sci_channel_w",
        "pcfm_xci_channel_w",
        "td_channel_w_scaling_exp",
        "td_gaussian_channel_w_scaling_exp",
        "pcfm_channel_w_scaling_exp",
        "pcfm_sci_channel_w_scaling_exp",
        "pcfm_xci_channel_w_scaling_exp",
    ]
    data = np.column_stack([[row[name] for row in rows] for name in header])
    np.savetxt(csv_path, data, delimiter=",", header=",".join(header), comments="", fmt="%s")

    plot_path = Path("media") / "PCFM" / "center_channel_spacing_scaling.png"
    _plot_scaling(rows, plot_path, center_idx)
    normalized_plot_path = Path("media") / "PCFM" / "center_channel_spacing_scaling_normalized.png"
    _plot_normalized_scaling(rows, normalized_plot_path, center_idx)

    summary_lines = [
        f"Center channel idx={center_idx}, band={band_label}",
        "Channel frequency range: {:.6f}-{:.6f} THz".format(
            float(np.min([row["channel_freq_thz"] for row in rows])),
            float(np.max([row["channel_freq_thz"] for row in rows])),
        ),
        f"TD 64-QAM exponent: {_fit_exponent(spacing_hz, np.array([row['td_channel_w'] for row in rows], dtype=float)):.6f}",
        f"TD Gaussian exponent: {_fit_exponent(spacing_hz, np.array([row['td_gaussian_channel_w'] for row in rows], dtype=float)):.6f}",
        f"PCFM total exponent: {_fit_exponent(spacing_hz, np.array([row['pcfm_channel_w'] for row in rows], dtype=float)):.6f}",
        f"PCFM SCI exponent: {_fit_exponent(spacing_hz, np.array([row['pcfm_sci_channel_w'] for row in rows], dtype=float)):.6f}",
        f"PCFM XCI exponent: {_fit_exponent(spacing_hz, np.array([row['pcfm_xci_channel_w'] for row in rows], dtype=float)):.6f}",
    ]
    if skipped_spacings:
        summary_lines.append("Skipped spacing points:")
        summary_lines.extend(
            [
                "S={:.2f} GHz: {}".format(spacing_ghz, reason)
                for spacing_ghz, reason in skipped_spacings
            ]
        )
    summary_path = out_dir / "center_channel_spacing_scaling_summary.txt"
    summary_path.write_text("\n".join(summary_lines) + "\n")
    return csv_path, summary_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep channel spacing and inspect center-channel TD/PCFM scaling."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="input/poggiolini_struct.toml",
        help="Path to system TOML.",
    )
    parser.add_argument(
        "--spacing-ghz",
        type=str,
        default="50,62.5,75,100,118.75,125",
        help="Comma-separated channel spacing values in GHz.",
    )
    parser.add_argument(
        "--launch-dbm",
        type=float,
        default=None,
        help="Optional uniform launch power override in dBm.",
    )
    parser.add_argument(
        "--pcfm-numeric-xci",
        action="store_true",
        help="Use numeric XCI in PCFM instead of closed-form XCI.",
    )
    parser.add_argument(
        "--recompute-td",
        action="store_true",
        help="Recompute TD collision caches.",
    )
    parser.add_argument(
        "--recompute-pcfm",
        action="store_true",
        help="Recompute PCFM caches.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="results/debug/poggiolini_spacing_scaling",
        help="Output directory for generated profiles, caches, and reports.",
    )
    args = parser.parse_args()

    csv_path, summary_path = run_spacing_sweep(
        cfg_path=Path(args.config),
        spacing_ghz_values=_parse_float_list(args.spacing_ghz),
        out_dir=Path(args.out_dir),
        launch_dbm=args.launch_dbm,
        pcfm_numeric_xci=bool(args.pcfm_numeric_xci),
        recompute_td=bool(args.recompute_td),
        recompute_pcfm=bool(args.recompute_pcfm),
    )
    lg.success(f"Saved scaling CSV to {csv_path}")
    lg.success(f"Saved scaling summary to {summary_path}")


if __name__ == "__main__":
    main()
