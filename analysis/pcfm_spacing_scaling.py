import argparse
from pathlib import Path

import matplotlib
import numpy as np
from loguru import logger as lg
from matplotlib.ticker import ScalarFormatter

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pynlin.constellation_stats import gaussian_mu0
from pynlin.nlin.nlin_estimator_uwb import collision_coeffs_system_uwb, total_nlin_uwb
from pynlin.nlin.pcfm_gn import PcfmConfig
from pynlin.system import System
from pynlin.utils import dBm2watt, watt2dBm
from pynlin.wdm import IrregularWDM, RegularWDM

from analysis.pcfm.config import (
    _channel_nonoverlap_spacing_hz,
    _format_scaling_plot_title,
    _load_pcfm_runtime_config,
    _resolve_scaling_run_flags,
    _select_scaling_channel,
    _wdm_nonoverlap_max_spacing_hz,
)
from analysis.pcfm.analytics import flat_profile_pcfm_xci_channel_power
from analysis.pcfm.figure_size import scale_figsize_to_ieee_column
from analysis.pcfm.io import _end_over_launch_power_ratio, _resolve_launch_powers, _write_flat_profile
from analysis.pcfm.models import _load_or_compute_pcfm
from analysis.pcfm.ssfm_interface import compute_ssfm_center_nli, prepare_ssfm_runtime
from analysis.pcfm.td import _td_modulation_components
from analysis.uwb_nlin import _nlin_cache_path

plt.rcParams["text.usetex"] = False


def _parse_float_list(values: str) -> list[float]:
    return [float(x.strip()) for x in values.split(",") if x.strip()]


def _safe_tag(value: float) -> str:
    text = f"{value:.6e}"
    return text.replace("+", "").replace("-", "m").replace(".", "p")


def _fit_exponent(x_values: np.ndarray, values_w: np.ndarray) -> float:
    mask = np.isfinite(x_values) & np.isfinite(values_w) & (x_values > 0.0) & (values_w > 0.0)
    if int(np.count_nonzero(mask)) < 2:
        return float("nan")
    coeffs = np.polyfit(np.log(x_values[mask]), np.log(values_w[mask]), 1)
    return float(coeffs[0])


def _disable_dbm_axis_grouping(ax: plt.Axes) -> None:
    formatter = ScalarFormatter(useOffset=False)
    formatter.set_scientific(False)
    ax.yaxis.set_major_formatter(formatter)
    ax.yaxis.offsetText.set_visible(False)


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


def _scaling_series(
    rows: list[dict],
    plot_pcfm_total_and_sci: bool,
) -> dict[str, np.ndarray]:
    series = {
        "TD 64-QAM": np.array([row["td_channel_w"] for row in rows], dtype=float),
        "TD Gaussian": np.array([row["td_gaussian_channel_w"] for row in rows], dtype=float),
        "PCFM XCI": np.array([row["pcfm_xci_channel_w"] for row in rows], dtype=float),
    }
    if rows and "pcfm_eq18_xci_channel_w" in rows[0]:
        series["PCFM XCI Eq18"] = np.array([row["pcfm_eq18_xci_channel_w"] for row in rows], dtype=float)
    if any("ssfm_channel_w" in row for row in rows):
        series["SSFM"] = np.array([row.get("ssfm_channel_w", np.nan) for row in rows], dtype=float)
    if plot_pcfm_total_and_sci:
        series["PCFM total"] = np.array([row["pcfm_channel_w"] for row in rows], dtype=float)
        series["PCFM SCI"] = np.array([row["pcfm_sci_channel_w"] for row in rows], dtype=float)
    return series


def _build_constant_launch_vector(system: System, launch_dbm: float | None) -> np.ndarray:
    if launch_dbm is not None:
        launch_w = float(dBm2watt(launch_dbm))
    else:
        launch_w = float(
            _resolve_launch_powers(
                system,
                profile_path=None,
                launch_csv_path=None,
                use_profile=False,
            )[0]
        )
    return np.full(system.n_channels, launch_w, dtype=float)


def _center_channel_eq18_xci(system: System, launch_channel_w: float, channel_idx: int) -> float:
    launch = np.full(system.wdm.frequency_grid().size, float(launch_channel_w), dtype=float)
    return flat_profile_pcfm_xci_channel_power(
        system,
        channel_idx=channel_idx,
        launch_powers_w=launch,
        use_beta2_eff=True,
        xci_model="eq18",
    )


def _save_figure(fig: plt.Figure, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=260)
    lg.success("Plot saved: {}", out_path.resolve())


def _plot_scaling(
    rows: list[dict],
    out_path: Path,
    base_system: System,
    channel_idx: int,
    band_label: str,
    launch_powers_w: np.ndarray,
    plot_pcfm_total_and_sci: bool,
) -> None:
    spacing_ghz = np.array([row["channel_spacing_ghz"] for row in rows], dtype=float)
    series = _scaling_series(rows, plot_pcfm_total_and_sci)
    freqs = np.array([row["channel_freq_thz"] for row in rows], dtype=float)
    fig, ax = plt.subplots(figsize=scale_figsize_to_ieee_column(5.2, 3.4))
    colors = plt.rcParams.get("axes.prop_cycle", None)
    palette = colors.by_key().get("color", []) if colors is not None else []
    for idx, (label, values) in enumerate(series.items()):
        color = palette[idx % len(palette)] if palette else None
        ax.plot(
            spacing_ghz,
            watt2dBm(np.maximum(np.asarray(values, dtype=float), 1e-18)),
            marker="o",
            markersize=3,
            markerfacecolor="none",
            markeredgewidth=0.9,
            linewidth=1.0,
            color=color,
            label=f"{label} (p={_fit_exponent(spacing_ghz * 1e9, values):.3f})",
        )
    ax.set_xscale("log")
    ax.set_xlabel("Channel spacing [GHz]")
    ax.set_ylabel("Center-channel NLI power [dBm]")
    _disable_dbm_axis_grouping(ax)
    ax.set_title(
        _format_scaling_plot_title(
            base_system,
            sweep_axis="spacing",
            channel_idx=channel_idx,
            band_label=band_label,
            launch_powers_w=launch_powers_w,
            channel_freq_range_thz=(float(np.min(freqs)), float(np.max(freqs))),
        )
    )
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=7)
    _save_figure(fig, out_path)
    plt.close(fig)


def _plot_normalized_scaling(
    rows: list[dict],
    out_path: Path,
    base_system: System,
    channel_idx: int,
    band_label: str,
    launch_powers_w: np.ndarray,
    plot_pcfm_total_and_sci: bool,
) -> None:
    spacing_ghz = np.array([row["channel_spacing_ghz"] for row in rows], dtype=float)
    series = _scaling_series(rows, plot_pcfm_total_and_sci)
    freqs = np.array([row["channel_freq_thz"] for row in rows], dtype=float)
    fig, ax = plt.subplots(figsize=scale_figsize_to_ieee_column(5.2, 3.4))
    colors = plt.rcParams.get("axes.prop_cycle", None)
    palette = colors.by_key().get("color", []) if colors is not None else []
    for idx, (label, values) in enumerate(series.items()):
        color = palette[idx % len(palette)] if palette else None
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
        _format_scaling_plot_title(
            base_system,
            sweep_axis="spacing",
            channel_idx=channel_idx,
            band_label=band_label,
            launch_powers_w=launch_powers_w,
            channel_freq_range_thz=(float(np.min(freqs)), float(np.max(freqs))),
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
    pcfm_numeric_xci: bool | None,
    pcfm_eq18_xci: bool | None,
    recompute_td: bool | None,
    recompute_pcfm: bool | None,
    exclude_self_channel: bool | None = None,
    run_ssfm_when_standalone: bool = False,
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    skipped_spacings: list[tuple[float, str]] = []

    base_system = System.from_toml(cfg_path)
    runtime_cfg = _load_pcfm_runtime_config(base_system)
    run_flags = _resolve_scaling_run_flags(
        base_system,
        pcfm_numeric_xci=pcfm_numeric_xci,
        pcfm_eq18_xci=pcfm_eq18_xci,
        recompute_td=recompute_td,
        recompute_pcfm=recompute_pcfm,
        exclude_self_channel=exclude_self_channel,
    )
    pcfm_numeric_xci = run_flags["pcfm_numeric_xci"]
    pcfm_eq18_xci = run_flags["pcfm_eq18_xci"]
    recompute_td = run_flags["recompute_td"]
    recompute_pcfm = run_flags["recompute_pcfm"]
    exclude_self_channel = run_flags["exclude_self_channel"]
    plot_pcfm_total_and_sci = bool(runtime_cfg["plot_pcfm_total_and_sci"])
    center_idx, band_label = _select_scaling_channel(base_system)
    if launch_dbm is not None:
        title_launch_vec = np.full(base_system.n_channels, float(dBm2watt(launch_dbm)), dtype=float)
    else:
        title_launch_vec = _build_constant_launch_vector(base_system, None)
    min_spacing_nonoverlap_hz = _channel_nonoverlap_spacing_hz(base_system)
    min_spacing_nonoverlap_ghz = min_spacing_nonoverlap_hz * 1e-9
    max_spacing_nonoverlap_hz = _wdm_nonoverlap_max_spacing_hz(base_system)
    max_spacing_nonoverlap_ghz = (
        None if max_spacing_nonoverlap_hz is None else max_spacing_nonoverlap_hz * 1e-9
    )
    lg.info("Selected center channel idx={} in band={}.".format(center_idx, band_label))
    lg.info(
        "Using spacing fit range S in [{:.3f}, {}] GHz.".format(
            min_spacing_nonoverlap_ghz,
            "inf"
            if max_spacing_nonoverlap_ghz is None
            else f"{max_spacing_nonoverlap_ghz:.3f}",
        )
    )

    ssfm_ctx = prepare_ssfm_runtime(out_dir) if run_ssfm_when_standalone else None

    for spacing_ghz in spacing_ghz_values:
        system = System.from_toml(cfg_path)
        spacing_hz = float(spacing_ghz) * 1e9
        if spacing_hz < min_spacing_nonoverlap_hz:
            reason = (
                "below non-overlap threshold "
                f"{min_spacing_nonoverlap_ghz:.3f} GHz"
            )
            skipped_spacings.append((float(spacing_ghz), reason))
            lg.warning("Skipping S={:.2f} GHz: {}".format(float(spacing_ghz), reason))
            continue
        if max_spacing_nonoverlap_hz is not None and spacing_hz > max_spacing_nonoverlap_hz:
            reason = (
                "above inter-band non-overlap threshold "
                f"{max_spacing_nonoverlap_ghz:.3f} GHz"
            )
            skipped_spacings.append((float(spacing_ghz), reason))
            lg.warning("Skipping S={:.2f} GHz: {}".format(float(spacing_ghz), reason))
            continue
        try:
            _rebuild_wdm_with_spacing(system, spacing_hz)
        except ValueError as exc:
            skipped_spacings.append((float(spacing_ghz), str(exc)))
            lg.warning("Skipping S={:.2f} GHz: {}".format(float(spacing_ghz), exc))
            continue
        freqs = system.wdm.frequency_grid()
        channel_freq_thz = float(freqs[center_idx] * 1e-12)

        launch_vec = _build_constant_launch_vector(system, launch_dbm)

        spacing_tag = f"S{_safe_tag(spacing_ghz)}GHz"
        profile_path = out_dir / f"flat_profile_{spacing_tag}.npy"
        _write_flat_profile(profile_path, system, launch_powers_w=launch_vec)
        nlin_power_scale = _end_over_launch_power_ratio(
            system=system,
            profile_path=profile_path,
            launch_powers_w=launch_vec,
        )

        ccfs = collision_coeffs_system_uwb(
            system,
            ipulse=1,
            recompute=recompute_td,
            profile_path=profile_path,
        )
        td_tag = f"{spacing_tag}_{'xci' if exclude_self_channel else 'all'}"
        td_cache = _nlin_cache_path(
            profile_path=profile_path,
            use_kappa=True,
            use_x_mode=True,
            extra_tag=td_tag,
        )
        nlin_td = total_nlin_uwb(
            system,
            ccfs,
            use_kappa=True,
            use_x_mode=True,
            launch_powers_w=launch_vec,
            exclude_self_channel=exclude_self_channel,
            cache_path=td_cache,
            recompute=recompute_td,
        )
        td_vec = np.asarray(nlin_td, dtype=float).reshape(-1) * nlin_power_scale

        const_pref, sum_a, sum_b = _td_modulation_components(
            system,
            ccfs,
            launch_vec,
            use_kappa=True,
            use_x_mode=True,
            exclude_self_channel=exclude_self_channel,
        )
        td_gaussian_vec = np.asarray(
            const_pref * (gaussian_mu0() * sum_a + sum_b),
            dtype=float,
        ).reshape(-1) * nlin_power_scale

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
        pcfm_total = np.asarray(pcfm_total, dtype=float).reshape(-1) * nlin_power_scale
        pcfm_sci = np.asarray(pcfm_sci, dtype=float).reshape(-1) * nlin_power_scale
        pcfm_xci = np.asarray(pcfm_xci, dtype=float).reshape(-1) * nlin_power_scale

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
        if pcfm_eq18_xci:
            row["pcfm_eq18_xci_channel_w"] = _center_channel_eq18_xci(
                system, float(launch_vec[center_idx]), center_idx
            ) * float(nlin_power_scale[center_idx])
        if ssfm_ctx is not None:
            try:
                ssfm_val = compute_ssfm_center_nli(
                    ssfm_ctx,
                    system=system,
                    launch_channel_w=float(launch_vec[center_idx]),
                    channel_idx=center_idx,
                    sweep_tag=spacing_tag,
                )
                if ssfm_val is not None:
                    row["ssfm_channel_w"] = float(ssfm_val)
                else:
                    lg.warning("SSFM returned no finite center-channel NLI for {}", spacing_tag)
            except Exception as exc:
                lg.warning("SSFM run failed for {}: {}", spacing_tag, exc)
        rows.append(row)
        msg = (
            "S={:.2f} GHz -> f={:.6f} THz, TD={:.3e} W, TD(Gauss)={:.3e} W, "
            "PCFM={:.3e} W, PCFM_SCI={:.3e} W, PCFM_XCI={:.3e} W"
        ).format(
            row["channel_spacing_ghz"],
            row["channel_freq_thz"],
            row["td_channel_w"],
            row["td_gaussian_channel_w"],
            row["pcfm_channel_w"],
            row["pcfm_sci_channel_w"],
            row["pcfm_xci_channel_w"],
        )
        if pcfm_eq18_xci:
            msg += ", PCFM_XCI_Eq18={:.3e} W".format(row["pcfm_eq18_xci_channel_w"])
        if "ssfm_channel_w" in row:
            msg += ", SSFM={:.3e} W".format(row["ssfm_channel_w"])
        lg.info(msg)

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
    if pcfm_eq18_xci:
        exponent = _fit_exponent(spacing_hz, np.array([row["pcfm_eq18_xci_channel_w"] for row in rows], dtype=float))
        for row in rows:
            row["pcfm_eq18_xci_channel_w_scaling_exp"] = exponent
    if any("ssfm_channel_w" in row for row in rows):
        exponent = _fit_exponent(spacing_hz, np.array([row.get("ssfm_channel_w", np.nan) for row in rows], dtype=float))
        for row in rows:
            if "ssfm_channel_w" in row:
                row["ssfm_channel_w_scaling_exp"] = exponent

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
    if pcfm_eq18_xci:
        header.extend([
            "pcfm_eq18_xci_channel_w",
            "pcfm_eq18_xci_channel_w_scaling_exp",
        ])
    if any("ssfm_channel_w" in row for row in rows):
        header.extend([
            "ssfm_channel_w",
            "ssfm_channel_w_scaling_exp",
        ])
    data = np.column_stack([[row.get(name, np.nan) for row in rows] for name in header])
    np.savetxt(csv_path, data, delimiter=",", header=",".join(header), comments="", fmt="%s")

    plot_path = Path("media") / "PCFM" / "center_channel_spacing_scaling.pdf"
    _plot_scaling(
        rows,
        plot_path,
        base_system,
        center_idx,
        band_label,
        title_launch_vec,
        plot_pcfm_total_and_sci=plot_pcfm_total_and_sci,
    )
    normalized_plot_path = Path("media") / "PCFM" / "center_channel_spacing_scaling_normalized.pdf"
    _plot_normalized_scaling(
        rows,
        normalized_plot_path,
        base_system,
        center_idx,
        band_label,
        title_launch_vec,
        plot_pcfm_total_and_sci=plot_pcfm_total_and_sci,
    )

    summary_lines = [
        f"Center channel idx={center_idx}, band={band_label}",
        "Spacing fit range: [{:.6f}, {}] GHz".format(
            min_spacing_nonoverlap_ghz,
            "inf"
            if max_spacing_nonoverlap_ghz is None
            else f"{max_spacing_nonoverlap_ghz:.6f}",
        ),
        f"TD mode: {'exclude self-channel (nuB==nuA)' if exclude_self_channel else 'include all channels'}",
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
    if pcfm_eq18_xci:
        summary_lines.append(
            f"PCFM XCI Eq18 exponent: {_fit_exponent(spacing_hz, np.array([row['pcfm_eq18_xci_channel_w'] for row in rows], dtype=float)):.6f}"
        )
    if any("ssfm_channel_w" in row for row in rows):
        summary_lines.append(
            f"SSFM exponent: {_fit_exponent(spacing_hz, np.array([row.get('ssfm_channel_w', np.nan) for row in rows], dtype=float)):.6f}"
        )
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
        default="input/pcfm_struct.toml",
        help="Path to system TOML.",
    )
    parser.add_argument(
        "--spacing-ghz",
        type=str,
        default="56.25,62.5,75,100,118.75,125",
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
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override [pcfm.run].pcfm_numeric_xci.",
    )
    parser.add_argument(
        "--recompute-td",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override [pcfm.run].td_mode.",
    )
    parser.add_argument(
        "--recompute-pcfm",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override [pcfm.run].pcfm_mode.",
    )
    parser.add_argument(
        "--exclude-self-channel",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override [pcfm.run].td_exclude_self_channel.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="results/debug/pcfm_spacing_scaling",
        help="Output directory for generated profiles, caches, and reports.",
    )
    args = parser.parse_args()

    csv_path, summary_path = run_spacing_sweep(
        cfg_path=Path(args.config),
        spacing_ghz_values=_parse_float_list(args.spacing_ghz),
        out_dir=Path(args.out_dir),
        launch_dbm=args.launch_dbm,
        pcfm_numeric_xci=args.pcfm_numeric_xci,
        pcfm_eq18_xci=None,
        recompute_td=args.recompute_td,
        recompute_pcfm=args.recompute_pcfm,
        exclude_self_channel=args.exclude_self_channel,
        run_ssfm_when_standalone=True,
    )
    lg.success(f"Saved scaling CSV to {csv_path}")
    lg.success(f"Saved scaling summary to {summary_path}")


if __name__ == "__main__":
    main()
