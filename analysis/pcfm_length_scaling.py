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

from analysis.pcfm.config import (
    _format_scaling_plot_title,
    _load_pcfm_runtime_config,
    _resolve_scaling_run_flags,
    _select_scaling_channel,
)
from analysis.pcfm.analytics import flat_profile_pcfm_xci_channel_power
from analysis.pcfm.figure_size import scale_figsize_to_ieee_column
from analysis.pcfm.io import _end_over_launch_power_ratio, _resolve_launch_powers, _write_flat_profile
from analysis.pcfm.models import _load_or_compute_pcfm
from analysis.pcfm.ssfm_interface import (
    compute_ssfm_center_nli,
    prepare_ssfm_runtime_with_template,
)
from analysis.pcfm.td import _td_modulation_components
from analysis.uwb_nlin import _nlin_cache_path

plt.rcParams["text.usetex"] = False


def _parse_float_list(values: str) -> list[float]:
    return [float(x.strip()) for x in values.split(",") if x.strip()]


def _safe_tag(value: float) -> str:
    text = f"{value:.6e}"
    return text.replace("+", "").replace("-", "m").replace(".", "p")


def _fit_exponent(lengths_m: np.ndarray, values_w: np.ndarray) -> float:
    mask = np.isfinite(lengths_m) & np.isfinite(values_w) & (lengths_m > 0.0) & (values_w > 0.0)
    if int(np.count_nonzero(mask)) < 2:
        return float("nan")
    coeffs = np.polyfit(np.log(lengths_m[mask]), np.log(values_w[mask]), 1)
    return float(coeffs[0])


def _disable_dbm_axis_grouping(ax: plt.Axes) -> None:
    formatter = ScalarFormatter(useOffset=False)
    formatter.set_scientific(False)
    ax.yaxis.set_major_formatter(formatter)
    ax.yaxis.offsetText.set_visible(False)


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


def _uses_dar2014_template(ssfm_template_path: Path | None) -> bool:
    if ssfm_template_path is None:
        return False
    return ssfm_template_path.name.lower() == "ssfm_dar_2014.toml"


def _load_dar2014_literature_points(
    csv_path: Path,
    reference_signal_dbm: float = -2.0,
) -> tuple[np.ndarray, np.ndarray]:
    lengths_km: list[float] = []
    nli_w: list[float] = []
    for raw_line in csv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if ";" not in line:
            continue
        left, right = line.split(";", 1)
        length_km = float(left.strip().replace(",", "."))
        normalized_db = float(right.strip().replace(",", "."))
        denorm_dbm = reference_signal_dbm + normalized_db
        lengths_km.append(length_km)
        nli_w.append(float(dBm2watt(denorm_dbm)))
    return np.asarray(lengths_km, dtype=float), np.asarray(nli_w, dtype=float)


def _plot_scaling(
    rows: list[dict],
    out_path: Path,
    base_system: System,
    channel_idx: int,
    band_label: str,
    channel_freq_thz: float,
    launch_powers_w: np.ndarray,
    plot_pcfm_total_and_sci: bool,
    literature_points: tuple[np.ndarray, np.ndarray] | None = None,
) -> None:
    lengths_km = np.array([row["length_km"] for row in rows], dtype=float)
    series = _scaling_series(rows, plot_pcfm_total_and_sci)
    fig, ax = plt.subplots(figsize=scale_figsize_to_ieee_column(5.2, 3.4))
    colors = plt.rcParams.get("axes.prop_cycle", None)
    palette = colors.by_key().get("color", []) if colors is not None else []
    for idx, (label, values) in enumerate(series.items()):
        color = palette[idx % len(palette)] if palette else None
        ax.plot(
            lengths_km,
            watt2dBm(np.maximum(np.asarray(values, dtype=float), 1e-18)),
            marker="o",
            markersize=3,
            markerfacecolor="none",
            markeredgewidth=0.9,
            linewidth=1.0,
            color=color,
            label=f"{label} (p={_fit_exponent(lengths_km * 1e3, values):.3f})",
        )
        
    ### put a linear scaling function compatible with the initial lenght data point of PCFM XCI
    initial_length_km = lengths_km[0]
    initial_value_w = series["PCFM XCI"][0]
    linear_values = initial_value_w * (lengths_km / initial_length_km)
    ax.plot(
        lengths_km,
        watt2dBm(np.maximum(np.asarray(linear_values, dtype=float), 1e-18)),
        linestyle="--",
        color="gray",
        label="Linear scaling (p=1.000)",
    )
    if literature_points is not None:
        lit_len_km, lit_nli_w = literature_points
        if lit_len_km.size and lit_nli_w.size:
            ax.plot(
                lit_len_km,
                watt2dBm(np.maximum(np.asarray(lit_nli_w, dtype=float), 1e-18)),
                linestyle="None",
                marker="x",
                markersize=6,
                markeredgewidth=1.0,
                color="black",
                label="Dar 2014 (digitized)",
            )
    ax.set_xscale("log")
    ax.set_xlabel("Fiber length [km]")
    ax.set_ylabel("Center-channel NLI power [dBm]")
    _disable_dbm_axis_grouping(ax)
    ax.set_title(
        _format_scaling_plot_title(
            base_system,
            sweep_axis="length",
            channel_idx=channel_idx,
            band_label=band_label,
            launch_powers_w=launch_powers_w,
            channel_freq_thz=channel_freq_thz,
        )
    )
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=7)
    _save_figure(fig, out_path)
    plt.close(fig)




def run_length_sweep(
    cfg_path: Path,
    lengths_km: list[float],
    out_dir: Path,
    launch_dbm: float | None,
    pcfm_numeric_xci: bool | None,
    pcfm_eq18_xci: bool | None,
    recompute_td: bool | None,
    recompute_pcfm: bool | None,
    exclude_self_channel: bool | None = None,
    run_ssfm_when_standalone: bool = False,
    ssfm_template_path: Path | None = None,
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    effective_cfg_path = cfg_path
    effective_ssfm_template_path = ssfm_template_path
    if ssfm_template_path is not None:
        tpl_text = str(ssfm_template_path).lower()
        if "struct" in tpl_text:
            if Path(ssfm_template_path).exists():
                effective_cfg_path = Path(ssfm_template_path)
                effective_ssfm_template_path = None
                lg.warning(
                    "--ssfm-template points to a system struct TOML; using it as --config and falling back to default gnlse SSFM template: {}",
                    ssfm_template_path,
                )
            else:
                lg.warning(
                    "Provided struct-like path in --ssfm-template does not exist: {}. Falling back to --config={} and default gnlse SSFM template.",
                    ssfm_template_path,
                    cfg_path,
                )
                effective_ssfm_template_path = None
        elif not Path(ssfm_template_path).exists():
            lg.warning(
                "SSFM template not found: {}. Falling back to default gnlse SSFM template.",
                ssfm_template_path,
            )
            effective_ssfm_template_path = None

    base_system = System.from_toml(effective_cfg_path)
    sweep_template_system = base_system
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
    center_idx, band_label = _select_scaling_channel(sweep_template_system)
    freqs = sweep_template_system.wdm.frequency_grid()
    channel_freq_thz = float(freqs[center_idx] * 1e-12)
    if launch_dbm is not None:
        title_launch_vec = np.full(sweep_template_system.n_channels, float(dBm2watt(launch_dbm)), dtype=float)
    else:
        title_launch_vec = _build_constant_launch_vector(sweep_template_system, None)
    lg.info(
        "Selected center channel idx={} in band={} at {:.6f} THz.".format(
            center_idx, band_label, channel_freq_thz
        )
    )

    ssfm_ctx = (
        prepare_ssfm_runtime_with_template(out_dir, template_path=effective_ssfm_template_path)
        if run_ssfm_when_standalone
        else None
    )
    literature_points: tuple[np.ndarray, np.ndarray] | None = None
    lit_csv = Path("input") / "literature_comparison" / "dar2014points.csv"
    if lit_csv.exists():
        try:
            literature_points = _load_dar2014_literature_points(lit_csv)
            lg.info("Loaded {} literature points from {}", len(literature_points[0]), lit_csv)
        except Exception as exc:
            lg.warning("Could not load literature comparison points from {}: {}", lit_csv, exc)
    else:
        lg.warning("Literature comparison CSV not found: {}", lit_csv)

    for length_km in lengths_km:
        system = System.from_toml(effective_cfg_path)
        system.fiber.length = float(length_km) * 1e3
        launch_vec = _build_constant_launch_vector(system, launch_dbm)

        profile_tag = f"L{_safe_tag(length_km)}km"
        profile_path = out_dir / f"flat_profile_{profile_tag}.npy"
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
        td_tag = f"{profile_tag}_{'xci' if exclude_self_channel else 'all'}"
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
        pcfm_path = out_dir / f"pcfm_{profile_tag}.npy"
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
            "length_km": float(length_km),
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
                    sweep_tag=profile_tag,
                )
                if ssfm_val is not None:
                    row["ssfm_channel_w"] = float(ssfm_val)
                else:
                    lg.warning("SSFM returned no finite center-channel NLI for {}", profile_tag)
            except Exception as exc:
                lg.warning("SSFM run failed for {}: {}", profile_tag, exc)
        rows.append(row)
        msg = (
            "L={:.1f} km -> TD={:.3e} W, TD(Gauss)={:.3e} W, PCFM={:.3e} W, "
            "PCFM_SCI={:.3e} W, PCFM_XCI={:.3e} W"
        ).format(
            row["length_km"],
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

    rows.sort(key=lambda item: item["length_km"])
    lengths_m = np.array([row["length_km"] * 1e3 for row in rows], dtype=float)
    for key in (
        "td_channel_w",
        "td_gaussian_channel_w",
        "pcfm_channel_w",
        "pcfm_sci_channel_w",
        "pcfm_xci_channel_w",
    ):
        exponent = _fit_exponent(lengths_m, np.array([row[key] for row in rows], dtype=float))
        for row in rows:
            row[f"{key}_scaling_exp"] = exponent
    if pcfm_eq18_xci:
        exponent = _fit_exponent(lengths_m, np.array([row["pcfm_eq18_xci_channel_w"] for row in rows], dtype=float))
        for row in rows:
            row["pcfm_eq18_xci_channel_w_scaling_exp"] = exponent
    if any("ssfm_channel_w" in row for row in rows):
        exponent = _fit_exponent(lengths_m, np.array([row.get("ssfm_channel_w", np.nan) for row in rows], dtype=float))
        for row in rows:
            if "ssfm_channel_w" in row:
                row["ssfm_channel_w_scaling_exp"] = exponent

    csv_path = out_dir / "center_channel_length_scaling.csv"
    header = [
        "length_km",
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

    plot_path = Path("media") / "PCFM" / "center_channel_length_scaling.pdf"
    _plot_scaling(
        rows,
        plot_path,
        base_system,
        center_idx,
        band_label,
        channel_freq_thz,
        title_launch_vec,
        plot_pcfm_total_and_sci=plot_pcfm_total_and_sci,
        literature_points=literature_points,
    )

    summary_lines = [
        f"Center channel idx={center_idx}, band={band_label}, freq={channel_freq_thz:.6f} THz",
        f"TD mode: {'exclude self-channel (nuB==nuA)' if exclude_self_channel else 'include all channels'}",
        f"TD 64-QAM exponent: {_fit_exponent(lengths_m, np.array([row['td_channel_w'] for row in rows], dtype=float)):.6f}",
        f"TD Gaussian exponent: {_fit_exponent(lengths_m, np.array([row['td_gaussian_channel_w'] for row in rows], dtype=float)):.6f}",
        f"PCFM total exponent: {_fit_exponent(lengths_m, np.array([row['pcfm_channel_w'] for row in rows], dtype=float)):.6f}",
        f"PCFM SCI exponent: {_fit_exponent(lengths_m, np.array([row['pcfm_sci_channel_w'] for row in rows], dtype=float)):.6f}",
        f"PCFM XCI exponent: {_fit_exponent(lengths_m, np.array([row['pcfm_xci_channel_w'] for row in rows], dtype=float)):.6f}",
    ]
    if pcfm_eq18_xci:
        summary_lines.append(
            f"PCFM XCI Eq18 exponent: {_fit_exponent(lengths_m, np.array([row['pcfm_eq18_xci_channel_w'] for row in rows], dtype=float)):.6f}"
        )
    if any("ssfm_channel_w" in row for row in rows):
        summary_lines.append(
            f"SSFM exponent: {_fit_exponent(lengths_m, np.array([row.get('ssfm_channel_w', np.nan) for row in rows], dtype=float)):.6f}"
        )
    if literature_points is not None:
        summary_lines.append(f"Literature points loaded: {int(literature_points[0].size)}")
    summary_path = out_dir / "center_channel_length_scaling_summary.txt"
    summary_path.write_text("\n".join(summary_lines) + "\n")
    return csv_path, summary_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep fiber length and inspect center-channel TD/PCFM scaling."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="input/pcfm_struct.toml",
        help="Path to system TOML.",
    )
    parser.add_argument(
        "--lengths-km",
        type=str,
        default="10,25,50,100,200",
        help="Comma-separated fiber lengths in km.",
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
        default="results/debug/pcfm_length_scaling",
        help="Output directory for generated profiles, caches, and reports.",
    )
    parser.add_argument(
        "--ssfm-template",
        type=str,
        default=None,
        help=(
            "Optional SSFM template TOML path (e.g., input/ssfm_dar_2014.toml). "
            "If omitted, uses the default gnlse wdm_nli_config.toml."
        ),
    )
    args = parser.parse_args()

    csv_path, summary_path = run_length_sweep(
        cfg_path=Path(args.config),
        lengths_km=_parse_float_list(args.lengths_km),
        out_dir=Path(args.out_dir),
        launch_dbm=args.launch_dbm,
        pcfm_numeric_xci=args.pcfm_numeric_xci,
        pcfm_eq18_xci=None,
        recompute_td=args.recompute_td,
        recompute_pcfm=args.recompute_pcfm,
        exclude_self_channel=args.exclude_self_channel,
        run_ssfm_when_standalone=True,
        ssfm_template_path=(Path(args.ssfm_template) if args.ssfm_template else None),
    )
    lg.success(f"Saved scaling CSV to {csv_path}")
    lg.success(f"Saved scaling summary to {summary_path}")
    # call the plotting functions
    # _plot_scaling(rows, plot_path, center_idx, channel_freq_thz)
    # _plot_normalized_scaling(rows, normalized_plot_path, center_idx, channel_freq


if __name__ == "__main__":
    main()
