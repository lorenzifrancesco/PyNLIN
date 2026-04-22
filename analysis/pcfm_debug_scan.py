import argparse
import csv
import hashlib
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger as lg
from scipy.constants import c

from pynlin.constellation_stats import gaussian_mu0, qam_mu0
from pynlin.nlin.nlin_estimator_uwb import collision_coeffs_system_uwb, total_nlin_uwb
from pynlin.nlin.pcfm_gn import PcfmConfig
from pynlin.system import System
from pynlin.utils import dBm2watt

try:
    from analysis.pcfm.config import _load_pcfm_runtime_config
    from analysis.pcfm.figure_size import (
        IEEE_DOUBLE_COLUMN_ONE_COLUMN_WIDTH_IN,
        scale_figsize_to_ieee_column,
    )
    from analysis.pcfm.io import _power_profile_hash, _resolve_signal_power, _write_flat_profile
    from analysis.pcfm.models import _load_or_compute_pcfm_I
    from analysis.pcfm.td import _td_modulation_components
    from analysis.uwb_nlin import _nlin_cache_path
except ModuleNotFoundError:
    from pcfm.config import _load_pcfm_runtime_config  # type: ignore[no-redef]
    from pcfm.figure_size import (  # type: ignore[no-redef]
        IEEE_DOUBLE_COLUMN_ONE_COLUMN_WIDTH_IN,
        scale_figsize_to_ieee_column,
    )
    from pcfm.io import _power_profile_hash, _resolve_signal_power, _write_flat_profile  # type: ignore[no-redef]
    from pcfm.models import _load_or_compute_pcfm_I  # type: ignore[no-redef]
    from pcfm.td import _td_modulation_components  # type: ignore[no-redef]
    from uwb_nlin import _nlin_cache_path  # type: ignore[no-redef]


def _parse_float_list(values: str) -> list[float]:
    return [float(x.strip()) for x in values.split(",") if x.strip()]


def _normalize_negative_csv_option(argv: list[str], option: str) -> list[str]:
    """Allow `--opt -1,-2` form for comma-separated negative lists."""
    normalized: list[str] = []
    i = 0
    while i < len(argv):
        token = argv[i]
        if token == option and (i + 1) < len(argv):
            candidate = argv[i + 1]
            if candidate.startswith("-") and not candidate.startswith("--"):
                try:
                    _parse_float_list(candidate)
                except ValueError:
                    pass
                else:
                    normalized.append(f"{option}={candidate}")
                    i += 2
                    continue
        normalized.append(token)
        i += 1
    return normalized


def _safe_tag(value: float) -> str:
    txt = f"{value:.6e}"
    return txt.replace("+", "").replace("-", "m").replace(".", "p")


def _set_constant_dispersion(system: System, beta2_value: float) -> None:
    """Force a constant beta2 while keeping beta1 profile coherent."""
    fiber = getattr(system, "fiber", None)
    if fiber is None:
        raise ValueError("System has no fiber object.")
    if not hasattr(fiber, "_force_constant_dispersion"):
        raise TypeError("Expected an SM fiber with force_constant_dispersion support.")

    fiber._force_constant_dispersion = True
    fiber.beta2 = float(beta2_value)
    fiber._beta2_profile = None
    fiber._beta_profile = None

    freq_profile = getattr(fiber, "_freq_profile", None)
    beta1_center = getattr(fiber, "beta1", None)
    if freq_profile is None or beta1_center is None:
        lg.warning(
            "Cannot rebuild beta1 profile (missing frequency profile or beta1 center). "
            "Using existing beta1 profile."
        )
        return

    freq_profile = np.asarray(freq_profile, dtype=float)
    omega = 2.0 * np.pi * freq_profile
    omega_c = 2.0 * np.pi * (
        float(system.center_frequency)
        if system.center_frequency is not None
        else float(np.mean(freq_profile))
    )
    beta1_values = float(beta1_center) + float(beta2_value) * (omega - omega_c)

    if getattr(fiber, "_beta1_profile", None) is not None:
        wl_profile = np.asarray(fiber._beta1_profile[0], dtype=float)
    else:
        wl_profile = c / freq_profile
    if wl_profile.size != beta1_values.size:
        wl_profile = c / freq_profile
    fiber._beta1_profile = (wl_profile, beta1_values)


def _dispersion_tag(system: System, freqs: np.ndarray) -> str:
    beta1_grid, beta2_grid = system.beta_grids(freqs=freqs)
    signature = np.ascontiguousarray(
        np.stack([beta1_grid, beta2_grid], axis=0),
        dtype=np.float64,
    ).view(np.uint8)
    return hashlib.sha1(signature).hexdigest()[:12]


def _scan_grids(
    rows: list[dict], beta2_values: list[float], power_dbm_values: list[float]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    b_map = {v: i for i, v in enumerate(beta2_values)}
    p_map = {v: i for i, v in enumerate(power_dbm_values)}
    shape = (len(beta2_values), len(power_dbm_values))
    td_db = np.full(shape, np.nan, dtype=float)
    td_gaussian_db = np.full(shape, np.nan, dtype=float)
    pcfm_db = np.full(shape, np.nan, dtype=float)
    pcfm_xci_db = np.full(shape, np.nan, dtype=float)
    diff_db = np.full(shape, np.nan, dtype=float)
    diff_xci_db = np.full(shape, np.nan, dtype=float)
    for row in rows:
        i = b_map[row["beta2_s2_per_m"]]
        j = p_map[row["launch_dbm"]]
        td_db[i, j] = row["eta_td_db"]
        td_gaussian_db[i, j] = row["eta_td_gaussian_db"]
        pcfm_db[i, j] = row["eta_pcfm_db"]
        pcfm_xci_db[i, j] = row["eta_pcfm_xci_db"]
        diff_db[i, j] = row["delta_db"]
        diff_xci_db[i, j] = row["eta_td_db"] - row["eta_pcfm_xci_db"]
    return td_db, td_gaussian_db, pcfm_db, pcfm_xci_db, diff_db, diff_xci_db


def _plot_heatmaps(
    rows: list[dict],
    beta2_values: list[float],
    power_dbm_values: list[float],
    out_path: Path,
    channel_label: str,
    plot_pcfm_total_and_sci: bool,
) -> None:
    td_db, _, pcfm_db, pcfm_xci_db, diff_db, diff_xci_db = _scan_grids(
        rows,
        beta2_values,
        power_dbm_values,
    )
    extent = [
        float(min(power_dbm_values)),
        float(max(power_dbm_values)),
        float(min(beta2_values)),
        float(max(beta2_values)),
    ]

    panel_height = scale_figsize_to_ieee_column(9.4, 2.9)[1]
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(
            IEEE_DOUBLE_COLUMN_ONE_COLUMN_WIDTH_IN,
            3.0 * panel_height,
        ),
        constrained_layout=True,
    )
    pcfm_plot_db = pcfm_db if plot_pcfm_total_and_sci else pcfm_xci_db
    diff_plot_db = diff_db if plot_pcfm_total_and_sci else diff_xci_db
    pcfm_label = "PCFM" if plot_pcfm_total_and_sci else "PCFM XCI"
    plots = [
        (td_db, f"{channel_label}: TD $10\\log_{{10}}(P_{{NLI}}/P_{{sig}})$ [dB]"),
        (
            pcfm_plot_db,
            f"{channel_label}: {pcfm_label} $10\\log_{{10}}(P_{{NLI}}/P_{{sig}})$ [dB]",
        ),
        (diff_plot_db, f"{channel_label}: TD - {pcfm_label} [dB]"),
    ]
    for ax, (z, title) in zip(np.atleast_1d(axes), plots):
        im = ax.imshow(
            z,
            aspect="auto",
            origin="lower",
            extent=extent,
            interpolation="nearest",
        )
        ax.set_title(title, fontsize=8)
        ax.set_xlabel("Launch power [dBm]")
        ax.set_ylabel(r"$\beta_2$ [s$^2$/m]")
        cbar = fig.colorbar(im, ax=ax, shrink=0.9)
        cbar.ax.tick_params(labelsize=7)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=260)
    plt.close(fig)


def _plot_linecuts(
    rows: list[dict],
    beta2_values: list[float],
    power_dbm_values: list[float],
    out_path: Path,
    channel_label: str,
    plot_pcfm_total_and_sci: bool,
) -> None:
    td_db, td_gaussian_db, pcfm_db, pcfm_xci_db, _, _ = _scan_grids(
        rows,
        beta2_values,
        power_dbm_values,
    )
    pcfm_plot_db = pcfm_db if plot_pcfm_total_and_sci else pcfm_xci_db
    pcfm_label = "PCFM" if plot_pcfm_total_and_sci else "PCFM XCI"
    panel_height = scale_figsize_to_ieee_column(10.2, 3.4)[1]
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(
            IEEE_DOUBLE_COLUMN_ONE_COLUMN_WIDTH_IN,
            2.0 * panel_height,
        ),
        constrained_layout=True,
    )
    ax_p, ax_b = axes

    beta_colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(beta2_values)))
    power_colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(power_dbm_values)))

    for i, beta2_value in enumerate(beta2_values):
        color = beta_colors[i]
        ax_p.plot(
            power_dbm_values,
            td_db[i, :],
            linestyle="-",
            linewidth=1.0,
            marker="o",
            markersize=3,
            markerfacecolor="none",
            markeredgewidth=0.9,
            color=color,
            label=fr"TD, $\beta_2$={beta2_value:.2e}",
        )
        ax_p.plot(
            power_dbm_values,
            td_gaussian_db[i, :],
            linestyle=":",
            linewidth=1.0,
            marker="o",
            markersize=3,
            markerfacecolor="none",
            markeredgewidth=0.9,
            color=color,
            label=fr"TD Gaussian, $\beta_2$={beta2_value:.2e}",
        )
        ax_p.plot(
            power_dbm_values,
            pcfm_plot_db[i, :],
            linestyle="--",
            linewidth=1.0,
            marker="o",
            markersize=3,
            markerfacecolor="none",
            markeredgewidth=0.9,
            color=color,
            label=fr"{pcfm_label}, $\beta_2$={beta2_value:.2e}",
        )
    ax_p.set_title(f"{channel_label}: line cuts vs launch power", fontsize=9)
    ax_p.set_xlabel("Launch power [dBm]")
    ax_p.set_ylabel(r"$\mathrm{NSR}\;[\mathrm{dB}]$")
    ax_p.grid(alpha=0.25)
    ax_p.legend(fontsize=5.5, ncol=1)

    for j, launch_dbm in enumerate(power_dbm_values):
        color = power_colors[j]
        ax_b.plot(
            beta2_values,
            td_db[:, j],
            linestyle="-",
            linewidth=1.0,
            marker="o",
            markersize=3,
            markerfacecolor="none",
            markeredgewidth=0.9,
            color=color,
            label=f"TD, P={launch_dbm:.1f} dBm",
        )
        ax_b.plot(
            beta2_values,
            td_gaussian_db[:, j],
            linestyle=":",
            linewidth=1.0,
            marker="o",
            markersize=3,
            markerfacecolor="none",
            markeredgewidth=0.9,
            color=color,
            label=f"TD Gaussian, P={launch_dbm:.1f} dBm",
        )
        ax_b.plot(
            beta2_values,
            pcfm_plot_db[:, j],
            linestyle="--",
            linewidth=1.0,
            marker="o",
            markersize=3,
            markerfacecolor="none",
            markeredgewidth=0.9,
            color=color,
            label=f"{pcfm_label}, P={launch_dbm:.1f} dBm",
        )
    ax_b.set_title(fr"{channel_label}: line cuts vs $\beta_2$", fontsize=9)
    ax_b.set_xlabel(r"$\beta_2$ [s$^2$/m]")
    ax_b.set_ylabel(r"$\mathrm{NSR}\;[\mathrm{dB}]$")
    ax_b.grid(alpha=0.25)
    ax_b.legend(fontsize=5.5, ncol=1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=260)
    plt.close(fig)


def _channel_targets(freqs: np.ndarray) -> list[tuple[str, int]]:
    n_channels = int(freqs.size)
    center_idx = n_channels // 2
    edge_idx = n_channels - 1
    targets = [("center", center_idx)]
    if edge_idx != center_idx:
        targets.append(("edge_upper", edge_idx))
    return targets


def run_scan(
    cfg_path: Path,
    beta2_values: list[float],
    power_dbm_values: list[float],
    pcfm_numeric_xci: bool,
    recompute_td: bool,
    recompute_pcfm: bool,
    out_dir: Path,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    base_system = System.from_toml(cfg_path)
    runtime_cfg = _load_pcfm_runtime_config(base_system)
    plot_pcfm_total_and_sci = bool(runtime_cfg["plot_pcfm_total_and_sci"])

    for beta2_value in beta2_values:
        system = System.from_toml(cfg_path)
        _set_constant_dispersion(system, beta2_value)
        freqs = system.wdm.frequency_grid()
        n_channels = freqs.size
        targets = _channel_targets(freqs)
        disp_tag = _dispersion_tag(system, freqs)

        lg.info(
            "Running beta2={:.3e} s^2/m (disp tag {}).".format(
                beta2_value, disp_tag
            )
        )

        for launch_dbm in power_dbm_values:
            launch_w = float(dBm2watt(launch_dbm))
            launch_vec = np.full(n_channels, launch_w, dtype=float)
            beta_tag = _safe_tag(beta2_value)
            p_tag = _safe_tag(launch_dbm)
            profile_path = out_dir / f"flat_profile_b{beta_tag}_p{p_tag}.npy"
            _write_flat_profile(profile_path, system, launch_powers_w=launch_vec)
            profile_power_tag = _power_profile_hash(system, profile_path)

            signal_power = _resolve_signal_power(system, profile_path, launch_vec)

            ccfs = collision_coeffs_system_uwb(
                system,
                ipulse=1,
                recompute=recompute_td,
                profile_path=profile_path,
            )
            td_cache = _nlin_cache_path(
                profile_path=profile_path,
                use_kappa=True,
                use_x_mode=True,
                extra_tag=f"disp{disp_tag}_p{p_tag}_prof{profile_power_tag}",
            )
            nlin_td = total_nlin_uwb(
                system,
                ccfs,
                use_kappa=True,
                use_x_mode=True,
                launch_powers_w=launch_vec,
                cache_path=td_cache,
                recompute=recompute_td,
            )
            td_vec_total = np.asarray(nlin_td, dtype=float).reshape(-1)
            const_pref, sum_a, sum_b = _td_modulation_components(
                system,
                ccfs,
                launch_vec,
                use_kappa=True,
                use_x_mode=True,
            )
            mu0_64 = qam_mu0(64)
            mu0_gaussian = gaussian_mu0()
            td_vec_qam64 = np.asarray(
                const_pref * (mu0_64 * sum_a + sum_b),
                dtype=float,
            ).reshape(-1)
            td_vec_gaussian = np.asarray(
                const_pref * (mu0_gaussian * sum_a + sum_b),
                dtype=float,
            ).reshape(-1)
            denom = np.maximum(np.abs(td_vec_total), 1e-30)
            rel_err = float(np.max(np.abs(td_vec_qam64 - td_vec_total) / denom))
            if rel_err > 1e-9:
                lg.warning(
                    "TD consistency check (QAM64 decomposition vs total_nlin_uwb) max rel err={:.3e}".format(
                        rel_err
                    )
                )
            pcfm_cfg = PcfmConfig(
                degree=9,
                include_mci=False,
                use_numeric_sci=True,
                use_numeric_xci=bool(pcfm_numeric_xci),
            )
            pcfm_path = out_dir / f"pcfm_b{beta_tag}_p{p_tag}_disp{disp_tag}_prof{profile_power_tag}.npy"
            nlin_pcfm, _, nlin_pcfm_xci = _load_or_compute_pcfm_I(
                system=system,
                profile_path=profile_path,
                launch_powers_w=launch_vec,
                output_path=pcfm_path,
                cfg=pcfm_cfg,
                recompute=recompute_pcfm,
                return_components=True,
            )
            pcfm_vec = np.asarray(nlin_pcfm, dtype=float).reshape(-1)
            pcfm_xci_vec = np.asarray(nlin_pcfm_xci, dtype=float).reshape(-1)
            for channel_label, channel_idx in targets:
                channel_freq_thz = float(freqs[channel_idx] * 1e-12)
                signal_ch = float(signal_power[channel_idx])
                td_ch = float(td_vec_qam64[channel_idx])
                td_gaussian_ch = float(td_vec_gaussian[channel_idx])
                pcfm_ch = float(pcfm_vec[channel_idx])
                pcfm_xci_ch = float(pcfm_xci_vec[channel_idx])

                eta_td = td_ch / max(signal_ch, 1e-30)
                eta_td_gaussian = td_gaussian_ch / max(signal_ch, 1e-30)
                eta_pcfm = pcfm_ch / max(signal_ch, 1e-30)
                eta_pcfm_xci = pcfm_xci_ch / max(signal_ch, 1e-30)
                delta_db = 10.0 * np.log10(max(eta_td, 1e-30)) - 10.0 * np.log10(max(eta_pcfm, 1e-30))

                rows.append(
                    {
                        "channel_label": channel_label,
                        "channel_idx": int(channel_idx),
                        "channel_freq_thz": channel_freq_thz,
                        "beta2_s2_per_m": float(beta2_value),
                        "launch_dbm": float(launch_dbm),
                        "launch_w": launch_w,
                        "signal_channel_w": signal_ch,
                        "td_channel_w": td_ch,
                        "td_gaussian_channel_w": td_gaussian_ch,
                        "pcfm_channel_w": pcfm_ch,
                        "pcfm_xci_channel_w": pcfm_xci_ch,
                        "eta_td": eta_td,
                        "eta_td_gaussian": eta_td_gaussian,
                        "eta_pcfm": eta_pcfm,
                        "eta_pcfm_xci": eta_pcfm_xci,
                        "eta_td_db": 10.0 * np.log10(max(eta_td, 1e-30)),
                        "eta_td_gaussian_db": 10.0 * np.log10(max(eta_td_gaussian, 1e-30)),
                        "eta_pcfm_db": 10.0 * np.log10(max(eta_pcfm, 1e-30)),
                        "eta_pcfm_xci_db": 10.0 * np.log10(max(eta_pcfm_xci, 1e-30)),
                        "delta_db": delta_db,
                        "ratio_td_over_pcfm": td_ch / max(pcfm_ch, 1e-30),
                        "dispersion_tag": disp_tag,
                    }
                )

                lg.info(
                    "b2={:.3e}, P={:.2f} dBm, {}(idx={}) -> TD(64QAM)={:.3e} W, TD(Gauss)={:.3e} W, "
                    "PCFM={:.3e} W, Δ={:.3f} dB".format(
                        beta2_value,
                        launch_dbm,
                        channel_label,
                        channel_idx,
                        td_ch,
                        td_gaussian_ch,
                        pcfm_ch,
                        delta_db,
                    )
                )

    csv_path = out_dir / "selected_channels_td_pcfm_scan.csv"
    if rows:
        fieldnames = list(rows[0].keys())
        with csv_path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        lg.success(f"Saved scan results to {csv_path}")
        channel_labels = sorted({str(row["channel_label"]) for row in rows})
        for channel_label in channel_labels:
            rows_ch = [row for row in rows if row["channel_label"] == channel_label]
            heatmap_path = out_dir / f"single_channel_td_pcfm_scan_heatmaps_{channel_label}.png"
            _plot_heatmaps(
                rows_ch,
                beta2_values=beta2_values,
                power_dbm_values=power_dbm_values,
                out_path=heatmap_path,
                channel_label=channel_label,
                plot_pcfm_total_and_sci=plot_pcfm_total_and_sci,
            )
            lg.success(f"Saved scan heatmaps to {heatmap_path}")
            linecuts_path = out_dir / f"single_channel_td_pcfm_scan_linecuts_{channel_label}.png"
            _plot_linecuts(
                rows_ch,
                beta2_values=beta2_values,
                power_dbm_values=power_dbm_values,
                out_path=linecuts_path,
                channel_label=channel_label,
                plot_pcfm_total_and_sci=plot_pcfm_total_and_sci,
            )
            lg.success(f"Saved scan line cuts to {linecuts_path}")
    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Debug scan for center/edge-channel TD vs PCFM NLI using the same "
            "functions as the PCFM workflow."
        )
    )
    parser.add_argument(
        "--config",
        type=str,
        default="./input/pcfm_struct.toml",
        help="Base system TOML.",
    )
    parser.add_argument(
        "--beta2-values",
        type=str,
        default="1e-27,2e-27,3e-27,4e-27",
        help="Comma-separated beta2 values [s^2/m].",
    )
    parser.add_argument(
        "--power-dbms",
        type=str,
        default="-10,-8,-6,-4,-2,0",
        help="Comma-separated launch powers [dBm], applied uniformly on all channels.",
    )
    parser.add_argument(
        "--pcfm-numeric-xci",
        action="store_true",
        help="Use numeric XCI in PCFM (default: closed-form XCI).",
    )
    parser.add_argument(
        "--recompute-td",
        action="store_true",
        help="Force TD collision/NLIN recomputation.",
    )
    parser.add_argument(
        "--recompute-pcfm",
        action="store_true",
        help="Force PCFM recomputation.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="results/debug/pcfm_single_channel_scan",
        help="Output directory for generated profiles, caches, and reports.",
    )
    argv = _normalize_negative_csv_option(sys.argv[1:], "--power-dbms")
    args = parser.parse_args(argv)

    cfg_path = Path(args.config)
    beta2_values = _parse_float_list(args.beta2_values)
    power_dbm_values = _parse_float_list(args.power_dbms)
    out_dir = Path(args.out_dir)

    run_scan(
        cfg_path=cfg_path,
        beta2_values=beta2_values,
        power_dbm_values=power_dbm_values,
        pcfm_numeric_xci=bool(args.pcfm_numeric_xci),
        recompute_td=bool(args.recompute_td),
        recompute_pcfm=bool(args.recompute_pcfm),
        out_dir=out_dir,
    )


if __name__ == "__main__":
    main()
