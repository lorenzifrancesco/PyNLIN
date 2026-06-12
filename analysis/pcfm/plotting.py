from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger as lg
from matplotlib.ticker import ScalarFormatter

from pynlin.nlin import pcfm_gn as pcfm
from pynlin.nlin.pcfm_gn import PcfmConfig
from pynlin.nlin.pcfm_gn import fit_spp_polynomials, load_signal_profiles, normalize_spp
from pynlin.system import System
from pynlin.utils import watt2dBm

from .figure_size import scale_figsize_to_ieee_column

GNUPLOT_RED = "#C00000"
GNUPLOT_GREEN = "#008A2E"
GNUPLOT_BLUE = "#0057D8"
GNUPLOT_ORANGE = "#D97706"
GNUPLOT_GRAY = "#202020"
GNUPLOT_MAGENTA = "#B000B0"
MARKER_TD = "o"
MARKER_TD_BY_LABEL = {
    "16-QAM": "s",
    "256-QAM": "^",
    "Gaussian": "X",
}
MARKER_PCFM = "^"
MARKER_PCFM_XCI = "s"
MARKER_PCFM_XCI_EQ18 = "D"
MARKER_GN = "x"
MARKER_GN_XCI = "+"
MARKER_GN_DIRECT = "v"
MARKER_GN_DIRECT_XCI = "P"
NLIN_MARKER_SIZE = 2.3
NLIN_MARKER_SIZE_KXCI = 3.2
NLIN_SCATTER_SIZE = 26
NLIN_MARKER_EDGE_WIDTH = 0.25
SAVEFIG_PAD_INCHES = 0.04
AXIS_LABEL_SIZE = 9.5
LEGEND_SIZE = 8.5
NLIN_POWER_XLIM_THZ = (184.0, 204.0)

plt.rcParams["xtick.labelsize"] = 8
plt.rcParams["ytick.labelsize"] = 8


def _disable_dbm_axis_grouping(ax: plt.Axes, which: str = "y") -> None:
    """Force plain tick labels on dBm axes without offset/scientific grouping."""
    axis = ax.yaxis if which == "y" else ax.xaxis
    formatter = ScalarFormatter(useOffset=False)
    formatter.set_scientific(False)
    axis.set_major_formatter(formatter)
    axis.offsetText.set_visible(False)


def _save_figure(fig: plt.Figure, out_path: Path, *, dpi: int = 300) -> None:
    """Save figure with explicit padding to avoid clipped borders."""
    fig.tight_layout(pad=0.45)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=SAVEFIG_PAD_INCHES)


def _ordered_legend(ax: plt.Axes, labels: list[str], **kwargs) -> None:
    handles, handle_labels = ax.get_legend_handles_labels()
    by_label = dict(zip(handle_labels, handles))
    ordered_labels = [label for label in labels if label in by_label]
    ordered_labels.extend(label for label in handle_labels if label not in ordered_labels)
    ax.legend([by_label[label] for label in ordered_labels], ordered_labels, **kwargs)


def _is_eq18_label(label: str) -> bool:
    return "eq18" in str(label).lower()


def _set_freq_xlim_thz(ax: plt.Axes) -> None:
    """Apply fixed frequency axis limits used across PCFM NLIN plots."""
    ax.set_xlim(*NLIN_POWER_XLIM_THZ)


def _safe_histogram_bins(values: np.ndarray, n_bins: int, *, log_scale: bool) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("Histogram bin builder received no finite values.")

    vmin = float(np.min(values))
    vmax = float(np.max(values))
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        raise ValueError("Histogram bin builder received non-finite extrema.")

    if np.isclose(vmin, vmax):
        if log_scale:
            lower = vmin / 1.05
            upper = vmax * 1.05
            if lower <= 0.0:
                lower = np.nextafter(vmin, 0.0)
            if not upper > lower:
                upper = np.nextafter(vmax, np.inf)
            return np.geomspace(lower, upper, max(int(n_bins), 3))
        half_width = max(abs(vmin) * 0.05, 1e-12)
        return np.linspace(vmin - half_width, vmax + half_width, max(int(n_bins), 3))

    if log_scale:
        bins = np.geomspace(vmin, vmax, int(n_bins))
    else:
        bins = np.linspace(vmin, vmax, int(n_bins))
    if not np.all(np.diff(bins) > 0.0):
        bins = np.linspace(vmin, np.nextafter(vmax, np.inf), max(int(n_bins), 3))
    return bins


def plot_pcfm_gsnr(
    freqs_hz: np.ndarray,
    gsnr_td: np.ndarray,
    gsnr_pcfm: dict[str, np.ndarray],
    gsnr_gn: dict[str, np.ndarray] | None,
    out_path: Path,
    gsnr_gn_direct: dict[str, np.ndarray] | None = None,
    title: str | None = None,
    plot_pcfm_total_and_sci: bool = False,
) -> None:
    """Plot GSNR_NLI overlays with the same style as existing noise plots."""
    dpi = 300
    marker_lw = 0.45
    fig, ax = plt.subplots(figsize=scale_figsize_to_ieee_column(3.6, 2.8))
    if title:
        ax.set_title(title, fontsize=9)
    ax.plot(
        freqs_hz * 1e-12,
        gsnr_td,
        color=GNUPLOT_RED,
        lw=0.45,
        marker="o",
        markersize=1.2,
        markerfacecolor="none",
        markeredgewidth=marker_lw,
        label="TD",
    )

    if plot_pcfm_total_and_sci:
        for label, gsnr in gsnr_pcfm.items():
            display = "" if label == "no_loss" else label
            suffix = f" {display}" if display else ""
            ax.plot(
                freqs_hz * 1e-12,
                gsnr,
                color=GNUPLOT_ORANGE,
                lw=0.45,
                marker="o",
                markersize=1.2,
                markerfacecolor="none",
                markeredgewidth=marker_lw,
                label=f"PCFM{suffix}",
            )

    if gsnr_gn:
        for label, gsnr in gsnr_gn.items():
            display = "" if label == "no_loss" else label
            suffix = f" {display}" if display else ""
            ax.scatter(
                freqs_hz * 1e-12,
                gsnr,
                s=8,
                marker="o",
                facecolors="none",
                edgecolors=GNUPLOT_GRAY,
                linewidths=marker_lw,
                label=f"GN{suffix}",
            )

    if gsnr_gn_direct:
        for label, gsnr in gsnr_gn_direct.items():
            display = "" if label == "no_loss" else label
            suffix = f" {display}" if display else ""
            ax.scatter(
                freqs_hz * 1e-12,
                gsnr,
                s=8,
                marker="o",
                facecolors="none",
                edgecolors=GNUPLOT_MAGENTA,
                linewidths=marker_lw,
                label=f"GN dir{suffix}",
            )

    ax.set_xlabel(r"$\mathnormal{f \; [\mathrm{THz}]}$", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel(r"$\mathnormal{GSNR_{NLI} \; [\mathrm{dB}]}$", fontsize=AXIS_LABEL_SIZE)
    _set_freq_xlim_thz(ax)
    ax.grid(False)
    ax.legend(loc="best", fontsize=LEGEND_SIZE)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _save_figure(fig, out_path, dpi=dpi)
    lg.success(f"Saved GSNR plot to {out_path}")


def plot_pcfm_nlin_power(
    freqs_hz: np.ndarray,
    signal_power_w: np.ndarray,
    nlin_td_w: np.ndarray,
    nlin_pcfm_w: dict[str, np.ndarray],
    nlin_gn_w: dict[str, np.ndarray] | None,
    out_path: Path,
    nlin_td_mod_w: dict[str, np.ndarray] | None = None,
    nlin_pcfm_xci_w: dict[str, np.ndarray] | None = None,
    nlin_gn_xci_w: dict[str, np.ndarray] | None = None,
    nlin_gn_direct_w: dict[str, np.ndarray] | None = None,
    nlin_gn_direct_xci_w: dict[str, np.ndarray] | None = None,
    gn_direct_is_ratio: bool = False,
    gn_direct_xci_is_ratio: bool = False,
    plot_pcfm_total_and_sci: bool = False,
) -> None:
    """Plot NLIN absolute power (dBm) and normalized-to-output power (dB)."""
    dpi = 300
    marker_lw = NLIN_MARKER_EDGE_WIDTH
    signal_power_w = np.asarray(signal_power_w, dtype=float).reshape(-1)
    if signal_power_w.size != freqs_hz.size:
        raise ValueError(
            f"signal_power_w size {signal_power_w.size} != freq size {freqs_hz.size}"
        )

    def _as_flat(nlin: np.ndarray) -> np.ndarray:
        arr = np.asarray(nlin, dtype=float).reshape(-1)
        if arr.size != signal_power_w.size:
            raise ValueError(f"NLIN size {arr.size} != signal power size {signal_power_w.size}")
        return arr

    def _power_dbm(nlin: np.ndarray, already_ratio: bool = False) -> np.ndarray:
        if already_ratio:
            raise ValueError("plot_pcfm_nlin_power expects absolute powers in W, not ratios.")
        arr = _as_flat(nlin)
        return watt2dBm(np.maximum(arr, 1e-18))

    def _over_pout_db(nlin: np.ndarray, already_ratio: bool = False) -> np.ndarray:
        arr = _as_flat(nlin)
        if already_ratio:
            ratio = arr
        else:
            ratio = arr / np.maximum(signal_power_w, 1e-18)
        return 10.0 * np.log10(np.maximum(ratio, 1e-18))

    def _plot_metric(
        metric_fn,
        *,
        ylabel: str,
        target_path: Path,
        success_msg: str,
    ) -> None:
        fig, ax = plt.subplots(figsize=scale_figsize_to_ieee_column(3.6, 2.8))

        if nlin_td_mod_w:
            for label, nlin in nlin_td_mod_w.items():
                if str(label).strip() == "64-QAM":
                    continue
                values = metric_fn(nlin)
                label_str = str(label).strip()
                marker = MARKER_TD_BY_LABEL.get(label_str, MARKER_TD)
                color = GNUPLOT_RED if label_str.lower() == "gaussian" else "black"
                markerfacecolor = GNUPLOT_RED if label_str.lower() == "gaussian" else "none"
                ax.plot(
                    freqs_hz * 1e-12,
                    values,
                    color=color,
                    lw=0.0,
                    ls="None",
                    marker=marker,
                    markersize=NLIN_MARKER_SIZE,
                    markerfacecolor=markerfacecolor,
                    markeredgewidth=marker_lw,
                    label=f"TD {label}",
                )
        else:
            values = metric_fn(nlin_td_w)
            ax.plot(
                freqs_hz * 1e-12,
                values,
                color=GNUPLOT_RED,
                lw=0.0,
                ls="None",
                marker=MARKER_TD,
                markersize=NLIN_MARKER_SIZE,
                markerfacecolor=GNUPLOT_RED,
                markeredgewidth=marker_lw,
                label="TD",
            )

        if plot_pcfm_total_and_sci:
            for label, nlin in nlin_pcfm_w.items():
                display = "" if label == "no_loss" else label
                suffix = f" {display}" if display else ""
                values = metric_fn(nlin)
                ax.plot(
                    freqs_hz * 1e-12,
                    values,
                    color=GNUPLOT_ORANGE,
                    lw=0.0,
                    ls="None",
                    marker=MARKER_PCFM,
                    markersize=NLIN_MARKER_SIZE,
                    markerfacecolor=GNUPLOT_ORANGE,
                    markeredgewidth=marker_lw,
                    label=f"PCFM{suffix}",
                )

        if nlin_pcfm_xci_w:
            for label, nlin in nlin_pcfm_xci_w.items():
                is_eq18 = _is_eq18_label(label)
                values = metric_fn(nlin)
                ax.plot(
                    freqs_hz * 1e-12,
                    values,
                    color=GNUPLOT_BLUE if is_eq18 else GNUPLOT_GREEN,
                    lw=0.0,
                    ls="None",
                    marker=MARKER_PCFM_XCI_EQ18 if is_eq18 else MARKER_PCFM_XCI,
                    markersize=NLIN_MARKER_SIZE if is_eq18 else NLIN_MARKER_SIZE_KXCI,
                    markerfacecolor="white" if is_eq18 else GNUPLOT_GREEN,
                    markeredgewidth=marker_lw,
                    label=f"PCFM {'II' if is_eq18 else 'I'}",
                )

        if nlin_gn_w:
            for label, nlin in nlin_gn_w.items():
                display = "" if label == "no_loss" else label
                suffix = f" {display}" if display else ""
                values = metric_fn(nlin)
                ax.scatter(
                    freqs_hz * 1e-12,
                    values,
                    s=NLIN_SCATTER_SIZE,
                    marker=MARKER_GN,
                    facecolors=GNUPLOT_GRAY,
                    edgecolors=GNUPLOT_GRAY,
                    linewidths=marker_lw,
                    label=f"GN{suffix}",
                )

        if nlin_gn_direct_w:
            for label, nlin in nlin_gn_direct_w.items():
                display = "" if label == "no_loss" else label
                suffix = f" {display}" if display else ""
                values = metric_fn(nlin, already_ratio=gn_direct_is_ratio)
                ax.scatter(
                    freqs_hz * 1e-12,
                    values,
                    s=NLIN_SCATTER_SIZE,
                    marker=MARKER_GN_DIRECT,
                    facecolors=GNUPLOT_MAGENTA,
                    edgecolors=GNUPLOT_MAGENTA,
                    linewidths=marker_lw,
                    label=f"GN dir{suffix}",
                )

        if nlin_gn_xci_w:
            for label, nlin in nlin_gn_xci_w.items():
                display = "" if label == "no_loss" else label
                suffix = f" {display}" if display else ""
                values = metric_fn(nlin)
                ax.scatter(
                    freqs_hz * 1e-12,
                    values,
                    s=NLIN_SCATTER_SIZE,
                    marker=MARKER_GN_XCI,
                    facecolors="none",
                    edgecolors=GNUPLOT_GRAY,
                    linewidths=marker_lw,
                    label=f"GN XCI{suffix}",
                )

        if nlin_gn_direct_xci_w:
            for label, nlin in nlin_gn_direct_xci_w.items():
                display = "" if label == "no_loss" else label
                suffix = f" {display}" if display else ""
                values = metric_fn(nlin, already_ratio=gn_direct_xci_is_ratio)
                ax.scatter(
                    freqs_hz * 1e-12,
                    values,
                    s=NLIN_SCATTER_SIZE,
                    marker=MARKER_GN_DIRECT_XCI,
                    facecolors="none",
                    edgecolors=GNUPLOT_MAGENTA,
                    linewidths=marker_lw,
                    label=f"GN dir XCI{suffix}",
                )

        ax.set_xlabel(r"$\mathnormal{f \; [\mathrm{THz}]}$", fontsize=AXIS_LABEL_SIZE)
        ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_SIZE)
        _set_freq_xlim_thz(ax)
        _disable_dbm_axis_grouping(ax)
        ax.grid(False)
        _ordered_legend(
            ax,
            ["PCFM I", "PCFM II", "TD Gaussian", "TD 16-QAM", "TD 256-QAM"],
            loc="best",
            fontsize=LEGEND_SIZE,
        )
        target_path.parent.mkdir(parents=True, exist_ok=True)
        _save_figure(fig, target_path, dpi=dpi)
        lg.success(success_msg.format(path=target_path))
        plt.close(fig)

    _plot_metric(
        _power_dbm,
        ylabel=r"$P_{NLI}\;[\mathrm{dBm}]$",
        target_path=out_path,
        success_msg="Saved NLIN power plot to {path}",
    )
    _plot_metric(
        _over_pout_db,
        ylabel=r"$\mathrm{NSR}\;[\mathrm{dB}]$",
        target_path=out_path.with_name(f"{out_path.stem}_over_pout_db{out_path.suffix}"),
        success_msg="Saved normalized NLIN plot to {path}",
    )


def plot_pcfm_diagnostics(
    system: System,
    profile_path: Path | str,
    launch_powers_w: np.ndarray,
    out_dir: Path,
    cfg: PcfmConfig | None = None,
) -> None:
    """Generate diagnostic plots for intermediate quantities."""
    cfg = cfg or PcfmConfig()
    out_dir.mkdir(parents=True, exist_ok=True)
    freqs = system.wdm.frequency_grid()
    freqs_thz = freqs * 1e-12
    launch_dbm = watt2dBm(np.maximum(launch_powers_w, 1e-18))

    fig, ax = plt.subplots(figsize=scale_figsize_to_ieee_column(3.6, 2.4))
    ax.plot(freqs_thz, launch_dbm, lw=0.8, color="black")
    ax.set_xlabel(r"$\mathnormal{f \; [\mathrm{THz}]}$", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel(r"$\mathnormal{P_\mathrm{launch}\;[\mathrm{dBm}]}$", fontsize=AXIS_LABEL_SIZE)
    _set_freq_xlim_thz(ax)
    _disable_dbm_axis_grouping(ax)
    ax.grid(False)
    _save_figure(fig, out_dir / "launch_power.pdf", dpi=300)
    lg.success(f"Saved launch power plot to {out_dir / 'launch_power.pdf'}")
    plt.close(fig)

    beta1, beta2 = system.beta_grids(freqs=freqs)
    beta1 = np.asarray(beta1, dtype=float)
    beta2 = np.asarray(beta2, dtype=float)
    if beta1.ndim == 2:
        if beta1.shape[0] != 1:
            raise ValueError(
                "plot_pcfm_diagnostics currently expects a single-mode system for the L/LW histogram."
            )
        beta1 = beta1[0]
    if beta2.ndim == 2:
        if beta2.shape[0] != 1:
            raise ValueError(
                "plot_pcfm_diagnostics currently expects a single-mode system for the L/LD histogram."
            )
        beta2 = beta2[0]

    length_m = float(system.fiber_length)
    baud_rate_hz = float(system.pulse.baud_rate)

    beta1_diff = np.abs(beta1[:, None] - beta1[None, :])
    pair_mask = np.triu(np.ones(beta1_diff.shape, dtype=bool), k=1)
    l_over_lw_pairs = (beta1_diff[pair_mask] * length_m * baud_rate_hz).reshape(-1)
    l_over_lw_pairs = l_over_lw_pairs[np.isfinite(l_over_lw_pairs) & (l_over_lw_pairs > 0.0)]

    fig, ax = plt.subplots(figsize=scale_figsize_to_ieee_column(3.6, 2.6))
    if l_over_lw_pairs.size:
        bins = _safe_histogram_bins(l_over_lw_pairs, 40, log_scale=True)
        ax.hist(l_over_lw_pairs, bins=bins, color="tab:blue", edgecolor="black", linewidth=0.5)
        ax.set_xscale("log")
    ax.set_xlabel(r"$\mathnormal{L/L_W}$", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel("channel-pair count", fontsize=AXIS_LABEL_SIZE)
    ax.grid(False)
    _save_figure(fig, out_dir / "l_over_lw_histogram.pdf", dpi=300)
    lg.success(f"Saved L/LW histogram to {out_dir / 'l_over_lw_histogram.pdf'}")
    plt.close(fig)

    if cfg.use_beta2_eff:
        coeffs_beta = pcfm._beta_coeffs_from_profile(
            system,
            float(system.center_frequency or np.mean(freqs)),
        )
        if coeffs_beta is not None:
            beta2_diag = np.array([pcfm._beta2_eff(float(f), float(f), coeffs_beta) for f in freqs], dtype=float)
        else:
            beta2_diag = beta2
    else:
        beta2_diag = beta2
    l_over_ld = np.abs(beta2_diag) * length_m * (baud_rate_hz ** 2)
    l_over_ld = l_over_ld[np.isfinite(l_over_ld) & (l_over_ld > 0.0)]

    fig, ax = plt.subplots(figsize=scale_figsize_to_ieee_column(3.6, 2.6))
    if l_over_ld.size:
        bins = _safe_histogram_bins(l_over_ld, 30, log_scale=True)
        ax.hist(l_over_ld, bins=bins, color="tab:orange", edgecolor="black", linewidth=0.5)
        ax.set_xscale("log")
    ax.set_xlabel(r"$\mathnormal{L/L_D}$", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel("channel count", fontsize=AXIS_LABEL_SIZE)
    ax.grid(False)
    _save_figure(fig, out_dir / "l_over_ld_histogram.pdf", dpi=300)
    lg.success(f"Saved L/LD histogram to {out_dir / 'l_over_ld_histogram.pdf'}")
    plt.close(fig)

    sig_ch_z, z_axis = load_signal_profiles(profile_path, system)
    span = float(z_axis[-1] - z_axis[0]) if z_axis.size else 0.0
    avg_power = np.trapezoid(sig_ch_z, z_axis, axis=1) / max(span, 1.0)
    out_power = sig_ch_z[:, -1]
    avg_dbm = watt2dBm(np.maximum(avg_power, 1e-18))
    out_dbm = watt2dBm(np.maximum(out_power, 1e-18))

    fig, ax = plt.subplots(figsize=scale_figsize_to_ieee_column(3.6, 2.4))
    ax.plot(freqs_thz, avg_dbm, lw=0.8, color="tab:blue", label="avg")
    ax.plot(freqs_thz, out_dbm, lw=0.8, color="tab:orange", label="out")
    ax.set_xlabel(r"$\mathnormal{f \; [\mathrm{THz}]}$", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel(r"$\mathnormal{P\;[\mathrm{dBm}]}$", fontsize=AXIS_LABEL_SIZE)
    _set_freq_xlim_thz(ax)
    _disable_dbm_axis_grouping(ax)
    ax.grid(False)
    ax.legend(loc="best", fontsize=LEGEND_SIZE)
    _save_figure(fig, out_dir / "profile_power.pdf", dpi=300)
    lg.success(f"Saved profile power plot to {out_dir / 'profile_power.pdf'}")
    plt.close(fig)

    spp = normalize_spp(sig_ch_z, z_axis)
    coeffs = fit_spp_polynomials(z_axis, spp, degree=9)
    length = float(z_axis[-1] - z_axis[0])
    z_norm = (z_axis - z_axis[0]) / length

    picks = []
    labels = []
    band_slices = (
        getattr(system.wdm, "_band_slices", {}) if hasattr(system.wdm, "_band_slices") else {}
    )
    if band_slices:
        for band_name in ("L", "C", "S"):
            for key, slc in band_slices.items():
                if str(key).lower().startswith(band_name.lower()):
                    idx = int((slc.start + slc.stop - 1) // 2)
                    picks.append(idx)
                    labels.append(f"Ch {idx}: center-{band_name}")
                    break
    if not picks:
        n_ch = spp.shape[0]
        picks = [n_ch // 6, n_ch // 2, 5 * n_ch // 6]
        labels = [f"Ch {p}" for p in picks]

    fig, ax = plt.subplots(figsize=scale_figsize_to_ieee_column(4.0, 2.8))
    first = True
    for idx, label in zip(picks, labels):
        p_fit = np.polynomial.polynomial.polyval(z_norm, coeffs[idx])
        ax.plot(
            z_axis / 1e3,
            spp[idx],
            color="tab:blue",
            lw=0.9,
            label="exact" if first else None,
        )
        ax.plot(
            z_axis / 1e3,
            p_fit,
            color="tab:red",
            lw=0.9,
            ls="--",
            label="polynomial Np=9" if first else None,
        )
        z_anno = z_axis[int(0.4 * (len(z_axis) - 1))] / 1e3
        y_anno = np.interp(z_anno * 1e3, z_axis, spp[idx])
        ax.annotate(
            label,
            xy=(z_anno, y_anno),
            xytext=(z_anno + 5, y_anno + 0.08),
            textcoords="data",
            fontsize=7,
            arrowprops=dict(arrowstyle="->", lw=0.6),
        )
        first = False
    ax.set_xlabel(r"$\mathnormal{z\;[\mathrm{km}]}$", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel(r"normalized power", fontsize=AXIS_LABEL_SIZE)
    ax.grid(False)
    ax.legend(loc="best", fontsize=LEGEND_SIZE)
    _save_figure(fig, out_dir / "spp_fit.pdf", dpi=300)
    lg.success(f"Saved SPP fit plot to {out_dir / 'spp_fit.pdf'}")
    plt.close(fig)

    p_l = np.array([np.polynomial.polynomial.polyval(1.0, coeff) for coeff in coeffs], dtype=float)
    poly_sum = np.array(
        [
            np.sum(np.convolve(coeff, coeff) / (np.arange(len(coeff) * 2 - 1) + 1.0))
            for coeff in coeffs
        ],
        dtype=float,
    )
    fig, ax1 = plt.subplots(figsize=scale_figsize_to_ieee_column(3.6, 2.4))
    ax1.plot(freqs_thz, p_l, lw=0.8, color="tab:blue")
    ax1.set_xlabel(r"$\mathnormal{f \; [\mathrm{THz}]}$", fontsize=AXIS_LABEL_SIZE)
    ax1.set_ylabel(r"$\mathnormal{p(L)}$", color="tab:blue", fontsize=AXIS_LABEL_SIZE)
    _set_freq_xlim_thz(ax1)
    ax2 = ax1.twinx()
    ax2.plot(freqs_thz, poly_sum, lw=0.8, color="tab:orange")
    ax2.set_ylabel(
        r"$\mathnormal{\sum a_n a_k/(n+k+1)}$",
        color="tab:orange",
        fontsize=AXIS_LABEL_SIZE,
    )
    ax1.grid(False)
    _save_figure(fig, out_dir / "pcfm_terms.pdf", dpi=300)
    lg.success(f"Saved PCFM terms plot to {out_dir / 'pcfm_terms.pdf'}")
    plt.close(fig)

    pump_specs = system.pump_specs or []
    if pump_specs:
        pump_freqs_thz = np.array([3e8 / p.wavelength for p in pump_specs], dtype=float) * 1e-12
        pump_powers_dbm = np.array([p.power_dbm for p in pump_specs], dtype=float)
    else:
        pump_freqs_thz = np.array([])
        pump_powers_dbm = np.array([])
    fig, ax = plt.subplots(figsize=scale_figsize_to_ieee_column(3.6, 2.4))
    ax.scatter(freqs_thz, launch_dbm, s=6, alpha=0.7, label="signals (launch)")
    ax.scatter(
        freqs_thz,
        out_dbm,
        s=6,
        facecolors="none",
        edgecolors="tab:blue",
        linewidths=0.6,
        label="signals (out)",
    )
    if pump_powers_dbm.size:
        ax.scatter(
            pump_freqs_thz,
            pump_powers_dbm,
            marker="x",
            s=16,
            color="tab:red",
            label="pumps",
        )
    ax.set_xlabel(r"$\mathnormal{f \; [\mathrm{THz}]}$", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel(r"$\mathnormal{P\;[\mathrm{dBm}]}$", fontsize=AXIS_LABEL_SIZE)
    _set_freq_xlim_thz(ax)
    _disable_dbm_axis_grouping(ax)
    ax.grid(False)
    ax.legend(loc="best", fontsize=LEGEND_SIZE)
    _save_figure(fig, out_dir / "launch_spectrum.pdf", dpi=300)
    lg.success(f"Saved launch spectrum plot to {out_dir / 'launch_spectrum.pdf'}")
    plt.close(fig)

    wl = 3e8 / freqs
    beta2 = np.array([system.fiber.beta2_at(float(w)) for w in wl], dtype=float)
    aeff = np.array([system.fiber.effective_area_at(float(w)) for w in wl], dtype=float)
    fig, ax1 = plt.subplots(figsize=scale_figsize_to_ieee_column(3.6, 2.4))
    ax1.plot(freqs_thz, beta2 * 1e24, lw=0.8, color="tab:blue")
    ax1.set_xlabel(r"$\mathnormal{f \; [\mathrm{THz}]}$", fontsize=AXIS_LABEL_SIZE)
    ax1.set_ylabel(
        r"$\mathnormal{\beta_2\;[10^{-24}\,s^2/m]}$",
        color="tab:blue",
        fontsize=AXIS_LABEL_SIZE,
    )
    _set_freq_xlim_thz(ax1)
    ax2 = ax1.twinx()
    ax2.plot(freqs_thz, aeff * 1e12, lw=0.8, color="tab:orange")
    ax2.set_ylabel(
        r"$\mathnormal{A_{eff}\;[\mu m^2]}$",
        color="tab:orange",
        fontsize=AXIS_LABEL_SIZE,
    )
    ax1.grid(False)
    _save_figure(fig, out_dir / "fiber_params.pdf", dpi=300)
    lg.success(f"Saved fiber parameters plot to {out_dir / 'fiber_params.pdf'}")
    plt.close(fig)
