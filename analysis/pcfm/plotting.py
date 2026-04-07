from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger as lg

from pynlin.nlin.pcfm_gn import fit_spp_polynomials, load_signal_profiles, normalize_spp
from pynlin.system import System


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
    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    if title:
        ax.set_title(title, fontsize=9)
    ax.plot(
        freqs_hz * 1e-12,
        gsnr_td,
        color="black",
        lw=0.45,
        marker="o",
        markersize=1.2,
        markerfacecolor="none",
        markeredgewidth=marker_lw,
        label="TD",
    )

    colors = ["tab:blue", "tab:orange", "tab:green"]
    if plot_pcfm_total_and_sci:
        for idx, (label, gsnr) in enumerate(gsnr_pcfm.items()):
            display = "" if label == "no_loss" else label
            suffix = f" {display}" if display else ""
            color = colors[idx % len(colors)]
            ax.plot(
                freqs_hz * 1e-12,
                gsnr,
                color=color,
                lw=0.45,
                marker="o",
                markersize=1.2,
                markerfacecolor="none",
                markeredgewidth=marker_lw,
                label=f"PCFM{suffix}",
            )

    if gsnr_gn:
        for idx, (label, gsnr) in enumerate(gsnr_gn.items()):
            display = "" if label == "no_loss" else label
            suffix = f" {display}" if display else ""
            color = colors[idx % len(colors)]
            ax.scatter(
                freqs_hz * 1e-12,
                gsnr,
                s=8,
                marker="o",
                facecolors="none",
                edgecolors=color,
                linewidths=marker_lw,
                label=f"GN{suffix}",
            )

    if gsnr_gn_direct:
        for idx, (label, gsnr) in enumerate(gsnr_gn_direct.items()):
            display = "" if label == "no_loss" else label
            suffix = f" {display}" if display else ""
            color = colors[idx % len(colors)]
            ax.scatter(
                freqs_hz * 1e-12,
                gsnr,
                s=8,
                marker="o",
                facecolors="none",
                edgecolors=color,
                linewidths=marker_lw,
                label=f"GN dir{suffix}",
            )

    ax.set_xlabel(r"$f \; [\mathrm{THz}]$")
    ax.set_ylabel(r"$GSNR_{NLI} \; [\mathrm{dB}]$")
    ax.grid(False)
    ax.legend(loc="best", fontsize=7)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
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
    """Plot per-channel NLIN normalized to end-of-fiber signal power (dB)."""
    dpi = 300
    marker_lw = 0.45
    signal_power_w = np.asarray(signal_power_w, dtype=float).reshape(-1)
    if signal_power_w.size != freqs_hz.size:
        raise ValueError(
            f"signal_power_w size {signal_power_w.size} != freq size {freqs_hz.size}"
        )
    denom = np.maximum(signal_power_w, 1e-18)

    def _ratio_db(nlin: np.ndarray, already_ratio: bool = False) -> np.ndarray:
        nlin = np.asarray(nlin, dtype=float).reshape(-1)
        if nlin.size != denom.size:
            raise ValueError(f"NLIN size {nlin.size} != signal power size {denom.size}")
        if already_ratio:
            return 10.0 * np.log10(np.maximum(nlin, 1e-18))
        return 10.0 * np.log10(np.maximum(nlin, 1e-18) / denom)

    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    if nlin_td_mod_w:
        styles = ["-", "--", ":"]
        style_idx = 0
        for label, nlin in nlin_td_mod_w.items():
            ratio_db = _ratio_db(nlin)
            is_gaussian = "gauss" in str(label).lower()
            color = "tab:red" if is_gaussian else "black"
            if is_gaussian:
                linestyle = "-"
            else:
                linestyle = styles[style_idx % len(styles)]
                style_idx += 1
            ax.plot(
                freqs_hz * 1e-12,
                ratio_db,
                color=color,
                lw=0.45,
                ls=linestyle,
                marker="o",
                markersize=1.2,
                markerfacecolor="none",
                markeredgewidth=marker_lw,
                label=f"TD {label}",
            )
    else:
        ratio_db = _ratio_db(nlin_td_w)
        ax.plot(
            freqs_hz * 1e-12,
            ratio_db,
            color="black",
            lw=0.45,
            marker="o",
            markersize=1.2,
            markerfacecolor="none",
            markeredgewidth=marker_lw,
            label="TD",
        )

    colors = ["tab:blue", "tab:orange", "tab:green"]
    if plot_pcfm_total_and_sci:
        for idx, (label, nlin) in enumerate(nlin_pcfm_w.items()):
            display = "" if label == "no_loss" else label
            suffix = f" {display}" if display else ""
            color = colors[idx % len(colors)]
            ratio_db = _ratio_db(nlin)
            ax.plot(
                freqs_hz * 1e-12,
                ratio_db,
                color=color,
                lw=0.45,
                marker="o",
                markersize=1.2,
                markerfacecolor="none",
                markeredgewidth=marker_lw,
                label=f"PCFM{suffix}",
            )

    if nlin_pcfm_xci_w:
        for idx, (label, nlin) in enumerate(nlin_pcfm_xci_w.items()):
            display = "" if label == "no_loss" else label
            suffix = f" {display}" if display else ""
            color = colors[idx % len(colors)]
            ratio_db = _ratio_db(nlin)
            ax.plot(
                freqs_hz * 1e-12,
                ratio_db,
                color=color,
                lw=0.45,
                ls="--",
                marker="o",
                markersize=1.2,
                markerfacecolor="none",
                markeredgewidth=marker_lw,
                label=f"PCFM XCI{suffix}",
            )

    if nlin_gn_w:
        for idx, (label, nlin) in enumerate(nlin_gn_w.items()):
            display = "" if label == "no_loss" else label
            suffix = f" {display}" if display else ""
            color = colors[idx % len(colors)]
            ratio_db = _ratio_db(nlin)
            ax.scatter(
                freqs_hz * 1e-12,
                ratio_db,
                s=8,
                marker="o",
                facecolors="none",
                edgecolors=color,
                linewidths=marker_lw,
                label=f"GN{suffix}",
            )

    if nlin_gn_direct_w:
        for idx, (label, nlin) in enumerate(nlin_gn_direct_w.items()):
            display = "" if label == "no_loss" else label
            suffix = f" {display}" if display else ""
            color = colors[idx % len(colors)]
            ratio_db = _ratio_db(nlin, already_ratio=gn_direct_is_ratio)
            ax.scatter(
                freqs_hz * 1e-12,
                ratio_db,
                s=8,
                marker="o",
                facecolors="none",
                edgecolors=color,
                linewidths=marker_lw,
                label=f"GN dir{suffix}",
            )

    if nlin_gn_xci_w:
        for idx, (label, nlin) in enumerate(nlin_gn_xci_w.items()):
            display = "" if label == "no_loss" else label
            suffix = f" {display}" if display else ""
            color = colors[idx % len(colors)]
            ratio_db = _ratio_db(nlin)
            ax.scatter(
                freqs_hz * 1e-12,
                ratio_db,
                s=8,
                marker="o",
                facecolors="none",
                edgecolors=color,
                linewidths=marker_lw,
                label=f"GN XCI{suffix}",
            )

    if nlin_gn_direct_xci_w:
        for idx, (label, nlin) in enumerate(nlin_gn_direct_xci_w.items()):
            display = "" if label == "no_loss" else label
            suffix = f" {display}" if display else ""
            color = colors[idx % len(colors)]
            ratio_db = _ratio_db(nlin, already_ratio=gn_direct_xci_is_ratio)
            ax.scatter(
                freqs_hz * 1e-12,
                ratio_db,
                s=8,
                marker="o",
                facecolors="none",
                edgecolors=color,
                linewidths=marker_lw,
                label=f"GN dir XCI{suffix}",
            )

    ax.set_xlabel(r"$f \; [\mathrm{THz}]$")
    ax.set_ylabel(r"$P_{NLI}/P_{sig}(L)\;[\mathrm{dB}]$")
    ax.grid(False)
    ax.legend(loc="best", fontsize=7)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    lg.success(f"Saved NLIN ratio plot to {out_path}")


def plot_pcfm_diagnostics(
    system: System,
    profile_path: Path | str,
    launch_powers_w: np.ndarray,
    out_dir: Path,
) -> None:
    """Generate diagnostic plots for intermediate quantities."""
    out_dir.mkdir(parents=True, exist_ok=True)
    freqs = system.wdm.frequency_grid()
    freqs_thz = freqs * 1e-12
    launch_dbm = 10.0 * np.log10(np.maximum(launch_powers_w, 1e-18) / 1e-3)

    fig, ax = plt.subplots(figsize=(3.6, 2.4))
    ax.plot(freqs_thz, launch_dbm, lw=0.8, color="black")
    ax.set_xlabel(r"$f \; [\mathrm{THz}]$")
    ax.set_ylabel(r"$P_\mathrm{launch}\;[\mathrm{dBm}]$")
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(out_dir / "launch_power.pdf", dpi=300)
    lg.success(f"Saved launch power plot to {out_dir / 'launch_power.pdf'}")
    plt.close(fig)

    sig_ch_z, z_axis = load_signal_profiles(profile_path, system)
    span = float(z_axis[-1] - z_axis[0]) if z_axis.size else 0.0
    avg_power = np.trapezoid(sig_ch_z, z_axis, axis=1) / max(span, 1.0)
    out_power = sig_ch_z[:, -1]
    avg_dbm = 10.0 * np.log10(np.maximum(avg_power, 1e-18) / 1e-3)
    out_dbm = 10.0 * np.log10(np.maximum(out_power, 1e-18) / 1e-3)

    fig, ax = plt.subplots(figsize=(3.6, 2.4))
    ax.plot(freqs_thz, avg_dbm, lw=0.8, color="tab:blue", label="avg")
    ax.plot(freqs_thz, out_dbm, lw=0.8, color="tab:orange", label="out")
    ax.set_xlabel(r"$f \; [\mathrm{THz}]$")
    ax.set_ylabel(r"$P\;[\mathrm{dBm}]$")
    ax.grid(False)
    ax.legend(loc="best", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "profile_power.pdf", dpi=300)
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

    fig, ax = plt.subplots(figsize=(4.0, 2.8))
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
    ax.set_xlabel(r"$z\;[\mathrm{km}]$")
    ax.set_ylabel(r"normalized power")
    ax.grid(False)
    ax.legend(loc="best", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "spp_fit.pdf", dpi=300)
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
    fig, ax1 = plt.subplots(figsize=(3.6, 2.4))
    ax1.plot(freqs_thz, p_l, lw=0.8, color="tab:blue")
    ax1.set_xlabel(r"$f \; [\mathrm{THz}]$")
    ax1.set_ylabel(r"$p(L)$", color="tab:blue")
    ax2 = ax1.twinx()
    ax2.plot(freqs_thz, poly_sum, lw=0.8, color="tab:orange")
    ax2.set_ylabel(r"$\\sum a_n a_k/(n+k+1)$", color="tab:orange")
    ax1.grid(False)
    fig.tight_layout()
    fig.savefig(out_dir / "pcfm_terms.pdf", dpi=300)
    lg.success(f"Saved PCFM terms plot to {out_dir / 'pcfm_terms.pdf'}")
    plt.close(fig)

    pump_specs = system.pump_specs or []
    if pump_specs:
        pump_freqs_thz = np.array([3e8 / p.wavelength for p in pump_specs], dtype=float) * 1e-12
        pump_powers_dbm = np.array([p.power_dbm for p in pump_specs], dtype=float)
    else:
        pump_freqs_thz = np.array([])
        pump_powers_dbm = np.array([])
    fig, ax = plt.subplots(figsize=(3.6, 2.4))
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
    ax.set_xlabel(r"$f \; [\mathrm{THz}]$")
    ax.set_ylabel(r"$P\;[\mathrm{dBm}]$")
    ax.grid(False)
    ax.legend(loc="best", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "launch_spectrum.pdf", dpi=300)
    lg.success(f"Saved launch spectrum plot to {out_dir / 'launch_spectrum.pdf'}")
    plt.close(fig)

    wl = 3e8 / freqs
    beta2 = np.array([system.fiber.beta2_at(float(w)) for w in wl], dtype=float)
    aeff = np.array([system.fiber.effective_area_at(float(w)) for w in wl], dtype=float)
    fig, ax1 = plt.subplots(figsize=(3.6, 2.4))
    ax1.plot(freqs_thz, beta2 * 1e24, lw=0.8, color="tab:blue")
    ax1.set_xlabel(r"$f \; [\mathrm{THz}]$")
    ax1.set_ylabel(r"$\\beta_2\\;[10^{-24}\\,s^2/m]$", color="tab:blue")
    ax2 = ax1.twinx()
    ax2.plot(freqs_thz, aeff * 1e12, lw=0.8, color="tab:orange")
    ax2.set_ylabel(r"$A_{eff}\\;[\\mu m^2]$", color="tab:orange")
    ax1.grid(False)
    fig.tight_layout()
    fig.savefig(out_dir / "fiber_params.pdf", dpi=300)
    lg.success(f"Saved fiber parameters plot to {out_dir / 'fiber_params.pdf'}")
    plt.close(fig)
