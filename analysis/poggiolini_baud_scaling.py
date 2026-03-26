import argparse
from pathlib import Path

import matplotlib
import numpy as np
from loguru import logger as lg
from scipy.constants import c

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pynlin.constellation_stats import gaussian_mu0
from pynlin.nlin.nlin_estimator_uwb import collision_coeffs_system_uwb, total_nlin_uwb
from pynlin.nlin.pcfm_gn import PcfmConfig
from pynlin.system import System

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


def _set_baud_rate(system: System, baud_rate_hz: float) -> None:
    system.pulse.baud_rate = float(baud_rate_hz)
    system.pulse.T0 = 1.0 / float(baud_rate_hz)
    if system.pulse_config is not None:
        system.pulse_config.baud_rate = float(baud_rate_hz)


def _scaling_series(rows: list[dict]) -> dict[str, np.ndarray]:
    return {
        "TD 64-QAM": np.array([row["td_channel_w"] for row in rows], dtype=float),
        "TD Gaussian": np.array([row["td_gaussian_channel_w"] for row in rows], dtype=float),
        "PCFM total": np.array([row["pcfm_channel_w"] for row in rows], dtype=float),
        "PCFM SCI": np.array([row["pcfm_sci_channel_w"] for row in rows], dtype=float),
        "PCFM XCI": np.array([row["pcfm_xci_channel_w"] for row in rows], dtype=float),
        "Analytic limit": np.array([row["analytic_limit_channel_w"] for row in rows], dtype=float),
    }


def _center_channel_analytic_limit(system: System, launch_channel_w: float, channel_idx: int) -> float:
    """Simple limiting-case estimate:

    noise = P^3 T^2 gamma^2 * (L / (2*pi*T*beta2*df))
    """
    freqs = system.wdm.frequency_grid()
    f_cut = float(freqs[channel_idx])

    spacing_hz = getattr(system.wdm, "spacing", None)
    if spacing_hz is None:
        diffs = np.diff(np.asarray(freqs, dtype=float))
        if diffs.size == 0:
            raise ValueError("Cannot infer channel spacing for analytic limit.")
        spacing_hz = float(np.median(np.abs(diffs)))
    spacing_hz = max(float(abs(spacing_hz)), 1e-30)

    fiber = system.fiber
    wavelength_m = c / f_cut
    if hasattr(fiber, "effective_area_at"):
        aeff = float(fiber.effective_area_at(wavelength_m))
    else:
        aeff = float(getattr(fiber, "effective_area"))

    if hasattr(fiber, "beta2_at"):
        beta2 = float(fiber.beta2_at(wavelength_m))
    else:
        beta2 = float(getattr(fiber, "beta2"))

    n2 = 2.6e-20
    gamma = n2 * (2.0 * np.pi * f_cut) / (aeff * c)
    pulse_time = 1.0 / float(system.pulse.baud_rate)
    fiber_length = float(system.fiber_length)

    denom = 2.0 * np.pi * pulse_time * max(abs(beta2), 1e-30) * spacing_hz
    analytic = (launch_channel_w ** 3) * (pulse_time ** 2) * (gamma ** 2) * (fiber_length / denom)
    return float(analytic)


def _save_figure(fig: plt.Figure, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=260)
    if out_path.suffix.lower() == ".png":
        fig.savefig(out_path.with_suffix(".pdf"))


def _plot_scaling(rows: list[dict], out_path: Path, channel_idx: int, channel_freq_thz: float) -> None:
    baud_gbaud = np.array([row["baud_rate_gbaud"] for row in rows], dtype=float)
    series = _scaling_series(rows)
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    colors = ["black", "tab:red", "tab:blue", "tab:green", "tab:orange", "tab:purple"]
    for (label, values), color in zip(series.items(), colors):
        ax.plot(
            baud_gbaud,
            values,
            marker="o",
            markersize=3,
            markerfacecolor="none",
            markeredgewidth=0.9,
            linewidth=1.0,
            color=color,
            label=f"{label} (p={_fit_exponent(baud_gbaud * 1e9, values):.3f})",
        )
    # plot also the scaling function for reference: linear relation in baud rate, normalized to the first point of PCFM XCI
    ref_baud = baud_gbaud[0] * 1e9
    ref_value = series["PCFM XCI"][0] if series["PCFM XCI"].size else 1.0
    scaling_baud = np.array([ref_baud
                            , baud_gbaud[-1] * 1e9], dtype=float)
    scaling_values = ref_value * (scaling_baud / ref_baud)**(-1.0)  # inverse linear scaling as a reference
    ax.plot(
        scaling_baud / 1e9,
        scaling_values,
        linestyle="--",
        color="gray",
        label="linear scaling reference",
    )
        
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Baud rate [GBaud]")
    ax.set_ylabel("Center-channel NLI power [W]")
    ax.set_title(f"Center channel idx={channel_idx}, f={channel_freq_thz:.3f} THz")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=7)
    _save_figure(fig, out_path)
    plt.close(fig)


def _plot_normalized_scaling(
    rows: list[dict], out_path: Path, channel_idx: int, channel_freq_thz: float
) -> None:
    baud_gbaud = np.array([row["baud_rate_gbaud"] for row in rows], dtype=float)
    series = _scaling_series(rows)
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    colors = ["black", "tab:red", "tab:blue", "tab:green", "tab:orange", "tab:purple"]
    for (label, values), color in zip(series.items(), colors):
        ref = float(values[0]) if values.size else 1.0
        normalized = values / ref if ref > 0.0 else values
        ax.plot(
            baud_gbaud,
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
    ax.set_xlabel("Baud rate [GBaud]")
    ax.set_ylabel("Normalized center-channel NLI power")
    ax.set_title(f"Center channel idx={channel_idx}, f={channel_freq_thz:.3f} THz")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=7)
    _save_figure(fig, out_path)
    plt.close(fig)


def run_baud_sweep(
    cfg_path: Path,
    baud_rates_gbaud: list[float],
    out_dir: Path,
    launch_dbm: float | None,
    pcfm_numeric_xci: bool,
    recompute_td: bool,
    recompute_pcfm: bool,
    exclude_self_channel: bool = False,
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    base_system = System.from_toml(cfg_path)
    center_idx, band_label = _center_channel(base_system)
    freqs = base_system.wdm.frequency_grid()
    channel_freq_thz = float(freqs[center_idx] * 1e-12)
    lg.info(
        "Selected center channel idx={} in band={} at {:.6f} THz.".format(
            center_idx, band_label, channel_freq_thz
        )
    )

    for baud_gbaud in baud_rates_gbaud:
        system = System.from_toml(cfg_path)
        baud_hz = float(baud_gbaud) * 1e9
        _set_baud_rate(system, baud_hz)
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

        baud_tag = f"R{_safe_tag(baud_gbaud)}GBd"
        profile_path = out_dir / f"flat_profile_{baud_tag}.npy"
        _write_flat_profile(profile_path, system, launch_powers_w=launch_vec)

        ccfs = collision_coeffs_system_uwb(
            system,
            ipulse=1,
            recompute=recompute_td,
            profile_path=profile_path,
        )
        td_tag = f"{baud_tag}_{'xci' if exclude_self_channel else 'all'}"
        td_cache = _nlin_cache_path(
            profile_path=profile_path,
            use_kappa=False,
            use_x_mode=True,
            extra_tag=td_tag,
        )
        nlin_td = total_nlin_uwb(
            system,
            ccfs,
            use_kappa=False,
            use_x_mode=True,
            launch_powers_w=launch_vec,
            exclude_self_channel=exclude_self_channel,
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
            exclude_self_channel=exclude_self_channel,
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
        pcfm_path = out_dir / f"pcfm_{baud_tag}.npy"
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
            "baud_rate_gbaud": float(baud_gbaud),
            "channel_idx": int(center_idx),
            "band_label": band_label,
            "channel_freq_thz": channel_freq_thz,
            "launch_channel_w": float(launch_vec[center_idx]),
            "td_channel_w": float(td_vec[center_idx]),
            "td_gaussian_channel_w": float(td_gaussian_vec[center_idx]),
            "pcfm_channel_w": float(pcfm_total[center_idx]),
            "pcfm_sci_channel_w": float(pcfm_sci[center_idx]),
            "pcfm_xci_channel_w": float(pcfm_xci[center_idx]),
            "analytic_limit_channel_w": _center_channel_analytic_limit(
                system, float(launch_vec[center_idx]), center_idx
            ),
        }
        rows.append(row)
        lg.info(
            "R={:.1f} GBd -> TD={:.3e} W, TD(Gauss)={:.3e} W, PCFM={:.3e} W, "
            "PCFM_SCI={:.3e} W, PCFM_XCI={:.3e} W, Analytic={:.3e} W".format(
                row["baud_rate_gbaud"],
                row["td_channel_w"],
                row["td_gaussian_channel_w"],
                row["pcfm_channel_w"],
                row["pcfm_sci_channel_w"],
                row["pcfm_xci_channel_w"],
                row["analytic_limit_channel_w"],
            )
        )
        

    rows.sort(key=lambda item: item["baud_rate_gbaud"])
    baud_rates_hz = np.array([row["baud_rate_gbaud"] * 1e9 for row in rows], dtype=float)
    for key in (
        "td_channel_w",
        "td_gaussian_channel_w",
        "pcfm_channel_w",
        "pcfm_sci_channel_w",
        "pcfm_xci_channel_w",
        "analytic_limit_channel_w",
    ):
        exponent = _fit_exponent(baud_rates_hz, np.array([row[key] for row in rows], dtype=float))
        for row in rows:
            row[f"{key}_scaling_exp"] = exponent

    csv_path = out_dir / "center_channel_baud_scaling.csv"
    header = [
        "baud_rate_gbaud",
        "channel_idx",
        "band_label",
        "channel_freq_thz",
        "launch_channel_w",
        "td_channel_w",
        "td_gaussian_channel_w",
        "pcfm_channel_w",
        "pcfm_sci_channel_w",
        "pcfm_xci_channel_w",
        "analytic_limit_channel_w",
        "td_channel_w_scaling_exp",
        "td_gaussian_channel_w_scaling_exp",
        "pcfm_channel_w_scaling_exp",
        "pcfm_sci_channel_w_scaling_exp",
        "pcfm_xci_channel_w_scaling_exp",
        "analytic_limit_channel_w_scaling_exp",
    ]
    data = np.column_stack([[row[name] for row in rows] for name in header])
    np.savetxt(csv_path, data, delimiter=",", header=",".join(header), comments="", fmt="%s")

    plot_path = Path("media") / "PCFM" / "center_channel_baud_scaling.png"
    _plot_scaling(rows, plot_path, center_idx, channel_freq_thz)

    summary_lines = [
        f"Center channel idx={center_idx}, band={band_label}, freq={channel_freq_thz:.6f} THz",
        f"TD mode: {'exclude self-channel (nuB==nuA)' if exclude_self_channel else 'include all channels'}",
        f"TD 64-QAM exponent: {_fit_exponent(baud_rates_hz, np.array([row['td_channel_w'] for row in rows], dtype=float)):.6f}",
        f"TD Gaussian exponent: {_fit_exponent(baud_rates_hz, np.array([row['td_gaussian_channel_w'] for row in rows], dtype=float)):.6f}",
        f"PCFM total exponent: {_fit_exponent(baud_rates_hz, np.array([row['pcfm_channel_w'] for row in rows], dtype=float)):.6f}",
        f"PCFM SCI exponent: {_fit_exponent(baud_rates_hz, np.array([row['pcfm_sci_channel_w'] for row in rows], dtype=float)):.6f}",
        f"PCFM XCI exponent: {_fit_exponent(baud_rates_hz, np.array([row['pcfm_xci_channel_w'] for row in rows], dtype=float)):.6f}",
        f"Analytic limit exponent: {_fit_exponent(baud_rates_hz, np.array([row['analytic_limit_channel_w'] for row in rows], dtype=float)):.6f}",
    ]
    summary_path = out_dir / "center_channel_baud_scaling_summary.txt"
    summary_path.write_text("\n".join(summary_lines) + "\n")
    return csv_path, summary_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep baud rate and inspect center-channel TD/PCFM scaling."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="input/poggiolini_struct.toml",
        help="Path to system TOML.",
    )
    parser.add_argument(
        "--baud-rates-gbaud",
        type=str,
        default="10, 50, 100",
        help="Comma-separated baud rates in GBaud.",
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
        "--exclude-self-channel",
        action="store_true",
        default=True,
        help="Exclude nuB==nuA contribution from TD aggregation (SCI-like term).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="results/debug/poggiolini_baud_scaling",
        help="Output directory for generated profiles, caches, and reports.",
    )
    args = parser.parse_args()

    csv_path, summary_path = run_baud_sweep(
        cfg_path=Path(args.config),
        baud_rates_gbaud=_parse_float_list(args.baud_rates_gbaud),
        out_dir=Path(args.out_dir),
        launch_dbm=args.launch_dbm,
        pcfm_numeric_xci=bool(args.pcfm_numeric_xci),
        recompute_td=bool(args.recompute_td),
        recompute_pcfm=bool(args.recompute_pcfm),
        exclude_self_channel=bool(args.exclude_self_channel),
    )
    lg.success(f"Saved scaling CSV to {csv_path}")
    lg.success(f"Saved scaling summary to {summary_path}")


if __name__ == "__main__":
    main()
