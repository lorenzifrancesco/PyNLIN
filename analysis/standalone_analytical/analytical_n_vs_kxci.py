"""Standalone non-interactive plotter for the analytical N vs K_XCI comparison.

This reproduces the analytical comparison used in the scratch ``N_vs_KXCI``
workflow, using the TD model

    N(x) T^2 L^-2 = 4/9 * (1 + (((2 pi)/lambda) * (L/L_eff) * x)^(1/eta))^(-eta)

against the flat-profile closed-form XCI kernel

    K_XCI(x) T^2 L^-2 = (1/(2 pi)) * (L_eff/L) * |ln((x + 1/2)/(x - 1/2))|

for x = Delta f / B.

"""

from __future__ import annotations

import argparse
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mplconfig"))

import matplotlib
import numpy as np
from loguru import logger as lg
from scipy.integrate import quad
from scipy.special import sici
from matplotlib.ticker import FixedLocator

matplotlib.use("Agg")
import matplotlib.pyplot as plt

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.pcfm.analytics import _g_kernel as _g_kernel_fast
from analysis.pcfm.figure_size import scale_figsize_to_ieee_column
from analysis.pcfm.io import _write_flat_profile
from pynlin.nlin.cache_names import s2a_lo_timeint_path, s2b_lo_extrema_path
from pynlin.nlin.nlin_estimation.ideal_fits_uwb import ideal_fit_coefficients, softplus
from pynlin.nlin.nlin_estimation.lo_correction_uwb import (
    build_lookup_integral_table_with_raman,
)
from pynlin.nlin.nlin_estimator_uwb import apply_plateau_correction
from pynlin.system import System
from pynlin.utils import _toml_load


DEFAULT_OUT_DIR = REPO_ROOT / "media" / "standalone_analytical"
DEFAULT_CONFIG_PATH = REPO_ROOT / "input" / "analytical_n_vs_kxci.toml"
DEFAULT_SYSTEM_CONFIG_PATH = REPO_ROOT / "input" / "pcfm_struct.toml"
DEFAULT_MPLRC_PATH = Path.home() / ".config" / "matplotlib" / "matplotlibrc"
DPI = 300
CONFIG_SECTION = "analytical_n_vs_kxci"
LINE_LW = 1.25
AXIS_LABEL_SIZE = 9.5
LEGEND_SIZE = 8.5
GNUPLOT_RED = "#e00000ff"
GNUPLOT_GREEN = "#00a500ff"
GNUPLOT_BLUE = "#0c22f0ff"
GNUPLOT_ORANGE = "#B0B000"
GNUPLOT_GRAY = "#606060"

COLOR_N = GNUPLOT_RED
COLOR_N_REFERENCE = GNUPLOT_GRAY
COLOR_KXCI = GNUPLOT_GREEN
COLOR_KXCI_EQ18 = GNUPLOT_BLUE
COLOR_PRE_UNITY_SHADE = "#F0F0F0"
DEFAULT_LLD_SWEEP_X = 1.0
DEFAULT_LLD_SWEEP_MIN = 1e-2
DEFAULT_LLD_SWEEP_MAX = 20.0
DEFAULT_LLD_SWEEP_NPTS = 400

###########
# Graphical
###########
def _configure_matplotlib() -> None:
    # Load the user's matplotlibrc explicitly because MPLCONFIGDIR is set to a
    # writable local directory for sandboxed runs, which would otherwise bypass
    # the user's sansmath / inward-tick defaults.
    if DEFAULT_MPLRC_PATH.exists():
        matplotlib.rc_file(str(DEFAULT_MPLRC_PATH))
    plt.rcParams["lines.linewidth"] = LINE_LW
    plt.rcParams["xtick.labelsize"] = 8
    plt.rcParams["ytick.labelsize"] = 8

def _style_axes(ax: plt.Axes, *, loglog: bool) -> None:
    if loglog:
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.grid(False)

def _make_x_grid(x_min: float, x_max: float, npts: int, loglog: bool) -> np.ndarray:
    if npts < 2:
        raise ValueError("npts must be at least 2.")
    if x_max <= x_min:
        raise ValueError("x_max must be greater than x_min.")
    if loglog:
        if x_min <= 0.0:
            raise ValueError("log-log plots require x_min > 0.")
        return np.geomspace(x_min, x_max, int(npts))
    return np.linspace(x_min, x_max, int(npts))

###########
# Physical
###########
def n_time_domain(x: np.ndarray, lam: float, l_over_leff: float, eta: float) -> np.ndarray:
    """TD curve with rescaling used in the original comparison."""
    scale = (2.0 * np.pi) * float(l_over_leff) / float(lam)
    return (4.0 / 9.0) * (1.0 + (scale * x) ** (1.0 / eta)) ** (-eta)


def n_softplus_scaled(
    x: np.ndarray,
    *,
    l_over_ld: float,
    ps: tuple[float, float, float] | np.ndarray,
) -> np.ndarray:
    """Evaluate the TD softplus using x_walkoff=(2 pi)(L/LD)x."""
    return softplus((2.0 * np.pi) * float(l_over_ld) * np.asarray(x, dtype=float), *ps)


def k_xci(x: np.ndarray, leff_over_l: float) -> np.ndarray:
    """Closed-form XCI kernel over the non-overlapping region x > 1/2."""
    out = np.full_like(x, np.nan, dtype=float)
    x = np.asarray(x, dtype=float)
    leff_over_l = np.asarray(leff_over_l, dtype=float)
    mask = x > 0.5
    if np.any(mask):
        scale = np.broadcast_to(leff_over_l, x.shape)
        out[mask] = (scale[mask] / (2.0 * np.pi)) * np.abs(
            np.log((x[mask] + 0.5) / (x[mask] - 0.5))
        )
    return out


@lru_cache(maxsize=None)
def _j_of_x_scalar(x_value: float) -> float:
    if np.isclose(x_value, 0.0):
        return 0.0
    sign = 1.0 if x_value >= 0.0 else -1.0
    x_abs = abs(x_value)

    def integrand(t: float) -> float:
        if np.isclose(t, 0.0):
            return 1.0
        si_value, _ = sici(t)
        return si_value / t

    value, _ = quad(integrand, 0.0, x_abs, limit=300, epsabs=1e-11, epsrel=1e-11)
    return sign * value


def _g_kernel(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    for idx, zi in np.ndenumerate(z):
        if np.isclose(zi, 0.0):
            out[idx] = 0.0
            continue
        si_value, _ = sici(float(zi))
        out[idx] = _j_of_x_scalar(float(zi)) - si_value + (1.0 - np.cos(zi)) / zi
    return out


def k_xci_eq18_normalized(x: np.ndarray, l_over_leff: float) -> np.ndarray:
    """Notebook Eq. (18) flat-Raman kernel in normalized variables."""
    x = np.asarray(x, dtype=float)
    l_over_leff = np.asarray(l_over_leff, dtype=float)
    if np.any(l_over_leff <= 0.0):
        raise ValueError("L/Leff must be strictly positive.")
    out = np.full_like(x, np.nan, dtype=float)
    # mask = x > 0.5
    mask = np.ones_like(x, dtype=bool)# FIXME
    if not np.any(mask):
        return out
    scale = np.broadcast_to(l_over_leff, x.shape)
    z_plus = np.pi * scale[mask] * (x[mask] + 0.5)
    z_minus = np.pi * scale[mask] * (x[mask] - 0.5)
    out[mask] = (
        (2.0 / (np.pi * scale[mask])) * (_g_kernel_fast(z_plus) - _g_kernel_fast(z_minus))
    ) / (2.0 * np.pi)
    return out

################
# Parsing and IO
################
def safe_ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    out = np.full_like(num, np.nan, dtype=float)
    mask = np.isfinite(num) & np.isfinite(den) & (np.abs(den) > 1e-14)
    out[mask] = num[mask] / den[mask]
    return out


def find_crossover(x: np.ndarray, y1: np.ndarray, y2: np.ndarray) -> float | None:
    diff = y1 - y2
    finite = np.isfinite(diff)
    idx = np.where(finite[:-1] & finite[1:] & (diff[:-1] * diff[1:] <= 0.0))[0]
    if idx.size == 0:
        return None
    i = int(idx[0])
    x1, x2 = float(x[i]), float(x[i + 1])
    d1, d2 = float(diff[i]), float(diff[i + 1])
    if np.isclose(d1, d2):
        return x1
    return float(x1 - d1 * (x2 - x1) / (d2 - d1))


def finite_min_max(*arrays: np.ndarray | float) -> tuple[float, float]:
    finite_values: list[np.ndarray] = []
    for array in arrays:
        values = np.asarray(array, dtype=float).ravel()
        mask = np.isfinite(values)
        if np.any(mask):
            finite_values.append(values[mask])
    if not finite_values:
        raise ValueError("No finite values available for plotting.")
    merged = np.concatenate(finite_values)
    return float(np.min(merged)), float(np.max(merged))

def _plot_pcfm_style_line(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    *,
    color: str,
    label: str,
    linestyle: str = "-",
) -> None:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    if not np.any(finite):
        return

    x_f = x[finite]
    y_f = y[finite]
    ax.plot(x_f, y_f, color=color, lw=LINE_LW, ls=linestyle, label=label)


def _shade_pre_unity_region(ax: plt.Axes, x: np.ndarray) -> None:
    x = np.asarray(x, dtype=float)
    finite = np.isfinite(x)
    if not np.any(finite):
        return
    x_min = float(np.min(x[finite]))
    if x_min >= 1.0:
        return
    ax.axvspan(
        x_min,
        1.0,
        facecolor=COLOR_PRE_UNITY_SHADE,
        alpha=0.5,
        zorder=-10,
    )


def _add_reference_y_ticks(ax: plt.Axes) -> tuple[float, float]:
    y_ln3 = float(np.log(3.0))
    y_four_ninths = 4.0 / 9.0

    current_ticks = np.asarray(ax.get_yticks(), dtype=float)
    merged_ticks = np.unique(
        np.concatenate([current_ticks, np.array([y_ln3, y_four_ninths])])
    )
    ax.yaxis.set_major_locator(FixedLocator(merged_ticks))

    fig = ax.figure
    fig.canvas.draw()

    for tick, value in zip(ax.yaxis.get_major_ticks(), ax.get_yticks()):
        if np.isclose(value, y_ln3):
            tick.label1.set_text(r"$\ln(3)$")
            tick.label1.set_color("tab:red")
            tick.tick1line.set_color("tab:red")
            tick.tick2line.set_color("tab:red")
        elif np.isclose(value, y_four_ninths):
            # tick.label1.set_text(r"$4/9$")
            tick.label1.set_color("tab:green")
            tick.tick1line.set_color("tab:green")
            tick.tick2line.set_color("tab:green")

    fig.canvas.draw()
    return y_ln3, y_four_ninths


def _comparison_figure(
    x: np.ndarray,
    y_n: np.ndarray,
    y_k: np.ndarray,
    *,
    y_k_eq18: np.ndarray,
    y_n_reference: np.ndarray | None,
    loglog: bool,
    normalize: bool,
    show_crossover: bool,
    show_kxci_alternative: bool,
    n_label: str,
    n_reference_label: str | None,
    show_unity_guides: bool = True,
) -> tuple[plt.Figure, float | None]:
    fig, ax = plt.subplots(figsize=scale_figsize_to_ieee_column(3.6, 2.8))
    if show_unity_guides:
        _shade_pre_unity_region(ax, x)
    _plot_pcfm_style_line(ax, x, y_n, color=COLOR_N, label=n_label)
    if y_n_reference is not None and n_reference_label is not None:
        _plot_pcfm_style_line(
            ax,
            x,
            y_n_reference,
            color=COLOR_N_REFERENCE,
            linestyle="--",
            label=n_reference_label,
        )
    _plot_pcfm_style_line(
        ax,
        x,
        y_k,
        color=COLOR_KXCI,
        linestyle="--",
        label=r"$\mathnormal{\mathcal{N}^{(\mathrm{PCFM-I})}}$",
    )
    if show_kxci_alternative:
        _plot_pcfm_style_line(
            ax,
            x,
            y_k_eq18,
            color=COLOR_KXCI_EQ18,
            linestyle="-.",
            label=r"$\mathnormal{\mathcal{N}^{(\mathrm{PCFM-II})}}$",
        )
    if show_unity_guides:
        ax.axvline(1.0, color="0.5", linestyle=":", linewidth=LINE_LW)

    y_ln3 = float(np.log(3.0))
    y_four_ninths = 4.0 / 9.0
    _style_axes(ax, loglog=loglog)

    if not loglog:
        arrays = [y_n, y_k, y_ln3, y_four_ninths]
        if y_n_reference is not None:
            arrays.append(y_n_reference)
        if show_kxci_alternative:
            arrays.append(y_k_eq18)
        ymin, ymax = finite_min_max(*arrays)
        pad = 0.06 * (ymax - ymin) if ymax > ymin else 0.1
        ax.set_ylim(ymin - pad, ymax + pad)

    _add_reference_y_ticks(ax)

    ax.set_xlabel(r"$\mathnormal{\Delta f / B}$", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel(
        r"$\mathcal{N}$ (normalized)" if normalize else r"$\mathcal{N}\,\mathnormal{T^2L^{-2}}$",
        fontsize=AXIS_LABEL_SIZE,
    )
    ax.legend(loc="best", fontsize=LEGEND_SIZE)

    crossover = None
    if show_crossover:
        crossover = find_crossover(x, y_n, y_k)
        if crossover is not None and np.isfinite(crossover):
            ax.axvline(crossover, color="0.35", linestyle="--", linewidth=LINE_LW)
            ax.annotate(
                rf"$x_{{\times}}\approx {crossover:.3f}$",
                xy=(
                    crossover,
                    ax.get_ylim()[1]
                    if not loglog
                    else y_n[np.nanargmax(np.nan_to_num(y_n, nan=-np.inf))],
                ),
                xytext=(4, -10),
                textcoords="offset points",
                color="0.25",
                # fontsize=7,
            )

    fig.tight_layout()
    return fig, crossover


def _slice_from_unity(
    x: np.ndarray,
    y_n: np.ndarray,
    y_k: np.ndarray,
    y_k_eq18: np.ndarray,
    y_n_reference: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    mask = np.isfinite(x) & (x >= 1.0)
    x_sel = np.asarray(x, dtype=float)[mask]
    y_n_sel = np.asarray(y_n, dtype=float)[mask]
    y_k_sel = np.asarray(y_k, dtype=float)[mask]
    y_k_eq18_sel = np.asarray(y_k_eq18, dtype=float)[mask]
    y_n_reference_sel = None if y_n_reference is None else np.asarray(y_n_reference, dtype=float)[mask]
    return x_sel, y_n_sel, y_k_sel, y_k_eq18_sel, y_n_reference_sel


def _ratio_figure(
    x: np.ndarray,
    ratio: np.ndarray,
    *,
    ratio_eq18: np.ndarray | None,
    loglog: bool,
    crossover: float | None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=scale_figsize_to_ieee_column(3.6, 2.4))
    _shade_pre_unity_region(ax, x)
    _plot_pcfm_style_line(
        ax,
        x,
        ratio,
        color=COLOR_KXCI,
        linestyle="--",
        label=r"$\mathnormal{\mathcal{N}^{(\mathrm{PCFM-I})}/\mathcal{N}^{(\mathrm{TD})}}$",
    )
    if ratio_eq18 is not None:
        _plot_pcfm_style_line(
            ax,
            x,
            ratio_eq18,
            color=COLOR_KXCI_EQ18,
            linestyle="-.",
            label=r"$\mathnormal{\mathcal{N}^{(\mathrm{PCFM-II})}/\mathcal{N}^{(\mathrm{TD})}}$",
        )
    if crossover is not None and np.isfinite(crossover):
        ax.axvline(crossover, color="0.35", linestyle="--", linewidth=LINE_LW)
    _style_axes(ax, loglog=loglog)
    ax.set_xlabel(r"$\mathnormal{\Delta f / B}$", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel(
        r"$\mathnormal{\mathcal{N}^{(\mathrm{PCFM})}/\mathcal{N}^{(\mathrm{TD})}}$",
        fontsize=AXIS_LABEL_SIZE,
    )
    ax.legend(loc="best", fontsize=LEGEND_SIZE)
    fig.tight_layout()
    return fig


def _lld_sweep_figure(
    l_over_ld: np.ndarray,
    y_n: np.ndarray,
    y_k: np.ndarray,
    *,
    y_k_eq18: np.ndarray,
    x_fixed: float,
    loglog: bool,
    n_label: str,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=scale_figsize_to_ieee_column(3.6, 2.8))
    _plot_pcfm_style_line(ax, l_over_ld, y_n * 1.0, color=COLOR_N, label=n_label) # FIXME manual rescaling
    _plot_pcfm_style_line(
        ax,
        l_over_ld,
        y_k,
        color=COLOR_KXCI,
        linestyle="--",
        label=r"$\mathnormal{\mathcal{N}^{(\mathrm{PCFM-I})}}$",
    )
    _plot_pcfm_style_line(
        ax,
        l_over_ld,
        y_k_eq18,
        color=COLOR_KXCI_EQ18,
        linestyle="-.",
        label=r"$\mathnormal{\mathcal{N}^{(\mathrm{PCFM-II})}}$",
    )
    ax.set_xscale("log")
    if loglog:
        ax.set_yscale("log")
    ax.set_xlabel(r"$\mathnormal{L/L_D}$", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel(r"$\mathnormal{\mathcal{N}\,T^2L^{-2}}$", fontsize=AXIS_LABEL_SIZE)
    ax.grid(False)
    ax.legend(loc="best", fontsize=LEGEND_SIZE)
    fig.tight_layout()
    return fig


def _combined_comparison_figure(
    x: np.ndarray,
    cases: list[dict[str, Any]],
    *,
    loglog: bool,
    normalize: bool,
    show_kxci_alternative: bool,
    show_unity_guides: bool = True,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=scale_figsize_to_ieee_column(3.6, 2.9))
    arrays: list[np.ndarray | float] = []
    if show_unity_guides:
        _shade_pre_unity_region(ax, x)

    for idx, case in enumerate(cases):
        label_n = r"$\mathnormal{\mathcal{N}^{(\mathrm{TD})}}$" if idx == 0 else "_nolegend_"
        label_k = r"$\mathnormal{\mathcal{N}^{(\mathrm{PCFM-I})}}$" if idx == 0 else "_nolegend_"
        label_k_eq18 = (
            r"$\mathnormal{\mathcal{N}^{(\mathrm{PCFM-II})}}$" if idx == 0 else "_nolegend_"
        )
        _plot_pcfm_style_line(
            ax,
            x,
            case["y_n"],
            color=COLOR_N,
            label=label_n,
        )
        _plot_pcfm_style_line(
            ax,
            x,
            case["y_k"],
            color=COLOR_KXCI,
            linestyle="--",
            label=label_k,
        )
        arrays.extend([case["y_n"], case["y_k"]])
        if show_kxci_alternative:
            _plot_pcfm_style_line(
                ax,
                x,
                case["y_k_eq18"],
                color=COLOR_KXCI_EQ18,
                linestyle="-.",
                label=label_k_eq18,
            )
            arrays.append(case["y_k_eq18"])

    if show_unity_guides:
        ax.axvline(1.0, color="0.5", linestyle=":", linewidth=LINE_LW)
    _style_axes(ax, loglog=loglog)

    if not loglog:
        ymin, ymax = finite_min_max(*arrays)
        pad = 0.06 * (ymax - ymin) if ymax > ymin else 0.1
        ax.set_ylim(ymin - pad, ymax + pad)

    ax.set_xlabel(r"$\mathnormal{\Delta f / B}$", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel(
        r"$\mathcal{N}$ (normalized)" if normalize else r"$\mathcal{N}\,\mathnormal{T^2L^{-2}}$",
        fontsize=AXIS_LABEL_SIZE,
    )
    ax.legend(loc="best", fontsize=LEGEND_SIZE)
    fig.tight_layout()
    return fig


def _combined_ratio_figure(
    x: np.ndarray,
    cases: list[dict[str, Any]],
    *,
    loglog: bool,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=scale_figsize_to_ieee_column(3.6, 2.5))
    _shade_pre_unity_region(ax, x)

    for case in cases:
        _plot_pcfm_style_line(
            ax,
            x,
            case["ratio"],
            color=COLOR_KXCI,
            linestyle="--",
            label=rf"$\mathnormal{{\mathcal{{N}}^{{(\mathrm{{PCFM-I}})}}/\mathcal{{N}}^{{(\mathrm{{TD}})}}}}$, {case['case_label']}",
        )
        crossover = case["crossover"]
        if crossover is not None and np.isfinite(crossover):
            ax.axvline(crossover, color=COLOR_N_REFERENCE, linestyle="--", linewidth=LINE_LW)

    _style_axes(ax, loglog=loglog)
    ax.set_xlabel(r"$\mathnormal{\Delta f / B}$", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel(
        r"$\mathnormal{\mathcal{N}^{(\mathrm{PCFM-I})}/\mathcal{N}^{(\mathrm{TD})}}$",
        fontsize=AXIS_LABEL_SIZE,
    )
    ax.legend(loc="best", fontsize=LEGEND_SIZE)
    fig.tight_layout()
    return fig


def _save_figure(fig: plt.Figure, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI)
    print(f"Saved {out_path}")


def _case_tag(value: float) -> str:
    return f"{value:.3f}".rstrip("0").rstrip(".").replace("-", "m").replace(".", "p")


def _as_float_list(value: Any) -> list[float] | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [float(item) for item in value]
    return [float(value)]


def _force_flat_profile_mode(system: System) -> None:
    raw = getattr(system, "raw_config", None)
    if not isinstance(raw, dict):
        raw = {}
        system.raw_config = raw

    pcfm = raw.get("pcfm")
    if not isinstance(pcfm, dict):
        pcfm = {}
        raw["pcfm"] = pcfm
    run = pcfm.get("run")
    if not isinstance(run, dict):
        run = {}
        pcfm["run"] = run
    run["power_profiles_mode"] = "flat"

    nlin_section = raw.get("nlin")
    if not isinstance(nlin_section, dict):
        nlin_section = {}
        raw["nlin"] = nlin_section
    nlin_section["flat_profiles"] = True


def _ensure_flat_profile(
    system: System,
    profile_path: Path,
    *,
    recompute: bool,
) -> Path:
    if recompute or not profile_path.exists():
        _write_flat_profile(profile_path, system)
    return profile_path


def _prepare_flat_td_corrections(
    system: System,
    *,
    profile_path: Path,
    ipulse: int,
    recompute: bool,
    max_l_over_ld: float,
    eta_override: float | None = None,
    m_lo_truncation: int = 2,
) -> dict[str, Any]:
    _force_flat_profile_mode(system)
    profile_path = _ensure_flat_profile(system, profile_path, recompute=recompute)
    pulse_name = "gaussian" if ipulse == 0 else "nyquist"
    lld_max = max(float(max_l_over_ld), 2.5)
    s2b_cache = s2b_lo_extrema_path(
        ipulse=ipulse,
        m_lo_truncation=m_lo_truncation,
        fiber_length=float(system.fiber_length),
        lld_max=lld_max,
    )
    s2a_caches = [
        s2a_lo_timeint_path(ipulse=ipulse, m_lo=m_lo)
        for m_lo in range(m_lo_truncation + 1)
    ]
    lg.info(
        "TD correction lookup setup: pulse={} flat_profile={} profile_path={} "
        "S2B cache={} source S2A caches={}".format(
            pulse_name,
            profile_path.exists(),
            profile_path,
            s2b_cache,
            [str(path) for path in s2a_caches],
        )
    )
    rmax_lookup, rmin_lookup = build_lookup_integral_table_with_raman(
        system,
        m_lo_truncation=m_lo_truncation,
        ipulse=ipulse,
        recompute=recompute,
        profile_path=profile_path,
        max_lld=max_l_over_ld,
    )
    ps_ideal_raw = tuple(
        float(v)
        for v in ideal_fit_coefficients(
            0.0,
            0.0,
            ipulse=ipulse,
            fiber_length=float(system.fiber_length),
            baud_rate=float(system.pulse.baud_rate),
        )
    )
    ps_ideal = (
        ps_ideal_raw[0],
        ps_ideal_raw[1],
        float(ps_ideal_raw[2] if eta_override is None else eta_override),
    )
    if eta_override is not None:
        lg.info(
            "TD correction setup: preserving user eta={:.6g} instead of ideal-fit eta={:.6g}".format(
                float(eta_override),
                ps_ideal_raw[2],
            )
        )
    return {
        "system": system,
        "profile_path": profile_path,
        "ps_ideal": ps_ideal,
        "ps_ideal_raw": ps_ideal_raw,
        "rmax_lookup": rmax_lookup,
        "rmin_lookup": rmin_lookup,
        "s2b_cache": s2b_cache,
        "s2a_caches": s2a_caches,
        "pulse_name": pulse_name,
        "m_lo_truncation": int(m_lo_truncation),
    }


def _td_corrected_softplus_params_flat(
    l_over_ld: float,
    td_ctx: dict[str, Any],
    *,
    verbose: bool = True,
) -> tuple[float, float, float]:
    lo_value = float(td_ctx["rmin_lookup"](float(l_over_ld), float(l_over_ld)))
    ps_ideal = tuple(float(v) for v in td_ctx["ps_ideal"])
    ps_corrected = tuple(float(v) for v in apply_plateau_correction(ps_ideal, lo_value))
    if verbose:
        lg.info(
            "TD flat-profile correction from cache {} at L/LD={:.6g}: "
            "using rmin_lookup(L/LD,L/LD)={:.6e}; ideal (a,Lambda,eta)=({:.6e},{:.6e},{:.6e}) "
            "-> corrected (a,Lambda,eta)=({:.6e},{:.6e},{:.6e})".format(
                td_ctx["s2b_cache"],
                float(l_over_ld),
                lo_value,
                ps_ideal[0],
                ps_ideal[1],
                ps_ideal[2],
                ps_corrected[0],
                ps_corrected[1],
                ps_corrected[2],
            )
        )
    return ps_corrected


def _write_summary(path: Path, rows: list[dict[str, float]]) -> None:
    if not rows:
        return
    header = (
        "l_over_ld,crossover_x,plateau_ideal,lambda_ideal,eta_ideal,"
        "plateau_corrected,lambda_corrected,eta_corrected"
    )
    lines = [header]
    for row in rows:
        lines.append(
            ",".join(
                [
                    f"{row['l_over_ld']:.12g}",
                    "nan" if not np.isfinite(row["crossover_x"]) else f"{row['crossover_x']:.12g}",
                    f"{row['plateau_ideal']:.12g}",
                    f"{row['lambda_ideal']:.12g}",
                    f"{row['eta_ideal']:.12g}",
                    f"{row['plateau_corrected']:.12g}",
                    f"{row['lambda_corrected']:.12g}",
                    f"{row['eta_corrected']:.12g}",
                ]
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="ascii")
    print(f"Saved {path}")


def _config_key_map() -> dict[str, str]:
    return {
        "lam": "lam",
        "lambda": "lam",
        "l_over_leff": "l_over_leff",
        "l_over_leff_values": "l_over_leff_values",
        "l_over_ld_values": "l_over_leff_values",
        "l-over-leff": "l_over_leff",
        "eta": "eta",
        "x_min": "x_min",
        "x-min": "x_min",
        "x_max": "x_max",
        "x-max": "x_max",
        "npts": "npts",
        "show_ratio": "show_ratio",
        "show-ratio": "show_ratio",
        "show_crossover": "show_crossover",
        "show-crossover": "show_crossover",
        "show_kxci_alternative": "show_kxci_alternative",
        "show-kxci-alternative": "show_kxci_alternative",
        "normalize": "normalize",
        "loglog": "loglog",
        "out_dir": "out_dir",
        "out-dir": "out_dir",
        "stem": "stem",
        "format": "format",
        "use_td_corrections": "use_td_corrections",
        "use-td-corrections": "use_td_corrections",
        "system_config": "system_config",
        "system-config": "system_config",
        "profile_path": "profile_path",
        "profile-path": "profile_path",
        "recompute_corrections": "recompute_corrections",
        "recompute-corrections": "recompute_corrections",
        "flat_profile": "flat_profile",
        "flat-profile": "flat_profile",
        "ipulse": "ipulse",
        "m_lo_truncation": "m_lo_truncation",
        "m-lo-truncation": "m_lo_truncation",
    }


def _load_config_defaults(config_path: Path | None) -> dict[str, object]:
    if config_path is None:
        return {}

    data = _toml_load(config_path)
    raw = data.get(CONFIG_SECTION, data)
    if not isinstance(raw, dict):
        raise ValueError(
            f"Expected TOML table '{CONFIG_SECTION}' or a flat mapping in {config_path}."
        )

    key_map = _config_key_map()
    defaults: dict[str, object] = {}
    unknown_keys: list[str] = []
    for raw_key, value in raw.items():
        mapped = key_map.get(str(raw_key))
        if mapped is None:
            unknown_keys.append(str(raw_key))
            continue
        defaults[mapped] = value

    if unknown_keys:
        allowed = ", ".join(sorted(key_map))
        raise ValueError(
            f"Unsupported config keys in {config_path}: {sorted(unknown_keys)}. "
            f"Allowed keys: {allowed}."
        )

    if "out_dir" in defaults:
        defaults["out_dir"] = (config_path.parent / Path(defaults["out_dir"])).resolve()
    for key in ("system_config", "profile_path"):
        if key in defaults and defaults[key] is not None:
            defaults[key] = (config_path.parent / Path(defaults[key])).resolve()
    if "l_over_leff_values" in defaults:
        defaults["l_over_leff_values"] = _as_float_list(defaults["l_over_leff_values"])

    return defaults


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH if DEFAULT_CONFIG_PATH.exists() else None,
        help=(
            "Optional TOML file with defaults. If omitted, the script auto-loads "
            f"{DEFAULT_CONFIG_PATH.relative_to(REPO_ROOT)} when present. "
            f"Accepted either as a [{CONFIG_SECTION}] table or as flat top-level keys. "
            "CLI flags override TOML values."
        ),
    )
    parser.add_argument("--lam", type=float, default=1.0, help="Lambda parameter in Eq. (18).")
    parser.add_argument("--l-over-leff", type=float, default=1.0, help="Ratio L / L_eff.")
    parser.add_argument(
        "--l-over-leff-values",
        type=float,
        nargs="+",
        default=None,
        help="Optional list of L/LD-style values to sweep in one run.",
    )
    parser.add_argument("--eta", type=float, default=1.0, help="Eta parameter in Eq. (18).")
    parser.add_argument("--x-min", type=float, default=0.0, help="Lower x bound.")
    parser.add_argument("--x-max", type=float, default=10.0, help="Upper x bound.")
    parser.add_argument("--npts", type=int, default=2000, help="Number of x samples.")
    parser.add_argument(
        "--show-ratio",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also save the K_XCI / N ratio plot.",
    )
    parser.add_argument(
        "--show-crossover",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Mark the first crossover point, if present in range.",
    )
    parser.add_argument(
        "--normalize",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Normalize each curve by its own finite maximum.",
    )
    parser.add_argument(
        "--show-kxci-alternative",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Overlay the notebook Eq. (18) K_XCI curve on the comparison plots.",
    )
    parser.add_argument(
        "--loglog",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use log-log axes. Requires x_min > 0.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Output directory for generated figures.",
    )
    parser.add_argument(
        "--stem",
        type=str,
        default="analytical_n_vs_kxci",
        help="Filename stem for saved figures.",
    )
    parser.add_argument(
        "--format",
        choices=("pdf", "png", "both"),
        default="pdf",
        help="Output format.",
    )
    parser.add_argument(
        "--use-td-corrections",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Apply the flat-profile TD plateau correction via the UWB lookup/cache path.",
    )
    parser.add_argument(
        "--system-config",
        type=Path,
        default=DEFAULT_SYSTEM_CONFIG_PATH if DEFAULT_SYSTEM_CONFIG_PATH.exists() else None,
        help="System TOML used to build/load the TD correction lookup tables.",
    )
    parser.add_argument(
        "--profile-path",
        type=Path,
        default=None,
        help="Optional flat-profile cache path used by the Raman lookup builder.",
    )
    parser.add_argument(
        "--recompute-corrections",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Force recomputation of the TD lookup tables instead of loading caches.",
    )
    parser.add_argument(
        "--flat-profile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use a synthetic flat power profile when building TD corrections.",
    )
    parser.add_argument(
        "--ipulse",
        type=int,
        choices=(0, 1),
        default=1,
        help="Pulse selector for the TD ideal fit and lookup caches: 0=Gaussian, 1=Nyquist.",
    )
    parser.add_argument(
        "--m-lo-truncation",
        type=int,
        default=2,
        help="Highest m_lo order included when building/loading TD correction lookups.",
    )
    return parser


def main() -> None:
    _configure_matplotlib()
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH if DEFAULT_CONFIG_PATH.exists() else None,
    )
    pre_args, _ = pre_parser.parse_known_args()

    parser = build_parser()
    parser.set_defaults(**_load_config_defaults(pre_args.config))
    args = parser.parse_args()

    if args.lam <= 0.0 or args.l_over_leff <= 0.0 or args.eta <= 0.0:
        raise ValueError("lambda, L/Leff, and eta must be strictly positive.")
    if int(args.m_lo_truncation) < 0:
        raise ValueError("m_lo_truncation must be non-negative.")
    l_over_leff_values = _as_float_list(args.l_over_leff_values) or [float(args.l_over_leff)]
    if any(value <= 0.0 for value in l_over_leff_values):
        raise ValueError("All L/Leff values must be strictly positive.")
    if args.use_td_corrections and args.system_config is None:
        raise ValueError("--system-config is required when --use-td-corrections is enabled.")

    x = _make_x_grid(args.x_min, args.x_max, args.npts, args.loglog)
    td_ctx: dict[str, Any] | None = None
    if args.use_td_corrections:
        system = System.from_toml(args.system_config)
        if args.flat_profile:
            profile_path = (
                args.profile_path
                if args.profile_path is not None
                else args.out_dir / f"{args.stem}_flat_profile.npy"
            )
        elif args.profile_path is None:
            raise ValueError("--profile-path is required when --no-flat-profile is used.")
        else:
            profile_path = args.profile_path
        td_ctx = _prepare_flat_td_corrections(
            system,
            profile_path=Path(profile_path),
            ipulse=int(args.ipulse),
            recompute=bool(args.recompute_corrections),
            max_l_over_ld=max(max(l_over_leff_values), DEFAULT_LLD_SWEEP_MAX),
            eta_override=float(args.eta),
            m_lo_truncation=int(args.m_lo_truncation),
        )
        lld_target = 1e1
        ps_lld_target = _td_corrected_softplus_params_flat(
            float(lld_target),
            td_ctx,
            verbose=False,
        )
        td_ctx["lld_target"] = float(lld_target)
        td_ctx["ps_lld_target"] = tuple(float(v) for v in ps_lld_target)

    suffixes = [".pdf", ".png"] if args.format == "both" else [f".{args.format}"]
    multi_case = len(l_over_leff_values) > 1
    summary_rows: list[dict[str, float]] = []
    case_results: list[dict[str, Any]] = []

    for l_over_leff in l_over_leff_values:
        leff_over_l = 1.0 / l_over_leff
        y_n_paper = n_time_domain(x, lam=args.lam, l_over_leff=l_over_leff, eta=args.eta)
        y_k = k_xci(x, leff_over_l=leff_over_l)
        y_k_eq18 = k_xci_eq18_normalized(x, l_over_leff=l_over_leff)

        y_n = y_n_paper
        y_n_reference = None
        n_label = r"$\mathnormal{\mathcal{N}^{(\mathrm{TD})}}$"
        n_reference_label = None
        plateau_ideal = float("nan")
        lambda_ideal = float("nan")
        eta_ideal = float("nan")
        plateau_corrected = float("nan")
        lambda_corrected = float("nan")
        eta_corrected = float("nan")

        if td_ctx is not None:
            ps_ideal = td_ctx["ps_ideal"]
            ps_corrected = _td_corrected_softplus_params_flat(l_over_leff, td_ctx)
            y_n = n_softplus_scaled(x, l_over_ld=l_over_leff, ps=ps_corrected)
            y_n_reference = y_n_paper
            n_label = r"$\mathnormal{\mathcal{N}^{(\mathrm{TD})}}$"
            n_reference_label = r"$\mathnormal \mathcal{N}_{\mathrm{Eq.18}}(x)\,T^2L^{-2}$"
            plateau_ideal, lambda_ideal, eta_ideal = ps_ideal
            plateau_corrected, lambda_corrected, eta_corrected = ps_corrected

        if args.normalize:
            arrays_to_scale = [y_n, y_k, y_k_eq18]
            if y_n_reference is not None:
                arrays_to_scale.append(y_n_reference)
            for y in arrays_to_scale:
                finite = y[np.isfinite(y)]
                if finite.size == 0:
                    continue
                scale = np.nanmax(np.abs(finite))
                if scale > 0.0:
                    y /= scale

        fig_cmp, crossover = _comparison_figure(
            x,
            y_n,
            y_k,
            y_k_eq18=y_k_eq18,
            y_n_reference=y_n_reference,
            loglog=args.loglog,
            normalize=args.normalize,
            show_crossover=args.show_crossover,
            show_kxci_alternative=args.show_kxci_alternative,
            n_label=n_label,
            n_reference_label=n_reference_label,
        )

        case_stem = args.stem if not multi_case else f"{args.stem}_lld{_case_tag(l_over_leff)}"
        for suffix in suffixes:
            _save_figure(fig_cmp, args.out_dir / f"{case_stem}{suffix}")
        plt.close(fig_cmp)

        x_x1, y_n_x1, y_k_x1, y_k_eq18_x1, y_n_ref_x1 = _slice_from_unity(
            x,
            y_n,
            y_k,
            y_k_eq18,
            y_n_reference,
        )
        if x_x1.size >= 2:
            fig_cmp_x1, _ = _comparison_figure(
                x_x1,
                y_n_x1,
                y_k_x1,
                y_k_eq18=y_k_eq18_x1,
                y_n_reference=y_n_ref_x1,
                loglog=args.loglog,
                normalize=args.normalize,
                show_crossover=args.show_crossover,
                show_kxci_alternative=args.show_kxci_alternative,
                n_label=n_label,
                n_reference_label=n_reference_label,
                show_unity_guides=False,
            )
            for suffix in suffixes:
                _save_figure(fig_cmp_x1, args.out_dir / f"{case_stem}_xge1{suffix}")
            plt.close(fig_cmp_x1)
        else:
            lg.warning("Skipping x>=1 comparison plot for {}: not enough points in selected range.", case_stem)

        if args.show_ratio:
            ratio = safe_ratio(y_k, y_n)
            ratio_eq18 = safe_ratio(y_k_eq18, y_n) if args.show_kxci_alternative else None
            fig_ratio = _ratio_figure(
                x,
                ratio,
                ratio_eq18=ratio_eq18,
                loglog=args.loglog,
                crossover=crossover,
            )
            for suffix in suffixes:
                _save_figure(fig_ratio, args.out_dir / f"{case_stem}_ratio{suffix}")
            plt.close(fig_ratio)
        else:
            ratio = safe_ratio(y_k, y_n)

        print(
            "Parameters: "
            f"lambda={args.lam}, L/Leff={l_over_leff}, eta={args.eta}, "
            f"x in [{args.x_min}, {args.x_max}]"
        )
        if td_ctx is not None:
            print(
                "TD flat-profile correction: "
                f"a={plateau_corrected:.6g}, Lambda={lambda_corrected:.6g}, eta={eta_corrected:.6g}"
            )
        if crossover is None:
            print("No crossover found in the selected x-range.")
        else:
            print(f"First crossover at x = Delta f / B ≈ {crossover:.6g}")

        summary_rows.append(
            {
                "l_over_ld": float(l_over_leff),
                "crossover_x": float(crossover) if crossover is not None else float("nan"),
                "plateau_ideal": plateau_ideal,
                "lambda_ideal": lambda_ideal,
                "eta_ideal": eta_ideal,
                "plateau_corrected": plateau_corrected,
                "lambda_corrected": lambda_corrected,
                "eta_corrected": eta_corrected,
            }
        )
        case_results.append(
            {
                "l_over_leff": float(l_over_leff),
                "case_label": rf"$L/L_D$",#={l_over_leff:g}$",
                "y_n": np.asarray(y_n, dtype=float).copy(),
                "y_k": np.asarray(y_k, dtype=float).copy(),
                "y_k_eq18": np.asarray(y_k_eq18, dtype=float).copy(),
                "ratio": np.asarray(ratio, dtype=float).copy(),
                "crossover": crossover,
            }
        )

    if pre_args.config is not None:
        print(f"Config: {pre_args.config}")
    if multi_case:
        fig_combined = _combined_comparison_figure(
            x,
            case_results,
            loglog=args.loglog,
            normalize=args.normalize,
            show_kxci_alternative=args.show_kxci_alternative,
        )
        for suffix in suffixes:
            _save_figure(fig_combined, args.out_dir / f"{args.stem}_combined{suffix}")
        plt.close(fig_combined)

        mask_xge1 = np.isfinite(x) & (x >= 1.0)
        x_xge1 = np.asarray(x, dtype=float)[mask_xge1]
        if x_xge1.size >= 2:
            case_results_xge1: list[dict[str, Any]] = []
            for case in case_results:
                case_results_xge1.append(
                    {
                        "l_over_leff": case["l_over_leff"],
                        "case_label": case["case_label"],
                        "y_n": np.asarray(case["y_n"], dtype=float)[mask_xge1],
                        "y_k": np.asarray(case["y_k"], dtype=float)[mask_xge1],
                        "y_k_eq18": np.asarray(case["y_k_eq18"], dtype=float)[mask_xge1],
                        "ratio": np.asarray(case["ratio"], dtype=float)[mask_xge1],
                        "crossover": case["crossover"],
                    }
                )

            fig_combined_xge1 = _combined_comparison_figure(
                x_xge1,
                case_results_xge1,
                loglog=args.loglog,
                normalize=args.normalize,
                show_kxci_alternative=args.show_kxci_alternative,
                show_unity_guides=False,
            )
            for suffix in suffixes:
                _save_figure(fig_combined_xge1, args.out_dir / f"{args.stem}_combined_xge1{suffix}")
            plt.close(fig_combined_xge1)
        else:
            lg.warning("Skipping combined x>=1 comparison plot: not enough points in selected range.")

        if args.show_ratio:
            fig_combined_ratio = _combined_ratio_figure(
                x,
                case_results,
                loglog=args.loglog,
            )
            for suffix in suffixes:
                _save_figure(fig_combined_ratio, args.out_dir / f"{args.stem}_combined_ratio{suffix}")
            plt.close(fig_combined_ratio)

    lld_sweep = np.geomspace(
        DEFAULT_LLD_SWEEP_MIN,
        DEFAULT_LLD_SWEEP_MAX,
        DEFAULT_LLD_SWEEP_NPTS,
    )
    x_fixed_grid = np.full_like(lld_sweep, DEFAULT_LLD_SWEEP_X)
    if td_ctx is not None:
        y_n_lld = np.array(
            [
                n_softplus_scaled(
                    np.array([DEFAULT_LLD_SWEEP_X], dtype=float),
                    l_over_ld=float(value),
                    ps=_td_corrected_softplus_params_flat(float(value), td_ctx, verbose=False),
                )[0]
                for value in lld_sweep
            ],
            dtype=float,
        )
        n_lld_label = r"$\mathnormal{\mathcal{N}^{(\mathrm{TD})}}$"
    else:
        y_n_lld = n_time_domain(
            x_fixed_grid,
            lam=args.lam,
            l_over_leff=lld_sweep,
            eta=args.eta,
        )
        n_lld_label = r"$\mathnormal{\mathcal{N}^{(\mathrm{TD})}}$"
    y_k_lld = k_xci(x_fixed_grid, leff_over_l=1.0 / lld_sweep)
    y_k_eq18_lld = k_xci_eq18_normalized(x_fixed_grid, l_over_leff=lld_sweep)

    final_idx = -1
    final_lld = float(lld_sweep[final_idx])
    final_x = float(DEFAULT_LLD_SWEEP_X)
    final_n = float(y_n_lld[final_idx])
    final_k = float(y_k_lld[final_idx])
    final_k_eq18 = float(y_k_eq18_lld[final_idx])
    final_ratio = float(final_k / final_n) if np.isfinite(final_n) and abs(final_n) > 1e-14 else float("nan")
    final_ratio_eq18 = (
        float(final_k_eq18 / final_n)
        if np.isfinite(final_n) and abs(final_n) > 1e-14 and np.isfinite(final_k_eq18)
        else float("nan")
    )

    table_rows: list[tuple[str, float]] = [
        ("N(x=1)", final_n),
        ("K_XCI(x=1)", final_k),
        ("K_XCI_Eq18(x=1)", final_k_eq18),
        ("K_XCI/N", final_ratio),
        ("K_XCI_Eq18/N", final_ratio_eq18),
    ]

    if td_ctx is not None:
        final_n_eq18 = float(
            n_time_domain(
                np.array([final_x], dtype=float),
                lam=args.lam,
                l_over_leff=final_lld,
                eta=args.eta,
            )[0]
        )
        table_rows.insert(1, ("N_Eq18(x=1)", final_n_eq18))

        ps_lld_target = tuple(float(v) for v in td_ctx["ps_lld_target"])
        n0_lambda_lld_target = float(ps_lld_target[0] * ps_lld_target[1])
        table_rows.extend(
            [
                ("a", ps_lld_target[0]),
                ("Lambda", ps_lld_target[1]),
                ("eta", ps_lld_target[2]),
                ("a*Lambda", n0_lambda_lld_target),
            ]
        )

    name_width = max(len(name) for name, _ in table_rows)
    value_col = "Value"
    header = f"{'Line/parameter':<{name_width}}  {value_col}"
    separator = f"{'-' * name_width}  {'-' * len(value_col)}"
    body = "\n".join(
        f"{name:<{name_width}}  {value:.12e}" if np.isfinite(value) else f"{name:<{name_width}}  nan"
        for name, value in table_rows
    )
    lg.info(
        "Final values table at x={:.6g}, L/LD={:.6g}:\n{}\n{}\n{}".format(
            final_x,
            final_lld,
            header,
            separator,
            body,
        )
    )

    fig_lld = _lld_sweep_figure(
        lld_sweep,
        y_n_lld,
        y_k_lld,
        y_k_eq18=y_k_eq18_lld,
        x_fixed=DEFAULT_LLD_SWEEP_X,
        loglog=args.loglog,
        n_label=n_lld_label,
    )
    for suffix in suffixes:
        _save_figure(fig_lld, args.out_dir / f"{args.stem}_x1_vs_lld{suffix}")
    plt.close(fig_lld)

    if td_ctx is not None:
        print(f"TD lookup system: {args.system_config}")
        print(f"TD flat profile: {td_ctx['profile_path']}")
        print(
            "Cache behavior: lookup tables and low-order datasets are loaded if present; "
            "missing ones are computed and saved unless --recompute-corrections forces regeneration."
        )
        _write_summary(args.out_dir / f"{args.stem}_td_corrections.csv", summary_rows)
    print("K_XCI is only defined for x > 1/2; values at x <= 1/2 are masked.")


if __name__ == "__main__":
    main()
