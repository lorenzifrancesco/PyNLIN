"""Standalone non-interactive plotter for the analytical N vs K_XCI comparison.

This reproduces the analytical comparison used in the scratch ``N_vs_KXCI``
workflow, using the Eq. (18) model

    N(x) T^2 L^-2 = 4/9 * (1 + ((1/lambda) * (L/L_eff) * x)^(1/eta))^(-eta)

against the flat-profile closed-form XCI kernel

    K_XCI(x) T^2 L^-2 = (L_eff/L) * |ln((x + 1/2)/(x - 1/2))|

for x = Delta f / B.

The script is intentionally non-interactive: it always writes publication-ready
figures to disk and never calls ``plt.show()``.
"""

from __future__ import annotations

import argparse
import os
import sys
from functools import lru_cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mplconfig"))

import matplotlib
import numpy as np
from scipy.integrate import quad
from scipy.special import sici

matplotlib.use("Agg")
import matplotlib.pyplot as plt

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.pcfm.figure_size import scale_figsize_to_ieee_column
from pynlin.utils import _toml_load


DEFAULT_OUT_DIR = REPO_ROOT / "media" / "standalone_analytical"
DEFAULT_CONFIG_PATH = REPO_ROOT / "input" / "analytical_n_vs_kxci.toml"
DEFAULT_MPLRC_PATH = Path.home() / ".config" / "matplotlib" / "matplotlibrc"
DPI = 300
CONFIG_SECTION = "analytical_n_vs_kxci"
LINE_LW = 0.75


def _configure_matplotlib() -> None:
    # Load the user's matplotlibrc explicitly because MPLCONFIGDIR is set to a
    # writable local directory for sandboxed runs, which would otherwise bypass
    # the user's sansmath / inward-tick defaults.
    if DEFAULT_MPLRC_PATH.exists():
        matplotlib.rc_file(str(DEFAULT_MPLRC_PATH))
    plt.rcParams["lines.linewidth"] = LINE_LW


def n_eq18(x: np.ndarray, lam: float, l_over_leff: float, eta: float) -> np.ndarray:
    """Eq. (18) curve used in the original comparison."""
    scale = float(l_over_leff) / float(lam)
    return (4.0 / 9.0) * (1.0 + (scale * x) ** (1.0 / eta)) ** (-eta)


def k_xci(x: np.ndarray, leff_over_l: float) -> np.ndarray:
    """Closed-form XCI kernel over the non-overlapping region x > 1/2."""
    out = np.full_like(x, np.nan, dtype=float)
    mask = x > 0.5
    if np.any(mask):
        out[mask] = float(leff_over_l) * np.abs(
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
    if l_over_leff <= 0.0:
        raise ValueError("L/Leff must be strictly positive.")
    out = np.full_like(x, np.nan, dtype=float)
    mask = x > 0.5
    if not np.any(mask):
        return out
    z_plus = np.pi * l_over_leff * (x[mask] + 0.5)
    z_minus = np.pi * l_over_leff * (x[mask] - 0.5)
    out[mask] = (2.0 / (np.pi * l_over_leff)) * (_g_kernel(z_plus) - _g_kernel(z_minus))
    return out


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


def _style_axes(ax: plt.Axes, *, loglog: bool) -> None:
    if loglog:
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.grid(False)


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

    if x_f[0] >= 1.0:
        ax.plot(x_f, y_f, color=color, lw=LINE_LW, ls=linestyle, label=label)
        return
    if x_f[-1] <= 1.0:
        ax.plot(x_f, y_f, color=color, lw=LINE_LW, ls="--", label=label)
        return

    y_split = float(np.interp(1.0, x_f, y_f))
    left_mask = x_f < 1.0
    right_mask = x_f > 1.0

    x_left = np.concatenate([x_f[left_mask], np.array([1.0])])
    y_left = np.concatenate([y_f[left_mask], np.array([y_split])])
    x_right = np.concatenate([np.array([1.0]), x_f[right_mask]])
    y_right = np.concatenate([np.array([y_split]), y_f[right_mask]])

    ax.plot(x_left, y_left, color=color, lw=LINE_LW, ls="--", label=label)
    ax.plot(x_right, y_right, color=color, lw=LINE_LW, ls=linestyle)


def _annotate_references(ax: plt.Axes, x: np.ndarray) -> tuple[float, float]:
    y_ln3 = float(np.log(3.0))
    y_four_ninths = 4.0 / 9.0
    x_anchor = float(x[min(len(x) - 1, int(0.82 * len(x)))])

    ax.axhline(y_ln3, color="tab:red", linestyle="--", linewidth=LINE_LW, label=r"$\ln(3)$")
    ax.axhline(
        y_four_ninths,
        color="tab:green",
        linestyle="--",
        linewidth=LINE_LW,
        label=r"$4/9$",
    )
    ax.annotate(
        r"$\ln(3)$",
        xy=(x_anchor, y_ln3),
        xytext=(5, 5),
        textcoords="offset points",
        color="tab:red",
    )
    ax.annotate(
        r"$4/9$",
        xy=(x_anchor, y_four_ninths),
        xytext=(5, -12),
        textcoords="offset points",
        color="tab:green",
    )
    return y_ln3, y_four_ninths


def _comparison_figure(
    x: np.ndarray,
    y_n: np.ndarray,
    y_k: np.ndarray,
    *,
    y_k_eq18: np.ndarray,
    loglog: bool,
    normalize: bool,
    show_crossover: bool,
    show_kxci_alternative: bool,
) -> tuple[plt.Figure, float | None]:
    fig, ax = plt.subplots(figsize=scale_figsize_to_ieee_column(3.6, 2.8))
    _plot_pcfm_style_line(ax, x, y_n, color="black", label=r"$\mathnormal \mathcal{N}(x)\,T^2L^{-2}$")
    _plot_pcfm_style_line(
        ax,
        x,
        y_k,
        color="tab:blue",
        label=r"$\mathnormal K_{\mathrm{XCI}}(x)\,T^2L^{-2}$",
    )
    if show_kxci_alternative:
        _plot_pcfm_style_line(
            ax,
            x,
            y_k_eq18,
            color="tab:orange",
            label=r"$\mathnormal K_{\mathrm{XCI}}^{\mathrm{Eq.18}}(x)\,T^2L^{-2}$",
        )
    ax.axvline(1.0, color="0.5", linestyle=":", linewidth=LINE_LW)

    y_ln3, y_four_ninths = _annotate_references(ax, x)
    _style_axes(ax, loglog=loglog)

    if not loglog:
        arrays = [y_n, y_k, y_ln3, y_four_ninths]
        if show_kxci_alternative:
            arrays.append(y_k_eq18)
        ymin, ymax = finite_min_max(*arrays)
        pad = 0.06 * (ymax - ymin) if ymax > ymin else 0.1
        ax.set_ylim(ymin - pad, ymax + pad)

    ax.set_xlabel(r"$\mathnormal{\Delta f / B}$")
    ax.set_ylabel("Normalized value" if normalize else r"$\mathnormal{T^2L^{-2}}$")
    ax.legend(loc="best", fontsize=7)

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
                fontsize=7,
            )

    fig.tight_layout()
    return fig, crossover


def _ratio_figure(
    x: np.ndarray,
    ratio: np.ndarray,
    *,
    ratio_eq18: np.ndarray | None,
    loglog: bool,
    crossover: float | None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=scale_figsize_to_ieee_column(3.6, 2.4))
    _plot_pcfm_style_line(ax, x, ratio, color="tab:blue", label=r"$\mathnormal{ K_{\mathrm{XCI}}/\mathcal{N}}$")
    if ratio_eq18 is not None:
        _plot_pcfm_style_line(
            ax,
            x,
            ratio_eq18,
            color="tab:orange",
            label=r"$K_{\mathrm{XCI}}^{\mathrm{Eq.18}}/\mathcal{N}$",
        )
    if crossover is not None and np.isfinite(crossover):
        ax.axvline(crossover, color="0.35", linestyle="--", linewidth=LINE_LW)
    _style_axes(ax, loglog=loglog)
    ax.set_xlabel(r"$\Delta f / B$")
    ax.set_ylabel(r"$\mathnormal{K_{\mathrm{XCI}} / \mathcal{N}}$")
    ax.legend(loc="best", fontsize=7)
    fig.tight_layout()
    return fig


def _save_figure(fig: plt.Figure, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI)
    print(f"Saved {out_path}")


def _config_key_map() -> dict[str, str]:
    return {
        "lam": "lam",
        "lambda": "lam",
        "l_over_leff": "l_over_leff",
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

    x = _make_x_grid(args.x_min, args.x_max, args.npts, args.loglog)
    leff_over_l = 1.0 / args.l_over_leff
    y_n = n_eq18(x, lam=args.lam, l_over_leff=args.l_over_leff, eta=args.eta)
    y_k = k_xci(x, leff_over_l=leff_over_l)
    y_k_eq18 = k_xci_eq18_normalized(x, l_over_leff=args.l_over_leff)

    if args.normalize:
        for y in (y_n, y_k, y_k_eq18):
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
        loglog=args.loglog,
        normalize=args.normalize,
        show_crossover=args.show_crossover,
        show_kxci_alternative=args.show_kxci_alternative,
    )

    suffixes = [".pdf", ".png"] if args.format == "both" else [f".{args.format}"]
    for suffix in suffixes:
        _save_figure(fig_cmp, args.out_dir / f"{args.stem}{suffix}")
    plt.close(fig_cmp)

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
            _save_figure(fig_ratio, args.out_dir / f"{args.stem}_ratio{suffix}")
        plt.close(fig_ratio)

    print(
        "Parameters: "
        f"lambda={args.lam}, L/Leff={args.l_over_leff}, eta={args.eta}, "
        f"x in [{args.x_min}, {args.x_max}]"
    )
    if pre_args.config is not None:
        print(f"Config: {pre_args.config}")
    if crossover is None:
        print("No crossover found in the selected x-range.")
    else:
        print(f"First crossover at x = Delta f / B ≈ {crossover:.6g}")
    print("K_XCI is only defined for x > 1/2; values at x <= 1/2 are masked.")


if __name__ == "__main__":
    main()
